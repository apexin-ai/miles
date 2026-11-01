"""CPU contracts for the Qwen3.8 vision audit with miles' real ModelCompanion (M-034).

Real: ``vision_audit.before_step`` and ``_verify_saved_model``, ``ModelCompanion`` with its weight-version bump,
``load_state_dict`` and sample-consumption rows, ``_gradient_samples`` over a single-process gloo group, the vision
plugin's ``_ZeroGradsFor`` for text microbatches, and torch SGD. The loop keeps ``train_one_step``'s order:
before-step hook -> forward/backward -> optimizer.step -> weight-version bump.

Stubbed: ``torch.cuda.synchronize`` (CPU), the Megatron distributed gradient buffer (one bucket, DP size 1) and the
Megatron checkpoint I/O, which is a ``state_dict`` round trip through ``torch.save``/``torch.load`` behind the real
``load_checkpoint`` signature. Multi-rank TP/DP behaviour is a GPU test (r3/req/VLFIX-*).
"""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.model_companion import ModelCompanion, ModelCompanionWeightVersionUtils
from miles.utils.types import SampleLineage
from miles_plugins.models.qwen3_8_next import vision_audit
from miles_plugins.models.qwen3_8_next.vision import _ZeroGradsFor

_X = torch.tensor([[1.0, -2.0]])


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    store = tmp_path_factory.mktemp("pg") / "store"
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(vision_audit, "_runtime_identity", lambda: {})


class _Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Linear(2, 2, bias=False)
        self.merger = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x, grid_thw=None, return_dict=True):
        return self.merger(self.blocks(x))


class Toy(torch.nn.Module):
    """A decoder, a native-named vision tower and miles' real companion, as one model chunk."""

    def __init__(self, *, freeze_language, scope="full", detach_visual=False, leak_visual=False):
        super().__init__()
        torch.manual_seed(0)
        self.decoder = torch.nn.Linear(2, 2, bias=False)
        self.decoder.requires_grad_(not freeze_language)
        self.vision_model = _Vision()
        for name, parameter in self.vision_model.named_parameters():
            parameter.requires_grad_(scope == "full" or (scope == "projector" and name.startswith("merger.")))
        self.model_companion = ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0))
        self.detach_visual = detach_visual
        self.leak_visual = leak_visual

    def forward(self, x, *, image):
        if image:
            visual = self.vision_model(x, grid_thw=torch.tensor([[1, 2, 2]]))
            hidden = x + (visual.detach() if self.detach_visual else visual)
            return self.decoder(hidden).sum()
        output = self.decoder(x).sum()
        if self.leak_visual:
            # A text microbatch whose loss reaches the tower without running it: the audit must notice.
            output = output + 1e-3 * self.vision_model.merger.weight.sum()
        trainable = [p for p in self.vision_model.parameters() if p.requires_grad]
        if trainable and torch.is_grad_enabled():
            output = _ZeroGradsFor.apply(output, *trainable)
        return output


class Optimizer:
    """Real SGD behind miles' distributed-optimizer gradient-buffer view (one bucket, DP size 1)."""

    def __init__(self, model, lr):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.inner = torch.optim.SGD(self.params, lr=lr)
        self.grad_data = torch.zeros(sum(p.numel() for p in self.params))
        mapping, start = {}, 0
        for p in self.params:
            mapping[p] = (start, start + p.numel(), 0)
            start += p.numel()
        bucket = SimpleNamespace(grad_data=self.grad_data, offset=0)
        self.buffers = [SimpleNamespace(param_index_map=mapping, buckets=[bucket], data_parallel_group=dist.group.WORLD)]

    def load_grads(self):
        start = 0
        for p in self.params:
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            self.grad_data[start : start + p.numel()] = grad.flatten()
            start += p.numel()

    def step(self):
        self.inner.step()
        return True, 1.0, 0


def _args(tmp_path, *, freeze_language, steps, interval=1):
    hf = tmp_path / "hf"
    hf.mkdir(parents=True, exist_ok=True)
    (hf / "config.json").write_text("{}")
    weight_map = {"model.visual.blocks.weight": "v.safetensors", "model.visual.merger.weight": "v.safetensors"}
    (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return SimpleNamespace(
        qwen38_audit_dir=str(tmp_path / "audit"),
        qwen38_freeze_language=freeze_language,
        qwen38_visual_snapshot_interval=0,
        qwen38_audit_interval=interval,
        hf_checkpoint=str(hf),
        num_rollout=steps,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
    )


def _train(model, args, images, *, lr=0.1, bump=True, before_update=None):
    optimizer = Optimizer(model, lr)
    for step, image in enumerate(images):
        vision_audit.before_step(args, step, 0, [model], optimizer, None)
        model(_X, image=image).backward()
        optimizer.load_grads()
        if before_update is not None:
            with torch.no_grad():
                before_update(step, model)
        optimizer.step()
        if bump:
            ModelCompanionWeightVersionUtils.bump_weight_version([model])
        optimizer.inner.zero_grad(set_to_none=True)
    return optimizer


def _record(args, update):
    return json.loads((Path(args.qwen38_audit_dir) / f"rank-000-update-{update:06d}.json").read_text())


def _last_step_write(module_path, steps):
    def write(step, model):
        if step == steps - 1:
            model.get_parameter(module_path)[0, 0] += 1.0

    return write


# --- language side: the companion counter is not a language weight ---


def test_frozen_language_with_the_real_companion_passes_three_updates(tmp_path):
    model = Toy(freeze_language=True)
    decoder = model.decoder.weight.detach().clone()
    args = _args(tmp_path, freeze_language=True, steps=3)
    _train(model, args, [True, True, True])
    assert torch.equal(model.decoder.weight, decoder)
    assert ModelCompanionWeightVersionUtils.weight_version([model]) == 3
    final = _record(args, 3)
    assert final["final_language_sha256"] == final["initial_language_sha256"]
    assert final["visual_sha256"] != json.loads((tmp_path / "audit/rank-000-initial.json").read_text())["visual_sha256"]


def test_frozen_language_still_rejects_a_real_decoder_write(tmp_path):
    model = Toy(freeze_language=True)
    args = _args(tmp_path, freeze_language=True, steps=3)
    with pytest.raises(ValueError, match="Frozen decoder weights changed"):
        _train(model, args, [True, True, True], before_update=_last_step_write("decoder.weight", 3))


def test_joint_run_without_a_decoder_update_is_rejected_despite_the_counter(tmp_path):
    # Nonzero language gradients, lr 0: the decoder never moves, only the companion counter does.
    # A frozen tower keeps the visual checks out of the way.
    model = Toy(freeze_language=False, scope="none")
    decoder = model.decoder.weight.detach().clone()
    args = _args(tmp_path, freeze_language=False, steps=2)
    with pytest.raises(ValueError, match="trainable decoder did not change"):
        _train(model, args, [True, True], lr=0.0)
    assert torch.equal(model.decoder.weight, decoder)
    assert ModelCompanionWeightVersionUtils.weight_version([model]) == 1


def test_joint_run_with_real_updates_passes(tmp_path):
    model = Toy(freeze_language=False)
    args = _args(tmp_path, freeze_language=False, steps=2)
    _train(model, args, [True, True])
    final = _record(args, 2)
    assert final["final_language_sha256"] != final["initial_language_sha256"]
    assert all(row["gradient_nonzero"] and row["delta_nonzero"] for row in final["language_parameters"])


# --- vision side: gradients follow the images the step actually ran ---


@pytest.mark.parametrize("freeze_language", [True, False])
def test_a_legal_text_only_step_in_a_mixed_run_is_accepted(tmp_path, freeze_language):
    model = Toy(freeze_language=freeze_language)
    args = _args(tmp_path, freeze_language=freeze_language, steps=3)
    _train(model, args, [True, False, True])
    text = _record(args, 2)
    assert text["images_in_step"] == 0
    assert text["parameters"] and not any(row["gradient_nonzero"] for row in text["parameters"])
    assert [_record(args, update)["images_in_step"] for update in (1, 2, 3)] == [1, 0, 1]
    assert _record(args, 3)["images_total"] == 2


def test_an_image_step_without_a_visual_gradient_is_rejected(tmp_path):
    model = Toy(freeze_language=False, detach_visual=True)
    args = _args(tmp_path, freeze_language=False, steps=2)
    with pytest.raises(ValueError, match="(?i)no visual gradient probe received a nonzero gradient"):
        _train(model, args, [True, True])


def test_a_visual_gradient_without_an_image_is_rejected(tmp_path):
    model = Toy(freeze_language=False, leak_visual=True)
    args = _args(tmp_path, freeze_language=False, steps=2)
    with pytest.raises(ValueError, match="visual gradient without an image"):
        _train(model, args, [False, True])


def test_a_run_that_never_saw_an_image_does_not_require_a_visual_change(tmp_path):
    model = Toy(freeze_language=False)
    visual = {n: p.detach().clone() for n, p in model.vision_model.named_parameters()}
    args = _args(tmp_path, freeze_language=False, steps=2)
    _train(model, args, [False, False])
    assert all(torch.equal(p, visual[n]) for n, p in model.vision_model.named_parameters())
    assert _record(args, 2)["images_total"] == 0


def test_a_run_with_images_still_requires_the_trainable_tower_to_change(tmp_path):
    # Frozen language, lr 0: images ran and gave visual gradients, but the tower never moved.
    model = Toy(freeze_language=True)
    args = _args(tmp_path, freeze_language=True, steps=2)
    with pytest.raises(ValueError, match="trainable visual tower did not change"):
        _train(model, args, [True, True], lr=0.0)


@pytest.mark.parametrize(
    "scope,frozen_path", [("none", "vision_model.merger.weight"), ("projector", "vision_model.blocks.weight")]
)
def test_frozen_visual_parameters_must_not_change(tmp_path, scope, frozen_path):
    model = Toy(freeze_language=False, scope=scope)
    args = _args(tmp_path, freeze_language=False, steps=2)
    _train(model, args, [True, True])
    model = Toy(freeze_language=False, scope=scope)
    args = _args(tmp_path / "write", freeze_language=False, steps=2)
    with pytest.raises(ValueError, match="Frozen visual weights changed"):
        _train(model, args, [True, True], before_update=_last_step_write(frozen_path, 2))


def test_audit_interval_records_interval_and_final_updates_and_counts_every_image(tmp_path):
    model = Toy(freeze_language=False)
    args = _args(tmp_path, freeze_language=False, steps=5, interval=2)
    _train(model, args, [True, False, True, True, False])
    written = sorted(path.name for path in Path(args.qwen38_audit_dir).glob("rank-000-update-*.json"))
    assert written == [f"rank-000-update-{u:06d}.json" for u in (2, 4, 5)]
    assert _record(args, 5)["images_total"] == 3


# --- final-save reload: model weights only, integers restored, companion state untouched ---


class _Backend(SimpleNamespace):
    @contextlib.contextmanager
    def mutable(self):
        yield


def _reload_model():
    model = Toy(freeze_language=False)
    model.register_parameter(
        "ple_offsets", torch.nn.Parameter(torch.arange(40, dtype=torch.int64) * 7, requires_grad=False)
    )
    model.register_parameter("final_norm", torch.nn.Parameter(torch.linspace(0, 1, 64), requires_grad=False))
    model.model_companion.weight_version.add_(3)
    model.model_companion.record_sample_consumptions([SampleLineage(source_sample_index=5, output_index=0, output_count=1)])
    return model


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _loader(model, path, calls, *, after_load=None):
    """A Megatron-free stand-in with the real load_checkpoint signature: a real state_dict round trip."""
    from miles.backends.megatron_utils.checkpoint import _CheckpointLoadResult

    def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt, *, args):
        calls.append({name: getattr(args.backend, name) for name in ("load", "ckpt_step", "finetune", "no_load_optim")})
        assert torch.isnan(model.decoder.weight.flatten()[0]) and torch.isnan(model.vision_model.blocks.weight.flatten()[0])
        assert not torch.equal(model.ple_offsets[:32], torch.arange(32) * 7)
        assert model.model_companion.weight_version.item() == 3
        ddp_model[0].load_state_dict(torch.load(path, weights_only=True))
        model.model_companion.clear_sample_consumptions()  # what load_checkpoint does for finetune=True
        if after_load is not None:
            with torch.no_grad():
                after_load(model)
        return _CheckpointLoadResult(iteration=0, restored_trained_iteration=False)

    return load_checkpoint


def _reload_args(tmp_path):
    return SimpleNamespace(backend=_Backend(save=str(tmp_path / "ckpt")), qwen38_audit_dir=str(tmp_path / "audit"))


def test_reload_restores_float_and_integer_weights_and_leaves_the_companion_alone(tmp_path, monkeypatch):
    from miles.backends.megatron_utils import checkpoint

    model = _reload_model()
    torch.save(_state(model), tmp_path / "saved.pt")
    before = _state(model)
    calls = []
    monkeypatch.setattr(checkpoint, "load_checkpoint", _loader(model, tmp_path / "saved.pt", calls))
    vision_audit._verify_saved_model(_reload_args(tmp_path), 7, [model])
    assert calls == [dict(load=str(tmp_path / "ckpt"), ckpt_step=7, finetune=True, no_load_optim=True)]
    after = _state(model)
    assert set(after) == set(before) and all(torch.equal(after[name], before[name]) for name in before), [
        name for name in before if not torch.equal(after[name], before[name])
    ]
    receipt = json.loads((tmp_path / "audit/checkpoint-reload/rank-000.json").read_text())
    assert receipt["passed"] and receipt["checkpoint_iteration"] == 7
    assert receipt["integer_tensors_perturbed"] == 1 and receipt["companion_tensors_excluded"] == 2
    assert receipt["companion_weight_version"] == {"live": 3, "checkpoint": 3}


def test_reload_mismatch_names_the_parameters_and_restores_the_live_model(tmp_path, monkeypatch):
    # The joint160 signature: the checkpoint holds TP rank 0's copy of a replicated tensor that drifted here.
    from miles.backends.megatron_utils import checkpoint

    model = _reload_model()
    torch.save(_state(model), tmp_path / "saved.pt")
    before = _state(model)

    def tp0_copy(model):
        model.final_norm[50] += 1e-3  # beyond the 32-element head the check perturbs

    monkeypatch.setattr(checkpoint, "load_checkpoint", _loader(model, tmp_path / "saved.pt", [], after_load=tp0_copy))
    with pytest.raises(ValueError, match=r"1 of \d+ model tensors differ after the reload: final_norm"):
        vision_audit._verify_saved_model(_reload_args(tmp_path), 7, [model])
    after = _state(model)
    assert all(torch.equal(after[name], before[name]) for name in before if name != "final_norm")
    # Only the perturbed head is put back; beyond it the live model now holds the checkpoint's copy.
    assert torch.equal(after["final_norm"][:32], before["final_norm"][:32])
    assert after["final_norm"][50] != before["final_norm"][50]
    assert not any(torch.isnan(value).any() for value in after.values() if value.is_floating_point())
    failure = json.loads((tmp_path / "audit/checkpoint-reload/rank-000.failed.json").read_text())
    assert failure["stage"] == "compare" and failure["loader_entered"] and failure["loader_returned"]
    assert [row["name"] for row in failure["mismatches"]] == ["final_norm"]
    assert failure["mismatches"][0]["tp_replicated"] is True
    assert not (tmp_path / "audit/checkpoint-reload/rank-000.json").exists()


def test_reload_failure_inside_the_loader_is_recorded_and_restores_nan_heads(tmp_path, monkeypatch):
    from miles.backends.megatron_utils import checkpoint

    model = _reload_model()
    before = _state(model)

    def broken(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt, *, args):
        raise RuntimeError("shard read failed")

    monkeypatch.setattr(checkpoint, "load_checkpoint", broken)
    with pytest.raises(RuntimeError, match="shard read failed"):
        vision_audit._verify_saved_model(_reload_args(tmp_path), 7, [model])
    after = _state(model)
    assert all(torch.equal(after[name], before[name]) for name in before)
    failure = json.loads((tmp_path / "audit/checkpoint-reload/rank-000.failed.json").read_text())
    assert failure["stage"] == "load" and failure["loader_entered"] and not failure["loader_returned"]
    assert failure["error_type"] == "RuntimeError" and "shard read failed" in failure["traceback"]
