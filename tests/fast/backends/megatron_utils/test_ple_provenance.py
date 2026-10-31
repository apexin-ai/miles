"""A resumed run must read the PLE table bytes its checkpoint trained with.

The PLE n-gram table is re-read from --hf-checkpoint on every start and no Megatron checkpoint holds it, so
each saved iteration carries a manifest of those bytes, and no save may publish an iteration without it.

The tests run the real ``model.save`` / ``model.load_model_state`` and miles' ``load_checkpoint`` wrapper on a
small synthetic safetensors base. Megatron's checkpoint I/O is replaced by ``FakeMegatron``, which keeps the
two facts that matter here: a sync ``save_checkpoint`` writes the tracker before it returns, an async one
leaves it to a finalize that runs later; a load reads the tracker with Megatron's own ``read_metadata``.
"""

import json
import logging
import shutil
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.training.checkpointing import read_metadata
from safetensors.torch import save_file
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import checkpoint as checkpoint_module
from miles.backends.megatron_utils import model as model_module
from miles.backends.megatron_utils import ple_provenance

# two spawned gloo cases; the rest are file-level and take milliseconds
register_cpu_ci(est_time=40, suite="stage-b-cpu", labels=["megatron"])

TRACKER = "latest_checkpointed_iteration.txt"
MANIFEST = "ple_provenance.json"
PLE_LAYERS = (1, 3)
SHARDS = 2


def _prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}.ple.ple_embedding"


SHARD_KEY = f"{_prefix(1)}.ngram_embedding.shard_1.weight"
MULTIPLIER_KEY = f"{_prefix(3)}.layer_multipliers"


def _ple_tensors(layer: int) -> dict[str, torch.Tensor]:
    tensors = {
        f"{_prefix(layer)}.ngram_embedding.shard_{i}.weight": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
        + layer
        + i
        for i in range(SHARDS)
    }
    tensors[f"{_prefix(layer)}.layer_multipliers"] = torch.tensor([3, 5, 7], dtype=torch.int64) * layer
    tensors[f"{_prefix(layer)}.ngram_heads_offsets"] = torch.tensor([0, 11], dtype=torch.int64)
    tensors[f"{_prefix(layer)}.ngram_heads_vocab_sizes"] = torch.tensor([11, 13], dtype=torch.int64)
    return tensors


def _make_base(directory: Path, *, changed: str | None = None, ple: bool = True, layers: int = 4) -> Path:
    """A sharded HF checkpoint with PLE tensors on two layers; ``changed`` gets other bytes, same dtype/shape."""
    files = {
        "model-00001-of-00002.safetensors": {"model.language_model.embed_tokens.weight": torch.ones(8, 6)},
        "model-00002-of-00002.safetensors": {"model.language_model.norm.weight": torch.ones(6)},
    }
    if ple:
        files["model-00001-of-00002.safetensors"] |= _ple_tensors(PLE_LAYERS[0])
        files["model-00002-of-00002.safetensors"] |= _ple_tensors(PLE_LAYERS[1])
    directory.mkdir(parents=True)
    weight_map = {}
    for name, tensors in files.items():
        if changed in tensors:
            tensors[changed] = tensors[changed] + 1
        save_file(tensors, directory / name)
        weight_map |= dict.fromkeys(tensors, name)
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps({"model_type": "toy", "num_hidden_layers": layers}))
    return directory


class ProcessKilled(BaseException):
    """The trainer process dies here; nothing after this point runs."""


class FakeMegatron:
    """The checkpoint files Megatron's save_checkpoint writes, in the order it writes them."""

    def __init__(self, save_root: Path, *, async_save: bool = False, kill: str | None = None):
        self.save_root, self.async_save, self.kill = save_root, async_save, kill
        self.pending = []
        self.called = []

    def save_checkpoint(self, iteration, *args, **kwargs) -> None:
        self.called.append((iteration, (self.save_root / f"iter_{iteration:07d}" / MANIFEST).is_file()))
        if self.kill == "on_entry":
            raise ProcessKilled
        shard_dir = self.save_root / f"iter_{iteration:07d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "__0_0.distcp").write_bytes(b"weights")
        (shard_dir / "metadata.json").write_text("{}")
        if self.async_save:
            self.pending.append(partial(self._publish, iteration))
        else:
            # rank 0 writes the tracker after the barrier, before save_checkpoint returns
            self._publish(iteration)

    def finalize_async_save(self) -> None:
        while self.pending:
            self.pending.pop(0)()

    def _publish(self, iteration: int) -> None:
        (self.save_root / TRACKER).write_text(str(iteration))
        if self.kill == "after_tracker":
            raise ProcessKilled


def _fake_megatron_load(args):
    """Megatron's load_checkpoint answer: the tracker's iteration; 0 for a release or --finetune."""

    def load(**kwargs):
        iteration, release = read_metadata(str(Path(args.backend.load) / TRACKER))
        if args.backend.ckpt_step is not None:
            iteration = args.backend.ckpt_step
        return (0 if release or args.backend.finetune else iteration), 0

    return load


@pytest.fixture(autouse=True)
def _no_gpu_side_effects(monkeypatch):
    # each test starts like a new trainer process, and leaves no binding to later test modules
    monkeypatch.setattr(ple_provenance, "_run_manifest", None)
    monkeypatch.delenv("MILES_PLE_PROVENANCE_ADOPT", raising=False)
    monkeypatch.delenv("MILES_PLE_MANIFEST", raising=False)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "clear_memory", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "is_first_replica_megatron_main_rank", lambda: False)
    monkeypatch.setattr(model_module, "should_disable_forward_pre_hook", lambda args: False)
    monkeypatch.setattr(model_module, "is_lora_model", lambda model: False)


def _args(base: Path, load: Path, save: Path, **overrides):
    load.mkdir(parents=True, exist_ok=True)
    values = dict(
        hf_checkpoint=str(base),
        load=str(load),
        save=str(save),
        ci_test=False,
        finetune=False,
        lora_rank=0,
        lora_adapter_path=None,
        spec=None,
    )
    return make_trainer_args(**(values | overrides))


def _load(monkeypatch, args):
    """A new trainer process loads --load."""
    monkeypatch.setattr(checkpoint_module, "_load_checkpoint_megatron", _fake_megatron_load(args))
    return model_module.load_model_state(
        args,
        model=[],
        optimizer=None,
        opt_param_scheduler=None,
        role="actor",
        checkpointing_context=None,
        resume_from_ckpt=True,
    )


def _save(monkeypatch, args, iteration: int, megatron: FakeMegatron) -> None:
    monkeypatch.setattr(model_module, "save_checkpoint", megatron.save_checkpoint)
    model_module.save(args, iteration, [torch.nn.Linear(2, 2)], None, None)


def _train(monkeypatch, tmp_path: Path, base: Path, iterations=(4,)) -> Path:
    """First run: starts from an empty --load and saves each iteration."""
    ckpt = tmp_path / "ckpt"
    args = _args(base, tmp_path / "empty", ckpt)
    _load(monkeypatch, args)
    megatron = FakeMegatron(ckpt)
    for iteration in iterations:
        _save(monkeypatch, args, iteration, megatron)
    return ckpt


def _resume(monkeypatch, tmp_path: Path, base: Path, ckpt: Path, **overrides):
    return _load(monkeypatch, _args(base, ckpt, ckpt, **overrides))


def _assert_latest_has_manifest(ckpt: Path) -> None:
    latest = ckpt / f"iter_{int((ckpt / TRACKER).read_text()):07d}"
    assert (latest / MANIFEST).is_file(), f"the tracker publishes {latest.name}, which has no {MANIFEST}"


class TestResume:
    def test_the_same_bytes_resume(self, tmp_path, monkeypatch) -> None:
        """Positive control: the manifest travels with the checkpoint and the unchanged base matches it."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        assert (ckpt / "iter_0000004" / MANIFEST).is_file()
        assert _resume(monkeypatch, tmp_path, base, ckpt).loaded_rollout_id == 4

    @pytest.mark.parametrize("key", [SHARD_KEY, MULTIPLIER_KEY])
    def test_the_same_path_with_other_ple_bytes_is_refused(self, tmp_path, monkeypatch, key) -> None:
        """A table shard or a hash parameter replaced under the same path fails the resume."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        shutil.rmtree(base)
        _make_base(base, changed=key)
        with pytest.raises(RuntimeError, match=r"1 PLE tensors differ in bytes.*" + key.replace(".", r"\.")):
            _resume(monkeypatch, tmp_path, base, ckpt)

    def test_another_path_with_the_same_bytes_resumes_with_a_warning(self, tmp_path, monkeypatch, caplog) -> None:
        """The binding is the content: a moved or re-downloaded base with identical bytes is fine."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        moved = tmp_path / "moved"
        shutil.copytree(base, moved)
        shutil.rmtree(base)
        with caplog.at_level(logging.WARNING):
            assert _resume(monkeypatch, tmp_path, moved, ckpt).loaded_rollout_id == 4
        assert f"hf_checkpoint: {base.resolve()} -> {moved.resolve()}" in caplog.text

    def test_a_changed_config_is_refused(self, tmp_path, monkeypatch) -> None:
        """The config decides the architecture the table plugs into."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        (base / "config.json").write_text(json.dumps({"model_type": "toy", "num_hidden_layers": 5}))
        with pytest.raises(RuntimeError, match="config.json differs"):
            _resume(monkeypatch, tmp_path, base, ckpt)

    @pytest.mark.parametrize("change", ["no_ple", "no_index"])
    def test_a_base_without_the_table_is_refused(self, tmp_path, monkeypatch, change) -> None:
        """A checkpoint bound to a PLE table cannot resume on a base that lost it."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        if change == "no_ple":
            shutil.rmtree(base)
            _make_base(base, ple=False)
        else:
            (base / "model.safetensors.index.json").unlink()
        with pytest.raises(RuntimeError, match="has no PLE tensors"):
            _resume(monkeypatch, tmp_path, base, ckpt)

    def test_a_trained_checkpoint_without_a_manifest_needs_an_explicit_adoption(self, tmp_path, monkeypatch) -> None:
        """Checkpoints written before this check: refused, or the current base adopted on request."""
        base = _make_base(tmp_path / "base")
        ckpt = tmp_path / "ckpt"
        FakeMegatron(ckpt).save_checkpoint(4)
        with pytest.raises(RuntimeError, match="iter_0000004/ple_provenance.json is missing"):
            _resume(monkeypatch, tmp_path, base, ckpt)
        monkeypatch.setenv("MILES_PLE_PROVENANCE_ADOPT", "1")
        assert _resume(monkeypatch, tmp_path, base, ckpt).loaded_rollout_id == 4

    def test_a_trained_rollout_0_checkpoint_is_a_resume(self, tmp_path, monkeypatch) -> None:
        """Iteration 0 written by training is checked like any other; only its absence of a manifest is refused."""
        base = _make_base(tmp_path / "base")
        legacy = tmp_path / "legacy"
        FakeMegatron(legacy).save_checkpoint(0)
        with pytest.raises(RuntimeError, match="iter_0000000/ple_provenance.json is missing"):
            _resume(monkeypatch, tmp_path, base, legacy)

        ckpt = _train(monkeypatch, tmp_path, base, iterations=(0,))
        shutil.rmtree(base)
        _make_base(base, changed=SHARD_KEY)
        with pytest.raises(RuntimeError, match="1 PLE tensors differ"):
            _resume(monkeypatch, tmp_path, base, ckpt)


class TestNewBinding:
    """A load that does not continue a trained checkpoint binds the run to the current base."""

    def _saved_manifest(self, monkeypatch, args, iteration=1) -> dict:
        _save(monkeypatch, args, iteration, FakeMegatron(Path(args.backend.save)))
        return json.loads((Path(args.backend.save) / f"iter_{iteration:07d}" / MANIFEST).read_text())

    def test_a_release_checkpoint_starts_a_binding(self, tmp_path, monkeypatch) -> None:
        """A converted checkpoint is untrained; it needs no manifest and no adoption."""
        base = _make_base(tmp_path / "base")
        release = tmp_path / "converted"
        (release / "release").mkdir(parents=True)
        (release / TRACKER).write_text("release")
        args = _args(base, release, tmp_path / "ckpt")
        assert _load(monkeypatch, args).loaded_rollout_id == 0
        assert self._saved_manifest(monkeypatch, args)["hf_checkpoint"] == str(base.resolve())

    def test_a_directory_named_release_with_a_trained_tracker_is_a_resume(self, tmp_path, monkeypatch) -> None:
        """What Megatron reads decides, not the directory name."""
        base = _make_base(tmp_path / "base")
        root = tmp_path / "release"
        FakeMegatron(root).save_checkpoint(0)
        with pytest.raises(RuntimeError, match="is missing"):
            _resume(monkeypatch, tmp_path, base, root)

    def test_an_empty_load_directory_starts_a_binding(self, tmp_path, monkeypatch) -> None:
        base = _make_base(tmp_path / "base")
        args = _args(base, tmp_path / "empty", tmp_path / "ckpt")
        _load(monkeypatch, args)
        assert self._saved_manifest(monkeypatch, args)["ple"][SHARD_KEY]["shape"] == [4, 6]

    def test_finetune_from_a_trained_checkpoint_binds_the_new_base(self, tmp_path, monkeypatch) -> None:
        """--finetune starts a new run; the next save records the base it now reads."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base)
        other = _make_base(tmp_path / "other", changed=SHARD_KEY)
        args = _args(other, ckpt, tmp_path / "finetuned", finetune=True)
        _load(monkeypatch, args)
        old = json.loads((ckpt / "iter_0000004" / MANIFEST).read_text())
        new = self._saved_manifest(monkeypatch, args)
        assert new["ple"][SHARD_KEY] != old["ple"][SHARD_KEY]

    def test_a_base_without_ple_writes_no_manifest(self, tmp_path, monkeypatch) -> None:
        """Every other model: no hashing, no sidecar, no collective."""
        base = _make_base(tmp_path / "base", ple=False)
        ckpt = tmp_path / "ckpt"
        args = _args(base, tmp_path / "empty", ckpt)
        _load(monkeypatch, args)
        megatron = FakeMegatron(ckpt)
        _save(monkeypatch, args, 3, megatron)
        assert megatron.called == [(3, False)]
        assert not (ckpt / "iter_0000003" / MANIFEST).exists()
        assert _resume(monkeypatch, tmp_path, base, ckpt).loaded_rollout_id == 3


class TestSaveNeverPublishesWithoutManifest:
    """The tracker is Megatron's commit point; the manifest must be durable before it can name an iteration."""

    @pytest.mark.parametrize("async_save", [False, True], ids=["sync", "async"])
    def test_a_kill_right_after_the_tracker_leaves_a_resumable_latest(self, tmp_path, monkeypatch, async_save) -> None:
        """Whether the tracker is written inside save_checkpoint or by a later finalize."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base, iterations=(1,))
        args = _args(base, ckpt, ckpt)
        _resume(monkeypatch, tmp_path, base, ckpt)
        megatron = FakeMegatron(ckpt, async_save=async_save, kill="after_tracker")
        with pytest.raises(ProcessKilled):
            _save(monkeypatch, args, 2, megatron)
            megatron.finalize_async_save()
        assert (ckpt / TRACKER).read_text() == "2"
        assert megatron.called == [(2, True)], "save_checkpoint ran before the manifest was on disk"
        _assert_latest_has_manifest(ckpt)
        assert _resume(monkeypatch, tmp_path, base, ckpt).loaded_rollout_id == 2

    def test_a_kill_between_the_manifest_and_megatron_keeps_the_previous_latest(self, tmp_path, monkeypatch) -> None:
        """An iteration directory with only a manifest is never named by the tracker."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base, iterations=(1,))
        args = _args(base, ckpt, ckpt)
        _resume(monkeypatch, tmp_path, base, ckpt)
        with pytest.raises(ProcessKilled):
            _save(monkeypatch, args, 2, FakeMegatron(ckpt, kill="on_entry"))
        assert (ckpt / TRACKER).read_text() == "1"
        _assert_latest_has_manifest(ckpt)
        assert _resume(monkeypatch, tmp_path, base, ckpt).loaded_rollout_id == 1

    def test_a_failed_manifest_write_saves_nothing(self, tmp_path, monkeypatch) -> None:
        """Megatron is not called, so the tracker stays at the last complete iteration."""
        base = _make_base(tmp_path / "base")
        ckpt = _train(monkeypatch, tmp_path, base, iterations=(1,))
        args = _args(base, ckpt, ckpt)
        _resume(monkeypatch, tmp_path, base, ckpt)
        (ckpt / "iter_0000002").write_text("a file where the iteration directory goes")
        megatron = FakeMegatron(ckpt)
        with pytest.raises(RuntimeError, match="not saving iteration 2"):
            _save(monkeypatch, args, 2, megatron)
        assert megatron.called == []
        assert (ckpt / TRACKER).read_text() == "1"
        assert _resume(monkeypatch, tmp_path, base, ckpt).loaded_rollout_id == 1


def _hash_worker(rank: int, world_size: int, port: int, *, base: str, out: str) -> None:
    from miles.backends.megatron_utils.ple_provenance import compute_manifest

    init_gloo(rank, world_size, port=port)
    try:
        manifest = compute_manifest(base)
        Path(out, f"rank{rank}.json").write_text(json.dumps(manifest))
    finally:
        dist.destroy_process_group()


def test_the_hashing_split_over_ranks_gives_the_single_process_manifest(tmp_path) -> None:
    """Each rank hashes a slice of the keys; every rank ends with the full, identical manifest."""
    from miles.backends.megatron_utils.ple_provenance import compute_manifest

    base = _make_base(tmp_path / "base")
    run_multiprocess(partial(_hash_worker, base=str(base), out=str(tmp_path)), world_size=2)
    expected = compute_manifest(base)
    assert len(expected["ple"]) == len(PLE_LAYERS) * (SHARDS + 3)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text()) == expected


def _failed_write_worker(rank: int, world_size: int, port: int, *, base: str, save: str, out: str) -> None:
    from miles.backends.megatron_utils import ple_provenance

    init_gloo(rank, world_size, port=port)
    try:
        ple_provenance.on_load(base, trained_checkpoint_dir=None)
        if rank == 0:
            Path(save).mkdir()
            (Path(save) / "iter_0000002").write_text("a file where the iteration directory goes")
        dist.barrier()
        megatron = FakeMegatron(Path(save))
        model_module.save_checkpoint = megatron.save_checkpoint
        model_module.should_disable_forward_pre_hook = lambda args: False
        model_module.is_lora_model = lambda model: False
        args = make_trainer_args(hf_checkpoint=base, save=save, ci_test=False)
        try:
            model_module.save(args, 2, [torch.nn.Linear(2, 2)], None, None)
            result = "saved"
        except RuntimeError as e:
            result = str(e)
        Path(out, f"rank{rank}.json").write_text(json.dumps({"result": result, "called": megatron.called}))
    finally:
        dist.destroy_process_group()


def test_a_manifest_write_failing_on_rank_0_stops_every_rank_before_megatron(tmp_path) -> None:
    """Only rank 0 writes; the others must not enter save_checkpoint's collectives alone."""
    base = _make_base(tmp_path / "base")
    worker = partial(_failed_write_worker, base=str(base), save=str(tmp_path / "ckpt"), out=str(tmp_path))
    run_multiprocess(worker, world_size=2)
    for rank in range(2):
        outcome = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert outcome["called"] == []
        assert outcome["result"].startswith("not saving iteration 2: cannot write")
