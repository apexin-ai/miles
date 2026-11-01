"""CPU contracts for lazy Adam on the merger's text-only steps (vision_lazy, user 2026-11-01 15:10 UTC).

Real: Megatron's HybridDeviceOptimizer (HDO) step, its once-attached CPU gradient copies, its param copy-back hook,
its state sync and load_state_dict hooks; torch's fused CPU AdamW exactly as Megatron builds it
(get_megatron_optimizer: lr, weight_decay, betas, eps, bias_correction, fused=True, param_update_in_fp32=True); and
Megatron's DistributedOptimizer.state_dict step extraction (one step per optimizer, asserted).

Stubbed: CUDA streams (no-ops), and HDO's _get_sub_optimizer_param_groups, which on a CPU-only host would make no
offload copies (it offloads only param.is_cuda tensors). _OffloadingHDO reproduces it for offload fraction 1.0 with
every parameter treated as a GPU parameter: a CPU copy, then an fp32 copy of a bf16 one, mapped exactly as
hybrid_optimizer.py:251-300 maps them. DistributedOptimizer's sharding is a stand-in with its real attribute names
(model_param_group_index_map, optimizer); the DP1 shard of a parameter is the parameter itself.
"""

import contextlib
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles_plugins.models.qwen3_8_next.vision import _ZeroGradsFor

BETAS, EPS, LR, WD = (0.9, 0.98), 1e-8, 1e-2, 0.1
_X = torch.tensor([[1.0, -2.0, 0.5, 3.0]])


class _FakeStream:
    def wait_stream(self, other):
        pass

    def record_event(self):
        return SimpleNamespace(synchronize=lambda: None, wait=lambda stream: None)


@pytest.fixture(autouse=True)
def cpu_streams(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path_factory.mktemp('pg') / 'store'}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()


def _hdo_class():
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    class _OffloadingHDO(HybridDeviceOptimizer):
        def _get_sub_optimizer_param_groups(self, offload_fraction):
            assert offload_fraction == 1.0
            cpu_param_groups, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param, param_to_fp32_param = [], {}, {}, {}
            for group in self.param_groups:
                cpu_group = group.copy()
                cpu_group["params"] = []
                for orig_param in group["params"]:
                    param = orig_param.detach().clone()  # .cpu().pin_memory() on a real GPU parameter
                    if self.param_update_in_fp32 and param.dtype != torch.float32:
                        param = param.detach().clone().float()
                        param_to_fp32_param[orig_param] = param
                    gpu_params_map_cpu_copy[orig_param] = param
                    cpu_copys_map_gpu_param[param] = orig_param
                    cpu_group["params"].append(param)
                cpu_param_groups.append(cpu_group)
            return cpu_param_groups, [], gpu_params_map_cpu_copy, cpu_copys_map_gpu_param, param_to_fp32_param

    return _OffloadingHDO


class _Vision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Linear(4, 4, bias=False)
        self.merger = torch.nn.Module()
        self.merger.norm = torch.nn.LayerNorm(4)
        self.merger.fc = torch.nn.Linear(4, 4)

    def forward(self, x, grid_thw=None, return_dict=True):
        return self.merger.fc(self.merger.norm(self.blocks(x)))


class Toy(torch.nn.Module):
    """vision_model + a decoder in bf16 (like the model's params), merger trainable (projector) or all (full)."""

    def __init__(self, scope="projector", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.decoder = torch.nn.Linear(4, 4)
        self.vision_model = _Vision()
        for name, p in self.vision_model.named_parameters():
            p.requires_grad_(scope == "full" or name.startswith("merger."))
        self.to(torch.bfloat16)

    def forward(self, x, *, image):
        x = x.to(torch.bfloat16)
        if image:
            return self.decoder(x + self.vision_model(x, grid_thw=torch.tensor([[1, 2, 2]]))).float().square().sum()
        out = self.decoder(x).float().square().sum()
        trainable = [p for p in self.vision_model.parameters() if p.requires_grad]
        return _ZeroGradsFor.apply(out, *trainable) if trainable and torch.is_grad_enabled() else out


class DistOpt:
    """DistributedOptimizer stand-in: its attribute names, DP1 shards (= the parameters), HDO inside."""

    def __init__(self, model):
        decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
        no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
        groups = [
            {"params": decay, "wd_mult": 1.0, "lr_mult": 1.0, "is_expert_parallel": False},
            {"params": no_decay, "wd_mult": 0.0, "lr_mult": 1.0, "is_expert_parallel": False, "weight_decay": 0.0},
        ]
        self.optimizer = _hdo_class()(
            groups,
            offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            overlap_cpu_optimizer_d2h_h2d=False,
            pin_cpu_grads=False,
            pin_cpu_params=False,
            param_update_in_fp32=True,
            lr=LR,
            weight_decay=WD,
            betas=BETAS,
            eps=EPS,
            bias_correction=True,
            fused=True,
        )
        self.model_param_group_index_map = {
            p: (g, o) for g, group in enumerate(self.optimizer.param_groups) for o, p in enumerate(group["params"])
        }

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none)

    def step(self):
        self.optimizer.step()
        return True, 1.0, 0

    def megatron_state_dict(self):
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

        return DistributedOptimizer.state_dict(SimpleNamespace(optimizer=self.optimizer, grad_scaler=None))


def _vision_lazy():
    from miles_plugins.models.qwen3_8_next import vision_lazy

    return vision_lazy


def _setup(scope="projector", mode="merger", seed=0):
    model = Toy(scope, seed)
    opt = DistOpt(model)
    lazy = _vision_lazy().install([model], opt, mode=mode) if mode != "off" else None
    return model, opt, lazy


def _train(model, opt, images):
    for image in images:
        opt.zero_grad()
        model(_X, image=image).backward()
        opt.step()


def _merger(model):
    return {n: p.detach().clone() for n, p in model.vision_model.merger.named_parameters()}


def _inner_state(opt, model, names):
    hdo = opt.optimizer
    named = dict(model.named_parameters())
    out = {}
    for name in names:
        inner = hdo.param_to_inner_param[named[name]]
        state = next(o.state[inner] for o in hdo.cpu_optimizers if inner in o.state)
        out[name] = {k: v.detach().clone() for k, v in state.items() if isinstance(v, torch.Tensor)} | {
            "master": inner.detach().clone()
        }
    return out


MERGER = ["vision_model.merger.norm.weight", "vision_model.merger.norm.bias", "vision_model.merger.fc.weight", "vision_model.merger.fc.bias"]


# --- the problem: without lazy, text-only steps move the merger ---


def test_without_lazy_a_text_step_moves_the_merger_by_momentum_and_decay():
    model, opt, _ = _setup(mode="off")
    _train(model, opt, [True])
    before = _merger(model)
    _train(model, opt, [False])
    assert any(not torch.equal(p, before[n]) for n, p in model.vision_model.merger.named_parameters())


# --- lazy: on a step whose global batch has no image the merger and its Adam state stay put ---


def test_a_text_step_leaves_the_merger_and_its_adam_state_bitwise_unchanged():
    model, opt, lazy = _setup()
    _train(model, opt, [True])
    params, states = _merger(model), _inner_state(opt, model, MERGER)
    _train(model, opt, [False, False])
    assert all(torch.equal(p, params[n]) for n, p in model.vision_model.merger.named_parameters())
    after = _inner_state(opt, model, MERGER)
    for name in MERGER:
        for key in ("exp_avg", "exp_avg_sq", "master"):
            assert torch.equal(after[name][key], states[name][key]), (name, key)
        assert after[name]["step"].item() == states[name]["step"].item() + 2  # step follows the global step
    assert lazy.skipped_updates == 2 and lazy.updates == 3


def test_the_language_trains_exactly_as_without_lazy():
    images = [True, False, False, True, False]
    off_model, off_opt, _ = _setup(mode="off", seed=1)
    on_model, on_opt, _ = _setup(seed=1)
    _train(off_model, off_opt, images)
    _train(on_model, on_opt, images)
    assert torch.equal(off_model.decoder.weight, on_model.decoder.weight)
    assert torch.equal(off_model.decoder.bias, on_model.decoder.bias)


def test_lazy_matches_an_fp64_lazy_adam_reference():
    """Merger update only on image steps; bias correction with the global step; AdamW decoupled decay."""
    images = [True, False, False, True, False, True]
    model, opt, _ = _setup(seed=2)
    named = dict(model.named_parameters())
    ref = {n: {"p": named[n].detach().double(), "m": torch.zeros_like(named[n], dtype=torch.float64),
               "v": torch.zeros_like(named[n], dtype=torch.float64)} for n in MERGER}
    for t, image in enumerate(images, start=1):
        opt.zero_grad()
        model(_X, image=image).backward()
        grads = {n: named[n].grad.detach().double().clone() for n in MERGER}
        opt.step()
        if not image:
            assert all(not g.any() for g in grads.values())
            continue
        for n in MERGER:
            r, g = ref[n], grads[n]
            wd = WD if named[n].ndim > 1 else 0.0
            r["p"] = r["p"] * (1 - LR * wd)
            r["m"] = BETAS[0] * r["m"] + (1 - BETAS[0]) * g
            r["v"] = BETAS[1] * r["v"] + (1 - BETAS[1]) * g * g
            m_hat, v_hat = r["m"] / (1 - BETAS[0] ** t), r["v"] / (1 - BETAS[1] ** t)
            r["p"] = r["p"] - LR * m_hat / (v_hat.sqrt() + EPS)
    state = _inner_state(opt, model, MERGER)
    for n in MERGER:
        # fused fp32 AdamW vs fp64: a few fp32 roundings per step
        torch.testing.assert_close(state[n]["master"].double(), ref[n]["p"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(state[n]["exp_avg"].double(), ref[n]["m"], rtol=1e-5, atol=1e-8)


def test_every_parameter_keeps_one_step_and_megatron_s_state_dict_accepts_it():
    model, opt, _ = _setup()
    _train(model, opt, [False, False, True, False])  # the merger's state is created on a skipped step
    hdo = opt.optimizer
    steps = {o.state[p]["step"].item() for o in hdo.cpu_optimizers for p in o.state}
    assert steps == {4.0}
    assert all(group["step"] == 4 for group in opt.megatron_state_dict()["optimizer"]["param_groups"])


def test_a_skipped_step_before_any_image_creates_zero_moments():
    model, opt, _ = _setup()
    before = _merger(model)
    _train(model, opt, [False])
    assert all(torch.equal(p, before[n]) for n, p in model.vision_model.merger.named_parameters())
    for state in _inner_state(opt, model, MERGER).values():
        assert state["step"].item() == 1 and not state["exp_avg"].any() and not state["exp_avg_sq"].any()


def test_the_once_attached_cpu_gradient_is_re_attached_after_a_skip():
    # HDO attaches the CPU copy's .grad once; lazy detaches it on a skipped step and must put it back.
    model, opt, lazy = _setup()
    _train(model, opt, [True, False])
    before = _merger(model)
    _train(model, opt, [True])
    assert all(not torch.equal(p, before[n]) for n, p in model.vision_model.merger.named_parameters() if p.ndim > 1)
    assert lazy.skipped_updates == 1


def test_full_scope_merger_mode_leaves_the_vit_to_adam_and_vision_mode_covers_it():
    model, opt, _ = _setup(scope="full", mode="merger")
    _train(model, opt, [True])
    vit = model.vision_model.blocks.weight.detach().clone()
    merger = _merger(model)
    _train(model, opt, [False])
    assert not torch.equal(model.vision_model.blocks.weight, vit)  # merger mode: the ViT still moves (wd, momentum)
    assert all(torch.equal(p, merger[n]) for n, p in model.vision_model.merger.named_parameters())
    model, opt, _ = _setup(scope="full", mode="vision")
    _train(model, opt, [True])
    vit = model.vision_model.blocks.weight.detach().clone()
    _train(model, opt, [False])
    assert torch.equal(model.vision_model.blocks.weight, vit)


# --- resume: a save between skips and a reload reproduce the uninterrupted run bit for bit ---


def _snapshot(model, opt):
    return {"model": {k: v.clone() for k, v in model.state_dict().items()}, "opt": opt.optimizer.state_dict()}


def test_resume_mid_text_run_matches_the_uninterrupted_run(tmp_path):
    images = [True, False, False, True, False, True]
    ref_model, ref_opt, _ = _setup(seed=3)
    _train(ref_model, ref_opt, images)
    model, opt, _ = _setup(seed=3)
    _train(model, opt, images[:2])
    torch.save(_snapshot(model, opt), tmp_path / "ckpt.pt")
    resumed, ropt, lazy = _setup(seed=4)  # different init: everything must come from the checkpoint
    saved = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    resumed.load_state_dict(saved["model"])
    ropt.optimizer.load_state_dict(saved["opt"])  # HDO's post-load hook rebuilds its CPU copies and optimizers
    _train(resumed, ropt, images[2:])
    assert lazy.skipped_updates == 2  # the wrappers survive the rebuild
    for (n, a), (_, b) in zip(ref_model.named_parameters(), resumed.named_parameters(), strict=True):
        assert torch.equal(a, b), n
    ref_state, state = _inner_state(ref_opt, ref_model, MERGER), _inner_state(ropt, resumed, MERGER)
    for n in MERGER:
        for key in ("step", "exp_avg", "exp_avg_sq", "master"):
            assert torch.equal(ref_state[n][key], state[n][key]), (n, key)


# --- the global decision: every rank skips or none does ---


def _decide(rank, world, store, local_images, out):
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        vision_lazy = _vision_lazy()
        tower = torch.nn.Linear(1, 1)
        tower._qwen38_lazy_images = local_images[rank]
        result = vision_lazy.global_images([tower], torch.device("cpu"))
        with open(f"{out}.{rank}", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("local_images,expected", [([1, 0], 1), ([0, 0], 0), ([0, 3], 3)])
def test_the_image_count_is_summed_over_all_ranks(tmp_path, local_images, expected):
    out = str(tmp_path / "result")
    mp.start_processes(
        _decide, args=(2, str(tmp_path / "store"), local_images, out), nprocs=2, join=True, start_method="spawn"
    )
    assert [json.load(open(f"{out}.{r}")) for r in range(2)] == [expected, expected]


def test_images_are_counted_only_in_gradient_enabled_forwards_and_reset_by_zero_grad():
    model, opt, lazy = _setup()
    with torch.no_grad():
        model(_X, image=True)  # a log-prob style forward: not a training image
    assert lazy.local_images() == 0
    model(_X, image=True).backward()
    assert lazy.local_images() == 1
    opt.zero_grad()
    assert lazy.local_images() == 0


# --- refusals ---


def test_unsupported_optimizers_and_scopes_are_refused():
    vision_lazy = _vision_lazy()
    model = Toy("projector")
    plain = SimpleNamespace(optimizer=torch.optim.AdamW(model.parameters()), model_param_group_index_map={})
    with pytest.raises(ValueError, match="HybridDeviceOptimizer"):
        vision_lazy.install([model], plain, mode="merger")
    with pytest.raises(ValueError, match="no trainable"):
        vision_lazy.install([Toy("none")], DistOpt(Toy("projector")), mode="merger")
    with pytest.raises(ValueError, match="mode"):
        vision_lazy.install([model], DistOpt(model), mode="sometimes")


def test_install_twice_is_refused():
    model, opt, _ = _setup()
    with pytest.raises(ValueError, match="already"):
        _vision_lazy().install([model], opt, mode="merger")
