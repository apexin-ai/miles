"""CPU contracts for lazy vision Adam with --optimizer-offload-fraction < 1, and for the HDO GPU group step (M-023).

Real: Megatron's HybridDeviceOptimizer (HDO) step, gradient hand-off, copy-back hooks, state sync, load_state_dict
hooks (which re-split the parameters with the running fraction); torch's fused CPU AdamW as Megatron builds it;
vision_lazy and hdo_group_step as installed by model.py (hdo_group_step first).

Stubbed:
  - CUDA streams (no-ops) and HDO's device moves: _FractionHDO reproduces hybrid_optimizer.py
    _get_sub_optimizer_param_groups with every parameter treated as a GPU parameter (the first offload_fraction of
    the elements, in param-group order, get a CPU copy; the rest stay "on the GPU"; bf16 ones get an fp32 copy) and
    keeps loaded state on the CPU instead of moving the GPU part to cuda.
  - TE FusedAdam (needs CUDA): _GroupStepAdam keeps its contract (fused_adam.py step: one "step" per param group,
    advanced on every step; zero moments created for every parameter; a parameter whose .grad is None is skipped)
    and uses torch's fused AdamW arithmetic, so that CPU-vs-"GPU" results can be compared bit for bit here. On a real
    GPU, TE FusedAdam and CPU AdamW round differently: there the comparison is approximate.
  - The checkpoint: _megatron_save / _megatron_load do what DistributedOptimizer.state_dict / load_state_dict do with
    an HDO (distrib_optimizer.py:897-935, 1087-1097): one step, taken from the CPU AdamW states, in the param groups;
    no per-parameter step (dp_reshardable stores it as LocalNonpersistentObject); on load every state gets it back.
"""

import io

import pytest
import torch
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from tests.fast.backends.megatron_utils.test_qwen38_vision_lazy import (  # noqa: F401 (autouse fixtures)
    _X,
    BETAS,
    EPS,
    LR,
    MERGER,
    WD,
    Toy,
    cpu_streams,
    process_group,
)
from torch.optim.adamw import adamw

from miles.backends.megatron_utils import hdo_group_step


class _GroupStepAdam(torch.optim.Optimizer):
    """TE FusedAdam's contract with torch's fused AdamW arithmetic (see the module docstring)."""

    def __init__(self, params):
        super().__init__(params, defaults=dict(lr=LR, betas=BETAS, eps=EPS, weight_decay=WD))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            if not group["params"]:
                continue
            group["step"] = group["step"] + 1 if "step" in group else 1
            for p in group["params"]:
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                if p.grad is None:
                    continue
                step = torch.tensor(float(group["step"] - 1))  # fused AdamW advances it to the group step
                adamw(
                    [p],
                    [p.grad],
                    [state["exp_avg"]],
                    [state["exp_avg_sq"]],
                    [],
                    [step],
                    fused=True,
                    amsgrad=False,
                    beta1=group["betas"][0],
                    beta2=group["betas"][1],
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=False,
                )


class _FractionHDO(HybridDeviceOptimizer):
    def _get_sub_optimizer_param_groups(self, offload_fraction):
        params = [p for group in self.param_groups for p in group["params"]]
        offload_threshold = sum(p.numel() for p in params) * offload_fraction
        offloaded = 0
        cpu_groups, gpu_groups, gpu_to_cpu, cpu_to_gpu, to_fp32 = [], [], {}, {}, {}
        for group in self.param_groups:
            gpu_group, cpu_group = group.copy(), group.copy()
            gpu_group["params"], cpu_group["params"] = [], []
            for orig in group["params"]:
                param, cpu_copy = orig, False
                if offloaded < offload_threshold:
                    param = param.detach().clone()  # .cpu().pin_memory() on a real GPU parameter
                    offloaded += param.numel()
                    cpu_copy = True
                if self.param_update_in_fp32 and param.dtype != torch.float32:
                    param = param.detach().clone().float()
                    to_fp32[orig] = param
                if cpu_copy:
                    gpu_to_cpu[orig] = param
                    cpu_to_gpu[param] = orig
                (cpu_group if cpu_copy else gpu_group)["params"].append(param)
            if gpu_group["params"]:
                gpu_groups.append(gpu_group)
            if cpu_group["params"]:
                cpu_groups.append(cpu_group)
        return cpu_groups, gpu_groups, gpu_to_cpu, cpu_to_gpu, to_fp32

    def _move_new_state_to_right_device(self):
        for optimizer in self.sub_optimizers:
            for param, state in optimizer.state.items():
                orig = self.inner_param_to_orig_param.get(param, param)
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        self.state[orig][k] = state[k] = v.to("cpu")


class FracDistOpt:
    """DistributedOptimizer stand-in (DP1 shards = the parameters) over an HDO with the given offload fraction."""

    def __init__(self, model, fraction):
        decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
        groups = [
            {"params": decay, "wd_mult": 1.0, "lr_mult": 1.0, "is_expert_parallel": False},
            {"params": no_decay, "wd_mult": 0.0, "lr_mult": 1.0, "is_expert_parallel": False, "weight_decay": 0.0},
        ]
        self.optimizer = _FractionHDO(
            groups,
            offload_fraction=fraction,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=_GroupStepAdam,
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


def _setup(fraction, mode="merger", seed=0, scope="projector", group_step=True):
    from miles_plugins.models.qwen3_8_next import vision_lazy

    model = Toy(scope, seed)
    opt = FracDistOpt(model, fraction)
    if group_step and fraction < 1.0:  # as model.py installs it
        hdo_group_step.install(opt)
    lazy = vision_lazy.install([model], opt, mode=mode) if mode != "off" else None
    return model, opt, lazy


def _train(model, opt, images):
    for image in images:
        opt.zero_grad()
        model(_X, image=image).backward()
        opt.step()


def _side(opt, model):
    """{name: 'cpu' | 'gpu'}: which HDO sub-optimizer steps each trainable parameter."""
    hdo = opt.optimizer
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = "cpu" if p in hdo.gpu_params_map_cpu_copy else "gpu"
            inner = hdo.param_to_inner_param[p]
            owner = hdo.gpu_optimizer if out[name] == "gpu" else None
            if owner is not None:
                assert any(q is inner for g in owner.param_groups for q in g["params"])
    return out


def _state(opt, model, names):
    hdo = opt.optimizer
    named = dict(model.named_parameters())
    out = {}
    for name in names:
        inner = hdo.param_to_inner_param[named[name]]
        state = next(o.state[inner] for o in hdo.sub_optimizers if inner in o.state)
        out[name] = {k: v.detach().clone() for k, v in state.items() if k in ("exp_avg", "exp_avg_sq")}
        out[name]["master"] = inner.detach().clone()
    return out


def _merger(model):
    return {n: p.detach().clone() for n, p in model.vision_model.merger.named_parameters()}


# Toy, projector scope, in HDO order: decay [decoder.weight 16, merger.fc.weight 16],
# no-decay [decoder.bias 4, merger.norm.weight 4, merger.norm.bias 4, merger.fc.bias 4]: 48 elements. HDO offloads
# whole parameters while fewer than fraction x 48 elements are offloaded: 0.3 -> decoder.weight only (the merger is all
# on the GPU side), 0.5 -> also merger.fc.weight (the merger is split across both sides).
SPLIT_03 = {
    "decoder.weight": "cpu",
    "vision_model.merger.fc.weight": "gpu",
    "decoder.bias": "gpu",
    "vision_model.merger.norm.weight": "gpu",
    "vision_model.merger.norm.bias": "gpu",
    "vision_model.merger.fc.bias": "gpu",
}
SPLIT_05 = SPLIT_03 | {"vision_model.merger.fc.weight": "cpu"}


# --- the split ---


@pytest.mark.parametrize("fraction,expected", [(0.3, SPLIT_03), (0.5, SPLIT_05)])
def test_hdo_splits_by_element_count_in_group_order_and_lazy_follows_the_split(fraction, expected):
    model, opt, lazy = _setup(fraction)
    assert _side(opt, model) == expected
    cpu, gpu = lazy._split(opt.optimizer)
    merger_sides = [expected[n] for n in MERGER]
    assert len(cpu) == merger_sides.count("cpu") and len(gpu) == merger_sides.count("gpu")


def test_fraction_1_keeps_every_lazy_parameter_on_the_cpu_and_installs_no_group_step():
    model, opt, lazy = _setup(1.0)
    assert set(_side(opt, model).values()) == {"cpu"} and opt.optimizer.gpu_optimizer is None
    assert len(lazy._split(opt.optimizer)[1]) == 0
    assert hdo_group_step.install(opt) == 0


# --- lazy on both sides ---


@pytest.mark.parametrize("fraction", [0.3, 0.5])
def test_a_text_step_leaves_cpu_and_gpu_lazy_parameters_and_their_adam_state_bitwise_unchanged(fraction):
    model, opt, lazy = _setup(fraction)
    _train(model, opt, [True])
    params, states = _merger(model), _state(opt, model, MERGER)
    _train(model, opt, [False, False])
    assert all(torch.equal(p, params[n]) for n, p in model.vision_model.merger.named_parameters())
    after = _state(opt, model, MERGER)
    for name in MERGER:
        for key in ("exp_avg", "exp_avg_sq", "master"):
            assert torch.equal(after[name][key], states[name][key]), (name, key)
    assert lazy.skipped_updates == 2
    _train(model, opt, [True])  # and an image step updates both sides again
    assert all(not torch.equal(p, params[n]) for n, p in model.vision_model.merger.named_parameters())


def test_without_lazy_a_text_step_moves_the_gpu_side_merger():
    model, opt, _ = _setup(0.3, mode="off")
    _train(model, opt, [True])
    before = model.vision_model.merger.fc.weight.detach().clone()  # on the GPU side at 0.3
    _train(model, opt, [False])
    assert not torch.equal(model.vision_model.merger.fc.weight, before)


IMAGES = [True, False, False, True, False, True, False]


@pytest.mark.parametrize("fraction", [0.3, 0.5])
def test_lazy_with_fraction_below_1_trains_bitwise_like_fraction_1(fraction):
    # same Adam arithmetic on both sides here (see _GroupStepAdam), so any difference is lazy / step bookkeeping
    ref_model, ref_opt, _ = _setup(1.0, seed=5)
    model, opt, _ = _setup(fraction, seed=5)
    _train(ref_model, ref_opt, IMAGES)
    _train(model, opt, IMAGES)
    for (n, a), (_, b) in zip(ref_model.named_parameters(), model.named_parameters(), strict=True):
        assert torch.equal(a, b), n
    ref, got = _state(ref_opt, ref_model, MERGER), _state(opt, model, MERGER)
    for n in MERGER:
        for key in ("exp_avg", "exp_avg_sq", "master"):
            assert torch.equal(ref[n][key], got[n][key]), (n, key)


def test_the_gpu_group_step_and_every_cpu_step_stay_equal_to_the_global_step():
    model, opt, _ = _setup(0.5)
    _train(model, opt, [False, True, False, False])
    hdo = opt.optimizer
    cpu_steps = {s["step"].item() for o in hdo.cpu_optimizers for s in o.state.values()}
    assert cpu_steps == {4.0} and set(hdo_group_step.gpu_steps(hdo)) == {4}


# --- resume: Megatron's checkpoint content does not depend on the split, the split follows the running fraction ---


def _megatron_save(opt):
    hdo = opt.optimizer
    sd = hdo.state_dict()
    step = hdo_group_step.cpu_step(hdo)
    for group in sd["param_groups"]:
        group["step"] = int(step)
    for state in sd["state"].values():
        state.pop("step", None)
    assert all(set(s) == {"exp_avg", "exp_avg_sq", "master_param"} for s in sd["state"].values())
    buffer = io.BytesIO()
    torch.save(sd, buffer)
    return buffer.getvalue()


def _megatron_load(opt, blob):
    sd = torch.load(io.BytesIO(blob), weights_only=False)
    (step,) = {group["step"] for group in sd["param_groups"]}
    for state in sd["state"].values():
        state["step"] = torch.tensor(step, dtype=torch.float32)
    opt.optimizer.load_state_dict(sd)  # HDO's post-load hook re-splits with the running fraction


def _resume(save_fraction, load_fraction, images, at, group_step=True):
    model, opt, _ = _setup(save_fraction, seed=6, group_step=group_step)
    _train(model, opt, images[:at])
    blob, weights = _megatron_save(opt), {k: v.clone() for k, v in model.state_dict().items()}
    resumed, ropt, lazy = _setup(load_fraction, seed=7, group_step=group_step)
    resumed.load_state_dict(weights)
    _megatron_load(ropt, blob)
    _train(resumed, ropt, images[at:])
    return resumed, ropt, lazy


@pytest.mark.parametrize("save_fraction,load_fraction", [(1.0, 0.5), (0.5, 1.0), (0.5, 0.5), (0.3, 0.5), (1.0, 0.3)])
def test_resume_with_a_changed_fraction_matches_the_uninterrupted_fraction_1_run(save_fraction, load_fraction):
    ref_model, ref_opt, _ = _setup(1.0, seed=6)
    _train(ref_model, ref_opt, IMAGES)
    resumed, ropt, lazy = _resume(save_fraction, load_fraction, IMAGES, at=2)
    assert lazy.skipped_updates == IMAGES[2:].count(False)
    for (n, a), (_, b) in zip(ref_model.named_parameters(), resumed.named_parameters(), strict=True):
        assert torch.equal(a, b), n
    ref, got = _state(ref_opt, ref_model, MERGER), _state(ropt, resumed, MERGER)
    for n in MERGER:
        for key in ("exp_avg", "exp_avg_sq", "master"):
            assert torch.equal(ref[n][key], got[n][key]), (n, key)
    if load_fraction < 1.0:
        assert set(hdo_group_step.gpu_steps(ropt.optimizer)) == {len(IMAGES)}


# --- M-023: without hdo_group_step the GPU group step sticks at S + 1 after a resume ---


def test_positive_control_m023_without_the_fix_the_gpu_group_step_sticks_after_resume():
    _, ropt, _ = _resume(0.5, 0.5, IMAGES, at=2, group_step=False)
    hdo = ropt.optimizer
    assert hdo_group_step.cpu_step(hdo) == len(IMAGES)
    assert set(hdo_group_step.gpu_steps(hdo)) == {3}  # S + 1 with S = 2, however many steps followed


def test_a_gpu_group_step_out_of_line_fails_the_step():
    model, opt, _ = _setup(0.5)
    _train(model, opt, [True, True])
    opt.optimizer.gpu_optimizer.param_groups[0]["step"] = 7
    with pytest.raises(RuntimeError, match="M-023"):
        _train(model, opt, [True])


def test_reset_optimizer_states_after_a_resume_keeps_the_gpu_step_at_the_reset_value():
    from miles.backends.megatron_utils.optimizer_state_reset import _reset

    resumed, ropt, _ = _resume(0.5, 0.5, IMAGES, at=2)
    hdo = ropt.optimizer
    for sub in hdo.sub_optimizers:  # what reset_optimizer_states does to HDO's leaves (Adam / CPUAdam schema)
        _reset(sub, frozenset({"exp_avg", "exp_avg_sq"}))
    _train(resumed, ropt, [True])  # HDO's own param groups still say step 2; the GPU groups must not go back to it
    assert hdo_group_step.cpu_step(hdo) == 1 and set(hdo_group_step.gpu_steps(hdo)) == {1}


def test_fraction_0_is_refused():
    model = Toy("projector", 0)
    with pytest.raises(ValueError, match="fraction > 0"):
        hdo_group_step.install(FracDistOpt(model, 0.0))


def test_install_twice_is_refused():
    model = Toy("projector", 0)
    opt = FracDistOpt(model, 0.5)
    hdo_group_step.install(opt)
    with pytest.raises(ValueError, match="already"):
        hdo_group_step.install(opt)
