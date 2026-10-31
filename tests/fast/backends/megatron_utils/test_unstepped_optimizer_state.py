"""A save after only skipped (unsupervised) steps must find the optimizer state Megatron's save reads, and filling
it must move no weight and leave the optimizer exactly where a fresh one starts."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.core.optimizer import CPUAdam
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from miles.backends.megatron_utils.unstepped_optimizer_state import materialize_unstepped_optimizer_states
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu")

LR, WD = 1e-2, 0.1
SHAPES = ((4, 3), (5,))


class _Stream:
    def wait_stream(self, stream):
        pass

    def record_event(self):
        return SimpleNamespace(synchronize=lambda: None, wait=lambda stream: None)


@contextmanager
def cpu_streams():
    """HDO keeps CUDA streams for its D2H/H2D copies; on the CPU pod they are no-ops."""
    with patch.object(torch.cuda, "current_stream", return_value=_Stream()), patch.object(
        torch.cuda, "stream", side_effect=lambda stream: nullcontext()
    ):
        yield


def _offload_everything(self, fraction):
    """HDO's real partition at offload_fraction=1 with param_update_in_fp32, minus pin_memory and is_cuda."""
    assert fraction == 1.0
    cpu_groups, to_copy, to_orig = [], {}, {}
    for group in self.param_groups:
        copies = []
        for param in group["params"]:
            copy = param.detach().clone().float()
            to_copy[param], to_orig[copy] = copy, param
            copies.append(copy)
        cpu_groups.append(dict(group, params=copies))
    return cpu_groups, [], to_copy, to_orig, dict(to_copy)


def bf16_params(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*shape, generator=gen).bfloat16()) for shape in SHAPES]


def hybrid(params):
    with patch.object(HybridDeviceOptimizer, "_get_sub_optimizer_param_groups", _offload_everything), cpu_streams():
        return HybridDeviceOptimizer(
            [{"params": params, "lr": LR, "weight_decay": WD}],
            offload_fraction=1.0,
            cpu_optimizer_cls=CPUAdam,
            gpu_optimizer_cls=None,
            param_update_in_fp32=True,
            pin_cpu_grads=False,
            pin_cpu_params=False,
            overlap_cpu_optimizer_d2h_h2d=False,
            lr=LR,
        )


def distributed(inner, precision_aware):
    dist_opt = DistributedOptimizer.__new__(DistributedOptimizer)
    dist_opt.optimizer = inner
    dist_opt.is_stub_optimizer = inner is None
    dist_opt.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware)
    params = [p for group in inner.param_groups for p in group["params"]] if inner is not None else []
    dist_opt.model_param_group_index_map = {p: (0, i) for i, p in enumerate(params)}
    return dist_opt


def saved_tensors(dist_opt):
    """What DistributedOptimizer's dp_reshardable save reads per param."""
    params = [p for group in dist_opt.optimizer.param_groups for p in group["params"]]
    return [DistributedOptimizer._get_main_param_and_optimizer_states(dist_opt, p) for p in params]


def hybrid_step(optimizer, grads):
    for param, grad in zip(optimizer.param_groups[0]["params"], grads, strict=True):
        param.decoupled_grad = grad
    with cpu_streams():
        optimizer.step()


def hybrid_snapshot(optimizer):
    out = [p.detach().clone() for p in optimizer.param_groups[0]["params"]]
    for state in optimizer.state.values():
        out += [state[k].detach().clone() for k in sorted(state)]
    return out


def assert_bitwise(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype and torch.equal(x, y)


def test_hybrid_save_after_skipped_steps_finds_fresh_state_and_moves_nothing():
    optimizer = hybrid(bf16_params())
    dist_opt = distributed(optimizer, precision_aware=True)
    with pytest.raises(KeyError, match="master_param"):  # SMOKE FIX-A-5 C2: the save before any step
        saved_tensors(dist_opt)
    weights = [p.detach().clone() for p in optimizer.param_groups[0]["params"]]
    masters = [p.detach().clone() for p in optimizer.param_to_fp32_param.values()]

    with cpu_streams():
        assert materialize_unstepped_optimizer_states(SimpleNamespace(chained_optimizers=[dist_opt])) == 1

    assert_bitwise([p.detach() for p in optimizer.param_groups[0]["params"]], weights)
    assert_bitwise([p.detach() for p in optimizer.param_to_fp32_param.values()], masters)
    for tensors in saved_tensors(dist_opt):
        assert set(tensors) == {"param", "exp_avg", "exp_avg_sq", "step"}
        assert not tensors["exp_avg"].any() and not tensors["exp_avg_sq"].any() and tensors["step"] == 0
    for opt in (optimizer, *optimizer.sub_optimizers):
        assert [(g["lr"], g["weight_decay"]) for g in opt.param_groups] == [(LR, WD)]
    assert all(p.decoupled_grad is None for p in optimizer.param_groups[0]["params"])

    # the first real step after the fill lands exactly where a never-filled optimizer's first step does
    fresh = hybrid(bf16_params())
    grads = [torch.randn(*shape, generator=torch.Generator().manual_seed(1)).bfloat16() for shape in SHAPES]
    hybrid_step(optimizer, grads)
    hybrid_step(fresh, grads)
    assert_bitwise(hybrid_snapshot(optimizer), hybrid_snapshot(fresh))

    before = hybrid_snapshot(optimizer)
    with cpu_streams():
        assert materialize_unstepped_optimizer_states(dist_opt) == 0
    assert_bitwise(hybrid_snapshot(optimizer), before)


def fp32_adamw(seed=0):
    gen = torch.Generator().manual_seed(seed)
    params = [torch.nn.Parameter(torch.randn(*shape, generator=gen)) for shape in SHAPES]
    return torch.optim.AdamW(params, lr=LR, weight_decay=WD)


def test_plain_grad_fill_moves_nothing_where_megatrons_load_fill_would():
    optimizer = fp32_adamw()
    weights = [p.detach().clone() for p in optimizer.param_groups[0]["params"]]

    assert materialize_unstepped_optimizer_states(distributed(optimizer, precision_aware=False)) == 1

    assert_bitwise([p.detach() for p in optimizer.param_groups[0]["params"]], weights)
    assert all(p.grad is None for p in optimizer.param_groups[0]["params"])
    for state in optimizer.state.values():
        assert state["step"] == 0 and not state["exp_avg"].any() and not state["exp_avg_sq"].any()

    # positive control: Megatron's load-time fill steps at the scheduled lr, so weight decay moves the weights
    control = fp32_adamw()
    DistributedOptimizer._init_optimizer_states_with_dummy_values(distributed(control, precision_aware=False))
    assert not torch.equal(control.param_groups[0]["params"][0].detach(), weights[0])


def test_stub_and_empty_shards_are_left_alone():
    empty = torch.optim.AdamW([torch.nn.Parameter(torch.empty(0))], lr=LR, weight_decay=WD)
    real = fp32_adamw()
    chain = SimpleNamespace(
        chained_optimizers=[
            distributed(None, precision_aware=False),
            distributed(empty, precision_aware=False),
            distributed(real, precision_aware=False),
        ]
    )

    assert materialize_unstepped_optimizer_states(chain) == 1
    assert not empty.state and len(real.state) == len(SHAPES)
    assert materialize_unstepped_optimizer_states(None) == 0
