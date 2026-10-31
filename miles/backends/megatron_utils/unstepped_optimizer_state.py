"""Give a never-stepped distributed optimizer the state a checkpoint needs, without moving any weight.

Megatron's DistributedOptimizer builds its state (master params, moments) on the first step, and saving reads
that state unconditionally: a run whose every step so far was skipped for having no supervised token fails the
save with ``KeyError: 'master_param'``. Megatron's own load-time fill cannot stand in, because it steps at the
scheduled lr (or, for HybridDeviceOptimizer.dummy_step, with random gradients).

The fill here steps once with zero gradients at lr = weight decay = 0, which leaves every parameter bit-identical,
then resets the moments and step clocks so the state equals what a fresh optimizer holds before its first step.
"""

import logging
from collections.abc import Iterator

import torch
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states

logger = logging.getLogger(__name__)


def materialize_unstepped_optimizer_states(optimizer) -> int:
    """Fill every DistributedOptimizer under ``optimizer`` that has not stepped yet; return how many were filled."""
    filled = 0
    for dist_opt in _distributed_optimizers(optimizer):
        if dist_opt.is_stub_optimizer or not _never_stepped(dist_opt.optimizer):
            continue
        _zero_effect_step(dist_opt)
        reset_optimizer_states(dist_opt)
        filled += 1
    return filled


def _distributed_optimizers(optimizer) -> Iterator[DistributedOptimizer]:
    if isinstance(optimizer, DistributedOptimizer):
        yield optimizer
    for child in getattr(
        optimizer, "chained_optimizers", ()
    ):  # config-access-exempt: only ChainedOptimizer has children
        yield from _distributed_optimizers(child)


def _never_stepped(inner) -> bool:
    params = [p for group in inner.param_groups for p in group["params"] if p.numel() > 0]
    # .get: other readers index the state defaultdict and leave empty entries behind
    return bool(params) and not any(inner.state.get(p) for p in params)


def _zero_effect_step(dist_opt: DistributedOptimizer) -> None:
    inner = dist_opt.optimizer
    # same attribute DistributedOptimizer fills before a real step; HDO and TE FusedAdam read it in that mode
    grad_attr = "decoupled_grad" if dist_opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 else "grad"
    saved = [(group["lr"], group["weight_decay"]) for group in inner.param_groups]
    try:
        for group in inner.param_groups:
            group["lr"] = 0.0
            group["weight_decay"] = 0.0
            for param in group["params"]:
                if param.numel() > 0:
                    setattr(param, grad_attr, torch.zeros_like(param))
        inner.step()
    finally:
        for group, (lr, weight_decay) in zip(inner.param_groups, saved, strict=True):
            group["lr"] = lr
            group["weight_decay"] = weight_decay
            for param in group["params"]:
                if getattr(param, grad_attr, None) is not None:  # config-access-exempt: grad attr chosen above
                    setattr(param, grad_attr, None)
        if isinstance(inner, HybridDeviceOptimizer):
            # its children copied lr = 0 at the start of the step; the next step would re-sync them anyway
            inner._sync_hdo_param_groups_to_sub_optimizers()
