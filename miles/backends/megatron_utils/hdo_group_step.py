"""Keep the Adam step of HybridDeviceOptimizer's GPU sub-optimizer across a resume (M-023, NVIDIA/Megatron-LM#7785).

With --optimizer-offload-fraction < 1, HDO (Megatron's --optimizer-cpu-offload) steps part of each parameter shard
on the GPU with TE FusedAdam, which keeps one step per param group, and the rest on the CPU with torch AdamW, which
keeps one step per parameter. Megatron saves one step (taken from the torch AdamW states) in the param groups, and a
resume writes it into HDO's own param groups. HDO then copies every key of its param groups into its sub-optimizers
at the start of every step (_sync_hdo_param_groups_to_sub_optimizers), "step" included, so after a resume at step S
the GPU groups go back to S on every step and stay at S + 1: their bias correction stops advancing. A fresh run is
not affected (HDO's own groups have no "step" until a resume).

install() makes that sync keep each GPU group's own step. Right after HDO rebuilds its sub-optimizers on a load, the
GPU groups start from the checkpoint's step (copied from HDO's groups), which is then kept and advanced. After every
HDO step the GPU groups' step must equal the newest CPU step; any difference fails the step.

Megatron only saves the step from torch AdamW states, so an HDO with a GPU optimizer and no CPU optimizer
(offload fraction 0) would save no step at all: refused. With offload fraction 1.0 there is no GPU optimizer and
nothing to install (the caller does not install then).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_MARKER = "_miles_hdo_group_step"


def _hdos(optimizer):
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    for inner in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        hdo = getattr(inner, "optimizer", None)
        if isinstance(hdo, HybridDeviceOptimizer):
            yield hdo


def _value(step) -> float:
    return float(step.item() if isinstance(step, torch.Tensor) else step)


def cpu_step(hdo):
    """The newest per-parameter step of HDO's CPU AdamW states, or None before any CPU state exists."""
    steps = [_value(state["step"]) for opt in hdo.cpu_optimizers for state in opt.state.values() if "step" in state]
    return max(steps) if steps else None


def gpu_steps(hdo) -> list:
    """The step of every non-empty GPU param group (None where it has none yet)."""
    if hdo.gpu_optimizer is None:
        return []
    return [group.get("step") for group in hdo.gpu_optimizer.param_groups if group["params"]]


def _wrap(hdo):
    original_sync, original_step = hdo._sync_hdo_param_groups_to_sub_optimizers, hdo.step

    def sync():
        gpu = hdo.gpu_optimizer
        kept = [group.get("step") for group in gpu.param_groups] if gpu is not None else []
        original_sync()
        if gpu is None or hdo.gpu_optimizer is not gpu:
            return
        for group, step in zip(gpu.param_groups, kept, strict=True):
            if step is not None:
                group["step"] = step

    def step(closure=None):
        result = original_step(closure)
        reference = cpu_step(hdo)
        if reference is not None:
            wrong = [s for s in gpu_steps(hdo) if s is not None and _value(s) != reference]
            if wrong:
                raise RuntimeError(
                    f"HDO GPU sub-optimizer step {sorted({_value(s) for s in wrong})} != CPU step {reference} (M-023)"
                )
        return result

    hdo._sync_hdo_param_groups_to_sub_optimizers, hdo.step = sync, step


def install(optimizer) -> int:
    """Wrap every HDO under the optimizer that has a GPU sub-optimizer; returns how many were wrapped."""
    wrapped = 0
    for hdo in _hdos(optimizer):
        if getattr(hdo, _MARKER, False):  # config-access-exempt: our own marker
            raise ValueError("hdo_group_step is already installed on this HybridDeviceOptimizer")
        if hdo.gpu_optimizer is None:
            continue
        if not hdo.cpu_optimizers:
            raise ValueError(
                "--optimizer-offload-fraction puts every parameter of a HybridDeviceOptimizer on the GPU: Megatron "
                "would save no Adam step (it reads it from the CPU AdamW states). Use a fraction > 0."
            )
        _wrap(hdo)
        setattr(hdo, _MARKER, True)
        wrapped += 1
    logger.info("HDO GPU group step kept across resumes on %d optimizer(s) (M-023)", wrapped)
    return wrapped
