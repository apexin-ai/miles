"""Cheap guards for --overlap-grad-reduce (F8) that stay on in the formal run (F8-XCHECK-A 4.1 / 5.3).

Installed on every DDP chunk with overlap_grad_reduce and the distributed optimizer. Q38_F8_GUARD=0 turns them all
off; Q38_F8_GUARD_FORCE=1 also installs them without overlap (an overlap-off control arm).

(a) Q38_F8_GUARD_NORMS (default 1): each rank's local L2 norm of every bucket right before its reduce-scatter. With
    Megatron's NaN / large-grad check on (check_for_nan_in_grad, the default) this is the tensor check_grads already
    computes (param_and_grad_buffer.py:417), kept instead of dropped: no extra kernel; otherwise one norm per bucket.
    No host sync in the backward; read only when an alert fires.
(b) Q38_F8_GUARD_WRITE_CHECK=N (default 8; 0 = off): "written after issue" on a rotating 1/N of the buckets (bucket k
    when k % N == step % N). At the reduce-scatter call (after Megatron's gradient scaling, on the current stream,
    before NCCL starts), the float64 sum of the bucket's elements outside this rank's shard; after finish_grad_sync's
    wait, the same sum. The reduce-scatter writes only this rank's shard, so a difference means the bucket was
    written after its reduce-scatter was issued. Compared once per step (one host sync, after the backward).
(c) Q38_F8_GUARD_ALERT_FACTOR (default 2; 0 = off), Q38_F8_GUARD_WARMUP (default 10 steps): a step whose grad norm is
    above factor x the median grad norm of the earlier steps (after the warm-up), or any (b) mismatch, prints
    `Q38_F8_ALERT step=<s> rank=<r> reason=<grad_norm|write_after_issue> ...` and writes this rank's per-bucket local
    norms and (b) results to Q38_F8_GUARD_DIR (default $Q38_RUN_DIR/audit/f8-guard) as step-<s>-rank-<rrrrr>.json.
    The grad norm is global, so every rank decides alike: rank 0 prints the grad-norm line, a rank with a (b)
    mismatch prints its own. The alert does not stop the run.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

_GUARD = None


def _env(name, default):
    return os.environ.get(name, default)


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _other_sum(group, data):
    """float64 sum of the bucket's elements outside this rank's reduce-scatter shard (no temporary copy)."""
    size, rank = group.intra_distributed_optimizer_instance_size, group.intra_distributed_optimizer_instance_rank
    shard = data.numel() // size
    total = torch.zeros((), dtype=torch.float64, device=data.device)
    for part in (data[: rank * shard], data[(rank + 1) * shard :]):
        if part.numel():
            total = total + torch.sum(part, dtype=torch.float64)
    return total


class _Guard:
    def __init__(self, *, norms: bool, every: int, factor: float, warmup: int, directory: Path | None):
        self.norms, self.every, self.factor, self.warmup, self.directory = norms, every, factor, warmup, directory
        self.order = {}  # tag -> position among all buckets (the write-check rotation)
        self.by_ptr = {}  # bucket grad_data pointer -> (tag, group)
        self.step = 0
        self.local_norms, self.pre, self.post = {}, {}, {}
        self.history = []

    def checked(self, tag) -> bool:
        return self.every > 0 and self.order[tag] % self.every == self.step % self.every

    # ---- per bucket group -----------------------------------------------------------------------------------
    def wrap(self, chunk_index, chunk) -> None:
        for g_index, group in enumerate(list(chunk.bucket_groups) + list(chunk.expert_parallel_bucket_groups)):
            tags = [f"{chunk_index}:{g_index}:{b}" for b in range(len(group.buckets))]
            for tag, bucket in zip(tags, group.buckets, strict=True):
                self.order[tag] = len(self.order)
                self.by_ptr[bucket.grad_data.data_ptr()] = (tag, group)
            self._wrap_group(group, tags)

    def _wrap_group(self, group, tags) -> None:
        ddp = group.ddp_config
        start, finish, check = group.start_grad_sync, group.finish_grad_sync, group.check_grads

        def check_grads(check_for_nan_or_inf, check_for_large):
            """Megatron's check_grads, keeping the per-bucket norm it hands the rerun state machine."""
            from megatron.core.rerun_state_machine import get_rerun_state_machine

            machine = get_rerun_state_machine()
            own = machine.__dict__.get("validate_result")
            validate, seen = own or type(machine).validate_result.__get__(machine), []

            def validate_result(result, *args, **kwargs):
                if not seen or seen[-1] is not result:
                    seen.append(result)
                return validate(result, *args, **kwargs)

            machine.validate_result = validate_result
            try:
                check(check_for_nan_or_inf, check_for_large)
            finally:
                if own is None:
                    del machine.validate_result
                else:
                    machine.validate_result = own
            for tag, norm in zip(tags, seen, strict=False):
                self.local_norms[tag] = norm.detach()

        def start_grad_sync(*args, **kwargs):
            issuing = not (group.is_first_batch and group.grad_reduce_handle is not None)
            if issuing and self.norms and not (ddp.check_for_nan_in_grad or ddp.check_for_large_grads):
                for tag, bucket in zip(tags, group.buckets, strict=True):
                    self.local_norms[tag] = bucket.grad_data.norm(p=2).detach()
            return start(*args, **kwargs)

        def finish_grad_sync(*args, **kwargs):
            out = finish(*args, **kwargs)
            for tag, bucket in zip(tags, group.buckets, strict=True):
                if tag in self.pre and tag not in self.post:
                    self.post[tag] = _other_sum(group, bucket.grad_data)
            return out

        if self.norms:
            group.check_grads = check_grads
        group.start_grad_sync = start_grad_sync
        group.finish_grad_sync = finish_grad_sync

    def reduce_scatter(self, real):
        """Wraps Megatron's module-level dist_reduce_scatter_func: (b)'s sum right before the collective."""

        def dist_reduce_scatter_func(output, input, *args, **kwargs):
            hit = self.by_ptr.get(input.data_ptr())
            if hit is not None and self.checked(hit[0]) and hit[0] not in self.pre:
                self.pre[hit[0]] = _other_sum(hit[1], input)
            return real(output, input, *args, **kwargs)

        dist_reduce_scatter_func._f8_guard_real = real
        return dist_reduce_scatter_func

    # ---- per step -------------------------------------------------------------------------------------------
    def begin_step(self, rollout_id: int) -> None:
        self.step = rollout_id + 1
        self.local_norms, self.pre, self.post = {}, {}, {}

    def end_step(self, grad_norm) -> list[str]:
        """Compare (b), decide (c); returns the alert reasons (already printed and written)."""
        tags = sorted(self.pre, key=self.order.get)
        mismatched = []
        if tags:
            nan = torch.tensor(float("nan"), dtype=torch.float64)
            pre = torch.stack([self.pre[t] for t in tags]).cpu().tolist()  # the step's one host sync
            post = torch.stack([self.post.get(t, nan.to(self.pre[t].device)) for t in tags]).cpu().tolist()
            mismatched = [
                {"bucket": t, "pre": a, "post": b} for t, a, b in zip(tags, pre, post, strict=True) if a != b
            ]
        reasons = []
        g = float(grad_norm) if grad_norm is not None else float("nan")
        if self.factor > 0 and len(self.history) >= max(self.warmup, 1):
            median = statistics.median(self.history)
            if g > self.factor * median:
                reasons.append(("grad_norm", f"grad_norm={g:.6g} median={median:.6g} factor={self.factor:g}"))
        if mismatched:
            reasons.append(("write_after_issue", f"buckets={[m['bucket'] for m in mismatched][:8]}"))
        if g == g and g > 0:
            self.history.append(g)
        if reasons:
            self._alert(reasons, g, mismatched, tags)
        return [r for r, _ in reasons]

    def _alert(self, reasons, grad_norm, mismatched, checked) -> None:
        rank = _rank()
        for reason, detail in reasons:
            if reason == "write_after_issue" or rank == 0:
                print(f"Q38_F8_ALERT step={self.step} rank={rank} reason={reason} {detail}", flush=True)
        if self.directory is None:
            return
        tags = sorted(self.local_norms, key=self.order.get)
        values = torch.stack([self.local_norms[t].float() for t in tags]).cpu().tolist() if tags else []
        record = {
            "step": self.step,
            "rank": rank,
            "reasons": [r for r, _ in reasons],
            "grad_norm": grad_norm,
            "local_bucket_norms": dict(zip(tags, values, strict=True)),
            "write_check": {"checked": checked, "mismatched": mismatched},
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"step-{self.step}-rank-{rank:05d}.json").write_text(json.dumps(record))


def install(model_chunks):
    """Install the guards on the chunks that need them (module docstring); returns the guard, or None."""
    global _GUARD
    from megatron.core.distributed import param_and_grad_buffer as pgb

    real = getattr(pgb.dist_reduce_scatter_func, "_f8_guard_real", None)  # config-access-exempt: our own marker
    if real is not None:  # a previous install: unwrap first
        pgb.dist_reduce_scatter_func = real
    _GUARD = None
    if _env("Q38_F8_GUARD", "1") == "0":
        return None
    force = _env("Q38_F8_GUARD_FORCE", "0") == "1"
    chunks = []
    for i, c in enumerate(model_chunks):
        ddp = getattr(c, "ddp_config", None)  # config-access-exempt: only Megatron DDP chunks have one
        if ddp is not None and ddp.use_distributed_optimizer and (ddp.overlap_grad_reduce or force):
            chunks.append((i, c))
    if not chunks:
        return None
    run = os.environ.get("Q38_RUN_DIR")
    directory = _env("Q38_F8_GUARD_DIR", str(Path(run) / "audit" / "f8-guard") if run else "")
    guard = _Guard(
        norms=_env("Q38_F8_GUARD_NORMS", "1") == "1",
        every=int(_env("Q38_F8_GUARD_WRITE_CHECK", "8")),
        factor=float(_env("Q38_F8_GUARD_ALERT_FACTOR", "2")),
        warmup=int(_env("Q38_F8_GUARD_WARMUP", "10")),
        directory=Path(directory) if directory else None,
    )
    for i, chunk in chunks:
        guard.wrap(i, chunk)
    if guard.every > 0:
        pgb.dist_reduce_scatter_func = guard.reduce_scatter(pgb.dist_reduce_scatter_func)
    _GUARD = guard
    return guard


def begin_step(rollout_id: int) -> None:
    if _GUARD is not None:
        _GUARD.begin_step(rollout_id)


def end_step(grad_norm) -> None:
    if _GUARD is not None:
        _GUARD.end_step(grad_norm)
