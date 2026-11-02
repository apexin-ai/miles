"""Fail-closed terminal checks of Megatron DDP's gradient-ready counting with --overlap-grad-reduce (CX3-F8-02).

Megatron (fdf82356 param_and_grad_buffer.py) counts, in the step's last microbatch, how often each parameter's
gradient accumulator reported (register_grad_ready) and issues a bucket group's reduce-scatter once the counts equal
the first step's ("golden"). Nothing checks what comes after: a parameter that reports again once its group was issued
(an extra reentrant backward, a late weight gradient) adds into a buffer that is already being reduced, and
finish_grad_sync only waits on the handle; reset() then drops the counts. On every bucket group (dense, expert and the
deferred vision groups) of chunks with --overlap-grad-reduce:
  (1) register_grad_ready in the last microbatch of a non-first step raises when the group's reduce-scatter was
      already issued (or finished) this step: group index, parameter name, its count and golden count;
  (2) finish_grad_sync of a non-first step first compares the whole parameter -> count dict of the last microbatch
      with the golden one and raises on any difference.
Plain Python on Megatron's own counters: no device work, no host sync. Step 1 (golden recording: Megatron issues every
group at finish_grad_sync) and no_sync microbatches (is_last_microbatch False: not counted) are left alone.
"""

from __future__ import annotations

MARKER = "_qwen38_grad_ready_checked"


class GradReadyCountError(AssertionError):
    pass


def _diff(counts: dict, golden: dict, names: dict) -> list[str]:
    out = []
    for p in set(counts) | set(golden):
        c, g = counts.get(p, 0), golden.get(p, 0)
        if c != g:
            out.append(f"{names.get(p, '?')}: count {c}, golden {g}")
    return sorted(out)


def _wrap(group, index: int, names: dict) -> None:
    if getattr(group, MARKER, False):  # config-access-exempt: our own marker on a Megatron bucket group
        return
    register, finish = group.register_grad_ready, group.finish_grad_sync

    def register_grad_ready(param, *args, **kwargs):
        if group.is_last_microbatch and not group.is_first_batch:
            if group.grad_reduce_handle is not None or group.grad_reduce_finished:
                count = group.per_param_grad_ready_counts.get(param, 0) + 1
                golden = group.golden_per_param_grad_ready_counts.get(param, 0)
                raise GradReadyCountError(
                    f"bucket group {index}: {names.get(param, '?')} reported a gradient after the group's "
                    f"reduce-scatter was issued this step (count {count}, golden {golden})"
                )
        return register(param, *args, **kwargs)

    def finish_grad_sync(*args, **kwargs):
        if not group.is_first_batch and not group.grad_reduce_finished:
            counts, golden = group.per_param_grad_ready_counts, group.golden_per_param_grad_ready_counts
            if counts != golden:
                raise GradReadyCountError(
                    f"bucket group {index}: the last microbatch's gradient reports differ from the first step's: "
                    f"{_diff(counts, golden, names)[:8]}"
                )
        return finish(*args, **kwargs)

    group.register_grad_ready = register_grad_ready
    group.finish_grad_sync = finish_grad_sync
    setattr(group, MARKER, True)


def install(model_chunks) -> int:
    """Wrap every bucket group of the chunks with --overlap-grad-reduce; returns how many groups were wrapped."""
    n = 0
    for chunk in model_chunks:
        ddp_config = getattr(chunk, "ddp_config", None)  # config-access-exempt: only Megatron DDP chunks have one
        if ddp_config is None or not ddp_config.overlap_grad_reduce:
            continue
        names = {p: name for name, p in chunk.named_parameters()}
        for index, group in enumerate(list(chunk.bucket_groups) + list(chunk.expert_parallel_bucket_groups)):
            _wrap(group, index, names)
            n += 1
    return n
