"""With --overlap-grad-reduce: reduce every DDP bucket group that holds a vision parameter at finish_grad_sync only.

Megatron DDP (fdf82356 param_and_grad_buffer.py) issues a bucket group's reduce-scatter from the gradient-accumulator
hook as soon as every parameter in it reported in the step's last microbatch (register_grad_ready, once the first step
recorded the golden counts). Every DP rank must issue its groups in one order on the DP communicator. The language
model's backward is the same graph on every microbatch, so its groups are ready in one order everywhere. The vision
tower's are not: an image microbatch reports them in the tower's own backward order (a block's norm2 before its
attention, a LayerNorm's weight before its bias, ...), a text microbatch in the order of vision._ZeroGradsFor's
chain. With a bucket boundary between such parameters, a DP rank on a text and one on an image microbatch issue the
vision groups in different orders (F8-GUARD, CX4 2026-11-02: tiny HF tower, DP 2, bucket sizes 33 / 64 / 65), and NCCL
pairs different buckets or hangs.

So a group holding any vision parameter is never issued from the hook. Its register_grad_ready keeps Megatron's
counting and issues nothing; its finish_grad_sync issues it (after checking this step's counts against the golden
ones, as the hook would have) and waits. DistributedDataParallel.finish_grad_sync calls the groups in index order,
after the backward, on every rank: the same order everywhere, whatever the microbatches held. The first step already
issues every group there (Megatron's own path). A vision group is ready only once the tower's backward ran, at the
end of the backward (vision.py reports a text microbatch's zero grads at the input embeddings), so this gives up only
the overlap with the word-embedding backward.
"""

from __future__ import annotations

import logging

import torch.distributed as dist

from miles_plugins.models.qwen3_8_next.vision_optimizer import is_vision_parameter_name

logger = logging.getLogger(__name__)

MARKER = "_qwen38_deferred_grad_sync"


def _defer(group) -> None:
    if getattr(group, MARKER, False):  # config-access-exempt: our own marker on a Megatron bucket group
        return
    finish = group.finish_grad_sync

    def register_grad_ready(param, force_all_reduce=False):
        """Megatron's bookkeeping (per-parameter report counts in the last microbatch), without the dispatch."""
        assert group.ddp_config.overlap_grad_reduce
        if group.is_last_microbatch:
            assert param in group.param_to_bucket, "Param is not in the bucket group"
            counts = group.per_param_grad_ready_counts
            counts[param] = counts.get(param, 0) + 1

    def finish_grad_sync(force_all_reduce=False):
        if not group.is_first_batch and not group.grad_reduce_finished and group.grad_reduce_handle is None:
            counts, golden = group.per_param_grad_ready_counts, group.golden_per_param_grad_ready_counts
            if counts != golden:
                raise AssertionError(
                    f"Deferred vision bucket group: {len(counts)}/{len(group.params)} params have grad available, "
                    "or a parameter reported a different number of times than in the first step"
                )
            group.start_grad_sync(force_all_reduce=force_all_reduce)
        return finish(force_all_reduce=force_all_reduce)

    group.register_grad_ready = register_grad_ready
    group.finish_grad_sync = finish_grad_sync
    setattr(group, MARKER, True)


def bucket_groups(chunk) -> list:
    """The chunk's bucket groups in the order DistributedDataParallel.finish_grad_sync calls them."""
    return list(chunk.bucket_groups) + list(chunk.expert_parallel_bucket_groups)


def defer_vision_bucket_groups(model_chunks) -> list[str]:
    """Defer every bucket group holding a vision parameter (chunks with --overlap-grad-reduce); returns their tags
    "<chunk>:<index>". Asserts that no such group is left undeferred, and logs one line per rank."""
    deferred = []
    for c, chunk in enumerate(model_chunks):
        ddp_config = getattr(chunk, "ddp_config", None)  # config-access-exempt: only Megatron DDP chunks have one
        if ddp_config is None or not ddp_config.overlap_grad_reduce:
            continue
        vision = {p for name, p in chunk.named_parameters() if is_vision_parameter_name(name)}
        groups = bucket_groups(chunk)
        for index, group in enumerate(groups):
            if vision & group.params:
                _defer(group)
                deferred.append(f"{c}:{index}")
        left = [
            i
            for i, g in enumerate(groups)
            if vision & g.params and not getattr(g, MARKER, False)  # config-access-exempt: our own marker
        ]
        if left:
            raise AssertionError(f"bucket groups {left} of chunk {c} hold vision parameters but are not deferred")
    if any(getattr(chunk, "ddp_config", None) is not None for chunk in model_chunks):  # config-access-exempt: DDP
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        print(f"Q38_F8_DEFERRED rank={rank} groups=[{','.join(deferred)}]", flush=True)
    return deferred
