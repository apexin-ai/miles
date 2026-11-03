"""Fail-closed check that every rank issues its bucket groups' gradient reductions in one order (CX3 F8-RC).

With --overlap-grad-reduce Megatron DDP issues a bucket group's reduce-scatter from the gradient hooks as soon as its
report counts equal the first step's (golden). Counts say nothing about order: if two ranks of one communicator issued
two groups in opposite orders, NCCL would pair different buckets; with equal shapes nothing fails and the gradients are
silently wrong. grad_ready_check and the vision deferral (vision_grad_sync) do not cover the ordinary groups' order.

On chunks with --overlap-grad-reduce, every bucket group's start_grad_sync records its index when it actually issues
(hook-issued groups during the backward, deferred vision groups and step 1's groups at finish_grad_sync). After the
chunk's finish_grad_sync (every group issued and waited), each communicator (the dense DP / DP x CP group, the
expert-DP group) gets one 64-bit hash of this rank's index sequence; one all-reduce (MAX of [h, -h], 16 bytes) per
communicator compares them, and the step's one host read follows. A mismatch prints
`Q38_F8_ORDER_MISMATCH rank=<r> comm=<dense|expert> seq=[...]` on every rank and raises before the optimizer step.
No device work or host sync during the backward; no_sync microbatches issue nothing and add nothing.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

MARKER = "_qwen38_issue_order_checked"
_FNV_OFFSET, _FNV_PRIME, _MASK = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1


class IssueOrderError(AssertionError):
    pass


def sequence_hash(seq) -> int:
    """FNV-1a 64 over the length and the indices, as a signed int64."""
    h = _FNV_OFFSET
    for value in (len(seq), *seq):
        for byte in int(value).to_bytes(8, "little", signed=True):
            h = ((h ^ byte) * _FNV_PRIME) & _MASK
    return h - (1 << 64) if h >= 1 << 63 else h


def _communicator(group):
    if group.ddp_config.use_distributed_optimizer:
        return group.intra_distributed_optimizer_instance_group
    return group.data_parallel_group


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class _Check:
    def __init__(self, chunk):
        self.chunk = chunk
        self.comms = []  # [(kind, process group, device)] in the order every rank reduces them
        self.issued = {}  # kind -> [group index, ...] this step
        groups = [("dense", g) for g in chunk.bucket_groups] + [
            ("expert", g) for g in chunk.expert_parallel_bucket_groups
        ]
        for index, (kind, group) in enumerate(groups):
            if kind not in self.issued:
                device = group.buckets[0].grad_data.device if group.buckets else torch.device("cpu")
                self.comms.append((kind, _communicator(group), device))
                self.issued[kind] = []
            self._wrap(group, kind, index)
        finish = chunk.finish_grad_sync

        def finish_grad_sync(*args, **kwargs):
            out = finish(*args, **kwargs)
            self.verify()
            return out

        chunk.finish_grad_sync = finish_grad_sync

    def _wrap(self, group, kind, index) -> None:
        start = group.start_grad_sync

        def start_grad_sync(*args, **kwargs):
            if not (group.is_first_batch and group.grad_reduce_handle is not None):  # Megatron issues now
                self.issued[kind].append(index)
            return start(*args, **kwargs)

        group.start_grad_sync = start_grad_sync
        setattr(group, MARKER, True)

    def verify(self) -> None:
        """One MAX all-reduce of [h, -h] per communicator, one host read; raises when the ranks' hashes differ."""
        sequences = {kind: list(seq) for kind, seq in self.issued.items()}
        for seq in self.issued.values():
            seq.clear()
        reduced = []
        for kind, pg, device in self.comms:
            h = sequence_hash(sequences[kind])
            t = torch.tensor([h, -h], dtype=torch.int64, device=device)
            if pg is not None and dist.is_available() and dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pg)
            reduced.append(t)
        values = torch.stack(reduced).cpu().tolist() if reduced else []  # the step's one host read
        bad = [kind for (kind, _, _), (hi, neg_lo) in zip(self.comms, values, strict=True) if hi != -neg_lo]
        if bad:
            rank = _rank()
            for kind in bad:
                print(f"Q38_F8_ORDER_MISMATCH rank={rank} comm={kind} seq={sequences[kind]}", flush=True)
            raise IssueOrderError(
                f"bucket-group issue order differs across the ranks of the {bad} communicator(s); this rank: "
                + "; ".join(f"{k}={sequences[k]}" for k in bad)
            )


def install(model_chunks) -> list:
    """Install on every chunk with --overlap-grad-reduce; returns the checks."""
    checks = []
    for chunk in model_chunks:
        ddp_config = getattr(chunk, "ddp_config", None)  # config-access-exempt: only Megatron DDP chunks have one
        if ddp_config is None or not ddp_config.overlap_grad_reduce:
            continue
        checks.append(_Check(chunk))
    return checks
