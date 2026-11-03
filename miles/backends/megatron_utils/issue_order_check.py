"""Fail-closed check that every rank issues its bucket groups' gradient reductions in one order (CX3 F8-RC).

With --overlap-grad-reduce Megatron DDP issues a bucket group's reduce-scatter from the gradient hooks as soon as its
report counts equal the first step's (golden). Counts say nothing about order: if two ranks of one communicator issued
two groups in opposite orders, NCCL would pair different buckets; with equal shapes nothing fails and the gradients are
silently wrong. grad_ready_check and the vision deferral (vision_grad_sync) do not cover the ordinary groups' order.

Only on chunks with --overlap-grad-reduce (otherwise nothing is wrapped and no collective ever runs):
  - identity: each bucket group's signature is a 64-bit hash of its buckets' ordered parameter names, numels, dtypes and
    gradient scaling factors. At install, every communicator (the dense DP / DP x CP group, the expert-DP group)
    checks that its ranks hold the same ordered list of signatures (the same layout), with the agreement below.
  - recording: every group's start_grad_sync appends the group when it actually issues (hook-issued groups during
    the backward, the deferred vision groups and step 1's groups at finish_grad_sync); no_sync microbatches issue
    nothing. No device work, no host sync during the backward.
  - verify_step(), called once per step by train_one_step after the backward and finalize_model_grads (every
    group finished, no reduction of the step still in flight) and before the optimizer: per communicator, h = the
    64-bit hash of the rank's issued signature sequence and ONE all-reduce (MAX) of the 16-byte int64 pair [h, -h]
    on that communicator (MAX of h and -MIN of h at once); then ONE all-reduce (MAX) of a 4-byte int32 mismatch flag
    over all ranks, so a mismatch on any communicator stops every rank; then one host read. Every rank raises
    IssueOrderError before the optimizer step; a rank on a mismatching communicator first prints
    `Q38_F8_ORDER_MISMATCH rank=<r> comm=<chunk>:<dense|expert> seq=[<group index>, ...]`.
Collectives per step: one 16-byte all-reduce per communicator (two per chunk: dense, expert) + one 4-byte all-reduce
over the world, all after the backward.
"""

from __future__ import annotations

import hashlib

import torch
import torch.distributed as dist

MARKER = "_qwen38_issue_order_checked"
_FNV_OFFSET, _FNV_PRIME, _MASK = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1

_CHECKS = []


class IssueOrderError(AssertionError):
    pass


def _signed(h: int) -> int:
    return h - (1 << 64) if h >= 1 << 63 else h


def sequence_hash(seq) -> int:
    """FNV-1a 64 over the length and the (signed 64-bit) values, as a signed int64."""
    h = _FNV_OFFSET
    for value in (len(seq), *seq):
        for byte in int(value).to_bytes(8, "little", signed=True):
            h = ((h ^ byte) * _FNV_PRIME) & _MASK
    return _signed(h)


def group_signature(group, names) -> int:
    """64-bit hash of the group's buckets: ordered parameter names, numels, dtypes, gradient scaling factors."""
    parts = []
    for bucket in group.buckets:
        parts.append(f"bucket scale={float(bucket.gradient_scaling_factor)!r}")
        parts.extend(f"{names.get(p, '?')}|{p.numel()}|{p.dtype}" for p in bucket.params_list)
    return _signed(int.from_bytes(hashlib.sha256("\n".join(parts).encode()).digest()[:8], "little"))


def _communicator(group):
    if group.ddp_config.use_distributed_optimizer:
        return group.intra_distributed_optimizer_instance_group
    return group.data_parallel_group


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


class _Check:
    def __init__(self, chunk_index, chunk):
        names = {p: n for n, p in chunk.named_parameters()}
        groups = [("dense", g) for g in chunk.bucket_groups] + [
            ("expert", g) for g in chunk.expert_parallel_bucket_groups
        ]
        self.comms = []  # [(label, kind, process group, device)] in the order every rank reduces them
        self.signatures = {}  # kind -> [signature per group, in DDP order]
        self.index_of = {}  # kind -> {group index: position in signatures}
        self.issued = {}  # kind -> [group index, ...] this step
        for index, (kind, group) in enumerate(groups):
            if kind not in self.issued:
                self.comms.append(
                    (f"{chunk_index}:{kind}", kind, _communicator(group), group.buckets[0].grad_data.device)
                )
                self.issued[kind], self.signatures[kind], self.index_of[kind] = [], [], {}
            self.index_of[kind][index] = len(self.signatures[kind])
            self.signatures[kind].append(group_signature(group, names))
            self._wrap(group, kind, index)

    def _wrap(self, group, kind, index) -> None:
        start = group.start_grad_sync

        def start_grad_sync(*args, **kwargs):
            if not (group.is_first_batch and group.grad_reduce_handle is not None):  # Megatron issues now
                self.issued[kind].append(index)
            return start(*args, **kwargs)

        group.start_grad_sync = start_grad_sync
        setattr(group, MARKER, True)

    def layout_entries(self):
        return [
            (label, pg, device, sequence_hash(self.signatures[kind]), None) for label, kind, pg, device in self.comms
        ]

    def step_entries(self):
        out = []
        for label, kind, pg, device in self.comms:
            seq = list(self.issued[kind])
            self.issued[kind].clear()
            sigs = [self.signatures[kind][self.index_of[kind][i]] for i in seq]
            out.append((label, pg, device, sequence_hash(sigs), seq))
        return out


def _agree(entries, what: str) -> None:
    """entries: (label, communicator, device, hash, details). Per communicator one MAX all-reduce of [h, -h]; one
    world MAX all-reduce of the mismatch flag; one host read. Raises on every rank when any communicator disagreed."""
    if not entries:
        return
    flags = []
    for _, pg, device, h, _ in entries:
        t = torch.tensor([h, -h], dtype=torch.int64, device=device)
        if pg is not None and _distributed():
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=pg)
        flags.append((t[0] + t[1] != 0).to(torch.int32))  # max(h) != min(h), decided on the device
    local = torch.stack(flags)
    anywhere = local.max().clone()
    if _distributed():
        dist.all_reduce(anywhere, op=dist.ReduceOp.MAX)
    local_flags, any_flag = local.cpu().tolist(), int(anywhere.cpu())  # the one host read
    if not any_flag:
        return
    rank = dist.get_rank() if _distributed() else 0
    mine = [e for e, f in zip(entries, local_flags, strict=True) if f]
    for label, _, _, _, details in mine:
        print(f"Q38_F8_ORDER_MISMATCH rank={rank} comm={label} what={what} seq={details}", flush=True)
    where = [label for label, *_ in mine] or "another rank's communicator"
    raise IssueOrderError(f"bucket-group {what} differs across ranks ({where}); stopping before the optimizer step")


def install(model_chunks) -> list:
    """Install on every chunk with --overlap-grad-reduce and check the layouts agree; returns the checks."""
    global _CHECKS
    _CHECKS = []
    for index, chunk in enumerate(model_chunks):
        ddp_config = getattr(chunk, "ddp_config", None)  # config-access-exempt: only Megatron DDP chunks have one
        if ddp_config is not None and ddp_config.overlap_grad_reduce:
            _CHECKS.append(_Check(index, chunk))
    _agree([e for c in _CHECKS for e in c.layout_entries()], "layout (parameter signatures)")
    return list(_CHECKS)


def verify_step() -> None:
    """Once per step after every bucket group finished (train_one_step, after the backward), before the optimizer."""
    _agree([e for c in _CHECKS for e in c.step_entries()], "issue order")
