"""Experimental launch ordering for CUDA QSA dK/dV; usable on CPU for schedule checks.

Only independent key tiles move. Their CSR lists, per-key summation order, and output
ownership stay unchanged. Workload ordering can lose query-cache locality, so callers
must measure it on real selections before enabling it in training.
"""

from dataclasses import replace

import torch

from miles.kernels.attention.qsa.block_index import QSABlockIndex


def reorder_qsa_key_tiles(index: QSABlockIndex, *, order: str) -> QSABlockIndex:
    """Return an index with positional or longest-query-list-first dK/dV launches.

    This is an explicit experiment API, not a change to the training default. No CUDA
    synchronization or tensor contents are read on the host. Equal workloads keep
    ascending tile ids, making the ordering deterministic.
    """
    if order == "position":
        key_order = torch.arange(index.k_offsets.numel() - 1, dtype=torch.int32, device=index.k_offsets.device)
    elif order == "workload":
        counts = index.k_offsets[1:] - index.k_offsets[:-1]
        key_order = torch.argsort(counts, descending=True, stable=True).to(torch.int32)
    else:
        raise ValueError(f"unknown QSA key-tile order: {order!r}; expected 'position' or 'workload'")
    return replace(index, k_order=key_order)
