"""Scheduling changes launch order, never key ownership or summation order."""

from dataclasses import fields

import pytest
import torch

from miles.kernels.attention.qsa.block_index import QSABlockIndex
from miles.kernels.attention.qsa.key_schedule import reorder_qsa_key_tiles


def _index(counts):
    counts = torch.tensor(counts, dtype=torch.int32)
    offsets = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)))
    empty = torch.empty(0, dtype=torch.int32)
    return QSABlockIndex(
        query_tile=10,
        key_tile=16,
        ratio=4,
        q_offsets=empty,
        q_key_start=empty,
        q_bits=empty,
        q_order=empty,
        k_offsets=offsets,
        k_query=torch.arange(int(offsets[-1]), dtype=torch.int32),
        k_bits=torch.ones(int(offsets[-1]), dtype=torch.int32),
        k_order=torch.arange(counts.numel(), dtype=torch.int32),
        block_start=empty,
        block_len=empty,
        last_key=empty,
    )


@pytest.mark.parametrize("counts,want", [([], []), ([0], [0]), ([2, 9, 9, 0, 3], [1, 2, 4, 0, 3])])
def test_workload_order_preserves_all_lists_and_stable_ties(counts, want):
    index = _index(counts)
    result = reorder_qsa_key_tiles(index, order="workload")
    assert result.k_order.tolist() == want
    assert result.k_order.dtype == torch.int32
    assert index.k_order.tolist() == list(range(len(counts)))
    for field in fields(index):
        if field.name != "k_order":
            assert getattr(result, field.name) is getattr(index, field.name)
    # Every original key tile retains exactly its original query traversal.
    visits = []
    for tile in result.k_order.tolist():
        lo, hi = result.k_offsets[tile : tile + 2].tolist()
        visits.extend(result.k_query[lo:hi].tolist())
    assert sorted(visits) == list(range(sum(counts)))


def test_position_restores_order_without_modifying_workload_index():
    original = _index([1, 3, 2])
    workload = reorder_qsa_key_tiles(original, order="workload")
    result = reorder_qsa_key_tiles(workload, order="position")
    assert result.k_order.tolist() == [0, 1, 2]
    assert workload.k_order.tolist() == [1, 2, 0]


def test_unknown_order_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="unknown QSA key-tile order"):
        reorder_qsa_key_tiles(_index([1]), order="worklod")
