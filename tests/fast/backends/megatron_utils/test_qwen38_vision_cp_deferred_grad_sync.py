"""CPU contracts: vision_grad_sync under allgather context parallelism.

Megatron DDP reduces the dense gradients, the vision tower's among them, over DP x CP (intra_dp_cp_group,
distributed_data_parallel.py at fdf82356), so every DP x CP rank must issue the bucket groups in one order. Under
allgather-CP each CP rank of an image microbatch runs the tower on every image and keeps the image tokens of its own
slice; a rank whose slice holds no image token still reaches every tower parameter through the zero anchor of
scatter_image_embeddings. These tests run the simulated DDP of test_qwen38_vision_deferred_grad_sync (Megatron's own
bucket-group methods and layout) on the CP harness of test_qwen38_vision_cp: text microbatches, an image in rank 1's
slice only (rank 0 has none), an image in rank 0's slice only, and an image across the boundary.
"""

import pytest
import torch
from tests.fast.backends.megatron_utils.test_qwen38_vision_cp import CASES, _allgather_pack, cp_world  # noqa: F401
from tests.fast.backends.megatron_utils.test_qwen38_vision_deferred_grad_sync import _Chunk, _layout
from tests.fast.backends.megatron_utils.test_qwen38_vision_packing import _packed
from tests.fast.backends.megatron_utils.test_qwen38_vision_text_fallback import _model

from miles_plugins.models.qwen3_8_next import vision_grad_sync

_CP = 2
_STEPS = ["text-only", "image-in-last-slice", "image-in-first-slice", "image-straddles-boundary"]


def _microbatch(chunk, case, rank, set_cp):
    """One CP rank's forward / backward of the case, then DDP's finish_grad_sync and reset; returns the issue order
    and the order of the gradient reports."""
    model = chunk.module
    ids, bounds, rows, grid, positions = _allgather_pack(CASES[case], _CP)
    local = ids.numel() // _CP
    set_cp(_CP, rank, ids)
    inputs = {"qwen38_position_ids": positions}
    if grid is not None:
        inputs |= {"pixel_values": torch.ones(int(grid.prod(-1).sum()), 24) / 7, "image_grid_thw": grid}
    reports = []
    names = {p: n for n, p in model.named_parameters()}
    handles = [p.register_post_accumulate_grad_hook(lambda p: reports.append(names[p])) for p in names]
    chunk.issued.clear()
    model.zero_grad(set_to_none=True)
    local_ids = ids[rank * local : (rank + 1) * local][None]
    model(input_ids=local_ids, packed_seq_params=_packed(bounds, rows), **inputs).backward()
    for h in handles:
        h.remove()
    for group in chunk.bucket_groups:
        group.finish_grad_sync()
    for group in chunk.bucket_groups:
        group.reset()
    return list(chunk.issued), reports


def _rank_run(bucket_size, rank, steps, set_cp, defer=True):
    torch.manual_seed(0)
    model = _model(scope="full", data="mixed")
    model.config.qwen3_8_next_allgather_cp = True
    chunk = _Chunk(model, _layout(model, bucket_size))
    if defer:
        vision_grad_sync.defer_vision_bucket_groups([chunk])
    return [_microbatch(chunk, case, rank, set_cp) for case in steps], chunk


@pytest.mark.parametrize("bucket_size", [33, 64, 65, 300, 20_000_000])
def test_every_dp_x_cp_rank_issues_the_bucket_groups_in_one_order(cp_world, bucket_size):  # noqa: F811
    """Two DP replicas x CP2 = four ranks; each DP replica's last microbatch is another case (text, an image only on
    the other CP rank's slice, an image across the boundary). From the second step every rank issues alike."""
    orders = {}
    for replica, steps in enumerate((["image-in-first-slice"] + _STEPS, ["image-in-first-slice"] + _STEPS[::-1])):
        for rank in range(_CP):
            results, chunk = _rank_run(bucket_size, rank, steps, cp_world)
            orders[(replica, rank)] = [issued for issued, _ in results[1:]]
    assert len({tuple(map(tuple, o)) for o in orders.values()}) == 1, orders
    deferred = [i for i, g in enumerate(chunk.bucket_groups) if getattr(g, vision_grad_sync.MARKER, False)]
    for order in orders[(0, 0)]:
        assert order[-len(deferred) :] == deferred and sorted(order) == list(range(len(chunk.bucket_groups)))


@pytest.mark.parametrize("case", _STEPS)
def test_every_cp_rank_reports_each_tower_parameter_once_after_the_language_model(cp_world, case):  # noqa: F811
    """Text, and an image whatever slice its tokens are on: one report per tower parameter, after the language
    model's (head) and before the word embeddings, on both CP ranks."""
    for rank in range(_CP):
        (result,), chunk = _rank_run(64, rank, [case], cp_world)
        _, reports = result
        tower = [n for n in reports if n.startswith("vision_model.")]
        assert sorted(tower) == sorted(f"vision_model.{n}" for n, _ in chunk.module.vision_model.named_parameters())
        assert len(set(tower)) == len(tower)
        first, last = reports.index(tower[0]), reports.index(tower[-1])
        assert all(reports.index(n) < first for n in reports if n.startswith("head."))
        assert last < reports.index("embedding.word_embeddings.weight")
        vision_grads = [p.grad for p in chunk.module.vision_model.parameters()]
        if case == "text-only" or (case == "image-in-last-slice" and rank == 0):
            assert all(not g.any() for g in vision_grads)  # no image token on this rank: exact zeros
