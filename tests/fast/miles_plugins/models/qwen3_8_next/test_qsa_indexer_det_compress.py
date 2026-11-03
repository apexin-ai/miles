"""``compress_keys_by_mean_packed`` sums each block in a fixed order, not with ``index_add_``.

On CUDA a float ``index_add_`` uses atomic adds, so a block's tokens were summed in a different
order on each run and the block keys (then the indexer's top-k) were not bitwise reproducible.
The new code copies every token to its own slot of a ``[num_blocks * r, dim]`` grid and sums each
block's ``r`` slots, as ``compress_keys_by_mean`` does for one sequence. These tests check, on CPU,
that the values stay those of the old code (kept here verbatim as the reference), that one
sequence gives the single-sequence path bitwise, and that ``index_add_`` is no longer called.
Run-to-run bitwise equality on a GPU is checked on a GPU, not here.
"""

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import (  # noqa: E402
    PackedBlockLayout,
    _indexer_acc_dtype,
    compress_keys_by_mean,
    compress_keys_by_mean_packed,
)

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["miles-plugin"])

DIM = 16

# lengths: several sequences, a get_batch pad segment at the end, lengths not a multiple of r,
# one-token sequences, exact multiples, and one sequence alone
LENS_CASES = [
    [37, 13, 22],
    [37, 13, 22, 3],
    [1, 1, 1],
    [1, 9, 1, 16, 5],
    [8, 16, 24],
    [64, 1],
    [37],
    [1],
    [64],
]
RATIOS = [3, 4, 8]


def _packed_positions(cu, total):
    idx = torch.arange(total)
    starts = cu[:-1].long()
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    return idx - starts[seg]


def _old_compress_keys_by_mean_packed(token_k, layout):
    """The code before this change, verbatim."""
    acc = _indexer_acc_dtype(token_k)
    dim = token_k.shape[-1]
    summed = torch.zeros(layout.num_blocks, dim, dtype=acc, device=token_k.device)
    summed.index_add_(0, layout.token_block, token_k.to(acc))
    counts = torch.zeros(layout.num_blocks, 1, dtype=acc, device=token_k.device)
    counts.index_add_(0, layout.token_block, torch.ones(token_k.shape[0], 1, dtype=acc, device=token_k.device))
    return (summed / counts.clamp_min(1)).to(token_k.dtype)


def _case(lens, ratio, dtype, seed=0):
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32)
    total = int(cu[-1])
    positions = _packed_positions(cu, total)
    gen = torch.Generator().manual_seed(seed)
    token_k = (torch.randn(total, DIM, generator=gen, dtype=torch.float64) * 3).to(dtype)
    layout = PackedBlockLayout(cu, positions, ratio)
    return cu, token_k, layout


@pytest.mark.parametrize("lens", LENS_CASES)
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_new_matches_old_index_add(lens, ratio, dtype):
    _, token_k, layout = _case(lens, ratio, dtype)
    new = compress_keys_by_mean_packed(token_k, layout)
    old = _old_compress_keys_by_mean_packed(token_k, layout)
    assert new.shape == old.shape == (layout.num_blocks, DIM)
    assert new.dtype == old.dtype == dtype
    tol = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(new, old, rtol=tol, atol=tol)


@pytest.mark.parametrize("lens", [[37], [1], [64], [3], [5]])
@pytest.mark.parametrize("ratio", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_sequence_equals_single_path_bitwise(lens, ratio, dtype):
    _, token_k, layout = _case(lens, ratio, dtype, seed=1)
    new = compress_keys_by_mean_packed(token_k, layout)
    single = compress_keys_by_mean(token_k, ratio)
    assert new.shape == single.shape
    assert torch.equal(new, single)


@pytest.mark.parametrize("lens", LENS_CASES)
@pytest.mark.parametrize("ratio", [4, 8])
def test_each_sequence_equals_single_path_bitwise(lens, ratio):
    cu, token_k, layout = _case(lens, ratio, torch.float32, seed=2)
    new = compress_keys_by_mean_packed(token_k, layout)
    expected = torch.cat(
        [compress_keys_by_mean(token_k[int(cu[i]) : int(cu[i + 1])], ratio) for i in range(len(lens))], dim=0
    )
    assert torch.equal(new, expected)


@pytest.mark.parametrize("lens", LENS_CASES)
@pytest.mark.parametrize("ratio", RATIOS)
def test_token_slots_are_unique_and_inside_their_block(lens, ratio):
    _, _, layout = _case(lens, ratio, torch.float32)
    slots = layout.token_slot
    assert slots.dtype == torch.long
    assert slots.unique().numel() == slots.numel()
    assert torch.equal(slots // ratio, layout.token_block)
    assert int(slots.max()) < layout.num_blocks * ratio


def test_index_add_is_not_called(monkeypatch):
    _, token_k, layout = _case([37, 13, 22, 3], 4, torch.float32)
    expected = _old_compress_keys_by_mean_packed(token_k, layout)

    def _raise(*args, **kwargs):
        raise AssertionError("compress_keys_by_mean_packed must not use index_add_ (atomic on CUDA)")

    monkeypatch.setattr(torch.Tensor, "index_add_", _raise)
    monkeypatch.setattr(torch, "index_add", _raise)
    new = compress_keys_by_mean_packed(token_k, layout)
    torch.testing.assert_close(new, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_dtype_and_shape_unchanged(dtype):
    _, token_k, layout = _case([37, 13, 22, 3], 4, dtype)
    new = compress_keys_by_mean_packed(token_k, layout)
    old = _old_compress_keys_by_mean_packed(token_k, layout)
    assert new.dtype == old.dtype == dtype
    assert new.shape == old.shape == (layout.num_blocks, DIM)
    # both accumulate in fp32 and round once at the end: low-precision results match to one ulp
    torch.testing.assert_close(new.float(), old.float(), rtol=1e-2, atol=1e-2)


def test_host_bounds_layout_has_the_same_slots():
    cu, _, layout = _case([37, 13, 22, 3], 4, torch.float32)
    total = int(cu[-1])
    host = PackedBlockLayout(cu, _packed_positions(cu, total), 4, cu_seqlens_host=tuple(int(b) for b in cu))
    assert torch.equal(host.token_slot, layout.token_slot)
    assert host.compress_ratio == layout.compress_ratio == 4
