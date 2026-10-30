"""QSA block bitmaps are stored one bit per (query, block); packing must lose nothing."""

import pytest

torch = pytest.importorskip("torch")

from miles.kernels.attention.qsa.block_sparse_attention import (  # noqa: E402
    _tile_mask,
    pack_block_bitmap,
    selection_to_block_bitmap,
    unpack_block_bitmap,
)


@pytest.mark.parametrize("num_blocks", [1, 7, 8, 9, 64, 1031])
def test_pack_round_trips_and_is_an_eighth(num_blocks):
    g = torch.Generator().manual_seed(num_blocks)
    flags = (torch.rand(37, num_blocks, generator=g) < 0.4).to(torch.uint8)
    packed = pack_block_bitmap(flags)
    assert packed.dtype == torch.uint8
    assert packed.shape == (37, -(-num_blocks // 8))
    assert torch.equal(unpack_block_bitmap(packed, num_blocks), flags)
    # block b is bit b % 8 of byte b // 8, the layout the kernels read
    b = num_blocks - 1
    assert torch.equal((packed[:, b // 8] >> (b % 8)) & 1, flags[:, b])


def _tile_mask_unpacked(sel, block_first, block_last, bq, bk):
    """``_tile_mask`` on a one-byte-per-block bitmap, as before packing."""
    T, nb = sel.shape
    nqt, nkt = -(-T // bq), -(-T // bk)
    if nqt * bq - T:
        sel = torch.nn.functional.pad(sel, (0, 0, 0, nqt * bq - T))
    picked = sel.view(nqt, bq, nb).amax(dim=1).to(torch.int32)
    tile = torch.zeros(nqt, nkt, dtype=torch.int32)
    tile.index_add_(1, block_first.long() // bk, picked)
    tile.index_add_(1, block_last.long() // bk, picked)
    return tile > 0


@pytest.mark.parametrize("T,ratio,bq,bk", [(300, 4, 64, 64), (257, 4, 64, 32), (1000, 4, 32, 64)])
def test_tile_mask_on_packed_bits_matches_unpacked(T, ratio, bq, bk):
    g = torch.Generator().manual_seed(T)
    nb = -(-T // ratio)
    flags = (torch.rand(T, nb, generator=g) < 0.05).to(torch.uint8)
    block_first = torch.arange(nb, dtype=torch.int32) * ratio
    block_last = (block_first + ratio).clamp_max(T) - 1
    got = _tile_mask(pack_block_bitmap(flags), block_first, block_last, bq, bk)
    assert torch.equal(got, _tile_mask_unpacked(flags, block_first, block_last, bq, bk))


def test_selection_bitmap_is_packed_block_flags():
    idx = torch.tensor([[0, 1, 2, 3, 9, -1], [-1, -1, -1, -1, -1, -1], [17, 4, -1, -1, -1, -1]], dtype=torch.int32)
    packed = selection_to_block_bitmap(idx, 18, 4)
    assert torch.equal(
        unpack_block_bitmap(packed, 5),
        torch.tensor([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 1]], dtype=torch.uint8),
    )


def test_every_bitmap_load_widens_the_row_before_the_stride():
    """At 262K tokens a packed row is ~8 KiB, so row x stride passes 2^31: int32 wraps negative.

    The widening must happen before the multiply (``offs_q.to(tl.int64) * stride_st``);
    converting the product is too late. Kernels only run on GPU, so pin it in the source.
    """
    import inspect

    from miles.kernels.attention.qsa import block_sparse_attention

    src = inspect.getsource(block_sparse_attention)
    loads = [line.strip() for line in src.splitlines() if line.strip().startswith("SEL +")]
    assert len(loads) == 3
    assert all(line.startswith("SEL + offs_q[:, None].to(tl.int64) * stride_st") for line in loads), loads
