"""The tiled QSA indexer must select exactly the blocks the untiled scorer selects.

``_reference_*`` below is the untiled implementation as of 65feb4a6: it builds the full
``[T, blocks, heads]`` fp32 score. ``torch.topk`` there leaves the choice among equal
scores unspecified; the tiled path fixes it (higher score, then lower block id), so rows
whose k-th score is tied are compared on the strictly-better blocks plus the tie count.
"""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops import qsa_indexer  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import (  # noqa: E402
    PackedBlockLayout,
    block_causal_mask,
    packed_block_causal_mask,
    plan_query_tiles,
    select_topk_blocks,
)

HIDDEN = 64
RATIO = 4


class _DuplicatedLinear(torch.nn.Module):
    """CPU stand-in for TELinear(parallel_mode="duplicated")."""

    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) / math.sqrt(input_size))

    def forward(self, x):
        return x @ self.weight.t(), None


def _indexer(monkeypatch, tile_bytes, budget=2048, heads=4, head_dim=128):
    monkeypatch.setattr(qsa_indexer, "TELinear", _DuplicatedLinear)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        init_method=None,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        qwen3_8_next_indexer_n_heads=heads,
        qwen3_8_next_indexer_kv_heads=1,
        qwen3_8_next_indexer_head_dim=head_dim,
        qwen3_8_next_indexer_budget=budget,
        qwen3_8_next_indexer_compress_ratio=RATIO,
        qwen3_8_next_indexer_tile_bytes=tile_bytes,
    )
    torch.manual_seed(0)
    indexer = qsa_indexer.Qwen38NextQSAIndexer(config, layer_number=1)
    with torch.no_grad():
        indexer.q_layernorm.normal_(0, 0.1)
        indexer.k_layernorm.normal_(0, 0.1)
    return indexer


def _packed_positions(cu, total):
    idx = torch.arange(total)
    starts = cu[:-1].long()
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    return idx - starts[seg]


def _inputs(lens, head_dim=128):
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32)
    total = int(cu[-1])
    positions = _packed_positions(cu, total) if len(lens) > 1 else torch.arange(total)
    torch.manual_seed(1)
    hidden = torch.randn(total, HIDDEN)
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    freqs = torch.outer(torch.arange(max(lens) + 1, dtype=torch.float32), inv_freq)
    rope = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    return hidden, positions, rope, (cu if len(lens) > 1 else None)


def _reference_logits(indexer, q, block_k, query_positions, layout):
    scores = torch.einsum("mhd,nd->mnh", q.float(), block_k.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(indexer.head_dim)
    if layout is None:
        valid = block_causal_mask(query_positions, block_k.shape[0], indexer.compress_ratio)
    else:
        valid = packed_block_causal_mask(query_positions, layout, indexer.compress_ratio)
    return logits.masked_fill(~valid, float("-inf"))


def _reference_forward(indexer, hidden_states, positions, rotary_pos_emb, cu_seqlens=None):
    """Untiled selection at 65feb4a6; also returns the logits it ranked."""
    layout = None
    if cu_seqlens is not None and cu_seqlens.numel() > 2:
        layout = PackedBlockLayout(cu_seqlens, positions, indexer.compress_ratio)
    q, block_k = indexer.project_qk(hidden_states, positions, rotary_pos_emb, layout=layout)
    logits = _reference_logits(indexer, q, block_k, positions, layout)
    k = min(indexer.block_topk, logits.shape[-1])
    block_scores, block_idx = torch.topk(logits, k, dim=-1)
    block_idx = block_idx.masked_fill(block_scores == float("-inf"), -1)
    offsets = torch.arange(indexer.compress_ratio)
    if layout is None:
        tokens = block_idx.unsqueeze(-1) * indexer.compress_ratio + offsets
    else:
        block_local = block_idx - layout.token_block_start.unsqueeze(-1)
        tokens = layout.token_start.unsqueeze(-1).unsqueeze(-1) + (
            block_local.unsqueeze(-1) * indexer.compress_ratio + offsets
        )
    tokens = tokens.masked_fill(block_idx.unsqueeze(-1) < 0, -1).flatten(-2)
    limit = positions if layout is None else layout.token_start + positions
    tokens = tokens.masked_fill(tokens > limit.unsqueeze(-1), -1)
    if layout is not None:
        tokens = tokens.masked_fill(tokens < layout.token_start.unsqueeze(-1), -1)
    return tokens[..., : indexer.token_topk].to(torch.int32), logits, layout


def _block_sets(tokens, layout, num_blocks):
    """``[T, k*ratio]`` token selection -> ``[T, num_blocks]`` bool of selected blocks."""
    valid = tokens >= 0
    t = tokens.long().clamp_min(0)
    if layout is None:
        blocks = t // RATIO
    else:
        blocks = layout.token_block_start.unsqueeze(1) + (t - layout.token_start.unsqueeze(1)) // RATIO
    out = torch.zeros(tokens.shape[0], num_blocks, dtype=torch.bool)
    rows = torch.arange(tokens.shape[0]).unsqueeze(1).expand_as(blocks)
    out[rows[valid], blocks[valid]] = True
    return out


def _compare(got, want, logits, k, layout):
    """Rows must select the same blocks, except where the k-th score is tied: there the
    strictly-better blocks and the tie count agree, and the tiled path keeps the lowest ids."""
    assert got.shape == want.shape
    g = _block_sets(got, layout, logits.shape[-1])
    w = _block_sets(want, layout, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
    tied = torch.isfinite(kth.squeeze(-1)) & ((logits >= kth).sum(-1) > k)
    differ = (g != w).any(dim=-1)
    assert not bool((differ & ~tied).any()), f"rows {torch.nonzero(differ & ~tied).flatten()[:8].tolist()} differ without a tie"
    strictly = logits > kth
    assert torch.equal(g & strictly, w & strictly)
    at = logits == kth
    assert torch.equal((g & at).sum(-1), (w & at).sum(-1))
    lowest = at & (at.cumsum(-1) <= (g & at).sum(-1, keepdim=True))
    assert torch.equal((g & at)[tied], lowest[tied]), "tie broken away from the lowest block ids"
    return int((differ & tied).sum())


CASES = [
    pytest.param([257], id="T257"),
    pytest.param([4096], id="T4096"),
    pytest.param([1, 3, 2055, 1, 5, 1000], id="packed-short-docs"),
    pytest.param([2600, 1497], id="packed-two-long"),
]


@pytest.mark.parametrize("lens", CASES)
@pytest.mark.parametrize("tile_bytes", [1 << 30, 64 << 10], ids=["1GiB", "64KiB"])
def test_tiled_selection_matches_untiled(monkeypatch, lens, tile_bytes):
    indexer = _indexer(monkeypatch, tile_bytes)
    hidden, positions, rope, cu = _inputs(lens)
    with torch.no_grad():
        want, logits, layout = _reference_forward(indexer, hidden, positions, rope, cu)
        got = indexer(hidden, positions, rope, cu_seqlens=cu)
    k = min(indexer.block_topk, logits.shape[-1])
    tie_rows = _compare(got, want, logits, k, layout)
    assert tie_rows <= max(2, want.shape[0] // 20)
    if lens == [4096]:
        assert tie_rows > 0, "the k-th-score tie path was not exercised"


def test_tiles_never_score_more_than_the_budget(monkeypatch):
    tile_bytes = 256 << 10
    indexer = _indexer(monkeypatch, tile_bytes)
    hidden, positions, rope, cu = _inputs([4096])
    sizes = []
    einsum = torch.einsum

    def spy(eq, *ops):
        out = einsum(eq, *ops)
        if eq == "mhd,nd->mnh":
            sizes.append(out.numel() * out.element_size())
        return out

    monkeypatch.setattr(qsa_indexer.torch, "einsum", spy)
    with torch.no_grad():
        indexer(hidden, positions, rope)
    full = 4096 * 1024 * 4 * 4
    assert len(sizes) > 1
    assert max(sizes) <= tile_bytes * 16 // 20 < full


def test_tie_rule_prefers_higher_score_then_lower_index():
    logits = torch.tensor(
        [
            [0.0, 3.0, 0.0, 0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [5.0, float("-inf"), float("-inf"), 4.0, float("-inf"), float("-inf")],
        ]
    )
    values, idx = select_topk_blocks(logits.clone(), 4)
    assert idx[0].tolist() == [1, 4, 0, 2]
    assert idx[1].tolist() == [0, 1, 2, 3]
    assert idx[2, :2].tolist() == [0, 3]
    assert values[2, 2:].tolist() == [float("-inf")] * 2
    # the rule does not depend on how the rows are batched
    for row in range(2):
        _, alone = select_topk_blocks(logits[row : row + 1].clone(), 4)
        assert alone[0].tolist() == idx[row].tolist()


@pytest.mark.parametrize("seed", range(5))
def test_plan_covers_every_row_once_within_budget(seed):
    g = torch.Generator().manual_seed(seed)
    lens = (torch.randint(1, 300, (8,), generator=g)).tolist()
    lo, hi = [], []
    block_start = 0
    for length in lens:
        for pos in range(length):
            lo.append(block_start)
            hi.append(block_start + (pos + 1) // RATIO)
        block_start += -(-length // RATIO)
    cap_bytes = 20 * 64
    tiles = plan_query_tiles(lo, hi, cap_bytes)
    assert tiles[0][0] == 0 and tiles[-1][1] == len(lo)
    assert all(a[1] == b[0] for a, b in zip(tiles, tiles[1:], strict=False))
    for s, e in tiles:
        assert e - s == 1 or (e - s) * max(1, hi[e - 1] - lo[s]) <= 64


def _reference_block_form(indexer, hidden, positions, rope, cu):
    """Selection merge + bitmap of ``Qwen38NextAttention`` at 65feb4a6 (thd path)."""
    seq = positions.shape[0]
    selection, logits, _ = _reference_forward(indexer, hidden, positions, rope, cu)
    seq_start = torch.arange(seq) - positions
    r = indexer.compress_ratio
    tail_in_seq = (positions + 1) // r * r
    offs = torch.arange(r)
    tail_pos = tail_in_seq.unsqueeze(1) + offs.unsqueeze(0)
    tail_idx = seq_start.unsqueeze(1) + tail_pos
    tail_idx = torch.where(tail_pos <= positions.unsqueeze(1), tail_idx, torch.full_like(tail_idx, -1))
    merged = torch.cat([selection, tail_idx.to(selection.dtype)], dim=1)
    pack_pos = torch.arange(seq).unsqueeze(1)
    ok = (merged >= seq_start.unsqueeze(1)) & (merged <= pack_pos) & (merged >= 0)
    selection = torch.where(ok, merged, torch.full_like(merged, -1))

    cu_ = cu if cu is not None and cu.numel() > 2 else torch.tensor([0, seq])
    layout = PackedBlockLayout(cu_, positions, r)
    blk_base = layout.token_block_start.to(torch.int32)
    tok_base = layout.token_start.to(torch.int32)
    num_blocks = layout.num_blocks
    sel_bitmap = torch.zeros(seq, num_blocks, dtype=torch.uint8)
    valid = selection >= 0
    if bool(valid.any()):
        rows = torch.arange(seq).unsqueeze(1).expand_as(selection)
        blk = blk_base.unsqueeze(1).long() + (selection - tok_base.unsqueeze(1)).div(r, rounding_mode="floor")
        sel_bitmap[rows[valid], blk[valid].long().clamp_(0, num_blocks - 1)] = 1
    hi = (positions + seq_start).to(torch.int32)
    form = (
        sel_bitmap,
        tok_base,
        hi,
        blk_base,
        tok_base,
        layout.block_first_token.to(torch.int32),
        layout.block_last_token.to(torch.int32),
        r,
    )
    return form, logits


@pytest.mark.parametrize("lens", CASES)
def test_block_form_matches_untiled_attention(monkeypatch, lens):
    from miles_plugins.models.qwen3_8_next.ops.attention import qsa_block_form

    indexer = _indexer(monkeypatch, 64 << 10)
    hidden, positions, rope, cu = _inputs(lens)
    with torch.no_grad():
        want, logits = _reference_block_form(indexer, hidden, positions, rope, cu)
        got = qsa_block_form(indexer, hidden, positions, rope, cu)
    for a, b in zip(got[1:7], want[1:7], strict=True):
        assert torch.equal(a, b)
    assert got[7] == want[7]
    k = min(indexer.block_topk, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
    tied = (torch.isfinite(kth.squeeze(-1))) & ((logits >= kth).sum(-1) > k)
    differ = (got[0] != want[0]).any(dim=-1)
    assert not bool((differ & ~tied).any()), "bitmap rows differ without a tied k-th score"
    # on tied rows the strictly-better blocks and the selected count still agree
    strictly = logits > kth
    assert torch.equal(got[0].bool() & strictly, want[0].bool() & strictly)
    assert torch.equal(got[0].sum(-1), want[0].sum(-1))
