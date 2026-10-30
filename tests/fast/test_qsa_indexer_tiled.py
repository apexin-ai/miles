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


def check_block_selection(got, want, logits, k, tol=0.0, lowest_index=True):
    """Per row, the selection ``got`` must be a valid top-k of ``logits``; returns the rows where it differs from ``want``.

    ``got`` / ``want`` are ``[T, blocks]`` 0/1; ``logits`` the untiled reference scores,
    ``-inf`` outside each row's candidate blocks. Every row of ``got`` must:

    - match ``want`` exactly outside the candidates (that is the query's partial tail block);
    - when a row has at most ``k`` candidates, select all of them;
    - otherwise select exactly ``k`` candidates, every one scoring above the k-th score by
      more than ``tol``, none below it by more than ``tol``; only blocks inside that band
      may differ from ``want``, and with ``lowest_index`` an exact tie keeps the lowest ids.

    ``tol`` is the fp band the GPU allows for scores summed in a different order.
    """
    got, want = got.bool(), want.bool()
    cand = torch.isfinite(logits)
    bad = lambda mask, what: f"{what}: rows {torch.nonzero(mask).flatten()[:8].tolist()}"  # noqa: E731
    outside = ((got ^ want) & ~cand).any(-1)
    assert not bool(outside.any()), bad(outside, "differs outside the candidate blocks")
    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
    full = torch.isfinite(kth.squeeze(-1))
    few = ~full & ((got & cand) != cand).any(-1)
    assert not bool(few.any()), bad(few, "misses a candidate in a row with at most k of them")
    band = tol * kth.abs()
    above = logits > kth + band
    below = cand & (logits < kth - band)
    missing = full & (above & ~got).any(-1)
    assert not bool(missing.any()), bad(missing, "drops a strictly-better block")
    extra = full & (got & below).any(-1)
    assert not bool(extra.any()), bad(extra, "keeps a block below the k-th score")
    count = full & ((got & cand).sum(-1) != k)
    assert not bool(count.any()), bad(count, "selects a number of blocks other than k")
    differ = (got != want).any(-1)
    if lowest_index:
        at = full.unsqueeze(-1) & (logits == kth)
        lowest = at & (at.cumsum(-1) <= (got & at).sum(-1, keepdim=True))
        wrong = ((got & at) != lowest).any(-1)
        assert not bool(wrong.any()), bad(wrong, "breaks a tie away from the lowest block ids")
    return differ


def _compare(got, want, logits, k, layout):
    assert got.shape == want.shape
    g = _block_sets(got, layout, logits.shape[-1])
    w = _block_sets(want, layout, logits.shape[-1])
    return int(check_block_selection(g, w, logits, k).sum())


@pytest.mark.parametrize(
    "logits,k,got,ok",
    [
        ([10.0, 2.0, 2.0, 0.0], 2, [1, 1, 0, 0], True),
        ([10.0, 2.0, 2.0, 0.0], 2, [1, 0, 1, 0], False),  # valid top-k, but not the lowest-id tie
        ([10.0, 2.0, 2.0, 0.0], 2, [0, 0, 0, 0], False),  # nothing selected
        ([10.0, 2.0, 2.0, 0.0], 2, [0, 1, 1, 0], False),  # drops the best block
        ([10.0, 2.0, 2.0, 0.0], 2, [1, 1, 1, 0], False),  # one too many
        ([10.0, 2.0, 2.0, 0.0], 2, [1, 0, 0, 1], False),  # keeps a block below the k-th score
        ([10.0, 2.000001, 2.0, 0.0], 2, [0, 0, 0, 0], False),
        ([5.0, float("-inf"), float("-inf"), float("-inf")], 2, [1, 0, 0, 0], True),
        ([5.0, float("-inf"), float("-inf"), float("-inf")], 2, [0, 0, 0, 0], False),
    ],
)
def test_selection_check_rejects_invalid_bitmaps(logits, k, got, ok):
    """The equivalence check itself: a wrong bitmap must not pass as a tie swap."""
    logits = torch.tensor([logits])
    want = torch.zeros(1, 4, dtype=torch.uint8)
    want[0, torch.topk(logits[0], k).indices[torch.isfinite(torch.topk(logits[0], k).values)]] = 1
    got = torch.tensor([got], dtype=torch.uint8)
    if ok:
        check_block_selection(got, want, logits, k)
    else:
        with pytest.raises(AssertionError):
            check_block_selection(got, want, logits, k)
    if logits[0, 1] == 2.000001 and not got.any():
        # a near tie inside the fp band still may not empty the row
        with pytest.raises(AssertionError):
            check_block_selection(got, want, logits, k, tol=1e-5, lowest_index=False)


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
    check_block_selection(got[0], want[0], logits, k)


@pytest.mark.parametrize("tile_bytes", [0, -1])
def test_plan_rejects_a_non_positive_budget(tile_bytes):
    with pytest.raises(ValueError, match="positive"):
        plan_query_tiles([0, 0], [1, 2], tile_bytes)


def test_a_row_wider_than_the_budget_is_scored_alone():
    tiles = plan_query_tiles([0, 0, 0], [4096, 4096, 4096], 65536)
    assert tiles == [(0, 1), (1, 2), (2, 3)]
