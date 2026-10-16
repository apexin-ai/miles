"""A packed (thd) QSA indexer batch must score each sequence as if it ran alone."""

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import (  # noqa: E402
    PackedBlockLayout,
    block_causal_mask,
    compress_keys_by_mean,
    compress_keys_by_mean_packed,
    packed_block_causal_mask,
)

RATIO = 4
LENS = [37, 13, 22]  # none is a multiple of RATIO
DIM = 8


def _packed_positions(cu, total):
    idx = torch.arange(total)
    starts = cu[:-1].long()
    seg = torch.zeros(total, dtype=torch.long)
    seg[starts[1:]] = 1
    seg = seg.cumsum(0)
    return idx - starts[seg]


def _fixture():
    cu = torch.tensor([0, *LENS]).cumsum(0).to(torch.int32)
    total = int(cu[-1])
    positions = _packed_positions(cu, total)
    torch.manual_seed(0)
    token_k = torch.randn(total, DIM)
    return cu, total, positions, token_k


def test_packed_blocks_never_mix_sequences():
    cu, total, positions, token_k = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    packed = compress_keys_by_mean_packed(token_k, layout)

    expected = torch.cat(
        [compress_keys_by_mean(token_k[int(cu[i]) : int(cu[i + 1])], RATIO) for i in range(len(LENS))],
        dim=0,
    )
    assert packed.shape == expected.shape
    torch.testing.assert_close(packed, expected)


def test_packed_mask_matches_per_sequence_mask():
    cu, total, positions, _ = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    packed = packed_block_causal_mask(positions, layout, RATIO)
    assert packed.shape == (total, layout.num_blocks)

    block_start = 0
    for i, length in enumerate(LENS):
        lo, hi = int(cu[i]), int(cu[i + 1])
        n_blocks = -(-length // RATIO)
        alone = block_causal_mask(positions[lo:hi], n_blocks, RATIO)

        window = packed[lo:hi, block_start : block_start + n_blocks]
        torch.testing.assert_close(window, alone)

        # nothing outside the sequence's own block range may be visible
        outside = packed[lo:hi].clone()
        outside[:, block_start : block_start + n_blocks] = False
        assert not bool(outside.any()), f"sequence {i} can see blocks of another sequence"
        block_start += n_blocks


def test_global_block_grid_points_at_the_wrong_sequence():
    """Characterises the bug the packed path exists to avoid.

    A global block grid combined with per-sequence positions sends every sequence
    after the first to the blocks at the front of the buffer, and hides its own.
    """
    cu, total, positions, _ = _fixture()
    num_blocks_global = -(-total // RATIO)
    global_mask = block_causal_mask(positions, num_blocks_global, RATIO)

    lo, hi = int(cu[1]), int(cu[2])  # the second sequence
    own_lo = lo // RATIO
    row = hi - 1  # its last query, which has the most history
    visible = global_mask[row].nonzero().flatten()

    assert visible.numel() > 0
    assert int(visible.max()) < own_lo, "expected the global grid to look before the sequence start"
    assert not bool(global_mask[row, own_lo:].any()), "expected the sequence's own blocks to be hidden"


def test_layout_block_local_positions_restart_per_sequence():
    cu, total, positions, _ = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)

    expected = torch.cat([torch.arange(-(-length // RATIO)) for length in LENS])
    torch.testing.assert_close(layout.block_local, expected)
    assert layout.num_blocks == sum(-(-length // RATIO) for length in LENS)


class _DuplicatedLinear(torch.nn.Module):
    """CPU stand-in for TELinear(parallel_mode="duplicated")."""

    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) / input_size**0.5)

    def forward(self, x):
        return x @ self.weight.t(), None


def _cpu_indexer(monkeypatch, budget):
    from types import SimpleNamespace

    from miles_plugins.models.qwen3_8_next.ops import qsa_indexer

    monkeypatch.setattr(qsa_indexer, "TELinear", _DuplicatedLinear)
    config = SimpleNamespace(
        hidden_size=32,
        init_method=None,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        qwen3_8_next_indexer_n_heads=2,
        qwen3_8_next_indexer_kv_heads=1,
        qwen3_8_next_indexer_head_dim=16,
        qwen3_8_next_indexer_budget=budget,
        qwen3_8_next_indexer_compress_ratio=RATIO,
        qwen3_8_next_indexer_tile_bytes=2 << 10,  # several query tiles per batch
    )
    torch.manual_seed(0)
    return qsa_indexer.Qwen38NextQSAIndexer(config, layer_number=1)


@pytest.mark.parametrize("budget", [8, 4 * RATIO * 64])  # 2 blocks; more blocks than any row can see
def test_selected_blocks_name_the_indexer_blocks(monkeypatch, budget):
    """The block ids handed to the packed kernel are the blocks of the indexer's token rows.

    ``qsa_selected_blocks`` writes the tiled scorer's block ids straight into ``[T, K]`` int32;
    folding ``Qwen38NextQSAIndexer.forward``'s token rows back to blocks (done here, in the test
    only) must give the same ids in the same order. Each id must be a distinct whole block of the
    row's own sequence before the query, so the kernel adds the partial tail exactly once.
    """
    from miles_plugins.models.qwen3_8_next.ops.attention import qsa_selected_blocks

    cu, total, positions, _ = _fixture()
    layout = PackedBlockLayout(cu, positions, RATIO)
    indexer = _cpu_indexer(monkeypatch, budget)
    hidden = torch.randn(total, 32, generator=torch.Generator().manual_seed(1))
    freqs = torch.outer(torch.arange(total, dtype=torch.float32), 1.0 / (1e4 ** (torch.arange(0, 8, 2) / 8)))
    rope = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    with torch.no_grad():
        blocks, cu_out = qsa_selected_blocks(indexer, hidden, positions, rope, cu)
        tokens = indexer(hidden, positions, rope, cu_seqlens=cu)

    assert cu_out is cu
    assert blocks.dtype == torch.int32
    assert blocks.shape == (total, min(indexer.block_topk, layout.num_blocks))
    first = tokens[:, ::RATIO].long()
    folded = torch.where(
        first >= 0, layout.token_block_start.unsqueeze(1) + (first - layout.token_start.unsqueeze(1)) // RATIO, -1
    )
    width = min(blocks.shape[1], folded.shape[1])
    assert torch.equal(blocks[:, :width].long(), folded[:, :width])
    assert bool((blocks[:, width:] == -1).all())

    lo = layout.token_block_start.unsqueeze(1)
    hi = lo + ((positions + 1) // RATIO).unsqueeze(1)
    used = blocks >= 0
    assert bool(((blocks >= lo) & (blocks < hi))[used].all()), "an id outside the row's whole blocks"
    for row in range(total):
        ids = blocks[row][used[row]].tolist()
        assert len(ids) == len(set(ids)), row
        # rows with at most K whole blocks keep all of them
        assert len(ids) == min(int(hi[row] - lo[row]), blocks.shape[1]), row


def test_selected_blocks_of_a_single_sequence_use_its_own_cu_seqlens(monkeypatch):
    from miles_plugins.models.qwen3_8_next.ops.attention import qsa_selected_blocks

    indexer = _cpu_indexer(monkeypatch, 8)
    total = 30
    hidden = torch.randn(total, 32)
    freqs = torch.outer(torch.arange(total, dtype=torch.float32), torch.ones(4))
    rope = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    with torch.no_grad():
        blocks, cu = qsa_selected_blocks(indexer, hidden, torch.arange(total), rope, None)
    assert cu.tolist() == [0, total] and cu.dtype == torch.int32
    assert blocks.shape == (total, 2)
    assert blocks[: RATIO - 1].eq(-1).all()


@pytest.mark.parametrize("lens", [[37, 13], [64, 64], [1, 41]])
def test_selection_equals_single_sequence_runs(lens):
    """Full indexer: packed selection == per-sequence selection + token offset."""
    if not torch.cuda.is_available():
        pytest.skip("indexer projection needs a CUDA device (TELinear)")

    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer

    hidden = 64
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    config.qwen3_8_next_indexer_n_heads = 2
    config.qwen3_8_next_indexer_kv_heads = 1
    config.qwen3_8_next_indexer_head_dim = 32
    config.qwen3_8_next_indexer_budget = 16
    config.qwen3_8_next_indexer_compress_ratio = RATIO

    indexer = Qwen38NextQSAIndexer(config, layer_number=1).cuda()
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32).cuda()
    total = int(cu[-1])
    positions = _packed_positions(cu.cpu(), total).cuda()
    torch.manual_seed(0)
    hidden_states = torch.randn(total, hidden, device="cuda", dtype=torch.bfloat16)
    rotary_dim = 8
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim))
    freqs = torch.outer(torch.arange(total, dtype=torch.float32, device="cuda"), inv_freq)
    rotary_pos_emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]

    with torch.no_grad():
        packed = indexer(hidden_states, positions, rotary_pos_emb, cu_seqlens=cu)
        for i, length in enumerate(lens):
            lo = int(cu[i])
            sub_cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
            alone = indexer(
                hidden_states[lo : lo + length].contiguous(),
                positions[lo : lo + length].contiguous(),
                rotary_pos_emb,
                cu_seqlens=sub_cu,
            )
            shifted = torch.where(alone >= 0, alone + lo, alone)
            got = packed[lo : lo + length]

            # a sequence scored alone can come back narrower, so compare token sets
            for row in range(length):
                got_set = {int(t) for t in got[row].tolist() if t >= 0}
                want_set = {int(t) for t in shifted[row].tolist() if t >= 0}
                assert got_set == want_set, (
                    f"sequence {i} row {row}: packed selection {sorted(got_set)} "
                    f"!= single-sequence {sorted(want_set)}"
                )
