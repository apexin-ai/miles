"""QSA indexer speed knobs: ``--qwen38-indexer-score-dtype`` and ``--qwen38-indexer-tp-shard``.

- bf16 scoring (bf16 operands, fp32 math) selects the same blocks as the fp32 reference when the scores are well
  separated, and gives the same logits on bf16 inputs up to fp32 rounding.
- The TP-sharded selection (rank r scores rows r, r + n, ...; all-gather; interleave) equals the replicated one.
- The flags reach the indexer through the TransformerConfig.

The fused Triton kernel itself needs a GPU; ``tests/manual``-style checks run it against the fp32 path.
"""

import math
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops import attention, qsa_indexer, qsa_score  # noqa: E402

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["miles-plugin"])

HIDDEN = 64
RATIO = 4


class _DuplicatedLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) / math.sqrt(input_size))

    def forward(self, x):
        return x @ self.weight.t(), None


def _indexer(monkeypatch, score_dtype="fp32", tile_bytes=1 << 20, budget=64, params_dtype=torch.float32):
    monkeypatch.setattr(qsa_indexer, "TELinear", _DuplicatedLinear)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        init_method=None,
        params_dtype=params_dtype,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        qwen3_8_next_indexer_n_heads=4,
        qwen3_8_next_indexer_kv_heads=1,
        qwen3_8_next_indexer_head_dim=32,
        qwen3_8_next_indexer_budget=budget,
        qwen3_8_next_indexer_compress_ratio=RATIO,
        qwen3_8_next_indexer_tile_bytes=tile_bytes,
        qwen3_8_next_indexer_score_dtype=score_dtype,
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
    return idx - starts[seg.cumsum(0)]


def _inputs(lens, dtype=torch.float32, rotary_dim=16):
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32)
    total = int(cu[-1])
    positions = _packed_positions(cu, total) if len(lens) > 1 else torch.arange(total)
    torch.manual_seed(1)
    hidden = torch.randn(total, HIDDEN).to(dtype)
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(max(lens) + 1, dtype=torch.float32), inv_freq)
    rope = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    return hidden, positions, rope, (cu if len(lens) > 1 else None)


def _well_separated(rows, heads, width, dim, seed=0):
    """bf16 q / keys whose block scores are distinct integers per row: every top-k is unique."""
    g = torch.Generator().manual_seed(seed)
    k = torch.zeros(width, dim)
    perm = torch.randperm(width, generator=g)
    k[:, 0] = perm.float() * 4  # exact in bf16 below 2^8 * 4
    q = torch.zeros(rows, heads, dim)
    q[:, :, 0] = 1.0
    return q.to(torch.bfloat16), k.to(torch.bfloat16)


@pytest.mark.parametrize("width", [7, 64, 200])
def test_bf16_scores_equal_fp32_on_bf16_inputs(width):
    torch.manual_seed(0)
    q = torch.randn(33, 4, 32).to(torch.bfloat16)
    k = torch.randn(width, 32).to(torch.bfloat16)
    row_lo = torch.zeros(33, dtype=torch.long)
    row_hi = torch.randint(0, width + 1, (33,))
    a = qsa_score.score_blocks(q, k, row_lo, row_hi, 0, math.sqrt(32), "fp32")
    b = qsa_score.score_blocks(q, k, row_lo, row_hi, 0, math.sqrt(32), "bf16")
    assert a.dtype == b.dtype == torch.float32
    assert torch.equal(torch.isfinite(a), torch.isfinite(b))
    fin = torch.isfinite(a)
    torch.testing.assert_close(a[fin], b[fin], rtol=1e-6, atol=1e-6)


def test_bf16_selection_equals_fp32_on_well_separated_scores():
    q, k = _well_separated(48, 4, 120, 32)
    row_lo = torch.zeros(48, dtype=torch.long)
    row_hi = torch.linspace(10, 120, 48).long()
    sel = {}
    for dtype in qsa_score.SCORE_DTYPES:
        logits = qsa_score.score_blocks(q, k, row_lo, row_hi, 0, math.sqrt(32), dtype)
        sel[dtype] = qsa_indexer.select_topk_blocks(logits, 8)[1]
    assert torch.equal(sel["fp32"], sel["bf16"])


def test_unknown_score_dtype_raises(monkeypatch):
    with pytest.raises(ValueError, match="score_dtype"):
        _indexer(monkeypatch, score_dtype="fp16")
    with pytest.raises(ValueError, match="score dtype"):
        qsa_score.score_blocks(torch.zeros(1, 1, 4), torch.zeros(1, 4), torch.zeros(1), torch.ones(1), 0, 2.0, "x")


def test_score_dtype_default_is_fp32(monkeypatch):
    indexer = _indexer(monkeypatch)
    assert indexer.score_dtype == "fp32"
    del indexer.config.qwen3_8_next_indexer_score_dtype
    monkeypatch.setattr(qsa_indexer, "TELinear", _DuplicatedLinear)
    assert qsa_indexer.Qwen38NextQSAIndexer(indexer.config, layer_number=1).score_dtype == "fp32"


@pytest.mark.parametrize("lens", [[300], [97, 160, 41]])
def test_indexer_bf16_matches_fp32_on_bf16_model(monkeypatch, lens):
    """A bf16 indexer (the training dtype): the bf16 knob changes only the fp32 summation order of exact products,
    so on these inputs the selection is the same in nearly every row and the logits agree to fp32 rounding."""
    hidden, positions, rope, cu = _inputs(lens, dtype=torch.bfloat16)
    got = {}
    for dtype in qsa_score.SCORE_DTYPES:
        indexer = _indexer(monkeypatch, score_dtype=dtype, params_dtype=torch.bfloat16)
        indexer.to(torch.bfloat16)
        with torch.no_grad():
            got[dtype] = attention.qsa_selected_blocks(indexer, hidden, positions, rope, cu)[0]
    a = got["fp32"].sort(-1).values
    b = got["bf16"].sort(-1).values
    assert a.shape == b.shape
    assert (a == b).all(-1).float().mean() >= 0.99


class _FakeTPGroup:
    """Runs ``qsa_selected_blocks`` once per rank; the last call's all-gather returns every rank's shard."""

    def __init__(self, n):
        self.n = n
        self.rank = 0
        self.shards = {}


def _fake_dist(monkeypatch, group):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda g: g.n)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda g: g.rank)

    def all_gather_into_tensor(out, inp, group):
        group.shards[group.rank] = inp.clone()
        if len(group.shards) == group.n:
            out.copy_(torch.cat([group.shards[r] for r in range(group.n)]))
        else:
            out.fill_(-7)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", all_gather_into_tensor)


@pytest.mark.parametrize("n", [2, 3, 4, 8])
@pytest.mark.parametrize("lens", [[300], [97, 160, 41], [5]])
@pytest.mark.parametrize("tile_bytes", [1 << 20, 4096])
def test_tp_sharded_selection_equals_replicated(monkeypatch, lens, n, tile_bytes):
    indexer = _indexer(monkeypatch, tile_bytes=tile_bytes)
    hidden, positions, rope, cu = _inputs(lens)
    with torch.no_grad():
        want, want_cu = attention.qsa_selected_blocks(indexer, hidden, positions, rope, cu)
        group = _FakeTPGroup(n)
        _fake_dist(monkeypatch, group)
        for rank in range(n):
            group.rank = rank
            got, got_cu = attention.qsa_selected_blocks(indexer, hidden, positions, rope, cu, tp_group=group)
    assert got.dtype == want.dtype == torch.int32
    assert torch.equal(got, want)
    assert torch.equal(got_cu, want_cu)


def test_tp_shard_rows_follow_the_shard(monkeypatch):
    """Row i of shard (r, n) is query r + i * n: compare each shard against the replicated rows."""
    indexer = _indexer(monkeypatch)
    hidden, positions, rope, cu = _inputs([97, 160, 41])
    with torch.no_grad():
        full = torch.cat([b for _, _, b in indexer.iter_selected_blocks(hidden, positions, rope, cu)])
        for r in range(3):
            part = torch.cat(
                [b for _, _, b in indexer.iter_selected_blocks(hidden, positions, rope, cu, row_shard=(r, 3))]
            )
            assert torch.equal(part, full[r::3])


def test_num_blocks_matches_layout(monkeypatch):
    indexer = _indexer(monkeypatch)
    for lens in ([300], [97, 160, 41], [1, 2, 3, 4, 5]):
        _, positions, _, cu = _inputs(lens)
        layout = indexer.layout_for(positions, cu)
        want = layout.num_blocks if layout is not None else -(-positions.shape[0] // RATIO)
        assert indexer.num_blocks(positions, cu) == want


def _attention_with_flag(flag, tp):
    layer = attention.Qwen38NextAttention.__new__(attention.Qwen38NextAttention)
    object.__setattr__(layer, "config", SimpleNamespace(qwen3_8_next_indexer_tp_shard=flag))
    return layer


@pytest.mark.parametrize("flag,tp,want", [(False, 8, False), (True, 1, False), (True, 8, True)])
def test_tp_group_follows_flag(monkeypatch, flag, tp, want):
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(attention, "get_tensor_model_parallel_group", lambda: "tp-group")
    layer = _attention_with_flag(flag, tp)
    assert (layer._indexer_tp_group() == "tp-group") is want


def test_spec_sets_indexer_knobs(monkeypatch):
    """get_qwen3_8_next_spec copies --qwen38-indexer-score-dtype / --qwen38-indexer-tp-shard to the config."""
    from miles_plugins.models.qwen3_8_next import qwen3_8_next

    config = SimpleNamespace()
    args = SimpleNamespace(qwen38_indexer_score_dtype="bf16", qwen38_indexer_tp_shard=True)

    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: None)
    monkeypatch.setattr(qwen3_8_next, "_get_text_config", lambda hf: None)
    monkeypatch.setattr(qwen3_8_next, "_apply_qwen3_8_next_config", lambda c, t: None)
    args.hf_checkpoint = "/x"
    args.backend = SimpleNamespace(position_embedding_type="rope")
    # the knobs are set before the VPP guard; the rest of the spec needs a checkpoint
    config.virtual_pipeline_model_parallel_size = 2
    with pytest.raises(NotImplementedError, match="interleaved pipeline"):
        qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)
    assert config.qwen3_8_next_indexer_score_dtype == "bf16"
    assert config.qwen3_8_next_indexer_tp_shard is True


def test_cli_defaults():
    from miles.utils.args.configs.custom_megatron_plugins import Qwen38VisionMegatronPluginsConfig

    fields = Qwen38VisionMegatronPluginsConfig.model_fields
    assert fields["qwen38_indexer_score_dtype"].default == "fp32"
    assert fields["qwen38_indexer_tp_shard"].default is False
