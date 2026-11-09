"""``--qwen38-indexer-fast-topk`` on GPU: the Triton top-k gives the reference block ids bit for bit.

- the CPU suite's adversarial cases (``tests/fast/test_qsa_indexer_fast_topk.py``), at two widths, with the
  small-scratch settings that force every branch and with the production settings;
- tiles of the production shape (4 heads x 128, k = 512 blocks, up to 65,536 blocks wide) scored by the
  bf16 indexer kernel;
- the indexer end to end (real TELinear, bf16 scores, packed and TP-sharded rows, host bounds): the same
  ``[T, K]`` ids with the flag on and off, and no synchronizing call in the tile loop with the flag on.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

import math
import warnings

import pytest
import torch

from tests.fast.test_qsa_indexer_fast_topk import (
    adversarial_cases,
    contract_oracle_ids,
    in_domain,
    outside_domain_cases,
    reference_block_ids,
)

HIDDEN = 2560

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _first_bad_row(got, want):
    bad = (got != want).any(1).nonzero().flatten()
    if bad.numel() == 0:
        return None
    r = int(bad[0])
    return f"{bad.numel()} rows differ; row {r}: got {got[r, :12].tolist()} want {want[r, :12].tolist()}"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scale", [1, 8])
@pytest.mark.parametrize("settings", ["case", "production"])
def test_adversarial_cases(seed, scale, settings):
    from miles_plugins.models.qwen3_8_next.ops.qsa_topk import select_topk_block_ids

    for name, logits, kk, col_lo, kw in adversarial_cases(seed, scale):
        assert all(in_domain(logits, kk)), name
        oracle = contract_oracle_ids(logits, kk, col_lo)
        logits = logits.cuda()
        want = reference_block_ids(logits, kk, col_lo)
        got = select_topk_block_ids(logits, kk, col_lo, **(kw if settings == "case" else {}))
        assert torch.equal(got, want), (name, _first_bad_row(got, want))
        assert torch.equal(got.cpu(), oracle), (name, "oracle")


@pytest.mark.parametrize("settings", ["case", "production"])
def test_outside_domain_follows_the_contract(settings):
    """NaN of either sign / any payload, several NaN, +inf at the cut: the kernel's contract (independent
    oracle) on the GPU too; the documented counterexample differs from the reference."""
    from miles_plugins.models.qwen3_8_next.ops.qsa_topk import select_topk_block_ids

    for name, logits, kk, col_lo, kw, must_differ in outside_domain_cases():
        oracle = contract_oracle_ids(logits, kk, col_lo)
        got = select_topk_block_ids(logits.cuda(), kk, col_lo, **(kw if settings == "case" else {})).cpu()
        assert torch.equal(got, oracle), (name, got.tolist(), oracle.tolist())
        if must_differ:
            assert not torch.equal(got, reference_block_ids(logits.cuda(), kk, col_lo).cpu()), name


def production_tile(rows, width, seed=0, causal_from=None, device="cuda"):
    """``[rows, width]`` fp32 logits as the bf16 indexer kernel scores them: 4 heads x 128, ReLU sum / sqrt(128).

    ``causal_from`` = the block count the first row sees; row ``i`` sees ``causal_from + i`` blocks (TP-sharded
    rows of one sequence step one block per row), the rest is ``-inf``.
    """
    from miles_plugins.models.qwen3_8_next.ops.qsa_score import score_blocks

    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(rows, 4, 128, generator=g, device=device).bfloat16()
    k = torch.randn(width, 128, generator=g, device=device).bfloat16()
    lo = torch.zeros(rows, dtype=torch.long, device=device)
    if causal_from is None:
        hi = torch.full((rows,), width, dtype=torch.long, device=device)
    else:
        hi = (causal_from + torch.arange(rows, device=device)).clamp(max=width)
    return score_blocks(q, k, lo, hi, 0, math.sqrt(128), "bf16")


@pytest.mark.parametrize(
    "rows,width,causal_from",
    [(1170, 45696, 45696 - 1170), (1200, 65536, None), (4800, 11000, 6200), (20000, 2048, 512), (300, 600, 1)],
)
def test_production_tiles(rows, width, causal_from):
    from miles_plugins.models.qwen3_8_next.ops.qsa_topk import select_topk_block_ids

    logits = production_tile(rows, width, causal_from=causal_from)
    kk = min(512, width)
    want = reference_block_ids(logits, kk, 3)
    got = select_topk_block_ids(logits, kk, 3)
    assert torch.equal(got, want), _first_bad_row(got, want)


def _indexer(fast_topk, tile_bytes=1 << 30):
    from megatron.core.transformer.transformer_config import TransformerConfig

    from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer

    config = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=32, params_dtype=torch.bfloat16, bf16=True
    )
    config.qwen3_8_next_indexer_n_heads = 4
    config.qwen3_8_next_indexer_kv_heads = 1
    config.qwen3_8_next_indexer_head_dim = 128
    config.qwen3_8_next_indexer_budget = 2048
    config.qwen3_8_next_indexer_compress_ratio = 4
    config.qwen3_8_next_indexer_tile_bytes = tile_bytes
    config.qwen3_8_next_indexer_score_dtype = "bf16"
    config.qwen3_8_next_indexer_fast_topk = fast_topk
    torch.manual_seed(0)
    indexer = Qwen38NextQSAIndexer(config, layer_number=1).cuda()
    with torch.no_grad():
        indexer.index_qk_proj.weight.normal_(0, HIDDEN**-0.5)
        indexer.q_layernorm.normal_(0, 0.1)
        indexer.k_layernorm.normal_(0, 0.1)
    return indexer


def indexer_inputs(lens, seed=1):
    from miles_plugins.models.qwen3_8_next.ops.attention import Qwen38NextAttention

    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32).cuda()
    total = int(cu[-1])
    positions = Qwen38NextAttention._packed_positions(cu, total)
    torch.manual_seed(seed)
    hidden = torch.randn(total, HIDDEN, device="cuda", dtype=torch.bfloat16)
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))
    freqs = torch.outer(torch.arange(max(lens) + 1, dtype=torch.float32, device="cuda"), inv_freq)
    rope = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    return hidden, positions, rope, cu, tuple(int(c) for c in cu.tolist())


def selected_blocks(indexer, hidden, positions, rope, cu, cu_host, shard, count_syncs=False):
    """The rows of ``row_shard = shard`` as ``qsa_selected_blocks`` writes them, the synchronizing calls torch
    reported while tiles 2.. ran (``torch.cuda.set_sync_debug_mode``), and the tile count."""
    parts, syncs = [], 0
    gen = indexer.iter_selected_blocks(hidden, positions, rope, cu, row_shard=shard, cu_seqlens_host=cu_host)
    with torch.no_grad():
        while True:
            if count_syncs:
                torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    item = next(gen, None)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            if count_syncs and parts:  # the first tile also runs the projection and layout set-up
                syncs += sum("synchroniz" in str(w.message) for w in caught)
            if item is None:
                break
            parts.append(item[2].to(torch.int32))
    return torch.cat(parts), syncs, len(parts)


@pytest.mark.parametrize("lens", [[65536], [20000, 30000, 15536], [1, 3, 5000, 1, 7, 27756]])
@pytest.mark.parametrize("shard", [None, (0, 4), (3, 4)])
def test_indexer_end_to_end(lens, shard):
    hidden, positions, rope, cu, cu_host = indexer_inputs(lens)
    # a small tile budget: many tiles, so the sync count has something to count
    want, ref_syncs, tiles = selected_blocks(
        _indexer(False, 1 << 27), hidden, positions, rope, cu, cu_host, shard, count_syncs=True
    )
    got, fast_syncs, _ = selected_blocks(_indexer(True, 1 << 27), hidden, positions, rope, cu, cu_host, shard, True)
    assert torch.equal(got, want), _first_bad_row(got, want)
    assert (want >= 0).any()
    print(f"\nlens={lens} shard={shard} tiles={tiles} reference syncs after tile 1: {ref_syncs}, fast: {fast_syncs}")
    assert tiles > 2 and ref_syncs >= tiles - 1  # positive control: the reference syncs once per tile
    assert fast_syncs == 0
