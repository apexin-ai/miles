"""Tiled QSA indexer on GPU: same selection as the untiled scorer, bounded peak memory.

The packed path's producer (``qsa_selected_blocks``) writes ``[T, K]`` int32 block ids for
``build_qsa_block_index``; peaks are reported for the scorer + producer and for the index build.

Uses the real TELinear projection and the Qwen3.8 indexer shape (4 heads x 128,
budget 2048, ratio 4). Run with ``-s`` to see the measured peaks.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest
import torch

from tests.fast.test_qsa_indexer_tiled import (
    _packed_positions,
    _reference_selected_blocks,
    check_block_selection,
    classify_differences,
    dense_blocks,
)

GiB = 1 << 30
HIDDEN = 2560


def _indexer(tile_bytes):
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
    torch.manual_seed(0)
    indexer = Qwen38NextQSAIndexer(config, layer_number=1).cuda()
    with torch.no_grad():
        indexer.index_qk_proj.weight.normal_(0, HIDDEN**-0.5)
        indexer.q_layernorm.normal_(0, 0.1)
        indexer.k_layernorm.normal_(0, 0.1)
    return indexer


def _inputs(lens):
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32)
    total = int(cu[-1])
    positions = (_packed_positions(cu, total) if len(lens) > 1 else torch.arange(total)).cuda()
    torch.manual_seed(1)
    hidden = torch.randn(total, HIDDEN, device="cuda", dtype=torch.bfloat16)
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device="cuda") / 64))
    freqs = torch.outer(torch.arange(max(lens) + 1, dtype=torch.float32, device="cuda"), inv_freq)
    rope = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    return hidden, positions, rope, (cu.cuda() if len(lens) > 1 else None)


def _peak(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) / GiB


def _row_mismatch(got_bitmap, want_bitmap, logits, k):
    """Rows whose block set differs from the untiled one, split into exact k-th-score tie swaps and
    fp near swaps by the scores of the blocks actually swapped. ``check_block_selection`` fails
    on anything else (missing strictly-better block, wrong count, block below the band,
    difference outside the candidates)."""
    check_block_selection(got_bitmap, want_bitmap, logits, k, tol=1e-5, lowest_index=False)
    exact, near = classify_differences(got_bitmap, want_bitmap, logits, k)
    return int(exact.sum()), int(near.sum()), 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "lens",
    [[4096], [32768], [1, 3, 5000, 1, 7, 27756], [65536]],
    ids=["T4096", "T32768", "T32768-packed", "T65536"],
)
def test_tiled_selected_blocks_match_untiled_and_stay_small(lens):
    from miles_plugins.models.qwen3_8_next.ops.attention import qsa_selected_blocks

    indexer = _indexer(GiB)
    hidden, positions, rope, cu = _inputs(lens)
    with torch.no_grad():
        (blocks, _), tiled_peak = _peak(lambda: qsa_selected_blocks(indexer, hidden, positions, rope, cu))
        (want, logits), ref_peak = _peak(lambda: _reference_selected_blocks_cuda(indexer, hidden, positions, rope, cu))
    k = min(indexer.block_topk, logits.shape[-1])
    assert blocks.dtype == torch.int32 and blocks.shape == (len(positions), k)
    got = dense_blocks(blocks, want.shape[1])
    assert torch.equal(got.sum(-1), (blocks >= 0).sum(-1)), "a row repeats a block id"
    ties, near, bad = _row_mismatch(got, want, logits, k)
    ids = blocks.numel() * blocks.element_size() / GiB
    print(
        f"\nT={sum(lens)} rows={len(positions)} tiled_peak={tiled_peak:.2f}GiB (block ids {ids:.2f}) "
        f"untiled_peak={ref_peak:.2f}GiB tie_rows={ties} near_tie_rows={near} mismatch_rows={bad}"
    )
    assert bad == 0
    assert near <= max(1, len(positions) // 1000)
    assert tiled_peak <= ids + 1.5


def _reference_selected_blocks_cuda(indexer, hidden, positions, rope, cu):
    # the CPU reference builds index tensors on the default device
    with torch.device("cuda"):
        return _reference_selected_blocks(indexer, hidden, positions, rope, cu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("lens", [[131072], [262144], [247145, 14999]], ids=["T131072", "T262144", "T262144-packed"])
def test_long_sequence_indexer_peak(lens):
    """No untiled reference here (it would need 144 / 576 GiB): peaks and runtime only.

    The packed case is the one where a tile's rows span both documents, so its scored window
    is narrow while the batch's block count is the full grid.
    """
    import time

    from miles.kernels.attention.qsa import build_qsa_block_index
    from miles_plugins.models.qwen3_8_next.ops.attention import qsa_selected_blocks

    indexer = _indexer(GiB)
    hidden, positions, rope, cu = _inputs(lens)
    with torch.no_grad():
        t0 = time.perf_counter()
        (blocks, cu_used), peak = _peak(lambda: qsa_selected_blocks(indexer, hidden, positions, rope, cu))
        elapsed = time.perf_counter() - t0
        # Qwen3.8-Flash-Next at TP 2: 12 query heads over 1 kv head
        _, index_peak = _peak(
            lambda: build_qsa_block_index(
                blocks, cu_used, compress_ratio=indexer.compress_ratio, num_heads=12, num_kv_heads=1
            )
        )
    ids = blocks.numel() * blocks.element_size() / GiB
    print(
        f"\nT={sum(lens)} lens={lens} tiled_peak={peak:.2f}GiB (block ids {ids:.2f}) time={elapsed:.2f}s "
        f"index_build_peak={index_peak:.2f}GiB"
    )
    assert peak <= ids + 2.5
