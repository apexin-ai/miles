"""Tiled QSA indexer on GPU: same selection as the untiled scorer, bounded peak memory.

Uses the real TELinear projection and the Qwen3.8 indexer shape (4 heads x 128,
budget 2048, ratio 4). Run with ``-s`` to see the measured peaks.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=600, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest
import torch

from tests.fast.test_qsa_indexer_tiled import _packed_positions, _reference_block_form, _reference_forward

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
    """Rows whose bitmaps differ, split into exact ties, fp near-ties and real mismatches."""
    values = torch.topk(logits, k + 1 if logits.shape[-1] > k else k, dim=-1).values
    kth = values[:, k - 1 : k]
    exact_tie = torch.isfinite(kth.squeeze(-1)) & ((logits >= kth).sum(-1) > k)
    if values.shape[-1] > k:
        gap = (values[:, k - 1] - values[:, k]).abs()
        near_tie = torch.isfinite(values[:, k]) & (gap <= 1e-5 * values[:, k - 1].abs().clamp_min(1e-6))
    else:
        near_tie = torch.zeros_like(exact_tie)
    differ = (got_bitmap != want_bitmap).any(dim=-1)
    return int((differ & exact_tie).sum()), int((differ & near_tie & ~exact_tie).sum()), int((differ & ~exact_tie & ~near_tie).sum())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "lens",
    [[4096], [32768], [1, 3, 5000, 1, 7, 27756], [65536]],
    ids=["T4096", "T32768", "T32768-packed", "T65536"],
)
def test_tiled_block_form_matches_untiled_and_stays_small(lens):
    from miles_plugins.models.qwen3_8_next.ops.attention import qsa_block_form

    indexer = _indexer(GiB)
    hidden, positions, rope, cu = _inputs(lens)
    with torch.no_grad():
        got, tiled_peak = _peak(lambda: qsa_block_form(indexer, hidden, positions, rope, cu))
        (want, logits), ref_peak = _peak(lambda: _reference_block_form_cuda(indexer, hidden, positions, rope, cu))
    for a, b in zip(got[1:7], want[1:7], strict=True):
        assert torch.equal(a, b)
    k = min(indexer.block_topk, logits.shape[-1])
    ties, near, bad = _row_mismatch(got[0], want[0], logits, k)
    bitmap = got[0].numel() / GiB
    print(
        f"\nT={sum(lens)} rows={len(positions)} tiled_peak={tiled_peak:.2f}GiB (bitmap {bitmap:.2f}) "
        f"untiled_peak={ref_peak:.2f}GiB tie_rows={ties} near_tie_rows={near} mismatch_rows={bad}"
    )
    assert bad == 0
    assert near <= max(1, len(positions) // 1000)
    assert tiled_peak <= bitmap + 1.5


def _reference_block_form_cuda(indexer, hidden, positions, rope, cu):
    # the CPU reference builds index tensors on the default device
    with torch.device("cuda"):
        return _reference_block_form(indexer, hidden, positions, rope, cu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("T", [131072, 262144])
def test_long_sequence_indexer_peak(T):
    """No untiled reference here (it would need 144 / 576 GiB): peak and runtime only."""
    import time

    from miles_plugins.models.qwen3_8_next.ops.attention import qsa_block_form

    indexer = _indexer(GiB)
    hidden, positions, rope, cu = _inputs([T])
    with torch.no_grad():
        t0 = time.perf_counter()
        got, peak = _peak(lambda: qsa_block_form(indexer, hidden, positions, rope, cu))
        elapsed = time.perf_counter() - t0
    bitmap = got[0].numel() / GiB
    print(f"\nT={T} tiled_peak={peak:.2f}GiB (bitmap {bitmap:.2f}) time={elapsed:.2f}s")
    assert peak <= bitmap + 2.5
