"""QSA indexer block scores: ``sum_h relu(q_h . k) / sqrt(d)``, masked to each row's visible blocks.

Two precisions, chosen by ``--qwen38-indexer-score-dtype``:

- ``fp32`` (default): fp32 operands through ``torch.einsum``. On Blackwell this is a SIMT SGEMM (no tensor
  cores) and materializes the ``[rows, blocks, heads]`` score before the head sum.
- ``bf16``: bf16 operands, fp32 accumulation, ReLU, head sum, scale and mask fused in one Triton kernel that
  writes only the fp32 ``[rows, blocks]`` logits. This is what sglang's prefill indexer does
  (``layers/attention/qsa/mqa.py``: bf16 TileLang GEMM, fp32 logits). The indexer's q and compressed keys are
  bf16 tensors already, so the bf16 operands are exact and the two modes differ only in fp32 summation order.
  Off CUDA (tests) the fused kernel is replaced by the same math in torch.
"""

import torch
from torch import Tensor

SCORE_DTYPES = ("fp32", "bf16")

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False


def _mask_logits(logits: Tensor, row_lo: Tensor, row_hi: Tensor, col_lo: int) -> Tensor:
    cols = torch.arange(col_lo, col_lo + logits.shape[1], device=logits.device).unsqueeze(0)
    valid = (cols >= row_lo.unsqueeze(1)) & (cols < row_hi.unsqueeze(1))
    return logits.masked_fill_(~valid, float("-inf"))


def score_blocks_fp32(q: Tensor, block_k: Tensor, row_lo: Tensor, row_hi: Tensor, col_lo: int, scale: float):
    """The reference path: fp32 einsum, ReLU, head sum, divide, mask. ``block_k`` is the visible key window."""
    scores = torch.einsum("mhd,nd->mnh", q.float(), block_k.float())
    logits = scores.relu_().sum(dim=-1)
    del scores
    logits.div_(scale)
    return _mask_logits(logits, row_lo, row_hi, col_lo)


def score_blocks_bf16_torch(q: Tensor, block_k: Tensor, row_lo: Tensor, row_hi: Tensor, col_lo: int, scale: float):
    """bf16 operands, fp32 math: the fused kernel's semantics without Triton (CPU tests, fallback)."""
    return score_blocks_fp32(q.to(torch.bfloat16), block_k.to(torch.bfloat16), row_lo, row_hi, col_lo, scale)


if HAS_TRITON:

    @triton.jit
    def _qsa_score_kernel(
        Q,
        K,
        OUT,
        ROW_LO,
        ROW_HI,
        rows,
        width,
        col_lo,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_om,
        scale,
        H: tl.constexpr,
        D: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < rows
        n_ok = offs_n < width
        lo = tl.load(ROW_LO + offs_m, mask=m_ok, other=2147483647).to(tl.int64)
        hi = tl.load(ROW_HI + offs_m, mask=m_ok, other=0).to(tl.int64)
        cols = col_lo + offs_n
        tile_first = col_lo + pid_n * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        # a tile no row of this block can see is written as -inf without the GEMM (causal upper half)
        if (tile_first < tl.max(hi, axis=0)) & (tile_first + BN > tl.min(lo, axis=0)):
            k = tl.load(K + offs_n[:, None] * stride_kn + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            kt = tl.trans(k)
            for h in tl.static_range(H):
                q = tl.load(
                    Q + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :], mask=m_ok[:, None], other=0.0
                )
                acc += tl.maximum(tl.dot(q, kt), 0.0)
            acc = acc / scale
        valid = (cols[None, :] >= lo[:, None]) & (cols[None, :] < hi[:, None])
        out = tl.where(valid, acc, float("-inf"))
        tl.store(OUT + offs_m[:, None] * stride_om + offs_n[None, :], out, mask=m_ok[:, None] & n_ok[None, :])


def score_blocks_bf16_triton(
    q: Tensor, block_k: Tensor, row_lo: Tensor, row_hi: Tensor, col_lo: int, scale: float
) -> Tensor:
    """Fused bf16 scorer. ``q`` ``[rows, H, D]``, ``block_k`` ``[width, D]`` (blocks ``col_lo ..``)."""
    rows, heads, dim = q.shape
    width = block_k.shape[0]
    qb = q.to(torch.bfloat16).contiguous()
    kb = block_k.to(torch.bfloat16).contiguous()
    lo = row_lo.contiguous()
    hi = row_hi.contiguous()
    out = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if rows == 0 or width == 0:
        return out
    bm, bn = 64, 128
    grid = (triton.cdiv(rows, bm), triton.cdiv(width, bn))
    _qsa_score_kernel[grid](
        qb,
        kb,
        out,
        lo,
        hi,
        rows,
        width,
        col_lo,
        qb.stride(0),
        qb.stride(1),
        kb.stride(0),
        out.stride(0),
        float(scale),
        H=heads,
        D=dim,
        BM=bm,
        BN=bn,
        num_warps=4,
        num_stages=1,
    )
    return out


def score_blocks(
    q: Tensor, block_k: Tensor, row_lo: Tensor, row_hi: Tensor, col_lo: int, scale: float, dtype: str = "fp32"
) -> Tensor:
    """``[rows, width]`` fp32 logits of rows ``q`` against blocks ``col_lo .. col_lo + width``; ``-inf`` where a row
    may not look (``[row_lo, row_hi)`` per row, absolute block ids)."""
    if dtype == "fp32":
        return score_blocks_fp32(q, block_k, row_lo, row_hi, col_lo, scale)
    if dtype == "bf16":
        if HAS_TRITON and q.is_cuda and q.shape[-1] in (16, 32, 64, 128, 256):
            return score_blocks_bf16_triton(q, block_k, row_lo, row_hi, col_lo, scale)
        return score_blocks_bf16_torch(q, block_k, row_lo, row_hi, col_lo, scale)
    raise ValueError(f"QSA indexer score dtype must be one of {SCORE_DTYPES}, got {dtype!r}")
