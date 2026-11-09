"""QSA indexer top-k in one Triton kernel: the selection of ``select_topk_blocks``, no host sync.

``select_topk_blocks`` (``qsa_indexer.py``) is ``torch.topk`` plus a tie fix-up: the rows whose k-th
score is shared by more than k entries are found with ``bool(tied.any())`` -- one host sync per
indexer tile -- and re-selected keeping the lower block ids; two sorts then order the result.
``--qwen38-indexer-fast-topk`` replaces the three with ``select_topk_block_ids``, one program per
query row:

1. one read: candidate count and the smallest / largest key (an order-preserving int of the fp32 score);
2. radix passes, ``DIGIT`` bits each, starting at the highest bit where those two keys differ, until
   the bucket that holds the k-th largest key, plus everything above it, fits ``CAP`` slots (usually
   one to three passes for 45K-block rows and k = 512);
3. one read that copies those candidates (``key >= bucket floor``) to a ``[CAP]`` scratch row;
4. a sort of the scratch row by (score descending, column ascending); its first ``take`` entries are
   the answer, ``-1`` fills the rest of the ``kk`` slots.

If even the last bucket (one exact key) is too large -- more than ``CAP`` equal scores at the k-th
place -- step 3 copies only the lowest columns of it that are needed, which is the tie rule itself.

The rule (``select_topk_blocks``'s docstring, written out):

- candidates: the entries that are not ``-inf`` (``-inf`` is the masked score);
- keep the ``take = min(kk, candidates)`` largest, equal scores by lower column;
- order: score descending, then column ascending; ``-1`` fills the rest of the ``kk`` slots;
- scores compare as floats: ``-0.0 == +0.0`` (the key maps both to the same integer).

Bitwise the same block ids as ``select_topk_blocks`` + ``torch.where(values == -inf, -1, idx + col_lo)``
on every row whose scores hold no NaN and whose k-th largest score is not ``+inf``. Those two cases
have no fixed answer in the reference either (its tie fix-up skips a non-finite k-th score, and
``torch.topk`` keeps an unspecified one of equal scores); here NaN ranks above ``+inf`` as in
``torch.topk``, and equal values keep the lower column. The indexer's scores are finite or ``-inf``
for finite inputs (a sum of ReLUs divided by sqrt(d), masked with ``-inf``).
"""

import os

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAS_TRITON = False

# one program per row; a row is read in chunks of BLOCK columns
DEFAULT_BLOCK = 2048
DEFAULT_NUM_WARPS = 8
# bits per radix pass: tl.histogram costs about one popcount per bin and lane, so 2^7 bins per pass
DEFAULT_DIGIT = 7
# scratch slots per row (power of two, >= kk): the candidates left after the radix passes are sorted here
DEFAULT_CAP = 2048


if HAS_TRITON:

    @triton.jit
    def _ukey(x):
        """fp32 -> (int64 key in [0, 2**32) with the float order, is-candidate).

        Integer ops on the bits only, so no flush-to-zero setting can touch a subnormal: -0.0 and +0.0
        share a key, NaN is the largest key, ``-inf`` (the mask) is not a candidate.
        """
        b = x.to(tl.int32, bitcast=True)
        mag = b & 0x7FFFFFFF
        ok = b != -8388608  # 0xFF800000, -inf
        s = tl.where(mag == 0, 0, b)
        # negative floats: flip the magnitude bits so a larger magnitude gives a smaller signed int
        s = tl.where(s < 0, s ^ 0x7FFFFFFF, s)
        s = tl.where(mag > 0x7F800000, 0x7FFFFFFF, s)
        return s.to(tl.int64) + 2147483648, ok

    @triton.jit
    def _qsa_topk_kernel(
        LOGITS,
        OUT,
        SCRATCH,
        width,
        kk,
        col_lo,
        stride_l,
        stride_o,
        CAP: tl.constexpr,
        DIGIT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        NB: tl.constexpr = 1 << DIGIT
        row = tl.program_id(0).to(tl.int64)
        row_ptr = LOGITS + row * stride_l
        out_ptr = OUT + row * stride_o
        scr_ptr = SCRATCH + row * CAP

        # 1. candidate count, smallest and largest key
        n_cand = tl.zeros([], dtype=tl.int32)
        kmin = tl.full([], 4294967295, dtype=tl.int64)
        kmax = tl.zeros([], dtype=tl.int64)
        for start in range(0, width, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            x = tl.load(row_ptr + offs, mask=offs < width, other=float("-inf"))
            key, ok = _ukey(x)
            n_cand += tl.sum(ok.to(tl.int32), axis=0)
            kmin = tl.minimum(kmin, tl.min(tl.where(ok, key, 4294967295), axis=0))
            kmax = tl.maximum(kmax, tl.max(tl.where(ok, key, 0), axis=0))
        take = tl.minimum(n_cand, kk)

        # the bucket: keys with key >> level == prefix; `above` keys are larger than the bucket, all kept;
        # `need` more come from the bucket (`in_bucket` keys)
        diff = kmax ^ kmin
        level = tl.zeros([], dtype=tl.int32)
        for b in tl.static_range(32):
            level = tl.where((diff >> b) > 0, b + 1, level)
        prefix = kmax >> level
        above = tl.zeros([], dtype=tl.int32)
        need = take
        in_bucket = n_cand

        # 2. radix passes until the bucket and everything above it fit CAP slots, or the key is exact
        bins = tl.arange(0, NB)
        while (level > 0) & (above + in_bucket > CAP):
            shift = tl.maximum(level - DIGIT, 0)
            hist = tl.zeros([NB], dtype=tl.int32)
            for start in range(0, width, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                x = tl.load(row_ptr + offs, mask=offs < width, other=float("-inf"))
                key, ok = _ukey(x)
                ok = ok & ((key >> level) == prefix)
                # only the bits below `level`: the ones above are the prefix
                digit = ((key >> shift) & ((1 << (level - shift)) - 1)).to(tl.int32)
                hist += tl.histogram(digit, NB, mask=ok)
            csum = tl.cumsum(hist, axis=0)
            at_least = in_bucket - csum + hist  # keys of this bucket with digit >= bin
            j = tl.max(tl.where(at_least >= need, bins, -1), axis=0)
            higher = in_bucket - tl.sum(tl.where(bins == j, csum, 0), axis=0)
            above += higher
            need -= higher
            in_bucket = tl.sum(tl.where(bins == j, hist, 0), axis=0)
            prefix = (prefix << (level - shift)) | j.to(tl.int64)
            level = shift
        # copy the whole bucket when it fits, else (one exact key, many ties) its `need` lowest columns
        limit = tl.where(above + in_bucket <= CAP, in_bucket, need)

        # 3. copy the candidates to the scratch row, in column order
        count = tl.zeros([], dtype=tl.int32)
        seen = tl.zeros([], dtype=tl.int32)
        for start in range(0, width, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            x = tl.load(row_ptr + offs, mask=offs < width, other=float("-inf"))
            key, ok = _ukey(x)
            ok = ok & (take > 0)
            top = key >> level
            eq = (ok & (top == prefix)).to(tl.int32)
            rank = seen + tl.cumsum(eq, axis=0)
            sel = (ok & (top > prefix)) | ((eq > 0) & (rank <= limit))
            sel_i = sel.to(tl.int32)
            pos = count + tl.cumsum(sel_i, axis=0) - 1
            # ascending order of this int64 = score descending, then column ascending
            packed = (((4294967295 - key) << 32) | offs.to(tl.int64)) ^ (-9223372036854775807 - 1)
            tl.store(scr_ptr + pos, packed, mask=sel)
            count += tl.sum(sel_i, axis=0)
            seen += tl.sum(eq, axis=0)

        # 4. sort the scratch row (written by other threads of this program) and keep the first `take`
        tl.debug_barrier()
        slots = tl.arange(0, CAP)
        picked = tl.load(scr_ptr + slots, mask=slots < count, other=9223372036854775807)
        picked = tl.sort(picked)
        ids = tl.where(slots < take, (picked & 4294967295) + col_lo, -1)
        tl.store(out_ptr + slots, ids, mask=slots < kk)


def fast_topk_supported(logits: Tensor) -> bool:
    """The fused kernel's inputs: a CUDA fp32 ``[rows, width]`` with unit column stride.

    Under ``TRITON_INTERPRET=1`` (Triton's CPU interpreter, used by the CPU tests) CPU tensors run too.
    """
    on_device = logits.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"
    return HAS_TRITON and on_device and logits.dtype == torch.float32 and logits.dim() == 2 and logits.stride(1) == 1


def select_topk_block_ids(
    logits: Tensor,
    kk: int,
    col_lo: int = 0,
    out: Tensor | None = None,
    block: int = DEFAULT_BLOCK,
    num_warps: int = DEFAULT_NUM_WARPS,
    digit: int = DEFAULT_DIGIT,
    cap: int = DEFAULT_CAP,
) -> Tensor:
    """``[rows, width]`` fp32 scores -> ``[rows, kk]`` int64 column ids + ``col_lo``, ``-1`` where unused.

    Same ids as ``qsa_indexer.select_topk_blocks(logits, kk)`` followed by
    ``torch.where(values == -inf, -1, idx + col_lo)`` (module docstring). ``out`` may be a wider int64
    buffer (``[rows, >= kk]``, unit column stride): its first ``kk`` columns are written, the rest are
    left as they are. No host sync. Transient memory: a ``[rows, cap']`` int64 scratch,
    ``cap' = max(next_pow2(kk), min(cap, next_pow2(width)))``, at most 16 bytes per score.
    """
    rows, width = logits.shape
    if not 0 < kk <= width:
        raise ValueError(f"top-k needs 0 < kk <= width, got kk={kk}, width={width}")
    if not fast_topk_supported(logits):
        raise ValueError("select_topk_block_ids needs a CUDA fp32 [rows, width] tensor with unit column stride")
    if width >= 1 << 31:
        raise ValueError(f"width {width} does not fit the kernel's 32-bit column field")
    if not 5 <= digit <= 10:
        raise ValueError(f"digit must be 5..10 bits (tl.histogram wants at least one bin per lane), got {digit}")
    if out is None:
        out = torch.empty((rows, kk), dtype=torch.long, device=logits.device)
    if out.dtype != torch.long or out.shape[0] != rows or out.shape[1] < kk or out.stride(1) != 1:
        raise ValueError(f"out must be int64 [{rows}, >= {kk}] with unit column stride, got {tuple(out.shape)}")
    if rows == 0:
        return out[:, :kk]
    kpad = triton.next_power_of_2(kk)
    cap_eff = max(kpad, min(triton.next_power_of_2(cap), triton.next_power_of_2(width)))
    scratch = torch.empty((rows, cap_eff), dtype=torch.long, device=logits.device)
    _qsa_topk_kernel[(rows,)](
        logits,
        out,
        scratch,
        width,
        kk,
        col_lo,
        logits.stride(0),
        out.stride(0),
        CAP=cap_eff,
        DIGIT=digit,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out[:, :kk]
