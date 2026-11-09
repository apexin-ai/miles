"""Hyper-connection mix / inject, kernel v2 (``--qwen38-hc-kernel-v2``, default off): GEMMs fused with their
elementwise prologues and epilogues.

Same math as ``hc._HCMixInject`` (see ``hc.py``). The fused path (``--qwen38-hc-fused``) fused the elementwise
chains but kept every GEMM as a separate cuBLAS call on fp32 ``[T, n*C]`` operands, so each call still wrote and
read the normed activations, the up projection, its gradient and the low-rank path's ``d normed`` in fp32
(about 205 KB / token forward, 570 KB / token backward at n = 4, C = 2560), and its backward re-ran the whole
forward chain. On B300 its per-token norm / gate kernels reached 2.5-3.7 TB/s and the GEMMs 0.55-0.7 PFLOP/s
(H2 trace, T = 16384). v2 moves the GEMMs into Triton so the ``[T, n*C]`` side of each GEMM is produced or consumed
in registers:

forward (3 kernels)
  ``_v2_stats_kernel``   per-stream rstd and the inject projection ``u`` (fp32 dot, as the fused path) and h_post,
                         ``BLOCK_T`` tokens per program so the inject weight is read once per block, not per token;
  ``_v2_down_kernel``    ``a = normed @ w_down^T`` with ``normed = x * rstd * (1 + w)`` built in the GEMM prologue
                         from the bf16 input; the epilogue writes ``a`` (saved for the backward) and ``silu(a / n)``;
  ``_v2_up_mix_kernel``  ``z2 = silu @ w_up^T`` per stream for one column block, the epilogue applies the sigmoid,
                         multiplies by the normed input (recomputed from x) and averages over the n streams: writes
                         only ``mixed`` ``[T, C]``.
backward (5 Triton kernels + 2 cuBLAS GEMMs, plus a cat, dtype casts and one torch sum)
  ``_v2_silu_kernel``    ``silu(a / n)`` again (same expression as the down epilogue, so the same bits);
  ``_v2_up_bwd_kernel``  recomputes ``z2`` with the forward's tiles (bitwise the forward's), writes ``dz2`` and
                         ``q = dm * gate`` (the mix part of ``d normed``);
  cuBLAS                 ``ds1 = dz2 @ w_up`` and ``dw_up = dz2^T @ silu`` (as the fused path);
  ``_v2_build_g_kernel`` ``G = [silu'(a) * ds1 / n | du]`` ``[T, R + n]``: the low-rank and inject gradients side by
                         side, so one GEMM pair handles both (``du`` from the saved h_post: sigmoid(u/n) = h_post/2);
  ``_v2_wgrad_kernel``   ``[dw_down; dw_inject] = G^T @ normed`` with normed built in the prologue (no fp32 operand);
  ``_v2_dx_kernel``      one program per (token block, stream): ``p = G @ [w_down; w_inject]`` in two passes over the
                         stream's columns (pass 1: the rmsnorm-backward row dot and the norm-weight partials, pass 2:
                         dx), so neither ``p`` nor ``d normed`` is written.

Traffic per token at n = 4, C = 2560: about 70 KB forward and 360 KB backward (fused: 205 / 570). Tile sizes are fixed
constants (no autotune, no split-K, T not specialized), so a token's forward outputs do not depend on T or on the other
rows when the token sits at the same place in its tile (e.g. a row moved by a multiple of 128 tokens); whether the MMA's
per-row accumulation also ignores the row's place inside the tile is not guaranteed and is only reported by the GPU
test. (The default path's cuBLAS inject GEMM changed with T, see the HC root-cause note of 2026-11-08.) Two runs on
the same GPU and toolchain are bitwise equal.

Numerics (relative to the default path; none of this changes parameters or checkpoints):

- every GEMM keeps ``matmul_precision``'s operand rounding (fp32 / TF32 / bf16 operands, fp32 accumulation); the
  operands are the same fp32 values the fused path feeds cuBLAS, rounded to TF32 the way cuBLAS does (nearest, ties
  away, ``_rna_tf32``; not Triton's truncation); the accumulation order is Triton's;
- the inject projection is an fp32 dot product (as the fused path), its two gradient GEMMs follow
  ``matmul_precision`` (as the default path);
- dweight is a sum of per-token-block partials (block = the dx kernel's token tile).

Memory: saves ``a`` ``[T, R]`` fp32 (1.25 KB / token at R = 320) and h_post besides the fused path's tensors; the
backward's transient peak is ``dz2`` + ``q`` (2 x ``[T, n*C]``) instead of the fused path's four.
"""

from __future__ import annotations

import json
import os

import torch
import triton
import triton.language as tl

from miles.kernels.hyper_connection.hc import _mm, _tf32_scope, check_matmul_precision

_PREC = {"fp32": 0, "tf32": 1, "bf16": 2}

# Fixed tiles (no autotune: the results must not depend on T). BM / BN / BK of each GEMM kernel, num_warps,
# num_stages. MILES_HCV2_CONFIG (a JSON object with the same keys) overrides them for microbenchmarks only.
_DEFAULT_CONFIG = {
    "stats": {"BT": 4, "num_warps": 16},
    "down": {"BM": 128, "BN": 64, "BK": 64, "num_warps": 8, "num_stages": 3},
    "up": {"BM": 64, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 3},
    "wgrad": {"BM": 64, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 2},
    "dx": {"BM": 32, "BN": 128, "BK": 64, "num_warps": 8, "num_stages": 3},  # E2 2026-11-09: ~20% faster than 64/128/32
    "small": {"BT": 8, "num_warps": 4},
}


def _config():
    cfg = {k: dict(v) for k, v in _DEFAULT_CONFIG.items()}
    override = os.environ.get("MILES_HCV2_CONFIG")
    if override:
        for k, v in json.loads(override).items():
            cfg[k].update(v)
    return cfg


# The Triton interpreter (CPU tests) multiplies bf16 tensors as their raw 16-bit patterns: there, bf16 operands are
# rounded to bf16 and multiplied as fp32 (exact products, fp32 accumulation: the same math as the bf16 MMA).
_INTERPRET_BF16 = tl.constexpr(os.environ.get("TRITON_INTERPRET") == "1")


@triton.jit
def _rna_tf32(x):
    """fp32 -> the nearest TF32 value (10-bit mantissa), ties away from zero, as ``cvt.rna.tf32.f32`` and as cuBLAS /
    CUTLASS convert fp32 GEMM operands. Triton's tf32 dot hands the fp32 bits to the MMA, which drops the low 13 bits
    (truncation): an operand bias of -2^-11 on average, 3.7x the default path's error with fp32 activations (E1
    2026-11-09, test_fp32_activations[tf32]). Integer ops, so the interpreter runs it too; bf16 values are TF32-exact
    and pass unchanged. Non-finite values (exponent all ones: inf and every NaN payload) pass through bit for bit: the
    rounding bias would turn a small-payload NaN into inf and carry the largest payloads across the sign bit into 0
    (PERFERECHECK6 T1); a finite value just below the largest rounds up to inf, as round-to-nearest does."""
    bits = x.to(tl.int32, bitcast=True)
    rounded = (bits + 0x1000) & -8192
    finite = (bits & 0x7F800000) != 0x7F800000
    return tl.where(finite, rounded, bits).to(tl.float32, bitcast=True)


@triton.jit
def _dot(a, b, acc, PREC: tl.constexpr):
    """acc + a @ b: PREC 0 fp32 (IEEE), 1 TF32 operands (rounded to nearest, see _rna_tf32), 2 bf16 operands; fp32
    accumulation."""
    if PREC == 2:
        if _INTERPRET_BF16:
            return tl.dot(
                a.to(tl.bfloat16).to(tl.float32), b.to(tl.bfloat16).to(tl.float32), acc, input_precision="ieee"
            )
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
    elif PREC == 1:
        return tl.dot(_rna_tf32(a), _rna_tf32(b), acc, input_precision="tf32")
    else:
        return tl.dot(a, b, acc, input_precision="ieee")


@triton.jit(do_not_specialize=["T"])
def _v2_stats_kernel(
    x_ptr,  # x dtype [T, n*C]
    w_ptr,  # norm weight [n*C]
    winj_ptr,  # inject weight [n, n*C] (HAS_INJECT)
    rstd_ptr,  # fp32 out [T, n]
    hpost_ptr,  # fp32 out [T, n] (HAS_INJECT)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    ts = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = ts < T
    offs = tl.arange(0, BLOCK_C)
    cmask = offs < C
    m2 = tmask[:, None] & cmask[None, :]
    js = tl.arange(0, BLOCK_N)
    jmask = js < N
    u = tl.zeros([BLOCK_T, BLOCK_N], dtype=tl.float32)
    for c in tl.static_range(N):
        x = tl.load(x_ptr + ts[:, None] * (N * C) + c * C + offs[None, :], mask=m2, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + EPS)
        tl.store(rstd_ptr + ts * N + c, rstd, mask=tmask)
        if HAS_INJECT:
            w = tl.load(w_ptr + c * C + offs, mask=cmask, other=0.0).to(tl.float32)
            normed = x * rstd[:, None] * (1.0 + w[None, :])
            if ROUND_BF16:
                normed = normed.to(tl.bfloat16).to(tl.float32)
            for j in tl.static_range(N):
                wj = tl.load(winj_ptr + j * (N * C) + c * C + offs, mask=cmask, other=0.0)
                if ROUND_BF16:
                    wj = wj.to(tl.bfloat16)
                uj = tl.sum(normed * wj.to(tl.float32)[None, :], axis=1)
                u += tl.where(js[None, :] == j, uj[:, None], 0.0)
    if HAS_INJECT:
        tl.store(
            hpost_ptr + ts[:, None] * N + js[None, :],
            2.0 * tl.sigmoid(u / N),
            mask=tmask[:, None] & jmask[None, :],
        )


@triton.jit(do_not_specialize=["T"])
def _v2_down_kernel(
    x_ptr,  # x dtype [T, n*C]
    w_ptr,  # norm weight [n*C]
    rstd_ptr,  # fp32 [T, n]
    wd_ptr,  # w_down [R, n*C]
    a_ptr,  # fp32 out [T, R]: normed @ w_down^T
    s_ptr,  # out, matmul operand dtype [T, R]: silu(a / n)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
):
    W: tl.constexpr = N * C
    ts = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    tmask = ts < T
    rs = tl.program_id(1) * BN + tl.arange(0, BN)
    rmask = rs < R
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for c in tl.static_range(N):
        rstd = tl.load(rstd_ptr + ts * N + c, mask=tmask, other=0.0)
        for k0 in range(0, C, BK):
            ks = k0 + tl.arange(0, BK)
            kmask = ks < C
            x = tl.load(
                x_ptr + ts[:, None] * W + c * C + ks[None, :], mask=tmask[:, None] & kmask[None, :], other=0.0
            ).to(tl.float32)
            w = tl.load(w_ptr + c * C + ks, mask=kmask, other=0.0).to(tl.float32)
            normed = x * rstd[:, None] * (1.0 + w[None, :])
            b = tl.load(
                wd_ptr + rs[None, :] * W + c * C + ks[:, None], mask=kmask[:, None] & rmask[None, :], other=0.0
            ).to(tl.float32)
            acc = _dot(normed, b, acc, PREC)
    om = tmask[:, None] & rmask[None, :]
    tl.store(a_ptr + ts[:, None] * R + rs[None, :], acc, mask=om)
    z = acc / N
    tl.store(s_ptr + ts[:, None] * R + rs[None, :], (z * tl.sigmoid(z)).to(s_ptr.dtype.element_ty), mask=om)


@triton.jit(do_not_specialize=["numel"])
def _v2_silu_kernel(a_ptr, s_ptr, numel, N: tl.constexpr, BLOCK: tl.constexpr):
    """silu(a / n) with the down kernel's epilogue expression (same bits)."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    z = tl.load(a_ptr + offs, mask=mask, other=0.0) / N
    tl.store(s_ptr + offs, (z * tl.sigmoid(z)).to(s_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _up_tile(s_ptr, wu_ptr, ts, tmask, cs, cmask, c, C, R: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr, PREC: tl.constexpr):  # fmt: skip
    """z2[ts, c*C + cs] = silu[ts, :] @ w_up[c*C + cs, :]^T (fp32 [BM, BN]); the forward and backward use this."""
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, R, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < R
        a = tl.load(s_ptr + ts[:, None] * R + ks[None, :], mask=tmask[:, None] & kmask[None, :], other=0.0).to(
            tl.float32
        )
        b = tl.load(
            wu_ptr + (c * C + cs)[None, :] * R + ks[:, None], mask=kmask[:, None] & cmask[None, :], other=0.0
        ).to(tl.float32)
        acc = _dot(a, b, acc, PREC)
    return acc


@triton.jit(do_not_specialize=["T"])
def _v2_up_mix_kernel(
    s_ptr,  # matmul operand dtype [T, R]
    wu_ptr,  # w_up [n*C, R]
    x_ptr,
    w_ptr,
    rstd_ptr,
    out_ptr,  # x dtype [T, C]: mixed
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
):
    W: tl.constexpr = N * C
    ts = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    tmask = ts < T
    cs = tl.program_id(1) * BN + tl.arange(0, BN)
    cmask = cs < C
    m2 = tmask[:, None] & cmask[None, :]
    mix = tl.zeros([BM, BN], dtype=tl.float32)
    for c in tl.static_range(N):
        g = tl.sigmoid(_up_tile(s_ptr, wu_ptr, ts, tmask, cs, cmask, c, C, R, BM, BN, BK, PREC))
        rstd = tl.load(rstd_ptr + ts * N + c, mask=tmask, other=0.0)
        x = tl.load(x_ptr + ts[:, None] * W + c * C + cs[None, :], mask=m2, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c * C + cs, mask=cmask, other=0.0).to(tl.float32)
        normed = x * rstd[:, None] * (1.0 + w[None, :])
        mix += g * normed
    mix = mix / N
    tl.store(out_ptr + ts[:, None] * C + cs[None, :], mix.to(out_ptr.dtype.element_ty), mask=m2)


@triton.jit(do_not_specialize=["T"])
def _v2_up_bwd_kernel(
    s_ptr,
    wu_ptr,
    x_ptr,
    w_ptr,
    rstd_ptr,
    dm_ptr,  # [T, C] (cast on load)
    dz2_ptr,  # out, matmul operand dtype [T, n*C]: d of the up projection
    q_ptr,  # fp32 out [T, n*C]: dm * gate / n (the mix part of d normed)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
):
    W: tl.constexpr = N * C
    ts = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    tmask = ts < T
    cs = tl.program_id(1) * BN + tl.arange(0, BN)
    cmask = cs < C
    m2 = tmask[:, None] & cmask[None, :]
    dm = tl.load(dm_ptr + ts[:, None] * C + cs[None, :], mask=m2, other=0.0).to(tl.float32) / N
    for c in tl.static_range(N):
        g = tl.sigmoid(_up_tile(s_ptr, wu_ptr, ts, tmask, cs, cmask, c, C, R, BM, BN, BK, PREC))
        rstd = tl.load(rstd_ptr + ts * N + c, mask=tmask, other=0.0)
        x = tl.load(x_ptr + ts[:, None] * W + c * C + cs[None, :], mask=m2, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + c * C + cs, mask=cmask, other=0.0).to(tl.float32)
        normed = x * rstd[:, None] * (1.0 + w[None, :])
        dgate = dm * normed
        dz2 = dgate * g * (1.0 - g)
        off = ts[:, None] * W + c * C + cs[None, :]
        tl.store(dz2_ptr + off, dz2.to(dz2_ptr.dtype.element_ty), mask=m2)
        tl.store(q_ptr + off, dm * g, mask=m2)


@triton.jit(do_not_specialize=["T"])
def _v2_build_g_kernel(
    ds_ptr,  # fp32 [T, R]: d silu
    a_ptr,  # fp32 [T, R]: the down projection before the 1/n
    hpost_ptr,  # fp32 [T, n] (HAS_INJECT)
    dhp_ptr,  # fp32 [T, n] (HAS_INJECT)
    g_ptr,  # out, matmul operand dtype [T, M]: [dz1 | du], M = R (+ n)
    T,
    N: tl.constexpr,
    R: tl.constexpr,
    M: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_INJECT: tl.constexpr,
):
    ts = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = ts < T
    rs = tl.arange(0, BLOCK_R)
    m2 = tmask[:, None] & (rs < R)[None, :]
    ds = tl.load(ds_ptr + ts[:, None] * R + rs[None, :], mask=m2, other=0.0)
    z = tl.load(a_ptr + ts[:, None] * R + rs[None, :], mask=m2, other=0.0) / N
    sig = 1.0 / (1.0 + tl.exp(-z))
    dz = ds * sig * (1.0 + z * (1.0 - sig))
    dz = dz / N
    tl.store(g_ptr + ts[:, None] * M + rs[None, :], dz.to(g_ptr.dtype.element_ty), mask=m2)
    if HAS_INJECT:
        js = tl.arange(0, BLOCK_N)
        mj = tmask[:, None] & (js < N)[None, :]
        sig_u = tl.load(hpost_ptr + ts[:, None] * N + js[None, :], mask=mj, other=0.0) * 0.5  # exact: h_post = 2 s
        dhp = tl.load(dhp_ptr + ts[:, None] * N + js[None, :], mask=mj, other=0.0)
        du = dhp * 2.0 * sig_u * (1.0 - sig_u)
        du = du / N
        tl.store(g_ptr + ts[:, None] * M + R + js[None, :], du.to(g_ptr.dtype.element_ty), mask=mj)


@triton.jit(do_not_specialize=["T"])
def _v2_wgrad_kernel(
    g_ptr,  # matmul operand dtype [T, M]
    x_ptr,
    w_ptr,
    rstd_ptr,
    dwc_ptr,  # fp32 out [M, n*C]: G^T @ normed
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
    ONE_STREAM: tl.constexpr,  # C % BN == 0: a column block lies in one stream, rstd is a [BK] vector
):
    W: tl.constexpr = N * C
    ms = tl.program_id(0) * BM + tl.arange(0, BM)  # program 0 varies fastest: the M blocks of one column block
    mmask = ms < M  # run together and share the x tiles in L2
    ks = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    kmask = ks < W
    stream = ks // C
    w = tl.load(w_ptr + ks, mask=kmask, other=0.0).to(tl.float32)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for t0 in range(0, T, BK):
        ts = t0 + tl.arange(0, BK).to(tl.int64)
        tmask = ts < T
        gt = tl.load(g_ptr + ts[None, :] * M + ms[:, None], mask=mmask[:, None] & tmask[None, :], other=0.0).to(
            tl.float32
        )
        m2 = tmask[:, None] & kmask[None, :]
        x = tl.load(x_ptr + ts[:, None] * W + ks[None, :], mask=m2, other=0.0).to(tl.float32)
        if ONE_STREAM:
            stream0 = (tl.program_id(1).to(tl.int64) * BN) // C
            rstd = tl.load(rstd_ptr + ts * N + stream0, mask=tmask, other=0.0)[:, None]
        else:
            rstd = tl.load(rstd_ptr + ts[:, None] * N + stream[None, :], mask=m2, other=0.0)
        normed = x * rstd * (1.0 + w[None, :])
        acc = _dot(gt, normed, acc, PREC)
    tl.store(dwc_ptr + ms[:, None] * W + ks[None, :], acc, mask=mmask[:, None] & kmask[None, :])


@triton.jit
def _p_tile(g_ptr, wc_ptr, ts, tmask, col, cmask, W, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr, PREC: tl.constexpr):  # fmt: skip
    """p[ts, col] = G[ts, :] @ Wcat[:, col] (fp32 [BM, BN]); both passes of the dx kernel call this (same bits)."""
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, M, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < M
        a = tl.load(g_ptr + ts[:, None] * M + ks[None, :], mask=tmask[:, None] & kmask[None, :], other=0.0).to(
            tl.float32
        )
        b = tl.load(wc_ptr + ks[:, None] * W + col[None, :], mask=kmask[:, None] & cmask[None, :], other=0.0).to(
            tl.float32
        )
        acc = _dot(a, b, acc, PREC)
    return acc


@triton.jit(do_not_specialize=["T"])
def _v2_dx_kernel(
    g_ptr,  # matmul operand dtype [T, M]
    wc_ptr,  # [M, n*C]: [w_down; w_inject]
    q_ptr,  # fp32 [T, n*C]
    x_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,  # out, x dtype [T, n*C]
    dwp_ptr,  # fp32 out [cdiv(T, BM), n*C]: per-token-block partials of the norm weight grad
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PREC: tl.constexpr,
):
    W: tl.constexpr = N * C
    tb = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    ts = tb * BM + tl.arange(0, BM)
    tmask = ts < T
    rstd = tl.load(rstd_ptr + ts * N + c, mask=tmask, other=0.0)
    dot = tl.zeros([BM], dtype=tl.float32)
    for j0 in range(0, C, BN):  # pass 1: sum_C(d normed * (1 + w) * x) and the norm-weight partials
        cs = j0 + tl.arange(0, BN)
        cmask = cs < C
        col = c * C + cs
        m2 = tmask[:, None] & cmask[None, :]
        p = _p_tile(g_ptr, wc_ptr, ts, tmask, col, cmask, W, M, BM, BN, BK, PREC)
        dn = tl.load(q_ptr + ts[:, None] * W + col[None, :], mask=m2, other=0.0) + p
        x = tl.load(x_ptr + ts[:, None] * W + col[None, :], mask=m2, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col, mask=cmask, other=0.0).to(tl.float32)
        gg = dn * (1.0 + w[None, :])
        dot += tl.sum(gg * x, axis=1)
        tl.store(dwp_ptr + tb * W + col, tl.sum(dn * (x * rstd[:, None]), axis=0), mask=cmask)
    r3 = rstd * rstd * rstd
    for j0 in range(0, C, BN):  # pass 2: dx
        cs = j0 + tl.arange(0, BN)
        cmask = cs < C
        col = c * C + cs
        m2 = tmask[:, None] & cmask[None, :]
        p = _p_tile(g_ptr, wc_ptr, ts, tmask, col, cmask, W, M, BM, BN, BK, PREC)
        dn = tl.load(q_ptr + ts[:, None] * W + col[None, :], mask=m2, other=0.0) + p
        x = tl.load(x_ptr + ts[:, None] * W + col[None, :], mask=m2, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + col, mask=cmask, other=0.0).to(tl.float32)
        gg = dn * (1.0 + w[None, :])
        dx = rstd[:, None] * gg - x * r3[:, None] * (dot / C)[:, None]
        tl.store(dx_ptr + ts[:, None] * W + col[None, :], dx.to(dx_ptr.dtype.element_ty), mask=m2)


def _launch(cfg_key):
    cfg = _config()[cfg_key]
    return cfg, {k: cfg[k] for k in ("num_warps", "num_stages") if k in cfg}


class _HCMixInjectV2(torch.autograd.Function):
    """``hc._HCMixInject`` with the GEMMs fused with their elementwise chains (``--qwen38-hc-kernel-v2``)."""

    @staticmethod
    def forward(ctx, x2d, weight, w_down, w_up, w_inject, n, eps, precision="fp32"):
        T, W = x2d.shape
        C = W // n
        R = w_down.shape[0]
        dev = x2d.device
        prec = _PREC[precision]
        mm_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        has_inject = w_inject is not None
        # the kernels address every weight as contiguous rows: a strided view is copied once here and the same copy
        # is saved for the backward (the gradients keep the parameters' shapes)
        weight = weight.contiguous()
        w_down = w_down.contiguous()
        w_up = w_up.contiguous()
        if has_inject:
            w_inject = w_inject.contiguous()
        rstd = torch.empty(T, n, dtype=torch.float32, device=dev)
        h_post = torch.empty(T, n, dtype=torch.float32, device=dev) if has_inject else x2d.new_empty(0, dtype=torch.float32)
        a = torch.empty(T, R, dtype=torch.float32, device=dev)
        s = torch.empty(T, R, dtype=mm_dtype, device=dev)
        mixed = torch.empty(T, C, dtype=x2d.dtype, device=dev)
        if T > 0:
            cfg, kw = _launch("stats")
            _v2_stats_kernel[(triton.cdiv(T, cfg["BT"]),)](
                x2d, weight, w_inject if has_inject else weight, rstd, h_post if has_inject else rstd, T,
                N=n, C=C, EPS=eps, BLOCK_T=cfg["BT"], BLOCK_C=triton.next_power_of_2(C),
                BLOCK_N=triton.next_power_of_2(n), HAS_INJECT=has_inject, ROUND_BF16=precision == "bf16", **kw,
            )  # fmt: skip
            cfg, kw = _launch("down")
            _v2_down_kernel[(triton.cdiv(T, cfg["BM"]), triton.cdiv(R, cfg["BN"]))](
                x2d, weight, rstd, w_down, a, s, T, N=n, C=C, R=R, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"],
                PREC=prec, **kw,
            )  # fmt: skip
            cfg, kw = _launch("up")
            _v2_up_mix_kernel[(triton.cdiv(T, cfg["BM"]), triton.cdiv(C, cfg["BN"]))](
                s, w_up, x2d, weight, rstd, mixed, T, N=n, C=C, R=R, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"],
                PREC=prec, **kw,
            )  # fmt: skip
        del s
        ctx.save_for_backward(x2d, weight, w_down, w_up, w_inject, rstd, a, h_post if has_inject else None)
        ctx.n, ctx.eps, ctx.precision = n, eps, precision
        return mixed, h_post

    @staticmethod
    def backward(ctx, dmixed, dh_post):
        x2d, weight, w_down, w_up, w_inject, rstd, a, h_post = ctx.saved_tensors
        n, precision = ctx.n, ctx.precision
        T, W = x2d.shape
        C = W // n
        R = w_down.shape[0]
        dev = x2d.device
        prec = _PREC[precision]
        mm_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        has_inject = w_inject is not None and dh_post is not None and dh_post.numel() > 0
        M = R + n if has_inject else R
        dmixed = dmixed.contiguous()

        s = torch.empty(T, R, dtype=mm_dtype, device=dev)
        dz2 = torch.empty(T, W, dtype=mm_dtype, device=dev)
        q = torch.empty(T, W, dtype=torch.float32, device=dev)
        if T > 0:
            cfg, kw = _launch("small")
            _v2_silu_kernel[(triton.cdiv(T * R, 1024),)](a, s, T * R, N=n, BLOCK=1024)
            cfg, kw = _launch("up")
            _v2_up_bwd_kernel[(triton.cdiv(T, cfg["BM"]), triton.cdiv(C, cfg["BN"]))](
                s, w_up, x2d, weight, rstd, dmixed, dz2, q, T, N=n, C=C, R=R, BM=cfg["BM"], BN=cfg["BN"],
                BK=cfg["BK"], PREC=prec, **kw,
            )  # fmt: skip
        with _tf32_scope(precision):
            dw_up = _mm(dz2.t(), s, precision)
            ds1 = _mm(dz2, w_up, precision)
        del dz2, s
        g = torch.empty(T, M, dtype=mm_dtype, device=dev)
        if T > 0:
            cfg, kw = _launch("small")
            _v2_build_g_kernel[(triton.cdiv(T, cfg["BT"]),)](
                ds1.contiguous(), a, h_post if has_inject else a, dh_post.float().contiguous() if has_inject else a,
                g, T, N=n, R=R, M=M, BLOCK_T=cfg["BT"], BLOCK_R=triton.next_power_of_2(R),
                BLOCK_N=triton.next_power_of_2(n), HAS_INJECT=has_inject, num_warps=cfg["num_warps"],
            )  # fmt: skip
        del ds1
        wcat = torch.cat([w_down, w_inject]) if has_inject else w_down  # [M, W], weight dtype
        dwc = torch.empty(M, W, dtype=torch.float32, device=dev)
        cfg_dx, kw_dx = _launch("dx")
        dx = torch.empty_like(x2d)
        dwp = torch.empty(triton.cdiv(T, cfg_dx["BM"]), W, dtype=torch.float32, device=dev)
        if T > 0:
            cfg, kw = _launch("wgrad")
            _v2_wgrad_kernel[(triton.cdiv(M, cfg["BM"]), triton.cdiv(W, cfg["BN"]))](
                g, x2d, weight, rstd, dwc, T, N=n, C=C, M=M, BM=cfg["BM"], BN=cfg["BN"], BK=cfg["BK"], PREC=prec,
                ONE_STREAM=C % cfg["BN"] == 0, **kw,
            )  # fmt: skip
            _v2_dx_kernel[(triton.cdiv(T, cfg_dx["BM"]), n)](
                g, wcat, q, x2d, weight, rstd, dx, dwp, T, N=n, C=C, M=M, BM=cfg_dx["BM"], BN=cfg_dx["BN"],
                BK=cfg_dx["BK"], PREC=prec, **kw_dx,
            )  # fmt: skip
        else:
            dwc.zero_()
        del q, g
        dweight = dwp.sum(dim=0).to(weight.dtype) if T > 0 else torch.zeros_like(weight)
        dw_down = dwc[:R].to(w_down.dtype)
        dw_inject = dwc[R:].to(w_inject.dtype) if has_inject else None
        return (
            dx,
            dweight,
            dw_down,
            dw_up.to(w_up.dtype),
            dw_inject,
            None,
            None,
            None,
        )


def hc_mix_inject_v2(x, weight, w_down, w_up, w_inject, n, eps, precision):
    """``hc.hc_mix_inject(..., v2=True)``: any leading shape (0 tokens included) and strides (the kernels address x as
    contiguous rows, so the call goes through the public entry, which flattens x and makes it contiguous)."""
    from miles.kernels.hyper_connection.hc import hc_mix_inject

    return hc_mix_inject(x, weight, w_down, w_up, w_inject, n, eps, matmul_precision=precision, v2=True)
