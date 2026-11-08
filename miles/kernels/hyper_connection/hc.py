"""Triton hyper-connection kernels for Qwen3.8-Next (Qwen4Exp).

Every reduction and elementwise step runs in fp32, with one cast onto the output dtype
at the end. Token indices are promoted before multiplying by the stream width:
long sequences can exceed 2**31 elements even when the token count fits in int32.

The gate chain's matmuls (the low-rank down/up projections, the inject gate and their
gradients) take ``matmul_precision`` (``--qwen38-hc-matmul-precision``):

- ``fp32``: fp32 operands, fp32 math (SIMT SGEMM on Blackwell). The reference and the default.
- ``tf32``: the same fp32 matmuls with TF32 tensor cores allowed for these calls only.
- ``bf16``: bf16 operands (the weights are bf16 already; activations are rounded), fp32
  accumulation and fp32 results, as sglang's fused ``hc_mix`` kernel.

Elementwise math (rmsnorm, silu, sigmoid, the gated mean) stays fp32 in every mode.

``fused`` (``--qwen38-hc-fused``, default off) selects ``_HCMixInjectFused``: the same math with the elementwise
chains fused into fewer kernels (see the comment above it).
"""

import contextlib

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from miles.kernels.norm.grouped_rmsnorm import block_c, grouped_rmsnorm_bwd_kernel, grouped_rmsnorm_fwd


@triton.jit(do_not_specialize=["T"])
def _gate_mul_mean_fwd_kernel(
    gate_ptr,  # fp32 [T, n*C]
    normed_ptr,  # fp32 [T, n*C]
    out_ptr,  # out dtype [T, C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        g = tl.load(gate_ptr + base + offs, mask=mask, other=0.0)
        nrm = tl.load(normed_ptr + base + offs, mask=mask, other=0.0)
        acc += g * nrm
    acc = acc / N
    tl.store(out_ptr + t * C + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _gate_mul_mean_bwd_kernel(
    dmixed_ptr,  # fp32 [T, C]
    gate_ptr,
    normed_ptr,
    dgate_ptr,  # fp32 out [T, n*C]
    dnormed_ptr,  # fp32 out [T, n*C] (mix contribution only)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    dm = tl.load(dmixed_ptr + t * C + offs, mask=mask, other=0.0) / N
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        g = tl.load(gate_ptr + base + offs, mask=mask, other=0.0)
        nrm = tl.load(normed_ptr + base + offs, mask=mask, other=0.0)
        tl.store(dgate_ptr + base + offs, dm * nrm, mask=mask)
        tl.store(dnormed_ptr + base + offs, dm * g, mask=mask)


@triton.jit(do_not_specialize=["T"])
def _combine_fwd_kernel(
    res_ptr,  # bf16 [T, n*C]
    y_ptr,  # bf16 [T, C]
    hpost_ptr,  # fp32 [T, n]
    out_ptr,  # res dtype [T, n*C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // N
    c = pid % N
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    base = t * (N * C) + c * C
    r = tl.load(res_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(hpost_ptr + t * N + c)
    tl.store(out_ptr + base + offs, (r + a * y).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _combine_bwd_kernel(
    dout_ptr,  # bf16 [T, n*C] (incoming grad, cast on load)
    y_ptr,  # bf16 [T, C]
    hpost_ptr,  # fp32 [T, n]
    dy_ptr,  # out y dtype [T, C]
    dhpost_ptr,  # fp32 out [T, n]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    y = tl.load(y_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        do = tl.load(dout_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(hpost_ptr + t * N + c)
        dy += a * do
        tl.store(dhpost_ptr + t * N + c, tl.sum(do * y, axis=0))
    tl.store(dy_ptr + t * C + offs, dy.to(dy_ptr.dtype.element_ty), mask=mask)


MATMUL_PRECISIONS = ("fp32", "tf32", "bf16")


def check_matmul_precision(precision: str) -> str:
    if precision not in MATMUL_PRECISIONS:
        raise ValueError(f"hyper-connection matmul precision must be one of {MATMUL_PRECISIONS}, got {precision!r}")
    return precision


@contextlib.contextmanager
def _tf32_scope(precision):
    """Allow TF32 for the fp32 matmuls inside, on this call only; restores the previous setting.

    The flag is process-global; it is set and restored around each group of matmuls in the same
    thread that launches them (forward, and backward in the autograd thread).
    """
    if precision != "tf32":
        yield
        return
    matmul = torch.backends.cuda.matmul
    prev = matmul.fp32_precision
    matmul.fp32_precision = "tf32"
    try:
        yield
    finally:
        matmul.fp32_precision = prev


def _bf16_mm(a, b):
    """a @ b with bf16 operands, fp32 accumulation and an fp32 result."""
    a, b = a.bfloat16(), b.bfloat16()
    if a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    # CPU has no mm.dtype: bf16 values are exact in fp32, so this is the same math
    return a.float() @ b.float()


def _mm(a, b, precision):
    """a @ b -> fp32. fp32/tf32: ``a @ b.float()`` (tf32 needs an enclosing ``_tf32_scope``)."""
    if precision == "bf16":
        return _bf16_mm(a, b)
    return a @ b.float()


def _linear(x, w, precision):
    """F.linear(x, w) -> fp32. fp32/tf32: ``F.linear(x, w.float())``, today's fp32 path."""
    if precision == "bf16":
        return _bf16_mm(x, w.t())
    return F.linear(x, w.float())


def _gate_chain_fwd(normed, w_down, w_up, n, precision="fp32"):
    """fp32 gate = sigmoid(Wup silu(Wdown N / n)). Returns (gate, z1) fp32."""
    z1 = _linear(normed, w_down, precision) / n
    gate = torch.sigmoid(_linear(F.silu(z1), w_up, precision))
    return gate, z1


class _HCMixInject(torch.autograd.Function):
    """Fused mix + inject gate."""

    @staticmethod
    def forward(ctx, x2d, weight, w_down, w_up, w_inject, n, eps, precision="fp32"):
        T, W = x2d.shape
        C = W // n
        normed, rstd = grouped_rmsnorm_fwd(x2d, weight, n, eps)
        mm_in = normed.bfloat16() if precision == "bf16" else normed  # one rounding, shared by both matmuls
        with _tf32_scope(precision):
            gate, _ = _gate_chain_fwd(mm_in, w_down, w_up, n, precision)
            h_post = None
            if w_inject is not None:
                h_post = 2.0 * torch.sigmoid(_linear(mm_in, w_inject, precision) / n)
        del mm_in
        mixed = torch.empty(T, C, dtype=x2d.dtype, device=x2d.device)
        if T > 0:
            _gate_mul_mean_fwd_kernel[(T,)](gate, normed, mixed, T, N=n, C=C, BLOCK_C=block_c(C))
        if h_post is None:
            h_post = x2d.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(x2d, weight, w_down, w_up, w_inject, rstd)
        ctx.n, ctx.eps, ctx.precision = n, eps, precision
        return mixed, h_post

    @staticmethod
    def backward(ctx, dmixed, dh_post):
        x2d, weight, w_down, w_up, w_inject, rstd = ctx.saved_tensors
        n, eps, precision = ctx.n, ctx.eps, ctx.precision
        with _tf32_scope(precision):
            return _HCMixInject._backward(
                x2d, weight, w_down, w_up, w_inject, rstd, n, eps, precision, dmixed, dh_post
            )

    @staticmethod
    def _backward(x2d, weight, w_down, w_up, w_inject, rstd, n, eps, precision, dmixed, dh_post):
        T, W = x2d.shape
        C = W // n
        normed, _ = grouped_rmsnorm_fwd(x2d, weight, n, eps)
        mm_in = normed.bfloat16() if precision == "bf16" else normed
        gate, z1 = _gate_chain_fwd(mm_in, w_down, w_up, n, precision)
        s1 = F.silu(z1)

        dmixed = dmixed.float()
        dgate = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        dnormed = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        if T > 0:
            _gate_mul_mean_bwd_kernel[(T,)](dmixed, gate, normed, dgate, dnormed, T, N=n, C=C, BLOCK_C=block_c(C))

        dz2 = dgate * gate * (1.0 - gate)
        if precision == "bf16":
            dz2 = dz2.bfloat16()  # one rounding for its two matmuls (fp32 elementwise math above)
        dw_up = _mm(dz2.t(), s1, precision)
        ds1 = _mm(dz2, w_up, precision)
        sig_z1 = torch.sigmoid(z1)
        dz1 = ds1 * sig_z1 * (1.0 + z1 * (1.0 - sig_z1))
        dz1 = dz1 / n  # z1 = (N @ w_down^T) / n
        dw_down = _mm(dz1.t(), mm_in, precision)
        dnormed += _mm(dz1, w_down, precision)

        dw_inject = None
        if w_inject is not None and dh_post is not None and dh_post.numel() > 0:
            u = _linear(mm_in, w_inject, precision) / n
            sig_u = torch.sigmoid(u)
            du = dh_post.float() * 2.0 * sig_u * (1.0 - sig_u)
            du = du / n
            dw_inject = _mm(du.t(), mm_in, precision).to(w_inject.dtype)
            dnormed += _mm(du, w_inject, precision)

        x_hat = (x2d.float().view(T, n, C) * rstd.unsqueeze(-1)).view(T, W)
        dweight = (dnormed * x_hat).sum(dim=0).to(weight.dtype)

        dx = torch.empty_like(x2d)
        if T > 0:
            grouped_rmsnorm_bwd_kernel[(T * n,)](x2d, weight, rstd, dnormed, dx, T, N=n, C=C, BLOCK_C=block_c(C))
        return (
            dx,
            dweight,
            dw_down.to(w_down.dtype),
            dw_up.to(w_up.dtype),
            dw_inject,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------------------------------------------
# Fused path (``--qwen38-hc-fused``, default off): the same math with the elementwise chains fused into 4 per-token
# Triton kernels and 2 small [T, R] kernels. The GEMMs are the same ``_linear`` / ``_mm`` calls with the same
# precision. Differences from the default path (numerics only, no parameter or checkpoint change):
#
# - the inject projection ``u = normed @ w_inject^T`` (n columns) is an fp32 dot product inside the norm kernel
#   (the operand rounded to bf16 first in bf16 mode) instead of an fp32 / TF32 GEMM: TF32 mode gets the fp32 value;
# - ``du @ w_inject`` (K = n) is n fp32 FMAs inside the norm-backward kernel instead of a GEMM;
# - dweight is a sum of per-token-block partial sums (fixed order for a given T) instead of torch's column sum;
# - sigmoid / silu use Triton's exp, and some multiply-adds may contract into FMAs.
#
# The gate (sigmoid of the up projection), its derivative chain, the gated mean and the rmsnorm backward are
# recomputed from x, rstd and the pre-sigmoid up projection instead of being materialized in fp32: the backward
# reads and writes about 56 [T, n*C] fp32-equivalents instead of about 160, the forward about 20 instead of 34.
# ---------------------------------------------------------------------------------------------------------------

_FUSED_NUM_WARPS = 8
_FUSED_SMALL_BLOCK = 1024


@triton.jit(do_not_specialize=["T"])
def _fused_norm_inject_kernel(
    x_ptr,  # x dtype [T, n*C]
    w_ptr,  # norm weight [n*C]
    winj_ptr,  # inject weight [n, n*C] (HAS_INJECT)
    mm_ptr,  # out, matmul operand dtype [T, n*C]: the normed activations
    rstd_ptr,  # fp32 [T, n]: written (LOAD_RSTD False) or read (LOAD_RSTD True)
    hpost_ptr,  # fp32 out [T, n] (HAS_INJECT)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    LOAD_RSTD: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    js = tl.arange(0, BLOCK_N)
    jmask = js < N
    u = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        if LOAD_RSTD:
            rstd = tl.load(rstd_ptr + t * N + c)
        else:
            var = tl.sum(x * x, axis=0) / C
            rstd = 1.0 / tl.sqrt(var + EPS)
            tl.store(rstd_ptr + t * N + c, rstd)
        w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        normed = x * rstd * (1.0 + w)
        tl.store(mm_ptr + base + offs, normed.to(mm_ptr.dtype.element_ty), mask=mask)
        if HAS_INJECT:
            if ROUND_BF16:
                normed = normed.to(tl.bfloat16).to(tl.float32)
            wj = tl.load(
                winj_ptr + js[:, None] * (N * C) + c * C + offs[None, :],
                mask=jmask[:, None] & mask[None, :],
                other=0.0,
            )
            if ROUND_BF16:
                wj = wj.to(tl.bfloat16)  # bf16 operands, as _bf16_mm (a no-op for the bf16 HC weights)
            u += tl.sum(wj.to(tl.float32) * normed[None, :], axis=1)
    if HAS_INJECT:
        tl.store(hpost_ptr + t * N + js, 2.0 * tl.sigmoid(u / N), mask=jmask)


@triton.jit(do_not_specialize=["numel"])
def _fused_silu_kernel(
    a_ptr,  # fp32 [T, R]: normed @ w_down^T (before the 1/n)
    s_ptr,  # out, matmul operand dtype [T, R]: silu(a / n)
    numel,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    z = tl.load(a_ptr + offs, mask=mask, other=0.0) / N
    tl.store(s_ptr + offs, (z / (1.0 + tl.exp(-z))).to(s_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["numel"])
def _fused_silu_bwd_kernel(
    ds_ptr,  # fp32 [T, R]: d silu
    a_ptr,  # fp32 [T, R]: normed @ w_down^T (before the 1/n)
    dz_ptr,  # out, matmul operand dtype [T, R]: d a (the 1/n applied)
    numel,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    ds = tl.load(ds_ptr + offs, mask=mask, other=0.0)
    z = tl.load(a_ptr + offs, mask=mask, other=0.0) / N
    sig = 1.0 / (1.0 + tl.exp(-z))
    dz = ds * sig * (1.0 + z * (1.0 - sig))
    dz = dz / N
    tl.store(dz_ptr + offs, dz.to(dz_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _fused_gate_mean_kernel(
    z2_ptr,  # fp32 [T, n*C]: the up projection before the sigmoid
    x_ptr,
    w_ptr,
    rstd_ptr,  # fp32 [T, n]
    out_ptr,  # x dtype [T, C]
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + t * N + c)
        w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        normed = x * rstd * (1.0 + w)
        g = tl.sigmoid(tl.load(z2_ptr + base + offs, mask=mask, other=0.0))
        acc += g * normed
    acc = acc / N
    tl.store(out_ptr + t * C + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["T"])
def _fused_mix_bwd_kernel(
    z2_ptr,  # fp32 [T, n*C]: the up projection before the sigmoid
    x_ptr,
    w_ptr,
    rstd_ptr,  # fp32 [T, n]
    dmixed_ptr,  # [T, C] (cast on load)
    winj_ptr,  # [n, n*C] (HAS_INJECT)
    dhpost_ptr,  # fp32 [T, n] (HAS_INJECT)
    dz2_ptr,  # out, matmul operand dtype [T, n*C]: d of the up projection
    du_ptr,  # fp32 out [T, n] (HAS_INJECT): d of the inject projection (the 1/n applied)
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    if t >= T:
        return
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    js = tl.arange(0, BLOCK_N)
    jmask = js < N
    dm = tl.load(dmixed_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32) / N
    u = tl.zeros([BLOCK_N], dtype=tl.float32)
    for c in tl.static_range(N):
        base = t * (N * C) + c * C
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + t * N + c)
        w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
        normed = x * rstd * (1.0 + w)
        g = tl.sigmoid(tl.load(z2_ptr + base + offs, mask=mask, other=0.0))
        dgate = dm * normed
        dz2 = dgate * g * (1.0 - g)
        tl.store(dz2_ptr + base + offs, dz2.to(dz2_ptr.dtype.element_ty), mask=mask)
        if HAS_INJECT:
            if ROUND_BF16:
                normed = normed.to(tl.bfloat16).to(tl.float32)
            wj = tl.load(
                winj_ptr + js[:, None] * (N * C) + c * C + offs[None, :],
                mask=jmask[:, None] & mask[None, :],
                other=0.0,
            )
            if ROUND_BF16:
                wj = wj.to(tl.bfloat16)  # bf16 operands, as _bf16_mm (a no-op for the bf16 HC weights)
            u += tl.sum(wj.to(tl.float32) * normed[None, :], axis=1)
    if HAS_INJECT:
        sig_u = tl.sigmoid(u / N)
        dhp = tl.load(dhpost_ptr + t * N + js, mask=jmask, other=0.0)
        du = dhp * 2.0 * sig_u * (1.0 - sig_u)
        du = du / N
        tl.store(du_ptr + t * N + js, du, mask=jmask)


@triton.jit(do_not_specialize=["T"])
def _fused_norm_bwd_kernel(
    z2_ptr,  # fp32 [T, n*C]: the up projection before the sigmoid
    p_ptr,  # fp32 [T, n*C]: dz1 @ w_down (the low-rank path's d normed)
    x_ptr,
    w_ptr,
    rstd_ptr,  # fp32 [T, n]
    dmixed_ptr,  # [T, C] (cast on load)
    winj_ptr,  # [n, n*C] (HAS_INJECT)
    du_ptr,  # fp32 [T, n] (HAS_INJECT)
    dx_ptr,  # out, x dtype [T, n*C]
    dw_part_ptr,  # fp32 out [cdiv(T, BLOCK_T), n*C]: per-token-block partial sums of the norm weight grad
    T,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    tb = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    w = tl.load(w_ptr + c * C + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_INJECT:
        js = tl.arange(0, BLOCK_N)
        jmask = js < N
        wj = tl.load(
            winj_ptr + js[:, None] * (N * C) + c * C + offs[None, :],
            mask=jmask[:, None] & mask[None, :],
            other=0.0,
        )
        if ROUND_BF16:
            wj = wj.to(tl.bfloat16)  # bf16 operands, as _bf16_mm (a no-op for the bf16 HC weights)
        # kept in the weight dtype across the token loop (bf16 in training: half the registers), cast per use
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for i in range(BLOCK_T):
        t = tb * BLOCK_T + i
        if t < T:
            base = t * (N * C) + c * C
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + t * N + c)
            dm = tl.load(dmixed_ptr + t * C + offs, mask=mask, other=0.0).to(tl.float32) / N
            g = tl.sigmoid(tl.load(z2_ptr + base + offs, mask=mask, other=0.0))
            dn = dm * g
            dn = dn + tl.load(p_ptr + base + offs, mask=mask, other=0.0)
            if HAS_INJECT:
                du = tl.load(du_ptr + t * N + js, mask=jmask, other=0.0)
                if ROUND_BF16:
                    du = du.to(tl.bfloat16).to(tl.float32)
                dn = dn + tl.sum(du[:, None] * wj.to(tl.float32), axis=0)
            acc += dn * (x * rstd)
            gg = dn * (1.0 + w)
            dot = tl.sum(gg * x, axis=0)
            dx = rstd * gg - x * (rstd * rstd * rstd) * (dot / C)
            tl.store(dx_ptr + base + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.store(dw_part_ptr + tb * (N * C) + c * C + offs, acc, mask=mask)


def _bwd_block_t(T: int, n: int) -> int:
    """Tokens per norm-backward program: the largest of 32..1 that still gives >= 1024 programs (a function of T and
    n only, so dweight's summation order is the same on every device and run)."""
    for bt in (32, 16, 8, 4, 2):
        if triton.cdiv(T, bt) * n >= 1024:
            return bt
    return 1


def _small_grid(numel):
    return (triton.cdiv(numel, _FUSED_SMALL_BLOCK),)


class _HCMixInjectFused(torch.autograd.Function):
    """``_HCMixInject`` with the elementwise chains fused (``--qwen38-hc-fused``); see the block comment above."""

    @staticmethod
    def forward(ctx, x2d, weight, w_down, w_up, w_inject, n, eps, precision="fp32"):
        T, W = x2d.shape
        C = W // n
        dev = x2d.device
        bf16 = precision == "bf16"
        has_inject = w_inject is not None
        mm_in = torch.empty(T, W, dtype=torch.bfloat16 if bf16 else torch.float32, device=dev)
        rstd = torch.empty(T, n, dtype=torch.float32, device=dev)
        h_post = torch.empty(T, n, dtype=torch.float32, device=dev) if has_inject else None
        if T > 0:
            _fused_norm_inject_kernel[(T,)](
                x2d, weight, w_inject if has_inject else weight, mm_in, rstd, h_post if has_inject else rstd, T,
                N=n, C=C, EPS=eps, BLOCK_C=block_c(C), BLOCK_N=triton.next_power_of_2(n),
                HAS_INJECT=has_inject, ROUND_BF16=bf16, LOAD_RSTD=False, num_warps=_FUSED_NUM_WARPS,
            )  # fmt: skip
        with _tf32_scope(precision):
            a = _linear(mm_in, w_down, precision)
            del mm_in
            s1 = torch.empty(a.shape, dtype=torch.bfloat16 if bf16 else torch.float32, device=dev)
            if a.numel() > 0:
                _fused_silu_kernel[_small_grid(a.numel())](a, s1, a.numel(), N=n, BLOCK=_FUSED_SMALL_BLOCK)
            del a
            z2 = _linear(s1, w_up, precision)
        del s1
        mixed = torch.empty(T, C, dtype=x2d.dtype, device=dev)
        if T > 0:
            _fused_gate_mean_kernel[(T,)](
                z2, x2d, weight, rstd, mixed, T, N=n, C=C, BLOCK_C=block_c(C), num_warps=_FUSED_NUM_WARPS
            )
        if h_post is None:
            h_post = x2d.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(x2d, weight, w_down, w_up, w_inject, rstd)
        ctx.n, ctx.eps, ctx.precision = n, eps, precision
        return mixed, h_post

    @staticmethod
    def backward(ctx, dmixed, dh_post):
        x2d, weight, w_down, w_up, w_inject, rstd = ctx.saved_tensors
        n, eps, precision = ctx.n, ctx.eps, ctx.precision
        T, W = x2d.shape
        C = W // n
        dev = x2d.device
        bf16 = precision == "bf16"
        mm_dtype = torch.bfloat16 if bf16 else torch.float32
        has_inject = w_inject is not None and dh_post is not None and dh_post.numel() > 0
        bn = triton.next_power_of_2(n)
        dmixed = dmixed.contiguous()
        dh_post = dh_post.float().contiguous() if has_inject else None

        mm_in = torch.empty(T, W, dtype=mm_dtype, device=dev)
        if T > 0:
            _fused_norm_inject_kernel[(T,)](
                x2d, weight, weight, mm_in, rstd, rstd, T, N=n, C=C, EPS=eps, BLOCK_C=block_c(C), BLOCK_N=bn,
                HAS_INJECT=False, ROUND_BF16=bf16, LOAD_RSTD=True, num_warps=_FUSED_NUM_WARPS,
            )  # fmt: skip
        with _tf32_scope(precision):
            a = _linear(mm_in, w_down, precision)
            s1 = torch.empty(a.shape, dtype=mm_dtype, device=dev)
            if a.numel() > 0:
                _fused_silu_kernel[_small_grid(a.numel())](a, s1, a.numel(), N=n, BLOCK=_FUSED_SMALL_BLOCK)
            z2 = _linear(s1, w_up, precision)

            dz2 = torch.empty(T, W, dtype=mm_dtype, device=dev)
            du = torch.empty(T, n, dtype=torch.float32, device=dev) if has_inject else None
            if T > 0:
                _fused_mix_bwd_kernel[(T,)](
                    z2, x2d, weight, rstd, dmixed, w_inject if has_inject else weight,
                    dh_post if has_inject else rstd, dz2, du if has_inject else rstd, T,
                    N=n, C=C, BLOCK_C=block_c(C), BLOCK_N=bn, HAS_INJECT=has_inject, ROUND_BF16=bf16,
                    num_warps=_FUSED_NUM_WARPS,
                )  # fmt: skip
            dw_up = _mm(dz2.t(), s1, precision)
            ds1 = _mm(dz2, w_up, precision)
            del dz2, s1
            dz1 = torch.empty(a.shape, dtype=mm_dtype, device=dev)
            if a.numel() > 0:
                _fused_silu_bwd_kernel[_small_grid(a.numel())](ds1, a, dz1, a.numel(), N=n, BLOCK=_FUSED_SMALL_BLOCK)
            del ds1, a
            dw_down = _mm(dz1.t(), mm_in, precision)
            p = _mm(dz1, w_down, precision)
            del dz1
            dw_inject = _mm(du.t(), mm_in, precision).to(w_inject.dtype) if has_inject else None
        del mm_in

        dx = torch.empty_like(x2d)
        bt = _bwd_block_t(T, n)
        dw_part = torch.empty(triton.cdiv(T, bt), W, dtype=torch.float32, device=dev)
        if T > 0:
            _fused_norm_bwd_kernel[(triton.cdiv(T, bt), n)](
                z2, p, x2d, weight, rstd, dmixed, w_inject if has_inject else weight, du if has_inject else rstd,
                dx, dw_part, T, N=n, C=C, BLOCK_C=block_c(C), BLOCK_N=bn, BLOCK_T=bt, HAS_INJECT=has_inject,
                ROUND_BF16=bf16, num_warps=_FUSED_NUM_WARPS,
            )  # fmt: skip
        dweight = dw_part.sum(dim=0).to(weight.dtype)
        return (
            dx,
            dweight,
            dw_down.to(w_down.dtype),
            dw_up.to(w_up.dtype),
            dw_inject,
            None,
            None,
            None,
        )


class _HCCombine(torch.autograd.Function):
    """out = residual + h_post[:, c, None] * y  (flattened [T, n*C])."""

    @staticmethod
    def forward(ctx, residual2d, y2d, h_post, n):
        T, W = residual2d.shape
        C = W // n
        ctx.h_post_dtype = h_post.dtype
        h_post = h_post.float().contiguous()
        out = torch.empty_like(residual2d)
        if T > 0:
            _combine_fwd_kernel[(T * n,)](residual2d, y2d, h_post, out, T, N=n, C=C, BLOCK_C=block_c(C))
        ctx.save_for_backward(y2d, h_post)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, dout):
        y2d, h_post = ctx.saved_tensors
        n = ctx.n
        T, W = dout.shape
        C = W // n
        dout = dout.contiguous()
        dy = torch.empty_like(y2d)
        dh_post = torch.empty(T, n, dtype=torch.float32, device=dout.device)
        if T > 0:
            _combine_bwd_kernel[(T,)](dout, y2d, h_post, dy, dh_post, T, N=n, C=C, BLOCK_C=block_c(C))
        return dout, dy, dh_post.to(ctx.h_post_dtype), None


def hc_mix_inject(
    x, weight, w_down, w_up, w_inject, n: int, eps: float, matmul_precision: str = "fp32", fused: bool = False
):
    """Fused (grouped_gemma_rmsnorm -> hc_mix, hc_inject_gate); ``matmul_precision``: see the module doc.

    ``fused`` (``--qwen38-hc-fused``): the elementwise chains in fewer kernels (``_HCMixInjectFused``); same math,
    different fp32 rounding order; the default path is unchanged."""
    check_matmul_precision(matmul_precision)
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    fn = _HCMixInjectFused if fused else _HCMixInject
    mixed, h_post = fn.apply(x2d, weight, w_down, w_up, w_inject, n, eps, matmul_precision)
    mixed = mixed.reshape(*lead, -1)
    if w_inject is not None:
        h_post = h_post.reshape(*lead, n)
    return mixed, h_post


def hc_combine(residual, block_output, h_post, n: int):
    """``X'_c = X_c + a_c * y`` -- fused elementwise."""
    lead = residual.shape[:-1]
    r2d = residual.reshape(-1, residual.shape[-1]).contiguous()
    y2d = block_output.reshape(-1, block_output.shape[-1]).contiguous()
    hp2d = h_post.reshape(-1, n)
    out = _HCCombine.apply(r2d, y2d, hp2d, n)
    return out.reshape(*lead, -1)
