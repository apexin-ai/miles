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


def hc_mix_inject(x, weight, w_down, w_up, w_inject, n: int, eps: float, matmul_precision: str = "fp32"):
    """Fused (grouped_gemma_rmsnorm -> hc_mix, hc_inject_gate); ``matmul_precision``: see the module doc."""
    check_matmul_precision(matmul_precision)
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    mixed, h_post = _HCMixInject.apply(x2d, weight, w_down, w_up, w_inject, n, eps, matmul_precision)
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
