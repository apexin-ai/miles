"""CPU test: ``hc_mix_inject(..., fused=True)`` (``--qwen38-hc-fused``) against the default path and fp64.

The fused path recomputes the gate chain's elementwise math inside 4 per-token Triton kernels (plus 2 small [T, R]
kernels) instead of ~25 separate fp32 torch kernels; the GEMMs are the same calls. It is a numerics change (fp32
rounding order, the n-column inject projection as an fp32 dot product, dweight from per-token-block partial sums),
so the checks are tolerances, not bitwise equality:

* every output and gradient is close to the default path (relative Frobenius difference, cosine);
* against an fp64 reference (operands rounded to bf16 where bf16 mode rounds them), the fused error is at most a
  small factor above the default path's error, i.e. the fusion does not make numerics materially worse;
* two fused runs are bitwise equal (no atomics; dweight's block size depends on T and n only).

Both paths run their Triton kernels under ``TRITON_INTERPRET=1`` on CPU, in a subprocess because the flag must be
set before ``triton.jit`` decorates the kernels. The interpreter (Triton 3.8) truncates fp32 -> bf16 casts, while
the GPU backend (``cvt.rn.bf16.f32``) and torch round to nearest even; the worker patches the interpreter's cast to
torch's rounding so the bf16 cases model the GPU (``_interpreter_bf16_rtne``). Cases: fp32 / tf32 (= fp32 on CPU) / bf16 matmul precision, with
and without the inject weight (head contraction), C not a power of two, n = 3 (masked stream lanes), forced
norm-backward token blocks with a ragged last block, T = 1, and T = 0 on the 2-D autograd functions. GPU numerics and speed: the GPU script
tests/fast-gpu/kernels/hyper_connection/test_hc_fused_gpu.py.
"""

import os
import subprocess
import sys
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["miles-plugin"])

ROOT = Path(__file__).resolve().parents[3]

# (T, C, n, R, x dtype, matmul precision, with inject, forced norm-backward BLOCK_T or None)
CASES = [
    (37, 32, 4, 8, "bfloat16", "fp32", True, None),
    (37, 32, 4, 8, "bfloat16", "tf32", True, None),
    (37, 32, 4, 8, "bfloat16", "bf16", True, None),
    (37, 40, 4, 8, "bfloat16", "fp32", True, 8),  # C = 40 (BLOCK_C 64), 5 token blocks, ragged last one
    (21, 24, 3, 6, "float32", "fp32", True, 4),  # n = 3: BLOCK_N 4 with a masked lane; fp32 activations
    (21, 24, 3, 6, "float32", "bf16", True, 32),  # one partial token block
    (37, 32, 4, 8, "bfloat16", "fp32", False, None),  # head contraction (no inject)
    (37, 40, 4, 8, "bfloat16", "bf16", False, 8),
    (1, 16, 4, 4, "bfloat16", "fp32", True, None),
]
NAMES = ["mixed", "h_post", "dx", "dweight", "dw_down", "dw_up", "dw_inject"]
EPS = 1e-6


def _inputs(T, C, n, R, dtype, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    W = n * C
    x = torch.randn(T, W, generator=g).to(dtype)
    weight = (0.1 * torch.randn(W, generator=g)).to(dtype)
    w_down = (torch.randn(R, W, generator=g) / W**0.5).to(dtype)
    w_up = (2.0 * torch.randn(W, R, generator=g) / R**0.5).to(dtype)  # gates away from 0.5
    w_inj = (torch.randn(n, W, generator=g) / W**0.5).to(dtype)
    dmixed = torch.randn(T, C, generator=g).to(dtype)
    dh_post = torch.randn(T, n, generator=g)
    return [x, weight, w_down, w_up, w_inj], dmixed, dh_post


def _run(fn, tensors, dmixed, dh_post, with_inject):
    import torch

    leaves = [t.clone().requires_grad_(True) for t in tensors]
    if not with_inject:
        leaves[4] = None
    mixed, h_post = fn(*leaves)
    loss = (mixed.double() * dmixed.double()).sum()
    if with_inject:
        loss = loss + (h_post.double() * dh_post.double()).sum()
    loss.backward()
    out = [mixed.detach(), h_post.detach()] + [leaf.grad for leaf in leaves if leaf is not None]
    if not with_inject:
        out = out[:1] + out[2:]
    return [torch.nan_to_num(o) for o in out]


def _reference(n, round_operands):
    """fp64 hc_mix_inject; ``round_operands`` rounds each matmul's activation operand to bf16 like bf16 mode."""
    import torch
    import torch.nn.functional as F

    rnd = (lambda t: t.bfloat16().double()) if round_operands else (lambda t: t)

    def fn(x, weight, w_down, w_up, w_inj):
        xg = x.unflatten(-1, (n, -1))
        normed = (xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + EPS)).flatten(-2) * (1.0 + weight)
        z1 = F.linear(rnd(normed), w_down) / n
        gate = torch.sigmoid(F.linear(rnd(F.silu(z1)), w_up))
        mixed = (gate * normed).unflatten(-1, (n, -1)).mean(dim=1)
        h_post = 2.0 * torch.sigmoid(F.linear(rnd(normed), w_inj) / n) if w_inj is not None else None
        return mixed, h_post if h_post is not None else x.new_empty(0)

    return fn


def _rel(a, b):
    a, b = a.double(), b.double()
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()


def _cos(a, b):
    import torch

    a, b = a.double().flatten(), b.double().flatten()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _interpreter_bf16_rtne():
    """fp32 -> bf16 casts in the Triton interpreter: round to nearest even (torch), as on the GPU, not truncation."""
    import numpy as np
    import torch
    import triton.language as tl
    import triton.runtime.interpreter as interp

    builder = interp.InterpreterBuilder
    orig = builder.cast_impl

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            data = torch.from_numpy(np.ascontiguousarray(src.data, dtype=np.float32)).bfloat16()
            return interp.TensorHandle(data.view(torch.int16).numpy().view(np.uint16), dst_type.scalar)
        return orig(self, src, dst_type)

    builder.cast_impl = cast_impl
    x = torch.randn(4096)
    y = torch.empty(4096, dtype=torch.bfloat16)
    _cast_probe[(1,)](x, y, 4096, BLOCK=4096)
    assert torch.equal(y, x.bfloat16()), "interpreter bf16 cast patch did not take effect"


def _worker():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import torch

    from miles.kernels.hyper_connection import hc

    _interpreter_bf16_rtne()

    real_block_t = hc._bwd_block_t
    for case_id, (T, C, n, R, dtype_name, precision, with_inject, forced_bt) in enumerate(CASES):
        dtype = getattr(torch, dtype_name)
        hc._bwd_block_t = (lambda T_, n_, bt=forced_bt: bt) if forced_bt else real_block_t
        tensors, dmixed, dh_post = _inputs(T, C, n, R, dtype, seed=case_id)

        def default(*a):
            return hc.hc_mix_inject(*a, n, EPS, matmul_precision=precision)

        def fused(*a):
            return hc.hc_mix_inject(*a, n, EPS, matmul_precision=precision, fused=True)

        want = _run(default, tensors, dmixed, dh_post, with_inject)
        got = _run(fused, tensors, dmixed, dh_post, with_inject)
        again = _run(fused, tensors, dmixed, dh_post, with_inject)
        ref = _run(
            _reference(n, precision == "bf16"),
            [t.double() for t in tensors],
            dmixed.double(),
            dh_post.double(),
            with_inject,
        )
        names = NAMES if with_inject else [m for m in NAMES if m not in ("h_post", "dw_inject")]
        assert len(got) == len(want) == len(ref) == len(names), (len(got), len(want), len(ref))
        for name, g, w, a, r in zip(names, got, want, again, ref, strict=True):
            assert g.shape == w.shape and g.dtype == w.dtype, (case_id, name, g.shape, w.shape, g.dtype, w.dtype)
            assert torch.equal(g, a), (case_id, name, "two fused runs differ")
            diff, cos = _rel(g, w), _cos(g, w)
            err_f, err_d = _rel(g, r), _rel(w, r)
            print(
                f"case {case_id} {precision} inject={with_inject} {name}: fused-vs-default rel {diff:.2e} cos "
                f"{cos:.8f}; vs fp64 fused {err_f:.2e} default {err_d:.2e}",
                flush=True,
            )
            low_precision = g.dtype != torch.float32 or precision == "bf16"
            assert diff <= (1e-2 if low_precision else 2e-5), (case_id, name, diff)
            assert cos >= (0.9999 if low_precision else 0.9999999), (case_id, name, cos)
            assert err_f <= 1.5 * err_d + 2e-6, (case_id, name, err_f, err_d)
    hc._bwd_block_t = real_block_t
    _check_empty(hc, torch)
    print("ALL-OK", flush=True)


def _check_empty(hc, torch):
    """T = 0 (hc_mix_inject's own reshape refuses 0 rows on both paths): the 2-D autograd functions agree."""
    for precision in ("fp32", "bf16"):
        tensors, _, _ = _inputs(0, 16, 4, 4, torch.bfloat16, seed=99)
        res = []
        for fn in (hc._HCMixInject, hc._HCMixInjectFused):
            leaves = [t.clone().requires_grad_(True) for t in tensors]
            mixed, h_post = fn.apply(*leaves, 4, EPS, precision)
            (mixed.float().sum() + h_post.sum()).backward()
            res.append([mixed, h_post] + [leaf.grad for leaf in leaves])
        for i, (a, b) in enumerate(zip(*res, strict=True)):
            assert (a is None) == (b is None), (precision, i)
            if a is None:
                continue
            assert a.shape == b.shape and a.dtype == b.dtype, (precision, i, a.shape, b.shape)
            assert not b.any(), (precision, i)
    print("empty-OK", flush=True)


def test_hc_fused_matches_default_and_fp64():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    proc = subprocess.run(
        [sys.executable, "-B", "-c", f"import runpy; runpy.run_path({str(Path(__file__))!r}, run_name='__worker__')"],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    sys.stdout.write(proc.stdout)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    assert "ALL-OK" in proc.stdout


def test_bwd_block_t_depends_on_shape_only():
    import pytest

    hc = pytest.importorskip("miles.kernels.hyper_connection.hc")
    assert hc._bwd_block_t(65536, 4) == 32
    assert hc._bwd_block_t(1024, 4) == 4
    assert hc._bwd_block_t(1, 4) == 1
    for T in (1, 7, 300, 4096, 70000):
        bt = hc._bwd_block_t(T, 4)
        assert bt in (1, 2, 4, 8, 16, 32)
        assert bt == 1 or -(-T // bt) * 4 >= 1024


if __name__ == "__worker__":
    import triton
    import triton.language as tl

    @triton.jit
    def _cast_probe(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask).to(tl.bfloat16), mask=mask)

    _worker()
