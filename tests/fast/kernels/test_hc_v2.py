"""CPU test: ``hc_mix_inject(..., v2=True)`` (``--qwen38-hc-kernel-v2``) against the default path and fp64.

v2 runs the gate chain's GEMMs as Triton kernels with the rmsnorm / silu / sigmoid / gated-mean / rmsnorm-backward
math in their prologues and epilogues (``miles/kernels/hyper_connection/hc_v2.py``). It is a numerics change (fp32
accumulation order of every GEMM, the inject projection as an fp32 dot product, dweight from per-token-block partial
sums), so the checks are tolerances, plus three exact properties:

* every output and gradient is close to the default path (relative Frobenius difference, cosine), and its error
  against an fp64 reference is at most a small factor above the default path's;
* two v2 runs are bitwise equal (no atomics, fixed tiles);
* a token's forward outputs (mixed, h_post) are bitwise the same whatever the other rows are and whatever T is
  (fixed tiles, no split-K: the shape dependence of the default path's cuBLAS inject GEMM cannot occur);
* strided weights give the same bits as their contiguous copies.

The Triton kernels run under ``TRITON_INTERPRET=1`` in a subprocess, with small tiles (``MILES_HCV2_CONFIG``) so the
shapes below cover several tiles in every dimension and ragged last tiles (T, C, R, n*C, M = R + n). fp32 -> bf16 casts
are patched to round to nearest even, as on the GPU (same patch as ``test_hc_fused.py``).
GPU numerics and speed: ``tests/fast-gpu/kernels/hyper_connection/test_hc_v2_gpu.py``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=300, suite="stage-b-cpu", labels=["miles-plugin"])

ROOT = Path(__file__).resolve().parents[3]

SMALL_TILES = {
    "stats": {"BT": 4, "num_warps": 4},
    "down": {"BM": 16, "BN": 16, "BK": 16, "num_warps": 4, "num_stages": 1},
    "up": {"BM": 16, "BN": 16, "BK": 16, "num_warps": 4, "num_stages": 1},
    "wgrad": {"BM": 16, "BN": 16, "BK": 16, "num_warps": 4, "num_stages": 1},
    "dx": {"BM": 16, "BN": 16, "BK": 16, "num_warps": 4, "num_stages": 1},
    "small": {"BT": 8, "num_warps": 4},
}

# (T, C, n, R, x dtype, matmul precision, with inject)
CASES = [
    (37, 32, 4, 8, "bfloat16", "fp32", True),
    (37, 32, 4, 8, "bfloat16", "tf32", True),
    (37, 32, 4, 8, "bfloat16", "bf16", True),
    (37, 40, 4, 20, "bfloat16", "fp32", True),  # C 40: ragged column tile; R 20: ragged K / N tiles; M 24
    (21, 24, 3, 6, "float32", "fp32", True),  # n = 3 (masked stream lane), fp32 activations
    (21, 24, 3, 6, "float32", "bf16", True),
    (37, 32, 4, 8, "bfloat16", "fp32", False),  # head contraction (no inject)
    (37, 40, 4, 20, "bfloat16", "bf16", False),
    (1, 16, 4, 4, "bfloat16", "fp32", True),
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
    w_up = (2.0 * torch.randn(W, R, generator=g) / R**0.5).to(dtype)
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
    for i, o in enumerate(out):
        assert o is not None and torch.isfinite(o).all(), f"output / grad {i} is not finite"
    return out


def _reference(n, round_operands):
    import torch
    import torch.nn.functional as F

    rnd = (lambda t: t.bfloat16().double()) if round_operands else (lambda t: t)

    def fn(x, weight, w_down, w_up, w_inj):
        xg = x.unflatten(-1, (n, -1))
        normed = (xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + EPS)).flatten(-2) * (1.0 + weight)
        # bf16 mode rounds both GEMM operands, weights included (a no-op for bf16 weights; fp32 weights round too)
        z1 = F.linear(rnd(normed), rnd(w_down)) / n
        gate = torch.sigmoid(F.linear(rnd(F.silu(z1)), rnd(w_up)))
        mixed = (gate * normed).unflatten(-1, (n, -1)).mean(dim=1)
        h_post = 2.0 * torch.sigmoid(F.linear(rnd(normed), rnd(w_inj)) / n) if w_inj is not None else None
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


def _paths(hc, n, precision):
    def default(*a):
        return hc.hc_mix_inject(*a, n, EPS, matmul_precision=precision)

    def v2(*a):
        return hc.hc_mix_inject(*a, n, EPS, matmul_precision=precision, v2=True)

    return default, v2


def _worker():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import torch

    from miles.kernels.hyper_connection import hc

    _interpreter_bf16_rtne()
    for case_id, (T, C, n, R, dtype_name, precision, with_inject) in enumerate(CASES):
        tensors, dmixed, dh_post = _inputs(T, C, n, R, getattr(torch, dtype_name), seed=case_id)
        default, v2 = _paths(hc, n, precision)
        want = _run(default, tensors, dmixed, dh_post, with_inject)
        got = _run(v2, tensors, dmixed, dh_post, with_inject)
        again = _run(v2, tensors, dmixed, dh_post, with_inject)
        ref = _run(
            _reference(n, precision == "bf16"),
            [t.double() for t in tensors],
            dmixed.double(),
            dh_post.double(),
            with_inject,
        )
        names = NAMES if with_inject else [m for m in NAMES if m not in ("h_post", "dw_inject")]
        assert len(got) == len(want) == len(ref) == len(names)
        for name, g, w, a, r in zip(names, got, want, again, ref, strict=True):
            assert g.shape == w.shape and g.dtype == w.dtype, (case_id, name, g.shape, w.shape, g.dtype, w.dtype)
            assert torch.equal(g, a), (case_id, name, "two v2 runs differ")
            diff, cos = _rel(g, w), _cos(g, w)
            err_v, err_d = _rel(g, r), _rel(w, r)
            print(
                f"case {case_id} {precision} inject={with_inject} {name}: v2-vs-default rel {diff:.2e} cos "
                f"{cos:.8f}; vs fp64 v2 {err_v:.2e} default {err_d:.2e}",
                flush=True,
            )
            low_precision = g.dtype != torch.float32 or precision == "bf16"
            assert diff <= (1e-2 if low_precision else 2e-5), (case_id, name, diff)
            assert cos >= (0.9999 if low_precision else 0.9999999), (case_id, name, cos)
            assert err_v <= 1.5 * err_d + 2e-6, (case_id, name, err_v, err_d)
    _check_length_invariance(hc, torch)
    _check_strided(hc, torch)
    _check_empty(hc, torch)
    print("ALL-OK", flush=True)


def _check_length_invariance(hc, torch):
    """Rows 3..10 of a 40-row batch give the same forward bits as the same rows alone (T 8), at offset 0 of a 19-row
    batch with other contents, and at offset 25 of a 61-row batch."""
    for precision in ("fp32", "tf32", "bf16"):
        C, n, R = 40, 4, 20
        base, _, _ = _inputs(40, C, n, R, torch.bfloat16, seed=7)
        rows = base[0][3:11]
        weights = base[1:]
        outs = []
        for T, off, seed in ((8, 0, 0), (40, 3, None), (19, 0, 11), (61, 25, 12)):
            if seed is None:
                x = base[0]
            else:
                x, _, _ = _inputs(T, C, n, R, torch.bfloat16, seed=seed)
                x = x[0].clone()
                x[off : off + 8] = rows
            with torch.no_grad():
                mixed, h_post = hc.hc_mix_inject(x, *weights, n, EPS, matmul_precision=precision, v2=True)
            outs.append((mixed[off : off + 8].clone(), h_post[off : off + 8].clone()))
        for i, (m, h) in enumerate(outs[1:], start=1):
            assert torch.equal(m, outs[0][0]), (precision, i, "mixed depends on T / row position")
            assert torch.equal(h, outs[0][1]), (precision, i, "h_post depends on T / row position")
    print("length-invariance-OK", flush=True)


def _check_strided(hc, torch):
    """Strided norm / down / up / inject weights give the bits of their contiguous copies, and the gradients land on
    the strided leaves with the parameters' shapes."""
    for case_id, (T, C, n, R, dtype_name, precision) in enumerate(
        [(7, 13, 3, 5, "float32", "fp32"), (37, 40, 4, 20, "bfloat16", "tf32"), (21, 24, 3, 6, "bfloat16", "bf16")]
    ):
        tensors, dmixed, dh_post = _inputs(T, C, n, R, getattr(torch, dtype_name), seed=50 + case_id)
        strided = list(tensors)
        strided[1] = torch.stack([tensors[1], tensors[1]], dim=1)[:, 0]  # stride 2
        for i in (2, 3, 4):
            strided[i] = tensors[i].t().contiguous().t()
            assert not strided[i].clone().is_contiguous()
        _, v2 = _paths(hc, n, precision)
        got = _run(v2, strided, dmixed, dh_post, True)
        ref = _run(v2, tensors, dmixed, dh_post, True)
        for name, g, r in zip(NAMES, got, ref, strict=True):
            assert g.shape == r.shape and g.dtype == r.dtype, (case_id, name)
            assert torch.equal(g, r), (case_id, name, "strided and contiguous weights differ")
        # clone() makes a 1-D stride-2 view dense, so feed the norm weight as a view of a [W, 2] leaf
        base = torch.stack([tensors[1], tensors[1]], dim=1).clone().requires_grad_(True)
        leaves = [t.clone().requires_grad_(True) for t in tensors]
        mixed, h_post = v2(leaves[0], base[:, 0], *leaves[2:])
        assert not base[:, 0].is_contiguous()
        ((mixed.double() * dmixed.double()).sum() + (h_post.double() * dh_post.double()).sum()).backward()
        assert torch.equal(mixed.detach(), ref[0]) and torch.equal(base.grad[:, 0], ref[3]), (case_id, "stride-2 norm weight")
        assert not base.grad[:, 1].any()
    # the direct helper (review F6): a column-major x gives the bits of its contiguous copy
    from miles.kernels.hyper_connection import hc_v2

    tensors, _, _ = _inputs(17, 16, 4, 8, torch.float32, seed=60)
    x_cm = tensors[0].t().contiguous().t()
    assert not x_cm.is_contiguous()
    for precision in ("fp32", "bf16"):
        with torch.no_grad():
            a = hc_v2.hc_mix_inject_v2(x_cm, *tensors[1:], 4, EPS, precision)
            b = hc_v2.hc_mix_inject_v2(tensors[0], *tensors[1:], 4, EPS, precision)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), (precision, "helper ignores x strides")
    print("strided-OK", flush=True)


def _check_empty(hc, torch):
    """T = 0 on the 2-D autograd function: empty outputs, zero weight gradients, same shapes as the default path."""
    from miles.kernels.hyper_connection import hc_v2

    for precision in ("fp32", "bf16"):
        tensors, _, _ = _inputs(0, 16, 4, 4, torch.bfloat16, seed=99)
        res = []
        for fn in (hc._HCMixInject, hc_v2._HCMixInjectV2):
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


def test_hc_v2_matches_default_and_fp64():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", MILES_HCV2_CONFIG=json.dumps(SMALL_TILES))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    proc = subprocess.run(
        [sys.executable, "-B", "-c", f"import runpy; runpy.run_path({str(Path(__file__))!r}, run_name='__worker__')"],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    sys.stdout.write(proc.stdout)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-8000:]
    assert "ALL-OK" in proc.stdout


def test_default_config_is_shape_independent():
    """The production tiles are constants: no entry depends on T (the length-invariance property relies on it)."""
    import pytest

    hc_v2 = pytest.importorskip("miles.kernels.hyper_connection.hc_v2")
    os.environ.pop("MILES_HCV2_CONFIG", None)
    cfg = hc_v2._config()
    assert cfg == hc_v2._DEFAULT_CONFIG
    for key in ("down", "up", "wgrad", "dx"):
        for dim in ("BM", "BN", "BK"):
            v = cfg[key][dim]
            assert v >= 16 and v & (v - 1) == 0, (key, dim, v)


if __name__ == "__worker__":
    _worker()
