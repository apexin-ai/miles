"""CPU test: ``ple_gate_conv(..., deterministic_conv_grad=True)`` computes the conv weight grad without atomics.

The default backward accumulates the conv weight grad ``[W, K]`` with fp32 ``tl.atomic_add`` over all tokens, so
its summation order (and its last bits) changes from run to run on a GPU. The deterministic path writes one
partial sum per (token block, width block) into a ``[num_token_blocks, W, KP]`` buffer and sums it over the token
blocks. This test runs the Triton kernels under ``TRITON_INTERPRET=1`` on CPU, in a subprocess because the
interpreter flag must be set before ``triton.jit`` decorates the kernels, and checks that:

* the output and the 6 other grads are bitwise equal to the default (atomic) path;
* the deterministic conv weight grad (fp32, before the cast onto the weight dtype) matches an fp64 reference
  computed from the same backward inputs, as closely as the atomic path does;
* two deterministic runs are bitwise equal.

Cases cover several token blocks with a ragged last one, a ragged last width block, segment starts inside a token
block and inside a ``BLOCK_T`` tile, T smaller than one tile, and bf16 / fp32. Bitwise run-to-run equality on a GPU
comes from the GPU script for the PERF executor (P5), not from this test.
"""

import os
import subprocess
import sys
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

# 6 cases x 3 backwards under the Triton interpreter; about 60 s on an 8-thread CPU
register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=["miles-plugin"])

ROOT = Path(__file__).resolve().parents[3]

# (T, C, n, segment bounds or None, dtype name)
CASES = [
    (5, 16, 4, None, "float32"),  # T < BLOCK_T: one partial tile
    (64, 64, 4, None, "bfloat16"),
    (257, 80, 4, [0, 100, 257], "bfloat16"),  # W = 320: the last 128-wide block is masked
    (600, 16, 4, [0, 3, 230, 231, 470, 600], "float32"),  # 3 token blocks of 224, ragged last one
    (1100, 16, 4, [0, 517, 1100], "bfloat16"),  # 5 token blocks
    (300, 40, 4, [0, 30, 64, 300], "float32"),  # C = 40, W = 160
]
NAMES = ["dhc", "dkey", "dvalue", "dwk", "dwq", "dwc", "dconvw"]


def _inputs(T, C, n, segs, dtype, seed):
    import torch

    g = torch.Generator().manual_seed(seed)
    W, K = n * C, 4

    def mk(*shape, scale=1.0):
        return (scale * torch.randn(*shape, generator=g)).to(dtype)

    hc = mk(T, W)
    key = mk(T, W)
    value = mk(T, C)
    wk, wq, wc = mk(W, scale=0.05), mk(W, scale=0.05), mk(W, scale=0.05)
    convw = mk(W, 1, K, scale=1.0 / K)
    cu = None if segs is None else torch.tensor(segs, dtype=torch.int32)
    dout = mk(T, W)
    return (hc, key, value, wk, wq, wc, convw), cu, dout


def _run(ple, inputs, cu, dout, deterministic):
    params = [p.clone().requires_grad_() for p in inputs]
    hc, key, value, wk, wq, wc, convw = params
    out = ple.ple_gate_conv(hc, key, value, wk, wq, wc, convw, 4, 1e-6, 3, cu, deterministic_conv_grad=deterministic)
    out.backward(dout)
    return out.detach(), [p.grad for p in params]


def _ref_dw(dout, conv_pre, normed, seg_lo, K, dil):
    """fp64 reference of the conv weight grad ``[W, K]`` from the backward's own inputs."""
    import torch

    do = dout.double()
    cv = conv_pre.double()
    sig = torch.sigmoid(cv)
    dconv = do * sig * (1.0 + cv * (1.0 - sig))
    x = normed.double()
    T, W = x.shape
    t = torch.arange(T)
    ref = torch.zeros(W, K, dtype=torch.float64)
    for j in range(K):
        src = t - (K - 1 - j) * dil
        ok = (src >= 0) & (src >= seg_lo.long())
        xs = torch.where(ok[:, None], x[src.clamp_min(0)], torch.zeros((), dtype=torch.float64))
        ref[:, j] = (dconv * xs).sum(0)
    return ref


def _worker():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import torch

    from miles.kernels.embedding import ple_gate_conv as ple

    real_dw = ple._ple_conv_dw_deterministic
    real_bwd = ple._ple_conv_bwd_kernel
    seen = {}

    def spy_dw(dout, conv_pre, normed, seg_lo, T, W, Kk, dilation):
        r = real_dw(dout, conv_pre, normed, seg_lo, T, W, Kk, dilation)
        seen.setdefault("dw", []).append(r.clone())
        seen["inputs"] = (dout.clone(), conv_pre.clone(), normed.clone(), seg_lo.clone(), Kk, dilation)
        return r

    class _SpyBwd:
        """Records the fp32 atomic dconvw of the default path (before the cast onto the weight dtype)."""

        def __getitem__(self, grid):
            launch = real_bwd[grid]

            def run(*args, **kw):
                launch(*args, **kw)
                if kw.get("ATOMIC_DW", True):
                    seen["atomic_dw"] = args[7].clone()

            return run

    ple._ple_conv_dw_deterministic = spy_dw
    ple._ple_conv_bwd_kernel = _SpyBwd()

    for i, (T, C, n, segs, dtype_name) in enumerate(CASES):
        dtype = getattr(torch, dtype_name)
        inputs, cu, dout = _inputs(T, C, n, segs, dtype, seed=200 + i)
        seen.clear()
        out_a, g_a = _run(ple, inputs, cu, dout, False)
        atomic_dw = seen.pop("atomic_dw")
        out_d1, g_d1 = _run(ple, inputs, cu, dout, True)
        out_d2, g_d2 = _run(ple, inputs, cu, dout, True)
        assert "atomic_dw" not in seen, "deterministic path ran the atomic accumulation"
        dw1, dw2 = seen["dw"]

        assert torch.equal(out_a, out_d1) and torch.equal(out_d1, out_d2)
        for name, a, b, c in zip(NAMES, g_a, g_d1, g_d2, strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert torch.isfinite(b.float()).all(), f"case {i} {name}: non-finite grad"
            assert b.float().abs().sum() > 0, f"case {i} {name}: all-zero grad"
            assert torch.equal(b, c), f"case {i} {name}: two deterministic runs differ"
            if name != "dconvw":
                assert torch.equal(a, b), f"case {i} {name}: max |diff| {(a.float() - b.float()).abs().max()}"
        assert torch.equal(dw1, dw2)
        assert torch.equal(g_d1[6], dw1.view_as(g_d1[6]).to(g_d1[6].dtype))

        d_out, conv_pre, normed, seg_lo, Kk, dil = seen["inputs"]
        ref = _ref_dw(d_out, conv_pre, normed, seg_lo, Kk, dil)
        scale = ref.abs().max().item()
        err_det = (dw1.double() - ref).abs().max().item()
        err_atomic = (atomic_dw.double() - ref).abs().max().item()
        # fp32 sums of T terms: |err| <~ T * eps32 * max|term|; both paths must sit well inside it
        tol = 1e-5 * scale * max(1.0, T / 64)
        print(
            f"case {i} T={T} W={n * C} segs={segs} {dtype_name}: 6 grads bitwise = atomic path; "
            f"dconvw max|ref| {scale:.3e} err det {err_det:.3e} atomic {err_atomic:.3e} tol {tol:.3e}",
            flush=True,
        )
        assert err_det <= tol, (err_det, tol)
        assert err_atomic <= tol, (err_atomic, tol)
    print("ALL-OK", flush=True)


def test_ple_gate_conv_deterministic_conv_weight_grad():
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


if __name__ == "__worker__":
    _worker()
