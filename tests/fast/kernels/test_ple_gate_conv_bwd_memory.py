"""CPU test: the PLE gate-conv backward frees its fp32 [T, W] buffers early and its grads stay bitwise equal.

``_PLEGateConv.backward`` used to keep every fp32 [T, W] local alive until it returned (9 of them at the
last allocation, 10 GiB each at T = 262144, W = 10240), which ran the 262,144-token calibration out of
memory. The fix only deletes buffers after their last read. This test runs the Triton kernels under
``TRITON_INTERPRET=1`` on CPU, in a subprocess because the interpreter flag must be set before
``triton.jit`` decorates the kernels. The subprocess runs the original backward (copied below
from ``af0e7996``) and the current one on the same inputs, and checks that:

* all 7 returned grads are ``torch.equal`` (bitwise), for 1 to 3 segments, bf16 and fp32, and ragged widths;
* the peak number of live fp32 [T, W]-sized tensors inside the backward drops from 10 to 6
  (the saved ``conv_pre`` is one of them in both).

The 262K-token memory figures come from the GPU script for the GPU executor, not from this test.
"""

import gc
import math
import os
import subprocess
import sys
import weakref
from pathlib import Path

from tests.ci.ci_register import register_cpu_ci

# 4 cases x 2 backwards under the Triton interpreter; about 60 s on an 8-thread CPU
register_cpu_ci(est_time=90, suite="stage-b-cpu", labels=["miles-plugin"])

ROOT = Path(__file__).resolve().parents[3]

# (T, C, n, segment bounds or None, dtype name)
CASES = [
    (64, 64, 4, None, "bfloat16"),
    (257, 64, 4, [0, 100, 257], "bfloat16"),
    (257, 80, 4, [0, 3, 140, 257], "bfloat16"),  # W = 320: two 256-wide conv blocks, the second one masked
    (64, 40, 4, [0, 30, 64], "float32"),  # C = 40: masked lanes in the gate / norm kernels
]
EXPECTED_PEAK = {"old": 10, "new": 6}  # liveness case: CASES[1]


def _old_backward(ctx, dout):
    """``_PLEGateConv.backward`` at ``af0e7996`` (before the early frees), same code (re-indented)."""
    import torch
    import triton

    from miles.kernels.embedding.ple_gate_conv import (
        _ple_conv_bwd_kernel,
        _ple_gate_bwd_kernel,
        _ple_gate_fwd_kernel,
        block_c,
        grouped_rmsnorm_bwd_kernel,
        grouped_rmsnorm_fwd,
    )

    hc_state, key, value, wk, wq, wc, convw2d, gate, rstdk, rstdq, rstdc, seg_lo, seg_hi, conv_pre = ctx.saved_tensors
    n, eps, dilation, Kk, conv_w_dtype = ctx.dims
    T, W = hc_state.shape
    C = W // n
    dev = hc_state.device
    dout = dout.contiguous()

    gated = torch.empty(T, W, dtype=torch.float32, device=dev)
    _g = torch.empty(T, n, dtype=torch.float32, device=dev)
    _rk = torch.empty(T, n, dtype=torch.float32, device=dev)
    _rq = torch.empty(T, n, dtype=torch.float32, device=dev)
    if T > 0:
        _ple_gate_fwd_kernel[(T * n,)](
            key,
            hc_state,
            value,
            wk,
            wq,
            gated,
            _g,
            _rk,
            _rq,
            T,
            N=n,
            C=C,
            EPS=eps,
            SQRTC=math.sqrt(C),
            BLOCK_C=block_c(C),
        )
    normed, _ = grouped_rmsnorm_fwd(gated, wc, n, eps)

    dnormed = torch.empty(T, W, dtype=torch.float32, device=dev)
    dconvw = torch.zeros(W, Kk, dtype=torch.float32, device=dev)
    dgated = torch.empty(T, W, dtype=torch.float32, device=dev)
    BW = 256
    if T > 0:
        _ple_conv_bwd_kernel[(T, triton.cdiv(W, BW))](
            dout,
            conv_pre,
            normed,
            convw2d,
            seg_lo,
            seg_hi,
            dnormed,
            dconvw,
            dgated,
            T,
            W,
            K=Kk,
            DIL=dilation,
            BLOCK_W=BW,
        )

    x_hat = (gated.view(T, n, C) * rstdc.unsqueeze(-1)).view(T, W)
    dwc = (dnormed * x_hat).sum(dim=0).to(wc.dtype)
    dgated_norm = torch.empty(T, W, dtype=torch.float32, device=dev)
    if T > 0:
        grouped_rmsnorm_bwd_kernel[(T * n,)](gated, wc, rstdc, dnormed, dgated_norm, T, N=n, C=C, BLOCK_C=block_c(C))
    dgated += dgated_norm

    dkey = torch.empty_like(key)
    dquery = torch.empty_like(hc_state)
    dvalue_pern = torch.empty(T, n, C, dtype=torch.float32, device=dev)
    dwk_part = torch.empty(T, W, dtype=torch.float32, device=dev)
    dwq_part = torch.empty(T, W, dtype=torch.float32, device=dev)
    if T > 0:
        _ple_gate_bwd_kernel[(T * n,)](
            dgated,
            key,
            hc_state,
            value,
            wk,
            wq,
            gate,
            rstdk,
            rstdq,
            dkey,
            dquery,
            dvalue_pern,
            dwk_part,
            dwq_part,
            T,
            N=n,
            C=C,
            SQRTC=math.sqrt(C),
            BLOCK_C=block_c(C),
        )
    dvalue = dvalue_pern.sum(dim=1).to(value.dtype)
    dwk = dwk_part.sum(dim=0).to(wk.dtype)
    dwq = dwq_part.sum(dim=0).to(wq.dtype)
    dconv_w = dconvw.view(W, 1, Kk).to(conv_w_dtype)

    return (dquery, dkey, dvalue, dwk, dwq, dwc, dconv_w, None, None, None, None)


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


def _in_triton_interpreter():
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename.endswith(os.path.join("triton", "runtime", "interpreter.py")):
            return True
        frame = frame.f_back
    return False


class _LiveCounter:
    """Peak number of distinct live fp32 buffers of T * W elements (one [T, W] each) created inside a backward."""

    def __init__(self, numel, extra):
        # ``extra``: such buffers alive before the backward starts (the saved conv_pre)
        self.numel, self.extra, self.refs, self.peak = numel, extra, [], 0

    def track(self, t):
        import torch

        if isinstance(t, torch.Tensor) and t.dtype == torch.float32 and t.numel() == self.numel:
            if _in_triton_interpreter():
                return t  # the interpreter re-wraps each kernel argument for a launch; a GPU launch does not
            # a view (x_hat) keeps its base's buffer alive: track every tensor, count distinct buffers
            self.refs = [r for r in self.refs if r() is not None] + [weakref.ref(t)]
            buffers = {r().untyped_storage().data_ptr() for r in self.refs if r() is not None}
            self.peak = max(self.peak, len(buffers) + self.extra)
        return t


def _run_backward(ple, which, inputs, cu, dout, counter=None):
    """Forward + backward through ``_PLEGateConv`` with the old or the current backward."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_map

    backward = _old_backward if which == "old" else ple._PLEGateConv.backward

    class _Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            tree_map(counter.track, out)
            return out

    def bwd(ctx, d):
        if counter is None:
            return backward(ctx, d)
        with _Mode():
            return backward(ctx, d)

    fn = type(
        f"_PLEGateConv_{which}",
        (torch.autograd.Function,),
        {"forward": staticmethod(ple._PLEGateConv.forward), "backward": staticmethod(bwd)},
    )
    params = [p.clone().requires_grad_() for p in inputs]
    hc, key, value, wk, wq, wc, convw = params
    out = fn.apply(hc, key, value, wk, wq, wc, convw, 4, 1e-6, 3, cu)
    out.backward(dout)
    gc.collect()
    return out.detach(), [p.grad for p in params]


def _worker():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    import torch

    from miles.kernels.embedding import ple_gate_conv as ple

    names = ["dhc", "dkey", "dvalue", "dwk", "dwq", "dwc", "dconvw"]
    for i, (T, C, n, segs, dtype_name) in enumerate(CASES):
        dtype = getattr(torch, dtype_name)
        inputs, cu, dout = _inputs(T, C, n, segs, dtype, seed=100 + i)
        out_old, g_old = _run_backward(ple, "old", inputs, cu, dout)
        out_new, g_new = _run_backward(ple, "new", inputs, cu, dout)
        assert torch.equal(out_old, out_new)
        for name, a, b in zip(names, g_old, g_new, strict=True):
            assert a is not None and b is not None, name
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert torch.isfinite(a.float()).all(), f"case {i} {name}: non-finite grad"
            assert a.float().abs().sum() > 0, f"case {i} {name}: all-zero grad"
            assert torch.equal(a, b), f"case {i} {name}: max |diff| {(a.float() - b.float()).abs().max()}"
        print(f"case {i} T={T} C={C} n={n} segs={segs} {dtype_name}: 7 grads bitwise equal", flush=True)

    T, C, n, segs, dtype_name = CASES[1]
    inputs, cu, dout = _inputs(T, C, n, segs, getattr(torch, dtype_name), seed=7)
    peaks = {}
    for which in ("old", "new"):
        counter = _LiveCounter(T * n * C, extra=1)
        _run_backward(ple, which, inputs, cu, dout, counter)
        peaks[which] = counter.peak
    print(f"peak live fp32 [T, W] tensors in the backward (incl. saved conv_pre): {peaks}", flush=True)
    assert peaks == EXPECTED_PEAK, peaks
    print("ALL-OK", flush=True)


def test_ple_gate_conv_backward_bitwise_and_early_free():
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
