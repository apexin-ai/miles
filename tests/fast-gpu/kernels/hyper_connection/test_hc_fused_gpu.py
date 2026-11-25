"""``hc_mix_inject(..., fused=True)`` (``--qwen38-hc-fused``) on the GPU: numerics against the default path and an
fp64 reference, determinism, and (``__main__``) a numerics report and a micro-benchmark on Qwen3.8-Flash-Next shapes.

pytest: moderate shapes (C 2560, n 4, R 320 included; T 4097 for a ragged norm-backward token block), every matmul
precision, with and without the inject weight, plus a strided inject weight and fp32 activations: every output and
gradient is finite, the fused error against fp64 is at most 1.5x the default path's (plus a floor), the cosine to the
default path is >= 0.9999, and two fused runs are bitwise equal.

Report and benchmark (single GPU, no distributed setup)::

    PYTHONPATH=<miles> python -B tests/fast-gpu/kernels/hyper_connection/test_hc_fused_gpu.py \\
        --report --bench --out /path/h2-gpu.json

``--report``: per output and gradient, fused vs default and both vs fp64 (max abs, max rel, rel Frobenius, cosine) at
T 4096 and 65536 (the SP slice of one 262,144-token row under TP4), fp32 / tf32 / bf16, xavier and 8x-wide weights.
``--bench``: median ms per call of forward and forward+backward, CUDA kernels per call and peak extra memory, both
paths, T 512 / 4096 / 16384 / 65536, tf32 (the live run's --qwen38-hc-matmul-precision).
"""

import argparse
import json
import statistics
import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.hyper_connection import hc  # noqa: E402

EPS = 1e-6
NAMES = ["mixed", "h_post", "dx", "dweight", "dw_down", "dw_up", "dw_inject"]


def make_inputs(T, C=2560, n=4, R=320, dtype=torch.bfloat16, seed=0, wide=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = n * C

    def rnd(*shape, std=1.0):
        return (std * torch.randn(*shape, device="cuda", generator=g)).to(dtype)

    xavier = lambda a, b: (2.0 / (a + b)) ** 0.5  # noqa: E731
    x = rnd(T, W)
    weight = rnd(W, std=0.05)
    w_down = rnd(R, W, std=wide * xavier(R, W))
    w_up = rnd(W, R, std=wide * xavier(W, R))
    w_inj = rnd(n, W, std=wide * xavier(n, W))
    dmixed = rnd(T, C)
    dh_post = torch.randn(T, n, device="cuda", generator=g)
    return [x, weight, w_down, w_up, w_inj], dmixed, dh_post


def run(fn, tensors, dmixed, dh_post, with_inject=True):
    leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
    if not with_inject:
        leaves[4] = None
    mixed, h_post = fn(*leaves)
    grads = [mixed]
    if with_inject:
        torch.autograd.backward([mixed, h_post], [dmixed.to(mixed.dtype), dh_post.to(h_post.dtype)])
        grads.append(h_post)
    else:
        mixed.backward(dmixed.to(mixed.dtype))
    grads += [leaf.grad for leaf in leaves if leaf is not None]
    return [t.detach() for t in grads]


def path(n, precision, fused):
    def fn(x, weight, w_down, w_up, w_inj):
        kwargs = {"fused": True} if fused else {}
        return hc.hc_mix_inject(x, weight, w_down, w_up, w_inj, n, EPS, matmul_precision=precision, **kwargs)

    return fn


def reference(n, round_operands):
    """fp64 math; ``round_operands`` rounds each matmul's operands to bf16 where bf16 mode does."""
    import torch.nn.functional as F

    rnd = (lambda t: t.bfloat16().double()) if round_operands else (lambda t: t)

    def fn(x, weight, w_down, w_up, w_inj):
        xg = x.unflatten(-1, (n, -1))
        normed = (xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + EPS)).flatten(-2) * (1.0 + weight)
        z1 = F.linear(rnd(normed), rnd(w_down)) / n
        gate = torch.sigmoid(F.linear(rnd(F.silu(z1)), rnd(w_up)))
        mixed = (gate * normed).unflatten(-1, (n, -1)).mean(dim=1)
        h_post = 2.0 * torch.sigmoid(F.linear(rnd(normed), rnd(w_inj)) / n) if w_inj is not None else None
        return mixed, h_post

    return fn


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    d = (a - b).abs()
    floor = 1e-3 * b.abs().max().clamp_min(1e-30)
    bn = b.norm().item()
    return {
        "max_abs": d.max().item() if d.numel() else 0.0,
        "max_rel": (d / (b.abs() + floor)).max().item() if d.numel() else 0.0,
        "rel_fro": (d.norm().item() / bn) if bn > 0 else d.norm().item(),
        "cos": torch.nn.functional.cosine_similarity(a, b, dim=0).item() if bn > 0 else 1.0,
    }


def compare(T, C, n, R, precision, with_inject, wide=1.0, seed=0, dtype=torch.bfloat16, strided_inject=False):
    tensors, dmixed, dh_post = make_inputs(T, C, n, R, dtype=dtype, seed=seed, wide=wide)
    if strided_inject:  # [n, n*C] with stride (1, n): a transpose of a contiguous [n*C, n] (review H2REV P2)
        tensors[4] = tensors[4].t().contiguous().t()
        assert not tensors[4].clone().is_contiguous()
    default = run(path(n, precision, False), tensors, dmixed, dh_post, with_inject)
    fused = run(path(n, precision, True), tensors, dmixed, dh_post, with_inject)
    again = run(path(n, precision, True), tensors, dmixed, dh_post, with_inject)
    ref = run(
        reference(n, precision == "bf16"),
        [t.double() for t in tensors],
        dmixed.double(),
        dh_post.double(),
        with_inject,
    )
    names = NAMES if with_inject else [m for m in NAMES if m not in ("h_post", "dw_inject")]
    rows = {}
    for name, f, d, a, r in zip(names, fused, default, again, ref, strict=True):
        assert torch.isfinite(f).all() and torch.isfinite(d).all(), name
        rows[name] = {
            "fused_vs_default": metrics(f, d),
            "fused_vs_fp64": metrics(f, r),
            "default_vs_fp64": metrics(d, r),
            "fused_deterministic": bool(torch.equal(f, a)),
            "dtype": str(f.dtype),
        }
    del tensors, default, fused, again, ref
    torch.cuda.empty_cache()
    return rows


# 4097: the norm backward runs 16-token blocks (_bwd_block_t) with a ragged last block on the GPU
SHAPES = [(7, 64, 4, 16), (300, 2560, 4, 320), (4096, 2560, 4, 320), (4097, 2560, 4, 320)]


@pytest.mark.parametrize("T,C,n,R", SHAPES)
@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
@pytest.mark.parametrize("with_inject", [True, False])
def test_fused_matches_default_and_fp64(T, C, n, R, precision, with_inject):
    _assert_rows(compare(T, C, n, R, precision, with_inject, seed=T), precision)


def _assert_rows(rows, precision):
    for name, r in rows.items():
        err_f, err_d = r["fused_vs_fp64"]["rel_fro"], r["default_vs_fp64"]["rel_fro"]
        floor = 2e-6 if r["dtype"] == "torch.float32" and precision == "fp32" else 1e-4
        assert r["fused_deterministic"], name
        assert err_f <= 1.5 * err_d + floor, (name, err_f, err_d)
        assert r["fused_vs_default"]["cos"] >= 0.9999, (name, r["fused_vs_default"])


@pytest.mark.parametrize("T,C,n,R", [(7, 13, 3, 5), (300, 2560, 4, 320)])
@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
def test_strided_inject_weight(T, C, n, R, precision):
    _assert_rows(compare(T, C, n, R, precision, True, seed=T + 1, strided_inject=True), precision)


@pytest.mark.parametrize("precision", ["fp32", "tf32"])
def test_fp32_activations(precision):
    _assert_rows(compare(300, 2560, 4, 320, precision, True, seed=5, dtype=torch.float32), precision)


# ---------------------------------------------------------------------------------------------------- report / bench


def _time_ms(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def _kernel_count(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


def bench_one(T, precision, fused, with_inject=True, iters=None):
    n, C, R = 4, 2560, 320
    tensors, dmixed, dh_post = make_inputs(T, C, n, R)
    leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
    if not with_inject:
        leaves[4] = None
    fn = path(n, precision, fused)
    grads = [dmixed, dh_post]

    def fwd():
        return fn(*leaves)

    def fwd_bwd():
        mixed, h_post = fn(*leaves)
        if with_inject:
            torch.autograd.backward([mixed, h_post], grads)
        else:
            mixed.backward(dmixed)
        for leaf in leaves:
            if leaf is not None:
                leaf.grad = None

    iters = iters or (10 if T >= 32768 else 30)
    t_fwd = _time_ms(fwd, iters)
    t_all = _time_ms(fwd_bwd, iters)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fwd_bwd()
    torch.cuda.synchronize()
    peak_gib = (torch.cuda.max_memory_allocated() - base) / 2**30
    k_fwd, k_all = _kernel_count(fwd), _kernel_count(fwd_bwd)
    del leaves, tensors
    torch.cuda.empty_cache()
    return {
        "fwd_ms": t_fwd,
        "fwd_bwd_ms": t_all,
        "bwd_ms": t_all - t_fwd,
        "kernels_fwd": k_fwd,
        "kernels_fwd_bwd": k_all,
        "peak_extra_gib": peak_gib,
    }


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--report-T", type=int, nargs="+", default=[4096, 65536])
    p.add_argument("--bench-T", type=int, nargs="+", default=[512, 4096, 16384, 65536])
    p.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "bf16"])
    p.add_argument("--bench-precisions", nargs="+", default=["tf32"])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "report": [], "bench": []}
    import triton

    out["triton"] = triton.__version__
    if a.report:
        for T in a.report_T:
            for precision in a.precisions:
                for wide in (1.0, 8.0):
                    for with_inject in (True, False):
                        rows = compare(T, 2560, 4, 320, precision, with_inject, wide=wide, seed=1)
                        out["report"].append(
                            {"T": T, "precision": precision, "wide": wide, "inject": with_inject, "rows": rows}
                        )
                        for name, r in rows.items():
                            fd, ff, dd = r["fused_vs_default"], r["fused_vs_fp64"], r["default_vs_fp64"]
                            print(
                                f"REPORT T={T} {precision} wide={wide:g} inject={with_inject} {name:9s} "
                                f"fused-vs-default max_abs {fd['max_abs']:.2e} max_rel {fd['max_rel']:.2e} "
                                f"rel {fd['rel_fro']:.2e} cos {fd['cos']:.8f} | vs fp64 rel: fused {ff['rel_fro']:.2e}"
                                f" default {dd['rel_fro']:.2e} | det {r['fused_deterministic']}",
                                flush=True,
                            )
    if a.bench:
        for precision in a.bench_precisions:
            for T in a.bench_T:
                for with_inject in (True, False):
                    d = bench_one(T, precision, False, with_inject)
                    f = bench_one(T, precision, True, with_inject)
                    row = {"T": T, "precision": precision, "inject": with_inject, "default": d, "fused": f}
                    row["speedup_fwd"] = d["fwd_ms"] / f["fwd_ms"]
                    row["speedup_fwd_bwd"] = d["fwd_bwd_ms"] / f["fwd_bwd_ms"]
                    out["bench"].append(row)
                    print(
                        f"BENCH T={T} {precision} inject={with_inject} default fwd {d['fwd_ms']:.3f} ms fwd+bwd "
                        f"{d['fwd_bwd_ms']:.3f} ms ({d['kernels_fwd_bwd']} kernels, {d['peak_extra_gib']:.2f} GiB) | "
                        f"fused fwd {f['fwd_ms']:.3f} ms fwd+bwd {f['fwd_bwd_ms']:.3f} ms ({f['kernels_fwd_bwd']} "
                        f"kernels, {f['peak_extra_gib']:.2f} GiB) | speedup fwd {row['speedup_fwd']:.2f}x "
                        f"fwd+bwd {row['speedup_fwd_bwd']:.2f}x",
                        flush=True,
                    )
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"WROTE {a.out}", flush=True)
    print("H2_GPU_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
