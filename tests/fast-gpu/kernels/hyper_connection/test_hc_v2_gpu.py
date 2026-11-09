"""``hc_mix_inject(..., v2=True)`` (``--qwen38-hc-kernel-v2``) on the GPU: numerics against the default path, the
fused path and an fp64 reference; determinism; length invariance; and (``__main__``) a report, a micro-benchmark and a
tile sweep on Qwen3.8-Flash-Next shapes.

pytest (one GPU, about 2 minutes): shapes up to C 2560 / R 320 / T 4097, every matmul precision, with and without the
inject weight, fp32 activations, strided weights: every output and gradient is finite, v2's error against fp64 is at
most 1.25x the default path's (plus a floor), the cosine to the default path is >= 0.9999, two v2 runs are bitwise
equal, strided weights give the contiguous bits, and a token's forward outputs are bitwise the same at T 896, 1408 and
2944 and at any row position (the shapes of the 2026-11-08 HC root-cause replay).

Report / benchmark / sweep (single GPU)::

    PYTHONPATH=<miles> python -B tests/fast-gpu/kernels/hyper_connection/test_hc_v2_gpu.py \\
        --report --bench --out /path/hcv2-gpu.json
    PYTHONPATH=<miles> python -B tests/fast-gpu/kernels/hyper_connection/test_hc_v2_gpu.py --sweep --out sweep.json

``--report``: per output / gradient, v2, fused and default against fp64 (max abs, max rel, rel Frobenius, cosine) at
T 4096 and 65536, fp32 / tf32 / bf16, xavier and 8x-wide weights. ``--bench``: median ms of forward and
forward+backward, kernels per call and peak extra memory for default / fused / v2 at T 512 / 4096 / 16384 / 65536
(tf32). ``--sweep``: v2 forward+backward ms for each candidate tile set at T 16384 and 65536 (speed only: other tiles
are other numerics, so a winner becomes the new fixed default, never a per-T choice).
"""

import argparse
import json
import os
import statistics
import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
# HCV2_TEST_DEVICE=cpu (with TRITON_INTERPRET=1) runs the numerics helpers on the CPU: a smoke test of this script only
DEVICE = os.environ.get("HCV2_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skipif(DEVICE == "cuda" and not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.hyper_connection import hc, hc_v2  # noqa: E402

EPS = 1e-6
NAMES = ["mixed", "h_post", "dx", "dweight", "dw_down", "dw_up", "dw_inject"]
PATHS = ("default", "fused", "v2")


def make_inputs(T, C=2560, n=4, R=320, dtype=torch.bfloat16, seed=0, wide=1.0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W = n * C

    def rnd(*shape, std=1.0):
        return (std * torch.randn(*shape, device=DEVICE, generator=g)).to(dtype)

    xavier = lambda a, b: (2.0 / (a + b)) ** 0.5  # noqa: E731
    x = rnd(T, W)
    weight = rnd(W, std=0.05)
    w_down = rnd(R, W, std=wide * xavier(R, W))
    w_up = rnd(W, R, std=wide * xavier(W, R))
    w_inj = rnd(n, W, std=wide * xavier(n, W))
    dmixed = rnd(T, C)
    dh_post = torch.randn(T, n, device=DEVICE, generator=g)
    return [x, weight, w_down, w_up, w_inj], dmixed, dh_post


def run(fn, tensors, dmixed, dh_post, with_inject=True):
    leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
    if not with_inject:
        leaves[4] = None
    mixed, h_post = fn(*leaves)
    outs = [mixed]
    if with_inject:
        torch.autograd.backward([mixed, h_post], [dmixed.to(mixed.dtype), dh_post.to(h_post.dtype)])
        outs.append(h_post)
    else:
        mixed.backward(dmixed.to(mixed.dtype))
    outs += [leaf.grad for leaf in leaves if leaf is not None]
    return [t.detach() for t in outs]


def path(n, precision, which):
    kwargs = {"default": {}, "fused": {"fused": True}, "v2": {"v2": True}}[which]

    def fn(x, weight, w_down, w_up, w_inj):
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
        "finite": bool(torch.isfinite(a).all().item()),
    }


def compare(T, C, n, R, precision, with_inject, wide=1.0, seed=0, dtype=torch.bfloat16, strided=False):
    tensors, dmixed, dh_post = make_inputs(T, C, n, R, dtype=dtype, seed=seed, wide=wide)
    if strided:  # every weight a strided view (transposes, stride-2 norm weight)
        tensors[1] = torch.stack([tensors[1], tensors[1]], dim=1)[:, 0]
        for i in (2, 3, 4):
            tensors[i] = tensors[i].t().contiguous().t()
    res = {p: run(path(n, precision, p), tensors, dmixed, dh_post, with_inject) for p in PATHS}
    again = run(path(n, precision, "v2"), tensors, dmixed, dh_post, with_inject)
    contiguous = None
    if strided:
        contiguous = run(path(n, precision, "v2"), [t.contiguous() for t in tensors], dmixed, dh_post, with_inject)
    ref = run(
        reference(n, precision == "bf16"), [t.double() for t in tensors], dmixed.double(), dh_post.double(), with_inject
    )
    names = NAMES if with_inject else [m for m in NAMES if m not in ("h_post", "dw_inject")]
    rows = {}
    for i, name in enumerate(names):
        v = res["v2"][i]
        rows[name] = {
            "v2_vs_default": metrics(v, res["default"][i]),
            "v2_vs_fused": metrics(v, res["fused"][i]),
            "v2_vs_fp64": metrics(v, ref[i]),
            "fused_vs_fp64": metrics(res["fused"][i], ref[i]),
            "default_vs_fp64": metrics(res["default"][i], ref[i]),
            "v2_deterministic": bool(torch.equal(v, again[i])),
            "strided_equals_contiguous": None if contiguous is None else bool(torch.equal(v, contiguous[i])),
            "dtype": str(v.dtype),
            "shape_ok": tuple(v.shape) == tuple(res["default"][i].shape) and v.dtype == res["default"][i].dtype,
        }
    del tensors, res, again, ref, contiguous
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return rows


def length_invariance(precision, C=2560, n=4, R=320, scale=1):
    """The same 512 rows at T 896 / 1408 / 2944 and at offsets 0 / 384 / 2432 (tile-aligned: bitwise equal by
    construction) and 1001 (not aligned: equal only if the MMA's per-row K order does not depend on the row's place in
    the tile; reported, not gated)."""
    L = 512 // scale
    base, _, _ = make_inputs(L, C, n, R, seed=3)
    rows, weights = base[0], base[1:]
    outs = []
    for T, off, seed in ((512, 0, None), (896, 384, 4), (1408, 0, 5), (2944, 2432, 6), (2944, 1001, 7)):
        T, off = T // scale, off // scale
        if seed is None:
            x = rows
        else:
            x = make_inputs(T, C, n, R, seed=seed)[0][0].clone()
            x[off : off + L] = rows
        with torch.no_grad():
            mixed, h_post = hc.hc_mix_inject(x, *weights, n, EPS, matmul_precision=precision, v2=True)
        outs.append((T, off, mixed[off : off + L].clone(), h_post[off : off + L].clone()))
    return [
        {"T": T, "offset": off, "aligned": off % (128 // scale) == 0, "mixed_equal": bool(torch.equal(m, outs[0][2])),
         "h_post_equal": bool(torch.equal(h, outs[0][3]))}
        for T, off, m, h in outs[1:]
    ]


def check_rows(rows, precision):
    """Numerics gate N1-N4 (RUNBOOK): finite, shape / dtype, determinism, v2 vs fp64 <= 1.25x default + floor, cos."""
    bad = []
    for name, r in rows.items():
        err_v, err_d = r["v2_vs_fp64"]["rel_fro"], r["default_vs_fp64"]["rel_fro"]
        floor = 2e-6 if r["dtype"] == "torch.float32" and precision == "fp32" else 2e-5
        if not (r["v2_vs_fp64"]["finite"] and r["shape_ok"] and r["v2_deterministic"]):
            bad.append((name, "finite/shape/det"))
        if r["strided_equals_contiguous"] is False:
            bad.append((name, "strided"))
        if not err_v <= 1.25 * err_d + floor:
            bad.append((name, "fp64", err_v, err_d))
        if not r["v2_vs_default"]["cos"] >= 0.9999:
            bad.append((name, "cos", r["v2_vs_default"]["cos"]))
    return bad


# 4097 / 300: ragged last token tile of every kernel; (7, 64, 4, 16): tiny R (masked K / N tiles)
SHAPES = [(7, 64, 4, 16), (300, 2560, 4, 320), (4096, 2560, 4, 320), (4097, 2560, 4, 320)]


@pytest.mark.parametrize("T,C,n,R", SHAPES)
@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
@pytest.mark.parametrize("with_inject", [True, False])
def test_v2_matches_default_and_fp64(T, C, n, R, precision, with_inject):
    assert check_rows(compare(T, C, n, R, precision, with_inject, seed=T), precision) == []


@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
def test_strided_weights(precision):
    assert check_rows(compare(300, 2560, 4, 320, precision, True, seed=11, strided=True), precision) == []


@pytest.mark.parametrize("precision", ["fp32", "tf32"])
def test_fp32_activations(precision):
    assert check_rows(compare(300, 2560, 4, 320, precision, True, seed=5, dtype=torch.float32), precision) == []


@pytest.mark.parametrize("precision", ["fp32", "tf32", "bf16"])
def test_length_invariance(precision):
    for row in length_invariance(precision):
        if row["aligned"]:
            assert row["mixed_equal"] and row["h_post_equal"], row


# ---------------------------------------------------------------------------------------------- report / bench / sweep


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


def _kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    per = {}
    count = 0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            count += 1
            k = e.name[:80]
            per[k] = per.get(k, 0.0) + e.device_time_total / 1000.0
    return count, dict(sorted(per.items(), key=lambda kv: -kv[1])[:20])


def bench_one(T, precision, which, with_inject=True, iters=None):
    n, C, R = 4, 2560, 320
    tensors, dmixed, dh_post = make_inputs(T, C, n, R)
    leaves = [t.detach().clone().requires_grad_(True) for t in tensors]
    if not with_inject:
        leaves[4] = None
    fn = path(n, precision, which)

    def fwd():
        return fn(*leaves)

    def fwd_bwd():
        mixed, h_post = fn(*leaves)
        if with_inject:
            torch.autograd.backward([mixed, h_post], [dmixed, dh_post])
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
    k_fwd, _ = _kernels(fwd)
    k_all, per_kernel = _kernels(fwd_bwd)
    del leaves, tensors
    torch.cuda.empty_cache()
    return {
        "fwd_ms": t_fwd,
        "fwd_bwd_ms": t_all,
        "bwd_ms": t_all - t_fwd,
        "kernels_fwd": k_fwd,
        "kernels_fwd_bwd": k_all,
        "peak_extra_gib": peak_gib,
        "top_kernels_ms": per_kernel,
    }


SWEEP = [
    {},
    {"down": {"BM": 64, "BN": 64, "BK": 64}},
    {"down": {"BM": 128, "BN": 64, "BK": 128, "num_stages": 2}},
    {"down": {"BM": 128, "BN": 128, "BK": 64}},
    {"up": {"BM": 128, "BN": 64, "BK": 64}, "dx": {"BM": 128, "BN": 64}},
    {"up": {"BM": 64, "BN": 64, "BK": 64}},
    {"up": {"BM": 128, "BN": 128, "BK": 32, "num_warps": 8}},
    {"wgrad": {"BM": 128, "BN": 128, "BK": 64}},
    {"wgrad": {"BM": 64, "BN": 256, "BK": 32}},
    {"dx": {"BM": 32, "BN": 128, "BK": 64}},
    {"dx": {"BM": 128, "BN": 128, "BK": 64, "num_warps": 8}},
    {"stats": {"BT": 8, "num_warps": 16}},
    {"stats": {"BT": 2, "num_warps": 8}},
]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--report", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--report-T", type=int, nargs="+", default=[4096, 65536])
    p.add_argument("--bench-T", type=int, nargs="+", default=[512, 4096, 16384, 65536])
    p.add_argument("--sweep-T", type=int, nargs="+", default=[16384, 65536])
    p.add_argument("--precisions", nargs="+", default=["fp32", "tf32", "bf16"])
    p.add_argument("--bench-precisions", nargs="+", default=["tf32"])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    import triton

    out = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "config": hc_v2._config(),
        "report": [],
        "invariance": {},
        "bench": [],
        "sweep": [],
    }
    if a.report:
        for precision in a.precisions:
            out["invariance"][precision] = length_invariance(precision)
            print(f"INVARIANCE {precision} {out['invariance'][precision]}", flush=True)
        for T in a.report_T:
            for precision in a.precisions:
                for wide in (1.0, 8.0):
                    for with_inject in (True, False):
                        rows = compare(T, 2560, 4, 320, precision, with_inject, wide=wide, seed=1)
                        bad = check_rows(rows, precision)
                        out["report"].append(
                            {"T": T, "precision": precision, "wide": wide, "inject": with_inject, "rows": rows,
                             "violations": [list(map(str, b)) for b in bad]}
                        )  # fmt: skip
                        for name, r in rows.items():
                            print(
                                f"REPORT T={T} {precision} wide={wide:g} inject={with_inject} {name:9s} vs fp64 rel: "
                                f"v2 {r['v2_vs_fp64']['rel_fro']:.3e} fused {r['fused_vs_fp64']['rel_fro']:.3e} "
                                f"default {r['default_vs_fp64']['rel_fro']:.3e} | v2-vs-default cos "
                                f"{r['v2_vs_default']['cos']:.8f} | det {r['v2_deterministic']}",
                                flush=True,
                            )
    if a.bench:
        for precision in a.bench_precisions:
            for T in a.bench_T:
                for with_inject in (True, False):
                    row = {"T": T, "precision": precision, "inject": with_inject}
                    for which in PATHS:
                        row[which] = bench_one(T, precision, which, with_inject)
                    row["speedup_v2_vs_default"] = row["default"]["fwd_bwd_ms"] / row["v2"]["fwd_bwd_ms"]
                    row["speedup_v2_vs_fused"] = row["fused"]["fwd_bwd_ms"] / row["v2"]["fwd_bwd_ms"]
                    out["bench"].append(row)
                    print(
                        "BENCH T={T} {p} inject={i} | ".format(T=T, p=precision, i=with_inject)
                        + " | ".join(
                            f"{w} fwd {row[w]['fwd_ms']:.3f} fwd+bwd {row[w]['fwd_bwd_ms']:.3f} ms "
                            f"({row[w]['kernels_fwd_bwd']} k, {row[w]['peak_extra_gib']:.2f} GiB)"
                            for w in PATHS
                        )
                        + f" | v2 vs default {row['speedup_v2_vs_default']:.2f}x vs fused {row['speedup_v2_vs_fused']:.2f}x",
                        flush=True,
                    )
    if a.sweep:
        for cand in SWEEP:
            os.environ["MILES_HCV2_CONFIG"] = json.dumps(cand)
            for T in a.sweep_T:
                try:
                    r = bench_one(T, "tf32", "v2", True)
                    row = {"candidate": cand, "T": T, "fwd_ms": r["fwd_ms"], "fwd_bwd_ms": r["fwd_bwd_ms"]}
                except Exception as exc:  # a tile set that does not compile / fit is a result, not a crash
                    row = {"candidate": cand, "T": T, "error": repr(exc)[:300]}
                out["sweep"].append(row)
                print(f"SWEEP {json.dumps(row)}", flush=True)
        os.environ.pop("MILES_HCV2_CONFIG", None)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"WROTE {a.out}", flush=True)
    print("HCV2_GPU_DONE", flush=True)


def cpu_smoke():
    """HCV2_TEST_DEVICE=cpu TRITON_INTERPRET=1: run compare / check_rows / length_invariance on small shapes."""
    assert DEVICE == "cpu"
    import numpy as np
    import triton.language as tl
    import triton.runtime.interpreter as interp

    # the interpreter truncates fp32 -> bf16 casts; the GPU and torch round to nearest even (as test_hc_v2.py)
    orig = interp.InterpreterBuilder.cast_impl

    def cast_impl(self, src, dst_type):
        if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
            data = torch.from_numpy(np.ascontiguousarray(src.data, dtype=np.float32)).bfloat16()
            return interp.TensorHandle(data.view(torch.int16).numpy().view(np.uint16), dst_type.scalar)
        return orig(self, src, dst_type)

    interp.InterpreterBuilder.cast_impl = cast_impl
    for precision in ("fp32", "tf32", "bf16"):
        for with_inject in (True, False):
            for strided in (False, True):
                rows = compare(37, 64, 4, 16, precision, with_inject, seed=2, strided=strided and with_inject)
                assert check_rows(rows, precision) == [], (precision, with_inject, strided, check_rows(rows, precision))
        inv = length_invariance(precision, C=64, R=16, scale=16)
        assert all(r["mixed_equal"] and r["h_post_equal"] for r in inv), inv
    # the gate must reject a broken row (positive control of check_rows)
    rows = compare(37, 64, 4, 16, "fp32", True, seed=2)
    rows["dx"]["v2_vs_fp64"]["rel_fro"] = 10 * rows["dx"]["default_vs_fp64"]["rel_fro"] + 1e-3
    rows["mixed"]["v2_deterministic"] = False
    assert {b[0] for b in check_rows(rows, "fp32")} == {"dx", "mixed"}
    print("CPU-SMOKE-OK", flush=True)


if __name__ == "__main__":
    if DEVICE == "cpu":
        cpu_smoke()
    else:
        main(sys.argv[1:])
