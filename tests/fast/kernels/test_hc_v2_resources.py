"""CPU test: the v2 GEMM kernels, compiled ahead of time for B300 (sm_103) with the locked default tiles, fit the
hardware's tensor memory (512 columns) and shared memory, for every matmul precision, bf16 and fp32 operands, and a
small (C 64, R 16: one K step) and the production (C 2560, R 320) shape.

E1 on 115 (2026-11-09, fc2dc2d5) failed two GPU cases with ``OutOfResources: tensor memory, Required: 576, Hardware
limit: 512`` in ``_v2_up_mix_kernel`` at R 16, tf32; this compile-only check reproduces that number on a CPU machine
(Triton's NVIDIA backend and ptxas, no GPU and no driver needed), so a tile or kernel change that would not launch
is caught before GPU time. It needs the production image (Triton with the NVIDIA backend); elsewhere it is skipped.
"""

import pytest

triton = pytest.importorskip("triton")

from tests.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=600, suite="stage-b-cpu", labels=["miles-plugin"])

TMEM_COLUMNS = 512
SMEM_BYTES = 232448  # B300 maximum dynamic shared memory per block
SHAPES = [(64, 4, 16), (2560, 4, 320)]  # (C, n, R)
KERNELS = {  # kernel -> (config key, pointer signature by role)
    "_v2_down_kernel": ("down", ["x", "w", "f32", "w", "f32", "s"]),
    "_v2_up_mix_kernel": ("up", ["s", "w", "x", "w", "f32", "x"]),
    "_v2_up_bwd_kernel": ("up", ["s", "w", "x", "w", "f32", "x", "s", "f32"]),
    "_v2_wgrad_kernel": ("wgrad", ["s", "x", "w", "f32", "f32"]),
    "_v2_dx_kernel": ("dx", ["s", "w", "f32", "x", "w", "f32", "x", "f32"]),
}


def _compile(hc_v2, name, C, n, R, prec, act):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = getattr(hc_v2, name)
    key, roles = KERNELS[name]
    cfg = hc_v2._DEFAULT_CONFIG[key]
    dtype = {"x": act, "w": act, "f32": "*fp32", "s": "*bf16" if prec == 2 else "*fp32"}
    consts = {"N": n, "C": C, "R": R, "M": R + n, "BM": cfg["BM"], "BN": cfg["BN"], "BK": cfg["BK"], "PREC": prec,
              "ONE_STREAM": C % cfg["BN"] == 0}  # fmt: skip
    consts = {k: v for k, v in consts.items() if k in fn.arg_names}
    ptrs = [a for a in fn.arg_names if a.endswith("_ptr")]
    assert len(ptrs) == len(roles), (name, ptrs)
    sig = {a: dtype[r] for a, r in zip(ptrs, roles, strict=True)}
    sig.update({a: "constexpr" for a in consts})
    sig.update({a: "i32" for a in fn.arg_names if a not in sig})
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", 103, 32),
                       options={"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"]})  # fmt: skip
    return getattr(k.metadata, "tmem_size", 0) or 0, getattr(k.metadata, "shared", 0) or 0


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_v2_kernels_fit_b300(name):
    try:
        from triton.backends.compiler import GPUTarget  # noqa: F401
    except ImportError:
        pytest.skip("no Triton NVIDIA backend")
    from miles.kernels.hyper_connection import hc_v2

    bad = []
    for C, n, R in SHAPES:
        for prec in (0, 1, 2):
            for act in ("*bf16", "*fp32"):
                if prec == 2 and act == "*fp32":
                    continue  # bf16 mode rounds operands to bf16; fp32-activation tests run fp32 / tf32 only
                tmem, smem = _compile(hc_v2, name, C, n, R, prec, act)
                print(f"{name} C={C} R={R} PREC={prec} {act}: tmem {tmem} smem {smem}", flush=True)
                if tmem > TMEM_COLUMNS or smem > SMEM_BYTES:
                    bad.append((C, R, prec, act, tmem, smem))
    assert bad == [], f"{name} exceeds B300 resources (C, R, PREC, dtype, tmem, smem): {bad}"
