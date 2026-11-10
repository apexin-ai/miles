"""CPU test: every QSA block-attention launch the GPU tests make, compiled ahead of time for B300 (sm_103), fits the
hardware's shared memory (232448 bytes per block) and tensor memory (512 columns).

GT's 115 run (2026-11-09) stopped at ``test_launch_variant_matches_reference_and_is_deterministic[g6-nu32]``:
``OutOfResources: shared memory, Required: 296992, Hardware limit: 232448`` in ``_fwd_kernel`` (NU 32 with the sm_100
row's num_stages 3). This compile-only check reproduces that number on a CPU machine (Triton's NVIDIA backend and
ptxas; no GPU, no driver) when it marks the pointers and int arguments 16-divisible, the most specialized class a
launch can get (lengths such as n_queries do not change the numbers; the unspecialized compile gives 231448 and would
miss it), so a launch that cannot run is caught before GPU time.

Covered: the table row sm_103 takes (sm_100's) at every GPU-test shape, every ``LAUNCH_VARIANTS`` entry of the GPU
test at its ``LAUNCH_CASES`` shapes (fwd, dq, dK / dV, and the reduce kernel when heavy_split is on), and the
regression ``nu32 at num_stages 3`` that must keep failing. Needs the production image; skipped elsewhere.
"""

import importlib.util
from pathlib import Path

import pytest

triton = pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[2]
SMEM_BYTES = 232448
TMEM_COLUMNS = 512
BF16 = {"Q", "K", "V", "OUT", "DO", "DQ", "DK", "DV"}
FP32 = {"LSE", "DELTA", "PART_DK", "PART_DV"}
FLOATS = {"qk_scale", "sm_scale"}


def _gpu_test_module():
    path = ROOT / "tests/fast-gpu/kernels/attention/qsa/test_qsa_block_attention.py"
    spec = importlib.util.spec_from_file_location("qsa_gpu_test_for_resources", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compile(kernel, consts, num_warps, num_stages):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from miles.kernels.attention.qsa import block_attention

    fn = getattr(block_attention, kernel)
    consts = {k: v for k, v in consts.items() if k in fn.arg_names}
    sig, attrs = {}, {}
    for i, a in enumerate(fn.arg_names):
        if a in consts:
            sig[a] = "constexpr"
        elif a in FLOATS:
            sig[a] = "fp32"
        else:
            sig[a] = ("*bf16" if a in BF16 else "*fp32" if a in FP32 else "*i32") if a[0].isupper() else "i32"
            attrs[(i,)] = [["tt.divisibility", 16]]  # the most specialized class (aligned pointers, 16-multiple ints)
    options = {"num_warps": num_warps} | ({} if num_stages is None else {"num_stages": num_stages})
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs),
                       target=GPUTarget("cuda", 103, 32), options=options)  # fmt: skip
    return int(getattr(k.metadata, "shared", 0) or 0), int(getattr(k.metadata, "tmem_size", 0) or 0)


def _launches(fields, heads, kv_heads, head_dim):
    """(kernel, constexprs, num_warps, num_stages) of one forward + backward with ``fields`` over the sm_103 row."""
    from miles.kernels.attention.qsa.launch import TUNED

    group = heads // kv_heads
    base = TUNED[10]  # sm_103 has no row of its own and takes sm_100's
    over = dict(fields)
    fwd = {**base.fwd, **over.get("fwd", {})}
    dq = {**base.dq, **over.get("dq", {})}
    dkdv = {**base.dkdv, **over.get("dkdv", {})}
    query_rows = over.get("query_rows", base.query_rows)
    key_tile = over.get("key_tile", base.key_tile)
    t = min(32, query_rows // group)
    bm = max(16, triton.next_power_of_2(t * group))
    out = []
    for kernel, row in (("_fwd_kernel", fwd), ("_dq_kernel", dq)):
        consts = {"HQ": heads, "G": group, "T": t, "D": head_dim, "BLOCK_M": bm, "NU": row["NU"], "RATIO": 4}
        out.append((kernel, consts, row["num_warps"], row["num_stages"]))
    rows = dkdv.get("ROWS", 64)
    split = bool(over.get("heavy_split", base.heavy_split))
    consts = {"HQ": heads, "G": group, "TB": rows // group, "D": head_dim, "BLOCK_M": rows, "NUK": key_tile,
              "RATIO": 4, "DSPLIT": dkdv.get("DSPLIT", 1), "SPLIT": split}  # fmt: skip
    out.append(("_dkdv_kernel", consts, dkdv["num_warps"], dkdv["num_stages"]))
    if split:
        out.append(("_dkdv_reduce_kernel", {"HKV": kv_heads, "D": head_dim, "NUK": key_tile, "RATIO": 4}, 8, None))
    return out


def _requirements():
    gpu = _gpu_test_module()
    cases = {(heads, kv_heads, head_dim) for _, heads, kv_heads, head_dim, _ in gpu.CASES}
    out = [("table", {}, *shape) for shape in sorted(cases)]
    for name, fields in gpu.LAUNCH_VARIANTS.items():
        for _, heads, kv_heads, head_dim, _ in gpu.LAUNCH_CASES:
            out.append((name, fields, heads, kv_heads, head_dim))
    return out


def _skip_without_backend():
    try:
        from triton.backends.compiler import GPUTarget  # noqa: F401
        from triton.compiler import ASTSource  # noqa: F401
    except ImportError:
        pytest.skip("no Triton NVIDIA backend")


def test_every_gpu_test_launch_fits_b300():
    _skip_without_backend()
    bad, seen = [], {}
    for name, fields, heads, kv_heads, head_dim in _requirements():
        for kernel, consts, w, s in _launches(fields, heads, kv_heads, head_dim):
            key = (kernel, tuple(sorted(consts.items())), w, s)
            if key not in seen:
                seen[key] = _compile(kernel, consts, w, s)
            smem, tmem = seen[key]
            print(f"{name} {heads}/{kv_heads} d{head_dim} {kernel} w{w} s{s}: smem {smem} tmem {tmem}", flush=True)
            if smem > SMEM_BYTES or tmem > TMEM_COLUMNS:
                bad.append((name, heads, kv_heads, head_dim, kernel, smem, tmem))
    assert bad == [], f"launches that do not fit B300 (variant, heads, kv heads, head_dim, kernel, smem, tmem): {bad}"


def test_the_check_reproduces_the_gt_failure():
    """Positive control: NU 32 at the sm_100 row's num_stages 3, G 6, head_dim 256 needs 296992 bytes (GT 115)."""
    _skip_without_backend()
    smem, _ = _compile("_fwd_kernel", {"HQ": 6, "G": 6, "T": 10, "D": 256, "BLOCK_M": 64, "NU": 32, "RATIO": 4}, 8, 3)
    assert smem == 296992
