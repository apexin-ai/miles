"""QSA block attention under every launch knob, on CPU through the Triton interpreter, against fp64.

The GPU suite (``tests/fast-gpu/kernels/attention/qsa/test_qsa_block_attention.py``) checks the compiled kernels;
this runs the same kernels' logic without a GPU: every launch shape the sweep may pick (``NU``, ``query_rows``,
``key_tile``, dK / dV ``ROWS`` and ``DSPLIT``, ``key_order``) must give the fp64 attention and its gradients over
exactly the selected blocks. The interpreter does not handle bfloat16 (numpy has none), so the inputs are fp32;
that checks which (query, key) pairs each program sums, not tensor-core rounding.

``TRITON_INTERPRET`` is read when a kernel is decorated, so the cases run in a child interpreter.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

LAUNCHES = {
    "table": {},
    "nu1-nu2": {"fwd": {"NU": 1}, "dq": {"NU": 2}},
    "nu32": {"fwd": {"NU": 32}, "dq": {"NU": 32}},
    "query-rows-128": {"query_rows": 128},
    "query-rows-16": {"query_rows": 16},
    "key-tile-1": {"key_tile": 1},
    "key-tile-32": {"key_tile": 32},
    "key-rows-16": {"dkdv": {"ROWS": 16}},
    "key-rows-128": {"dkdv": {"ROWS": 128}},
    "dsplit-2": {"dkdv": {"DSPLIT": 2}},
    "busiest": {"key_order": "busiest"},
    "heavy-1": {"heavy_split": 1},
    "heavy-1-dsplit-busiest": {"heavy_split": 1, "dkdv": {"DSPLIT": 2}, "key_order": "busiest"},
    "heavy-3-key-tile-1": {"heavy_split": 3, "key_tile": 1},
    "all": {
        "fwd": {"NU": 4},
        "dq": {"NU": 4},
        "query_rows": 128,
        "key_tile": 8,
        "dkdv": {"ROWS": 32, "DSPLIT": 2},
        "key_order": "busiest",
        "heavy_split": 2,
    },
}
# lens (one packed batch), query heads, kv heads, blocks per query
SHAPES = {
    "tp4-packed": ([37, 0, 50, 9], 6, 1, 6),  # Qwen3.8-Flash-Next at TP4: G = 6
    "two-kv-heads": ([70], 4, 2, 5),
    "dense": ([45], 3, 1, 64),  # budget above the blocks: every row selects every whole block
    "sink": ([160], 6, 1, 3),  # every query with a whole block selects block 0: one key tile far above the mean
}
RATIO = 4
HEAD_DIM = 32
TOL = 1e-4


def _cases():
    import torch

    from miles.kernels.attention.qsa import block_attention, build_qsa_block_index, qsa_block_attention
    from miles.kernels.attention.qsa.launch import qsa_launch_override

    launched = {}

    class Recorder:
        """Records each kernel's grid and launch keywords, so a knob that never reaches a kernel fails."""

        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            def run(*args, **kw):
                launched[self.name] = {"grid": list(grid), **{k: v for k, v in kw.items() if k != "D"}}
                return self.kernel[grid](*args, **kw)

            return run

    for name in ("_fwd_kernel", "_dq_kernel", "_dkdv_kernel", "_dkdv_reduce_kernel"):
        setattr(block_attention, name, Recorder(name, getattr(block_attention, name)))

    def select(lens, budget, g, sink=False):
        rows, first = [], 0
        for length in lens:
            n_blocks = -(-length // RATIO)
            for pos in range(length):
                whole = (pos + 1) // RATIO
                chosen = (torch.randperm(whole, generator=g)[:budget] + first).tolist() if whole else []
                if sink and whole and first not in chosen:
                    chosen[-1] = first
                rows.append(chosen + [-1] * (budget - len(chosen)))
            first += n_blocks
        rows += [[0] * budget] * 3  # rows past the last sequence: garbage the index must ignore
        return torch.tensor(rows, dtype=torch.int32)

    def keys_of(blocks, lens):
        out, start, first = [], 0, 0
        for length in lens:
            for pos in range(length):
                ks = [start + (b - first) * RATIO + j for b in blocks[start + pos].tolist() if b >= 0 for j in range(4)]
                ks += [start + t for t in range((pos + 1) // RATIO * RATIO, pos + 1)]
                out.append(ks)
            start += length
            first += -(-length // RATIO)
        return out + [[] for _ in range(blocks.shape[0] - start)]

    def reference(q, k, v, keys, scale):
        group = q.shape[1] // k.shape[1]
        rows = []
        for i, ks in enumerate(keys):
            if not ks:
                rows.append(torch.zeros(q.shape[1:], dtype=torch.float64))
                continue
            kk = k[ks].repeat_interleave(group, 1).transpose(0, 1)  # [H, n, D]
            vv = v[ks].repeat_interleave(group, 1).transpose(0, 1)
            s = torch.einsum("hd,hnd->hn", q[i], kk) * scale
            rows.append(torch.einsum("hn,hnd->hd", torch.softmax(s, -1), vv))
        return torch.stack(rows)

    def _programs_cover_each_list_once(index):
        """heavy_split programs: per tile, consecutive [lo, hi) ranges covering its list; slots of cut tiles
        numbered in chunk order from the tile's r_slot0, -1 for whole tiles."""
        ok = True
        tiles = index.k_prog_tile.tolist()
        lo, hi, slot = index.k_prog_lo.tolist(), index.k_prog_hi.tolist(), index.k_prog_slot.tolist()
        offs = index.k_offsets.tolist()
        firsts = zip(index.r_slot0.tolist(), index.r_count.tolist(), strict=True)
        cut = dict(zip(index.r_tile.tolist(), firsts, strict=True))
        for t in range(len(offs) - 1):
            mine = [i for i, x in enumerate(tiles) if x == t]
            ranges = [(lo[i], hi[i]) for i in mine]
            ok &= ranges[0][0] == offs[t] and ranges[-1][1] == offs[t + 1]
            ok &= all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
            if t in cut:
                s0, n = cut[t]
                ok &= len(mine) == n > 1 and [slot[i] for i in mine] == list(range(s0, s0 + n))
            else:
                ok &= len(mine) == 1 and slot[mine[0]] == -1
        return ok

    results = {}
    for shape_name, (lens, hq, hkv, budget) in SHAPES.items():
        g = torch.Generator().manual_seed(len(lens) * 100 + hq)
        blocks = select(lens, budget, g, sink=shape_name == "sink")
        n = blocks.shape[0]
        cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32)
        q, k, v, gout = (torch.randn(n, h, HEAD_DIM, generator=g) for h in (hq, hkv, hkv, hq))
        scale = HEAD_DIM**-0.5
        leaves = [t.double().requires_grad_(True) for t in (q, k, v)]
        reference(*leaves, keys_of(blocks, lens), scale).backward(gout.double())
        want = [None] + [t.grad for t in leaves]
        want[0] = reference(*(t.detach() for t in leaves), keys_of(blocks, lens), scale)
        for launch_name, fields in LAUNCHES.items():
            launched.clear()
            with qsa_launch_override(**fields):
                index = build_qsa_block_index(blocks, cu, compress_ratio=RATIO, num_heads=hq, num_kv_heads=hkv)
                if index.k_prog_tile is None:
                    counts = (index.k_offsets[1:] - index.k_offsets[:-1])[index.k_order.long()]
                    order_ok = sorted(index.k_order.tolist()) == list(range(index.k_offsets.numel() - 1))
                else:
                    counts = (index.k_prog_hi - index.k_prog_lo)[index.k_order.long()]
                    order_ok = sorted(index.k_order.tolist()) == list(range(index.k_prog_tile.numel()))
                    order_ok &= _programs_cover_each_list_once(index)
                if fields.get("key_order") == "busiest":
                    order_ok &= bool((counts[:-1] >= counts[1:]).all())
                ins = [t.clone().requires_grad_(True) for t in (q, k, v)]
                out = qsa_block_attention(*ins, index, scale)
            # the backward runs outside the override and must launch as its forward did
            out.backward(gout)
            got = [out.detach()] + [t.grad for t in ins]
            errs = {
                name: ((a.double() - b).norm() / b.norm().clamp_min(1e-30)).item()
                for name, a, b in zip(("out", "dq", "dk", "dv"), got, want, strict=True)
            }
            past = all(not t[sum(lens) :].any() for t in got)
            results[f"{shape_name}/{launch_name}"] = {
                "errs": errs,
                "order_ok": order_ok,
                "past_rows_zero": past,
                "query_tile": index.query_tile,
                "key_tile": index.key_tile,
                "launched": dict(launched),
                "cut_tiles": 0 if index.r_tile is None else index.r_tile.numel(),
            }
    return results


@pytest.fixture(scope="module")
def results():
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.run(
        [sys.executable, "-B", __file__], env=env, cwd=ROOT, capture_output=True, text=True, timeout=1200
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("launch", LAUNCHES)
@pytest.mark.parametrize("shape", SHAPES)
def test_every_launch_matches_fp64(results, shape, launch):
    r = results[f"{shape}/{launch}"]
    assert r["order_ok"], r
    assert r["past_rows_zero"], r
    assert max(r["errs"].values()) < TOL, r


@pytest.mark.parametrize("launch", LAUNCHES)
def test_every_knob_reaches_its_kernel(results, launch):
    """Positive control: the launch each kernel got is the table's sm_90 row (CPU) with the override on top."""
    fields = LAUNCHES[launch]
    got = results[f"tp4-packed/{launch}"]["launched"]
    for kernel, part, default in (
        ("_fwd_kernel", "fwd", {"NU": 8, "num_warps": 4, "num_stages": 3}),
        ("_dq_kernel", "dq", {"NU": 8, "num_warps": 4, "num_stages": 1}),
    ):
        for key, value in {**default, **fields.get(part, {})}.items():
            assert got[kernel][key] == value, (kernel, key)
    dkdv = {"num_warps": 4, "num_stages": 3, "DSPLIT": 1, "ROWS": 64, **fields.get("dkdv", {})}
    k = got["_dkdv_kernel"]
    assert (k["num_warps"], k["num_stages"], k["DSPLIT"], k["grid"][2]) == (
        dkdv["num_warps"],
        dkdv["num_stages"],
        dkdv["DSPLIT"],
        dkdv["DSPLIT"],
    )
    assert (k["BLOCK_M"], k["TB"], k["NUK"]) == (dkdv["ROWS"], dkdv["ROWS"] // 6, fields.get("key_tile", 4))


@pytest.mark.parametrize("launch", [name for name, f in LAUNCHES.items() if f.get("heavy_split")])
def test_heavy_split_cuts_tiles_and_reduces_them(results, launch):
    """Positive control: the sink shape does cut tiles, the reduce kernel ran on them, and without heavy_split
    nothing is cut."""
    r = results[f"sink/{launch}"]
    assert r["cut_tiles"] > 0
    assert r["launched"]["_dkdv_reduce_kernel"]["grid"][0] == r["cut_tiles"]
    assert results["sink/table"]["cut_tiles"] == 0
    assert "_dkdv_reduce_kernel" not in results["sink/table"]["launched"]


def test_the_knobs_reach_the_index(results):
    assert results["tp4-packed/table"]["query_tile"] == 10  # 64 rows / 6 heads
    assert results["tp4-packed/query-rows-128"]["query_tile"] == 21
    assert results["tp4-packed/key-tile-32"]["key_tile"] == 32
    assert results["tp4-packed/table"]["key_tile"] == 4  # the CPU (sm_90) row


if __name__ == "__main__":
    print(json.dumps(_cases()))
