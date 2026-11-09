"""QSA block attention must match an fp64 reference built from the selected blocks, forward and backward; its
index must list every attended (query, block) pair exactly once, in both directions, in ascending order."""

import dataclasses
import random

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=180, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.qsa import build_qsa_block_index, qsa_block_attention  # noqa: E402
from miles.kernels.attention.qsa.launch import qsa_launch_override  # noqa: E402

RATIO = 4
TOL = 1e-2


def _cu(lens):
    return torch.tensor([0, *lens], device="cuda").cumsum(0).to(torch.int32)


def _select(lens, budget, seed, pad_rows=0):
    """Per query, up to ``budget`` random whole blocks of its own sequence, as packed block ids."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, first_block = [], 0
    for length in lens:
        n_blocks = -(-length // RATIO)
        pos = torch.arange(length, device="cuda")
        allowed = torch.arange(n_blocks, device="cuda")[None, :] < ((pos + 1) // RATIO)[:, None]
        scores = torch.rand(length, n_blocks, device="cuda", generator=g).masked_fill(~allowed, -1.0)
        top = scores.topk(min(budget, n_blocks), dim=1)
        blocks = torch.where(top.values >= 0, top.indices + first_block, -1)
        rows.append(torch.nn.functional.pad(blocks, (0, budget - blocks.shape[1]), value=-1))
        first_block += n_blocks
    # rows past the last sequence carry garbage the index must ignore
    rows.append(torch.randint(0, max(first_block, 1), (pad_rows, budget), device="cuda", generator=g))
    return torch.cat(rows).to(torch.int32)


def _keys(blocks, lens):
    """Key indices each query attends: its selected whole blocks, then its causal tail (``-1`` pad)."""
    parts, start, first_block = [], 0, 0
    for length in lens:
        b = blocks[start : start + length].long()
        first = torch.where(b >= 0, start + (b - first_block) * RATIO, -1)
        whole = torch.where(first[..., None] >= 0, first[..., None] + torch.arange(RATIO, device="cuda"), -1)
        pos = torch.arange(length, device="cuda")
        tail = ((pos + 1) // RATIO * RATIO)[:, None] + torch.arange(RATIO, device="cuda")
        tail = torch.where(tail <= pos[:, None], start + tail, -1)
        parts.append(torch.cat([whole.flatten(1), tail], dim=1))
        start += length
        first_block += -(-length // RATIO)
    return torch.cat(parts).to(torch.int32)


def _reference(q, k, v, keys, scale):
    """fp64 attention over exactly ``keys`` per row; a row with no key gives zeros."""
    n, n_keys = q.shape[0], k.shape[0]
    mask = torch.zeros(n, n_keys, dtype=torch.bool, device=q.device)
    rows = torch.arange(keys.shape[0], device=q.device)[:, None].expand_as(keys)
    valid = keys >= 0
    mask[rows[valid], keys[valid].long()] = True
    empty = ~mask.any(-1)
    group = q.shape[1] // k.shape[1]
    qh = q.double().transpose(0, 1)
    kh = k.double().transpose(0, 1).repeat_interleave(group, 0)
    vh = v.double().transpose(0, 1).repeat_interleave(group, 0)
    s = torch.einsum("htd,hsd->hts", qh, kh) * scale
    s = s.masked_fill(~mask, float("-inf")).masked_fill(empty[None, :, None], 0.0)
    p = torch.softmax(s, dim=-1) * (~empty)[None, :, None]
    return torch.einsum("hts,hsd->htd", p, vh).transpose(0, 1)


def _same_bits(a, b):
    """Bit-for-bit equality (torch.equal treats +0 and -0, and NaN payloads, by value)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _inputs(n, heads, kv_heads, head_dim, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(n, heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(n, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(n, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    gout = torch.randn(n, heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    return q, k, v, gout


def _run(fn, q, k, v, gout):
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    out = fn(*leaves)
    out.backward(gout)
    return [out.detach()] + [t.grad for t in leaves]


CASES = [
    # lens, query heads, kv heads, head_dim, budget in blocks
    ([384], 12, 1, 256, 16),  # Qwen3.8-Flash-Next at TP 2
    ([384], 24, 2, 256, 16),  # TP 1
    ([301, 211], 6, 1, 256, 16),  # TP 4, packed off the block grid
    ([61, 128], 3, 1, 256, 8),  # TP 8
    ([1, 1, 1, 1, 1, 200], 12, 1, 128, 8),
    ([37, 0, 13, 22], 4, 4, 128, 4),  # an empty sequence: a repeated boundary in cu_seqlens
    ([512], 12, 1, 256, 512),  # budget above the sequence: every whole block
    ([97], 12, 1, 256, 0),  # tails only; rows on a block's last token see nothing
]


@pytest.mark.parametrize("lens,heads,kv_heads,head_dim,budget", CASES)
def test_matches_reference(lens, heads, kv_heads, head_dim, budget):
    q, k, v, gout = _inputs(sum(lens), heads, kv_heads, head_dim, seed=sum(lens) + heads)
    blocks = _select(lens, budget, seed=len(lens))
    index = build_qsa_block_index(blocks, _cu(lens), compress_ratio=RATIO, num_heads=heads, num_kv_heads=kv_heads)
    scale = head_dim**-0.5
    got = _run(lambda a, b, c: qsa_block_attention(a, b, c, index, scale), q, k, v, gout)
    want = _run(lambda a, b, c: _reference(a, b, c, _keys(blocks, lens), scale), q, k, v, gout)
    for name, a, b in zip(("out", "dq", "dk", "dv"), got, want, strict=True):
        assert _rel(a, b) < TOL, (name, _rel(a, b))


def test_rows_past_the_last_sequence_see_nothing():
    lens, pad = [100, 61], 31
    q, k, v, gout = _inputs(sum(lens) + pad, 12, 1, 256, seed=7)
    blocks = _select(lens, 8, seed=7, pad_rows=pad)
    index = build_qsa_block_index(blocks, _cu(lens), compress_ratio=RATIO, num_heads=12, num_kv_heads=1)
    out, dq, dk, dv = _run(lambda a, b, c: qsa_block_attention(a, b, c, index, 256**-0.5), q, k, v, gout)
    n = sum(lens)
    for t in (out, dq, dk, dv):
        assert not t[n:].any()
    want = _run(lambda a, b, c: _reference(a, b, c, _keys(blocks[:n], lens), 256**-0.5), q[:n], k[:n], v[:n], gout[:n])
    for a, b in zip((out, dq, dk, dv), want, strict=True):
        assert _rel(a[:n], b) < TOL


def test_deterministic():
    lens = [3000, 1096]
    q, k, v, gout = _inputs(sum(lens), 12, 1, 256, seed=11)
    blocks = _select(lens, 512, seed=11)

    def attend(a, b, c):
        index = build_qsa_block_index(blocks, _cu(lens), compress_ratio=RATIO, num_heads=12, num_kv_heads=1)
        return qsa_block_attention(a, b, c, index, 256**-0.5)

    first = _run(attend, q, k, v, gout)
    second = _run(attend, q, k, v, gout)
    for a, b in zip(first, second, strict=True):
        assert _same_bits(a, b)


# launch shapes a B300 sweep may pick; each must stay within TOL of fp64 and be deterministic
LAUNCH_VARIANTS = {
    "nu4-w4-s2": {"fwd": {"NU": 4, "num_warps": 4, "num_stages": 2}, "dq": {"NU": 4, "num_warps": 4, "num_stages": 2}},
    "nu32": {"fwd": {"NU": 32}, "dq": {"NU": 32}},
    "query-rows-128": {"query_rows": 128},
    "key-tile-4": {"key_tile": 4},
    "key-tile-8-rows-32": {"key_tile": 8, "dkdv": {"ROWS": 32}},
    "key-rows-128": {"dkdv": {"ROWS": 128, "num_warps": 8}},
    "dsplit-2": {"dkdv": {"DSPLIT": 2}},
    "dsplit-4-w4": {"dkdv": {"DSPLIT": 4, "num_warps": 4, "num_stages": 2}},
    "busiest": {"key_order": "busiest"},
    "heavy-2": {"heavy_split": 2},
    "heavy-1-dsplit-2-busiest": {"heavy_split": 1, "dkdv": {"DSPLIT": 2}, "key_order": "busiest"},
}
LAUNCH_CASES = [([1500, 301, 211], 6, 1, 256, 512), ([384], 12, 1, 256, 16), ([61, 128], 3, 1, 256, 8)]


def _attend_with(blocks, lens, heads, kv_heads, scale, fields):
    def attend(a, b, c):
        with qsa_launch_override(**fields):
            index = build_qsa_block_index(
                blocks, _cu(lens), compress_ratio=RATIO, num_heads=heads, num_kv_heads=kv_heads
            )
            return qsa_block_attention(a, b, c, index, scale)

    return attend


@pytest.mark.parametrize("variant", LAUNCH_VARIANTS)
@pytest.mark.parametrize("lens,heads,kv_heads,head_dim,budget", LAUNCH_CASES, ids=["g6", "g12", "g3"])
def test_launch_variant_matches_reference_and_is_deterministic(variant, lens, heads, kv_heads, head_dim, budget):
    q, k, v, gout = _inputs(sum(lens), heads, kv_heads, head_dim, seed=sum(lens) + heads)
    blocks = _select(lens, budget, seed=len(lens))
    scale = head_dim**-0.5
    attend = _attend_with(blocks, lens, heads, kv_heads, scale, LAUNCH_VARIANTS[variant])
    got = _run(attend, q, k, v, gout)
    again = _run(attend, q, k, v, gout)
    want = _run(lambda a, b, c: _reference(a, b, c, _keys(blocks, lens), scale), q, k, v, gout)
    for name, a, b, c in zip(("out", "dq", "dk", "dv"), got, again, want, strict=True):
        assert _same_bits(a, b), (name, "not deterministic")
        assert torch.isfinite(a).all(), (name, "not finite")
        assert _rel(a, c) < TOL, (name, _rel(a, c))


def test_key_order_does_not_change_any_bit():
    """The dK / dV programs are independent; only their launch order changes."""
    lens, heads = [1500, 301, 211], 6
    q, k, v, gout = _inputs(sum(lens), heads, 1, 256, seed=5)
    blocks = _select(lens, 512, seed=5)
    by_position = _run(_attend_with(blocks, lens, heads, 1, 256**-0.5, {}), q, k, v, gout)
    busiest = _run(_attend_with(blocks, lens, heads, 1, 256**-0.5, {"key_order": "busiest"}), q, k, v, gout)
    for a, b in zip(by_position, busiest, strict=True):
        assert _same_bits(a, b)


# ---- index ----
_rng = random.Random(0)
PACKS = [
    [61, 128],
    [61, 61, 61, 200],
    [64, 0, 128],
    [1, 1, 1, 1, 1, 200],
    *[[_rng.randint(1, 300) for _ in range(_rng.randint(2, 6))] for _ in range(4)],
]


def _pair_case(lens, budget, heads, kv_heads, pad_rows=0):
    """Random causal whole-block selections, the attended (query, block) pairs, and the index."""
    g = torch.Generator().manual_seed(len(lens))
    rows, pairs, start, first_block = [], set(), 0, 0
    for length in lens:
        for pos in range(length):
            allowed = (pos + 1) // RATIO
            chosen = torch.randperm(allowed, generator=g)[:budget].tolist() if allowed else []
            rows.append(torch.tensor([first_block + b for b in chosen] + [-1] * (budget - len(chosen))))
            pairs |= {(start + pos, first_block + b) for b in chosen}
            if (pos + 1) % RATIO:
                pairs.add((start + pos, first_block + pos // RATIO))  # the causal tail
        start += length
        first_block += -(-length // RATIO)
    # rows past the last sequence carry garbage the index must ignore
    rows += [torch.randint(0, max(first_block, 1), (budget,), generator=g) for _ in range(pad_rows)]
    blocks = torch.stack(rows).to(torch.int32).cuda()
    index = build_qsa_block_index(blocks, _cu(lens), compress_ratio=RATIO, num_heads=heads, num_kv_heads=kv_heads)
    return index, pairs


def _decode(offsets, members, bits, width):
    """CSR tile lists with per-entry bit sets -> {(tile * width + bit, member)}; each list must ascend."""
    offsets, members, bits = offsets.tolist(), members.tolist(), bits.tolist()
    out = []
    for tile in range(len(offsets) - 1):
        listed = members[offsets[tile] : offsets[tile + 1]]
        assert listed == sorted(set(listed)), (tile, listed)
        for e in range(offsets[tile], offsets[tile + 1]):
            pattern = bits[e] & 0xFFFFFFFF
            assert pattern, (tile, e)
            out += [(tile * width + i, members[e]) for i in range(width) if pattern >> i & 1]
    return out


def _assert_lists(index, pairs):
    block_of_key = {s: b for b, s in enumerate(index.block_start.tolist())}
    by_query = _decode(index.q_offsets, index.q_key_start, index.q_bits, index.query_tile)
    forward = [(query, block_of_key[key]) for query, key in by_query]
    by_block = _decode(index.k_offsets, index.k_query, index.k_bits, index.key_tile)
    backward = [(query, block) for block, query in by_block]

    assert len(forward) == len(set(forward)) and set(forward) == pairs
    assert len(backward) == len(set(backward)) and set(backward) == pairs
    assert sorted(index.q_order.tolist()) == list(range(index.q_offsets.numel() - 1))
    assert sorted(index.k_order.tolist()) == list(range(index.k_offsets.numel() - 1))
    assert index.k_offsets.numel() - 1 == -(-index.block_start.numel() // index.key_tile)


@pytest.mark.parametrize("lens", PACKS)
@pytest.mark.parametrize("heads,kv_heads", [(12, 1), (6, 1), (3, 1), (2, 2)])
def test_index_lists_the_attended_pairs_once_in_order(lens, heads, kv_heads):
    _assert_lists(*_pair_case(lens, budget=5, heads=heads, kv_heads=kv_heads, pad_rows=7))



def _dense_case(lens, k, seed):
    """Production-budget selections: per query ``min(k, whole)`` distinct whole blocks of its own sequence, so the
    rows before ``k`` whole blocks select all of them (dense rows, whole query tiles of them) and later rows a
    stratified spread; with the attended (query, block) pairs, causal tails included."""
    g = torch.Generator().manual_seed(seed)
    rows, pairs, start, first = [], set(), 0, 0
    j = torch.arange(k)
    for length in lens:
        whole = ((torch.arange(length) + 1) // RATIO).unsqueeze(1)
        u = (torch.rand(length, 1, generator=g) * whole).long()
        ids = torch.where(whole >= k, (j * whole + u) // k, torch.where(j < whole, j, -1))
        part = torch.where(ids >= 0, ids + first, -1).to(torch.int32)
        rows.append(part)
        for pos, chosen in enumerate(part.tolist()):
            pairs |= {(start + pos, b) for b in chosen if b >= 0}
            if (pos + 1) % RATIO:
                pairs.add((start + pos, first + pos // RATIO))
        start += length
        first += -(-length // RATIO)
    return torch.cat(rows).cuda(), pairs


@pytest.mark.parametrize(
    "heads,kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1), (1, 1)], ids=["g12", "g12-tp2", "g6", "g3", "g1"]
)
def test_index_at_the_production_budget(heads, kv_heads):
    """K = 512 blocks per query, so the fill kernel's dense query tiles span 512-wide entry chunks. Every group gives
    a different query tile (5, 5, 10, 21, 32); group 3 (24 / 2 heads under TP8) once faulted there with a
    misaligned address."""
    lens = [2600, 700, 2049]
    blocks, pairs = _dense_case(lens, k=512, seed=heads)
    index = build_qsa_block_index(blocks, _cu(lens), compress_ratio=RATIO, num_heads=heads, num_kv_heads=kv_heads)
    _assert_lists(index, pairs)

def test_index_of_a_long_sequence():
    """Long enough that the prefix scans over the query tiles and over a key tile's bitmap take several chunks."""
    n = 33_000
    pos = torch.arange(n)
    whole = (pos + 1) // RATIO
    block = torch.where(whole > 0, (torch.rand(n, generator=torch.Generator().manual_seed(0)) * whole).long(), -1)
    tail = torch.where((pos + 1) % RATIO != 0, pos // RATIO, -1)
    pairs = {(q, b) for q, b in enumerate(block.tolist()) if b >= 0} | {
        (q, b) for q, b in enumerate(tail.tolist()) if b >= 0
    }
    blocks = block[:, None].to(torch.int32).cuda()
    _assert_lists(build_qsa_block_index(blocks, _cu([n]), compress_ratio=RATIO, num_heads=12, num_kv_heads=1), pairs)


def test_index_ignores_blocks_a_query_may_not_select():
    lens = [9, 9]  # blocks 0-2, then 3-5
    clean = torch.full((18, 4), -1, dtype=torch.int32)
    clean[15, 0] = 3  # row 15, position 6 of the second sequence, has one whole block before it
    dirty = clean.clone()
    dirty[15, 1:] = torch.tensor([0, 4, 5])  # the first sequence's, its own incomplete one, a later one
    a, b = (
        build_qsa_block_index(x.cuda(), _cu(lens), compress_ratio=RATIO, num_heads=12, num_kv_heads=1)
        for x in (clean, dirty)
    )
    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y, field.name


def test_index_cuts_blocks_from_each_sequence_start():
    lens = [6, 3, 9]
    index, _ = _pair_case(lens, budget=1, heads=12, kv_heads=1, pad_rows=2)
    assert index.block_start.tolist() == [0, 4, 6, 9, 13, 17]
    assert index.block_len.tolist() == [4, 2, 3, 4, 4, 1]
    assert index.last_key.tolist() == list(range(sum(lens))) + [-1, -1]
