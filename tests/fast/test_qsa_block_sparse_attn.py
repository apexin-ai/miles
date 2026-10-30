"""The tensor-core QSA kernel must match the gather kernel, including off-grid packed boundaries."""

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # pragma: no cover - needs a GPU
    pytest.skip("QSA kernels need CUDA", allow_module_level=True)

from miles.kernels.attention.qsa import (  # noqa: E402
    qsa_block_sparse_attention,
    qsa_sparse_attention,
    qsa_sparse_attention_from_indices,
)
from miles.kernels.attention.qsa.block_sparse_attention import (  # noqa: E402
    build_tile_index_pair,
    pack_block_bitmap,
    selection_to_block_bitmap,
)

BLK = 4
HQ, HKV, D = 4, 1, 128
SCALE = D**-0.5


def _single_sequence_case(T=384, budget=64, seed=0):
    """Selection as the indexer emits it: distinct blocks, tail clamped to the position."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(T, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    rows = torch.arange(T, device="cuda")
    nblk = budget // BLK
    scores = torch.rand(T, T // BLK + 1, device="cuda", generator=g)
    allowed = torch.arange(scores.shape[1], device="cuda").unsqueeze(0) <= (rows // BLK).unsqueeze(1)
    scores = torch.where(allowed, scores, torch.full_like(scores, -1.0))
    blk = scores.topk(min(nblk, scores.shape[1]), dim=1).indices
    tok = (blk.unsqueeze(-1) * BLK + torch.arange(BLK, device="cuda")).reshape(T, -1)
    idx = torch.where(tok <= rows.unsqueeze(1), tok, torch.full_like(tok, -1)).to(torch.int32)
    return q, k, v, idx


def _reference(q, k, v, idx, scale):
    """fp64, one query at a time, straight from the documented semantics."""
    T, hq, d = q.shape
    group = hq // k.shape[1]
    out = torch.zeros(T, hq, d, device=q.device, dtype=torch.float64)
    qd, kd, vd = q.double(), k.double(), v.double()
    for t in range(T):
        sel = idx[t][idx[t] >= 0].long()
        if sel.numel() == 0:
            continue
        for h in range(hq):
            s = (qd[t, h].unsqueeze(0) * kd[sel, h // group]).sum(-1) * scale
            p = torch.softmax(s, dim=0)
            out[t, h] = (p.unsqueeze(-1) * vd[sel, h // group]).sum(0)
    return out


def _rel(a, b):
    d = (a.double() - b.double()).abs()
    return d.max().item(), (d.mean() / (b.double().abs().mean() + 1e-12)).item()


def test_matches_gather_kernel_and_reference():
    q, k, v, idx = _single_sequence_case()
    ref = _reference(q, k, v, idx, SCALE)
    old = qsa_sparse_attention(q, k, v, idx, SCALE)
    new = qsa_sparse_attention_from_indices(q, k, v, idx, SCALE)

    old_max, _ = _rel(old, ref)
    new_max, new_rel = _rel(new, ref)
    # neither is closer to exact than the other by more than the bf16 floor
    assert new_max <= old_max * 4, (new_max, old_max)
    assert new_rel < 1e-2, new_rel
    assert _rel(new, old)[1] < 1e-2


def test_gradients_match_gather_kernel():
    q, k, v, idx = _single_sequence_case()
    gout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16)

    def run(fn):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        fn(qq, kk, vv, idx, SCALE).backward(gout)
        return qq.grad, kk.grad, vv.grad

    for a, b, name in zip(run(qsa_sparse_attention_from_indices), run(qsa_sparse_attention), "qkv", strict=True):
        rel = _rel(a, b)[1]
        assert rel < 1e-2, (name, rel)


@pytest.mark.parametrize("lens", [[301, 211], [61, 128]])
def test_packed_boundary_not_a_multiple_of_the_block(lens):
    """A sequence starting off the global block grid must not borrow its neighbour's blocks."""
    T = sum(lens)
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(T, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)

    cu = torch.tensor([0, *lens], device="cuda").cumsum(0)
    tok_base = torch.zeros(T, dtype=torch.int32, device="cuda")
    blk_base = torch.zeros(T, dtype=torch.int32, device="cuda")
    pos = torch.zeros(T, dtype=torch.int64, device="cuda")
    block_first, block_last = [], []
    boff = 0
    for i, length in enumerate(lens):
        s0 = int(cu[i])
        tok_base[s0 : s0 + length] = s0
        blk_base[s0 : s0 + length] = boff
        pos[s0 : s0 + length] = torch.arange(length, device="cuda")
        boff += -(-length // BLK)
        block_first += list(range(s0, s0 + length, BLK))
        block_last += [min(b + BLK, s0 + length) - 1 for b in range(s0, s0 + length, BLK)]
    block_first = torch.tensor(block_first, dtype=torch.int32, device="cuda")
    block_last = torch.tensor(block_last, dtype=torch.int32, device="cuda")

    sel = torch.zeros(T, boff, dtype=torch.uint8, device="cuda")
    idx = torch.full((T, 64), -1, dtype=torch.int32, device="cuda")
    for t in range(T):
        p = int(pos[t])
        cand = torch.arange((p + 1) // BLK, device="cuda")
        chosen = cand[torch.randperm(cand.numel(), device="cuda", generator=g)[:7]] if cand.numel() else cand
        blocks = torch.cat([chosen, torch.tensor([p // BLK], device="cuda")]).unique()
        sel[t, int(blk_base[t]) + blocks] = 1
        toks = (blocks.unsqueeze(1) * BLK + torch.arange(BLK, device="cuda")).reshape(-1) + int(tok_base[t])
        toks = toks[(toks >= int(tok_base[t])) & (toks <= t)]
        idx[t, : toks.numel()] = toks.to(torch.int32)

    hi = torch.arange(T, dtype=torch.int32, device="cuda")
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = qsa_block_sparse_attention(
        qq, kk, vv, pack_block_bitmap(sel), tok_base, hi, blk_base, tok_base, block_first, block_last, SCALE, BLK
    )
    ref = _reference(q, k, v, idx, SCALE)
    assert _rel(out, ref)[1] < 1e-2
    # the second sequence is the one a global grid would corrupt
    assert _rel(out[lens[0] :], ref[lens[0] :])[1] < 1e-2

    gout = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16, generator=g)
    out.backward(gout)
    ref_grads = [t.clone().requires_grad_(True) for t in (q, k, v)]
    qsa_sparse_attention(*ref_grads, idx, SCALE).backward(gout)
    for got, want, name in zip((qq, kk, vv), ref_grads, "qkv", strict=True):
        assert _rel(got.grad, want.grad)[1] < 1e-2, name


def test_tile_index_pair_is_consistent_both_ways():
    q, k, v, idx = _single_sequence_case(T=256, budget=32)
    sel = selection_to_block_bitmap(idx, 256, BLK)
    block_first = torch.arange(256 // BLK, dtype=torch.int32, device="cuda") * BLK
    klist, kcnt, qlist, qcnt = build_tile_index_pair(sel, block_first, block_first + BLK - 1, 64, 32)

    forward = {(qt, int(klist[qt, i])) for qt in range(kcnt.numel()) for i in range(int(kcnt[qt]))}
    backward = {(int(qlist[kt, i]), kt) for kt in range(qcnt.numel()) for i in range(int(qcnt[kt]))}
    assert forward == backward


def test_bitmap_rows_past_int32_byte_offsets():
    """262K packed tokens (247145 + 14999): the last rows' bitmap offsets pass 2^31 bytes.

    Forward and dq of those rows must match an fp64 reference built from the same bitmap.
    """
    lens = [247145, 14999]
    T = sum(lens)
    hq, hkv, d = 1, 1, 64
    g = torch.Generator(device="cuda").manual_seed(7)
    q = torch.randn(T, hq, d, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, hkv, d, device="cuda", dtype=torch.bfloat16, generator=g)

    starts = torch.tensor([0, lens[0]], device="cuda")
    nblk = [-(-n // BLK) for n in lens]
    rows = torch.arange(T, device="cuda")
    doc = (rows >= lens[0]).long()
    tok_base = starts[doc].to(torch.int32)
    blk_base = torch.tensor([0, nblk[0]], device="cuda")[doc].to(torch.int32)
    pos = rows - tok_base
    num_blocks = sum(nblk)
    firsts, lasts = [], []
    for s0, n in zip([0, lens[0]], lens, strict=True):
        f = torch.arange(s0, s0 + n, BLK, device="cuda")
        firsts.append(f)
        lasts.append((f + BLK).clamp_max(s0 + n) - 1)
    block_first, block_last = torch.cat(firsts), torch.cat(lasts)

    # each row: its own block plus up to 3 random earlier blocks of its document
    own = blk_base.long() + pos // BLK
    picks = [own] + [blk_base.long() + (torch.rand(T, device="cuda", generator=g) * (pos // BLK + 1)).long() for _ in range(3)]
    nbytes = -(-num_blocks // 8)
    sel = torch.zeros(T, nbytes, dtype=torch.uint8, device="cuda")
    for b in picks:
        sel[rows, b >> 3] |= (1 << (b & 7)).to(torch.uint8)
    assert (T - 1) * nbytes > 2**31

    hi = rows.to(torch.int32)
    qq = q.clone().requires_grad_(True)
    out = qsa_block_sparse_attention(
        qq, k, v, sel, tok_base, hi, blk_base, tok_base, block_first.to(torch.int32), block_last.to(torch.int32), SCALE, BLK
    )
    tail = torch.arange(T - 40, T, device="cuda")
    gout = torch.zeros_like(out)
    gout[tail] = torch.randn(40, hq, d, device="cuda", dtype=out.dtype, generator=g)
    out.backward(gout)

    for t in tail.tolist():
        keys = torch.arange(int(tok_base[t]), t + 1, device="cuda")
        kb = blk_base[t].long() + (keys - tok_base[t]) // BLK
        keys = keys[((sel[t, kb >> 3] >> (kb & 7).to(torch.uint8)) & 1) != 0]
        qt = q[t, 0].double().requires_grad_(True)
        s = k[keys, 0].double() @ qt * SCALE
        ref = torch.softmax(s, 0) @ v[keys, 0].double()
        assert (out[t, 0].double() - ref).abs().max() < 2e-2, t
        ref.backward(gout[t, 0].double())
        assert (qq.grad[t, 0].double() - qt.grad).abs().max() < 2e-2 * max(1.0, qt.grad.abs().max().item()), t
