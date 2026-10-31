"""262K packed QSA attention through the selected-block index: forward and backward against fp64.

The batch is the CRV-FIX-013 case, documents of 247145 and 14999 tokens. Each query selects
K = 512 distinct whole blocks (budget 2048 / ratio 4) spread over its whole document, so the
index lists cover the full 262K block grid. Gradients flow only from the last rows of each
document; out, dq, dk and dv must match an fp64 reference built from the same block ids.

The old bitmap kernel crossed 2^31 at ~92.7K tokens through its ``[T, blocks]`` row offsets.
At these head counts no Q / K row offset of the new kernel reaches 2^31; the CPU source check
``tests/fast/test_qsa_block_kernel_int64_offsets.py`` pins the int64 widening, and this test
checks the full path at the production length. Run with ``-s`` to see the peaks.
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

RATIO = 4
K = 512
LENS = [247145, 14999]
CHECK = 40  # rows with a gradient at the end of each document
GiB = 1 << 30


def _blocks(lens, k, seed):
    """``[T, k]`` int32: per query ``min(k, whole)`` distinct whole blocks of its own document, ``-1`` pad.

    Row ids are stratified, ``(j * whole + u) // k`` for ``j < k``: one per disjoint interval, so they
    are distinct without a sort, and they reach the far start of the document.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    parts, first = [], 0
    j = torch.arange(k, device="cuda")
    for length in lens:
        whole = ((torch.arange(length, device="cuda") + 1) // RATIO).unsqueeze(1)
        u = (torch.rand(length, 1, device="cuda", generator=g) * whole).long()
        spread = (j * whole + u) // k
        ids = torch.where(whole >= k, spread, torch.where(j < whole, j, -1))
        parts.append(torch.where(ids >= 0, ids + first, -1).to(torch.int32))
        first += -(-length // RATIO)
    return torch.cat(parts)


def _keys_of_row(blocks, row, doc_start, doc_first_block):
    """Key indices the kernel attends for ``row``: its selected whole blocks, then its causal tail."""
    ids = blocks[row][blocks[row] >= 0].long() - doc_first_block
    whole = (doc_start + ids * RATIO).unsqueeze(1) + torch.arange(RATIO, device="cuda")
    pos = row - doc_start
    tail = torch.arange(doc_start + (pos + 1) // RATIO * RATIO, row + 1, device="cuda")
    return torch.cat([whole.flatten(), tail])


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _peak(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    return out, (torch.cuda.max_memory_allocated() - base) / GiB


@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (12, 1), (6, 1), (3, 1)], ids=["TP1", "TP2", "TP4", "TP8"])
def test_long_packed_matches_fp64_at_the_far_end(heads, kv_heads):
    from miles.kernels.attention.qsa import build_qsa_block_index, qsa_block_attention

    head_dim, total = 256, sum(LENS)
    scale = head_dim**-0.5
    g = torch.Generator(device="cuda").manual_seed(heads)
    q = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(total, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(total, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    blocks = _blocks(LENS, K, seed=heads)
    cu = torch.tensor([0, *LENS], device="cuda").cumsum(0).to(torch.int32)

    index, index_peak = _peak(
        lambda: build_qsa_block_index(blocks, cu, compress_ratio=RATIO, num_heads=heads, num_kv_heads=kv_heads)
    )

    doc_start = [0, LENS[0]]
    doc_first_block = [0, -(-LENS[0] // RATIO)]
    rows = [(d, r) for d in range(2) for r in range(doc_start[d] + LENS[d] - CHECK, doc_start[d] + LENS[d])]
    gout = torch.zeros_like(q)
    row_ids = torch.tensor([r for _, r in rows], device="cuda")
    gout[row_ids] = torch.randn(len(rows), heads, head_dim, device="cuda", dtype=q.dtype, generator=g)

    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]

    def fwd_bwd():
        out = qsa_block_attention(*leaves, index, scale)
        out.backward(gout)
        return out.detach()

    out, attn_peak = _peak(fwd_bwd)
    dq, dk, dv = (t.grad for t in leaves)

    group = heads // kv_heads
    kd = k.double().requires_grad_(True)
    vd = v.double().requires_grad_(True)
    want_out, want_dq = [], []
    for d, row in rows:
        keys = _keys_of_row(blocks, row, doc_start[d], doc_first_block[d])
        qd = q[row].double().requires_grad_(True)
        kk = kd[keys].repeat_interleave(group, 1)  # [n, heads, D]
        vv = vd[keys].repeat_interleave(group, 1)
        p = torch.softmax(torch.einsum("hd,nhd->hn", qd, kk) * scale, dim=-1)
        o = torch.einsum("hn,nhd->hd", p, vv)
        o.backward(gout[row].double())
        want_out.append(o.detach())
        want_dq.append(qd.grad)
    print(
        f"\nheads={heads}/{kv_heads} T={total} index_build_peak={index_peak:.2f}GiB "
        f"attention_fwd_bwd_peak={attn_peak:.2f}GiB q_list={index.q_key_start.numel()} k_list={index.k_query.numel()}"
    )
    assert _rel(out[row_ids], torch.stack(want_out)) < 1e-2
    assert _rel(dq[row_ids], torch.stack(want_dq)) < 1e-2
    # only the checked rows carry a gradient, so dk / dv are theirs alone, over the whole batch
    assert _rel(dk, kd.grad) < 1e-2
    assert _rel(dv, vd.grad) < 1e-2
    other = torch.ones(total, dtype=torch.bool, device="cuda")
    other[row_ids] = False
    assert not dq[other].any()
