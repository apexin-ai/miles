"""QSA recompute reuse on the real Triton kernels: the forward without autograd plus the replay give bitwise the
output and the q / k / v gradients of qsa_block_attention, and the index rebuilt from the same block ids is
bitwise the kept one (ops/qsa_recompute_reuse.py modes index / output rely on both).

One GPU. Shapes: the production per-rank head layout at TP4 (6 query heads on 1 KV head, head_dim 256), an irregular
13 / 1 group and 12 / 2 (two KV heads); packed sequences with rows past the last sequence; the production budget 512
blocks on a 16K row. A head layout the index refuses (13 / 2: heads not a multiple of KV heads) is a refusal case,
not a positive one; tests/fast/.../test_qsa_recompute_reuse.py checks on CPU that every positive case here is a
layout build_qsa_block_index accepts."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.qsa import (  # noqa: E402
    build_qsa_block_index,
    qsa_block_attention,
    qsa_block_attention_forward,
    qsa_block_attention_replay,
)
from miles_plugins.models.qwen3_8_next.ops.qsa_recompute_reuse import bitwise_equal  # noqa: E402

RATIO = 4


def _cu(lens):
    return torch.tensor([0, *lens], device="cuda").cumsum(0).to(torch.int32)


def _select(lens, budget, seed, pad_rows=0):
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
    rows.append(torch.randint(0, max(first_block, 1), (pad_rows, budget), device="cuda", generator=g))
    return torch.cat(rows).to(torch.int32)


CASES = [
    # lens, heads, kv_heads, head_dim, budget, pad_rows
    ((700, 333, 1024), 6, 1, 256, 64, 5),
    ((517, 2048), 13, 1, 64, 32, 0),
    ((900, 300), 12, 2, 128, 48, 3),
    ((16384,), 6, 1, 256, 512, 0),
]
REFUSED = [((64,), 13, 2)]  # heads not a multiple of KV heads: build_qsa_block_index asserts


def supported(heads: int, kv_heads: int, head_dim: int) -> bool:
    """What build_qsa_block_index / qsa_block_attention accept: heads a multiple of KV heads, a group of at most the
    64 tile rows, a power-of-two head_dim of at least 16."""
    return heads % kv_heads == 0 and heads // kv_heads <= 64 and head_dim >= 16 and head_dim & (head_dim - 1) == 0


@pytest.mark.parametrize("lens, heads, kv_heads, head_dim, budget, pad_rows", CASES)
def test_forward_plus_replay_is_bitwise_the_attention(lens, heads, kv_heads, head_dim, budget, pad_rows):
    torch.manual_seed(0)
    n = sum(lens) + pad_rows
    q = torch.randn(n, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    gout = torch.randn(n, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    blocks, cu = _select(lens, budget, seed=11, pad_rows=pad_rows), _cu(lens)
    scale = head_dim**-0.5

    def index():
        return build_qsa_block_index(blocks, cu, compress_ratio=RATIO, num_heads=heads, num_kv_heads=kv_heads)

    kept_index = index()
    assert bitwise_equal(index(), kept_index)  # mode index: the replay's rebuilt index equals the kept one

    def grads(fn):
        a, b, c = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        out = fn(a, b, c)
        out.backward(gout)
        return out.detach(), a.grad, b.grad, c.grad

    reference = grads(lambda a, b, c: qsa_block_attention(a, b, c, kept_index, scale))
    with torch.no_grad():  # the forward pass of a checkpointed layer
        kept_out, kept_lse = qsa_block_attention_forward(q, k, v, kept_index, scale)
    assert bitwise_equal(kept_out, reference[0])
    replayed = grads(lambda a, b, c: qsa_block_attention_replay(a, b, c, kept_index, scale, kept_out, kept_lse))
    for name, got, want in zip(("out", "dq", "dk", "dv"), replayed, reference):
        assert bitwise_equal(got, want), name
    # negative control: the replay's backward reads the kept lse (one row off by 0.5 changes dq)
    bad_lse = kept_lse.clone()
    bad_lse[sum(lens) // 2] += 0.5
    off = grads(lambda a, b, c: qsa_block_attention_replay(a, b, c, kept_index, scale, kept_out, bad_lse))
    assert not bitwise_equal(off[1], reference[1])


@pytest.mark.parametrize("lens, heads, kv_heads", REFUSED)
def test_an_unsupported_head_layout_is_refused_by_the_index(lens, heads, kv_heads):
    with pytest.raises(AssertionError):
        build_qsa_block_index(
            _select(lens, 8, 1), _cu(lens), compress_ratio=RATIO, num_heads=heads, num_kv_heads=kv_heads
        )


def test_replay_refuses_a_kept_output_of_another_shape():
    q = torch.randn(64, 6, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(64, 1, 64, device="cuda", dtype=torch.bfloat16)
    index = build_qsa_block_index(_select((64,), 8, 1), _cu((64,)), compress_ratio=RATIO, num_heads=6, num_kv_heads=1)
    out, lse = qsa_block_attention_forward(q, k, k, index, 0.125)
    with pytest.raises(ValueError, match="do not fit"):
        qsa_block_attention_replay(q, k, k, index, 0.125, out[:32], lse)
    with pytest.raises(ValueError, match="do not fit"):
        qsa_block_attention_replay(q, k, k, index, 0.125, out, lse.to(torch.bfloat16))
