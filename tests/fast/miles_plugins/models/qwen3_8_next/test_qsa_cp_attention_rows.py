"""QSA attention under allgather-CP: this rank's query / key rows get their own RoPE and attend as global rows.

Under ``--allgather-cp`` a CP rank holds the contiguous rows ``[lo, hi)`` of the packed stream, cu_seqlens stays
global. Two consumers inside Megatron's SelfAttention.forward read the CP layout on their own:

* RoPE. ``apply_rotary_pos_emb(cp_group=pg_collection.cp)`` maps a THD token to its position with the zigzag split
  (``rope_utils._apply_rotary_pos_emb_thd``; the TE and Triton fused kernels take ``cp_size`` / ``cp_rank`` the same
  way): on a contiguous slice every rank but the first gets wrong positions, silently. Qwen38NextAttention therefore
  hands SelfAttention this rank's rows of the frequencies, a one-segment packing of its rows and a CP group of one,
  so RoPE reads exactly the rows the CP1 run reads for these tokens (the negative control below shows the stock call
  does not).
* The QSA block index. ``build_qsa_block_index`` takes query row ``i`` to be packed row ``i`` of ``cu_seqlens``; a
  rank's rows start at ``lo``. The core attention pads the queries and block ids to the whole stream (rows outside
  the window select no block and get zero queries, whose output is dropped), so the kernel sees global rows.

The kernels are Triton; the block-attention test runs a torch reference with the same row contract. CPU only.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer.attention import SelfAttention
from miles_plugins.models.qwen3_8_next.ops import attention
from miles_plugins.models.qwen3_8_next.ops.attention import Qwen38NextAttention, Qwen38NextQSACoreAttention
from miles_plugins.models.qwen3_8_next.ops.cp import cp_local_packed_seq_params, cp_local_rope

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["miles-plugin"])

HEAD_DIM = 16
MROPE = [4, 2, 2]  # rotary_dim / 2 = 8 -> rotary_dim 16


class _Group:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _config(mrope=True):
    return SimpleNamespace(
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=False,
        mrope_section=list(MROPE) if mrope else None,
        mrope_interleaved=False,
        sequence_parallel=False,
        qwen3_8_next_allgather_cp=True,
        qwen3_8_next_materialized_rope=mrope,
        qwen3_8_next_indexer_tp_shard=False,
    )


def _packed(cu):
    from miles.backends.megatron_utils.parallel import PackedSeqParamsWithHostCuSeqlens

    lengths = [b - a for a, b in zip(cu, cu[1:])]
    return PackedSeqParamsWithHostCuSeqlens(
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(cu, dtype=torch.int32),
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        qkv_format="thd",
        cu_seqlens_host=tuple(cu),
        total_tokens=cu[-1],
        sample_count=len(cu) - 1,
    )


def _rope(kind, cu):
    """The rotary_pos_emb GPTModel hands every layer for this pack: raw mRoPE freqs, the materialized mRoPE table
    (both per token, image-like positions), or a text RoPE table indexed by in-sequence position."""
    total = cu[-1]
    inv_freq = 1.0 / (1e4 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    if kind == "text":
        freqs = torch.outer(torch.arange(max(b - a for a, b in zip(cu, cu[1:])), dtype=torch.float32), inv_freq)
        return torch.cat((freqs, freqs), -1)[:, None, None, :]
    torch.manual_seed(3)
    positions = torch.randint(0, 64, (3, 1, total)).float()  # three distinct axes, as image tokens have
    raw = (positions[..., None] * inv_freq[: sum(MROPE)]).contiguous()  # [3, 1, T, rotary_dim / 2]
    if kind == "raw":
        return raw
    from megatron.core.fusions.fused_mrope import mrope_freqs_to_rotary_emb

    return mrope_freqs_to_rotary_emb(raw, list(MROPE), interleaved_mrope=False, rotary_interleaved=False)


CUS = [(0, 32), (0, 13, 32), (0, 5, 20, 27, 32)]


def _apply(t, freqs, config, packed, group):
    return apply_rotary_pos_emb(
        t, freqs, config=config, cu_seqlens=packed.cu_seqlens_q, cp_group=group, max_seqlen=packed.max_seqlen_q
    )


@pytest.mark.parametrize("kind", ["raw", "materialized", "text"])
@pytest.mark.parametrize("cu", CUS, ids=str)
@pytest.mark.parametrize("cp_size", [2, 4])
def test_local_rope_rows_equal_the_cp1_rope_rows(kind, cu, cp_size):
    config = _config(mrope=kind != "text")
    total = cu[-1]
    freqs, packed = _rope(kind, cu), _packed(list(cu))
    torch.manual_seed(0)
    t = torch.randn(total, 2, HEAD_DIM)
    whole = _apply(t, freqs, config, packed, _Group(1, 0))
    positions = Qwen38NextAttention._packed_positions(packed.cu_seqlens_q, total)
    local = total // cp_size
    ours, stock = [], []
    for rank in range(cp_size):
        lo, hi = rank * local, (rank + 1) * local
        rows = cp_local_rope(freqs, positions, (lo, hi), config)
        ours.append(_apply(t[lo:hi], rows, config, cp_local_packed_seq_params(packed, hi - lo), _Group(1, 0)))
        stock.append(_apply(t[lo:hi], freqs, config, packed, _Group(cp_size, rank)))
    assert torch.equal(torch.cat(ours), whole)
    # negative control: Megatron's own CP RoPE reads the zigzag split, wrong on a contiguous slice
    assert not torch.allclose(torch.cat(stock), whole)


def test_a_tuple_of_query_and_key_rope_is_sliced_per_element():
    config = _config()
    freqs = _rope("materialized", (0, 16))
    positions = torch.arange(16)
    q, k = cp_local_rope((freqs, freqs * 2), positions, (8, 16), config)
    assert torch.equal(q, freqs[8:16]) and torch.equal(k, (freqs * 2)[8:16])


def test_the_local_packing_is_one_segment_of_this_rank_s_rows():
    packed = _packed([0, 13, 32])
    local = cp_local_packed_seq_params(packed, 16)
    assert local.cu_seqlens_q.tolist() == [0, 16] and local.cu_seqlens_kv.tolist() == [0, 16]
    assert local.max_seqlen_q == local.max_seqlen_kv == 16 and local.cu_seqlens_host == (0, 16)
    assert packed.cu_seqlens_q.tolist() == [0, 13, 32]  # the layer's own packing is not touched


# ── Qwen38NextAttention.forward hands SelfAttention the local rows ───────────────────────────────────────


def _attention_layer(config, cp_group):
    layer = Qwen38NextAttention.__new__(Qwen38NextAttention)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.indexer = None
    layer.compress_ratio = 4
    layer.pg_collection = SimpleNamespace(cp=cp_group)
    layer._qsa_selection = layer._qsa_blocks = layer._qsa_cp_rows = None
    return layer


@pytest.mark.parametrize("kind", ["raw", "materialized", "text"])
@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_layer_applies_cp1_rope_to_its_rows(monkeypatch, kind, cp_size):
    """SelfAttention.forward is replaced by what it does with RoPE (apply_rotary_pos_emb with the packing and
    pg_collection.cp it sees); the ranks' outputs concatenated equal the CP1 layer's, and pg_collection.cp is put
    back afterwards."""
    from megatron.core import parallel_state

    cu = [0, 13, 32]
    total = cu[-1]
    config = _config(mrope=kind != "text")
    freqs, packed = _rope(kind, cu), _packed(cu)
    torch.manual_seed(0)
    hidden = torch.randn(total, 1, HEAD_DIM)

    def self_attention_rope(self, hidden_states, *args, **kwargs):
        seen = kwargs["packed_seq_params"]
        return _apply(hidden_states[:, 0], kwargs["rotary_pos_emb"], self.config, seen, self.pg_collection.cp)

    monkeypatch.setattr(SelfAttention, "forward", self_attention_rope)
    monkeypatch.setattr(attention, "qsa_selected_blocks", lambda *a, **k: (None, packed.cu_seqlens_q))
    state = {"size": 1, "rank": 0}
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: state["size"])
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: state["rank"])
    monkeypatch.setattr(attention, "gather_sequence_no_grad", lambda x: hidden)

    cp1 = _attention_layer(config, _Group(1, 0))
    whole = cp1(hidden, packed_seq_params=packed, rotary_pos_emb=freqs)
    local = total // cp_size
    pieces = []
    for rank in range(cp_size):
        state.update(size=cp_size, rank=rank)
        group = _Group(cp_size, rank)
        layer = _attention_layer(config, group)
        piece = hidden[rank * local : (rank + 1) * local]
        pieces.append(layer(piece, packed_seq_params=packed, rotary_pos_emb=freqs))
        assert layer.pg_collection.cp is group
    assert torch.equal(torch.cat(pieces), whole)


# ── the QSA core attention reads global rows ──────────────────────────────────────────────────────────────


def _first_blocks(cu, ratio):
    first = [0]
    for a, b in zip(cu, cu[1:]):
        first.append(first[-1] + -(-(b - a) // ratio))
    return first


def _reference_index(blocks, cu_seqlens, *, compress_ratio, num_heads, num_kv_heads):
    """``build_qsa_block_index``'s row contract: query row ``i`` is packed row ``i`` of ``cu_seqlens``."""
    return SimpleNamespace(blocks=blocks.clone(), cu=[int(c) for c in cu_seqlens], ratio=compress_ratio)


def _reference_attention(q, k, v, index, scale):
    cu, ratio = index.cu, index.ratio
    first = _first_blocks(cu, ratio)
    out = []
    for i in range(q.shape[0]):
        seq = next((s for s in range(len(cu) - 1) if cu[s] <= i < cu[s + 1]), None)
        if seq is None:
            out.append(torch.zeros_like(q[i]))
            continue
        own_start = cu[seq] + (i - cu[seq]) // ratio * ratio
        keys = set(range(own_start, i + 1))
        for b in index.blocks[i].tolist():
            if first[seq] <= b < first[seq + 1]:
                start = cu[seq] + (b - first[seq]) * ratio
                if start + ratio <= own_start:
                    keys.update(range(start, start + ratio))
        keys = torch.tensor(sorted(keys))
        group = q.shape[1] // k.shape[1]
        kk, vv = k[keys].repeat_interleave(group, 1), v[keys].repeat_interleave(group, 1)
        weights = torch.softmax(torch.einsum("hd,khd->hk", q[i], kk) * scale, -1)
        out.append(torch.einsum("hk,khd->hd", weights, vv))
    return torch.stack(out)


def _core(config, blocks, cu):
    owner = SimpleNamespace(
        _qsa_selection=None, _qsa_blocks=(blocks, cu), compress_ratio=4, indexer=SimpleNamespace(block_topk=2)
    )
    core = Qwen38NextQSACoreAttention.__new__(Qwen38NextQSACoreAttention)
    torch.nn.Module.__init__(core)
    core.config = config
    core.layer_number = 1
    core.softmax_scale = HEAD_DIM**-0.5
    object.__setattr__(core, "_owner", owner)
    return core


@pytest.mark.parametrize("cu", [(0, 32), (0, 13, 32), (0, 5, 20, 27, 32), (0, 10, 25, 32)], ids=str)
@pytest.mark.parametrize("cp_size", [2, 4])
def test_core_attention_rows_and_grads_equal_cp1(monkeypatch, cu, cp_size):
    from megatron.core import parallel_state

    cu = list(cu)
    total = cu[-1]
    monkeypatch.setattr(attention, "build_qsa_block_index", _reference_index)
    monkeypatch.setattr(attention, "qsa_block_attention", _reference_attention)
    state = {"size": 1, "rank": 0}
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: state["size"])
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: state["rank"])
    torch.manual_seed(0)
    n_blocks = _first_blocks(cu, 4)[-1]
    blocks = torch.randint(-1, n_blocks, (total, 3), dtype=torch.int32)
    q = torch.randn(total, 4, HEAD_DIM, dtype=torch.float64, requires_grad=True)
    k = torch.randn(total, 2, HEAD_DIM, dtype=torch.float64, requires_grad=True)
    v = torch.randn(total, 2, HEAD_DIM, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(total, 4 * HEAD_DIM, dtype=torch.float64)
    config = _config()
    cu_t = torch.tensor(cu, dtype=torch.int32)

    whole = _core(config, blocks, cu_t)(q, k, v)
    (whole * weight).sum().backward()
    expected = [x.grad.clone() for x in (q, k, v)]
    for x in (q, k, v):
        x.grad = None

    # the gather hands back the whole key / value; their gradient accumulates over the ranks as the reduce-scatter
    gathered = iter([])
    monkeypatch.setattr(attention, "gather_sequence", lambda x: next(gathered))
    local = total // cp_size
    pieces = []
    for rank in range(cp_size):
        state.update(size=cp_size, rank=rank)
        lo, hi = rank * local, (rank + 1) * local
        gathered = iter([k, v])
        out = _core(config, blocks[lo:hi], cu_t)(q[lo:hi], k[lo:hi], v[lo:hi])
        (out * weight[lo:hi]).sum().backward()
        pieces.append(out.detach())
    torch.testing.assert_close(torch.cat(pieces), whole.detach(), rtol=0, atol=1e-12)
    for got, want in zip((q.grad, k.grad, v.grad), expected):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)
