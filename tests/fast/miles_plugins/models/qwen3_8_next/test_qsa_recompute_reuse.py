"""QSA recompute reuse (ops/qsa_recompute_reuse.py): the replay of a checkpointed QSA layer reuses the forward's
block selection (mode blocks), its block index (index) or its attention output and lse (output).

CPU only. The layers run through Megatron's real CheckpointFunction (its RNG save / restore replaced, the CPU has no
CUDA RNG); SelfAttention.forward is replaced by a small projection -> core attention -> projection, so the real
Qwen38NextAttention.forward and Qwen38NextQSACoreAttention.forward run. The Triton pieces are replaced by torch
references with the same contracts: ``build_qsa_block_index`` by a dense mask, and block_attention's ``_forward`` /
``_backward_*`` by an explicit softmax forward and the flash-attention backward formulas, which read the saved out
and lse. So the real ``_QSABlockAttention`` and ``_QSABlockAttentionReplay`` run, and a replay that took a wrong kept
output would give wrong gradients (negative control below). The indexer (``qsa_selected_blocks``) is a seeded
random selection, deterministic or not per test.

Checked: every mode gives gradients bitwise equal to reuse off (deterministic indexer) and runs the indexer / index
build / forward kernel the expected number of times; with an indexer that selects differently each call (as the
index_add_ compression can), reuse off trains on the replay's selection and reuse on on the forward's one (equal
to a run whose replay is given the forward's selection); audit counts 0 mismatches when deterministic and finds the
nondeterministic selection; max_tokens keeps markers only; stale entries are dropped; a missing entry, an in-place
write or a released storage of a kept tensor raise; eval / no-recompute / unpacked forwards keep nothing; PP, CP
and fp8 are refused; configure validates and clears.
"""

import contextlib
import dataclasses
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from megatron.core.tensor_parallel import random as mrandom
from megatron.core.transformer.attention import SelfAttention
from miles.kernels.attention.qsa import block_attention
from miles_plugins.models.qwen3_8_next.ops import attention
from miles_plugins.models.qwen3_8_next.ops import qsa_recompute_reuse as reuse
from miles_plugins.models.qwen3_8_next.ops.attention import Qwen38NextAttention, Qwen38NextQSACoreAttention

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["miles-plugin"])

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, RATIO, TOPK = 32, 4, 2, 16, 4, 3
CU = (0, 23, 40)  # two packed sequences


# ── torch references for the Triton pieces ─────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class RefIndex:
    ratio: int
    mask: torch.Tensor  # [T, T] bool: query row -> attended key rows (selected whole blocks + causal tail)


def ref_build_index(blocks, cu_seqlens, *, compress_ratio, num_heads, num_kv_heads):
    COUNTS["build"] += 1
    cu = [int(x) for x in cu_seqlens.tolist()]
    total = blocks.shape[0]
    mask = torch.zeros(total, total, dtype=torch.bool)
    first_block = 0
    for a, b in zip(cu[:-1], cu[1:]):
        for row in range(a, b):
            pos = row - a
            tail = (pos + 1) // compress_ratio * compress_ratio
            mask[row, a + tail : row + 1] = True
            for blk in blocks[row].tolist():
                if blk < 0:
                    continue
                local = blk - first_block
                if 0 <= local and (local + 1) * compress_ratio <= pos + 1 - ((pos + 1) % compress_ratio):
                    mask[row, a + local * compress_ratio : a + (local + 1) * compress_ratio] = True
        first_block += -(-(b - a) // compress_ratio)
    return RefIndex(compress_ratio, mask)


def _expand(k, group):
    return k.repeat_interleave(group, dim=1)  # [T, Hkv, D] -> [T, Hq, D]


def ref_forward(q, k, v, index, scale):
    COUNTS["attn_fwd"] += 1
    group = q.shape[1] // k.shape[1]
    kk, vv = _expand(k, group).float(), _expand(v, group).float()
    s = torch.einsum("thd,shd->hts", q.float(), kk) * scale
    s = s.masked_fill(~index.mask[None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)  # [H, T]
    p = torch.exp(s - lse[..., None])
    out = torch.einsum("hts,shd->thd", p, vv).to(q.dtype)
    return out, lse.transpose(0, 1).contiguous()


def ref_delta(out, grad_out):
    return (out.float() * grad_out.float()).sum(-1)  # [T, H]


def _probs(q, k, lse, index, scale):
    group = q.shape[1] // k.shape[1]
    s = torch.einsum("thd,shd->hts", q.float(), _expand(k, group).float()) * scale
    s = s.masked_fill(~index.mask[None], float("-inf"))
    return torch.exp(s - lse.transpose(0, 1)[..., None])  # [H, T, S]


def ref_dq(q, k, v, grad_out, lse, delta, index, scale):
    group = q.shape[1] // k.shape[1]
    p = _probs(q, k, lse, index, scale)
    dp = torch.einsum("thd,shd->hts", grad_out.float(), _expand(v, group).float())
    ds = p * (dp - delta.transpose(0, 1)[..., None])
    return (torch.einsum("hts,shd->thd", ds, _expand(k, group).float()) * scale).to(q.dtype)


def ref_dkdv(q, k, v, grad_out, lse, delta, index, scale):
    group = q.shape[1] // k.shape[1]
    p = _probs(q, k, lse, index, scale)
    dp = torch.einsum("thd,shd->hts", grad_out.float(), _expand(v, group).float())
    ds = p * (dp - delta.transpose(0, 1)[..., None])
    dk = torch.einsum("hts,thd->shd", ds, q.float()) * scale
    dv = torch.einsum("hts,thd->shd", p, grad_out.float())
    fold = lambda x: x.view(x.shape[0], k.shape[1], group, -1).sum(2)  # noqa: E731
    return fold(dk).to(k.dtype), fold(dv).to(v.dtype)


COUNTS = {"indexer": 0, "build": 0, "attn_fwd": 0}


class Indexer:
    """qsa_selected_blocks stand-in: per query row up to TOPK whole blocks of its own sequence, seeded. ``drift``:
    every call after the first selects with another seed (as a nondeterministic compression can)."""

    def __init__(self, drift=False):
        self.drift, self.calls = drift, 0

    def __call__(self, indexer, states, positions, rope, cu_seqlens, **kw):
        COUNTS["indexer"] += 1
        self.calls += 1
        seed = 1234 + (self.calls if self.drift else 0) + int(states.shape[0])
        g = torch.Generator().manual_seed(seed)
        cu = [int(x) for x in cu_seqlens.tolist()]
        rows, first_block = [], 0
        for a, b in zip(cu[:-1], cu[1:]):
            n_blocks = -(-(b - a) // RATIO)
            for pos in range(b - a):
                whole = (pos + 1) // RATIO
                pick = (torch.randperm(max(whole, 1), generator=g)[: min(TOPK, whole)] + first_block).tolist()
                rows.append(pick + [-1] * (TOPK - len(pick)))
            first_block += n_blocks
        return torch.tensor(rows, dtype=torch.int32), cu_seqlens


# ── the layer ────────────────────────────────────────────────────────────────────────────────────────────


def _config(**kw):
    base = dict(
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=False,
        mrope_section=None,
        sequence_parallel=False,
        qwen3_8_next_allgather_cp=False,
        qwen3_8_next_materialized_rope=False,
        qwen3_8_next_indexer_tp_shard=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        kv_channels=HEAD_DIM,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class QSALayer(Qwen38NextAttention):
    """Qwen38NextAttention with its own small q / k / v / output projections (SelfAttention.forward patched)."""

    @classmethod
    def build(cls, seed, layer_number, config):
        layer = cls.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.config = config
        layer.layer_number = layer_number
        layer.indexer = SimpleNamespace(block_topk=TOPK)
        layer.compress_ratio = RATIO
        layer.pg_collection = SimpleNamespace(cp=None)
        layer._qsa_selection = layer._qsa_blocks = layer._qsa_cp_rows = None
        layer._qsa_reuse_phase = layer._qsa_reuse_entry = None
        layer._qsa_reuse_stash = reuse.LayerStash(layer_number)
        core = Qwen38NextQSACoreAttention.__new__(Qwen38NextQSACoreAttention)
        torch.nn.Module.__init__(core)
        core.config, core.layer_number, core.softmax_scale = config, layer_number, HEAD_DIM**-0.5
        object.__setattr__(core, "_owner", layer)
        layer.core_attention = core
        g = torch.Generator().manual_seed(seed)
        w = lambda *s: torch.nn.Parameter(torch.randn(*s, generator=g) * 0.3)  # noqa: E731
        layer.wq, layer.wk, layer.wv = (
            w(HIDDEN, HEADS * HEAD_DIM),
            w(HIDDEN, KV_HEADS * HEAD_DIM),
            w(HIDDEN, KV_HEADS * HEAD_DIM),
        )
        layer.wo = w(HEADS * HEAD_DIM, HIDDEN)
        return layer


def _self_attention(self, hidden_states, *args, **kwargs):
    h = hidden_states[:, 0]
    q = (h @ self.wq).view(-1, HEADS, HEAD_DIM)
    k = (h @ self.wk).view(-1, KV_HEADS, HEAD_DIM)
    v = (h @ self.wv).view(-1, KV_HEADS, HEAD_DIM)
    out = self.core_attention(q, k, v, None, packed_seq_params=kwargs["packed_seq_params"])
    return (out @ self.wo)[:, None], None


def _packed(cu=CU):
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


def _rope(total):
    inv = 1.0 / (1e4 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    f = torch.outer(torch.arange(total, dtype=torch.float32), inv)
    return torch.cat((f, f), -1)[:, None, None, :]


@pytest.fixture
def env(monkeypatch):
    for key in COUNTS:
        COUNTS[key] = 0
    monkeypatch.setattr(SelfAttention, "forward", _self_attention)
    monkeypatch.setattr(attention, "build_qsa_block_index", ref_build_index)
    monkeypatch.setattr(block_attention, "_forward", ref_forward)
    monkeypatch.setattr(block_attention, "_backward_delta", ref_delta)
    monkeypatch.setattr(block_attention, "_backward_dq", ref_dq)
    monkeypatch.setattr(block_attention, "_backward_dkdv", ref_dkdv)
    # Megatron's CheckpointFunction saves / restores the CUDA RNG; there is none here
    monkeypatch.setattr(mrandom, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(mrandom, "_set_all_rng_states", lambda *a, **k: None)
    monkeypatch.setattr(mrandom, "_fork_rng", contextlib.nullcontext)
    # a test that raises inside a checkpoint leaves Megatron's global flag set (CheckpointFunction does not unset
    # it on an exception); every test starts outside a checkpoint
    monkeypatch.setattr(mrandom, "IS_CHECKPOINTING", False)
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)
    reuse.configure("off")
    reuse.reset_stats()
    yield monkeypatch
    reuse.configure("off")
    reuse.reset_stats()


def _run(
    env,
    mode="off",
    *,
    indexer=None,
    checkpoint=True,
    training=True,
    max_tokens=None,
    audit=False,
    config=None,
    tamper=None,
    layers=2,
):
    """Two (default) checkpointed QSA layers with residuals, a weighted sum as loss; returns the gradients."""
    env.setattr(attention, "qsa_selected_blocks", indexer or Indexer())
    reuse.configure(mode, max_tokens=max_tokens, audit=audit)
    config = config or _config()
    stack = [QSALayer.build(10 + i, 4 * i + 4, config) for i in range(layers)]
    for layer in stack:
        layer.train(training)
    packed, total = _packed(), CU[-1]
    rope = _rope(total)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(total, 1, HIDDEN, generator=g, requires_grad=True)
    weight = torch.randn(total, 1, HIDDEN, generator=g)
    h = x
    for layer in stack:

        def fn(inp, layer=layer):
            return layer(inp, packed_seq_params=packed, rotary_pos_emb=rope)[0] + inp

        h = mrandom.checkpoint(fn, False, h) if checkpoint else fn(h)
    if tamper is not None:
        tamper(stack)
    (h * weight).sum().backward()
    grads = {"x": x.grad}
    for i, layer in enumerate(stack):
        for name in ("wq", "wk", "wv", "wo"):
            grads[f"{i}.{name}"] = getattr(layer, name).grad
    return grads, stack


def _equal(a, b):
    return a.keys() == b.keys() and all(reuse.bitwise_equal(a[k], b[k]) for k in a)


# ── equality and work skipped ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["blocks", "index", "output", "blocks_output"])
def test_every_mode_gives_the_gradients_of_reuse_off(env, mode):
    off, _ = _run(env, "off")
    counts_off = dict(COUNTS)
    assert counts_off == {"indexer": 4, "build": 4, "attn_fwd": 4}  # 2 layers x (forward + replay)
    for key in COUNTS:
        COUNTS[key] = 0
    reuse.reset_stats()
    on, stack = _run(env, mode)
    assert _equal(on, off)
    expected = {
        "blocks": {"indexer": 2, "build": 4, "attn_fwd": 4},
        "index": {"indexer": 2, "build": 2, "attn_fwd": 4},
        "output": {"indexer": 2, "build": 2, "attn_fwd": 2},
        "blocks_output": {"indexer": 2, "build": 4, "attn_fwd": 2},
    }[mode]
    assert COUNTS == expected
    s = reuse.stats()
    assert s["pushes"] == 2 and s["hits"] == 2 and s["skipped_indexer_selections"] == 2
    assert s.get("skipped_index_builds", 0) == (2 if mode in ("index", "output") else 0)
    assert s.get("skipped_attention_forwards", 0) == (2 if mode in ("output", "blocks_output") else 0)
    assert s.get("stale_dropped", 0) == 0 and s.get("markers", 0) == 0
    assert s["held_bytes"] == 0 and s["held_entries"] == 0 and s["peak_held_bytes"] > 0
    assert all(not layer._qsa_reuse_stash.entries for layer in stack)


def test_kept_bytes_per_mode(env):
    """blocks keeps the [T, K] ids; output keeps the index plus out [T, Hq, D] and lse [T, Hq] fp32."""
    peaks = {}
    for mode in ("blocks", "index", "output", "blocks_output"):
        reuse.reset_stats()
        _run(env, mode, layers=1)
        peaks[mode] = reuse.stats()["peak_held_bytes"]
    total = CU[-1]
    assert peaks["blocks"] == total * TOPK * 4
    assert peaks["index"] == total * total  # the reference index: one bool mask
    assert peaks["output"] == peaks["index"] + total * HEADS * HEAD_DIM * 4 + total * HEADS * 4
    assert peaks["blocks_output"] == peaks["blocks"] + total * HEADS * HEAD_DIM * 4 + total * HEADS * 4


def test_reuse_off_keeps_nothing_and_counts_nothing(env):
    _, stack = _run(env, "off")
    assert reuse.stats() == {"held_bytes": 0, "held_entries": 0, "peak_held_bytes": 0}
    assert all(not layer._qsa_reuse_stash.entries for layer in stack)


# ── a selection that changes between forward and replay ─────────────────────────────────────────────────


class ForwardSelectionTwice(Indexer):
    """Drifting indexer whose replay calls are given the forward calls' selections (what reuse should equal). The
    indexer always runs under no_grad, so the calls are told apart by order: the forward pass calls each layer once
    (``layers`` calls), then the replays run the layers in reverse."""

    def __init__(self, layers=2):
        super().__init__(drift=True)
        self.layers, self.kept = layers, []

    def __call__(self, *args, **kwargs):
        if self.calls < self.layers:
            self.kept.append(super().__call__(*args, **kwargs))
            return self.kept[-1]
        COUNTS["indexer"] += 1
        return self.kept.pop()


def test_drifting_selection_reuse_trains_on_the_forward_selection(env):
    drift_off, _ = _run(env, "off", indexer=Indexer(drift=True))
    consistent, _ = _run(env, "off", indexer=ForwardSelectionTwice())
    assert not _equal(drift_off, consistent)  # the replay's other selection changes the gradients
    for mode in ("blocks", "index", "output", "blocks_output"):
        on, _ = _run(env, mode, indexer=Indexer(drift=True))
        assert _equal(on, consistent), mode


def test_audit_counts_no_mismatch_when_deterministic(env):
    off, _ = _run(env, "off")
    for mode in ("blocks", "index", "output", "blocks_output"):
        reuse.reset_stats()
        on, _ = _run(env, mode, audit=True)
        assert _equal(on, off)
        s = reuse.stats()
        assert s["hits"] == 2 and s["audit_selection_checks"] == 2 and s["audit_selection_mismatches"] == 0
        assert s.get("audit_index_mismatches", 0) == 0 and s.get("audit_output_mismatches", 0) == 0
        assert s.get("audit_lse_mismatches", 0) == 0
        assert s.get("audit_index_checks", 0) == (2 if mode in ("index", "output") else 0)
        assert s.get("audit_output_checks", 0) == (2 if mode in ("output", "blocks_output") else 0)
        assert s.get("skipped_indexer_selections", 0) == 0  # the audit recomputes everything


def test_audit_finds_the_drifting_selection_and_still_uses_the_kept_one(env):
    consistent, _ = _run(env, "off", indexer=ForwardSelectionTwice())
    reuse.reset_stats()
    on, _ = _run(env, "output", indexer=Indexer(drift=True), audit=True)
    assert _equal(on, consistent)
    s = reuse.stats()
    assert s["audit_selection_mismatches"] == 2 and s["audit_selection_checks"] == 2
    # the index and output the audit rebuilds come from the kept selection, so they agree
    assert s["audit_index_mismatches"] == 0 and s["audit_output_mismatches"] == 0


def test_negative_control_a_wrong_kept_output_changes_the_gradients(env):
    """The replay's backward reads the kept out / lse: a perturbed kept output gives other gradients (and the
    audit reports it), so the equality above is not vacuous."""
    off, _ = _run(env, "off")

    def perturb(stack):
        for layer in stack:
            for entry in layer._qsa_reuse_stash.entries:
                entry.out = entry.out + 1e-3  # a new tensor: no version-counter change
                entry.lse = entry.lse.clone()

    reuse.reset_stats()
    bad, _ = _run(env, "output", tamper=perturb, audit=True)
    assert not _equal(bad, off)
    assert reuse.stats()["audit_output_mismatches"] == 2


# ── limits, pairing, safety ─────────────────────────────────────────────────────────────────────────────


def test_max_tokens_keeps_markers_above_the_limit(env):
    off, _ = _run(env, "off")
    for key in COUNTS:
        COUNTS[key] = 0
    reuse.reset_stats()
    on, _ = _run(env, "output", max_tokens=CU[-1] - 1)
    assert _equal(on, off) and COUNTS == {"indexer": 4, "build": 4, "attn_fwd": 4}
    s = reuse.stats()
    assert s["markers"] == 2 and s["marker_replays"] == 2 and s.get("hits", 0) == 0 and s["peak_held_bytes"] == 0
    reuse.reset_stats()
    _run(env, "output", max_tokens=CU[-1])
    assert reuse.stats()["hits"] == 2


@pytest.mark.parametrize("mode, fallback", [("output", "index"), ("blocks_output", "blocks")])
def test_output_max_tokens_keeps_the_cheaper_part_of_longer_microbatches(env, mode, fallback):
    peaks, counts, grads = {}, {}, {}
    for m, cap in (("off", None), (fallback, None), (mode, CU[-1] - 1), (mode, CU[-1])):
        for key in COUNTS:
            COUNTS[key] = 0
        reuse.reset_stats()
        env.setattr(attention, "qsa_selected_blocks", Indexer())
        reuse.configure(m, output_max_tokens=cap)
        layers = [QSALayer.build(10 + i, 4 * i + 4, _config()) for i in range(2)]
        packed, rope = _packed(), _rope(CU[-1])
        x = torch.randn(CU[-1], 1, HIDDEN, generator=torch.Generator().manual_seed(7), requires_grad=True)
        weight = torch.randn(CU[-1], 1, HIDDEN, generator=torch.Generator().manual_seed(8))
        h = x
        for layer in layers:
            h = mrandom.checkpoint(
                lambda inp, layer=layer: layer(inp, packed_seq_params=packed, rotary_pos_emb=rope)[0] + inp, False, h
            )
        (h * weight).sum().backward()
        peaks[(m, cap)], counts[(m, cap)], grads[(m, cap)] = reuse.stats()["peak_held_bytes"], dict(COUNTS), x.grad
    below, above = (mode, CU[-1]), (mode, CU[-1] - 1)
    assert peaks[above] == peaks[(fallback, None)] < peaks[below]  # over the cap: the fallback mode's bytes
    assert counts[above] == counts[(fallback, None)] and counts[below]["attn_fwd"] == 2
    assert all(reuse.bitwise_equal(g, grads[("off", None)]) for g in grads.values())
    assert reuse.settings()["output_max_tokens"] == CU[-1]


def test_a_forward_without_backward_leaves_a_stale_entry_that_the_next_forward_drops(env):
    env.setattr(attention, "qsa_selected_blocks", Indexer())
    reuse.configure("output")
    layer = QSALayer.build(10, 4, _config())
    packed, rope = _packed(), _rope(CU[-1])
    x = torch.randn(CU[-1], 1, HIDDEN, requires_grad=True)

    def fn(inp):
        return layer(inp, packed_seq_params=packed, rotary_pos_emb=rope)[0]

    mrandom.checkpoint(fn, False, x)  # no backward
    assert len(layer._qsa_reuse_stash.entries) == 1 and reuse.stats()["held_entries"] == 1
    packed2 = _packed()
    out = mrandom.checkpoint(lambda inp: layer(inp, packed_seq_params=packed2, rotary_pos_emb=rope)[0], False, x)
    assert reuse.stats()["stale_dropped"] == 1 and len(layer._qsa_reuse_stash.entries) == 1
    out.sum().backward()
    assert reuse.stats()["hits"] == 1 and reuse.stats()["held_entries"] == 0


def test_a_replay_without_an_entry_raises(env):
    def drop(stack):
        for layer in stack:
            layer._qsa_reuse_stash.clear()

    with pytest.raises(RuntimeError, match="found no entry"):
        _run(env, "index", tamper=drop)


def test_an_in_place_write_to_a_kept_tensor_raises(env):
    def write(stack):
        stack[-1]._qsa_reuse_stash.entries[0].out.mul_(1.0)

    with pytest.raises(RuntimeError, match="was changed after the forward"):
        _run(env, "output", tamper=write)


def test_a_released_storage_of_a_kept_tensor_raises(env):
    def release(stack):
        stack[-1]._qsa_reuse_stash.entries[0].lse.untyped_storage().resize_(0)

    with pytest.raises(RuntimeError, match="was changed after the forward"):
        _run(env, "output", tamper=release)


@pytest.mark.parametrize("case", ["eval", "no_checkpoint"])
def test_forwards_without_a_replay_keep_nothing(env, case):
    if case == "eval":
        x_grads, stack = None, None
        with torch.no_grad():
            env.setattr(attention, "qsa_selected_blocks", Indexer())
            reuse.configure("output")
            layer = QSALayer.build(10, 4, _config())
            layer.eval()
            layer(torch.randn(CU[-1], 1, HIDDEN), packed_seq_params=_packed(), rotary_pos_emb=_rope(CU[-1]))
            stack = [layer]
    else:  # the short-microbatch no-recompute path: the layers run without Megatron's checkpoint
        x_grads, stack = _run(env, "output", checkpoint=False)
        off, _ = _run(env, "off", checkpoint=False)
        assert _equal(x_grads, off)
    assert reuse.stats().get("pushes", 0) == 0 and reuse.stats()["peak_held_bytes"] == 0
    assert all(not layer._qsa_reuse_stash.entries for layer in stack)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"pipeline_model_parallel_size": 2}, "pipeline"),
        ({"virtual_pipeline_model_parallel_size": 2}, "pipeline"),
        ({"context_parallel_size": 2}, "context parallelism"),
        ({"fp8": "hybrid"}, "fp8"),
        ({"fp4": "nvfp4"}, "fp8 / fp4"),
    ],
)
@pytest.mark.parametrize("checkpoint", [True, False])
def test_refused_layouts(env, change, match, checkpoint):
    """Refused at the first training forward whether or not Megatron's checkpoint runs it: fp8 / fp4 recompute goes
    through TE's checkpoint, which leaves is_checkpointing() False (checkpoint=False is that route)."""
    with pytest.raises(NotImplementedError, match=match):
        _run(env, "blocks", config=_config(**change), checkpoint=checkpoint)
    _run(env, "off", config=_config(**change), checkpoint=checkpoint)  # off: no check


def test_sbhd_is_refused_in_training_and_passes_in_eval_or_off(env):
    reuse.configure("output")
    layer = QSALayer.build(10, 4, _config())
    with pytest.raises(NotImplementedError, match="packed"):
        layer._reuse_phase(None)
    layer.eval()
    assert layer._reuse_phase(None) == (None, None)
    layer.train()
    reuse.configure("off")
    assert layer._reuse_phase(None) == (None, None)


def _seal_one(env, mode="output"):
    env.setattr(attention, "qsa_selected_blocks", Indexer())
    reuse.configure(mode)
    layer = QSALayer.build(10, 4, _config())
    packed, rope = _packed(), _rope(CU[-1])
    x = torch.randn(CU[-1], 1, HIDDEN, requires_grad=True)
    out = mrandom.checkpoint(lambda inp: layer(inp, packed_seq_params=packed, rotary_pos_emb=rope)[0], False, x)
    return layer, packed, out


@pytest.mark.parametrize("how", ["write", "replace_storage"])
def test_a_changed_cu_after_the_forward_raises(env, how):
    """The pairing key and the index input are guarded like the kept tensors: an in-place boundary change or a
    swapped storage of the same cu object raises at the replay."""
    layer, packed, out = _seal_one(env)
    cu = packed.cu_seqlens_q
    if how == "write":
        cu[1] = 5
    else:
        cu.set_(cu.clone().untyped_storage())
    assert layer._qsa_reuse_stash.entries[0].cu is cu
    with pytest.raises(RuntimeError, match="was changed after the forward"):
        out.sum().backward()


def test_a_data_write_is_not_detected_without_audit(env):
    """Documented limit: a write through .data bypasses the version counter; the guard does not see it (audit does)."""
    layer, _, out = _seal_one(env)
    layer._qsa_reuse_stash.entries[0].out.data.mul_(1.0)
    out.sum().backward()
    assert reuse.stats()["hits"] == 1


def test_gpu_replay_cases_are_supported_layouts():
    """Every positive case of the 1-GPU replay test is a head layout the index accepts; the refused one is not."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[5] / "tests/fast-gpu/kernels/attention/qsa/test_qsa_block_attention_replay.py"
    )
    spec = importlib.util.spec_from_file_location("_perfb_gpu_replay_cases", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CASES and all(mod.supported(h, kv, d) for _, h, kv, d, _, _ in mod.CASES)
    assert any((h, kv, d) == (6, 1, 256) and b == 512 for _, h, kv, d, b, _ in mod.CASES)  # the production layout
    assert mod.REFUSED and not any(mod.supported(h, kv, 64) for _, h, kv in mod.REFUSED)
    assert not mod.supported(13, 2, 64) and not mod.supported(6, 1, 48)


def test_configure_validates_and_clears_kept_entries(env):
    for bad in [
        dict(mode="on"),
        dict(mode="index", max_tokens=0),
        dict(mode="index", max_tokens=True),
        dict(mode="index", max_tokens=1.5),
        dict(mode="index", audit="yes"),
    ]:
        with pytest.raises(ValueError):
            reuse.configure(**bad)
    env.setattr(attention, "qsa_selected_blocks", Indexer())
    reuse.configure("output")
    layer = QSALayer.build(10, 4, _config())
    x = torch.randn(CU[-1], 1, HIDDEN, requires_grad=True)
    mrandom.checkpoint(lambda inp: layer(inp, packed_seq_params=_packed(), rotary_pos_emb=_rope(CU[-1]))[0], False, x)
    assert layer._qsa_reuse_stash.entries
    reuse.configure("output")  # same settings: kept
    assert layer._qsa_reuse_stash.entries
    reuse.configure("index")  # other settings: cleared
    assert not layer._qsa_reuse_stash.entries and reuse.stats()["held_entries"] == 0
    assert reuse.settings() == {"mode": "index", "max_tokens": None, "audit": False, "output_max_tokens": None}
    with pytest.raises(ValueError, match="output_max_tokens"):
        reuse.configure("output", output_max_tokens=0)


def test_bitwise_equal_tells_nan_payloads_and_signed_zeros_apart():
    a = torch.tensor([0.0, 1.0])
    assert reuse.bitwise_equal(a, a.clone())
    assert not reuse.bitwise_equal(a, torch.tensor([-0.0, 1.0]))
    assert not reuse.bitwise_equal(a, a.to(torch.float64))
    assert not reuse.bitwise_equal(a, a[:1])
    assert reuse.bitwise_equal(RefIndex(4, torch.ones(2, 2, dtype=torch.bool)), RefIndex(4, torch.ones(2, 2) > 0))
    assert not reuse.bitwise_equal(RefIndex(4, torch.ones(2, 2) > 0), RefIndex(2, torch.ones(2, 2) > 0))
