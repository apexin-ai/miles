"""CPU tests: the PLE n-gram batch under the EP-overlap schedule (combined 1F1B) is bound per microbatch.

That schedule never calls the model, so the model hooks that publish the PLE batch do not run, and its full
recompute is a hand-rolled replay (``RecomputeSegment``), not ``CheckpointFunction``: ``is_checkpointing()`` stays
False and the PLE layer's recompute FIFO is never used. Microbatch i's replay runs while microbatch i + 1's plan
is already built (and, in a deep model, after its PLE layer's forward), so one batch published per microbatch is
read by the wrong replay. With equal lengths that is silent: PLE trains on the next microbatch's n-grams.

The tests run Megatron's real ``TransformerLayerNode`` (the attention node, whose ``submodule`` both the first
pass and the replay call) and its real ``RecomputeSegment`` around the real ``Qwen38NextPLEHyperConnection``
(fp64 torch references for the Triton kernels, as in test_ple_sequence_parallel_grads), in the schedule's order,
on two equal-length microbatches with different tokens. Each microbatch's input and parameter gradients must
equal its eager run; the same schedule with the batch published at plan-build time (the positive control) must
not.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.miles_plugins.models.qwen3_8_next import test_ple_sequence_parallel_grads as m001

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["miles-plugin"])

SEQ = m001.SEQ
HEADS = (m001.NGRAM_SIZE - 1) * m001.HEADS_PER_NGRAM
CU = [0, 5, 31, SEQ]


class _FakeEvent:
    def wait(self, stream=None):
        pass

    def record(self, stream=None):
        pass


class _NGramIds:
    """Stands in for the frozen n-gram embedding's hash: ids that differ wherever the token windows differ."""

    ngram_size = m001.NGRAM_SIZE
    eos_token_id = 0

    def compute_ngram_ids(self, contexts):
        weights = torch.arange(1, contexts.shape[1] + 1, dtype=torch.long)
        base = (contexts * weights).sum(dim=1, keepdim=True)
        return (base + 7 * torch.arange(HEADS, dtype=torch.long)) % m001.TABLE_ROWS


@pytest.fixture
def cpu(monkeypatch):
    import megatron.core.tensor_parallel.random as mp_random

    from miles_plugins.models.qwen3_8_next import hyper_connection, model_provider
    from miles_plugins.models.qwen3_8_next.ops import ple

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(ple, "ple_gate_conv", m001._ple_gate_conv_reference)
    monkeypatch.setattr(ple, "TELinear", m001._DuplicatedLinear)
    monkeypatch.setattr(ple, "Qwen38NextFrozenNGramEmbedding", m001._HostTable)
    monkeypatch.setattr(hyper_connection, "hc_mix_inject", m001._hc_mix_inject_reference)
    monkeypatch.setattr(mp_random, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(mp_random, "_set_all_rng_states", lambda *states: None)
    # the schedule nodes record their tensors on CUDA streams; there are none on CPU
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None, raising=False)
    monkeypatch.setattr(model_provider, "_find_ple_embedding", lambda model: _NGramIds())
    ple.clear_ple_batch()
    yield SimpleNamespace(ple=ple, model_provider=model_provider)
    ple.clear_ple_batch()


def _build_hc():
    hc = m001._build_layer()
    hc.config.sequence_parallel = False  # one rank: the scheduling, not SP, is under test
    return hc


class _LayerPlan:
    """The parts of ``TransformerLayerSchedulePlan`` the binder and ``RecomputeSegment`` use, around a real
    attention node; its submodule is the attention site: the PLE hyper-connection's forward."""

    def __init__(self, layer, config):
        from megatron.core.models.gpt.fine_grained_callables import TransformerLayerNode

        def attn(node, hidden_states):
            aggregated, h_res, h_post, residual = layer.self_attention_hyper_connection(hidden_states)
            assert h_res is None
            return aggregated + h_post.sum(dim=-1, keepdim=True)

        self.layer = layer
        self.attn = TransformerLayerNode(
            lambda: None, _FakeEvent(), SimpleNamespace(), None, attn, name="attn", extra_args={"config": config}
        )
        self.recompute_segment = None

    def set_forward_no_grad(self, no_grad):
        self.attn.forward_no_grad = no_grad

    def recompute_forward(self, f_input):
        return self.attn.forward(f_input)


class _Plan:
    def __init__(self, model, input_ids):
        from megatron.core.models.common.model_chunk_schedule_plan import RecomputeSegment

        self.state = SimpleNamespace(
            input_ids=input_ids,
            position_ids=None,
            padding_mask=None,
            mtp_hidden_states=None,
            mhc_multistream=None,
            mhc_grad_carrier=None,
        )
        self.layers = [_LayerPlan(model.decoder_layer, model.config)]
        self.segment = RecomputeSegment(self.layers, self.state, 0)  # full recompute, one layer per segment

    def num_layers(self):
        return len(self.layers)

    def get_layer(self, i):
        return self.layers[i]

    def forward(self, x):
        """``TransformerLayerSchedulePlan.run``'s forward half: capture the segment input, first pass under no_grad."""
        layer = self.layers[0]
        self.segment.capture(layer, x)
        return layer.attn.forward(x)

    def backward(self, grad):
        """The backward half: replay the segment with grad, then the node's backward; returns d loss / d input."""
        layer = self.layers[0]
        self.segment.recompute(layer)
        return layer.attn.backward(grad)


class _Model(torch.nn.Module):
    def __init__(self, hc, publish_at_build=False):
        super().__init__()
        self.decoder_layer = torch.nn.Module()
        self.decoder_layer.self_attention_hyper_connection = hc
        self.config = SimpleNamespace(
            moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None, fp8=None, fp4=None
        )
        self.publish_at_build = publish_at_build

    def build_schedule_plan(self, input_ids, packed_seq_params=None, **_kwargs):
        if self.publish_at_build:  # the positive control: what publishing once per microbatch would do
            from miles_plugins.models.qwen3_8_next.model_provider import _ple_batch
            from miles_plugins.models.qwen3_8_next.ops.ple import publish_ple_batch

            publish_ple_batch(*_ple_batch(_NGramIds(), input_ids, packed_seq_params))
        return _Plan(self, input_ids)


def _microbatches():
    gen = torch.Generator().manual_seed(3)
    width = m001.N_STREAMS * m001.HIDDEN
    batches = []
    for _ in range(2):  # equal length, different tokens: a mix-up does not change any shape
        ids = torch.randint(1, 50, (1, SEQ), generator=gen)
        x = torch.randn(SEQ, 1, width, dtype=torch.float64, generator=gen)
        grad = torch.randn(SEQ, 1, m001.HIDDEN, dtype=torch.float64, generator=gen)
        batches.append(SimpleNamespace(ids=ids, x=x, grad=grad))
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor(CU, dtype=torch.int32))
    return batches, packed


def _eager(hc, mb, packed, cpu):
    """The non-overlap path: the model hook publishes the batch, the layer runs with grad."""
    from miles_plugins.models.qwen3_8_next.model_provider import _ple_batch

    hc.zero_grad(set_to_none=True)
    x = mb.x.clone().requires_grad_()
    cpu.ple.publish_ple_batch(*_ple_batch(_NGramIds(), mb.ids, packed))
    aggregated, _, h_post, _ = hc(x)
    cpu.ple.clear_ple_batch()
    torch.autograd.backward(aggregated + h_post.sum(dim=-1, keepdim=True), mb.grad)
    return x.grad, _param_grads(hc)


def _param_grads(hc):
    return {name: p.grad.clone() for name, p in hc.named_parameters() if p.grad is not None}


def _assert_matches(got, ref, what):
    dx, grads = got
    ref_dx, ref_grads = ref
    torch.testing.assert_close(dx, ref_dx, rtol=0, atol=1e-12, msg=f"{what}: input grad")
    assert sorted(grads) == sorted(ref_grads), f"{what}: parameters with a gradient differ"
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=0, atol=1e-12, msg=f"{what}: {name} grad")
    assert "ple.key_proj.weight" in grads and "ple.value_proj.weight" in grads


def _overlap_schedule(model, batches, packed, replay_first):
    """combined_1f1b_schedule_for_no_pipelining with two microbatches and one (PLE) layer, full recompute.

    phase 0: build plan 0, forward 0.  phase 1: build plan 1, then forward 1 and backward 0 (its replay first
    when ``replay_first``, as for a layer in the back half of a deep model; after forward 1 for the front half).
    phase 2: backward 1. Returns each microbatch's (input grad, param grads).
    """
    hc = model.decoder_layer.self_attention_hyper_connection
    build = model.miles_build_schedule_plan
    hc.zero_grad(set_to_none=True)
    kwargs = [dict(input_ids=mb.ids, packed_seq_params=packed) for mb in batches]
    xs = [mb.x.clone() for mb in batches]

    plan0 = build(kwargs[0])
    plan0.forward(xs[0])
    plan1 = build(kwargs[1])
    if replay_first:
        dx0 = plan0.backward(batches[0].grad)
        grads0 = _param_grads(hc)
        plan1.forward(xs[1])
    else:
        plan1.forward(xs[1])
        dx0 = plan0.backward(batches[0].grad)
        grads0 = _param_grads(hc)
    hc.zero_grad(set_to_none=True)
    dx1 = plan1.backward(batches[1].grad)
    return [(dx0, grads0), (dx1, _param_grads(hc))]


@pytest.mark.parametrize("replay_first", [True, False], ids=["replay-before-next-forward", "replay-after"])
def test_each_microbatch_trains_on_its_own_ngrams(cpu, replay_first):
    hc = _build_hc()
    batches, packed = _microbatches()
    refs = [_eager(hc, mb, packed, cpu) for mb in batches]
    model = _Model(hc)
    cpu.model_provider.install_schedule_plan_builder(model)

    got = _overlap_schedule(model, batches, packed, replay_first)

    for i in range(2):
        _assert_matches(got[i], refs[i], f"microbatch {i}")
    # nothing leaks out of the nodes: outside them PLE has no batch
    assert cpu.ple.scoped_ple_batch() is None
    with pytest.raises(RuntimeError, match="no n-gram ids published"):
        cpu.ple.current_ple_batch()


def test_positive_control_publishing_per_microbatch_trains_on_the_wrong_ngrams(cpu):
    """Without the per-node binding, microbatch 0's replay reads microbatch 1's batch: no error, wrong grads."""
    hc = _build_hc()
    batches, packed = _microbatches()
    refs = [_eager(hc, mb, packed, cpu) for mb in batches]
    model = _Model(hc, publish_at_build=True)
    model.miles_build_schedule_plan = lambda kwargs, multimodal=None: model.build_schedule_plan(**kwargs)

    got = _overlap_schedule(model, batches, packed, replay_first=True)

    with pytest.raises(AssertionError):
        _assert_matches(got[0], refs[0], "microbatch 0")
    _assert_matches(got[1], refs[1], "microbatch 1")  # the last one published: right by luck


def test_an_unbound_replay_fails_loudly(cpu):
    """A PLE node the binder missed must not fall back to a stale batch: with nothing published it raises."""
    hc = _build_hc()
    batches, packed = _microbatches()
    model = _Model(hc)
    plan = model.build_schedule_plan(input_ids=batches[0].ids, packed_seq_params=packed)
    with pytest.raises(RuntimeError, match="no n-gram ids published"):
        plan.forward(batches[0].x.clone())


def test_binder_counts_the_ple_layers(cpu):
    hc = _build_hc()
    batches, packed = _microbatches()
    plan = _Model(hc).build_schedule_plan(input_ids=batches[0].ids, packed_seq_params=packed)
    ids = torch.zeros(SEQ, HEADS, dtype=torch.long)
    with pytest.raises(RuntimeError, match="expected 2"):
        cpu.model_provider.bind_ple_batch_to_plan(plan, ids, None, expected_layers=2)


def test_scope_wins_over_the_published_batch_and_the_recompute_fifo(cpu, monkeypatch):
    from miles_plugins.models.qwen3_8_next import hyper_connection

    hc = _build_hc()
    a = (torch.zeros(SEQ, HEADS, dtype=torch.long), None)
    b = (torch.ones(SEQ, HEADS, dtype=torch.long), None)
    c = (torch.full((SEQ, HEADS), 2, dtype=torch.long), None)
    cpu.ple.publish_ple_batch(*a)
    hc._ple_recompute_fifo = [b]
    monkeypatch.setattr(hyper_connection, "is_checkpointing", lambda: True)
    with cpu.ple.ple_batch_scope(*c):
        assert hc._resolve_ple_batch() is not None and hc._resolve_ple_batch()[0] is c[0]
        with cpu.ple.ple_batch_scope(*a):
            assert hc._resolve_ple_batch()[0] is a[0]
        assert hc._resolve_ple_batch()[0] is c[0]
    assert hc._ple_recompute_fifo == [b], "the scoped path must not touch the eager path's FIFO"
    with contextlib.suppress(RuntimeError), cpu.ple.ple_batch_scope(*c):
        raise RuntimeError("a node that fails")
    assert cpu.ple.scoped_ple_batch() is None, "the scope is restored when the node raises"


def test_megatron_runs_the_attention_submodule_for_the_first_pass_and_the_replay():
    """The binding relies on these Megatron fdf82356 internals; fail here, not silently, if they change."""
    import inspect

    from megatron.core.models.common.model_chunk_schedule_plan import RecomputeSegment, TransformerLayerSchedulePlan
    from megatron.core.models.gpt.fine_grained_callables import TransformerLayerNode

    assert "self.submodule(self, *args)" in inspect.getsource(TransformerLayerNode.forward_impl)
    assert "seg_layer.recompute_forward(" in inspect.getsource(RecomputeSegment.recompute)
    assert "yield self.attn" in inspect.getsource(TransformerLayerSchedulePlan._iter_layer_nodes)
    assert "node.forward(f_input)" in inspect.getsource(TransformerLayerSchedulePlan.recompute_forward)
    assert 'self.attn = create_node(comp_stream, attn_module, "attn")' in inspect.getsource(
        TransformerLayerSchedulePlan._build_callable_nodes
    )
