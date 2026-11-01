"""CPU tests: what the EP-overlap schedule (combined 1F1B) needs besides the PLE batch binding.

* Delayed weight gradients (``--delay-wgrad-compute``): the PLE projections (and the frozen QSA indexer) are TE
  linears that no ``backward_dw`` reaches, so they are built without delay; a step fails if any TE module still
  holds deferred work after the backward (``assert_delayed_wgrad_drained``).
* Both GDN ``self_attention`` classes (replicated and head-sharded) have the ``backward_dw`` Megatron reads.
* ``build_schedule_plan`` (miles model.py) goes through the provider's builder and carries ``fp32_output``.
* The vision provider's plan inputs equal what its ``forward`` passes to the language model: native mRoPE
  positions, the image decoder input, and on a text microbatch the zero tower gradients.
"""

import inspect
import queue
from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci
from tests.fast.backends.megatron_utils import test_qwen38_vision_text_fallback as vt
from tests.fast.miles_plugins.models.qwen3_8_next import test_ple_sequence_parallel_grads as m001

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["miles-plugin"])

# ----------------------------------------------------------------------------------------------
# delayed weight gradients


def test_without_delayed_wgrad_copies_only_when_needed():
    from miles_plugins.models.qwen3_8_next.ops.delayed_wgrad import without_delayed_wgrad

    off = SimpleNamespace(delay_wgrad_compute=False, hidden_size=8)
    assert without_delayed_wgrad(off) is off
    plain = SimpleNamespace(hidden_size=8)
    assert without_delayed_wgrad(plain) is plain
    on = SimpleNamespace(delay_wgrad_compute=True, hidden_size=8)
    copy = without_delayed_wgrad(on)
    assert copy is not on and copy.delay_wgrad_compute is False and copy.hidden_size == 8
    assert on.delay_wgrad_compute is True, "the model's config must keep delayed wgrad for everything else"


def test_ple_projections_never_defer_their_weight_gradient(monkeypatch):
    from miles_plugins.models.qwen3_8_next.ops import ple

    seen = []

    class _Recorder(m001._DuplicatedLinear):
        def __init__(self, input_size, output_size, *, config, **kwargs):
            seen.append(config)
            super().__init__(input_size, output_size, config=config, **kwargs)

    monkeypatch.setattr(ple, "TELinear", _Recorder)
    monkeypatch.setattr(ple, "Qwen38NextFrozenNGramEmbedding", m001._HostTable)
    config = m001._config()
    config.delay_wgrad_compute = True
    ple.Qwen38NextPLE(config, layer_number=1)
    assert len(seen) == 2 and all(c.delay_wgrad_compute is False for c in seen), "key_proj / value_proj"
    assert config.delay_wgrad_compute is True


def test_qsa_indexer_projection_is_built_without_delayed_wgrad():
    from miles_plugins.models.qwen3_8_next.ops import qsa_indexer

    source = inspect.getsource(qsa_indexer.Qwen38NextQSAIndexer.__init__)
    assert "config=without_delayed_wgrad(config)" in source


class _Store:
    def __init__(self, delayed, pending):
        self._delayed = delayed
        self.context = queue.Queue() if delayed else None
        for _ in range(pending):
            self.context.put(([], None))

    def delay_wgrad_compute(self):
        return self._delayed


def _chunk(**stores):
    chunk = torch.nn.Module()
    for name, store in stores.items():
        module = torch.nn.Module()
        module.wgrad_store = store
        chunk.add_module(name, module)
    return chunk


def test_a_step_fails_when_deferred_weight_gradients_were_never_computed():
    from miles.backends.megatron_utils.model import assert_delayed_wgrad_drained

    assert_delayed_wgrad_drained([_chunk(a=_Store(True, 0), b=_Store(False, 0))])
    with pytest.raises(RuntimeError, match=r"never computed.*ple_key_proj \(2\)"):
        assert_delayed_wgrad_drained([_chunk(ok=_Store(True, 0), ple_key_proj=_Store(True, 2))])


def test_train_one_step_checks_the_queues_when_delayed_wgrad_is_on():
    from miles.backends.megatron_utils import model

    source = inspect.getsource(model.train_one_step)
    check = source.index("assert_delayed_wgrad_drained(model)")
    assert source.index("losses_reduced = run_forward_backward_pass(") < check
    assert "if " not in source[source.rindex("\n", 0, check) : check], "the check runs every step"


# ----------------------------------------------------------------------------------------------
# GDN backward_dw


def test_both_gdn_attention_classes_have_backward_dw(gdn):
    from miles_plugins.models.linear_attn import LinearAttentionLayer

    for cls in (gdn.ReplicatedGDNAttention, LinearAttentionLayer):
        assert cls.backward_dw(object()) is None, cls.__name__


def test_megatron_reads_self_attention_backward_dw_when_building_a_plan():
    from megatron.core.models.gpt.fine_grained_callables import _BackwardDWWrapper, build_transformer_layer_callables

    assert "layer.self_attention.backward_dw" in inspect.getsource(_BackwardDWWrapper.__init__)
    assert "layer.init_backward_dw_wrapper()" in inspect.getsource(build_transformer_layer_callables)


# ----------------------------------------------------------------------------------------------
# miles build_schedule_plan


class _PlainModel:
    def __init__(self):
        self.calls = []

    def build_schedule_plan(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(state=SimpleNamespace())


def test_build_schedule_plan_uses_the_provider_builder_and_sets_the_output_dtype():
    from miles.backends.megatron_utils.model import build_schedule_plan

    model = _PlainModel()
    seen = []

    def builder(plan_kwargs, multimodal):
        seen.append((plan_kwargs, multimodal))
        return SimpleNamespace(state=SimpleNamespace())

    model.miles_build_schedule_plan = builder
    plan = build_schedule_plan(model, {"input_ids": 1}, {"pixel_values": 2}, fp32_output=False)
    assert seen == [({"input_ids": 1}, {"pixel_values": 2})] and model.calls == []
    assert plan.state.fp32_output is False


def test_build_schedule_plan_without_a_builder_refuses_multimodal_inputs():
    from miles.backends.megatron_utils.model import build_schedule_plan

    model = _PlainModel()
    plan = build_schedule_plan(model, {"input_ids": 1}, None, fp32_output=True)
    assert model.calls == [{"input_ids": 1}] and plan.state.fp32_output is True
    with pytest.raises(NotImplementedError, match="pixel_values"):
        build_schedule_plan(model, {"input_ids": 1}, {"pixel_values": 2}, fp32_output=True)


def test_forward_step_sends_the_overlap_path_through_build_schedule_plan():
    from miles.backends.megatron_utils import model

    source = inspect.getsource(model.run_forward_backward_pass)
    assert "model.build_schedule_plan(" not in source
    assert 'batch["multimodal_train_inputs"],' in source


# ----------------------------------------------------------------------------------------------
# vision inputs for the plan


class _PlanLanguage(vt._Language):
    """vt._Language plus ``build_schedule_plan``: the plan runs the same language forward on what it was given."""

    def build_schedule_plan(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in ("input_ids", "position_ids", "decoder_input")}
        return SimpleNamespace(state=SimpleNamespace(), loss=type(self).forward(self, **kwargs))


def _vision_model(*, scope, data, seed=0):
    from miles_plugins.models.qwen3_8_next import vision
    from miles_plugins.models.qwen3_8_next.model_provider import install_schedule_plan_builder

    torch.manual_seed(seed)
    model = _PlanLanguage()
    vision.wire_visual(model, vt._CONFIG, scope=scope, recompute=False, freeze_language=False, data=data)
    install_schedule_plan_builder(model)
    return model


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_an_image_microbatch_plan_gets_the_forward_positions_decoder_input_and_gradients(tp_rank_zero):
    ids, inputs = vt._image_batch()
    eager = _vision_model(scope="full", data="mixed")
    loss = eager(input_ids=ids, **inputs)
    loss.backward()
    eager_call = eager.calls[-1]

    planned = _vision_model(scope="full", data="mixed")
    plan = planned.miles_build_schedule_plan({"input_ids": ids, "position_ids": None}, inputs)
    plan.loss.backward()
    plan_call = planned.calls[-1]

    assert torch.equal(plan_call["position_ids"], eager_call["position_ids"])
    assert torch.equal(plan_call["decoder_input"], eager_call["decoder_input"])
    assert torch.equal(plan.loss, loss)
    eager_grads, plan_grads = _grads(eager), _grads(planned)
    assert sorted(plan_grads) == sorted(eager_grads)
    assert any(n.startswith("vision_model.") for n in plan_grads), "the plan's backward must reach the tower"
    for name, grad in eager_grads.items():
        assert torch.equal(plan_grads[name], grad), name


@pytest.mark.parametrize("scope", ["full", "projector", "none"])
def test_a_text_microbatch_plan_matches_the_forward(scope, tp_rank_zero, monkeypatch):
    ids = vt._text_ids()
    eager = _vision_model(scope=scope, data="mixed")
    vt._forbid_vision_forward(eager, monkeypatch)
    eager_reports = vt._grad_reports(eager)
    loss = eager(input_ids=ids)
    loss.backward()
    eager_call = eager.calls[-1]

    planned = _vision_model(scope=scope, data="mixed")
    vt._forbid_vision_forward(planned, monkeypatch)
    plan_reports = vt._grad_reports(planned)
    plan = planned.miles_build_schedule_plan({"input_ids": ids, "position_ids": None}, {})
    plan.loss.backward()
    plan_call = planned.calls[-1]

    assert torch.equal(plan_call["position_ids"], eager_call["position_ids"])
    assert torch.equal(plan.loss, loss)
    assert plan_reports == eager_reports, "a trainable tower must report its zero grads as on the eager path"
    eager_grads, plan_grads = _grads(eager), _grads(planned)
    assert sorted(plan_grads) == sorted(eager_grads)
    for name, grad in eager_grads.items():
        assert torch.equal(plan_grads[name], grad), name
    if scope == "none":
        assert plan_call["decoder_input"] is None, "no tower to report for: the plan embeds as usual"


def test_the_plan_refuses_caller_positions_and_multimodal_inputs_without_the_vision_provider():
    from miles_plugins.models.qwen3_8_next.model_provider import install_schedule_plan_builder

    model = _vision_model(scope="none", data="mixed")
    with pytest.raises(ValueError, match="owns image injection and native mRoPE positions"):
        model.miles_build_schedule_plan({"input_ids": vt._text_ids(), "position_ids": torch.zeros(3, 1, 12)}, {})

    text_only = _PlanLanguage()
    install_schedule_plan_builder(text_only)
    with pytest.raises(ValueError, match="without the vision provider"):
        text_only.miles_build_schedule_plan({"input_ids": vt._text_ids()}, {"pixel_values": torch.zeros(1)})


@pytest.fixture
def tp_rank_zero(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
