"""Custom provider: miles' default model + forward hooks publishing the PLE
n-gram side channel (input token ids never reach a transformer layer).
"""

import logging
from typing import TYPE_CHECKING

from megatron.core.models.gpt import GPTModel

from miles_plugins.models.qwen3_8_next.hyper_connection import Qwen38NextPLEHyperConnection
from miles_plugins.models.qwen3_8_next.ops.ple import (
    Qwen38NextFrozenNGramEmbedding,
    build_ngram_contexts_packed,
    clear_ple_batch,
    ple_batch_scope,
    publish_ple_batch,
)

if TYPE_CHECKING:
    from miles.utils.args.runtime import TrainerConfig

logger = logging.getLogger(__name__)


def _find_ple_embedding(model):
    for _, module in model.named_modules():
        if isinstance(module, Qwen38NextFrozenNGramEmbedding):
            return module
    return None


def _ple_batch(ple_embedding, input_ids, packed_seq_params):
    """The PLE side channel for one microbatch: ``(ngram_ids [T, heads], cu_seqlens)``."""
    cu_seqlens = (
        getattr(packed_seq_params, "cu_seqlens_q", None) if packed_seq_params is not None else None
    )  # config-access-exempt: packed-sequence implementations may omit cumulative lengths
    flat = input_ids.reshape(-1)
    contexts = build_ngram_contexts_packed(flat, cu_seqlens, ple_embedding.ngram_size, ple_embedding.eos_token_id)
    return ple_embedding.compute_ngram_ids(contexts), cu_seqlens


def _install_ple_context_hooks(model: GPTModel) -> None:
    install_schedule_plan_builder(model)
    ple_embedding = _find_ple_embedding(model)
    if ple_embedding is None:
        return

    def pre_hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return
        publish_ple_batch(*_ple_batch(ple_embedding, input_ids, kwargs.get("packed_seq_params")))

    def post_hook(_module, _args, _output):
        clear_ple_batch()

    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.register_forward_hook(post_hook)
    logger.info("PLE n-gram context hooks installed on the stage hosting the PLE layer")


def bind_ple_batch_to_plan(plan, ngram_ids, cu_seqlens, expected_layers: int) -> int:
    """Run every PLE layer's attention node of ``plan`` inside ``ple_batch_scope`` with this microbatch's batch.

    The node's ``submodule`` is what both its first pass (``ScheduleNode.forward`` -> ``forward_impl``) and its
    full-recompute replay (``RecomputeSegment.recompute`` -> ``recompute_forward`` -> ``node.forward``) call, so
    wrapping it binds the batch to this microbatch for both, whatever the other microbatch in flight is doing.
    Raises unless exactly ``expected_layers`` layers were bound: an unbound PLE layer would read whatever the side
    channel holds.
    """
    bound = 0
    for i in range(plan.num_layers()):
        layer_plan = plan.get_layer(i)
        hc = getattr(
            layer_plan.layer, "self_attention_hyper_connection", None
        )  # config-access-exempt: only hyper-connection layers carry it
        if not isinstance(hc, Qwen38NextPLEHyperConnection):
            continue
        node = layer_plan.attn
        inner = node.submodule

        def submodule(node_, *inputs, _inner=inner):
            with ple_batch_scope(ngram_ids, cu_seqlens):
                return _inner(node_, *inputs)

        node.submodule = submodule
        bound += 1
    if bound != expected_layers:
        raise RuntimeError(
            f"bound the PLE batch to {bound} attention nodes of the schedule plan, expected {expected_layers}: "
            "a PLE layer outside the plan's decoder layers would read the wrong (or no) n-gram batch"
        )
    return bound


def install_schedule_plan_builder(model) -> None:
    """``model.miles_build_schedule_plan(plan_kwargs, multimodal)``: the EP-overlap (combined 1F1B) entry point.

    That schedule calls ``GPTModel.build_schedule_plan`` instead of the model, so neither the PLE forward hooks
    nor the vision provider's ``forward`` run. This builder does their work per microbatch: the vision inputs
    (mRoPE positions, image embeddings) through ``model.qwen38_prepare_schedule_plan_inputs`` when the vision
    provider set it, and the PLE batch bound to the plan's PLE layers (``bind_ple_batch_to_plan``).
    """

    ple = {}

    def ple_layers():
        if not ple:  # the model is complete once the provider returns; walk it once
            ple["embedding"] = _find_ple_embedding(model)
            ple["layers"] = sum(isinstance(m, Qwen38NextPLEHyperConnection) for m in model.modules())
        return ple["embedding"], ple["layers"]

    def build(plan_kwargs: dict, multimodal: dict | None = None):
        kwargs = dict(plan_kwargs)
        prepare = getattr(
            model, "qwen38_prepare_schedule_plan_inputs", None
        )  # config-access-exempt: set only by the Qwen3.8 vision provider
        if prepare is not None:
            kwargs = prepare(kwargs, **(multimodal or {}))
        elif multimodal:
            raise ValueError(
                f"multimodal inputs {sorted(multimodal)} reached a Qwen3.8 model without the vision provider"
            )
        plan = model.build_schedule_plan(**kwargs)
        ple_embedding, expected = ple_layers()
        if ple_embedding is not None:
            ngram_ids, cu_seqlens = _ple_batch(ple_embedding, kwargs["input_ids"], kwargs.get("packed_seq_params"))
            bind_ple_batch_to_plan(plan, ngram_ids, cu_seqlens, expected)
        return plan

    model.miles_build_schedule_plan = build


def assert_qsa_indexers_frozen(model) -> None:
    """Fail if any QSA indexer parameter would reach the optimizer."""
    trainable = [name for name, p in model.named_parameters() if ".indexer." in name and p.requires_grad]
    if trainable:
        raise RuntimeError(
            f"QSA indexer parameters require grad: {trainable[:4]}{'...' if len(trainable) > 4 else ''}. "
            "They have no training objective; left trainable, weight decay shrinks them every step."
        )


def get_qwen3_8_next_model_provider(
    pre_process: bool = True, post_process: bool = True, vp_stage=None, *, args: "TrainerConfig"
):
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    base_provider = get_model_provider_func(
        type(args).model_validate(dict(args) | {"custom_model_provider_path": None})
    )

    model = base_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    _install_ple_context_hooks(model)
    assert_qsa_indexers_frozen(model)
    return model
