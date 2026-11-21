"""The full Qwen3.8 trunk spec remains equivalent to the V4.2 builder (38d88efa)."""

import copy
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.qwen3_8_next import qwen3_8_next as qwen
from miles_plugins.models.qwen3_8_next.mtp import get_qwen38_mtp_block_spec


def _structure(value):
    # functools.partial has identity-based equality. Compare its callable, bound arguments and keywords.
    if isinstance(value, partial):
        return (partial, value.func, _structure(value.args), _structure(value.keywords))
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), {field.name: _structure(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return {key: _structure(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_structure(item) for item in value)
    return value


def _v42_decoder(template, args, text):
    """Independent pre-MTP overrides from 38d88efa, applied to the actual MCore spec."""
    expected = copy.deepcopy(template)
    for i, layer in enumerate(expected.layer_specs):
        # The MCore template may reuse its layer specs across indices.
        layer = expected.layer_specs[i] = copy.deepcopy(layer)
        parts = layer.submodules
        hc = qwen.Qwen38NextPLEHyperConnection if i + 1 in text.ple_layer_ids else qwen.Qwen38NextHyperConnection
        parts.self_attention_hyper_connection = ModuleSpec(module=hc)
        parts.mlp_hyper_connection = ModuleSpec(module=qwen.Qwen38NextHyperConnection)
        if text.layer_types[i] == "linear_attention":
            gdn = qwen.HeadShardedGDNAttention if args.qwen38_gdn_head_sharded else qwen.Qwen38NextLinearAttention
            parts.self_attention = ModuleSpec(module=gdn, params={"args": args})
        else:
            attention = parts.self_attention
            parts.self_attention = ModuleSpec(
                module=qwen.Qwen38NextAttention, params=dict(attention.params or {}), submodules=attention.submodules
            )
            parts.self_attention.submodules.linear_qkv = TEColumnParallelLinear
        parts.input_layernorm = parts.pre_mlp_layernorm = IdentityOp
    expected.hc_head_contraction = ModuleSpec(module=qwen.Qwen38NextHCHeadContraction)
    expected.layer_norm = IdentityOp
    return expected


@pytest.mark.parametrize("mtp_depth", [None, 1], ids=["mtp-off", "mtp-on"])
@pytest.mark.parametrize("sharded", [False, True], ids=["replicated-gdn", "sharded-gdn"])
def test_every_decoder_layer_matches_v42(mtp_depth, sharded):
    text = SimpleNamespace(
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 12,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[2],
        ple_embed_dim=8,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=37,
        split_ngram_parts=1,
        ple_conv_kernel_size=3,
        eos_token_id=7,
        indexer_budget=8,
        indexer_compress_ratio=2,
        indexer_n_heads=2,
        indexer_head_dim=8,
        indexer_kv_heads=1,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        mtp=dict(num_hidden_layers=1, layer_types=["full_attention"], rope_theta=10000000),
        rope_parameters=dict(rope_theta=10000000, mrope_interleaved=True, mrope_section=[11, 11, 10]),
    )
    args = SimpleNamespace(
        hf_checkpoint="qwen38-v42",
        qwen38_gdn_head_sharded=sharded,
        backend=SimpleNamespace(num_experts=4, position_embedding_type="mrope", mrope_section=[11, 11, 10]),
    )
    cfg = TransformerConfig(
        num_layers=48,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=8,
        ffn_hidden_size=16,
        num_moe_experts=4,
        moe_ffn_hidden_size=16,
        moe_shared_expert_intermediate_size=16,
        moe_shared_expert_gate=True,
        moe_grouped_gemm=True,
        mtp_num_layers=mtp_depth,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
    )
    builder = qwen.get_gpt_decoder_block_spec
    templates = []

    def capture(*a, **kw):
        template = builder(*a, **kw)
        templates.append(copy.deepcopy(template))
        return template

    with ExitStack() as stack:
        stack.enter_context(patch.object(qwen, "load_hf_config", return_value=text))
        stack.enter_context(patch.object(qwen, "get_gpt_decoder_block_spec", side_effect=capture))
        stack.enter_context(patch.object(parallel_state, "get_pipeline_model_parallel_rank", return_value=0))
        stack.enter_context(patch.object(parallel_state, "get_pipeline_model_parallel_world_size", return_value=1))
        candidate = qwen.get_qwen3_8_next_spec(args, cfg)
        if mtp_depth:
            # Building the draft must also leave the trunk's nested/submodule specs intact.
            draft = get_qwen38_mtp_block_spec(cfg, candidate, use_transformer_engine=True)
            assert len(draft.layer_specs) == 1
    expected = _v42_decoder(templates[0], args, text)
    assert len(candidate.layer_specs) == len(expected.layer_specs) == 48
    for i, (actual, previous) in enumerate(zip(candidate.layer_specs, expected.layer_specs, strict=True)):
        assert _structure(actual) == _structure(previous), f"V4.2 decoder layer {i} changed"
    # Includes block-level contraction and norms as well as module, params, metainfo and nested submodules.
    assert _structure(candidate) == _structure(expected)
