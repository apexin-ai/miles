"""Qwen4Exp MTP layer on the Megatron CUDA training backend."""

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs, get_gpt_mtp_block_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from torch import nn

from miles_plugins.models.qwen3_8_next.hyper_connection import Qwen38NextHCHeadContraction


class Qwen38MTPHiddenNorm(MegatronModule):
    """Normalize one flat HC vector, preserving MCore's four-dimensional stream view."""

    def __init__(self, config, hidden_size, eps):
        super().__init__(config)
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size * config.num_residual_streams, dtype=config.params_dtype))
        self.weight.sequence_parallel = config.sequence_parallel

    def forward(self, hidden_states):
        flat = hidden_states.flatten(-2).float()
        normalized = torch.nn.functional.rms_norm(flat, self.weight.shape, 1.0 + self.weight.float(), self.eps)
        return normalized.to(hidden_states.dtype).reshape_as(hidden_states)


class Qwen38MTPLayer(MultiTokenPredictionLayer):
    """Keep MCore's rolls, dual TP projections and detach policy; use the native HC readout."""

    def __init__(self, config, submodules, **kwargs):
        super().__init__(config=config, submodules=submodules, **kwargs)
        # MCore's generic HC readout is hard-coded in the layer constructor. Those
        # three DeepSeek-specific parameters have no Qwen4Exp checkpoint entries.
        del self.hc_head_fn, self.hc_head_base, self.hc_head_scale
        self.hc_head_contraction = Qwen38NextHCHeadContraction(config)

    def _get_embeddings(self, input_ids, position_ids, embedding, hidden_states, packed_seq_params=None, **kwargs):
        # Only mixed packs supply this callable, which splices the shifted visual embeddings.
        embedding = getattr(packed_seq_params, "qwen38_mtp_embedding", embedding)
        return super()._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=embedding,
            hidden_states=hidden_states,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )

    def _postprocess(self, hidden_states):
        return self.hc_head_contraction(hidden_states)


def get_qwen38_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine, vp_stage=None):
    from miles_plugins.models.qwen3_8_next.qwen3_8_next import make_qwen38_full_attention_spec

    if config.mtp_num_layers != 1 or config.num_residual_streams != 4:
        raise ValueError("Qwen3.8-Flash-Next MTP requires one draft layer and four residual streams")
    if not config.enable_hyper_connections or not config.layernorm_zero_centered_gamma:
        raise ValueError("Qwen3.8 MTP requires hyper-connections and zero-centered RMSNorm")
    if transformer_layer_spec.hc_head_contraction.module is not Qwen38NextHCHeadContraction:
        raise ValueError("Qwen3.8 MTP requires the native Qwen3.8 decoder spec")

    # Request the native MoE attention template before any trunk GDN/PLE substitutions.
    raw_spec = get_gpt_decoder_layer_specs(config, use_transformer_engine=use_transformer_engine)[-1]
    full_spec = make_qwen38_full_attention_spec(raw_spec, with_ple=False)
    block = get_gpt_mtp_block_spec(config, full_spec, use_transformer_engine=use_transformer_engine, vp_stage=vp_stage)
    if block is not None:
        for spec in block.layer_specs:
            spec.module = Qwen38MTPLayer
            spec.submodules.hnorm = Qwen38MTPHiddenNorm
            spec.submodules.layer_norm = IdentityOp
    return block
