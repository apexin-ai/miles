"""Attach Qwen's complete native ViT before Megatron DDP/optimizer construction."""

import json
from copy import copy
from pathlib import Path

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions, scatter_image_embeddings


def set_visual_scope(visual, scope, *, sequence_parallel=False):
    if scope not in ("none", "projector", "full"):
        raise ValueError(f"Unknown Qwen visual scope: {scope}")
    for name, parameter in visual.named_parameters():
        parameter.requires_grad_(scope == "full" or (scope == "projector" and name.startswith("merger.")))
        parameter.sequence_parallel = sequence_parallel


def wire_visual(model, hf_config, *, scope, recompute, freeze_language):
    if freeze_language:
        if scope == "none":
            raise ValueError("Freezing language and vision would leave no trainable parameters")
        model.requires_grad_(False)
    else:
        # QSA selects discrete indices under no_grad; weight decay must not masquerade as learning.
        for name, parameter in model.named_parameters():
            if ".self_attention.indexer." in name:
                parameter.requires_grad_(False)
    if hf_config["vision_config"].get("deepstack_visual_indexes"):
        raise ValueError("This Qwen3.8-Next integration does not implement deepstack injection")
    if model.pre_process:
        vision_config = Qwen3VLVisionConfig(**hf_config["vision_config"])
        vision_config._attn_implementation = "sdpa"
        with torch.device(model.embedding.word_embeddings.weight.device):
            visual = Qwen3VLVisionModel(vision_config).to(dtype=model.config.params_dtype)
        set_visual_scope(visual, scope, sequence_parallel=model.config.sequence_parallel)
        if recompute and scope == "full":
            visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # The existing Qwen38NextBridge maps this exact prefix, including every frozen tensor.
        model.add_module("vision_model", visual)
    original_forward = model.forward

    def forward(*args, pixel_values=None, image_grid_thw=None, qwen38_position_ids=None, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Qwen image SFT requires a single packed input sequence per microbatch")
        if pixel_values is None or image_grid_thw is None or qwen38_position_ids is None:
            raise ValueError("Qwen image SFT requires genuine image patches, grids and mRoPE positions")
        if kwargs.get("decoder_input") is not None or kwargs.get("position_ids") is not None:
            raise ValueError("Qwen image SFT owns image injection and native mRoPE positions")
        length = qwen38_position_ids.shape[0]
        if qwen38_position_ids.shape != (length, 3) or length > ids.shape[1]:
            raise ValueError("Qwen mRoPE positions do not match the input sequence")
        expected = image_rope_positions(
            ids[0, :length],
            image_grid_thw,
            image_token_id=hf_config["image_token_id"],
            vision_start_token_id=hf_config["vision_start_token_id"],
            vision_end_token_id=hf_config["vision_end_token_id"],
            merge_size=hf_config["vision_config"]["spatial_merge_size"],
        )
        if not torch.equal(qwen38_position_ids.to(expected.device), expected):
            raise ValueError("Provided mRoPE positions differ from the native image/grid contract")
        positions = torch.zeros((ids.shape[1], 3), dtype=torch.long, device=ids.device)
        positions[:length] = expected
        kwargs["position_ids"] = positions.T.unsqueeze(1).contiguous()
        if model.pre_process:
            from megatron.core import parallel_state

            decoder = model.embedding(input_ids=ids, position_ids=None)
            with torch.set_grad_enabled(torch.is_grad_enabled() and scope != "none"):
                result = model.vision_model(
                    pixel_values.to(device=decoder.device, dtype=model.config.params_dtype),
                    grid_thw=image_grid_thw,
                    return_dict=True,
                )
            image_positions = (ids[0] == hf_config["image_token_id"]).nonzero().flatten()
            kwargs["decoder_input"] = scatter_image_embeddings(
                decoder,
                image_positions,
                result.pooler_output,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                sequence_parallel=model.config.sequence_parallel,
            )
        return original_forward(*args, **kwargs)

    model.forward = forward


def get_qwen38_vision_model_provider(pre_process=True, post_process=True, vp_stage=None):
    from megatron.training import get_args

    from miles.backends.megatron_utils.model_provider import get_model_provider_func
    from miles_plugins.models.qwen3_8_next.model_provider import _install_ple_context_hooks

    args = get_args()
    if args.context_parallel_size != 1 or args.qkv_format != "thd" or args.micro_batch_size != 1:
        raise ValueError("Qwen image SFT currently requires CP1, THD and micro-batch-size 1")
    if args.use_dynamic_batch_size or (args.tensor_model_parallel_size > 1 and not args.sequence_parallel):
        raise ValueError("Qwen image SFT requires static microbatches and sequence parallelism at TP>1")
    if args.use_rollout_routing_replay or args.enable_mtp_training or args.virtual_pipeline_model_parallel_size:
        raise ValueError("Qwen image SFT does not support routing replay, MTP or interleaved pipelines")
    if args.position_embedding_type != "mrope" or args.apply_rope_fusion:
        raise ValueError("Qwen image SFT needs materialized mRoPE (not text RoPE or fused raw frequencies)")
    config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or config.get("language_model_only"):
        raise ValueError("Expected a complete native Qwen3.8-Flash-Next model")
    provider_args = copy(args)
    provider_args.custom_model_provider_path = None
    model = get_model_provider_func(provider_args)(
        pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
    )
    wire_visual(
        model,
        config,
        scope=args.qwen38_vision_trainable,
        recompute=args.qwen38_vision_recompute,
        freeze_language=args.qwen38_freeze_language,
    )
    _install_ple_context_hooks(model)
    return model
