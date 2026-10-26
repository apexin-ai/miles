"""Attach native image embeddings before DDP, preserving language weight names."""

import json
from copy import copy
from pathlib import Path

import torch
from safetensors import safe_open

from miles_plugins.models.deepseek_v4_1.image_processing import span_length
from miles_plugins.models.deepseek_v4_1.vision import DeepSeekV41Visual, VisionConfig
from miles_plugins.models.deepseek_v4_1.weights import decode_weight


def load_visual_weights(visual, checkpoint):
    root = Path(checkpoint)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as shard:
            index = {name: "model.safetensors" for name in shard.keys()}
    state = {}
    grouped = {}
    for name in visual.state_dict():
        if name not in index:
            raise ValueError(f"Missing native V4.1 visual tensor: {name}")
        grouped.setdefault(index[name], []).append(name)
    for filename, names in grouped.items():
        with safe_open(root / filename, framework="pt", device="cpu") as shard:
            for name in names:
                value = shard.get_tensor(name)
                scale_name = name.removesuffix(".weight") + ".scale"
                if value.dtype not in (torch.bfloat16, torch.float32, torch.float16):
                    if scale_name not in index:
                        raise ValueError(f"Missing V4.1 visual quantization scale: {name}")
                    with safe_open(root / index[scale_name], framework="pt", device="cpu") as scales:
                        value = decode_weight(value, scales.get_tensor(scale_name), name=name)
                state[name] = value
    visual.load_state_dict(state, strict=True)


def scatter_image_embeddings(decoder_input, positions, image_embeddings, *, tp_rank=0, sequence_parallel=False):
    if decoder_input.shape[1] != 1:
        raise ValueError("V4.1 image SFT requires micro-batch-size 1")
    if positions.numel() != image_embeddings.shape[0]:
        raise ValueError("V4.1 image token positions and feature rows do not match")
    local_positions = positions.to(decoder_input.device)
    if sequence_parallel:
        local_positions = local_positions - tp_rank * decoder_input.shape[0]
    selected = (local_positions >= 0) & (local_positions < decoder_input.shape[0])
    if not sequence_parallel and not selected.all():
        raise ValueError("Image span lies outside the input sequence")
    output = decoder_input.clone()
    output[local_positions[selected], 0] = image_embeddings[selected].to(output.dtype)
    # Every TP replica must fire its visual DDP hooks, even on text-only slices.
    return output + image_embeddings.sum().to(output.dtype) * 0


def image_positions(starts, grids, ratio, device):
    if len(starts) != len(grids):
        raise ValueError("Each image needs a matching span start and patch grid")
    positions = [
        torch.arange(start, start + span_length(h, w, ratio), device=device)
        for start, (h, w) in zip(starts.tolist(), grids.tolist(), strict=True)
    ]
    if not positions:
        raise ValueError("Empty V4.1 image batch")
    positions = torch.cat(positions)
    if positions.unique().numel() != positions.numel():
        raise ValueError("V4.1 image spans overlap")
    return positions


def wire_visual(model, checkpoint, *, scope, recompute=False, freeze_language=False):
    # Hard top-k indexers have no task-loss backward in the native plugin.
    # Exclude them from Adam instead of letting weight decay mimic learning.
    for name, parameter in model.named_parameters():
        if ".core_attention.indexer." in name:
            parameter.requires_grad_(False)
        elif name.endswith(".router.weight") and not freeze_language and not parameter.requires_grad:
            raise ValueError("Joint V4.1 SFT must not inherit the text recipe's frozen router gates")
    if freeze_language:
        if scope == "none":
            raise ValueError("Vision-only SFT requires trainable visual parameters")
        model.requires_grad_(False)
    original_forward = model.forward
    if model.pre_process:
        config = json.loads((Path(checkpoint) / "config.json").read_text())
        visual_config = VisionConfig.from_dict(config["vision_config"])
        with torch.device(model.embedding.word_embeddings.weight.device):
            visual = DeepSeekV41Visual(visual_config, model.config.hidden_size, recompute=recompute)
        visual.to(dtype=model.config.params_dtype)
        load_visual_weights(visual, checkpoint)
        visual.set_trainable_scope(scope, sequence_parallel=model.config.sequence_parallel)
        # Include frozen visual weights too: native saves must be self-consistent.
        model.add_module("visual", visual)

    def forward(*args, dsv41_patches=None, dsv41_grids=None, dsv41_image_positions=None, **kwargs):
        if not model.pre_process:
            return original_forward(*args, **kwargs)
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.shape[0] != 1:
            raise ValueError("V4.1 image SFT expects [1, sequence] token IDs")
        placeholder_positions = (ids[0] == model.config.v41_image_token_id).nonzero().flatten()
        if dsv41_patches is None:
            if placeholder_positions.numel() or dsv41_grids is not None or dsv41_image_positions is not None:
                raise ValueError("V4.1 image placeholders require image patches, grids and starts")
            if scope != "none":
                raise ValueError("Trainable visual SFT requires images in every micro-batch")
            return original_forward(*args, **kwargs)
        if dsv41_grids is None or dsv41_image_positions is None or kwargs.get("decoder_input") is not None:
            raise ValueError("V4.1 image SFT needs complete image metadata and no precomputed decoder input")
        positions = image_positions(dsv41_image_positions, dsv41_grids, visual_config.downsample_ratio, ids.device)
        if not torch.equal(positions, placeholder_positions):
            raise ValueError("Expanded V4.1 image spans do not match input token positions")
        decoder = model.embedding(input_ids=ids, position_ids=kwargs.get("position_ids"))
        with torch.set_grad_enabled(torch.is_grad_enabled() and scope != "none"):
            embeds = model.visual(
                dsv41_patches.to(device=decoder.device, dtype=model.config.params_dtype), dsv41_grids
            )
        from megatron.core import parallel_state

        kwargs["decoder_input"] = scatter_image_embeddings(
            decoder,
            positions,
            embeds,
            tp_rank=parallel_state.get_tensor_model_parallel_rank(),
            sequence_parallel=model.config.sequence_parallel,
        )
        return original_forward(*args, **kwargs)

    model.forward = forward


def deepseek_v41_vlm_model_provider(pre_process=True, post_process=True, vp_stage=None):
    from megatron.training import get_args

    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    args = get_args()
    if args.context_parallel_size != 1 or args.qkv_format != "bshd" or args.micro_batch_size != 1:
        raise ValueError("V4.1 image SFT currently requires CP1, BSHD and micro-batch-size 1")
    if not args.sequence_parallel and args.tensor_model_parallel_size > 1:
        raise ValueError("Replicated V4.1 visual gradients require sequence parallelism at TP>1")
    if args.use_rollout_routing_replay or args.enable_mtp_training:
        raise ValueError("V4.1 image SFT does not yet support routing replay or MTP")
    provider_args = copy(args)
    provider_args.custom_model_provider_path = None
    provider_args.dsv41_image_sft = True
    model = get_model_provider_func(provider_args, role="actor")(
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
    wire_visual(
        model,
        args.hf_checkpoint,
        scope=args.dsv41_vision_trainable,
        recompute=args.dsv41_vision_recompute,
        freeze_language=args.dsv41_freeze_language,
    )
    from miles_plugins.models.deepseek_v4_1.audit import record_inventory

    record_inventory(model)
    return model
