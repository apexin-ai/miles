"""Attach Qwen's complete native ViT before Megatron DDP/optimizer construction."""

import json
from pathlib import Path

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from miles_plugins.models.qwen3_8_next.vision_host_lengths import install_host_cu_seqlens
from miles_plugins.models.qwen3_8_next.vision_inputs import (
    image_span_starts,
    packed_rope_positions,
    scatter_image_embeddings,
)

# What the training data holds: "image" = every microbatch has images; "text" = none does (the tower is held but
# frozen and never run); "mixed" = a microbatch (one row, or several rows packed by THD) may have either.
VISION_DATA_MODES = ("image", "text", "mixed")


class _ZeroGradsFor(torch.autograd.Function):
    """Passes the input through unchanged; its backward gives each parameter an exact zero gradient.

    With --overlap-grad-reduce, Megatron DDP reduces a bucket only after every parameter in it reported a gradient in
    the last microbatch of the step. A trainable tower skipped on a text microbatch would never report.

    The result is not a view of the input (Megatron refuses views of tensors it frees, e.g. deallocate_output_tensor).
    """

    @staticmethod
    def forward(ctx, output, *parameters):
        ctx.shapes = [(p.shape, p.dtype, p.device) for p in parameters]
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        zeros = (torch.zeros((), dtype=dtype, device=device).expand(shape) for shape, dtype, device in ctx.shapes)
        return (grad_output, *zeros)


def _zero_grads_in_tower_backward_order(hidden, parameters):
    """``hidden`` unchanged; its backward reports a zero gradient for each parameter, last parameter first.

    One autograd node per parameter, chained in registration order: autograd runs a node's gradient accumulation
    (sequence number "max") before the chain's earlier node, so the reports come in reverse registration order. That
    is the order of Megatron DDP's buckets (laid out over the reversed parameter list), and the tower's own backward
    on an image microbatch completes its buckets in that order too: it runs the modules last to first, and only swaps
    inside a block (a block's norm2 reports before its attention) reorder parameters, never a whole bucket. A single
    node for every parameter would report them in the order of autograd's ready-queue heap instead.
    """
    for parameter in parameters:
        hidden = _ZeroGradsFor.apply(hidden, parameter)
    return hidden


def check_overlap_bucket_layout(model_chunks):
    """With --overlap-grad-reduce: refuse a DDP layout that splits one tower module's trainable parameters over two
    buckets.

    A text microbatch reports the tower's zero gradients in reverse registration order (the bucket order); the
    tower's own backward on an image microbatch reports a module's parameters in autograd's order (a LayerNorm or
    Conv3d reports its weight before its bias, a Linear its bias before its weight). Inside one bucket that makes no
    difference. A bucket boundary between such two parameters can complete the two buckets in opposite orders on a
    text and an image microbatch (e.g. the patch embedding's bias closes one bucket and its weight the next, which
    also holds the language model's output layer), and the DP ranks deadlock. Another --ddp-bucket-size moves the
    boundary.
    """
    for chunk in model_chunks:
        ddp_config = getattr(chunk, "ddp_config", None)  # config-access-exempt: only Megatron DDP chunks have one
        if ddp_config is None or not ddp_config.overlap_grad_reduce:
            continue
        bucket_of = {}
        for index, buffer in enumerate(chunk.buffers):
            for param, (_, _, bucket) in buffer.param_index_map.items():
                bucket_of[param] = (index, bucket)
        for name, module in chunk.named_modules():
            if "vision_model" not in name.split("."):
                continue
            held = {bucket_of[p] for p in module.parameters(recurse=False) if p in bucket_of}
            if len(held) > 1:
                raise ValueError(
                    f"--overlap-grad-reduce: the DDP buckets split the trainable parameters of {name} "
                    f"(buffer, bucket) {sorted(held)}; text and image microbatches would issue the reduce-scatters "
                    "in different orders and deadlock. Choose another --ddp-bucket-size."
                )


def set_visual_scope(visual, scope, *, sequence_parallel=False):
    if scope not in ("none", "projector", "full"):
        raise ValueError(f"Unknown Qwen visual scope: {scope}")
    for name, parameter in visual.named_parameters():
        parameter.requires_grad_(scope == "full" or (scope == "projector" and name.startswith("merger.")))
        parameter.sequence_parallel = sequence_parallel


def wire_visual(model, hf_config, *, scope, recompute, freeze_language, data="image"):
    if data not in VISION_DATA_MODES:
        raise ValueError(f"Unknown Qwen vision data mode: {data}")
    if data == "text" and scope != "none":
        # A tower that is never run but still in the optimizer gets zero grads, and AdamW weight decay shrinks it.
        raise ValueError("Text-only data needs a frozen vision tower: run with --qwen38-vision-trainable none")
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
        install_host_cu_seqlens(visual)  # sdpa splits per image on host lengths: one sync per tower forward
        # The existing Qwen38NextBridge maps this exact prefix, including every frozen tensor.
        model.add_module("vision_model", visual)
    original_forward = model.forward

    def forward(*args, pixel_values=None, image_grid_thw=None, qwen38_position_ids=None, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Qwen image SFT requires a single packed input sequence per microbatch")
        if kwargs.get("decoder_input") is not None or kwargs.get("position_ids") is not None:
            raise ValueError("Qwen image SFT owns image injection and native mRoPE positions")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("Qwen image SFT requires genuine image patches and grids together")
        # every integer check below runs on one host copy of the token ids (one device sync per microbatch
        # instead of one per check); the results go back with a non-blocking copy, values unchanged
        ids_host = ids[0].to("cpu")
        has_images = bool(image_span_starts(ids_host, hf_config["image_token_id"]))
        if data == "image" and not has_images:
            raise ValueError("Qwen image SFT requires genuine image patches, grids and mRoPE positions")
        if has_images and data == "text":
            raise ValueError("Image inputs reached a model configured with --qwen38-vision-data text")
        if has_images != (pixel_values is not None):
            raise ValueError("Image tokens and image patches must come together in a microbatch")
        packed = kwargs.get("packed_seq_params")
        boundaries, rows = ((0, ids.shape[1]), 1) if packed is None else _host_boundaries(packed, ids.shape[1])
        # One row, or THD rows back to back: each row restarts at 0 (text) or has its native image mRoPE.
        positions = packed_rope_positions(
            ids_host,
            boundaries,
            qwen38_position_ids,
            image_grid_thw,
            rows=rows,
            image_token_id=hf_config["image_token_id"],
            vision_start_token_id=hf_config["vision_start_token_id"],
            vision_end_token_id=hf_config["vision_end_token_id"],
            merge_size=hf_config["vision_config"]["spatial_merge_size"],
        )
        kwargs["position_ids"] = _to_device(positions.T.unsqueeze(1).contiguous(), ids.device)
        if not has_images:
            # A text microbatch: the vision tower does not run.
            trainable = [p for p in model.vision_model.parameters() if p.requires_grad] if model.pre_process else []
            if trainable and torch.is_grad_enabled():
                # The tower's zero gradients are reported where an image microbatch merges the tower's output into
                # the input embeddings, so they become ready at the same point of the backward and in the same
                # order as on an image microbatch: after the whole language model, before the word embeddings.
                # With --overlap-grad-reduce every DP rank must issue its bucket reduce-scatters in one order; a DP
                # rank on a text microbatch and one on an image microbatch otherwise deadlock (F8, 2026-11-02).
                kwargs["decoder_input"] = _zero_grads_in_tower_backward_order(
                    _text_decoder_input(model, ids, kwargs), trainable
                )
            return original_forward(*args, **kwargs)
        if model.pre_process:
            from megatron.core import parallel_state

            decoder = model.embedding(input_ids=ids, position_ids=None)
            with torch.set_grad_enabled(torch.is_grad_enabled() and scope != "none"):
                result = model.vision_model(
                    pixel_values.to(device=decoder.device, dtype=model.config.params_dtype),
                    grid_thw=image_grid_thw,
                    return_dict=True,
                )
            # images in row order, rows in pack order: the tower's outputs line up with the pad tokens
            image_positions = _to_device((ids_host == hf_config["image_token_id"]).nonzero().flatten(), ids.device)
            kwargs["decoder_input"] = scatter_image_embeddings(
                decoder,
                image_positions,
                result.pooler_output,
                tp_rank=parallel_state.get_tensor_model_parallel_rank(),
                sequence_parallel=model.config.sequence_parallel,
            )
        return original_forward(*args, **kwargs)

    model.forward = forward


def _text_decoder_input(model, ids, kwargs):
    """The input embeddings GPTModel._preprocess computes for ``ids`` (no learned positions: mRoPE), as the image path
    computes them."""
    if kwargs.get("padding_mask") is not None:
        raise ValueError("Qwen image SFT does not take a padding_mask: GPTModel skips it once decoder_input is given")
    scatters = getattr(  # config-access-exempt: older Megatron embeddings have no switch and always scatter
        model.embedding, "scatter_to_sequence_parallel", True
    )
    if model.config.sequence_parallel and not scatters:
        raise ValueError("Qwen image SFT needs the embedding to scatter its output to the sequence-parallel region")
    return model.embedding(input_ids=ids, position_ids=None)


def _to_device(host_tensor, device):
    """A host tensor on ``device`` without waiting for the device: pinned memory, non-blocking copy (the caching
    host allocator keeps the pinned buffer until the copy has run)."""
    if host_tensor.device == device:
        return host_tensor
    if device.type == "cuda":
        return host_tensor.pin_memory().to(device, non_blocking=True)
    return host_tensor.to(device)


def _host_boundaries(packed_seq_params, total):
    """The pack's cu_seqlens (host copy miles keeps, no device sync) and how many of its segments are rows."""
    host = getattr(packed_seq_params, "cu_seqlens_host", None)
    if host is None:
        host = packed_seq_params.cu_seqlens_q.tolist()
    bounds = tuple(int(b) for b in host)
    if bounds[-1] != total:
        raise ValueError(f"cu_seqlens end at {bounds[-1]}, the microbatch has {total} tokens (CP must be 1)")
    # rows and get_batch's trailing pad cannot be told apart from token values (a real row may be all token 0)
    rows = getattr(packed_seq_params, "sample_count", None)
    if rows is None:
        raise ValueError("Packed Qwen image SFT needs packed_seq_params.sample_count (miles get_batch sets it)")
    return bounds, int(rows)


def get_qwen38_vision_model_provider(pre_process=True, post_process=True, vp_stage=None, *, args):
    from miles.backends.megatron_utils.model_provider import get_model_provider_func
    from miles_plugins.models.qwen3_8_next.model_provider import _install_ple_context_hooks

    if args.context_parallel_size != 1 or args.qkv_format != "thd":
        raise ValueError("Qwen image SFT currently requires CP1, THD (rows are packed by cu_seqlens)")
    if args.tensor_model_parallel_size > 1 and not args.sequence_parallel:
        raise ValueError("Qwen image SFT requires sequence parallelism at TP>1")
    if args.use_rollout_routing_replay or args.enable_mtp_training or args.virtual_pipeline_model_parallel_size:
        raise ValueError("Qwen image SFT does not support routing replay, MTP or interleaved pipelines")
    if args.position_embedding_type != "mrope":
        # rope fusion is fine: attention takes MCore's raw mRoPE freqs, the QSA indexer a materialized table
        raise ValueError("Qwen image SFT needs mRoPE positions (--position-embedding-type mrope)")
    config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or config.get("language_model_only"):
        raise ValueError("Expected a complete native Qwen3.8-Flash-Next model")
    provider_args = type(args).model_validate(dict(args) | {"custom_model_provider_path": None})
    model = get_model_provider_func(provider_args)(
        pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
    )
    wire_visual(
        model,
        config,
        scope=args.qwen38_vision_trainable,
        recompute=args.qwen38_vision_recompute,
        freeze_language=args.qwen38_freeze_language,
        data=args.qwen38_vision_data,
    )
    _install_ple_context_hooks(model)
    return model
