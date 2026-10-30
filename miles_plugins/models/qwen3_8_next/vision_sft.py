"""Whole-conversation SFT masks aligned with Qwen's actual expanded image tokens."""

import json
from functools import lru_cache
from pathlib import Path

from miles.utils.processing_utils import load_processor
from miles.utils.types import Sample
from miles_plugins.models.qwen3_8_next.vision_inputs import (
    apply_final_answer_format,
    image_rope_positions,
    normalize_image_messages,
)


@lru_cache(maxsize=1)
def _processor(checkpoint):
    processor = load_processor(checkpoint, trust_remote_code=True)
    if processor is None:
        raise ValueError("Qwen image SFT needs the checkpoint's native image processor")
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or not config.get("vision_config"):
        raise ValueError("Expected the complete Qwen3.8-Flash-Next multimodal checkpoint")
    return processor, config


def encode_sft_sample(
    processor, config, messages, images, *, template_kwargs=None, tools=None, final_answer_format="dataset"
):
    normalized = normalize_image_messages(messages, len(images))
    if not images or not normalized or normalized[-1]["role"] != "assistant":
        raise ValueError("Image SFT needs real images and a final assistant target")
    normalized = apply_final_answer_format(normalized, final_answer_format)
    options = dict(template_kwargs or {})
    if any(key in options for key in ("tokenize", "add_generation_prompt", "tools")):
        raise ValueError("Template options cannot override SFT rendering or tool schemas")
    tokenizer = processor.tokenizer

    def render(conversation, *, generation=False):
        return tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=generation, tools=tools, **options
        )

    text = render(normalized)
    spans = []
    for index, message in enumerate(normalized):
        if message["role"] != "assistant" or not message.get("step_loss_mask", 1):
            continue
        prefix = render(normalized[:index], generation=True)
        complete = render(normalized[: index + 1])
        if not text.startswith(complete) or not complete.startswith(prefix):
            raise ValueError("This conversation/template does not preserve assistant prefix boundaries")
        spans.append((len(prefix), len(complete)))
    raw = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    raw_mask = [
        int(any(start <= left < right <= end for start, end in spans)) for left, right in raw["offset_mapping"]
    ]
    encoded = processor(text=[text], images=images, return_tensors="pt", add_special_tokens=False)
    if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
        raise ValueError("The native processor did not return image patches and grids")
    grids = encoded["image_grid_thw"]
    merge = config["vision_config"]["spatial_merge_size"]
    counts = iter((grids.prod(-1) // merge**2).tolist())
    expected, mask = [], []
    for token, active in zip(raw["input_ids"], raw_mask, strict=True):
        if token == config["image_token_id"]:
            count = next(counts, None)
            if count is None or active:
                raise ValueError("Image placeholders must match grids and must not receive SFT loss")
            expected.extend([token] * count)
            mask.extend([0] * count)
        else:
            expected.append(token)
            mask.append(active)
    tokens = encoded["input_ids"][0].tolist()
    if next(counts, None) is not None or tokens != expected or not any(mask):
        raise ValueError("Native processor tokens differ from the whole-conversation loss alignment")
    positions = image_rope_positions(
        tokens,
        grids,
        image_token_id=config["image_token_id"],
        vision_start_token_id=config["vision_start_token_id"],
        vision_end_token_id=config["vision_end_token_id"],
        merge_size=merge,
    )
    inputs = {
        "pixel_values": encoded["pixel_values"],
        "image_grid_thw": grids,
        "qwen38_position_ids": positions,
    }
    return tokens, mask, inputs


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    if evaluation or not args.rollout_global_dataset:
        raise ValueError("Qwen image SFT consumes a global supervised dataset, not generated rollouts")
    processor, config = _processor(args.hf_checkpoint)
    samples = []
    for (sample,) in data_buffer.get_samples(args.rollout_batch_size):
        if not isinstance(sample.prompt, list):
            raise ValueError("Pass messages to image SFT without --apply-chat-template")
        images = (sample.multimodal_inputs or {}).get("images", [])
        tokens, mask, inputs = encode_sft_sample(
            processor,
            config,
            sample.prompt,
            images,
            template_kwargs=args.apply_chat_template_kwargs,
            tools=sample.metadata.get("tools"),
            final_answer_format=getattr(args, "qwen38_sft_final_answer_format", "dataset"),
        )
        if len(tokens) > args.seq_length:
            raise ValueError(f"Expanded sample {sample.index} exceeds --seq-length: {len(tokens)}")
        response_length = len(mask) - mask.index(1)
        sample.tokens = tokens
        sample.loss_mask = mask[-response_length:]
        sample.response_length = response_length
        sample.multimodal_train_inputs = inputs
        sample.reward = 0
        sample.response = processor.tokenizer.decode(tokens[-response_length:], skip_special_tokens=False)
        sample.status = Sample.Status.COMPLETED
        samples.append([sample])
    return samples
