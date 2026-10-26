"""Image SFT with native V4.1 rendering and whole-conversation loss offsets."""

from copy import deepcopy
from functools import lru_cache

from miles.utils.types import Sample
from miles_plugins.models.deepseek_v4_1.image_processing import expand_image_tokens


@lru_cache(maxsize=1)
def _processor(checkpoint):
    from miles_plugins.models.deepseek_v4_1.processor import DeepSeekV41Processor

    return DeepSeekV41Processor(checkpoint)


def encode_sft_sample(processor, messages, images, *, reasoning_effort="high"):
    from sglang.srt.entrypoints.openai import encoding_dsv41 as native

    normalized = processor.normalize_messages(messages, images)
    prepared, extracted = native.process_image_messages(deepcopy(normalized))
    if len(extracted) != len(images):
        raise ValueError("Native image blocks do not match the resolved training inputs")
    prepared = native.sort_tool_results_by_call_order(native.merge_tool_messages(prepared))
    chunks, spans = [native.bos_token], []
    length = len(chunks[0])
    for index, message in enumerate(prepared):
        chunk = native.render_message(
            index, prepared, thinking_mode="thinking", drop_thinking=False, reasoning_effort=reasoning_effort
        )
        if message["role"] == "assistant" and message.get("step_loss_mask", 1):
            spans.append((length, length + len(chunk)))
        chunks.append(chunk)
        length += len(chunk)
    text = "".join(chunks)
    expected = native.encode_messages(
        deepcopy(normalized), thinking_mode="thinking", drop_thinking=False, reasoning_effort=reasoning_effort
    )
    if text != expected:
        raise ValueError("SFT span rendering differs from the native V4.1 conversation")
    encoded = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    masks = [
        int(any(start <= left < right <= end for start, end in spans)) for left, right in encoded["offset_mapping"]
    ]
    tokens, masks, inputs = expand_image_tokens(
        encoded["input_ids"], images, processor.image_token_id, processor.config, loss_mask=masks
    )
    if not any(masks) or inputs is None:
        raise ValueError("Image SFT requires both genuine images and assistant supervision")
    return tokens, masks, inputs


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    if evaluation or not args.rollout_global_dataset:
        raise ValueError("V4.1 image SFT uses an existing global supervised dataset, not generation")
    processor = _processor(args.hf_checkpoint)
    samples = []
    for (sample,) in data_buffer.get_samples(args.rollout_batch_size):
        if not isinstance(sample.prompt, list):
            raise ValueError("Image SFT needs messages, not an already rendered prompt")
        if sample.metadata.get("tools"):
            raise ValueError("Image SFT tool schemas must be part of the native system message")
        images = (sample.multimodal_inputs or {}).get("images", [])
        tokens, mask, inputs = encode_sft_sample(
            processor,
            sample.prompt,
            images,
            reasoning_effort=args.apply_chat_template_kwargs.get("reasoning_effort", "high"),
        )
        if len(tokens) > args.seq_length:
            raise ValueError(f"Expanded image SFT sample {sample.index} exceeds --seq-length: {len(tokens)}")
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
