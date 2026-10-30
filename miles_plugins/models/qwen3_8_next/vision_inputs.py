"""Native image spans and three-axis positions, without importing Megatron."""

from copy import deepcopy

import torch

INTEGER_FINAL_INSTRUCTION = (
    "Keep all reasoning in the thinking block. Your final answer after thinking must consist of "
    "exactly one integer and nothing else. Do not include explanations, headings, lists, units, "
    "or Markdown in the final answer."
)


def apply_final_answer_format(messages, final_answer_format):
    """Apply the same optional output instruction in training and evaluation."""
    result = deepcopy(messages)
    if final_answer_format == "dataset":
        return result
    if final_answer_format != "integer-only":
        raise ValueError("Unknown final-answer format")
    if result and result[0]["role"] == "system":
        if not isinstance(result[0]["content"], str):
            raise ValueError("An existing system instruction must be text")
        result[0]["content"] += "\n\n" + INTEGER_FINAL_INSTRUCTION
    else:
        result.insert(0, {"role": "system", "content": INTEGER_FINAL_INSTRUCTION})
    return result


def normalize_image_messages(messages, image_count):
    """Accept Miles' <image> convention or structured native image blocks."""
    normalized = deepcopy(messages)
    found = 0
    for message in normalized:
        content = message.get("content", "")
        if isinstance(content, str):
            parts = content.split("<image>")
            if len(parts) > 1:
                content = []
                for index, text in enumerate(parts):
                    if index:
                        content.append({"type": "image"})
                    if text:
                        content.append({"type": "text", "text": text})
                message["content"] = content
        if isinstance(content, list):
            for item in content:
                if item.get("type") in ("image", "image_url") or "image" in item or "image_url" in item:
                    if message["role"] != "user":
                        raise ValueError("Qwen image SFT accepts images only in user messages")
                    found += 1
                elif item.get("type") != "text":
                    raise ValueError("Qwen image SFT does not support video or audio blocks")
    if found != image_count:
        raise ValueError(f"Image blocks ({found}) do not match resolved images ({image_count})")
    return normalized


def image_rope_positions(token_ids, grids, *, image_token_id, vision_start_token_id, vision_end_token_id, merge_size):
    """Qwen3-VL image mRoPE, returned as [tokens, 3] for Miles' concatenation."""
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    grids = torch.as_tensor(grids, dtype=torch.long)
    if ids.ndim != 1 or grids.ndim != 2 or grids.shape[1] != 3 or merge_size < 1:
        raise ValueError("Expected one token sequence and [images, 3] patch grids")
    image_mask = ids == image_token_id
    previous = torch.cat((image_mask.new_zeros(1), image_mask[:-1]))
    starts = (image_mask & ~previous).nonzero().flatten().tolist()
    if len(starts) != len(grids):
        raise ValueError("Each patch grid must have exactly one complete native image span")
    chunks, cursor, next_position = [], 0, 0
    for start, (temporal, height, width) in zip(starts, grids.tolist(), strict=True):
        if temporal != 1 or min(height, width) <= 0 or height % merge_size or width % merge_size:
            raise ValueError("Image grids require T=1 and positive spatial multiples of the merge size")
        height, width = height // merge_size, width // merge_size
        count = height * width
        end = start + count
        if (
            start <= cursor
            or ids[start - 1] != vision_start_token_id
            or end >= len(ids)
            or ids[end] != vision_end_token_id
            or not image_mask[start:end].all()
        ):
            raise ValueError("Native image boundaries or expanded patch-token counts are invalid")
        text = torch.arange(start - cursor, device=ids.device) + next_position
        chunks.append(text[:, None].expand(-1, 3))
        base = next_position + len(text)
        h, w = torch.meshgrid(
            torch.arange(height, device=ids.device), torch.arange(width, device=ids.device), indexing="ij"
        )
        chunks.append(torch.stack((torch.zeros_like(h), h, w), dim=-1).reshape(-1, 3) + base)
        cursor, next_position = end, base + max(height, width)
    tail = torch.arange(len(ids) - cursor, device=ids.device) + next_position
    chunks.append(tail[:, None].expand(-1, 3))
    positions = torch.cat(chunks, dim=0)
    if positions.shape != (len(ids), 3):
        raise ValueError("Image positions do not cover the complete input")
    return positions


def scatter_image_embeddings(decoder_input, token_positions, image_embeddings, *, tp_rank=0, sequence_parallel=False):
    """Replace only image tokens, retaining gradients on every visual TP replica."""
    if decoder_input.ndim != 3 or decoder_input.shape[1] != 1:
        raise ValueError("Qwen image SFT requires micro-batch-size 1")
    if token_positions.numel() != image_embeddings.shape[0]:
        raise ValueError("Image embeddings and expanded placeholder counts differ")
    local = token_positions.to(decoder_input.device)
    if sequence_parallel:
        local = local - tp_rank * decoder_input.shape[0]
    selected = (local >= 0) & (local < decoder_input.shape[0])
    if not sequence_parallel and not selected.all():
        raise ValueError("An image token lies outside the decoder input")
    output = decoder_input.clone()
    output[local[selected], 0] = image_embeddings[selected].to(output.dtype)
    # Even a TP slice without image tokens must participate in visual gradient reduction.
    return output + image_embeddings.sum().to(output.dtype) * 0
