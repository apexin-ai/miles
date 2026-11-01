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


def text_rope_positions(length, *, device=None):
    """Positions of a sequence without images: every mRoPE axis carries the token index, as text RoPE does."""
    if length < 1:
        raise ValueError("A text sequence needs at least one token")
    return torch.arange(length, device=device)[:, None].expand(-1, 3)


def image_span_starts(token_ids, image_token_id):
    """Start index of every run of image pad tokens."""
    ids = torch.as_tensor(token_ids)
    image_mask = ids == image_token_id
    previous = torch.cat((image_mask.new_zeros(1), image_mask[:-1]))
    return (image_mask & ~previous).nonzero().flatten().tolist()


def packed_rope_positions(
    token_ids,
    boundaries,
    positions,
    grids,
    *,
    image_token_id,
    vision_start_token_id,
    vision_end_token_id,
    merge_size,
    pad_token_id=0,
):
    """[tokens, 3] mRoPE positions of a packed microbatch (THD rows back to back, then miles' trailing pad).

    ``boundaries`` are the cu_seqlens of the pack (rows, then the pad segment if any). ``positions`` are the rows'
    ``qwen38_position_ids`` concatenated in row order (miles concatenates them per microbatch), or None when no row
    of the microbatch has any (then no row may hold an image). Every row restarts at 0. Each row's positions must
    be exactly what its own tokens and images give (text: the token index on every axis; image: native image
    mRoPE), so a row that came without positions cannot be mistaken for another row. Only miles' trailing pad
    segment (all ``pad_token_id``) may be left uncovered; it gets 0.
    """
    ids = torch.as_tensor(token_ids)
    total = ids.shape[0]
    bounds = [int(b) for b in boundaries]
    if ids.ndim != 1 or not bounds or bounds[0] != 0 or bounds[-1] != total:
        raise ValueError(f"cu_seqlens {bounds[:4]}... do not describe a packed sequence of {total} tokens")
    if any(b <= a for a, b in zip(bounds, bounds[1:], strict=False)):
        raise ValueError("cu_seqlens must be strictly increasing")
    out = torch.zeros((total, 3), dtype=torch.long, device=ids.device)
    if positions is None:
        if image_span_starts(ids, image_token_id):
            raise ValueError("Image tokens reached the model without mRoPE positions")
        for start, end in zip(bounds, bounds[1:], strict=False):
            out[start:end] = text_rope_positions(end - start, device=ids.device)
        return out
    positions = torch.as_tensor(positions).to(device=ids.device, dtype=torch.long)
    covered = positions.shape[0]
    if positions.ndim != 2 or positions.shape[1] != 3 or covered not in bounds or covered == 0:
        raise ValueError(f"mRoPE positions of {covered} tokens do not end at a row boundary of {bounds[:4]}...")
    rows = bounds.index(covered)
    if rows != len(bounds) - 1 and (rows != len(bounds) - 2 or not bool((ids[covered:] == pad_token_id).all())):
        raise ValueError("Rows without mRoPE positions in a microbatch that has them (every row must carry them)")
    grids = torch.zeros((0, 3), dtype=torch.long) if grids is None else torch.as_tensor(grids)
    used = 0
    for start, end in zip(bounds[:rows], bounds[1 : rows + 1], strict=True):
        row = ids[start:end]
        count = len(image_span_starts(row, image_token_id))
        if count:
            expected = image_rope_positions(
                row,
                grids[used : used + count],
                image_token_id=image_token_id,
                vision_start_token_id=vision_start_token_id,
                vision_end_token_id=vision_end_token_id,
                merge_size=merge_size,
            )
        else:
            expected = text_rope_positions(end - start, device=ids.device)
        used += count
        if not torch.equal(positions[start:end], expected.to(ids.device)):
            raise ValueError(f"mRoPE positions of the row at [{start}, {end}) differ from its tokens and images")
        out[start:end] = positions[start:end]
    if used != grids.shape[0]:
        raise ValueError(f"{grids.shape[0]} image grids for {used} image spans in the microbatch")
    return out


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
