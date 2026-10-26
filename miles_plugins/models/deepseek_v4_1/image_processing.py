"""Native 1017 image resize, patchification and image-token expansion.

The grid plan and channel/patch order match the pinned DeepSeek MIT reference;
image span positions use Miles' *_positions batching contract.
"""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def image_grid(width, height, config):
    if width <= 0 or height <= 0 or config.max_image_tokens < 4:
        raise ValueError("Image dimensions and image-token budget must be positive")
    if config.max_wh_ratio is not None and width > height * config.max_wh_ratio:
        width = height * config.max_wh_ratio
    if width * height < config.min_pixels:
        ratio = (config.min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    p, r = config.patch_size, config.downsample_ratio
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p
    if span_length(best_h // p, best_w // p, r) > config.max_image_tokens:
        aspect = height / width
        cols = math.sqrt((config.max_image_tokens - 2) / aspect + 0.25) - 0.5
        rows = cols * aspect
        cell = p * r
        if cols < 1:
            best_h, best_w = (config.max_image_tokens - 2) // 2 * cell, cell
        elif rows < 1:
            best_h, best_w = cell, (config.max_image_tokens - 3) * cell
        else:
            beta = min(math.floor(cols) * cell / width, math.floor(rows) * cell / height)
            best_h, best_w = math.floor(height * beta / p) * p, math.floor(width * beta / p) * p
    if span_length(best_h // p, best_w // p, r) > config.max_image_tokens:
        raise ValueError("Native V4.1 resize plan exceeds the image-token budget")
    return best_h, best_w


def span_length(height, width, ratio):
    return math.ceil(height / ratio) * (math.ceil(width / ratio) + 1) + 2


def patchify(image, config):
    if not isinstance(image, Image.Image):
        with Image.open(Path(image)) as source:
            image = source.convert("RGB")
    else:
        image = image.convert("RGB")
    height, width = image_grid(image.width, image.height, config)
    if config.max_wh_ratio is not None and image.width >= config.max_wh_ratio * image.height:
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    p = config.patch_size
    h, w = height // p, width // p
    values = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    values = ((values - 0.5) / 0.5).to(torch.bfloat16)
    patches = values.reshape(3, h, p, w, p).permute(1, 3, 0, 2, 4).reshape(h * w, 3, p, p)
    return patches, (h, w)


def expand_image_tokens(tokens, images, image_token_id, config, *, loss_mask=None):
    count = tokens.count(image_token_id)
    if count != len(images):
        raise ValueError(f"Found {count} image placeholders but received {len(images)} images")
    if loss_mask is not None and len(loss_mask) != len(tokens):
        raise ValueError("Unexpanded SFT tokens and loss mask must have equal length")
    expanded, mask, starts, patches, grids = [], [], [], [], []
    image_iter = iter(images)
    for index, token in enumerate(tokens):
        if token != image_token_id:
            expanded.append(token)
            if loss_mask is not None:
                mask.append(loss_mask[index])
            continue
        if loss_mask is not None and loss_mask[index]:
            raise ValueError("Image spans cannot be assistant supervision targets")
        values, (h, w) = patchify(next(image_iter), config)
        starts.append(len(expanded))
        patches.append(values)
        grids.append((h, w))
        count = span_length(h, w, config.downsample_ratio)
        expanded.extend([image_token_id] * count)
        if loss_mask is not None:
            mask.extend([0] * count)
    inputs = (
        None
        if not images
        else {
            "dsv41_patches": torch.cat(patches),
            "dsv41_grids": torch.tensor(grids, dtype=torch.long),
            "dsv41_image_positions": torch.tensor(starts, dtype=torch.long),
        }
    )
    return expanded, mask if loss_mask is not None else None, inputs
