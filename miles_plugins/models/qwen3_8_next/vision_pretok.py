"""Image inputs for pre-tokenized Qwen3.8 SFT rows (ids rendered offline, images referenced in the row's meta).

A pre-tokenized image row already holds every token: each image is ``<|vision_start|>``, N x ``<|image_pad|>``,
``<|vision_end|>`` in ``input_ids``, with labels -100 on the pads. Training does not render or tokenize; it only
loads the images and builds the three tensors the vision provider takes (``pixel_values``, ``image_grid_thw``,
``qwen38_position_ids``). A row without images gets only its text positions: with THD packing every row of a
microbatch must carry positions, so the model can check each row against its own tokens.

The offline converter and the training-time loader both use ``load_images`` / ``image_pad_counts``, so N is the
same in both: images are resized by the same function as the dataset path of image SFT (miles'
``process_vision_info``, i.e. qwen_vl_utils with the processor's patch size).
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch

from miles.utils.processing_utils import load_processor, process_vision_info
from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions, text_rope_positions


@lru_cache(maxsize=1)
def processor_and_config(hf_checkpoint: str):
    """The checkpoint's native processor and config.json (one per process)."""
    processor = load_processor(hf_checkpoint, trust_remote_code=True)
    if processor is None:
        raise ValueError(f"{hf_checkpoint} has no image processor")
    config = json.loads((Path(hf_checkpoint) / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp" or not config.get("vision_config"):
        raise ValueError("Expected the complete Qwen3.8-Flash-Next multimodal checkpoint")
    return processor, config


def _entry_path(entry) -> tuple[str, str | None]:
    if isinstance(entry, str):
        return entry, None
    return entry["path"], entry.get("sha256")


def load_images(entries, processor):
    """The resized PIL images of ``entries`` (paths, or {"path", "sha256"}), in order.

    A given sha256 must match the file's bytes: the pad counts in the row were computed from those bytes.
    """
    content = []
    for entry in entries:
        path, sha256 = _entry_path(entry)
        if sha256 is not None:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != sha256:
                raise ValueError(f"{path}: sha256 {actual} differs from the row's {sha256}")
        content.append({"type": "image", "image": path})
    media = process_vision_info([{"role": "user", "content": content}], processor)
    images = media.get("images") or []
    if len(images) != len(entries):
        raise ValueError(f"Resolved {len(images)} images for {len(entries)} image entries")
    return images


def image_tensors(images, processor) -> tuple[torch.Tensor, torch.Tensor]:
    """pixel_values and image_grid_thw of the resized images, as the native processor makes them."""
    encoded = processor.image_processor(images=images, return_tensors="pt")
    if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
        raise ValueError("The native image processor did not return image patches and grids")
    return encoded["pixel_values"], encoded["image_grid_thw"]


def pad_counts(grids: torch.Tensor, merge_size: int) -> list[int]:
    return (torch.as_tensor(grids).prod(-1) // merge_size**2).tolist()


def image_pad_counts(entries, hf_checkpoint: str) -> list[int]:
    """How many <|image_pad|> tokens each image of ``entries`` takes (for the offline converter)."""
    processor, config = processor_and_config(hf_checkpoint)
    _, grids = image_tensors(load_images(entries, processor), processor)
    return pad_counts(grids, config["vision_config"]["spatial_merge_size"])


def image_spans(input_ids, config) -> list[tuple[int, int]]:
    """[start, end) of each run of image pads, each enclosed by vision_start / vision_end."""
    pad, start_id, end_id = config["image_token_id"], config["vision_start_token_id"], config["vision_end_token_id"]
    spans, index, n = [], 0, len(input_ids)
    while index < n:
        if input_ids[index] != pad:
            index += 1
            continue
        end = index
        while end < n and input_ids[end] == pad:
            end += 1
        if index == 0 or input_ids[index - 1] != start_id or end == n or input_ids[end] != end_id:
            raise ValueError(f"Image pads at [{index}, {end}) are not enclosed by vision_start / vision_end")
        spans.append((index, end))
        index = end
    if sum(1 for token in input_ids if token == start_id) != len(spans):
        raise ValueError("Every vision_start must open one run of image pads")
    return spans


def train_inputs_for_row(input_ids, labels, entries, hf_checkpoint: str):
    """The vision provider's inputs for one pre-tokenized row of a vision-on run.

    A text row gets ``{"qwen38_position_ids"}`` only; an image row also ``pixel_values`` and ``image_grid_thw``.
    ``input_ids`` is not changed. Checks: labels are given and as long as the row; every image entry is a
    ``{"path", "sha256"}`` whose file has that sha256; one pad run per image, the run length equals the image's
    grid product / merge², and the pads carry no SFT label.
    """
    processor, config = processor_and_config(hf_checkpoint)
    if labels is None or len(labels) != len(input_ids):
        raise ValueError("A pre-tokenized row needs labels as long as its input_ids")
    if entries is None:
        entries = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError(f"Image entries must be {{'path', 'sha256'}}, got {entry!r}")
    spans = image_spans(input_ids, config)
    if not entries:
        if spans:
            raise ValueError(f"A row without meta images holds {len(spans)} image pad runs")
        return {"qwen38_position_ids": text_rope_positions(len(input_ids)).clone()}
    if len(spans) != len(entries):
        raise ValueError(f"The row has {len(spans)} image pad runs for {len(entries)} images")
    if any(labels[i] != -100 for start, end in spans for i in range(start, end)):
        raise ValueError("Image pads must not receive SFT labels")
    pixel_values, grids = image_tensors(load_images(entries, processor), processor)
    expected = pad_counts(grids, config["vision_config"]["spatial_merge_size"])
    actual = [end - start for start, end in spans]
    if actual != expected:
        raise ValueError(f"Image pad counts {actual} differ from the images' grids {expected}")
    positions = image_rope_positions(
        input_ids,
        grids,
        image_token_id=config["image_token_id"],
        vision_start_token_id=config["vision_start_token_id"],
        vision_end_token_id=config["vision_end_token_id"],
        merge_size=config["vision_config"]["spatial_merge_size"],
    )
    return {"pixel_values": pixel_values, "image_grid_thw": grids, "qwen38_position_ids": positions}
