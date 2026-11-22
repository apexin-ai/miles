"""Ray pickle protocol 5 buffers for immutable image patches in pretokenized training shards."""

import numpy as np
import torch

# Other payloads can carry custom tensors or references that alias the image storage.
_SFT_COLUMNS = frozenset(
    {
        "tokens",
        "partition",
        "sample_indices",
        "multimodal_train_inputs",
        "total_lengths",
        "response_lengths",
        "rewards",
        "raw_reward",
        "truncated",
        "loss_masks",
        "rollout_mask_sums",
        "rollout_ids",
        "dynamic_global_batch_size",
    }
)


class _Pixels:
    def __init__(self, tensor: torch.Tensor):
        self.array = np.from_dlpack(tensor)

    def __reduce__(self):
        return _writable_tensor, (self.array,)


def _writable_tensor(array: np.ndarray) -> torch.Tensor:
    # Ray can map the pickle buffer read-only. Each trainer gets private writable storage as before.
    return torch.tensor(array, device="cpu")


def pack_train_pixels(shard: dict) -> dict:
    rows = shard.get("multimodal_train_inputs")
    if type(rows) is not list or not rows or not shard.keys() <= _SFT_COLUMNS:
        return shard
    # Known token/mask sequences contain scalar IDs. Plugins may instead supply tensors
    # for those sequences, rewards, or metadata; retain their native storage/alias graph.
    for key, values in shard.items():
        if key == "multimodal_train_inputs":
            continue
        if key == "dynamic_global_batch_size":
            if type(values) is not int:
                return shard
        elif key in ("tokens", "loss_masks"):
            if not isinstance(values, list) or any(type(seq) is not list for seq in values):
                return shard
        elif not isinstance(values, (list, tuple, range)) or any(type(x) not in (int, float, bool) for x in values):
            return shard
    storage_ids = set()
    aliased = set()
    for row in rows:
        if type(row) is not dict:
            return shard
        for tensor in row.values():
            if type(tensor) is not torch.Tensor or tensor.layout != torch.strided or tensor.device.type != "cpu":
                return shard
            pointer = tensor.untyped_storage().data_ptr()
            if pointer in storage_ids:
                aliased.add(pointer)
            storage_ids.add(pointer)

    packed = None
    for index, row in enumerate(rows):
        t = row.get("pixel_values")
        if t is None or t.dtype not in (torch.float32, torch.float16) or t.ndim != 2 or t.requires_grad:
            continue
        if (
            t.stride() != (t.shape[1], 1)
            or t.storage_offset()
            or t.is_neg()
            or t.is_conj()
            or t.__dict__
            or getattr(t, "names", (None, None)) != (None, None)  # Named tensors were removed in newer torch.
            or t.untyped_storage().nbytes() != t.numel() * t.element_size()
            or t.untyped_storage().data_ptr() in aliased
        ):
            continue
        if packed is None:
            packed = rows.copy()
        packed[index] = {**row, "pixel_values": _Pixels(t)}
    return shard if packed is None else {**shard, "multimodal_train_inputs": packed}
