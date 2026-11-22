import hashlib
import pickle

import numpy as np
import pytest
import torch

from miles.ray.rollout.train_pixels import pack_train_pixels


def _sha(value):
    return hashlib.sha256(value).digest()


def _wire(value):
    buffers = []
    header = pickle.dumps(value, protocol=5, buffer_callback=buffers.append)
    return pickle.loads(header, buffers=buffers), buffers


def test_all_64_shards_preserve_order_bytes_and_private_writable_pixels():
    for i in range(64):
        pixels = torch.arange(48, dtype=torch.float32).reshape(4, 12).clone().add_(i)
        grid = torch.tensor([[1, 2, 2]])
        shard = {
            "sample_indices": [i],
            "tokens": [[i, i + 1]],
            "multimodal_train_inputs": [
                {"pixel_values": pixels, "image_grid_thw": grid},
                {"qwen38_position_ids": grid},
            ],
        }
        ref, _ = _wire(shard)
        out, buffers = _wire(pack_train_pixels(shard))
        assert len(buffers) == 1
        assert pixels.untyped_storage().resizable()
        assert list(out) == list(ref)
        assert out["sample_indices"] == ref["sample_indices"]
        assert _sha(np.array(out["tokens"]).tobytes()) == _sha(np.array(ref["tokens"]).tobytes())
        for actual, expected in zip(out["multimodal_train_inputs"], ref["multimodal_train_inputs"], strict=True):
            assert list(actual) == list(expected)
            for name, tensor in actual.items():
                assert type(tensor) is type(expected[name])
                assert tensor.dtype == expected[name].dtype and tensor.stride() == expected[name].stride()
                assert _sha(tensor.clone().numpy().tobytes()) == _sha(expected[name].clone().numpy().tobytes())
        out["multimodal_train_inputs"][0]["pixel_values"].zero_()
        assert torch.count_nonzero(pixels) > 0
        out["multimodal_train_inputs"][0]["pixel_values"].resize_(100)


@pytest.mark.parametrize("kind", ["text", "view", "grad", "alias", "custom", "reward_alias", "tuple_rows"])
def test_unsupported_shards_keep_native_representation(kind):
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4).clone()
    row = {"pixel_values": tensor}
    shard = {"multimodal_train_inputs": [row]}
    if kind == "text":
        row.clear()
    elif kind == "view":
        row["pixel_values"] = tensor[:, ::2]
    elif kind == "grad":
        tensor.requires_grad_()
    elif kind == "alias":
        row["image_grid_thw"] = tensor.view(-1)
    elif kind == "reward_alias":
        shard["raw_reward"] = [tensor]
    elif kind == "tuple_rows":
        shard["multimodal_train_inputs"] = (row,)
    else:
        shard["custom"] = tensor
    assert pack_train_pixels(shard) is shard
