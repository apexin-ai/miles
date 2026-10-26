"""Decode native FP8 dense/FP4 expert tensors into trainable BF16 weights.

Engram lookup tables are intentionally not mapped through this decoder. They
remain immutable native FP8 assets; their learned projections train normally.
"""

import torch

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def decode_weight(weight, scale=None, *, name=""):
    if weight.dtype in (torch.bfloat16, torch.float32, torch.float16):
        return weight
    if scale is None or weight.ndim != 2:
        raise ValueError(f"Quantized V4.1 weight needs a 2D scale: {name}")
    rows, cols = weight.shape
    if weight.dtype == torch.int8:
        if ".ffn.experts." not in name or cols * 2 % 32 or tuple(scale.shape) != (rows, cols * 2 // 32):
            raise ValueError(f"Unsupported native V4.1 FP4 layout: {name}")
        packed = weight.view(torch.uint8)
        table = torch.tensor(FP4_VALUES, dtype=torch.float32, device=weight.device)
        values = torch.stack((table[(packed & 15).long()], table[(packed >> 4).long()]), dim=-1)
        values = values.reshape(rows, -1, 32) * scale.float().unsqueeze(-1)
        return values.reshape(rows, cols * 2).bfloat16()
    if weight.dtype == torch.float8_e4m3fn:
        if rows % 32 or cols % 32 or tuple(scale.shape) != (rows // 32, cols // 32):
            raise ValueError(f"Unsupported native V4.1 FP8 block scale: {name}")
        values = weight.float().reshape(rows // 32, 32, cols // 32, 32)
        return (values * scale.float()[:, None, :, None]).reshape(rows, cols).bfloat16()
    raise ValueError(f"Unsupported V4.1 trainable dtype: {name}: {weight.dtype}")
