"""Differentiable V4.1 ViT and 3x3 aligner, with the native checkpoint names.

Arithmetic and spatial ordering follow deepseek-ai/DeepSeek-V4.1-Flash-1017
revision 1bae0e1088993b9b82de08f59de5be5a34e42170, inference/vision.py (MIT).
Unlike the inference implementation, this path retains the autograd graph.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class VisionConfig:
    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    max_image_tokens: int = 1024
    min_pixels: int = 295936
    max_wh_ratio: float | None = None

    @classmethod
    def from_dict(cls, values):
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})


def vision_rope(height, width, head_dim, theta, device):
    dim = head_dim // 2
    if head_dim % 4:
        raise ValueError("V4.1 vision head dimension must be divisible by four")
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    phase = (torch.stack((h, w), dim=-1).reshape(-1, 2, 1).float() * inv).flatten(1).unsqueeze(1)
    return phase.cos(), phase.sin()


def apply_vision_rope(value, cos, sin):
    first, second = value.float().chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(value.dtype)


class VisionRMSNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=torch.float32))
        self.weight.keep_in_fp32 = True

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # The native vision RMSNorm keeps its affine weights in FP32 even when
        # the surrounding linears use BF16; Megatron also honors this marker.
        self.weight.data = self.weight.data.float()
        self.weight.keep_in_fp32 = True
        return self

    def forward(self, value):
        x = value.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight.float()).to(value.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches):
        return self.proj(patches.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, cos, sin):
        q, k, v = (part.reshape(len(x), self.n_heads, -1) for part in self.wqkv(x).chunk(3, dim=-1))
        q, k = apply_vision_rope(q, cos, sin), apply_vision_rope(k, cos, sin)
        value = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(value.transpose(0, 1).reshape(len(x), -1))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = VisionRMSNorm(config.hidden_size)
        self.attn = VisionAttention(config)
        self.norm2 = VisionRMSNorm(config.hidden_size)
        self.mlp = VisionMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class VisionTower(nn.Module):
    def __init__(self, config, *, recompute=False):
        super().__init__()
        self.config = config
        self.recompute = recompute
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = VisionRMSNorm(config.hidden_size)

    def forward(self, patches, height, width):
        if len(patches) != height * width:
            raise ValueError("ViT patch count does not match its spatial grid")
        x = self.patch_embed(patches)
        cos, sin = vision_rope(
            height, width, self.config.hidden_size // self.config.num_attention_heads, self.config.rope_theta, x.device
        )
        for block in self.blocks:
            if self.recompute and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, config, language_size):
        super().__init__()
        self.ratio = config.downsample_ratio
        self.w1 = nn.Linear(config.hidden_size * self.ratio**2, language_size)
        self.w2 = nn.Linear(language_size, language_size)

    def forward(self, x, height, width):
        x = x.reshape(height, width, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -width % self.ratio, 0, -height % self.ratio))
        # Channel-major unfold order is part of the pretrained aligner contract.
        x = F.unfold(x.unsqueeze(0), self.ratio, stride=self.ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


class DeepSeekV41Visual(nn.Module):
    def __init__(self, config, language_size, *, recompute=False):
        super().__init__()
        self.vision = VisionTower(config, recompute=recompute)
        self.aligner = Aligner(config, language_size)
        self.image_start = nn.Parameter(torch.zeros(language_size))
        self.image_end = nn.Parameter(torch.zeros(language_size))
        self.image_newline = nn.Parameter(torch.zeros(language_size))

    def set_trainable_scope(self, scope, *, sequence_parallel=False):
        if scope not in ("none", "projector", "full"):
            raise ValueError(f"Unknown V4.1 visual scope: {scope}")
        for name, param in self.named_parameters():
            param.requires_grad_(scope == "full" or (scope == "projector" and not name.startswith("vision.")))
            # Megatron sums the partial gradients from TP-local sequence slices.
            param.sequence_parallel = sequence_parallel
        self.vision.recompute = self.vision.recompute and scope == "full"

    def forward(self, patches, grids):
        outputs, offset = [], 0
        for height, width in grids.tolist():
            count = height * width
            features = self.aligner(self.vision(patches[offset : offset + count], height, width), height, width)
            rows = (height + self.aligner.ratio - 1) // self.aligner.ratio
            cols = (width + self.aligner.ratio - 1) // self.aligner.ratio
            features = features.reshape(rows, cols, -1)
            newline = self.image_newline.expand(rows, 1, -1)
            outputs.append(
                torch.cat(
                    (
                        self.image_start[None],
                        torch.cat((features, newline), dim=1).flatten(0, 1),
                        self.image_end[None],
                    ),
                    dim=0,
                )
            )
            offset += count
        if offset != len(patches) or not outputs:
            raise ValueError("V4.1 images require one nonempty grid per complete patch sequence")
        return torch.cat(outputs, dim=0)
