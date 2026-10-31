"""Restore the base checkpoint's floating dtype for tensors that training holds wider.

Some parameters are deliberately trained in a wider dtype than the base HF checkpoint stores
(e.g. gated delta net ``linear_attn.A_log`` is kept in fp32 while the base stores bf16). A raw
HF export should keep the base layout, so tensors on an explicit allow-list are cast back to
the dtype the base safetensors header declares. Everything else is written as is.
"""

import json
import re
import struct
from collections.abc import Sequence
from pathlib import Path

import torch

# Names whose training dtype intentionally differs from the base checkpoint.
DEFAULT_RESTORE_BASE_DTYPE = (re.compile(r".+\.linear_attn\.A_log"),)

_SAFETENSORS_FLOAT_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}


def read_safetensors_dtypes(path: str | Path) -> dict[str, str]:
    """``{name: safetensors dtype}`` from the file header, without reading any tensor data."""
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return {name: info["dtype"] for name, info in header.items()}


def _base_shards(base_dir: Path) -> list[Path]:
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        return [base_dir / file for file in dict.fromkeys(weight_map.values())]
    return sorted(base_dir.glob("*.safetensors"))


class BaseDtypeCaster:
    """Casts allow-listed floating tensors to the dtype their base checkpoint copy has."""

    def __init__(self, dtypes: dict[str, torch.dtype]) -> None:
        self._dtypes = dtypes

    @classmethod
    def from_checkpoint(
        cls, base_dir: str | Path, patterns: Sequence[re.Pattern] = DEFAULT_RESTORE_BASE_DTYPE
    ) -> "BaseDtypeCaster":
        dtypes = {}
        for shard in _base_shards(Path(base_dir)):
            for name, dtype in read_safetensors_dtypes(shard).items():
                if dtype in _SAFETENSORS_FLOAT_DTYPES and any(p.fullmatch(name) for p in patterns):
                    dtypes[name] = _SAFETENSORS_FLOAT_DTYPES[dtype]
        return cls(dtypes)

    def cast(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        dtype = self._dtypes.get(name)
        if dtype is None or tensor.dtype == dtype or not tensor.is_floating_point():
            return tensor
        return tensor.to(dtype)
