"""Select the storage dtype for raw HF exports.

Legacy exports cast allow-listed tensors back to their base storage dtype. Qwen3.8
GDN decays keep the FP32 values used for training and offline DCP exports.
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


def checkpoint_model_type(directory: Path) -> str | None:
    config = directory / "config.json"
    return json.loads(config.read_text()).get("model_type") if config.is_file() else None


def native_export_dtype(model_type: str | None, name: str, base_dtype: str) -> str | None:
    """Wider native storage explicitly supported by both the HF writer and the verifier."""
    if (
        model_type == "qwen4_exp"
        and base_dtype in {"BF16", "F16", "F32"}
        and re.fullmatch(r"model\.language_model\.layers\.\d+\.linear_attn\.A_log", name)
    ):
        return "F32"
    return None


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
        base_dir = Path(base_dir)
        model_type = checkpoint_model_type(base_dir)
        dtypes = {}
        for shard in _base_shards(base_dir):
            for name, dtype in read_safetensors_dtypes(shard).items():
                if (
                    dtype in _SAFETENSORS_FLOAT_DTYPES
                    and any(p.fullmatch(name) for p in patterns)
                    and native_export_dtype(model_type, name, dtype) is None
                ):
                    dtypes[name] = _SAFETENSORS_FLOAT_DTYPES[dtype]
        return cls(dtypes)

    def cast(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        dtype = self._dtypes.get(name)
        if dtype is None or tensor.dtype == dtype or not tensor.is_floating_point():
            return tensor
        return tensor.to(dtype)
