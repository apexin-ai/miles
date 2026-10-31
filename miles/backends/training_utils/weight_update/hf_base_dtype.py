"""Restore the base checkpoint's dtype for tensors training deliberately holds wider.

Qwen3.5/3.8 GDN keeps ``linear_attn.A_log`` in fp32 (mbridge upcasts it on load), but the
base checkpoint stores it as bf16, so a raw HF export must cast it back to load like the
base. Only names on the allowlist are cast; any other dtype difference is still an error
for the export completeness check.
"""

import json
import re
from pathlib import Path

import torch

from miles.backends.training_utils.weight_update.hf_passthrough import INDEX, read_header

# tensors whose training dtype intentionally differs from the base checkpoint
RESTORE_BASE_DTYPE = re.compile(r".+\.linear_attn\.A_log")

_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}


class BaseDtypeError(RuntimeError):
    pass


def base_dtypes(base_dir, pattern: re.Pattern = RESTORE_BASE_DTYPE) -> dict[str, torch.dtype]:
    """``{name: torch dtype}`` for base tensors that match ``pattern``."""
    index_path = Path(base_dir) / INDEX
    if not index_path.is_file():
        return {}
    weight_map = json.loads(index_path.read_text())["weight_map"]
    headers = {}
    dtypes = {}
    for name, file in weight_map.items():
        if pattern.fullmatch(name) is None:
            continue
        if file not in headers:
            headers[file] = read_header(Path(base_dir) / file)[1]
        declared = headers[file][name]["dtype"]
        if declared not in _SAFETENSORS_DTYPES:
            raise BaseDtypeError(f"{name}: unsupported base dtype {declared}")
        dtypes[name] = _SAFETENSORS_DTYPES[declared]
    return dtypes


class BaseDtypeCaster:
    def __init__(self, dtypes: dict[str, torch.dtype]) -> None:
        self._dtypes = dtypes

    @classmethod
    def for_base(cls, base_dir) -> "BaseDtypeCaster":
        return cls(base_dtypes(base_dir))

    def cast(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        dtype = self._dtypes.get(name)
        return tensor if dtype is None or tensor.dtype == dtype else tensor.to(dtype)
