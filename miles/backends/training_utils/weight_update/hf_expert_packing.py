"""Pack per-expert HF weights into the fused expert layout of the base checkpoint.

The raw Megatron -> HF converters emit one ``...mlp.experts.{e}.{gate,up,down}_proj.weight`` triple per
expert. Some base checkpoints (e.g. Qwen3.5-MoE and later) store each MoE layer as two fused tensors
instead: ``...mlp.experts.gate_up_proj`` of shape ``[E, 2I, H]`` (gate rows, then up rows, as in
Megatron's ``linear_fc1``) and ``...mlp.experts.down_proj`` of shape ``[E, H, I]``. A raw export of such a
model should keep the base layout, so it can be loaded like the base.

Packing applies only to prefixes that the base stores fused. A layer is emitted once all its experts have
arrived, in any order; a duplicate, out-of-range or missing expert, or a packed shape that differs from
the base, is an error.
"""

import json
import re
import struct
from collections.abc import Iterable
from pathlib import Path

import torch

_SPLIT = re.compile(r"(?P<prefix>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight")
_FUSED = re.compile(r"(?P<prefix>.+\.mlp\.experts)\.(?P<kind>gate_up_proj|down_proj)")


class ExpertPackingError(RuntimeError):
    pass


def _read_safetensors_shapes(path: Path) -> dict[str, list[int]]:
    """``{name: shape}`` from a safetensors header, without reading any tensor data."""
    with open(path, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return {name: info["shape"] for name, info in header.items()}


def fused_expert_shapes(base_dir: str | Path) -> dict[str, dict[str, list[int]]]:
    """``{prefix: {"gate_up_proj": shape, "down_proj": shape}}`` of the experts the base stores fused."""
    base_dir = Path(base_dir)
    index_path = base_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        files = sorted({file for name, file in weight_map.items() if _FUSED.fullmatch(name)})
    else:
        files = sorted(file.name for file in base_dir.glob("*.safetensors"))
    fused: dict[str, dict[str, list[int]]] = {}
    for file in files:
        for name, shape in _read_safetensors_shapes(base_dir / file).items():
            if (match := _FUSED.fullmatch(name)) is not None:
                fused.setdefault(match["prefix"], {})[match["kind"]] = shape
    for prefix, kinds in fused.items():
        if set(kinds) != {"gate_up_proj", "down_proj"}:
            raise ExpertPackingError(f"base checkpoint stores only {sorted(kinds)} fused for {prefix}")
    return fused


class ExpertPacker:
    """Buffers the per-expert tensors of fused prefixes and emits each layer's two fused tensors.

    The base layout is read on the first ``pack`` call, so a broken base checkpoint is reported where
    every other packing error is.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = base_dir
        self._shapes: dict[str, dict[str, list[int]]] | None = None
        self._pending: dict[str, dict[tuple[int, str], torch.Tensor]] = {}

    def pack(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """The tensors to write now: everything but buffered experts, plus each layer that became complete."""
        if self._shapes is None:
            self._shapes = fused_expert_shapes(self._base_dir)
        out = []
        for name, tensor in named_tensors:
            match = _SPLIT.fullmatch(name)
            if match is None or match["prefix"] not in self._shapes:
                out.append((name, tensor))
            else:
                out += self._feed(match["prefix"], int(match["expert"]), match["proj"], name, tensor)
        return out

    def _feed(self, prefix: str, expert: int, proj: str, name: str, tensor: torch.Tensor):
        num_experts = self._shapes[prefix]["gate_up_proj"][0]
        if not 0 <= expert < num_experts:
            raise ExpertPackingError(f"{name}: expert {expert} is outside the base's {num_experts} experts")
        pending = self._pending.setdefault(prefix, {})
        if (expert, proj) in pending:
            raise ExpertPackingError(f"{name} was exported twice")
        pending[(expert, proj)] = tensor.detach().to("cpu")
        if len(pending) < 3 * num_experts:
            return []
        del self._pending[prefix]
        gate_up = torch.stack(
            [torch.cat([pending[(e, "gate_proj")], pending[(e, "up_proj")]], dim=0) for e in range(num_experts)]
        )
        down = torch.stack([pending[(e, "down_proj")] for e in range(num_experts)])
        packed = [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)]
        for packed_name, packed_tensor in packed:
            expected = self._shapes[prefix][packed_name.rsplit(".", 1)[1]]
            if list(packed_tensor.shape) != list(expected):
                raise ExpertPackingError(f"{packed_name}: packed shape {list(packed_tensor.shape)} != base {expected}")
        return packed

    def finish(self) -> None:
        if self._pending:
            received = {prefix: len(slots) for prefix, slots in self._pending.items()}
            raise ExpertPackingError(f"expert layers left incomplete (tensors received per prefix): {received}")
