"""Pack per-expert HF weights into the base checkpoint's fused expert layout.

The raw Megatron -> HF converters emit one ``...mlp.experts.{e}.{gate,up,down}_proj.weight``
triple per expert. Qwen3.5/3.8 base checkpoints instead store each MoE layer as two fused
tensors, ``...mlp.experts.gate_up_proj`` of shape ``[E, 2I, H]`` (gate rows, then up rows,
exactly Megatron's ``linear_fc1``) and ``...mlp.experts.down_proj`` of shape ``[E, H, I]``.
mbridge loads expert ``e`` as ``gate_up_proj[e]`` / ``down_proj[e]``; this module is the
inverse, so an export keeps the base's key set and can be loaded like the base.

Packing applies only to prefixes the base index stores fused. A layer is emitted once every
expert's three tensors have arrived; ``finish`` fails on a layer left incomplete.
"""

import json
import re
from pathlib import Path

import torch

from miles.backends.training_utils.weight_update.hf_passthrough import INDEX, read_header

_SPLIT = re.compile(r"(?P<prefix>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight")
_FUSED = re.compile(r"(?P<prefix>.+\.mlp\.experts)\.(?P<kind>gate_up_proj|down_proj)")


class ExpertPackingError(RuntimeError):
    pass


def fused_expert_shapes(base_dir) -> dict[str, dict[str, tuple[list[int], str]]]:
    """``{prefix: {"gate_up_proj" | "down_proj": (shape, safetensors dtype)}}`` for fused base experts."""
    index_path = Path(base_dir) / INDEX
    if not index_path.is_file():
        return {}
    weight_map = json.loads(index_path.read_text())["weight_map"]
    fused = {}
    headers = {}
    for name, file in weight_map.items():
        match = _FUSED.fullmatch(name)
        if match is None:
            continue
        if file not in headers:
            headers[file] = read_header(Path(base_dir) / file)[1]
        entry = headers[file][name]
        fused.setdefault(match["prefix"], {})[match["kind"]] = (entry["shape"], entry["dtype"])
    for prefix, kinds in fused.items():
        if set(kinds) != {"gate_up_proj", "down_proj"}:
            raise ExpertPackingError(f"base checkpoint stores only {sorted(kinds)} fused for {prefix}")
    return fused


class ExpertPacker:
    """Buffers per-expert tensors of fused prefixes and emits each layer's two fused tensors."""

    def __init__(self, fused_shapes: dict[str, dict[str, tuple[list[int], str]]]) -> None:
        self._shapes = fused_shapes
        self._pending: dict[str, dict[tuple[int, str], torch.Tensor]] = {}

    @classmethod
    def for_base(cls, base_dir) -> "ExpertPacker":
        return cls(fused_expert_shapes(base_dir))

    def feed(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        match = _SPLIT.fullmatch(name)
        if match is None or match["prefix"] not in self._shapes:
            return [(name, tensor)]
        prefix = match["prefix"]
        num_experts = self._shapes[prefix]["gate_up_proj"][0][0]
        expert = int(match["expert"])
        if not 0 <= expert < num_experts:
            raise ExpertPackingError(f"{name}: expert {expert} is outside the base's {num_experts} experts")
        slot = (expert, match["proj"])
        pending = self._pending.setdefault(prefix, {})
        if slot in pending:
            raise ExpertPackingError(f"{name} was exported twice")
        pending[slot] = tensor.detach().to("cpu")
        if len(pending) < 3 * num_experts:
            return []
        del self._pending[prefix]
        return self._pack(prefix, pending, num_experts)

    def _pack(self, prefix, pending, num_experts) -> list[tuple[str, torch.Tensor]]:
        gate_up = torch.stack(
            [torch.cat([pending[(e, "gate_proj")], pending[(e, "up_proj")]], dim=0) for e in range(num_experts)]
        )
        down = torch.stack([pending[(e, "down_proj")] for e in range(num_experts)])
        packed = [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)]
        for name, tensor in packed:
            expected = self._shapes[prefix][name.rsplit(".", 1)[1]][0]
            if list(tensor.shape) != list(expected):
                raise ExpertPackingError(f"{name}: packed shape {list(tensor.shape)} != base shape {expected}")
        return packed

    def finish(self) -> None:
        if self._pending:
            incomplete = {prefix: len(slots) for prefix, slots in self._pending.items()}
            raise ExpertPackingError(f"expert layers left incomplete (tensors received per prefix): {incomplete}")
