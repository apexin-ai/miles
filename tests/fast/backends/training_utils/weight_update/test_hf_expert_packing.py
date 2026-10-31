"""Packing per-expert HF tensors into the fused expert layout of the base checkpoint."""

import json
import random
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.training_utils.weight_update.hf_expert_packing import (
    ExpertPacker,
    ExpertPackingError,
    fused_expert_shapes,
)

NUM_EXPERTS, HIDDEN, INTER = 3, 4, 2
PREFIX = "model.layers.0.mlp.experts"
EMBED = "model.embed_tokens.weight"


def _experts(seed: int = 0) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(gate [I, H], up [I, H], down [H, I]) per expert, distinct values."""
    gen = torch.Generator().manual_seed(seed)
    return {
        e: (
            torch.randn(INTER, HIDDEN, generator=gen),
            torch.randn(INTER, HIDDEN, generator=gen),
            torch.randn(HIDDEN, INTER, generator=gen),
        )
        for e in range(NUM_EXPERTS)
    }


def _split(experts) -> list[tuple[str, torch.Tensor]]:
    out = []
    for e, (gate, up, down) in experts.items():
        out += [
            (f"{PREFIX}.{e}.gate_proj.weight", gate),
            (f"{PREFIX}.{e}.up_proj.weight", up),
            (f"{PREFIX}.{e}.down_proj.weight", down),
        ]
    return out


def _fused(experts) -> dict[str, torch.Tensor]:
    return {
        f"{PREFIX}.gate_up_proj": torch.stack([torch.cat([g, u]) for g, u, _ in experts.values()]),
        f"{PREFIX}.down_proj": torch.stack([d for _, _, d in experts.values()]),
    }


def _write_base(directory: Path, tensors: dict[str, torch.Tensor], *, index: bool = True) -> Path:
    directory.mkdir()
    save_file(tensors, str(directory / "model-00001-of-00001.safetensors"))
    if index:
        weight_map = dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
        (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return directory


@pytest.mark.parametrize("index", [True, False])
def test_the_base_layout_is_read_from_the_headers(tmp_path, index):
    base = _write_base(tmp_path / "base", _fused(_experts()) | {EMBED: torch.zeros(2)}, index=index)
    assert fused_expert_shapes(base) == {
        PREFIX: {"gate_up_proj": [NUM_EXPERTS, 2 * INTER, HIDDEN], "down_proj": [NUM_EXPERTS, HIDDEN, INTER]}
    }


def test_any_arrival_order_and_bucketing_packs_into_the_base_layout(tmp_path):
    experts = _experts()
    packer = ExpertPacker(_write_base(tmp_path / "base", _fused(experts)))
    split = _split(experts)
    random.Random(0).shuffle(split)

    emitted = []
    for i in range(0, len(split), 2):
        emitted += packer.pack(split[i : i + 2])
    packer.finish()

    assert [name for name, _ in emitted] == [f"{PREFIX}.gate_up_proj", f"{PREFIX}.down_proj"]
    for name, tensor in emitted:
        assert torch.equal(tensor, _fused(experts)[name])


def test_other_tensors_and_split_bases_are_left_alone(tmp_path):
    experts = _experts()
    embed = torch.ones(2)
    fused_base = ExpertPacker(_write_base(tmp_path / "fused", _fused(experts)))
    assert fused_base.pack([(EMBED, embed)]) == [(EMBED, embed)]

    split_base = ExpertPacker(_write_base(tmp_path / "split", dict(_split(experts))))
    assert split_base.pack(_split(experts)) == _split(experts)
    split_base.finish()


def test_a_missing_expert_fails_at_finish(tmp_path):
    experts = _experts()
    packer = ExpertPacker(_write_base(tmp_path / "base", _fused(experts)))
    assert packer.pack([item for item in _split(experts) if ".experts.2." not in item[0]]) == []
    with pytest.raises(ExpertPackingError, match="left incomplete"):
        packer.finish()


def test_a_duplicate_or_out_of_range_expert_is_rejected(tmp_path):
    packer = ExpertPacker(_write_base(tmp_path / "base", _fused(_experts())))
    packer.pack([(f"{PREFIX}.0.down_proj.weight", torch.zeros(HIDDEN, INTER))])
    with pytest.raises(ExpertPackingError, match="twice"):
        packer.pack([(f"{PREFIX}.0.down_proj.weight", torch.zeros(HIDDEN, INTER))])
    with pytest.raises(ExpertPackingError, match="outside"):
        packer.pack([(f"{PREFIX}.{NUM_EXPERTS}.down_proj.weight", torch.zeros(HIDDEN, INTER))])


def test_a_packed_shape_that_differs_from_the_base_is_rejected(tmp_path):
    experts = _experts()
    packer = ExpertPacker(_write_base(tmp_path / "base", _fused(experts)))
    wide = {e: (torch.cat([g, g], dim=1), torch.cat([u, u], dim=1), d) for e, (g, u, d) in experts.items()}
    with pytest.raises(ExpertPackingError, match="!= base"):
        packer.pack(_split(wide))


def test_a_base_with_only_half_of_a_fused_layer_is_rejected(tmp_path):
    base = _write_base(tmp_path / "base", {f"{PREFIX}.down_proj": torch.zeros(NUM_EXPERTS, HIDDEN, INTER)})
    with pytest.raises(ExpertPackingError, match=r"stores only \['down_proj'\] fused"):
        ExpertPacker(base).pack([(EMBED, torch.ones(2))])
