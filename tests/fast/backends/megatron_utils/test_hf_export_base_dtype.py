"""Raw HF exports keep the base checkpoint dtype of tensors that training holds wider (GDN ``A_log``)."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils import hf_export
from miles.backends.training_utils.weight_update import snapshot_publisher
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

A_LOG = "model.layers.0.linear_attn.A_log"
EMBED = "model.embed_tokens.weight"
INDEX = "model.safetensors.index.json"


class _Iterator:
    placement = SimpleNamespace(is_full_gather=True)

    def __init__(self, buckets: list[list[tuple[str, torch.Tensor]]]) -> None:
        self._buckets = buckets

    def iter_hf_weights(self, weights):
        yield from self._buckets


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot_publisher.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *args, **kwargs: [])
    monkeypatch.setattr(hf_export, "is_lora_model", lambda model: False)
    rank0 = SimpleNamespace(rank=0)
    monkeypatch.setattr(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=rank0, tp=rank0))


def _write_base(directory: Path) -> Path:
    directory.mkdir()
    save_file(
        {A_LOG: torch.zeros(8, dtype=torch.bfloat16), EMBED: torch.zeros(2, dtype=torch.bfloat16)},
        str(directory / "model-00001-of-00001.safetensors"),
    )
    weight_map = {A_LOG: "model-00001-of-00001.safetensors", EMBED: "model-00001-of-00001.safetensors"}
    (directory / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text("{}")
    return directory


def _export(tmp_path: Path, hf_checkpoint: str, buckets) -> Path:
    args = SimpleNamespace(megatron_to_hf_mode="raw", hf_checkpoint=hf_checkpoint, save_hf=None)
    target = tmp_path / "export"
    hf_export.save_hf_model(
        args, 0, [], publisher=SnapshotPublisher(_Iterator(buckets)), path=target, raise_on_error=True
    )
    return target


def _read_export(target: Path) -> dict[str, torch.Tensor]:
    weight_map = json.loads((target / INDEX).read_text())["weight_map"]
    tensors = {}
    for name, file in weight_map.items():
        with safe_open(str(target / file), framework="pt") as f:
            tensors[name] = f.get_tensor(name)
    return tensors


def _trained_a_log() -> torch.Tensor:
    # values bf16 cannot hold exactly, so a missing cast or a different rounding shows
    return torch.linspace(-3.0, 1.0, 8, dtype=torch.float32) + 1e-4


def test_raw_export_writes_a_log_in_the_base_dtype_and_leaves_other_tensors_alone(tmp_path) -> None:
    base = _write_base(tmp_path / "base")
    a_log = _trained_a_log()
    embed = torch.randn(2, dtype=torch.float32)
    ids = torch.arange(4)

    target = _export(tmp_path, str(base), [[(EMBED, embed)], [(A_LOG, a_log), ("model.ids", ids)]])

    assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
    exported = _read_export(target)
    assert exported[A_LOG].dtype == torch.bfloat16
    assert torch.equal(exported[A_LOG], a_log.to(torch.bfloat16))
    # not on the allow-list: written in the training dtype even though the base differs
    assert exported[EMBED].dtype == torch.float32 and torch.equal(exported[EMBED], embed)
    assert exported["model.ids"].dtype == torch.int64 and torch.equal(exported["model.ids"], ids)
    assert a_log.dtype == torch.float32  # the training tensor itself is not changed


def test_raw_export_without_a_local_base_keeps_the_training_dtype(tmp_path) -> None:
    a_log = _trained_a_log()
    target = _export(tmp_path, "org/remote-model", [[(A_LOG, a_log)]])

    exported = _read_export(target)
    assert exported[A_LOG].dtype == torch.float32 and torch.equal(exported[A_LOG], a_log)


def test_raw_export_fails_without_a_marker_when_the_base_header_is_unreadable(tmp_path) -> None:
    base = _write_base(tmp_path / "base")
    (base / "model-00001-of-00001.safetensors").write_bytes(b"\x00")

    with pytest.raises(struct.error):
        _export(tmp_path, str(base), [[(A_LOG, _trained_a_log())]])
    assert not (tmp_path / "export").exists()
