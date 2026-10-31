"""Qwen3.8 raw HF export: how the export fixes work together on a qwen4_exp base.

The single fixes have their own tests (test_hf_export_base_dtype, test_hf_export_fused_experts,
test_hf_export_complete). These tests cover what only the combination decides: a dtype difference off the
A_log allow-list is still refused by the completion check.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils import hf_export, ple_provenance
from miles.backends.training_utils.weight_update import hf_passthrough, snapshot_publisher
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

INDEX = "model.safetensors.index.json"
A_LOG = "model.language_model.layers.0.linear_attn.A_log"
EMBED = "model.language_model.embed_tokens.weight"


class _Iterator:
    placement = SimpleNamespace(is_full_gather=True)

    def __init__(self, buckets) -> None:
        self._buckets = buckets

    def iter_hf_weights(self, weights):
        yield from self._buckets


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot_publisher.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(hf_export.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(ple_provenance, "_run_manifest", None)
    monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *args, **kwargs: [])
    monkeypatch.setattr(hf_export, "is_lora_model", lambda model: False)
    rank0 = SimpleNamespace(rank=0)
    monkeypatch.setattr(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=rank0, tp=rank0))


def _write_base(directory: Path) -> Path:
    directory.mkdir()
    tensors = {A_LOG: torch.zeros(48, dtype=torch.bfloat16), EMBED: torch.zeros(2, dtype=torch.bfloat16)}
    save_file(tensors, str(directory / "model-00001.safetensors"))
    (directory / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": dict.fromkeys(tensors, "model-00001.safetensors")}))
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    return directory


def _trained_a_log() -> torch.Tensor:
    # values that bf16 cannot hold exactly, so a missing cast or a wrong rounding shows
    return torch.linspace(-3.0, 1.0, 48, dtype=torch.float32) + 1e-4


def _export(tmp_path: Path, bucket) -> Path:
    base = _write_base(tmp_path / "base")
    args = SimpleNamespace(megatron_to_hf_mode="raw", enable_mtp_training=False, hf_checkpoint=str(base), save_hf=None)
    target = tmp_path / "export"
    hf_export.save_hf_model(
        args, 0, [], publisher=SnapshotPublisher(_Iterator([bucket])), path=target, raise_on_error=True
    )
    return target


class TestBaseDtypeWithCompletion:
    def test_the_restored_a_log_passes_the_completion_check(self, tmp_path) -> None:
        a_log = _trained_a_log()
        target = _export(tmp_path, [(A_LOG, a_log), (EMBED, torch.ones(2, dtype=torch.bfloat16))])

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        assert (target / hf_passthrough.RECEIPT).is_file()
        weight_map = json.loads((target / INDEX).read_text())["weight_map"]
        with safe_open(str(target / weight_map[A_LOG]), framework="pt") as f:
            assert torch.equal(f.get_tensor(A_LOG), a_log.to(torch.bfloat16))

    def test_a_dtype_difference_outside_the_allow_list_still_fails(self, tmp_path) -> None:
        """The caster leaves EMBED in fp32 (base bf16); the completion check refuses it."""
        with pytest.raises(hf_passthrough.ExportIncomplete, match="differ from the base in dtype or shape"):
            _export(tmp_path, [(A_LOG, _trained_a_log()), (EMBED, torch.ones(2, dtype=torch.float32))])
        assert not (tmp_path / "export").exists()
