"""Raw HF exports restore the base dtype of GDN A_log, and only of A_log (CRV P1-INT-6).

Training keeps ``linear_attn.A_log`` in fp32; the Qwen3.8 base stores it as bf16, so the
completeness check rejected every milestone export. Other dtype differences must still fail.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils import hf_export, ple_provenance
from miles.backends.training_utils.weight_update import hf_passthrough
from miles.backends.training_utils.weight_update.hf_base_dtype import BaseDtypeCaster
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

A_LOG = "model.language_model.layers.0.linear_attn.A_log"
EMBED = "model.language_model.embed_tokens.weight"


def _write_base(directory: Path, tensors: dict[str, torch.Tensor]) -> Path:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    save_file(tensors, str(directory / "model-00001.safetensors"))
    (directory / hf_passthrough.INDEX).write_text(
        json.dumps({"metadata": {}, "weight_map": {name: "model-00001.safetensors" for name in tensors}})
    )
    return directory


def _trained_a_log() -> torch.Tensor:
    # values that bf16 cannot hold exactly, so a missing cast or a wrong rounding shows
    return torch.linspace(-3.0, 1.0, 48, dtype=torch.float32) + 1e-4


class TestBaseDtypeCaster:
    def test_a_log_is_cast_to_the_base_dtype_and_nothing_else_is(self, tmp_path) -> None:
        base = _write_base(
            tmp_path / "base",
            {A_LOG: torch.zeros(48, dtype=torch.bfloat16), EMBED: torch.zeros(2, dtype=torch.bfloat16)},
        )
        caster = BaseDtypeCaster.for_base(base)
        a_log = _trained_a_log()

        cast = caster.cast(A_LOG, a_log)
        assert cast.dtype == torch.bfloat16 and torch.equal(cast, a_log.to(torch.bfloat16))
        embed = torch.ones(2, dtype=torch.float32)
        assert caster.cast(EMBED, embed) is embed

    def test_a_base_without_the_key_or_already_matching_is_a_no_op(self, tmp_path) -> None:
        base = _write_base(tmp_path / "base", {A_LOG: torch.zeros(48, dtype=torch.float32)})
        a_log = _trained_a_log()
        assert BaseDtypeCaster.for_base(base).cast(A_LOG, a_log) is a_log
        assert BaseDtypeCaster.for_base(tmp_path / "missing").cast(A_LOG, a_log) is a_log


class TestRawExportEndToEnd:
    @pytest.fixture(autouse=True)
    def _single_process(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(ple_provenance, "_run_manifest", None)
        monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *a, **kw: [])
        monkeypatch.setattr(hf_export, "is_lora_model", lambda model: False)
        zero = SimpleNamespace(rank=0)
        monkeypatch.setattr(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=zero, tp=zero))

    def _export(self, tmp_path, exported: list[tuple[str, torch.Tensor]]) -> Path:
        base = _write_base(
            tmp_path / "base",
            {A_LOG: torch.zeros(48, dtype=torch.bfloat16), EMBED: torch.zeros(2, dtype=torch.bfloat16)},
        )
        args = SimpleNamespace(
            megatron_to_hf_mode="raw", enable_mtp_training=False, hf_checkpoint=str(base), save_hf=None
        )

        class Iterator:
            placement = SimpleNamespace(is_full_gather=True)

            def iter_hf_weights(self, weights):
                yield exported

        target = tmp_path / "export"
        hf_export.save_hf_model(args, 0, [], publisher=SnapshotPublisher(Iterator()), path=target, raise_on_error=True)
        return target

    def test_an_fp32_a_log_is_published_as_the_base_bf16(self, tmp_path) -> None:
        a_log = _trained_a_log()
        target = self._export(tmp_path, [(A_LOG, a_log), (EMBED, torch.ones(2, dtype=torch.bfloat16))])

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        weight_map = json.loads((target / hf_passthrough.INDEX).read_text())["weight_map"]
        with safe_open(str(target / weight_map[A_LOG]), framework="pt") as f:
            published = f.get_tensor(A_LOG)
        assert published.dtype == torch.bfloat16
        assert torch.equal(published, a_log.to(torch.bfloat16))

    def test_a_dtype_difference_outside_the_allowlist_still_fails(self, tmp_path) -> None:
        with pytest.raises(hf_passthrough.ExportIncomplete, match="dtype|F32"):
            self._export(tmp_path, [(A_LOG, _trained_a_log()), (EMBED, torch.ones(2, dtype=torch.float32))])
        assert not (tmp_path / "export").exists()
