"""A raw Qwen3.8 HF export completed from the base checkpoint, on a small fake checkpoint with the real key layout."""

import json
import os
import re
import signal
import zlib
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.training_utils.weight_update import hf_passthrough
from miles.backends.training_utils.weight_update.hf_passthrough import (
    ExportIncomplete,
    complete_hf_export,
    main,
    verify_export,
)

LM = "model.language_model"
PLE = f"{LM}.layers.1.ple.ple_embedding"
# base shard -> keys, laid out like the real checkpoint: a PLE-only shard, shards mixing trained keys with
# PLE metadata / MTP / vision keys, and trained-only shards
BASE_LAYOUT = {
    "model-00001-of-00004.safetensors": [f"{LM}.embed_tokens.weight", "model.visual.blocks.0.attn.qkv.weight", "model.visual.merger.linear.weight"],
    "model-00002-of-00004.safetensors": [f"{PLE}.ngram_embedding.shard_0.weight", f"{PLE}.ngram_embedding.shard_1.weight"],
    "model-00003-of-00004.safetensors": [
        f"{LM}.layers.1.ple.key_proj.weight",
        f"{PLE}.layer_multipliers",
        f"{PLE}.ngram_heads_offsets",
        f"{PLE}.ngram_heads_vocab_sizes",
        "mtp.fc_hidden.weight",
        "mtp.layers.0.mlp.experts.down_proj",
    ],
    "model-00004-of-00004.safetensors": [f"{LM}.layers.0.linear_attn.out_proj.weight", f"{LM}.norm.weight", "lm_head.weight"],
}
TRAINED = [k for keys in BASE_LAYOUT.values() for k in keys if not (k.startswith(("model.visual.", "mtp.")) or ".ple_embedding." in k)]
VISUAL = [k for keys in BASE_LAYOUT.values() for k in keys if k.startswith("model.visual.")]
MTP = [k for keys in BASE_LAYOUT.values() for k in keys if k.startswith("mtp.")]


def _tensor(key: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(zlib.crc32(f"{key}/{seed}".encode()))
    if key.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")):
        return torch.randint(0, 1 << 40, (3,), generator=generator, dtype=torch.int64)
    return torch.randn(4, 6, generator=generator).to(torch.bfloat16)


def _write(directory: Path, layout: dict[str, list[str]], tensors: dict[str, torch.Tensor]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for filename, keys in layout.items():
        save_file({k: tensors[k] for k in keys}, str(directory / filename), metadata={"format": "pt"})
        weight_map.update({k: filename for k in keys})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    directory = tmp_path / "base"
    _write(directory, BASE_LAYOUT, {k: _tensor(k, 0) for keys in BASE_LAYOUT.values() for k in keys})
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    return directory


def _export(tmp_path: Path, keys, *, seed: int = 1, override: dict | None = None) -> Path:
    """What SnapshotPublisher.write_model leaves: exporter-named shards with the live model's tensors only."""
    tensors = {k: _tensor(k, seed) for k in keys} | (override or {})
    directory = tmp_path / "export"
    _write(directory, {"model-00001.safetensors": sorted(keys)}, tensors)
    return directory


def _truncate(path: Path, nbytes: int = 8) -> None:
    """Drop the last bytes of a shard's payload; the header stays valid."""
    path.write_bytes(path.read_bytes()[:-nbytes])


def _read(directory: Path, key: str) -> torch.Tensor:
    filename = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"][key]
    with safe_open(str(directory / filename), framework="pt") as f:
        return f.get_tensor(key)


class TestCompletion:
    def test_the_completed_export_has_every_base_key_with_frozen_bytes_from_the_base(self, tmp_path, base) -> None:
        """Text mode: PLE table, MTP and vision come from the base; trained keys keep the exported values."""
        export = _export(tmp_path, TRAINED)
        receipt = complete_hf_export(export, base, mtp_trained=False)

        index = json.loads((export / "model.safetensors.index.json").read_text())
        base_index = json.loads((base / "model.safetensors.index.json").read_text())
        assert set(index["weight_map"]) == set(base_index["weight_map"])
        assert receipt["passthrough_groups"] == ["mtp", "ple_table", "visual"]
        assert receipt["exported_keys"] == len(TRAINED)
        assert receipt["passthrough_keys"] == len(base_index["weight_map"]) - len(TRAINED)
        for key in base_index["weight_map"]:
            expected = _tensor(key, 1) if key in TRAINED else _tensor(key, 0)
            assert torch.equal(_read(export, key), expected), key
        # the PLE-only base shard is copied whole; the two mixed ones (vision + embed, PLE metadata + MTP + trained)
        # become passthrough shards
        assert receipt["whole_base_shards"] == ["model-00002-of-00004.safetensors"]
        assert receipt["passthrough_shards"] == 2
        assert index["metadata"]["total_size"] == sum(
            _read(export, k).numel() * _read(export, k).element_size() for k in index["weight_map"]
        )

    def test_copy_is_the_default_and_hardlink_is_opt_in(self, tmp_path, base, monkeypatch) -> None:
        """A copied shard survives a later change of the base; a link shares the base inode."""
        export = _export(tmp_path, TRAINED)
        complete_hf_export(export, base, mtp_trained=False)
        shard = "model-00002-of-00004.safetensors"
        assert os.stat(export / shard).st_ino != os.stat(base / shard).st_ino

        monkeypatch.setenv("MILES_HF_PASSTHROUGH", "hardlink")
        linked = _export(tmp_path / "second", TRAINED)
        complete_hf_export(linked, base, mtp_trained=False)
        assert os.stat(linked / shard).st_ino == os.stat(base / shard).st_ino

    def test_a_built_vision_tower_and_a_trained_mtp_are_taken_from_the_export(self, tmp_path, base) -> None:
        """Only groups the model did not export are passed through."""
        export = _export(tmp_path, TRAINED + VISUAL + MTP)
        receipt = complete_hf_export(export, base, mtp_trained=True)
        assert receipt["passthrough_groups"] == ["ple_table"]
        assert torch.equal(_read(export, "mtp.fc_hidden.weight"), _tensor("mtp.fc_hidden.weight", 1))
        assert torch.equal(_read(export, VISUAL[0]), _tensor(VISUAL[0], 1))

    def test_models_without_a_policy_are_left_alone(self, tmp_path, base) -> None:
        """Other architectures keep the plain raw export."""
        (base / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
        export = _export(tmp_path, TRAINED)
        before = (export / "model.safetensors.index.json").read_text()
        assert complete_hf_export(export, base, mtp_trained=False) is None
        assert (export / "model.safetensors.index.json").read_text() == before


class TestRefusals:
    @pytest.mark.parametrize(
        ("keys", "mtp_trained", "message"),
        [
            (TRAINED[1:], False, "neither exported nor passed through"),
            (TRAINED + ["model.language_model.layers.9.extra.weight"], False, "not in the base checkpoint"),
            (TRAINED + [f"{PLE}.ngram_embedding.shard_0.weight"], False, "group ple_table is passed through"),
            (TRAINED + MTP[:1], False, "group mtp is passed through"),
            (TRAINED, True, "group mtp is trained but 2 of its keys are missing"),
            (TRAINED + MTP[:1], True, "group mtp is trained but 1 of its keys are missing"),
            (TRAINED + VISUAL[:1], False, "group visual is trained but 1 of its keys are missing"),
        ],
    )
    def test_key_set_mismatches_fail_before_anything_is_written(self, tmp_path, base, keys, mtp_trained, message) -> None:
        """Missing trained keys, foreign keys and half-exported groups are errors, not passthroughs."""
        export = _export(tmp_path, keys)
        before = sorted(p.name for p in export.iterdir())
        with pytest.raises(ExportIncomplete, match=message):
            complete_hf_export(export, base, mtp_trained=mtp_trained)
        assert sorted(p.name for p in export.iterdir()) == before

    @pytest.mark.parametrize("change", ["dtype", "shape"])
    def test_an_exported_tensor_must_keep_the_base_dtype_and_shape(self, tmp_path, base, change) -> None:
        """A converter that changes layout or precision is caught by the header check."""
        key = f"{LM}.norm.weight"
        bad = _tensor(key, 1).float() if change == "dtype" else torch.zeros(6, 4, dtype=torch.bfloat16)
        export = _export(tmp_path, TRAINED, override={key: bad})
        with pytest.raises(ExportIncomplete, match="differ from the base in dtype or shape"):
            complete_hf_export(export, base, mtp_trained=False)

    def test_a_passthrough_byte_that_differs_from_the_base_fails(self, tmp_path, base, monkeypatch) -> None:
        """The raw-byte copy is re-read and hashed against the base."""
        original = hf_passthrough._write_raw_shard

        def corrupting(target, source, keys):
            original(target, source, keys)
            data = bytearray(target.read_bytes())
            data[-1] ^= 1
            target.write_bytes(bytes(data))

        monkeypatch.setattr(hf_passthrough, "_write_raw_shard", corrupting)
        with pytest.raises(ExportIncomplete, match="passthrough tensors differ from the base bytes"):
            complete_hf_export(_export(tmp_path, TRAINED), base, mtp_trained=False)


class TestTruncatedShards:
    """CRV-FIX-014: a shard whose payload is shorter than its header declares is refused before .complete."""

    @pytest.fixture(autouse=True)
    def _fail_instead_of_hanging(self):
        """Before the fix the raw copy spun forever on a truncated source."""

        def timeout(signum, frame):
            raise TimeoutError("the export did not return")

        previous = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(20)
        yield
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    def test_a_truncated_mixed_base_shard_fails_before_anything_is_written(self, tmp_path, base) -> None:
        """model-00003 mixes trained keys with PLE metadata and MTP, so it would be copied key by key."""
        _truncate(base / "model-00003-of-00004.safetensors")
        export = _export(tmp_path, TRAINED)
        before = sorted(p.name for p in export.iterdir())
        with pytest.raises(ExportIncomplete, match="truncated or padded"):
            complete_hf_export(export, base, mtp_trained=False)
        assert sorted(p.name for p in export.iterdir()) == before

    def test_a_truncated_trained_export_shard_fails(self, tmp_path, base) -> None:
        """Only passthrough tensors are hashed against the base; the trained ones need the size check."""
        export = _export(tmp_path, TRAINED)
        _truncate(export / "model-00001.safetensors")
        with pytest.raises(ExportIncomplete, match="model-00001.safetensors: header declares"):
            complete_hf_export(export, base, mtp_trained=False)
        assert not (export / "passthrough.json").exists()

    def test_a_tensor_whose_span_does_not_match_its_shape_fails(self, tmp_path, base) -> None:
        """A header that declares fewer bytes than dtype x shape is as broken as a short file."""
        export = _export(tmp_path, TRAINED)
        shard = export / "model-00001.safetensors"
        data_start, header = hf_passthrough.read_header(shard)
        key = sorted(header)[0]
        header[key]["shape"] = [4, 7]
        blob = json.dumps(header).encode()
        blob += b" " * (-len(blob) % 8)
        shard.write_bytes(len(blob).to_bytes(8, "little") + blob + shard.read_bytes()[data_start:])
        with pytest.raises(ExportIncomplete, match=re.escape(f"{key} spans 48 bytes, BF16[4, 7] is 56")):
            hf_passthrough.check_payloads(export, ["model-00001.safetensors"])

    def test_the_raw_copy_raises_at_an_early_end_of_file_instead_of_spinning(self, tmp_path, base) -> None:
        """The copy loop must not wait forever for bytes a truncated source does not have."""
        source = base / "model-00003-of-00004.safetensors"
        _truncate(source)
        with pytest.raises(ExportIncomplete, match="ends 8 bytes before the end of"):
            hf_passthrough._write_raw_shard(tmp_path / "out.safetensors", source, MTP)


class TestVerifyExport:
    def test_a_completed_export_verifies_and_reports_changes_by_group(self, tmp_path, base, capsys) -> None:
        """Trained keys changed, frozen groups byte-identical."""
        export = _export(tmp_path, TRAINED)
        complete_hf_export(export, base, mtp_trained=False)
        report = verify_export(export, base)
        assert report["changed"] == len(TRAINED)
        assert report["changed_by_group"]["ple_table"] == [0, 5]
        assert report["changed_by_group"]["mtp"] == [0, 2]
        assert report["changed_by_group"]["other"] == [len(TRAINED), len(TRAINED)]
        assert main([str(export), str(base)]) == 0

    def test_a_key_expected_unchanged_that_changed_fails(self, tmp_path, base) -> None:
        """--expect-unchanged pins frozen-but-exported params (e.g. the indexer) to the base bytes."""
        export = _export(tmp_path, TRAINED)
        complete_hf_export(export, base, mtp_trained=False)
        with pytest.raises(ExportIncomplete, match="must equal the base changed"):
            verify_export(export, base, expect_unchanged=[r"\.norm\.weight$"])
        assert main([str(export), str(base), "--expect-unchanged", r"\.norm\.weight$"]) == 1

    def test_an_export_of_untrained_weights_fails(self, tmp_path, base) -> None:
        """Exporting the base back unchanged means the trained weights never reached the export."""
        export = _export(tmp_path, TRAINED, seed=0)
        complete_hf_export(export, base, mtp_trained=False)
        with pytest.raises(ExportIncomplete, match="untrained weights"):
            verify_export(export, base)

    def test_unchanged_trained_keys_are_listed_not_failed(self, tmp_path, base) -> None:
        """A few steps in bf16 may leave some trained tensor bitwise equal; that is reported for review."""
        key = f"{LM}.norm.weight"
        export = _export(tmp_path, TRAINED, override={key: _tensor(key, 0)})
        complete_hf_export(export, base, mtp_trained=False)
        assert verify_export(export, base)["unchanged_trained_keys"] == [key]


class TestRawExportPath:
    def test_save_hf_model_completes_the_raw_export_from_the_base(self, tmp_path, base, monkeypatch) -> None:
        """The milestone export written during training is the completed one, before .complete is written."""
        from types import SimpleNamespace

        from miles.backends.megatron_utils import hf_export

        class Publisher:
            def write_model(self, path, *, weights, hf_checkpoint):
                _write(Path(path), {"model-00001.safetensors": sorted(TRAINED)}, {k: _tensor(k, 1) for k in TRAINED})

        monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *a, **k: [])
        monkeypatch.setattr(hf_export, "is_lora_model", lambda model: False)
        monkeypatch.setattr(hf_export.torch.distributed, "get_rank", lambda: 0)
        rank0 = SimpleNamespace(rank=0)
        monkeypatch.setattr(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=rank0, tp=rank0))
        args = SimpleNamespace(
            megatron_to_hf_mode="raw",
            hf_checkpoint=str(base),
            enable_mtp_training=False,
            save_hf=str(tmp_path / "hf" / "rollout-{rollout_id}"),
        )

        hf_export.save_hf_model(args, 7, [torch.nn.Linear(2, 2)], publisher=Publisher(), raise_on_error=True)

        export = tmp_path / "hf" / "rollout-7"
        assert (export / ".complete").exists()
        assert json.loads((export / "passthrough.json").read_text())["passthrough_groups"] == ["mtp", "ple_table", "visual"]
        assert verify_export(export, base)["changed"] == len(TRAINED)
