"""Completing a raw HF export from its base checkpoint, on a small synthetic checkpoint."""

import json
import re
import shutil
import signal
import zlib
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.training_utils.weight_update import hf_passthrough
from miles.backends.training_utils.weight_update.hf_passthrough import (
    ExportIncomplete,
    ExportPassthrough,
    PassthroughGroup,
    complete_hf_export,
    hf_export_passthrough,
    register_hf_export_passthrough,
)

INDEX = "model.safetensors.index.json"
# base shard -> keys: a shard mixing trained keys with a tower, a frozen-only shard, a shard mixing trained,
# frozen and head keys, and a trained-only shard
BASE_LAYOUT = {
    "model-00001-of-00004.safetensors": ["model.embed_tokens.weight", "tower.blocks.0.weight", "tower.proj.weight"],
    "model-00002-of-00004.safetensors": ["model.layers.1.table.shard_0.weight", "model.layers.1.table.shard_1.weight"],
    "model-00003-of-00004.safetensors": [
        "model.layers.1.proj.weight",
        "model.layers.1.table.offsets",
        "head.fc.weight",
        "head.norm.weight",
    ],
    "model-00004-of-00004.safetensors": ["model.layers.0.proj.weight", "model.norm.weight", "lm_head.weight"],
}
ALL = [k for keys in BASE_LAYOUT.values() for k in keys]
TOWER = [k for k in ALL if k.startswith("tower.")]
HEAD = [k for k in ALL if k.startswith("head.")]
FROZEN = [k for k in ALL if ".table." in k]
TRAINED = [k for k in ALL if k not in TOWER + HEAD + FROZEN]
POLICY = ExportPassthrough(
    "toy",
    {
        "frozen": PassthroughGroup(r"model\.layers\.\d+\.table\..+"),
        "tower": PassthroughGroup(r"tower\..+", only_if_absent=True),
        "head": PassthroughGroup(r"head\..+", only_if_absent=True),
    },
)


def _tensor(key: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(zlib.crc32(f"{key}/{seed}".encode()))
    if key.endswith("offsets"):
        return torch.randint(0, 1 << 40, (3,), generator=generator, dtype=torch.int64)
    return torch.randn(4, 6, generator=generator).to(torch.bfloat16)


def _write(directory: Path, layout: dict[str, list[str]], tensors: dict[str, torch.Tensor]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for filename, keys in layout.items():
        save_file({k: tensors[k] for k in keys}, str(directory / filename), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(keys, filename))
    (directory / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    directory = tmp_path / "base"
    _write(directory, BASE_LAYOUT, {k: _tensor(k, 0) for k in ALL})
    (directory / "config.json").write_text(json.dumps({"model_type": "toy"}))
    return directory


def _export(tmp_path: Path, keys, *, override: dict | None = None) -> Path:
    """What SnapshotPublisher.write_model leaves: exporter-named shards with the training model's tensors."""
    tensors = {k: _tensor(k, 1) for k in keys} | (override or {})
    directory = tmp_path / "export"
    _write(directory, {"model-00001.safetensors": sorted(keys)}, tensors)
    return directory


def _read(directory: Path, key: str) -> torch.Tensor:
    filename = json.loads((directory / INDEX).read_text())["weight_map"][key]
    with safe_open(str(directory / filename), framework="pt") as f:
        return f.get_tensor(key)


def _truncate(path: Path, nbytes: int = 8) -> None:
    """Drop the last bytes of a shard's payload; the header stays valid."""
    path.write_bytes(path.read_bytes()[:-nbytes])


class TestCompletion:
    def test_the_completed_export_has_every_base_key_with_passthrough_bytes_from_the_base(self, tmp_path, base):
        export = _export(tmp_path, TRAINED)
        receipt = complete_hf_export(export, base, POLICY)

        index = json.loads((export / INDEX).read_text())
        assert set(index["weight_map"]) == set(ALL)
        assert receipt["passthrough_groups"] == ["frozen", "head", "tower"]
        assert (receipt["exported_keys"], receipt["passthrough_keys"]) == (len(TRAINED), len(ALL) - len(TRAINED))
        for key in ALL:
            assert torch.equal(_read(export, key), _tensor(key, 1 if key in TRAINED else 0)), key
        # the frozen-only base shard is copied whole; the two mixed ones become passthrough shards
        assert receipt["whole_base_shards"] == ["model-00002-of-00004.safetensors"]
        assert receipt["passthrough_shards"] == 2
        assert index["metadata"]["total_size"] == sum(
            _read(export, k).numel() * _read(export, k).element_size() for k in ALL
        )
        assert json.loads((export / hf_passthrough.RECEIPT).read_text()) == receipt

    def test_a_tower_and_a_head_the_model_exported_are_kept_from_the_export(self, tmp_path, base):
        export = _export(tmp_path, TRAINED + TOWER + HEAD)
        receipt = complete_hf_export(export, base, POLICY)
        assert receipt["passthrough_groups"] == ["frozen"]
        for key in TOWER + HEAD:
            assert torch.equal(_read(export, key), _tensor(key, 1)), key

    def test_a_single_file_base_without_an_index(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        keys = ["model.norm.weight", "model.layers.0.table.shard_0.weight"]
        save_file({k: _tensor(k, 0) for k in keys}, str(base / "model.safetensors"))
        (base / "config.json").write_text(json.dumps({"model_type": "toy"}))
        export = _export(tmp_path, ["model.norm.weight"])

        receipt = complete_hf_export(export, base, POLICY)

        assert receipt["passthrough_groups"] == ["frozen"]
        assert torch.equal(_read(export, keys[1]), _tensor(keys[1], 0))


class TestRegistry:
    def test_the_groups_are_built_from_the_run_args_and_unknown_models_get_none(self):
        register_hf_export_passthrough(
            "toy-registry", lambda args: {"head": PassthroughGroup(r"head\..+")} if args.flag else {}
        )
        assert hf_export_passthrough("toy-registry", Namespace(flag=True)).groups == {
            "head": PassthroughGroup(r"head\..+")
        }
        assert hf_export_passthrough("toy-registry", Namespace(flag=False)).groups == {}
        assert hf_export_passthrough("not-registered", Namespace()) is None
        assert hf_export_passthrough(None, Namespace()) is None


class TestRefusals:
    @pytest.mark.parametrize(
        ("keys", "message"),
        [
            (TRAINED[1:], "neither exported nor passed through"),
            (TRAINED + ["model.layers.9.extra.weight"], "not in the base checkpoint"),
            (TRAINED + FROZEN[:1], "group frozen comes from the base but 1 keys were exported"),
            (TRAINED + TOWER[:1], "group tower is exported but 1 of its keys are missing"),
            (TRAINED + HEAD[1:], "group head is exported but 1 of its keys are missing"),
        ],
    )
    def test_key_set_mismatches_fail_before_anything_is_written(self, tmp_path, base, keys, message):
        export = _export(tmp_path, keys)
        before = sorted(p.name for p in export.iterdir())
        with pytest.raises(ExportIncomplete, match=message):
            complete_hf_export(export, base, POLICY)
        assert sorted(p.name for p in export.iterdir()) == before

    @pytest.mark.parametrize("change", ["dtype", "shape"])
    def test_an_exported_tensor_must_keep_the_base_dtype_and_shape(self, tmp_path, base, change):
        key = "model.norm.weight"
        bad = _tensor(key, 1).float() if change == "dtype" else torch.zeros(6, 4, dtype=torch.bfloat16)
        export = _export(tmp_path, TRAINED, override={key: bad})
        with pytest.raises(ExportIncomplete, match="differ from the base in dtype or shape"):
            complete_hf_export(export, base, POLICY)

    @pytest.mark.parametrize(
        ("damage", "message"),
        [
            (lambda base: (base / "config.json").unlink(), "has no config.json"),
            (lambda base: (base / "config.json").write_text('{"model_type": "other"}'), "is now model_type 'other'"),
            (lambda base: (base / INDEX).unlink(), "has neither"),
            (shutil.rmtree, "is not a local directory"),
        ],
    )
    def test_a_base_that_is_no_longer_the_one_the_run_started_from_fails(self, tmp_path, base, damage, message):
        export = _export(tmp_path, TRAINED)
        damage(base)
        with pytest.raises(ExportIncomplete, match=message):
            complete_hf_export(export, base, POLICY)

    def test_a_passthrough_byte_that_differs_from_the_base_fails(self, tmp_path, base, monkeypatch):
        """The copied bytes are read back and hashed against the base."""
        original = hf_passthrough._write_raw_shard

        def corrupting(target, source, keys):
            original(target, source, keys)
            data = bytearray(target.read_bytes())
            data[-1] ^= 1
            target.write_bytes(bytes(data))

        monkeypatch.setattr(hf_passthrough, "_write_raw_shard", corrupting)
        with pytest.raises(ExportIncomplete, match="passthrough tensors differ from the base bytes"):
            complete_hf_export(_export(tmp_path, TRAINED), base, POLICY)


class TestTruncatedShards:
    @pytest.fixture(autouse=True)
    def _fail_instead_of_hanging(self):
        """A copy loop that waits for bytes a truncated file does not have would never return."""

        def timeout(signum, frame):
            raise TimeoutError("the export did not return")

        previous = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(20)
        yield
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    def test_a_truncated_mixed_base_shard_fails_before_anything_is_written(self, tmp_path, base):
        _truncate(base / "model-00003-of-00004.safetensors")
        export = _export(tmp_path, TRAINED)
        before = sorted(p.name for p in export.iterdir())
        with pytest.raises(ExportIncomplete, match="truncated or padded"):
            complete_hf_export(export, base, POLICY)
        assert sorted(p.name for p in export.iterdir()) == before

    def test_a_truncated_export_shard_fails(self, tmp_path, base):
        """Only passthrough tensors are hashed against the base; the exported ones need the size check."""
        export = _export(tmp_path, TRAINED)
        _truncate(export / "model-00001.safetensors")
        with pytest.raises(ExportIncomplete, match="model-00001.safetensors: header declares"):
            complete_hf_export(export, base, POLICY)
        assert not (export / hf_passthrough.RECEIPT).exists()

    def test_a_tensor_whose_span_does_not_match_its_shape_fails(self, tmp_path):
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

    def test_the_raw_copy_raises_at_an_early_end_of_file(self, tmp_path, base):
        source = base / "model-00003-of-00004.safetensors"
        _, header = hf_passthrough.read_header(source)
        last = max(header, key=lambda k: header[k]["data_offsets"][1])
        _truncate(source)
        with pytest.raises(ExportIncomplete, match=f"ends 8 bytes before the end of {re.escape(last)}"):
            hf_passthrough._write_raw_shard(tmp_path / "out.safetensors", source, [last])
