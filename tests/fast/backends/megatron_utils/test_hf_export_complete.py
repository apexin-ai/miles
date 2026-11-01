"""A raw HF export holds every tensor of the base checkpoint before it is marked complete.

Runs the real ``save_hf_model`` -> ``SnapshotPublisher`` -> ``write_checkpoint_dir`` path with the
Qwen3.8-Next passthrough registration; only parameter enumeration and the weight gather are replaced.
"""

import ast
import json
import re
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tests.fast.source_scan import FRAMEWORK_ROOT

from miles.backends.megatron_utils import hf_export
from miles.backends.megatron_utils.megatron_to_hf import qwen3_8_next  # noqa: F401  (registers the passthrough)
from miles.backends.training_utils.weight_update import hf_passthrough, snapshot_publisher
from miles.backends.training_utils.weight_update.hf_passthrough import hf_export_passthrough
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

INDEX = "model.safetensors.index.json"
LM = "model.language_model"
EMBED = f"{LM}.embed_tokens.weight"
LM_HEAD = "lm_head.weight"
PLE_SHARD = f"{LM}.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
PLE_MULTIPLIERS = f"{LM}.layers.0.ple.ple_embedding.layer_multipliers"
MTP = "mtp.fc.weight"
VISUAL = "model.visual.merger.linear.weight"
BASE_SHARDS = {
    "model-00001-of-00002.safetensors": [EMBED, PLE_SHARD, PLE_MULTIPLIERS],
    "model-00002-of-00002.safetensors": [MTP, VISUAL, LM_HEAD],
}


class _Iterator:
    placement = SimpleNamespace(is_full_gather=True)

    def __init__(self, buckets, *, collective: bool = False) -> None:
        self._buckets = buckets
        self._collective = collective

    def iter_hf_weights(self, weights):
        for bucket in self._buckets:
            if self._collective:  # stands in for the TP/EP weight gather
                torch.distributed.all_reduce(torch.ones(1))
            yield bucket


def _base_tensor(key: str) -> torch.Tensor:
    if key == PLE_MULTIPLIERS:
        return torch.arange(3, dtype=torch.int64)
    return torch.full((2, 4), float(len(key)), dtype=torch.bfloat16)


def _write_base(directory: Path, shards: dict[str, list[str]] = BASE_SHARDS) -> Path:
    directory.mkdir()
    weight_map = {}
    for shard, keys in shards.items():
        save_file({k: _base_tensor(k) for k in keys}, str(directory / shard))
        weight_map.update(dict.fromkeys(keys, shard))
    (directory / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    return directory


def _trained() -> list[tuple[str, torch.Tensor]]:
    return [(EMBED, torch.ones(2, 4, dtype=torch.bfloat16)), (LM_HEAD, torch.ones(2, 4, dtype=torch.bfloat16) * 2)]


def _args(base: Path, *, mtp: bool = False) -> Namespace:
    return Namespace(megatron_to_hf_mode="raw", hf_checkpoint=str(base), save_hf=None, enable_mtp_training=mtp)


def _publisher(args: Namespace, buckets, **kwargs) -> SnapshotPublisher:
    # what the Megatron actor builds when the run starts
    return SnapshotPublisher(_Iterator(buckets, **kwargs), export_passthrough=hf_export_passthrough("qwen4_exp", args))


def _read(target: Path) -> dict[str, torch.Tensor]:
    out = {}
    for name, file in json.loads((target / INDEX).read_text())["weight_map"].items():
        with safe_open(str(target / file), framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out


def _patch_single_process(patch) -> None:
    rank0 = SimpleNamespace(rank=0)
    patch(hf_export, "named_params_and_buffers", lambda *args, **kwargs: [])
    patch(hf_export, "is_lora_model", lambda model: False)
    patch(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=rank0, tp=rank0))


class TestSingleRank:
    @pytest.fixture(autouse=True)
    def _single_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(snapshot_publisher.dist, "get_rank", lambda: 0)
        _patch_single_process(monkeypatch.setattr)

    def test_a_text_only_export_is_completed_from_the_base(self, tmp_path):
        base = _write_base(tmp_path / "base")
        args = _args(base)
        target = tmp_path / "export"

        hf_export.save_hf_model(
            args, 0, [], publisher=_publisher(args, [_trained()]), path=target, raise_on_error=True
        )

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        exported = _read(target)
        assert set(exported) == {k for keys in BASE_SHARDS.values() for k in keys}
        for name, tensor in _trained():
            assert torch.equal(exported[name], tensor)
        for name in (PLE_SHARD, PLE_MULTIPLIERS, MTP, VISUAL):
            assert torch.equal(exported[name], _base_tensor(name)), name
        receipt = json.loads((target / "passthrough.json").read_text())
        assert receipt["passthrough_groups"] == ["mtp", "ple_table", "visual"]

    @pytest.mark.parametrize(
        ("mtp", "buckets", "message"),
        [
            (False, [_trained()[:1]], "neither exported nor passed through"),
            (True, [_trained()], "neither exported nor passed through"),
            (False, [_trained() + [(PLE_SHARD, _base_tensor(PLE_SHARD))]], "group ple_table comes from the base"),
        ],
        ids=["missing-trained-tensor", "trained-mtp-not-exported", "frozen-table-exported"],
    )
    def test_an_export_that_does_not_cover_the_base_is_never_marked_complete(self, tmp_path, mtp, buckets, message):
        base = _write_base(tmp_path / "base")
        args = _args(base, mtp=mtp)
        target = tmp_path / "export"

        with pytest.raises(Exception, match=message):
            hf_export.save_hf_model(args, 0, [], publisher=_publisher(args, buckets), path=target, raise_on_error=True)
        assert not target.exists()

    def test_a_base_whose_config_disappeared_after_the_run_started_fails_the_export(self, tmp_path):
        base = _write_base(tmp_path / "base")
        args = _args(base)
        publisher = _publisher(args, [_trained()])
        (base / "config.json").unlink()
        target = tmp_path / "export"

        with pytest.raises(Exception, match="has no config.json"):
            hf_export.save_hf_model(args, 0, [], publisher=publisher, path=target, raise_on_error=True)
        assert not target.exists()

    def test_a_model_without_a_passthrough_keeps_the_plain_raw_export(self, tmp_path):
        base = _write_base(tmp_path / "base")
        target = tmp_path / "export"

        hf_export.save_hf_model(
            _args(base), 0, [], publisher=SnapshotPublisher(_Iterator([_trained()])), path=target, raise_on_error=True
        )

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        assert set(_read(target)) == {EMBED, LM_HEAD}


SHARD = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")
# one whole base shard (the PLE table) between two mixed ones, so the completed export has all three kinds
MIXED_BASE_SHARDS = {
    "model-00001-of-00003.safetensors": [EMBED, MTP],
    "model-00002-of-00003.safetensors": [PLE_SHARD, PLE_MULTIPLIERS],
    "model-00003-of-00003.safetensors": [VISUAL, LM_HEAD],
}


def _assert_uniform(target: Path) -> dict[str, str]:
    """Every indexed shard is model-i-of-N with i = 1..N, and no other safetensors file is in the directory."""
    weight_map = json.loads((target / INDEX).read_text())["weight_map"]
    files = sorted(set(weight_map.values()))
    matches = [SHARD.fullmatch(f) for f in files]
    assert all(matches), files
    assert [int(m.group(1)) for m in matches] == list(range(1, len(files) + 1)), files
    assert {int(m.group(2)) for m in matches} == {len(files)}, files
    assert sorted(p.name for p in target.glob("*.safetensors")) == files
    return weight_map


def _write_export(directory: Path, layout: dict[str, list[str]]) -> None:
    directory.mkdir()
    weight_map = {}
    for shard, keys in layout.items():
        save_file({k: _base_tensor(k) for k in keys}, str(directory / shard))
        weight_map.update(dict.fromkeys(keys, shard))
    (directory / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


class TestShardNames:
    """SMOKE REPO-3: a completed export mixed model-NNNNN, the base's model-NNNNN-of-00131 and passthrough-NNNNN,
    so a tool counting shards by -of- saw 31 of 131. Every raw export is named model-i-of-N, trained shards first."""

    @pytest.fixture(autouse=True)
    def _single_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(snapshot_publisher.dist, "get_rank", lambda: 0)
        _patch_single_process(monkeypatch.setattr)

    def test_a_completed_export_is_named_model_i_of_n_trained_shards_first(self, tmp_path):
        base = _write_base(tmp_path / "base", MIXED_BASE_SHARDS)
        args = _args(base)
        target = tmp_path / "export"

        hf_export.save_hf_model(args, 0, [], publisher=_publisher(args, [_trained()]), path=target, raise_on_error=True)

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        weight_map = _assert_uniform(target)
        receipt = json.loads((target / "passthrough.json").read_text())
        order = [*receipt["exported_shards"], *receipt["whole_base_shards"], *receipt["passthrough_files"]]
        assert receipt["whole_base_shards"] == ["model-00002-of-00003.safetensors"]  # where it came from
        assert receipt["passthrough_files"] == ["passthrough-00001.safetensors", "passthrough-00002.safetensors"]
        total = len(order)
        expected = {old: f"model-{i:05d}-of-{total:05d}.safetensors" for i, old in enumerate(order, 1)}
        assert receipt["shard_renames"] == {old: new for old, new in expected.items() if old != new}
        for name, _ in _trained():  # the exporter's shards come first
            assert int(SHARD.fullmatch(weight_map[name]).group(1)) <= len(receipt["exported_shards"])
        assert weight_map[PLE_SHARD] == expected["model-00002-of-00003.safetensors"]
        exported = _read(target)
        for name, tensor in _trained():
            assert torch.equal(exported[name], tensor)
        for name in (PLE_SHARD, PLE_MULTIPLIERS, MTP, VISUAL):
            assert torch.equal(exported[name], _base_tensor(name)), name
        assert hf_passthrough.verify_export(target, base)["changed"] == len(_trained())

    def test_a_raw_export_without_a_passthrough_is_named_too(self, tmp_path):
        base = _write_base(tmp_path / "base")
        target = tmp_path / "export"

        hf_export.save_hf_model(
            _args(base), 0, [], publisher=SnapshotPublisher(_Iterator([_trained()])), path=target, raise_on_error=True
        )

        weight_map = _assert_uniform(target)
        assert set(weight_map) == {EMBED, LM_HEAD}
        assert not (target / "passthrough.json").exists()

    def test_a_safetensors_file_outside_the_index_fails_the_export(self, tmp_path, monkeypatch):
        base = _write_base(tmp_path / "base")
        args = _args(base)
        real = snapshot_publisher.complete_hf_export

        def leave_a_stray_file(export_dir, *a, **k):
            receipt = real(export_dir, *a, **k)
            save_file({"x": torch.zeros(1)}, str(Path(export_dir) / "stray.safetensors"))
            return receipt

        monkeypatch.setattr(snapshot_publisher, "complete_hf_export", leave_a_stray_file)
        target = tmp_path / "export"
        with pytest.raises(Exception, match="stray.safetensors"):
            hf_export.save_hf_model(args, 0, [], publisher=_publisher(args, [_trained()]), path=target, raise_on_error=True)
        assert not target.exists()  # write_checkpoint_dir removed the failed export

    def test_a_target_name_held_by_another_shard_is_renamed_in_two_steps(self, tmp_path):
        """The base shard model-00001-of-00002 must become model-00002-of-00002 and give its name to the trained one."""
        target = tmp_path / "export"
        _write_export(target, {"model-00001.safetensors": [EMBED], "model-00001-of-00002.safetensors": [MTP]})
        before = _read(target)
        receipt = {
            "exported_shards": ["model-00001.safetensors"],
            "whole_base_shards": ["model-00001-of-00002.safetensors"],
            "passthrough_files": [],
        }
        (target / hf_passthrough.RECEIPT).write_text(json.dumps(receipt))

        renames = hf_passthrough.normalize_shard_names(target, receipt)

        assert renames == {
            "model-00001.safetensors": "model-00001-of-00002.safetensors",
            "model-00001-of-00002.safetensors": "model-00002-of-00002.safetensors",
        }
        assert _assert_uniform(target) == {EMBED: "model-00001-of-00002.safetensors", MTP: "model-00002-of-00002.safetensors"}
        after = _read(target)
        assert all(torch.equal(after[k], before[k]) for k in before)
        assert json.loads((target / hf_passthrough.RECEIPT).read_text())["shard_renames"] == renames
        assert not list(target.glob("*.tmp")) and not list(target.glob(".rename-*"))

    def test_normalizing_twice_changes_nothing(self, tmp_path):
        target = tmp_path / "export"
        _write_export(target, {"model-00001.safetensors": [EMBED], "model-00002.safetensors": [MTP]})
        hf_passthrough.normalize_shard_names(target)
        index = (target / INDEX).read_text()
        inodes = {p.name: p.stat().st_ino for p in target.glob("*.safetensors")}

        assert hf_passthrough.normalize_shard_names(target) == {}
        assert (target / INDEX).read_text() == index
        assert {p.name: p.stat().st_ino for p in target.glob("*.safetensors")} == inodes


def test_the_actor_builds_the_publisher_with_the_passthrough_of_the_model_it_trains():
    source = (FRAMEWORK_ROOT / "backends" / "megatron_utils" / "actor.py").read_text()
    (call,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SnapshotPublisher"
    ]
    (keyword,) = [k for k in call.keywords if k.arg == "export_passthrough"]
    assert ast.unparse(keyword.value) == "hf_export_passthrough(self.hf_config.model_type, args)"


def _two_rank_worker(rank: int, world_size: int, port: int, base: str, target: str, out: str) -> None:
    from tests.fast.dist_utils import init_gloo

    from miles.utils import distributed_utils

    init_gloo(rank, world_size, port=port)
    distributed_utils.GLOO_GROUP = torch.distributed.new_group(backend="gloo")
    _patch_single_process(setattr)
    args = _args(Path(base))
    publisher = _publisher(args, [_trained()[:1]], collective=True)
    try:
        hf_export.save_hf_model(args, 0, [], publisher=publisher, path=target, raise_on_error=True)
        error = None
    except Exception as exc:  # noqa: BLE001 - the test records which rank failed and why
        error = f"{type(exc).__name__}: {exc}"
    after = torch.ones(1)
    torch.distributed.all_reduce(after)
    Path(out, f"rank{rank}.json").write_text(json.dumps({"error": error, "next_collective": int(after.item())}))
    torch.distributed.destroy_process_group()


def test_a_completion_failure_on_the_writer_fails_every_rank_without_a_marker(tmp_path):
    from tests.fast.dist_utils import find_free_port

    base = _write_base(tmp_path / "base")
    target = tmp_path / "export"
    torch.multiprocessing.spawn(
        _two_rank_worker, args=(2, find_free_port(), str(base), str(target), str(tmp_path)), nprocs=2, join=True
    )

    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert results[0]["error"].startswith("ExportIncomplete")
    assert results[1]["error"].startswith("RuntimeError: Checkpoint write failed: rank 0: ExportIncomplete")
    assert all(r["next_collective"] == 2 for r in results)
    assert not target.exists()
