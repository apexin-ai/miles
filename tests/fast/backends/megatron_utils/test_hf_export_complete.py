"""A raw HF export holds every tensor of the base checkpoint before it is marked complete.

Runs the real ``save_hf_model`` -> ``SnapshotPublisher`` -> ``write_checkpoint_dir`` path with the
Qwen3.8-Next passthrough registration; only parameter enumeration and the weight gather are replaced.
"""

import ast
import json
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
from miles.backends.training_utils.weight_update import snapshot_publisher
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


def _write_base(directory: Path) -> Path:
    directory.mkdir()
    weight_map = {}
    for shard, keys in BASE_SHARDS.items():
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
