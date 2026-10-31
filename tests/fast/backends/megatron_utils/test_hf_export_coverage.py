"""CPU tests for the HF export coverage check: a partial export must not be marked complete."""

import ast
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture(scope="module")
def missing_hf_weights() -> Callable:
    # Execute the production helper without importing GPU-only Megatron modules.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/hf_export.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if (isinstance(node, ast.FunctionDef) and node.name == "missing_hf_weights") or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_SAFETENSORS_INDEX" for t in node.targets))]
    namespace = {"json": json, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["missing_hf_weights"]


SOURCE = {
    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
    "lm_head.weight": "model-00002-of-00002.safetensors",
    "mtp.fc.weight": "model-00002-of-00002.safetensors",
}


def _checkpoint(root: Path, weight_map: dict[str, str] | None, shards: set[str]) -> Path:
    root.mkdir(parents=True)
    if weight_map is not None:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard in shards:
        (root / shard).write_bytes(b"")
    return root


def test_complete_export(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(tmp_path / "out", SOURCE, set(SOURCE.values()))
    assert missing_hf_weights(source, export) == []


def test_dropped_shard_takes_its_mapped_neighbours_with_it(tmp_path, missing_hf_weights):
    """Qwen3.8-27B: unmapped MTP tensors kept shard 2 from being written, and lm_head with it."""
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(
        tmp_path / "out",
        {"model.embed_tokens.weight": "model-00001-of-00002.safetensors"},
        {"model-00001-of-00002.safetensors"},
    )
    assert missing_hf_weights(source, export) == ["lm_head.weight", "mtp.fc.weight"]


def test_indexed_shard_that_was_never_written(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(tmp_path / "out", SOURCE, {"model-00001-of-00002.safetensors"})
    assert missing_hf_weights(source, export) == ["lm_head.weight", "mtp.fc.weight"]


def test_no_export_index(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", SOURCE, set())
    export = _checkpoint(tmp_path / "out", None, {"model.safetensors"})
    assert missing_hf_weights(source, export) == sorted(SOURCE)


def test_source_without_a_local_index_is_not_checked(tmp_path, missing_hf_weights):
    source = _checkpoint(tmp_path / "src", None, {"model.safetensors"})
    export = _checkpoint(tmp_path / "out", None, {"model.safetensors"})
    assert missing_hf_weights(source, export) == []
    assert missing_hf_weights("Qwen/Qwen3-8B", export) == []


# --- the collective boundary around the check (REBASE-plan, "#3691 必须补的集体异常边界") ---------------
# The check runs on rank 0 between the bridge write and a broadcast every rank waits in. These tests run
# the real save_hf_model and write_checkpoint_dir on two Gloo ranks; only the bridge write is faked.

_WORKER_TIMEOUT_S = 30
_BAD_JSON = "{not json"
_SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"


def _bridge_export_worker(rank: int, world_size: int, port: int, case: str, src: str, root: str) -> None:
    import datetime
    import os
    import time
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=_WORKER_TIMEOUT_S)
    )
    from miles.backends.megatron_utils import hf_export
    from miles.utils import distributed_utils

    distributed_utils.GLOO_GROUP = dist.group.WORLD
    hf_export.get_parallel_state = lambda: SimpleNamespace(effective_dp_cp=SimpleNamespace(rank=rank), tp=SimpleNamespace(rank=0))
    hf_export.is_lora_model = lambda model: False
    hf_export.patch_megatron_model = lambda model: nullcontext()

    class Bridge:
        def save_hf_pretrained(self, model, path):
            if rank != 0:
                return
            weight_map = dict(SOURCE)
            shards = set(SOURCE.values())
            if case == "missing_tensor":
                shards.discard("model-00002-of-00002.safetensors")
            for shard in shards:
                (Path(path) / shard).write_bytes(b"")
            index = _BAD_JSON if case == "bad_export_json" else json.dumps({"weight_map": weight_map})
            (Path(path) / _SAFETENSORS_INDEX_NAME).write_text(index)

    hf_export._get_hf_bridge = lambda hf_checkpoint: Bridge()
    if case == "unreadable_source_index":
        # root ignores mode bits, so the PermissionError is injected at the read
        read_text = Path.read_text

        def unreadable(self, *args, **kwargs):
            if self == Path(src) / _SAFETENSORS_INDEX_NAME:
                raise PermissionError(13, "Permission denied", str(self))
            return read_text(self, *args, **kwargs)

        Path.read_text = unreadable

    args = SimpleNamespace(megatron_to_hf_mode="bridge", hf_checkpoint=src, save_hf=None)
    start = time.monotonic()
    try:
        hf_export.save_hf_model(args, 0, [], publisher=None, path=Path(root) / "export", raise_on_error=True)
        error = None
    except Exception as exc:  # noqa: BLE001 - the test records what each rank raised
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - start
    after = torch.ones(1)
    dist.all_reduce(after)
    (Path(root) / f"rank{rank}.json").write_text(
        json.dumps({"error": error, "elapsed": elapsed, "next_collective": int(after.item())})
    )
    dist.destroy_process_group()


def _run_bridge_export(tmp_path: Path, case: str) -> tuple[list[dict], Path]:
    import torch

    from tests.fast.dist_utils import find_free_port

    src = tmp_path / "src"
    src.mkdir()
    index = _BAD_JSON if case == "bad_source_json" else json.dumps({"weight_map": SOURCE})
    (src / _SAFETENSORS_INDEX_NAME).write_text(index)
    root = tmp_path / "out"
    root.mkdir()
    torch.multiprocessing.spawn(
        _bridge_export_worker, args=(2, find_free_port(), case, str(src), str(root)), nprocs=2, join=True
    )
    return [json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)], root / "export"


def _assert_every_rank_failed_without_hanging(results: list[dict], export: Path, match: str) -> None:
    for rank, result in enumerate(results):
        assert result["error"] is not None and result["error"].startswith("RuntimeError: "), (rank, result)
        assert match in result["error"], (rank, result)
        assert result["elapsed"] < _WORKER_TIMEOUT_S / 2, (rank, result)
        assert result["next_collective"] == 2, (rank, result)
    assert not (export / ".complete").exists()
    assert not export.exists()  # write_checkpoint_dir removes a failed export


def test_two_ranks_publish_a_complete_bridge_export(tmp_path) -> None:
    """Positive control for the failure cases below: same harness, nothing broken."""
    results, export = _run_bridge_export(tmp_path, "complete")
    assert [r["error"] for r in results] == [None, None]
    assert all(r["next_collective"] == 2 for r in results)
    assert (export / ".complete").exists()


def test_two_ranks_refuse_a_bridge_export_missing_a_tensor(tmp_path) -> None:
    results, export = _run_bridge_export(tmp_path, "missing_tensor")
    _assert_every_rank_failed_without_hanging(results, export, "2 tensor(s)")


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("bad_source_json", "coverage check failed: JSONDecodeError"),
        ("bad_export_json", "coverage check failed: JSONDecodeError"),
        ("unreadable_source_index", "coverage check failed: PermissionError"),
    ],
)
def test_a_rank0_coverage_check_error_reaches_every_rank(tmp_path, case: str, match: str) -> None:
    """Rank 0's local check error must not leave rank 1 in the broadcast while rank 0 enters
    write_checkpoint_dir's all_gather: every rank finishes the broadcast, then all raise; no .complete."""
    results, export = _run_bridge_export(tmp_path, case)
    _assert_every_rank_failed_without_hanging(results, export, match)
