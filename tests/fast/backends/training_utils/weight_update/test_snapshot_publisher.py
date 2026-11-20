"""A publisher must finish the weight collectives before reporting a writer-local error."""

import json
from datetime import timedelta
from unittest.mock import patch

import pytest
import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils import distributed_utils


class _CannotStage(torch.Tensor):
    def contiguous(self):
        # A non-contiguous weight needs a second allocation when staged for safetensors.
        assert self.device.type == "cpu" and not self.is_contiguous()
        raise torch.OutOfMemoryError("cannot allocate contiguous CPU weight")


class _Buckets:
    placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, *, distributed=False, fail_staging=False):
        self.distributed = distributed
        self.fail_staging = fail_staging
        self.yielded = 0

    def iter_hf_weights(self, weights):
        for i in range(3):
            local = torch.full((2, 2), 2 * i + dist.get_rank(), dtype=torch.float32)
            shards = [local]
            if self.distributed:
                shards = [torch.empty_like(local) for _ in range(dist.get_world_size())]
                dist.all_gather(shards, local)
            weight = torch.cat(shards).T
            if self.fail_staging and i == 1:
                weight = weight.as_subclass(_CannotStage)
            self.yielded += 1
            yield [(f"weight_{i}", weight)]


@pytest.mark.parametrize("failure", ["staging", "disk"])
def test_writer_drains_remaining_buckets(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    buckets = _Buckets(fail_staging=failure == "staging")
    writes = []
    save = safetensors.torch.save_file

    def write(tensors, path):
        writes.append(path.name)
        if failure == "disk" and len(writes) == 2:
            raise OSError("disk full")
        save(tensors, path)

    monkeypatch.setattr(safetensors.torch, "save_file", write)
    error, message = (OSError, "disk full") if failure == "disk" else (torch.OutOfMemoryError, "cannot allocate")
    with pytest.raises(error, match=message):
        SnapshotPublisher(buckets).write_model(tmp_path, weights={}, hf_checkpoint=str(tmp_path / "base"))

    assert buckets.yielded == 3
    assert len(writes) == (2 if failure == "disk" else 1)


def _distributed_writer(rank, directory):
    timeout = timedelta(seconds=10)
    dist.init_process_group(
        "gloo", init_method=(directory / "store").as_uri(), rank=rank, world_size=2, timeout=timeout
    )
    try:
        # The checkpoint error exchange and the weight gathers use separate groups in production.
        control = dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
        path = directory / "checkpoint"

        def publish(buckets):
            publisher = SnapshotPublisher(buckets)
            write_checkpoint_dir(
                path,
                lambda dest: publisher.write_model(dest, weights={}, hf_checkpoint=str(directory / "base")),
                completion_marker=".complete",
            )

        with patch.object(distributed_utils, "GLOO_GROUP", control):
            buckets = _Buckets(distributed=True, fail_staging=True)
            with pytest.raises(RuntimeError) as failure:
                publish(buckets)
            outcomes = [None, None]
            dist.all_gather_object(outcomes, (buckets.yielded, str(failure.value)), group=control)
            assert [count for count, _ in outcomes] == [3, 3], outcomes
            assert all("cannot allocate contiguous CPU weight" in message for _, message in outcomes), outcomes
            if rank == 0:
                assert not path.exists()
            publish(_Buckets(distributed=True))
    finally:
        dist.destroy_process_group()


def test_all_ranks_report_staging_error_and_can_retry(tmp_path):
    mp.spawn(_distributed_writer, args=(tmp_path,), nprocs=2)
    path = tmp_path / "checkpoint"
    assert (path / ".complete").is_file()
    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert index["weight_map"].keys() == {f"weight_{i}" for i in range(3)}
    for i in range(3):
        name = f"weight_{i}"
        tensor = safetensors.torch.load_file(path / index["weight_map"][name])[name]
        torch.testing.assert_close(tensor, torch.tensor([[2 * i] * 2 + [2 * i + 1] * 2] * 2, dtype=torch.float32))
    assert index["metadata"]["total_size"] == 3 * 8 * 4
