"""get_batch takes max_seqlen from the host boundaries and copies cu_seqlens to the device without blocking: no
``.max().item()`` and no pageable H2D every microbatch, with the same values as before."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils


class _HostReads(TorchDispatchMode):
    """Counts the aten ops that read a device value on the host (each one a device sync on CUDA)."""

    OPS = {torch.ops.aten._local_scalar_dense.default, torch.ops.aten.nonzero.default, torch.ops.aten.equal.default}

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += func in self.OPS
        return func(*args, **(kwargs or {}))


def _batch(monkeypatch, lengths, *, cp_size, cp_rank, allgather_cp):
    rollout = {
        "tokens": [torch.arange(1, n + 1) for n in lengths],
        "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n // 2 for n in lengths],
    }
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))
    state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    with _HostReads() as reads:
        batch = data_utils.get_batch(
            data_utils.DataIterator(rollout, micro_batch_size=len(lengths)),
            list(rollout),
            pad_multiplier=7,
            qkv_format="thd",
            allgather_cp=allgather_cp,
        )
    return batch, reads.count


@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("lengths", [[13, 11, 5, 1], [7], [1, 1, 64]])
def test_max_seqlen_and_cu_seqlens_are_unchanged_and_read_nothing_back(monkeypatch, allgather_cp, cp_size, lengths):
    for cp_rank in range(cp_size):
        batch, reads = _batch(monkeypatch, lengths, cp_size=cp_size, cp_rank=cp_rank, allgather_cp=allgather_cp)
        cu = batch["cu_seqlens"]
        # the old code: max over the device tensor, read back with .item()
        assert type(batch["max_seqlen"]) is int and batch["max_seqlen"] == (cu[1:] - cu[:-1]).max().item()
        assert cu.dtype == torch.int32 and cu.tolist() == list(batch["cu_seqlens_host"])
        assert reads == 0


def test_the_counter_sees_the_old_read_back():
    """Positive control for _HostReads: the old max_seqlen expression is one host read."""
    cu = torch.tensor([0, 3, 9], dtype=torch.int)
    with _HostReads() as reads:
        (cu[1:] - cu[:-1]).max().item()
    assert reads.count == 1


def test_a_cuda_copy_goes_from_pinned_memory_without_blocking(monkeypatch):
    calls = []

    class _Pinned:
        def to(self, device, non_blocking=False):
            calls.append((device, non_blocking))
            return "on-device"

    def pin_memory(self):
        calls.append(("pinned", self.dtype, self.tolist()))
        return _Pinned()

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin_memory)
    device = torch.device("cuda", 0)
    assert data_utils._host_ints_to_device([0, 3, 8], torch.int, device) == "on-device"
    assert calls == [("pinned", torch.int32, [0, 3, 8]), (device, True)]
    calls.clear()
    on_cpu = data_utils._host_ints_to_device([0, 3], torch.int, torch.device("cpu"))
    assert on_cpu.dtype == torch.int32 and on_cpu.tolist() == [0, 3]
    assert calls == []
