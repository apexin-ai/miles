"""Host syncs removed from the PLE and get_batch per-microbatch path give the same values as before (GBSMIG,
taken over from PERF H1 2026-11-01; the QSA indexer part stays with PERF Q1).

Each removed device read is replaced by the same ints from the host copy of ``cu_seqlens`` that
miles' ``get_batch`` keeps (``cu_seqlens_host``), or by a device op that needs no readback. The
tests compare the new values with the old code (kept here verbatim as the reference), bitwise,
and check that the new path does not read the device: no ``.tolist()`` / ``int(tensor)`` and no
``repeat_interleave`` without ``output_size``. Whether a GPU run really stops syncing is checked
on a GPU with ``torch.cuda.set_sync_debug_mode`` (not here: CPU tensors never sync).
"""

import itertools
import random
import math
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")

from miles.kernels.embedding import ple_gate_conv  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.ple import build_ngram_contexts_packed  # noqa: E402

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["miles-plugin"])


# boundaries the device path accepts: one row, rows + get_batch's pad segment, empty rows in the middle
CU_CASES = [
    (0, 1),
    (0, 3),
    (0, 4),
    (0, 37),
    (0, 64),
    (0, 37, 40),
    (0, 64, 128),
    (0, 37, 50, 72),
    (0, 1, 2, 3),
    (0, 5, 5, 11),
    (0, 3, 3, 7, 10),
    (0, 0, 9, 20),
    (0, 8, 8, 8, 30, 31),
]


def _random_cases(n=40, seed=0):
    rng = random.Random(seed)
    cases = []
    for _ in range(n):
        lens = [rng.choice([0, 1, 2, 3, 4, 5, 7, 8, 13, 31, 64, 97]) for _ in range(rng.randint(1, 6))]
        if sum(lens) == 0:
            lens[0] = 5
        cu = [0]
        for length in lens:
            cu.append(cu[-1] + length)
        cases.append(tuple(cu))
    return cases


ALL_CU = CU_CASES + _random_cases()


class _NoDeviceReads:
    """Fail on the reads the host path must not do: ``.tolist()``, ``int()`` / ``bool()`` of a tensor, and
    ``repeat_interleave`` without ``output_size``. The tie check's ``bool(tied.any())`` is allowed (counted)."""

    def __init__(self, monkeypatch):
        self.bool_reads = 0
        real_tolist = torch.Tensor.tolist
        real_bool = torch.Tensor.__bool__
        real_repeat = torch.repeat_interleave

        def tolist(t):
            raise AssertionError(f"device read: tolist of {tuple(t.shape)}")

        def as_int(t):
            raise AssertionError(f"device read: int() of {tuple(t.shape)}")

        def as_bool(t):
            self.bool_reads += 1
            return real_bool(t)

        def repeat(*args, **kwargs):
            if len(args) >= 2 and isinstance(args[1], torch.Tensor) and kwargs.get("output_size") is None:
                raise AssertionError("repeat_interleave by tensor repeats without output_size")
            return real_repeat(*args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "tolist", tolist)
        monkeypatch.setattr(torch.Tensor, "__int__", as_int)
        monkeypatch.setattr(torch.Tensor, "__index__", as_int)
        monkeypatch.setattr(torch.Tensor, "__bool__", as_bool)
        monkeypatch.setattr(torch, "repeat_interleave", repeat)
        self._real_tolist = real_tolist


def _old_seg_bounds(T, cu_seqlens, device):
    """``ple_gate_conv._seg_bounds`` before the change (repeat_interleave by the segment lengths)."""
    if cu_seqlens is None:
        lo = torch.zeros(T, dtype=torch.int32, device=device)
        hi = torch.full((T,), T, dtype=torch.int32, device=device)
        return lo, hi
    cu = cu_seqlens.to(torch.long)
    lens = (cu[1:] - cu[:-1]).clamp_min(0)
    lo = torch.repeat_interleave(cu[:-1], lens).to(torch.int32)
    hi = torch.repeat_interleave(cu[1:], lens).to(torch.int32)
    return lo, hi


def _window_cus():
    """PLE's own bounds: ``window_cu_seqlens`` of packs (leading filler segment, clipped rows) and plain packs."""
    from miles_plugins.models.qwen3_8_next.ops.ple import window_cu_seqlens

    out = []
    for cu in ALL_CU:
        dev = torch.tensor(cu, dtype=torch.int32)
        out.append((cu[-1] - cu[0], dev))
        total = cu[-1]
        for start, length in ((-3, 9), (0, total), (2, max(1, total // 2)), (total // 3, total - total // 3)):
            if length > 0:
                out.append((length, window_cu_seqlens(dev, total, start, length)))
    return out


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ple_segment_bounds_equal_the_old_ones(dtype):
    cases = _window_cus()
    cases.append((10, torch.tensor([3, 7, 7, 13])))  # a pack that does not start at 0
    for T, cu in cases:
        cu = cu.to(dtype)
        assert int(cu[-1] - cu[0]) == T
        want = _old_seg_bounds(T, cu, "cpu")
        got = ple_gate_conv._seg_bounds(T, cu, "cpu")
        for a, b in zip(want, got, strict=True):
            assert a.dtype == b.dtype == torch.int32 and torch.equal(a, b), (T, cu.tolist())
    for a, b in zip(_old_seg_bounds(5, None, "cpu"), ple_gate_conv._seg_bounds(5, None, "cpu"), strict=True):
        assert torch.equal(a, b)


def test_ple_segment_bounds_need_no_device_read(monkeypatch):
    cu = torch.tensor([0, 5, 5, 11, 20], dtype=torch.int32)
    with monkeypatch.context() as m:
        _NoDeviceReads(m)
        lo, hi = ple_gate_conv._seg_bounds(20, cu, "cpu")
    want = _old_seg_bounds(20, cu, "cpu")
    assert torch.equal(lo, want[0]) and torch.equal(hi, want[1])


@pytest.mark.parametrize("cu_host", [(0, 9), (0, 9, 16), (0, 4, 4, 11, 16), (0, 1, 2, 3)])
def test_ngram_contexts_from_host_bounds(monkeypatch, cu_host):
    tokens = torch.arange(100, 100 + cu_host[-1])
    cu = torch.tensor(cu_host, dtype=torch.int32)
    old = build_ngram_contexts_packed(tokens, cu, 3, 7)
    with monkeypatch.context() as m:
        _NoDeviceReads(m)
        new = build_ngram_contexts_packed(tokens, cu, 3, 7, cu_seqlens_host=cu_host)
    assert torch.equal(old, new)
    # a host copy that does not describe the same boundaries falls back to the device read
    assert torch.equal(build_ngram_contexts_packed(tokens, cu, 3, 7, cu_seqlens_host=cu_host[:-1]), old)


def test_get_batch_max_seqlen_from_host(monkeypatch):
    import miles.backends.training_utils.data.context_parallel as cp_utils_mod
    import miles.backends.training_utils.data.rollout as data_mod

    lengths = [7, 13, 5, 9, 4]

    class _Iterator:
        rollout_data = {}

        def get_next(self, keys):
            batch = {
                "tokens": [torch.arange(1, n + 1, dtype=torch.long) for n in lengths],
                "loss_masks": [torch.ones(max(1, n // 2), dtype=torch.int) for n in lengths],
                "total_lengths": list(lengths),
                "response_lengths": [max(1, n // 2) for n in lengths],
                "adapter_slots": None,
            }
            return {k: batch[k] for k in keys}

    monkeypatch.setattr(data_mod.current_platform, "current_device", lambda: torch.device("cpu"))
    for cp_size, cp_rank, allgather in itertools.product((1, 2), (0, 1), (False, True)):
        if cp_rank >= cp_size:
            continue
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(size=1))
        monkeypatch.setattr(data_mod, "get_parallel_state", lambda state=state: state)
        monkeypatch.setattr(cp_utils_mod, "get_parallel_state", lambda state=state: state)
        out = data_mod.get_batch(
            _Iterator(),
            ["tokens", "loss_masks", "total_lengths", "response_lengths"],
            pad_multiplier=8,
            qkv_format="thd",
            allgather_cp=allgather,
        )
        cu = out["cu_seqlens"]
        want = (cu[1:] - cu[:-1]).max().item()
        assert out["max_seqlen"] == want and type(out["max_seqlen"]) is type(want)
        assert tuple(cu.tolist()) == tuple(out["cu_seqlens_host"])
