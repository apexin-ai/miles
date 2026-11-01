"""Host syncs removed from the Qwen3.8-Next QSA indexer give the same values as before.

Each removed device read is replaced by the same ints from the host copy of ``cu_seqlens`` that
miles' ``get_batch`` keeps (``cu_seqlens_host``), or by a device op that needs no readback. The
tests compare the new values with the old code (kept here verbatim as the reference), bitwise,
and check that the new path does not read the device: no ``.tolist()`` / ``int(tensor)`` and no
``repeat_interleave`` without ``output_size``. Whether a GPU run really stops syncing is checked
on a GPU with ``torch.cuda.set_sync_debug_mode`` (not here: CPU tensors never sync).
"""

import itertools
import math
import random
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops import qsa_indexer  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.attention import Qwen38NextAttention, qsa_selected_blocks  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import HostRowBounds, PackedBlockLayout  # noqa: E402

register_cpu_ci(est_time=60, suite="stage-b-cpu", labels=["miles-plugin"])

RATIO = 4
HIDDEN = 64

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


def _device_row_bounds(cu_host, total, ratio):
    """The device path of ``iter_selected_blocks`` before the change: positions, layout, then ``.tolist()``."""
    cu = torch.tensor(cu_host, dtype=torch.int32)
    positions = Qwen38NextAttention._packed_positions(cu, total)
    if cu.numel() > 2:
        layout = PackedBlockLayout(cu, positions, ratio)
        row_lo = layout.token_block_start
    else:
        row_lo = torch.zeros_like(positions)
    row_hi = row_lo + (positions + 1) // ratio
    return row_lo.tolist(), row_hi.tolist()


def _usable_total(cu):
    # rows past cu[-1] belong to the last segment, as on the device; inner starts must be < total
    return [t for t in (cu[-1], cu[-1] + 3) if t > 0 and HostRowBounds.usable(cu, t)]


@pytest.mark.parametrize("cu", ALL_CU)
def test_host_row_bounds_equal_the_device_ones(cu):
    for total in _usable_total(cu):
        want_lo, want_hi = _device_row_bounds(cu, total, RATIO)
        host = HostRowBounds(cu, total, RATIO)
        assert len(host.lo) == len(host.hi) == total
        assert [host.lo[i] for i in range(total)] == want_lo
        assert [host.hi[i] for i in range(total)] == want_hi


def test_host_row_bounds_refuse_what_the_device_path_rejects():
    assert not HostRowBounds.usable(None, 10)
    assert not HostRowBounds.usable((0,), 10)
    assert not HostRowBounds.usable((0, 6, 4, 10), 10)  # decreasing
    assert not HostRowBounds.usable((0, 10, 10), 10)  # an inner start at the row count indexes past it
    assert HostRowBounds.usable((0, 10), 10)
    with pytest.raises(ValueError):
        HostRowBounds((0, 10, 10), 10, RATIO)


@pytest.mark.parametrize("cu", [c for c in ALL_CU if len(c) > 2])
def test_packed_layout_with_host_bounds_is_identical(cu):
    total = cu[-1]
    if not HostRowBounds.usable(cu, total):
        pytest.skip("boundaries the device path rejects")
    dev = torch.tensor(cu, dtype=torch.int32)
    positions = Qwen38NextAttention._packed_positions(dev, total)
    old = PackedBlockLayout(dev, positions, RATIO)
    new = PackedBlockLayout(dev, positions, RATIO, cu_seqlens_host=cu)
    assert new.num_blocks == old.num_blocks
    for name in ("token_seq", "token_block", "token_block_start", "token_start", "block_seq", "block_local"):
        a, b = getattr(old, name), getattr(new, name)
        assert a.dtype == b.dtype and torch.equal(a, b), name
    for name in ("block_first_token", "block_last_token"):
        assert torch.equal(getattr(old, name), getattr(new, name)), name


class _DuplicatedLinear(torch.nn.Module):
    """CPU stand-in for TELinear(parallel_mode="duplicated")."""

    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) / math.sqrt(input_size))

    def forward(self, x):
        return x @ self.weight.t(), None


def _indexer(monkeypatch, tile_bytes, budget):
    monkeypatch.setattr(qsa_indexer, "TELinear", _DuplicatedLinear)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        init_method=None,
        params_dtype=torch.float32,
        layernorm_epsilon=1e-6,
        sequence_parallel=False,
        qwen3_8_next_indexer_n_heads=4,
        qwen3_8_next_indexer_kv_heads=1,
        qwen3_8_next_indexer_head_dim=32,
        qwen3_8_next_indexer_budget=budget,
        qwen3_8_next_indexer_compress_ratio=RATIO,
        qwen3_8_next_indexer_tile_bytes=tile_bytes,
    )
    torch.manual_seed(0)
    indexer = qsa_indexer.Qwen38NextQSAIndexer(config, layer_number=1)
    with torch.no_grad():
        indexer.q_layernorm.normal_(0, 0.1)
        indexer.k_layernorm.normal_(0, 0.1)
    return indexer


def _inputs(cu_host, total, rotary_dim=32):
    cu = torch.tensor(cu_host, dtype=torch.int32)
    positions = Qwen38NextAttention._packed_positions(cu, total)
    torch.manual_seed(1)
    hidden = torch.randn(total, HIDDEN)
    inv_freq = 1.0 / (1e7 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(total + 1, dtype=torch.float32), inv_freq)
    rope = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]
    return hidden, positions, rope, cu


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


SELECTION_CASES = [
    ((0, 37), 64),
    ((0, 200), 64),
    ((0, 200, 256), 64),
    ((0, 37, 50, 72), 32),
    ((0, 97, 97, 160, 256), 64),
    ((0, 300, 301, 333), 128),
]


@pytest.mark.parametrize("tile_bytes", [1 << 30, 64 << 10, 9000])
@pytest.mark.parametrize("cu_host,budget", SELECTION_CASES)
def test_selected_blocks_are_bitwise_identical_with_host_bounds(monkeypatch, cu_host, budget, tile_bytes):
    """The packed attention path's block ids, old (device reads) vs new (host bounds), bit for bit."""
    indexer = _indexer(monkeypatch, tile_bytes, budget)
    total = cu_host[-1]
    hidden, positions, rope, cu = _inputs(cu_host, total)
    with torch.no_grad():
        old, old_cu = qsa_selected_blocks(indexer, hidden, positions, rope, cu)
        with monkeypatch.context() as m:
            guard = _NoDeviceReads(m)
            new, new_cu = qsa_selected_blocks(indexer, hidden, positions, rope, cu, cu_seqlens_host=cu_host)
    assert old.dtype == new.dtype and old.shape == new.shape
    assert torch.equal(old, new)
    assert torch.equal(old_cu, new_cu)
    # what is left per tile: the tie check of select_topk_blocks
    assert guard.bool_reads >= 1


def test_tile_plan_is_the_same_with_host_bounds(monkeypatch):
    """Same tiles, so the same einsum shapes (a tile's fp32 scores depend on its shape)."""
    seen = {}

    def record(tag):
        real = qsa_indexer.plan_query_tiles

        def plan(lo, hi, *args, **kwargs):
            tiles = real(lo, hi, *args, **kwargs)
            seen[tag] = tiles
            return tiles

        return plan

    cu_host = (0, 97, 97, 160, 256)
    indexer = _indexer(monkeypatch, 9000, 64)
    hidden, positions, rope, cu = _inputs(cu_host, cu_host[-1])
    with torch.no_grad():
        monkeypatch.setattr(qsa_indexer, "plan_query_tiles", record("old"))
        qsa_selected_blocks(indexer, hidden, positions, rope, cu)
        monkeypatch.setattr(qsa_indexer, "plan_query_tiles", record("new"))
        qsa_selected_blocks(indexer, hidden, positions, rope, cu, cu_seqlens_host=cu_host)
    assert seen["old"] == seen["new"] and len(seen["old"]) > 1


def test_host_bounds_of_another_length_are_rejected(monkeypatch):
    indexer = _indexer(monkeypatch, 1 << 30, 64)
    hidden, positions, rope, cu = _inputs((0, 37, 50, 72), 72)
    with torch.no_grad(), pytest.raises(ValueError, match="bounds"):
        qsa_selected_blocks(indexer, hidden, positions, rope, cu, cu_seqlens_host=(0, 50, 72))
