"""The QSA indexer under context parallelism selects exactly what the unsharded indexer does.

Under allgather-CP a rank holds queries for one contiguous window of the sequence, while the
keys -- and so the candidate blocks -- stay whole-sequence. ``qsa_selected_blocks(cp_rows=...)``
must therefore return precisely the unsharded selection restricted to those rows.

This is the part of the CP change that fails silently: a wrong row window or a wrong
composition with the indexer's TP row stride (``--qwen38-indexer-tp-shard``, on in production)
changes which keys a query attends to, with the right shape, no error, and no warning. So the
comparison is bitwise against the unsharded call, over every (cp_size, window) combination,
and three control tests assert the restriction is real rather than a no-op.

The two shardings compose in ``ops/cp.py:compose_row_shard``: CP contributes a contiguous
window ``[lo, hi)``, the TP shard a stride ``n``, and together they are the stride restricted
to the window -- which is why ``iter_selected_blocks`` needs ``row_stop``. The arithmetic of
that composition (rows partition the sequence exactly once) is checked here too, without any
process group, since it is pure index math.

CPU only: ``TELinear`` is replaced by a plain linear, as the other indexer tests do. No
process groups are built -- ``cp_rows`` is passed directly, which is what a CP rank computes
from ``local_rows()``.
"""

import math
from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops import qsa_indexer  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.attention import Qwen38NextAttention, qsa_selected_blocks  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.cp import compose_row_shard  # noqa: E402

register_cpu_ci(est_time=90, suite="stage-b-cpu", labels=["miles-plugin"])

RATIO = 4
HIDDEN = 64


class _DuplicatedLinear(torch.nn.Module):
    """``TELinear(parallel_mode="duplicated")`` without Transformer Engine."""

    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__()
        torch.manual_seed(7)
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) / math.sqrt(input_size))

    def forward(self, x):
        return x @ self.weight.t(), None


def _indexer(monkeypatch, budget=16, tile_bytes=1 << 30):
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


def _select(indexer, hidden, positions, rope, cu, cp_rows=None, cu_host=None):
    return qsa_selected_blocks(indexer, hidden, positions, rope, cu, cp_rows=cp_rows, cu_seqlens_host=cu_host)


# packed boundaries: one document, several, ragged, and a trailing pad segment as get_batch writes
CU_CASES = [
    (0, 64),
    (0, 128),
    (0, 32, 64),
    (0, 20, 52, 64),
    (0, 37, 64),
    (0, 50, 100, 128),
]
CP_SIZES = [2, 4]


@pytest.mark.parametrize("cu_host", CU_CASES, ids=[str(c) for c in CU_CASES])
@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_cp_window_matches_the_unsharded_selection(monkeypatch, cu_host, cp_size) -> None:
    total = cu_host[-1]
    indexer = _indexer(monkeypatch)
    hidden, positions, rope, cu = _inputs(cu_host, total)

    reference, ref_cu = _select(indexer, hidden, positions, rope, cu)
    assert reference.shape[0] == total

    local = total // cp_size
    for cp_rank in range(cp_size):
        lo, hi = cp_rank * local, (cp_rank + 1) * local
        blocks, cu_out = _select(indexer, hidden, positions, rope, cu, cp_rows=(lo, hi))
        assert blocks.shape == (hi - lo, reference.shape[1]), f"rank {cp_rank} shape"
        assert torch.equal(blocks, reference[lo:hi]), f"rank {cp_rank} selection"
        assert torch.equal(cu_out, ref_cu), f"rank {cp_rank} cu_seqlens"


@pytest.mark.parametrize("cu_host", [(0, 64), (0, 20, 52, 64)], ids=["one-doc", "ragged"])
@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_cp_window_matches_with_host_bounds(monkeypatch, cu_host, cp_size) -> None:
    """The host-bounds tile plan (``cu_seqlens_host``) must restrict rows the same way."""
    total = cu_host[-1]
    indexer = _indexer(monkeypatch)
    hidden, positions, rope, cu = _inputs(cu_host, total)

    reference, _ = _select(indexer, hidden, positions, rope, cu, cu_host=cu_host)
    local = total // cp_size
    for cp_rank in range(cp_size):
        lo, hi = cp_rank * local, (cp_rank + 1) * local
        blocks, _ = _select(indexer, hidden, positions, rope, cu, cp_rows=(lo, hi), cu_host=cu_host)
        assert torch.equal(blocks, reference[lo:hi]), f"rank {cp_rank} with host bounds"


@pytest.mark.parametrize("cu_host", [(0, 64), (0, 20, 52, 64)], ids=["one-doc", "ragged"])
def test_small_tiles_do_not_change_the_windowed_selection(monkeypatch, cu_host) -> None:
    """A tile budget that forces several tiles must not change which blocks a row picks."""
    total = cu_host[-1]
    big = _indexer(monkeypatch, tile_bytes=1 << 30)
    hidden, positions, rope, cu = _inputs(cu_host, total)
    reference, _ = _select(big, hidden, positions, rope, cu)

    small = _indexer(monkeypatch, tile_bytes=4096)
    local = total // 2
    for cp_rank in range(2):
        lo, hi = cp_rank * local, (cp_rank + 1) * local
        blocks, _ = _select(small, hidden, positions, rope, cu, cp_rows=(lo, hi))
        assert torch.equal(blocks, reference[lo:hi]), f"rank {cp_rank} under a small tile budget"


def test_each_rank_gets_a_different_window(monkeypatch) -> None:
    """Control: the window must actually select, not return rank 0's rows to everyone."""
    indexer = _indexer(monkeypatch)
    hidden, positions, rope, cu = _inputs((0, 128), 128)
    first, _ = _select(indexer, hidden, positions, rope, cu, cp_rows=(0, 64))
    second, _ = _select(indexer, hidden, positions, rope, cu, cp_rows=(64, 128))
    assert not torch.equal(first, second)


def test_block_ids_are_whole_sequence_not_window_local(monkeypatch) -> None:
    """A later window's ids must reach blocks before it: the keys are the whole sequence.

    If the implementation had rebased ids onto the window, the second half's ids could not
    exceed the number of blocks in half a sequence.
    """
    indexer = _indexer(monkeypatch)
    hidden, positions, rope, cu = _inputs((0, 128), 128)
    blocks, _ = _select(indexer, hidden, positions, rope, cu, cp_rows=(64, 128))
    blocks_in_half = 64 // RATIO
    assert int(blocks[blocks >= 0].max()) >= blocks_in_half


# ---------------------------------------------------------------- composition arithmetic


@pytest.mark.parametrize("total", [64, 128, 256])
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_composed_rows_partition_the_sequence_exactly_once(total, cp_size, tp_size) -> None:
    """Every (cp_rank, tp_rank) pair together covers each query row once and only once."""
    if total % cp_size:
        pytest.skip("allgather-CP slices an already padded stream")
    local = total // cp_size
    seen: list[int] = []
    for cp_rank in range(cp_size):
        lo, hi = cp_rank * local, (cp_rank + 1) * local
        for tp_rank in range(tp_size):
            shard, rows = compose_row_shard((lo, hi), (tp_rank, tp_size) if tp_size > 1 else None, total)
            start, step = (lo, 1) if shard is None else shard
            owned = list(range(start, hi, step))
            assert rows == len(owned), f"cp{cp_rank} tp{tp_rank}: claims {rows} rows, owns {len(owned)}"
            seen.extend(owned)
    assert sorted(seen) == list(range(total))


@pytest.mark.parametrize("tp_size", [2, 4])
def test_the_window_stop_is_load_bearing(tp_size) -> None:
    """Without the window's upper bound a non-final rank's stride runs into the next window."""
    total, cp_size = 128, 4
    local = total // cp_size
    for cp_rank in range(cp_size - 1):  # the last window ends at the sequence; nothing to run into
        lo, hi = cp_rank * local, (cp_rank + 1) * local
        shard, rows = compose_row_shard((lo, hi), (0, tp_size), total)
        unclamped = len(range(shard[0], total, shard[1]))
        assert unclamped > rows, f"cp{cp_rank}: stride to the sequence end is not longer than to the window end"
