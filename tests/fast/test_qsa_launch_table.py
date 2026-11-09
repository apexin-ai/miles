"""The QSA launch table: which row a device takes, the rows training ran with, and the sweep override."""

import pytest
import torch

from miles.kernels.attention.qsa import launch as qsa_launch_mod
from miles.kernels.attention.qsa.launch import QSALaunch, TUNED, qsa_launch, qsa_launch_override, table_launch

CUDA = torch.device("cuda", 0)


@pytest.fixture
def capability(monkeypatch):
    def set_cap(cap):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cap)

    return set_cap


def test_rows_training_ran_with_are_unchanged():
    """The sm_90 / sm_100 rows of block_attention._TUNED and block_index._KEY_TILE_BLOCKS before the table moved."""
    assert TUNED[9] == QSALaunch(
        fwd={"NU": 8, "num_warps": 4, "num_stages": 3},
        dq={"NU": 8, "num_warps": 4, "num_stages": 1},
        dkdv={"num_warps": 4, "num_stages": 3},
        key_tile=4,
    )
    assert TUNED[10] == QSALaunch(
        fwd={"NU": 16, "num_warps": 8, "num_stages": 3},
        dq={"NU": 16, "num_warps": 8, "num_stages": 3},
        dkdv={"num_warps": 8, "num_stages": 1},
        key_tile=16,
    )
    for row in (TUNED[9], TUNED[10]):
        assert (row.query_rows, row.key_rows, row.dsplit, row.key_order) == (64, 64, 1, "position")


@pytest.mark.parametrize(
    "cap,want", [((9, 0), 9), ((10, 0), 10), ((10, 3), 10), ((12, 0), 9), ((8, 0), 9)], ids=str
)
def test_a_device_takes_its_major_row_when_no_exact_row(capability, cap, want):
    capability(cap)
    if cap in TUNED:
        pytest.skip(f"{cap} has an exact row")
    assert table_launch(CUDA) is TUNED[want]


def test_an_exact_row_wins_over_the_major_row(capability, monkeypatch):
    b300 = QSALaunch(fwd={"NU": 4}, dq={"NU": 4}, dkdv={"DSPLIT": 2}, key_tile=8)
    monkeypatch.setitem(TUNED, (10, 3), b300)
    capability((10, 3))
    assert table_launch(CUDA) is b300
    capability((10, 0))
    assert table_launch(CUDA) is TUNED[10]


def test_cpu_takes_the_sm90_row():
    assert table_launch(torch.device("cpu")) is TUNED[9]


def test_override_merges_dicts_replaces_scalars_and_restores(capability):
    capability((10, 0))
    base = qsa_launch(CUDA)
    with qsa_launch_override(dkdv={"DSPLIT": 2, "ROWS": 32}, key_tile=8, key_order="busiest"):
        got = qsa_launch(CUDA)
        assert got.dkdv == {"num_warps": 8, "num_stages": 1, "DSPLIT": 2, "ROWS": 32}
        assert got.kernel_dkdv() == {"num_warps": 8, "num_stages": 1, "DSPLIT": 2}
        assert (got.key_tile, got.key_order, got.key_rows, got.dsplit) == (8, "busiest", 32, 2)
        assert got.fwd == base.fwd and got.query_rows == base.query_rows
        with qsa_launch_override(fwd={"NU": 4}):
            inner = qsa_launch(CUDA)
            assert inner.fwd == {"NU": 4, "num_warps": 8, "num_stages": 3}
            assert inner.key_tile == 8  # the outer override stays
        assert qsa_launch(CUDA).fwd == base.fwd
    assert qsa_launch(CUDA) == base
    assert qsa_launch_mod._OVERRIDE.fields == {}


def test_override_restores_after_an_exception():
    with pytest.raises(RuntimeError):
        with qsa_launch_override(key_tile=2):
            raise RuntimeError
    assert qsa_launch_mod._OVERRIDE.fields == {}


@pytest.mark.parametrize(
    "fields",
    [
        {"fwd": {"BLOCK": 32}},
        {"dkdv": {"NU": 4}},
        {"dkdv": {"DSPLIT": 3}},
        {"dkdv": {"ROWS": 8}},
        {"key_tile": 64},
        {"key_tile": 6},
        {"query_rows": 48},
        {"key_order": "random"},
        {"fwd": {"NU": 0}},
        {"block_m": 64},
    ],
    ids=str,
)
def test_override_rejects_bad_fields(fields):
    with pytest.raises(ValueError):
        with qsa_launch_override(**fields):
            pass
    assert qsa_launch_mod._OVERRIDE.fields == {}


def test_kernel_dkdv_always_passes_dsplit():
    assert TUNED[10].kernel_dkdv() == {"num_warps": 8, "num_stages": 1, "DSPLIT": 1}
