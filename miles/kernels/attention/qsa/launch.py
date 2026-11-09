"""Launch parameters of the QSA block-attention kernels, per GPU arch, and a scoped override for sweeps.

One row per arch holds everything that changes how the kernels are cut, not what they compute:

* ``fwd`` / ``dq``: ``NU`` blocks per step of a forward / dQ program, ``num_warps``, ``num_stages``;
* ``dkdv``: ``num_warps``, ``num_stages``, ``DSPLIT`` (programs per key tile, each owning ``D / DSPLIT`` columns of
  dK and dV; every one recomputes the scores over the whole head_dim) and ``ROWS`` (rows of a dK / dV step:
  ``ROWS // G`` queries x the ``G`` heads of one KV head);
* ``key_tile``: blocks per dK / dV program (the index's key tiles);
* ``query_rows``: rows of a forward / dQ tile (``min(32, query_rows // G)`` queries x ``G`` heads);
* ``key_order``: launch order of the dK / dV programs, ``"position"`` or ``"busiest"`` (most queries first).

``key_order`` and ``num_stages`` only reorder or pipeline the same arithmetic. ``DSPLIT`` splits the output
columns of dK / dV; the sum behind each element is the same. ``NU``, ``query_rows``, ``key_tile`` and ``ROWS``
regroup the fp32 sums (forward: where the online softmax rescales; dQ / dK / dV: which terms one tensor-core
product adds before the fp32 accumulator), so they change results by rounding only. Every row is
deterministic: no autotuning at run time (ranks could pick differently), no atomics.

``qsa_launch_override`` is for benchmarks and tests. Training reads the table.
"""

import contextlib
from dataclasses import dataclass, field, replace

import torch

_STEP_KEYS = {"NU", "num_warps", "num_stages"}
_DKDV_KEYS = {"num_warps", "num_stages", "DSPLIT", "ROWS"}
_KEY_ORDERS = ("position", "busiest")


@dataclass(frozen=True)
class QSALaunch:
    fwd: dict
    dq: dict
    dkdv: dict
    key_tile: int
    query_rows: int = 64
    key_order: str = "position"

    def __post_init__(self):
        for name, allowed in (("fwd", _STEP_KEYS), ("dq", _STEP_KEYS), ("dkdv", _DKDV_KEYS)):
            extra = set(getattr(self, name)) - allowed
            if extra:
                raise ValueError(f"QSA launch {name}: unknown keys {sorted(extra)}, allowed {sorted(allowed)}")
        for name in ("fwd", "dq"):
            _check_pow2(f"{name}.NU", getattr(self, name).get("NU", 1), 1, 64)
        _check_pow2("dkdv.DSPLIT", self.dkdv.get("DSPLIT", 1), 1, 16)
        _check_pow2("dkdv.ROWS", self.dkdv.get("ROWS", 64), 16, 256)
        _check_pow2("key_tile", self.key_tile, 1, 32)  # one int32 bit per block of a key tile
        _check_pow2("query_rows", self.query_rows, 16, 256)
        if self.key_order not in _KEY_ORDERS:
            raise ValueError(f"QSA launch key_order {self.key_order!r}, allowed {_KEY_ORDERS}")

    @property
    def key_rows(self) -> int:
        return self.dkdv.get("ROWS", 64)

    @property
    def dsplit(self) -> int:
        return self.dkdv.get("DSPLIT", 1)

    def kernel_dkdv(self) -> dict:
        """The dK / dV kernel's launch keywords (``ROWS`` sets the step shape, not a kernel argument)."""
        out = {k: v for k, v in self.dkdv.items() if k != "ROWS"}
        out.setdefault("DSPLIT", 1)
        return out


def _check_pow2(name: str, value, lo: int, hi: int) -> None:
    if not isinstance(value, int) or value < lo or value > hi or value & (value - 1):
        raise ValueError(f"QSA launch {name} = {value!r}: a power of two in [{lo}, {hi}]")


# Measured at head_dim 256, 12 query heads per KV head, 8K and 64K tokens of indexer selections: sm_90 on H20 and
# H100, sm_100 on B200. Keys are (major, minor) or major; a device takes its exact (major, minor) row, else its
# major's row, else the sm_90 row. sm_103 (B300) has no row of its own yet and takes the sm_100 row.
TUNED: dict = {
    9: QSALaunch(
        fwd={"NU": 8, "num_warps": 4, "num_stages": 3},
        dq={"NU": 8, "num_warps": 4, "num_stages": 1},
        dkdv={"num_warps": 4, "num_stages": 3},
        key_tile=4,
    ),
    10: QSALaunch(
        fwd={"NU": 16, "num_warps": 8, "num_stages": 3},
        dq={"NU": 16, "num_warps": 8, "num_stages": 3},
        dkdv={"num_warps": 8, "num_stages": 1},
        key_tile=16,
    ),
}


@dataclass
class _Override:
    fields: dict = field(default_factory=dict)


_OVERRIDE = _Override()


def table_launch(device: torch.device) -> QSALaunch:
    """The table row for ``device`` (a non-CUDA device, e.g. the Triton interpreter, takes the sm_90 row)."""
    if device.type != "cuda":
        return TUNED[9]
    cap = tuple(torch.cuda.get_device_capability(device))
    return TUNED.get(cap, TUNED.get(cap[0], TUNED[9]))


def qsa_launch(device: torch.device) -> QSALaunch:
    """The launch the kernels use on ``device``: its table row with the active override applied."""
    base = table_launch(device)
    if not _OVERRIDE.fields:
        return base
    merged = {}
    for name, value in _OVERRIDE.fields.items():
        if name in ("fwd", "dq", "dkdv"):
            merged[name] = {**getattr(base, name), **value}
        else:
            merged[name] = value
    return replace(base, **merged)


@contextlib.contextmanager
def qsa_launch_override(**fields):
    """Within the block, kernels launch with ``fields`` on top of the table row: ``fwd`` / ``dq`` / ``dkdv`` dicts
    are merged key by key, ``key_tile`` / ``query_rows`` / ``key_order`` replace. An index built inside the
    block keeps its tiles, and an attention whose forward ran inside it runs its backward with the same launch,
    also after the block."""
    unknown = set(fields) - {"fwd", "dq", "dkdv", "key_tile", "query_rows", "key_order"}
    if unknown:
        raise ValueError(f"qsa_launch_override: unknown fields {sorted(unknown)}")
    previous = _OVERRIDE.fields
    _OVERRIDE.fields = {**previous, **{k: (dict(v) if isinstance(v, dict) else v) for k, v in fields.items()}}
    try:
        # validate now, on a CPU row, rather than at the first launch
        qsa_launch(torch.device("cpu"))
        yield
    finally:
        _OVERRIDE.fields = previous
