"""QSA recompute reuse: the replay of a checkpointed QSA layer takes the forward's selection / index / output.

Default off (mode "off": every code path is the one without this module). Turned on per process with
:func:`configure` (the Qwen3.8 training repo calls it from its init hook, schedule field qsa_recompute_reuse).

Under full activation recompute (Megatron ``CheckpointFunction``, ``--recompute-granularity full``) every layer's
forward runs twice: under ``no_grad`` in the forward pass, then again in backward (the replay) to rebuild the
activations. In a QSA layer (``ops/attention.py``) the replay repeats work whose result it cannot change:

* the indexer's block selection (``no_grad``, frozen indexer: the SP gather of its input, key compression, scores,
  top-k, and with ``--qwen38-indexer-tp-shard`` the TP all-gather of the block ids),
* the block index built from that selection (``build_qsa_block_index``, one host sync),
* the sparse attention forward kernel (output and log-sum-exp from the same q, k, v and index).

With reuse on, the forward pass keeps these per layer in an :class:`Entry` and the replay takes them. The values the
replay uses are the forward's own: the replay's result is bitwise the result of a replay that recomputes them,
whenever that recomputation is deterministic. The indexer's key compression is not (``index_add_`` with float
atomics, M-041): there the recomputing replay can select other blocks than the forward did, and with reuse the
backward always uses the forward's selection.

Modes (what an entry keeps; bytes per token of the microbatch per QSA layer at TP4 / CP1, K = 512 block ids, 6 query
heads x 256 per rank):

* ``blocks``: the ``[T, K]`` int32 block ids (2 KiB). The replay skips the indexer; it builds the index and runs the
  forward kernel again.
* ``index``: the ``QSABlockIndex`` (its size depends on the selection; the block ids are not kept). The replay also
  skips the index build.
* ``output``: the index, the attention output ``[T, Hq, D]`` (bf16, 3 KiB) and its log-sum-exp ``[T, Hq]`` fp32. The
  replay also skips the forward kernel; the backward kernels read the kept output / lse, exactly as they read the
  ones a recomputed forward returns (``qsa_block_attention_replay``).
* ``blocks_output``: the block ids, the output and the lse (5 KiB): the replay skips the indexer and the forward
  kernel and builds the index again (for when the index is larger than the block ids).

``max_tokens``: only microbatches whose QSA sequence is at most this many tokens keep entries (the kept bytes grow
with T); ``None`` = every recomputed microbatch. The QSA sequence is the indexer's rows: the whole SP-gathered
microbatch, padding included (not the longest sample in it). Both limits include the boundary (``<=``). Which
microbatches are recomputed at all is decided outside this module (the training repo's no-recompute hook): a
microbatch that runs without Megatron's checkpoint keeps nothing.

``output_max_tokens``: the output modes keep the output / lse only for microbatches of at most this many tokens; a
longer one keeps what the mode keeps besides (output -> the index, as mode index; blocks_output -> the block ids, as
mode blocks). The output is the largest part (3 KiB per token per layer), so a long row can keep the cheap part
under a memory budget. ``None`` = no such limit.

``audit`` (tests): the replay reruns the indexer and compares its selection with the kept one; then, from the kept
selection, rebuilds the index and reruns the forward kernel and compares those with the kept index / out / lse
(bitwise, counts in :func:`stats`; nothing raises). The replay then uses the kept values. So zero index / output /
lse mismatches show that the recomputation with the forward's selection is reproducible; they say nothing about the
other selection a reuse-off replay may draw (that is the selection count). Audit keeps the block ids in every mode
(about 2 KiB per token per layer more than without it) and adds the recomputation's temporaries.

Pairing. An entry belongs to one layer and one microbatch, keyed by the ``cu_seqlens_q`` tensor object of the
microbatch's ``packed_seq_params``: Megatron's checkpoint closes over ``packed_seq_params`` and hands the same object
to the replay. The forward pass pushes an entry when Megatron's checkpoint runs the layer (``is_checkpointing()``
under ``no_grad``, training); the replay (``is_checkpointing()`` with grad) pops the entry of its ``cu_seqlens_q``.
A microbatch over ``max_tokens`` pushes a marker entry that keeps nothing, so a replay without any entry is an error
(raised, not recomputed: every rank of the TP group would take the same path, a silent fallback would only hide a
pairing bug). An entry of the layer still there when the next forward pushes is stale (a checkpointed forward that
got no backward) and is dropped (counted). Before the replay uses an entry, the kept tensors and the pairing / index
inputs (``cu`` and the ``cu_seqlens`` the selection came with) are checked: autograd version counter, storage pointer
and storage size as at the end of the forward. This catches ordinary in-place tensor ops, a replaced storage and a
released storage (offloading). It is not a content check: writes that bypass the version counter (``.data`` ops, a
raw kernel writing the memory) are not detected; only audit compares values.

Refused (``NotImplementedError`` at the first training forward with reuse on, recomputed or not): pipeline
parallelism (several microbatches in flight per layer), context parallelism (the CP paths of the core attention are
not covered), fp8 / fp4 (Megatron routes their recompute through TE's checkpoint, which does not set
``is_checkpointing``: reuse would switch itself off silently), the sbhd (unpacked) layout. Mode off and eval forwards
are not checked.
"""

from __future__ import annotations

import collections
import dataclasses
import weakref

import torch
from torch import Tensor

MODES = ("off", "blocks", "index", "output", "blocks_output")
KEEPS_BLOCKS = ("blocks", "blocks_output")  # the replay builds the index from the kept block ids
KEEPS_INDEX = ("index", "output")
KEEPS_OUTPUT = ("output", "blocks_output")


def keeps_blocks() -> bool:
    """The forward keeps the block ids: the mode rebuilds the index from them, or the audit compares them."""
    return _SETTINGS.mode in KEEPS_BLOCKS or _SETTINGS.audit


def keeps_index() -> bool:
    return _SETTINGS.mode in KEEPS_INDEX


def keeps_output() -> bool:
    return _SETTINGS.mode in KEEPS_OUTPUT


@dataclasses.dataclass
class _Settings:
    mode: str = "off"
    max_tokens: int | None = None
    audit: bool = False
    output_max_tokens: int | None = None


_SETTINGS = _Settings()
_STATS: collections.Counter = collections.Counter()
_HELD = {"bytes": 0, "peak_bytes": 0, "entries": 0}
_STASHES: "weakref.WeakSet[LayerStash]" = weakref.WeakSet()


def _limit(name: str, value) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"qsa recompute reuse {name} must be None or a positive int, got {value!r}")


def configure(
    mode: str = "off", max_tokens: int | None = None, audit: bool = False, output_max_tokens: int | None = None
) -> dict:
    """Set this process's reuse mode (see the module doc). Returns :func:`settings`."""
    if mode not in MODES:
        raise ValueError(f"qsa recompute reuse mode must be one of {MODES}, got {mode!r}")
    _limit("max_tokens", max_tokens)
    _limit("output_max_tokens", output_max_tokens)
    if not isinstance(audit, bool):
        raise ValueError(f"qsa recompute reuse audit must be true or false, got {audit!r}")
    new = _Settings(mode, max_tokens, audit, output_max_tokens)
    if new != _SETTINGS:
        for stash in list(_STASHES):  # entries kept under the old settings are not replayed under the new ones
            stash.clear()
    for f in dataclasses.fields(_Settings):
        setattr(_SETTINGS, f.name, getattr(new, f.name))
    return settings()


def settings() -> dict:
    return dataclasses.asdict(_SETTINGS)


def mode() -> str:
    return _SETTINGS.mode


def audit() -> bool:
    return _SETTINGS.audit


def stats() -> dict:
    """Counters since the last :func:`reset_stats` plus the kept bytes now / at peak (all layers of this process).

    pushes / markers: entries kept / marker entries (over max_tokens) in the forward pass; hits: replays that reused
    an entry; marker_replays: replays of a marker (recomputed as without reuse); stale_dropped: entries dropped
    unused; skipped_*: the work the hits did not repeat; audit_*: the audit's comparisons and bitwise mismatches."""
    return {
        **dict(_STATS),
        "held_bytes": _HELD["bytes"],
        "held_entries": _HELD["entries"],
        "peak_held_bytes": _HELD["peak_bytes"],
    }


def reset_stats() -> None:
    """Clear the counters; the peak restarts from the bytes kept now."""
    _STATS.clear()
    _HELD["peak_bytes"] = _HELD["bytes"]


def _count(name: str, n: int = 1) -> None:
    _STATS[name] += n


def tensor_bytes(x) -> int:
    """Bytes of the distinct storages behind the tensors in ``x`` (a tensor, a dataclass of tensors / ints, a tuple /
    list, or None): what keeping ``x`` keeps alive (the index's lists are views of a few shared buffers)."""
    seen, total = set(), 0
    for t in _tensors(x):
        storage = t.untyped_storage()
        if storage.data_ptr() not in seen:
            seen.add(storage.data_ptr())
            total += storage.nbytes()
    return total


def _tensors(x):
    if isinstance(x, Tensor):
        yield x
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            yield from _tensors(getattr(x, f.name))
    elif isinstance(x, (tuple, list)):
        for t in x:
            yield from _tensors(t)


def bitwise_equal(a, b) -> bool:
    """Same structure, shapes, dtypes and bits (NaN payloads included) for tensors; ``==`` for the rest."""
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        if not (isinstance(a, Tensor) and isinstance(b, Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype.is_floating_point or a.dtype.is_complex:
            width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
            a, b = a.contiguous().view(width), b.contiguous().view(width)
        return torch.equal(a, b)
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            return False
        return all(bitwise_equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(bitwise_equal(x, y) for x, y in zip(a, b))
    return a == b


@dataclasses.dataclass(eq=False)
class Entry:
    """What one QSA layer keeps from one microbatch's forward pass for its replay."""

    cu: Tensor  # the microbatch's packed_seq_params.cu_seqlens_q (identity key, kept alive by this reference)
    seq: int  # the indexer's rows (whole microbatch)
    hs_shape: tuple  # the layer input's shape, checked at the replay
    keep: bool  # False: a marker (over max_tokens), the replay recomputes
    keep_output: bool = False  # an output mode and seq <= output_max_tokens: the forward keeps out / lse
    blocks: Tensor | None = None  # [rows, K] int32 (mode blocks; any mode with audit)
    blocks_cu: Tensor | None = None  # the cu_seqlens qsa_selected_blocks returned next to them
    index: object | None = None  # QSABlockIndex (modes index, output)
    out: Tensor | None = None  # [T, Hq, D] (mode output)
    lse: Tensor | None = None  # [T, Hq] fp32 (mode output)
    versions: tuple = ()
    nbytes: int = 0
    held: bool = False  # counted in the kept bytes (sealed, not yet popped or dropped)

    def seal(self) -> None:
        """Record version counter, storage pointer and size of the kept tensors and of the pairing / index inputs
        (cu, blocks_cu), and the kept bytes (distinct storages of the kept tensors only), at the end of the forward."""
        kept = [self.blocks, self.index, self.out, self.lse]
        guarded = [*kept, self.cu, self.blocks_cu]
        seen, versions = set(), []
        for t in _tensors(guarded):
            if id(t) not in seen:
                seen.add(id(t))
                versions.append((t, t._version, t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()))
        self.versions = tuple(versions)
        self.nbytes = tensor_bytes(kept)

    def check_unmodified(self, layer_number) -> None:
        """Version counters and storages as sealed (module doc: not a content check)."""
        for t, version, ptr, size in self.versions:
            s = t.untyped_storage()
            if t._version != version or s.data_ptr() != ptr or s.nbytes() != size:
                raise RuntimeError(
                    f"QSA recompute reuse: a tensor layer {layer_number} kept or paired from its forward pass was "
                    f"changed after the forward (version {version} -> {t._version}, storage {size} -> {s.nbytes()} "
                    f"bytes, moved {s.data_ptr() != ptr}); its replay cannot reuse it"
                )


class LayerStash:
    """The entries of one QSA layer (at most one: pipeline parallelism is refused)."""

    def __init__(self, layer_number):
        self.layer_number = layer_number
        self.entries: list[Entry] = []
        _STASHES.add(self)

    def push(self, entry: Entry) -> None:
        if self.entries:
            _count("stale_dropped", len(self.entries))
            for old in self.entries:
                _release(old)
            self.entries = []
        self.entries.append(entry)
        _count("pushes" if entry.keep else "markers")

    def account(self, entry: Entry) -> None:
        entry.seal()
        entry.held = True
        _HELD["bytes"] += entry.nbytes
        _HELD["entries"] += 1
        _HELD["peak_bytes"] = max(_HELD["peak_bytes"], _HELD["bytes"])

    def pop(self, cu: Tensor) -> Entry | None:
        for i, entry in enumerate(self.entries):
            if entry.cu is cu:
                for old in self.entries[:i]:
                    _count("stale_dropped")
                    _release(old)
                self.entries = self.entries[i + 1 :]
                _release(entry)
                return entry
        return None

    def clear(self) -> None:
        for old in self.entries:
            _count("stale_dropped")
            _release(old)
        self.entries = []


def _release(entry: Entry) -> None:
    """Take the entry out of the kept bytes (popped by its replay, or dropped); its version record stays."""
    if entry.held:
        _HELD["bytes"] -= entry.nbytes
        _HELD["entries"] -= 1
        entry.held = False


def check_config(config) -> None:
    """Refuse layouts the pairing does not cover (module doc). ``config`` is the layer's TransformerConfig."""
    pp = getattr(config, "pipeline_model_parallel_size", 1) or 1  # config-access-exempt: hand-built test configs
    vpp = getattr(config, "virtual_pipeline_model_parallel_size", None)  # config-access-exempt: set only when enabled
    cp = getattr(config, "context_parallel_size", 1) or 1  # config-access-exempt: hand-built test configs
    fp8 = getattr(config, "fp8", None) or getattr(config, "fp4", None)  # config-access-exempt: optional precision
    problems = []
    if pp > 1 or vpp:
        problems.append(
            f"pipeline parallelism (PP {pp}, virtual {vpp}) keeps several microbatches per layer in flight"
        )
    if cp > 1:
        problems.append(f"context parallelism (CP {cp}) is not covered")
    if fp8:
        problems.append("fp8 / fp4 recompute runs through TE's checkpoint, which does not set is_checkpointing")
    if problems:
        raise NotImplementedError("QSA recompute reuse (mode %r): %s" % (_SETTINGS.mode, "; ".join(problems)))


def wants_entry(seq: int) -> bool:
    return _SETTINGS.max_tokens is None or seq <= _SETTINGS.max_tokens


def wants_output(seq: int) -> bool:
    """An output mode keeps the output of a microbatch of ``seq`` tokens (output_max_tokens)."""
    return keeps_output() and (_SETTINGS.output_max_tokens is None or seq <= _SETTINGS.output_max_tokens)
