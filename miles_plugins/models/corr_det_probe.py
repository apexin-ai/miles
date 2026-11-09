"""Test-only probe for the M039 hdirect whole-model determinism run. Off unless MILES_CORR_DET_PROBE=1.

Never use it in a training run: it adds per-call and per-parameter GPU work and writes evidence files.

What it records (all GPU-side, deterministic integer checksums; one host transfer per step):

* every call of FlashQLA's backward CP state correction ``cp_context.correct_terminal_states`` (whatever is
  installed after ``flashqla_corr_sync.maybe_apply``: the hdirect replacement in the B arm, the original kernel
  in the optional A3 arm): a checksum of each CONSUMED input region and of the output.
  Consumed regions (the reverse scan reads idx = end-1 .. start+1 of every raw sequence): dht_buffer blocks of
  those segments; mt_buffer blocks of those segments where fallback is set; fallback_mask on those segments;
  seq_map_r2c whole. raw_dht must be None and state_v_first False (else the row is flagged).
* per parameter, the pre-reduction main_grad of every DDP bucket at the moment the F8 guard sees the bucket's
  gradient sync issued (same moment as the guard's local bucket norm).

Checksum of a tensor region: x = its bits as int64 (masked-out elements = 0), i = element index;
(sum x, sum x * (i * 0x9E3779B97F4A7C15 + 1)) mod 2**64. Integer sums are order independent, so the checksum is
bitwise reproducible for bitwise equal data.

Files (Q38_RUN_DIR/gt-evidence): corrdet-<rank>.jsonl and gradhash-<rank>.jsonl, one row per step, written at the
F8 guard's end_step. A call key is (rank, ordinal of the call within the step): the data, the microbatch order and
the layer order are the same every step of the frozen LR0 replay, so an ordinal names one (layer, microbatch).
"""
from __future__ import annotations

import functools
import json
import logging
import os
import threading
from pathlib import Path

ENV = "MILES_CORR_DET_PROBE"
MARK = "MILES_CORR_DET_PROBE=1: correction digests and per-parameter main_grad hashes enabled (test only)"
MAX_CALLS = 8192
CHUNK = 1 << 24
_GOLD = -7046029254386353131          # 0x9E3779B97F4A7C15 as a signed int64
_lock = threading.Lock()
logger = logging.getLogger(__name__)
_probe = None


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() == "1"


def _bits64(t):
    import torch
    t = t.detach().contiguous()
    if t.dtype == torch.bool:
        v = t.view(torch.uint8)
    else:
        v = t.view({8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}[t.element_size()])
    return v.reshape(-1).to(torch.int64)


def checksum(t, mask=None):
    """int64[2] on t's device. mask: bool tensor broadcastable to t's leading dims (block mask [S, H]) or None."""
    import torch
    x = _bits64(t)
    if mask is not None:
        m = mask.to(t.device)
        while m.dim() < t.dim():
            m = m.unsqueeze(-1)
        x = x * m.expand(t.shape).contiguous().reshape(-1).to(torch.int64)
    out = torch.zeros(2, dtype=torch.int64, device=x.device)
    gold = torch.tensor(_GOLD, dtype=torch.int64, device=x.device)
    for s in range(0, x.numel(), CHUNK):
        part = x[s:s + CHUNK]
        w = torch.arange(s, s + part.numel(), dtype=torch.int64, device=x.device) * gold + 1
        out[0] += part.sum()
        out[1] += (part * w).sum()
    return out


def consumed_masks(fallback_mask, seq_map_r2c):
    """GPU-side, no host sync: (rows [S,H], mt [S,H]) bool masks of the regions the correction can read."""
    import torch
    fb = fallback_mask.bool()
    S, H = fb.shape
    read = torch.ones(S, dtype=torch.bool, device=fb.device)
    read[seq_map_r2c[:-1].long()] = False
    rows = read[:, None].expand(S, H)
    return rows, rows & fb


class _Probe:
    def __init__(self, out_dir: Path, rank: int):
        self.out, self.rank = out_dir, rank
        self.buf = None
        self.n = 0
        self.overflow = 0
        self.fn_names = set()
        self.grad = {}            # tag -> int64 [n_params, 2]
        self.grad_calls = {}      # tag -> number of issuing start_grad_sync calls this step
        self.groups_wrapped = 0

    # ---- correction calls ------------------------------------------------------------------------------
    def record_inputs(self, kw):
        """Checksums of the consumed input regions, taken BEFORE the correction runs. Returns the row or None."""
        import torch
        if self.buf is None:
            self.buf = torch.zeros(MAX_CALLS, 11, dtype=torch.int64, device=kw["dht_buffer"].device)
        if self.n >= MAX_CALLS:
            self.overflow += 1
            return None
        row = self.buf[self.n]
        self.n += 1
        row[10] = 0 if (kw.get("raw_dht") is None and not kw.get("state_v_first")) else 1
        rows, mt = consumed_masks(kw["fallback_mask"], kw["seq_map_r2c"])
        row[0:2] = checksum(kw["dht_buffer"], rows)
        row[2:4] = checksum(kw["mt_buffer"], mt)
        row[4:6] = checksum(kw["fallback_mask"], rows)
        row[6:8] = checksum(kw["seq_map_r2c"])
        return row

    @staticmethod
    def record_output(row, out):
        if row is not None:
            row[8:10] = checksum(out)

    def wrap_correction(self, cp_context) -> None:
        fn = cp_context.correct_terminal_states
        if getattr(fn, "_corr_det_probe", False):
            return

        @functools.wraps(fn)
        def probed(*args, **kw):
            if args:
                raise RuntimeError(f"{ENV}: correct_terminal_states called with positional arguments")
            row = self.record_inputs(kw)
            out = fn(**kw)
            self.record_output(row, out)
            return out

        probed._corr_det_probe = True
        probed._corr_det_inner = fn
        cp_context.correct_terminal_states = probed
        self.fn_names.add(f"{getattr(fn, '__module__', None)}.{getattr(fn, '__qualname__', None)}")

    # ---- gradients -------------------------------------------------------------------------------------
    def hash_bucket(self, tag, bucket) -> None:
        import torch
        rows = [checksum(p.main_grad) for p in bucket.params_list]
        self.grad[tag] = torch.stack(rows) if rows else torch.zeros(0, 2, dtype=torch.int64)
        self.grad_calls[tag] = self.grad_calls.get(tag, 0) + 1

    def wrap_guard(self, guard) -> None:
        groups = {}
        for _ptr, (_tag, group) in guard.by_ptr.items():
            groups.setdefault(id(group), group)
        for group in groups.values():
            tags = [guard.by_ptr[b.grad_data.data_ptr()][0] for b in group.buckets]
            start = group.start_grad_sync

            def start_grad_sync(*a, _group=group, _tags=tags, _start=start, **k):
                issuing = not (_group.is_first_batch and _group.grad_reduce_handle is not None)
                if issuing:
                    for tag, bucket in zip(_tags, _group.buckets, strict=True):
                        self.hash_bucket(tag, bucket)
                return _start(*a, **k)

            group.start_grad_sync = start_grad_sync
            self.groups_wrapped += 1

    # ---- per step ---------------------------------------------------------------------------------------
    def flush(self, step: int, grad_norm) -> None:
        digests = self.buf[: self.n].cpu().tolist() if self.buf is not None and self.n else []
        tags = sorted(self.grad)
        stacked = [self.grad[t].cpu().tolist() for t in tags]
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / f"corrdet-{self.rank:05d}.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "rank": self.rank, "n_calls": self.n, "overflow": self.overflow,
                                "corr_fn": sorted(self.fn_names), "digests": digests}) + "\n")
        with (self.out / f"gradhash-{self.rank:05d}.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "rank": self.rank, "grad_norm": float(grad_norm),
                                "groups_wrapped": self.groups_wrapped, "calls": {t: self.grad_calls[t] for t in tags},
                                "hashes": dict(zip(tags, stacked, strict=True))}) + "\n")
        self.n, self.overflow = 0, 0
        self.grad, self.grad_calls = {}, {}


def _install_probe(cp_context, f8_guard_module, rank: int, out_dir: Path) -> _Probe:
    """Create the probe once: wrap the guard's bucket groups and its end_step (the step flush)."""
    global _probe
    guard = f8_guard_module._GUARD
    if guard is None:
        raise RuntimeError(f"{ENV} needs the F8 guard (Q38_F8_GUARD_FORCE=1 in the GT driver)")
    probe = _Probe(out_dir, rank)
    probe.wrap_guard(guard)
    original_end = f8_guard_module._Guard.end_step

    def end_step(gself, grad_norm, _original=original_end):
        reasons = _original(gself, grad_norm)
        probe.flush(gself.step, grad_norm)
        return reasons

    f8_guard_module._Guard.end_step = end_step
    logger.warning(MARK)   # stderr, like the corr-sync marker: one line per rank in the worker logs
    _probe = probe
    return probe


def ensure(cp_context, f8_guard_module=None, rank=None, out_dir=None) -> _Probe:
    """Idempotent: install the probe (first call) and wrap whatever correction is installed now."""
    with _lock:
        probe = _probe
        if probe is None:
            if f8_guard_module is None:
                from miles.backends.megatron_utils import f8_guard as f8_guard_module
            if rank is None:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
            if out_dir is None:
                out_dir = Path(os.environ["Q38_RUN_DIR"]) / "gt-evidence"
            probe = _install_probe(cp_context, f8_guard_module, rank, out_dir)
        probe.wrap_correction(cp_context)
        return probe


def install_for_backend(backend: str, corr_sync) -> None:
    """Called by flashqla_corr_sync.apply_for_backend when MILES_CORR_DET_PROBE=1. For backend "fla" it wraps
    fla's FlashQLABackend.chunk_gated_delta_rule: before each call, first corr_sync.maybe_apply (so the hdirect
    replacement, if requested, is installed and verified against the ORIGINAL function), then ensure() wraps the
    installed correction. "flashqla" does the same at once."""
    import importlib
    cp_context = importlib.import_module(corr_sync.CP_CONTEXT_MODULE) if backend == "flashqla" else None
    if backend == "flashqla":
        corr_sync.maybe_apply(strict_arch=True)
        ensure(cp_context)
        return
    if backend != "fla":
        return
    module = importlib.import_module(corr_sync.FLA_FLASHQLA_BACKEND_MODULE)
    cls = module.FlashQLABackend
    original = cls.chunk_gated_delta_rule
    if getattr(original, "_corr_det_probe_hook", False):
        return

    @functools.wraps(original)
    def chunk_gated_delta_rule(self, *args, **kwargs):
        corr_sync.maybe_apply(strict_arch=False)
        ensure(importlib.import_module(corr_sync.CP_CONTEXT_MODULE))
        return original(self, *args, **kwargs)

    chunk_gated_delta_rule._corr_det_probe_hook = True
    cls.chunk_gated_delta_rule = chunk_gated_delta_rule
