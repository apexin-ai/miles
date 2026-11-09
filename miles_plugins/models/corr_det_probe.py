"""Test-only probe for the M039 hdirect whole-model determinism run (v2). Off unless MILES_CORR_DET_PROBE=1.

Never use it in a training run: it keeps GPU copies of every correction call, adds work and writes evidence files.

v2 (RECHECK18): the verdict no longer rests on a checksum. Per rank:

* Call identity. Every GDN core forward that fla dispatches to FlashQLA with grad enabled gets a tag = its ordinal
  among this step's tagged forwards; the core output passes through an identity autograd node whose backward sets
  the "current tag" right before the core's own backward runs, so each backward correction call carries the tag of
  the forward it belongs to (layer x microbatch). The forward manifest (tag, q/k/v shapes, dtype, and a digest of
  cu_seqlens) is written every step; the checker requires the same manifest every step, every tag used by exactly
  one correction call, and n_GDN_layers x n_microbatches tags.
* Exact comparison. The reference step (step 1) keeps GPU clones of each call's inputs and output. In later steps
  each call is compared BITWISE with the reference of its tag: consumed input region equal? output equal? The flags
  stay on the GPU; the step's current inputs/outputs are kept on the GPU too, so at the step's end (one host
  transfer of the flags) every "same consumed input, different output" call and the first input-change calls are
  saved in full (live + reference tensors) as evidence, without any per-call host sync.
* Digests (descriptive, for grouping steps whose input differs from the reference): two independent splitmix64
  lanes over (index << 32 | raw bits) of each consumed region, seeded by dtype / shape / stride; summed mod 2**64
  (additive, position- and value-sensitive, non-linear). Equal digests are probabilistic evidence only
  (about 2**-64 per lane for a random difference; not a cryptographic hash).
* Per-parameter pre-reduction main_grad digests (same function) at the F8 guard's issuing start_grad_sync, after
  the original call order is preserved (hash first, then the original start_grad_sync).

Consumed regions (the reverse scan reads idx = end-1 .. start+1 of every raw sequence): dht_buffer blocks of those
segments; mt_buffer blocks of those segments where fallback is set; fallback_mask on those segments; seq_map_r2c.

Files (Q38_RUN_DIR/gt-evidence): corrdet-<rank>.jsonl, gradhash-<rank>.jsonl (one row per step, written at the F8
guard's end_step), corrdet-event-*.pt (full tensors).
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import threading
from pathlib import Path

ENV = "MILES_CORR_DET_PROBE"
MARK = "MILES_CORR_DET_PROBE=1: correction digests and per-parameter main_grad hashes enabled (test only, v2)"
MAX_CALLS = 1024
MAX_SAVED_FAIL = 8
MAX_SAVED_INPUT = 2
CHUNK = 1 << 24
MASK32 = 0xFFFFFFFF
_lock = threading.Lock()
logger = logging.getLogger(__name__)
_probe = None


def _signed(c: int) -> int:
    return c - (1 << 64) if c >= 1 << 63 else c


_GOLDEN, _M1, _M2 = _signed(0x9E3779B97F4A7C15), _signed(0xBF58476D1CE4E5B9), _signed(0x94D049BB133111EB)
_LANE_KEYS = (_signed(0x243F6A8885A308D3), _signed(0x13198A2E03707344))     # pi digits: two independent lanes


def enabled() -> bool:
    return os.environ.get(ENV, "").strip() == "1"


def _shr(z, s):
    return (z >> s) & ((1 << (64 - s)) - 1)


def _mix64(z):
    z = z + _GOLDEN
    z = (z ^ _shr(z, 30)) * _M1
    z = (z ^ _shr(z, 27)) * _M2
    return z ^ _shr(z, 31)


def _meta_seed(t, region: str) -> int:
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|{tuple(t.stride())}|{region}".encode()).digest()
    return _signed(int.from_bytes(h[:8], "little"))


def _bits(t):
    import torch
    t = t.detach().contiguous()
    if t.dtype == torch.bool:
        return t.view(torch.uint8), 0xFF
    size = t.element_size()
    if size == 8:
        return t.view(torch.int64), None
    return t.view({4: torch.int32, 2: torch.int16, 1: torch.uint8}[size]), {4: MASK32, 2: 0xFFFF, 1: 0xFF}[size]


def digest(t, mask=None, region="all"):
    """int64[4] on t's device: two splitmix64 lanes, each summed over the selected elements as (low 32-bit half,
    high 32-bit half) so no int64 sum overflows (< 2**31 elements). mask: block mask over the leading dims, or None
    = all. Element value AND index (and dtype/shape/stride/region through the seed) are mixed in."""
    import torch
    bits, m = _bits(t)
    flat = bits.reshape(-1)
    sel = None
    if mask is not None:
        mm = mask.to(t.device)
        while mm.dim() < t.dim():
            mm = mm.unsqueeze(-1)
        sel = mm.expand(t.shape).contiguous().reshape(-1)
    seed = _meta_seed(t, region)
    out = torch.zeros(4, dtype=torch.int64, device=t.device)
    for a in range(0, flat.numel(), CHUNK):
        b = min(flat.numel(), a + CHUNK)
        x = flat[a:b].to(torch.int64)
        idx = torch.arange(a, b, dtype=torch.int64, device=t.device)
        for lane, key in enumerate(_LANE_KEYS):
            if m is None:
                z = _mix64(_mix64((idx << 32) ^ (x & MASK32) ^ key ^ seed) ^ _shr(x, 32))
            else:
                z = _mix64((idx << 32) ^ (x & m) ^ key ^ seed)
            z = _mix64(z ^ seed)
            if sel is not None:
                z = z * sel[a:b].to(torch.int64)
            out[2 * lane] += (z & MASK32).sum()
            out[2 * lane + 1] += _shr(z, 32).sum()
    return out


def consumed_masks(fallback_mask, seq_map_r2c):
    """GPU-side, no host sync: (rows [S,H], mt [S,H]) bool masks of the regions the correction can read."""
    import torch
    fb = fallback_mask.bool()
    S, H = fb.shape
    read = torch.ones(S, dtype=torch.bool, device=fb.device)
    read[seq_map_r2c[:-1].long()] = False
    rows = read[:, None].expand(S, H).contiguous()
    return rows, rows & fb


def _bits_ne(a, b):
    import torch
    ba, _ = _bits(a)
    bb, _ = _bits(b)
    return ba != bb


def consumed_equal(kw, ref):
    """GPU bool scalar: the consumed input regions of kw equal those of ref bitwise (shapes/dtypes must match;
    the masks are the reference's; fallback and seq_map are compared on their consumed part / whole)."""
    import torch
    dev = kw["dht_buffer"].device
    for k in ("dht_buffer", "mt_buffer", "fallback_mask", "seq_map_r2c"):
        if kw[k].shape != ref[k].shape or kw[k].dtype != ref[k].dtype:
            return torch.zeros((), dtype=torch.bool, device=dev)
    rows, mt = ref["_rows"], ref["_mt"]
    ne = (_bits_ne(kw["dht_buffer"], ref["dht_buffer"]).flatten(2).any(-1) & rows).any()
    ne = ne | (_bits_ne(kw["mt_buffer"], ref["mt_buffer"]).flatten(2).any(-1) & mt).any()
    ne = ne | ((kw["fallback_mask"].bool() != ref["fallback_mask"].bool()) & rows).any()
    ne = ne | (kw["seq_map_r2c"] != ref["seq_map_r2c"]).any()
    return ~ne


class _Tag:
    """Identity autograd node on a GDN core output; its backward sets the probe's current tag."""
    fn = None

    @classmethod
    def get(cls):
        if cls.fn is None:
            import torch

            class TagFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, probe, tag):
                    ctx.probe, ctx.tag = probe, tag
                    return x.view_as(x)

                @staticmethod
                def backward(ctx, g):
                    ctx.probe.current_tag = ctx.tag
                    return g, None, None
            cls.fn = TagFn
        return cls.fn


class _Probe:
    def __init__(self, out_dir: Path, rank: int):
        self.out, self.rank = out_dir, rank
        self.step_index = 0                  # flushes so far; the reference step is the first one
        self.current_tag = None
        self.fwd = []                        # this step: [tag, shapes, dtype] per tagged forward
        self.fwd_cu = []                     # GPU int64[2] cu_seqlens digests, same order
        self.untagged_fwd = 0                # forwards with grad that could not be tagged
        self.nograd_fwd = 0
        self.calls = []                      # this step: dict(tag, flags (GPU), din, dout, live, out, flag)
        self.refs = {}                       # tag -> reference inputs/output (GPU clones) from the reference step
        self.untagged_calls = 0
        self.overflow = 0
        self.fn_names = set()
        self.grad = {}
        self.grad_calls = {}
        self.groups_total = 0
        self.events_saved = {"fail": 0, "input": 0}

    # ---- forward tags ----------------------------------------------------------------------------------
    def tag_forward(self, result, args, kwargs):
        import torch
        out = result[0] if isinstance(result, tuple) else result
        if not torch.is_tensor(out) or not out.requires_grad:
            self.nograd_fwd += 1
            return result
        tag = len(self.fwd)
        qkv = [args[i] if len(args) > i else kwargs.get(n) for i, n in enumerate(("q", "k", "v"))]
        q = qkv[0]
        cu = kwargs.get("cu_seqlens")
        shapes = [list(a.shape) if torch.is_tensor(a) else None for a in qkv]
        self.fwd.append([tag, shapes, str(getattr(q, "dtype", None)), cu is not None])
        self.fwd_cu.append(digest(cu, region="cu") if torch.is_tensor(cu) else torch.zeros(4, dtype=torch.int64,
                                                                                               device=out.device))
        tagged = _Tag.get().apply(out, self, tag)
        return (tagged,) + tuple(result[1:]) if isinstance(result, tuple) else tagged

    # ---- correction calls ------------------------------------------------------------------------------
    def call(self, fn, kw):
        import torch
        tag, self.current_tag = self.current_tag, None
        if tag is None:
            self.untagged_calls += 1
            return fn(**kw)
        if len(self.calls) >= MAX_CALLS:
            self.overflow += 1
            return fn(**kw)
        flag = 0 if (kw.get("raw_dht") is None and not kw.get("state_v_first")) else 1
        live = {k: kw[k].detach().clone() for k in ("dht_buffer", "mt_buffer", "fallback_mask", "seq_map_r2c")}
        rows, mt = consumed_masks(live["fallback_mask"], live["seq_map_r2c"])
        out = fn(**kw)                                       # the call itself is untouched
        o = out.detach().clone()
        drift = torch.stack([_bits_ne(kw[k], live[k]).any() if kw[k].shape == live[k].shape else
                             torch.ones((), dtype=torch.bool, device=o.device) for k in live]).any()
        din = torch.stack([digest(live["dht_buffer"], rows, "dht"), digest(live["mt_buffer"], mt, "mt"),
                           digest(live["fallback_mask"], rows, "fb"), digest(live["seq_map_r2c"], None, "r2c")])
        dout = digest(o, None, "out")
        ref = self.refs.get(tag)
        if self.step_index == 0:
            ieq = oeq = torch.ones((), dtype=torch.bool, device=o.device)
        elif ref is None:
            ieq = oeq = torch.zeros((), dtype=torch.bool, device=o.device)   # tag unknown at the reference step
        else:
            ieq = consumed_equal(live, ref)
            oeq = (~_bits_ne(o, ref["out"]).any()) if o.shape == ref["out"].shape and o.dtype == ref["out"].dtype \
                else torch.zeros((), dtype=torch.bool, device=o.device)
        self.calls.append({"tag": tag, "flags": torch.stack([ieq, oeq, drift]), "din": din, "dout": dout,
                           "flag": flag, "live": live, "out": o, "rows": rows, "mt": mt})
        return out

    def wrap_correction(self, cp_context) -> None:
        fn = cp_context.correct_terminal_states
        if getattr(fn, "_corr_det_probe", False):
            return

        @functools.wraps(fn)
        def probed(*args, **kw):
            if args:
                raise RuntimeError(f"{ENV}: correct_terminal_states called with positional arguments")
            return self.call(fn, kw)

        probed._corr_det_probe = True
        probed._corr_det_inner = fn
        cp_context.correct_terminal_states = probed
        self.fn_names.add(f"{getattr(fn, '__module__', None)}.{getattr(fn, '__qualname__', None)}")

    # ---- gradients -------------------------------------------------------------------------------------
    def hash_bucket(self, tag, bucket) -> None:
        import torch
        rows = [digest(p.main_grad, None, "grad") for p in bucket.params_list]
        self.grad[tag] = torch.stack(rows) if rows else torch.zeros(0, 4, dtype=torch.int64)
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
        self.groups_total = len(groups)

    # ---- per step ---------------------------------------------------------------------------------------
    def _save(self, kind, step, c, extra):
        import torch
        p = self.out / f"corrdet-event-{kind}-r{self.rank:05d}-s{step:03d}-t{c['tag']:04d}.pt"
        torch.save({"meta": {"kind": kind, "step": step, "rank": self.rank, "tag": c["tag"]},
                    "live_inputs": {k: v.cpu() for k, v in c["live"].items()}, "live_out": c["out"].cpu(),
                    "consumed_rows": c["rows"].cpu(), "consumed_mt": c["mt"].cpu(), **extra}, p)
        return {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def flush(self, step: int, grad_norm) -> None:
        import torch
        self.out.mkdir(parents=True, exist_ok=True)
        n = len(self.calls)
        if n:
            packed = torch.cat([torch.stack([c["flags"].to(torch.int64) for c in self.calls]),
                                torch.stack([c["din"].reshape(-1) for c in self.calls]),
                                torch.stack([c["dout"] for c in self.calls])], dim=1).cpu().tolist()   # one transfer
        else:
            packed = []
        cu = torch.stack(self.fwd_cu).cpu().tolist() if self.fwd_cu else []
        tags = sorted(self.grad)
        gh = torch.cat([self.grad[t].to(self.grad[tags[0]].device).reshape(-1, 4) for t in tags]).cpu().tolist() if tags else []
        events = []
        rows = []
        for c, p in zip(self.calls, packed):
            ieq, oeq, drift = bool(p[0]), bool(p[1]), bool(p[2])
            rec = {"tag": c["tag"], "in_eq_ref": ieq, "out_eq_ref": oeq, "in_call_drift": drift, "din": p[3:19],
                   "dout": p[19:23], "flag": c["flag"]}
            if self.step_index > 0 and c["tag"] in self.refs:
                ref = self.refs[c["tag"]]
                if ieq and not oeq and self.events_saved["fail"] < MAX_SAVED_FAIL:
                    rec["event"] = self._save("fail", step, c, {"ref_inputs": {k: v.cpu() for k, v in ref.items()
                                                                               if not k.startswith("_") and k != "out"},
                                                                "ref_out": ref["out"].cpu()})
                    self.events_saved["fail"] += 1
                elif not ieq and self.events_saved["input"] < MAX_SAVED_INPUT:
                    rec["event"] = self._save("input", step, c, {"ref_inputs": {k: v.cpu() for k, v in ref.items()
                                                                                if not k.startswith("_") and k != "out"}})
                    self.events_saved["input"] += 1
            rows.append(rec)
        if self.step_index == 0:
            for c in self.calls:
                self.refs[c["tag"]] = dict(c["live"], out=c["out"], _rows=c["rows"], _mt=c["mt"])
        with (self.out / f"corrdet-{self.rank:05d}.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "rank": self.rank, "reference_step": self.step_index == 0,
                                "n_calls": n, "overflow": self.overflow, "untagged_calls": self.untagged_calls,
                                "untagged_forwards": self.untagged_fwd, "nograd_forwards": self.nograd_fwd,
                                "forwards": [fw + [cu[i]] for i, fw in enumerate(self.fwd)],
                                "corr_fn": sorted(self.fn_names), "calls": rows}) + "\n")
        hashes, i = {}, 0
        for t in tags:
            k = self.grad[t].shape[0]
            hashes[t] = gh[i:i + k]; i += k
        with (self.out / f"gradhash-{self.rank:05d}.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "rank": self.rank, "grad_norm": float(grad_norm),
                                "groups_wrapped": self.groups_total, "calls": {t: self.grad_calls[t] for t in tags},
                                "hashes": hashes}) + "\n")
        self.step_index += 1
        self.calls, self.fwd, self.fwd_cu = [], [], []
        self.untagged_calls = self.untagged_fwd = self.nograd_fwd = self.overflow = 0
        self.grad, self.grad_calls = {}, {}
        self.current_tag = None


def _install_probe(f8_guard_module, rank: int, out_dir: Path) -> _Probe:
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
    logger.warning(MARK)
    _probe = probe
    return probe


def ensure(cp_context, f8_guard_module=None, rank=None, out_dir=None) -> _Probe:
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
            probe = _install_probe(f8_guard_module, rank, out_dir)
        probe.wrap_correction(cp_context)
        return probe


def install_for_backend(backend: str, corr_sync) -> None:
    """Called by flashqla_corr_sync.apply_for_backend when MILES_CORR_DET_PROBE=1 (backend "fla" only: the v14
    configuration; any other backend is refused so the run cannot silently lose its call identity). Wraps fla's
    FlashQLABackend.chunk_gated_delta_rule: corr_sync.maybe_apply first (hdirect, if requested, is verified against
    the ORIGINAL function), then ensure() wraps the installed correction, then the forward runs and its output is
    tagged."""
    import importlib
    if backend != "fla":
        raise RuntimeError(f"{ENV} supports the 'fla' GDN backend only, got {backend!r}")
    module = importlib.import_module(corr_sync.FLA_FLASHQLA_BACKEND_MODULE)
    cls = module.FlashQLABackend
    original = cls.chunk_gated_delta_rule
    if getattr(original, "_corr_det_probe_hook", False):
        return

    @functools.wraps(original)
    def chunk_gated_delta_rule(self, *args, **kwargs):
        corr_sync.maybe_apply(strict_arch=False)
        probe = ensure(importlib.import_module(corr_sync.CP_CONTEXT_MODULE))
        return probe.tag_forward(original(self, *args, **kwargs), args, kwargs)

    chunk_gated_delta_rule._corr_det_probe_hook = True
    cls.chunk_gated_delta_rule = chunk_gated_delta_rule
