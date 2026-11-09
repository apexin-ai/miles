"""Opt-in workaround for the nondeterministic FlashQLA CP backward state correction.

FlashQLA ``0.1.2+7c7dfe1`` (QwenLM/FlashQLA, Blackwell path): with intra-card CP enabled, the backward
correction ``cp_context.correct_terminal_states`` runs ``blackwell.cp_fwd.tilelang_correct_h0``
(``reverse=True, transpose_m=True``, FP32 buffers), whose segment loop is a 2-stage ``T.Pipelined``.
On B300 with concurrent NCCL traffic it sometimes returns a different ``dht`` for the first CP
segments of the reverse scan (one 8-column V group of one head). The error then reaches dk/dv/dbeta/dg
of that segment (dq never changes). Measured on node 127 with frozen inputs: 20/48000 calls differed;
with ``num_stages=1``: 0/48000. The fault is in this kernel; ``fused_gdr_dh`` and ``fused_gdr_bwd`` only
consume or propagate it.

``MILES_FLASHQLA_CORR_SYNC`` selects a replacement for the backward correction only (default: off).

* ``stages1``: the same TileLang kernel compiled with ``num_stages=1``. Same arithmetic per element;
  the next segment's TMA load starts only after the current one is consumed.
* ``torch``: a pure-PyTorch reverse scan. Deterministic; non-fallback segments are exact copies, while
  fallback segments use a PyTorch matmul instead of the kernel's TF32 MMA (small numeric difference).

The forward correction (``correct_initial_states``) is not touched. The patch refuses to apply when
the installed ``cp_fwd.py`` differs from the reviewed version.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import tempfile
import threading
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ENV = "MILES_FLASHQLA_CORR_SYNC"
MODES = ("stages1", "torch")
# sha256 of flash_qla/ops/gated_delta_rule/chunk/blackwell/cp_fwd.py that this patch was reviewed against.
REVIEWED_CP_FWD_SHA256 = {
    "8cfc9059acffd08c968cfcf9622e0527eb961d77de5c57b91ffb08b02fe7ff27": "0.1.2+7c7dfe1",
}
_ANCHOR = "        for i_s in T.Pipelined(num_iters - 1, num_stages=2):\n"
_REPLACEMENT = "        for i_s in T.Pipelined(num_iters - 1, num_stages=1):  # miles flashqla_corr_sync\n"

_lock = threading.Lock()
_applied: str | None = None


def requested_mode() -> str | None:
    mode = os.environ.get(ENV, "").strip().lower()
    if mode in ("", "0", "off", "none", "false"):
        return None
    if mode not in MODES:
        raise ValueError(f"{ENV}={mode!r}: expected one of {MODES} or unset")
    return mode


def patch_cp_fwd_source(text: str) -> str:
    """Return cp_fwd.py source with tilelang_correct_h0's segment loop at num_stages=1."""
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha not in REVIEWED_CP_FWD_SHA256:
        raise RuntimeError(f"flash_qla cp_fwd.py sha256 {sha} was not reviewed for {ENV}")
    if text.count(_ANCHOR) != 1:
        raise RuntimeError("tilelang_correct_h0 pipeline anchor not found exactly once")
    return text.replace(_ANCHOR, _REPLACEMENT)


def correct_terminal_states_torch(raw_dht, dht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
                                  state_v_first: bool = False):
    """PyTorch version of ``correct_terminal_states``.

    For each raw sequence, scan its CP segments from the end:
    ``cp[idx] = h; h = buf[idx] + fallback[idx] * M[idx]^T @ h``, and finally ``cp[start] = h``.
    """
    if state_v_first:
        raise NotImplementedError("state_v_first=True is not covered by this workaround")
    res_dtype = torch.float32 if raw_dht is None else raw_dht.dtype
    cp = torch.empty(dht_buffer.shape, dtype=res_dtype, device=dht_buffer.device)
    r2c = [int(x) for x in seq_map_r2c.tolist()]
    fb = fallback_mask.bool()
    for b in range(len(r2c) - 1):
        start, end = r2c[b], r2c[b + 1]
        n = end - start
        if raw_dht is not None:
            h = raw_dht[b].float()
        else:
            h = torch.zeros(dht_buffer.shape[1:], dtype=torch.float32, device=dht_buffer.device)
        for i in range(n - 1):
            idx = start + n - 1 - i
            cp[idx] = h.to(res_dtype)
            buf = dht_buffer[idx].float()
            # Unwritten (non-fallback) mt blocks may hold garbage; torch.where never reads them back.
            corr = torch.matmul(mt_buffer[idx].float().transpose(-1, -2), h)
            h = torch.where(fb[idx][:, None, None], buf + corr, buf)
        cp[start] = h.to(res_dtype)
    return cp


def _load_stages1_module(cp_fwd_module):
    text = Path(cp_fwd_module.__file__).read_text()
    patched = patch_cp_fwd_source(text)
    digest = hashlib.sha256(patched.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("MILES_FLASHQLA_PATCH_DIR", tempfile.gettempdir())) / "miles_flashqla_corr_sync"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"cp_fwd_stages1_{digest}.py"
    if not path.exists() or path.read_text() != patched:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(patched)
        os.replace(tmp, path)  # atomic; every rank writes the same bytes
    name = f"miles_flashqla_cp_fwd_stages1_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def maybe_apply() -> str | None:
    """Apply the replacement selected by ``MILES_FLASHQLA_CORR_SYNC`` once per process; return the mode."""
    global _applied
    mode = requested_mode()
    if mode is None:
        return None
    with _lock:
        if _applied is not None:
            if _applied != mode:
                raise RuntimeError(f"{ENV} changed from {_applied!r} to {mode!r} within one process")
            return mode
        import flash_qla.ops.gated_delta_rule.chunk.cp_context as cp_context

        original = cp_context.correct_terminal_states
        if not original.__module__.endswith("blackwell.cp_fwd"):
            raise RuntimeError(f"{ENV} covers the Blackwell (sm100/sm103) path only, got {original.__module__}")
        if mode == "stages1":
            import importlib

            cp_fwd = importlib.import_module(original.__module__)
            cp_context.correct_terminal_states = _load_stages1_module(cp_fwd).correct_terminal_states
        else:
            cp_context.correct_terminal_states = correct_terminal_states_torch
        _applied = mode
        logger.warning("%s=%s: FlashQLA backward CP state correction replaced", ENV, mode)
        return mode
