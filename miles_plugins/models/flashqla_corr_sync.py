"""Opt-in workaround for the nondeterministic FlashQLA CP backward state correction.

FlashQLA ``0.1.2+7c7dfe1`` (QwenLM/FlashQLA, Blackwell path): with intra-card CP enabled, the backward
correction ``cp_context.correct_terminal_states`` runs ``blackwell.cp_fwd.tilelang_correct_h0``
(``reverse=True, transpose_m=True``, FP32 buffers), whose segment loop is a 2-stage ``T.Pipelined``.
On B300 with concurrent NCCL traffic it sometimes returns a different ``dht`` for the first CP
segments of the reverse scan (one 8-column V group of one head). The error then reaches dk/dv/dbeta/dg
of that segment (dq never changes). Measured on node 127 with frozen inputs: 20/48000 calls differed;
with ``num_stages=1``: 0/48000 (one run; an upper bound, not a proof that the race is gone). The fault
is in this kernel; ``fused_gdr_dh`` and ``fused_gdr_bwd`` only consume or propagate it.

``MILES_FLASHQLA_CORR_SYNC`` selects a replacement for the backward correction only (default: off).

* ``stages1``: the same TileLang kernel compiled with ``num_stages=1``. Same arithmetic per element;
  the next segment's loads are not prefetched into a second buffer.
* ``torch``: a PyTorch reverse scan. Deterministic. Non-fallback segments are exact copies; fallback
  segments compute ``M^T @ h`` in FP64 (independent of the global matmul precision setting) instead of
  the kernel's TF32 MMA, so they differ from the kernel by roughly the TF32 error.

Rules:

* Both modes refuse to apply unless the installed ``cp_fwd.py`` and ``cp_context.py`` match the
  reviewed sha256 values, and ``cp_context.correct_terminal_states`` is still the original function.
* Once applied, the mode cannot change or be switched off in the same process (an off/unset value after
  applying raises), because callers may already hold the replaced function.
* The forward correction (``correct_initial_states``) is not touched.
* ``gdn_kernel`` calls :func:`apply_for_backend`. ``backend="flashqla"`` always uses FlashQLA, so the
  patch is applied there (non-Blackwell is an error). ``backend="fla"`` is patched only when fla would
  dispatch to FlashQLA (dispatch not disabled, ``FLA_FLASH_QLA`` not ``"0"``, ``flash_qla`` installed);
  a non-Blackwell FlashQLA path is skipped with a warning, because the defect is Blackwell-specific.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ENV = "MILES_FLASHQLA_CORR_SYNC"
MODES = ("stages1", "torch")
# sha256 of the FlashQLA sources this patch was reviewed against (flash_qla 0.1.2+7c7dfe1).
REVIEWED_CP_FWD_SHA256 = {"8cfc9059acffd08c968cfcf9622e0527eb961d77de5c57b91ffb08b02fe7ff27": "0.1.2+7c7dfe1"}
REVIEWED_CP_CONTEXT_SHA256 = {"4272d8d983b0af64aeb9a9fb47032a8a46a899376e2c032719b634c18510e2b8": "0.1.2+7c7dfe1"}
CP_CONTEXT_MODULE = "flash_qla.ops.gated_delta_rule.chunk.cp_context"
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


def _sha256_file(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_cp_fwd_source(text: str) -> str:
    """Return cp_fwd.py source with tilelang_correct_h0's segment loop at num_stages=1."""
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha not in REVIEWED_CP_FWD_SHA256:
        raise RuntimeError(f"flash_qla cp_fwd.py sha256 {sha} was not reviewed for {ENV}")
    if text.count(_ANCHOR) != 1:
        raise RuntimeError("tilelang_correct_h0 pipeline anchor not found exactly once")
    return text.replace(_ANCHOR, _REPLACEMENT)


def verify_sources(cp_context, cp_fwd) -> None:
    """Refuse unless both FlashQLA source files are the reviewed versions and nothing patched them yet."""
    for module, reviewed in ((cp_fwd, REVIEWED_CP_FWD_SHA256), (cp_context, REVIEWED_CP_CONTEXT_SHA256)):
        path = getattr(module, "__file__", None)
        if not path or not Path(path).is_file():
            raise RuntimeError(f"{ENV}: cannot locate the source of {module.__name__}")
        sha = _sha256_file(path)
        if sha not in reviewed:
            raise RuntimeError(f"{ENV}: {module.__name__} sha256 {sha} was not reviewed")
    if getattr(cp_fwd, "correct_terminal_states", None) is not cp_context.correct_terminal_states:
        raise RuntimeError(f"{ENV}: cp_context.correct_terminal_states is not the original cp_fwd function")


def correct_terminal_states_torch(raw_dht, dht_buffer, mt_buffer, fallback_mask, seq_map_r2c,
                                  state_v_first: bool = False):
    """PyTorch version of ``correct_terminal_states``.

    For each raw sequence, scan its CP segments from the end:
    ``cp[idx] = h; h = buf[idx] + fallback[idx] * M[idx]^T @ h``, and finally ``cp[start] = h``.
    ``seq_map_r2c`` is read on the host once per call (a device-to-host sync).
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
            # The matmul also reads mt blocks of non-fallback heads (possibly never written); torch.where
            # masks those results out, so they cannot reach the output.
            corr = torch.matmul(mt_buffer[idx].double().transpose(-1, -2), h.double()).float()
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
    if Path(module.__file__).read_text() != patched:
        raise RuntimeError(f"{ENV}: loaded module does not hold the patched source")
    return module


def maybe_apply(strict_arch: bool = True) -> str | None:
    """Apply the replacement selected by ``MILES_FLASHQLA_CORR_SYNC`` once per process; return the mode.

    With ``strict_arch=False`` a non-Blackwell FlashQLA path is skipped (returns None) instead of raising.
    """
    global _applied
    mode = requested_mode()
    with _lock:
        if _applied is not None:
            if mode != _applied:
                raise RuntimeError(f"{ENV} was applied as {_applied!r}; changing it to {mode!r} (including "
                                   "off/unset) within one process is not supported")
            return mode
        if mode is None:
            return None
        cp_context = importlib.import_module(CP_CONTEXT_MODULE)
        original = cp_context.correct_terminal_states
        if not original.__module__.endswith("blackwell.cp_fwd"):
            msg = f"{ENV} covers the Blackwell (sm100/sm103) path only, got {original.__module__}"
            if strict_arch:
                raise RuntimeError(msg)
            logger.warning("%s; not applied", msg)
            return None
        cp_fwd = importlib.import_module(original.__module__)
        verify_sources(cp_context, cp_fwd)
        if mode == "stages1":
            replacement = _load_stages1_module(cp_fwd).correct_terminal_states
        else:
            replacement = correct_terminal_states_torch
        cp_context.correct_terminal_states = replacement
        _applied = mode
        logger.warning("%s=%s: FlashQLA backward CP state correction replaced", ENV, mode)
        return mode


def _flash_qla_installed() -> bool:
    if "flash_qla" in sys.modules:
        return sys.modules["flash_qla"] is not None
    try:
        return importlib.util.find_spec("flash_qla") is not None
    except (ImportError, ValueError):
        return False


def fla_dispatches_to_flashqla() -> bool:
    """Mirror fla's backend switch: dispatch on unless FLA_DISABLE_BACKEND_DISPATCH=1, FlashQLA on unless
    FLA_FLASH_QLA=0, and FlashQLA must be installed."""
    if os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1":
        return False
    if os.environ.get("FLA_FLASH_QLA", "1") == "0":
        return False
    return _flash_qla_installed()


def apply_for_backend(backend: str) -> str | None:
    """Hook for ``gdn_kernel``: apply only on paths that actually run FlashQLA."""
    if backend == "flashqla":
        return maybe_apply(strict_arch=True)
    if backend == "fla":
        if requested_mode() is None or not fla_dispatches_to_flashqla():
            with _lock:
                if _applied is not None and requested_mode() != _applied:
                    raise RuntimeError(f"{ENV} was applied as {_applied!r}; it cannot be changed or switched off")
            return _applied
        return maybe_apply(strict_arch=False)
    return None
