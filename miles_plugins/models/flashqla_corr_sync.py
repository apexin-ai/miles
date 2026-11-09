"""Opt-in workaround for the nondeterministic FlashQLA CP backward state correction.

FlashQLA ``0.1.2+7c7dfe1`` (QwenLM/FlashQLA, Blackwell path): with intra-card CP enabled, the backward
correction ``cp_context.correct_terminal_states`` runs ``blackwell.cp_fwd.tilelang_correct_h0``
(``reverse=True, transpose_m=True``, FP32 buffers), whose segment loop is a 2-stage ``T.Pipelined``.
On B300 with concurrent NCCL traffic it sometimes returns a different ``dht`` for the first CP
segments of the reverse scan (one 8-column V group of one head). The error then reaches dk/dv/dbeta/dg
of that segment (dq never changes). Measured on node 127 with frozen inputs: 20/48000 calls differed;
with ``num_stages=1``: 0/48000 in one run, but 2/48000 in a second run (v11 shipping arm), so
``num_stages=1`` only lowers the rate. All captured events (31) are in the copy of ``ht`` through
``h_shared`` (the consumer reads ``h_shared`` and releases it for the next TMA into the same buffer); no
event started on an iteration that applied a nonzero correction. The fault is in this kernel;
``fused_gdr_dh`` and ``fused_gdr_bwd`` only consume or propagate it.

``MILES_FLASHQLA_CORR_SYNC`` selects a replacement for the backward correction only (default: off).

* ``hdirect`` (candidate primary): the same TileLang kernel with ``num_stages=1`` and ``h`` read
  straight from ``ht_buffer`` in global memory into registers (no ``h_shared``, no TMA and no mbarrier
  for ``h``; out-of-range segments read 0, as the TMA zero fill does). ``M`` still arrives by TMA. Same
  arithmetic per element (FP32 copy, TF32 MMA), so it is expected to equal the original kernel bit for bit
  whenever the original is correct (to be shown on GPU by the equiv arm).
* ``stages1`` (kept for comparison, not recommended): the same TileLang kernel compiled with
  ``num_stages=1``. Same arithmetic per element; lowers but does not remove the race.
* ``torch``: a PyTorch reverse scan. Deterministic. Non-fallback segments are exact copies; fallback
  segments compute ``M^T @ h`` in FP64 (independent of the global matmul precision setting) instead of
  the kernel's TF32 MMA, so they differ from the kernel by roughly the TF32 error.

Rules:

* All modes refuse to apply unless the installed ``cp_fwd.py`` and ``cp_context.py`` match the
  reviewed sha256 values, and ``cp_context.correct_terminal_states`` is still the original function.
* Once applied, the mode cannot change or be switched off in the same process (an off/unset value after
  applying raises), because callers may already hold the replaced function.
* The forward correction (``correct_initial_states``) is not touched.
* ``gdn_kernel`` calls :func:`apply_for_backend`. ``backend="flashqla"`` always uses FlashQLA, so the
  patch is applied there; a device that is not Blackwell (compute capability 10.x) is an error, checked
  before FlashQLA is imported. ``backend="fla"`` decides per call, exactly as fla does: the hook wraps
  fla's ``FlashQLABackend.chunk_gated_delta_rule``, which fla's dispatcher calls only after its own
  checks pass (dispatch state fixed at fla import, ``FLA_FLASH_QLA``, package present, and the verifier:
  architecture, dtype, K=V=128, gate options, ``cp_context is None``). The patch is applied on the first
  such call; a non-Blackwell device is skipped with a warning (the defect is Blackwell-specific).
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import logging
import os
import tempfile
import threading
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

ENV = "MILES_FLASHQLA_CORR_SYNC"
MODES = ("stages1", "hdirect", "torch")
# sha256 of the FlashQLA sources this patch was reviewed against (flash_qla 0.1.2+7c7dfe1).
REVIEWED_CP_FWD_SHA256 = {"8cfc9059acffd08c968cfcf9622e0527eb961d77de5c57b91ffb08b02fe7ff27": "0.1.2+7c7dfe1"}
REVIEWED_CP_CONTEXT_SHA256 = {"4272d8d983b0af64aeb9a9fb47032a8a46a899376e2c032719b634c18510e2b8": "0.1.2+7c7dfe1"}
CP_CONTEXT_MODULE = "flash_qla.ops.gated_delta_rule.chunk.cp_context"
FLA_FLASHQLA_BACKEND_MODULE = "fla.ops.gated_delta_rule.backends.flash_qla"
_ANCHOR = "        for i_s in T.Pipelined(num_iters - 1, num_stages=2):\n"
_REPLACEMENT = "        for i_s in T.Pipelined(num_iters - 1, num_stages=1):  # miles flashqla_corr_sync\n"
# hdirect: the h_shared load (both layouts) is removed and the h_shared -> h_fragment copy becomes a direct
# global -> h_fragment copy of the same slice.
_H_LOAD_ANCHOR = (
    "            if state_v_first:\n"
    "                T.copy(\n"
    "                    ht_buffer[idx, bh, DV_start:DV_end, 0:DK],\n"
    "                    h_shared,\n"
    "                )\n"
    "            else:\n"
    "                T.copy(\n"
    "                    ht_buffer[idx, bh, 0:DK, DV_start:DV_end],\n"
    "                    h_shared,\n"
    "                )\n"
)
_H_LOAD_REPLACEMENT = "            pass  # miles flashqla_corr_sync hdirect: h is read from global below, not via h_shared\n"
_H_USE_ANCHOR = "            T.copy(h_shared, h_fragment)\n"
_H_USE_REPLACEMENT = (
    "            if state_v_first:  # miles flashqla_corr_sync hdirect\n"
    "                T.copy(ht_buffer[idx, bh, DV_start:DV_end, 0:DK], h_fragment)\n"
    "            else:\n"
    "                T.copy(ht_buffer[idx, bh, 0:DK, DV_start:DV_end], h_fragment)\n"
)
KERNEL_MODES = ("stages1", "hdirect")

_lock = threading.Lock()
_applied: str | None = None
_fla_hooked = False


def _blackwell_device() -> bool:
    """True on a CUDA device with compute capability 10.x (sm100 / sm103). Never imports FlashQLA."""
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
    except Exception:
        return False


def requested_mode() -> str | None:
    mode = os.environ.get(ENV, "").strip().lower()
    if mode in ("", "0", "off", "none", "false"):
        return None
    if mode not in MODES:
        raise ValueError(f"{ENV}={mode!r}: expected one of {MODES} or unset")
    return mode


def _sha256_file(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_cp_fwd_source(text: str, mode: str = "stages1") -> str:
    """Return cp_fwd.py source with tilelang_correct_h0's segment loop at num_stages=1 and, for
    ``mode="hdirect"``, with h read from global memory instead of through h_shared."""
    if mode not in KERNEL_MODES:
        raise ValueError(f"no source patch for mode {mode!r}")
    sha = hashlib.sha256(text.encode()).hexdigest()
    if sha not in REVIEWED_CP_FWD_SHA256:
        raise RuntimeError(f"flash_qla cp_fwd.py sha256 {sha} was not reviewed for {ENV}")
    anchors = [(_ANCHOR, _REPLACEMENT)]
    if mode == "hdirect":
        anchors += [(_H_LOAD_ANCHOR, _H_LOAD_REPLACEMENT), (_H_USE_ANCHOR, _H_USE_REPLACEMENT)]
    for anchor, _ in anchors:
        if text.count(anchor) != 1:
            raise RuntimeError(f"tilelang_correct_h0 anchor not found exactly once: {anchor.strip()[:60]!r}")
    for anchor, replacement in anchors:
        text = text.replace(anchor, replacement)
    return text


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
    return _load_patched_module(cp_fwd_module, "stages1")


def _load_patched_module(cp_fwd_module, mode: str):
    text = Path(cp_fwd_module.__file__).read_text()
    patched = patch_cp_fwd_source(text, mode)
    digest = hashlib.sha256(patched.encode()).hexdigest()[:16]
    cache = Path(os.environ.get("MILES_FLASHQLA_PATCH_DIR", tempfile.gettempdir())) / "miles_flashqla_corr_sync"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"cp_fwd_{mode}_{digest}.py"
    if not path.exists() or path.read_text() != patched:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(patched)
        os.replace(tmp, path)  # atomic; every rank writes the same bytes
    name = f"miles_flashqla_cp_fwd_{mode}_{digest}"
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
        if not _blackwell_device():  # before importing FlashQLA: its import raises on e.g. sm80
            msg = f"{ENV} covers Blackwell (compute capability 10.x) devices only"
            if strict_arch:
                raise RuntimeError(msg)
            logger.warning("%s; not applied", msg)
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
        if mode in KERNEL_MODES:
            replacement = _load_patched_module(cp_fwd, mode).correct_terminal_states
        else:
            replacement = correct_terminal_states_torch
        cp_context.correct_terminal_states = replacement
        _applied = mode
        logger.warning("%s=%s: FlashQLA backward CP state correction replaced", ENV, mode)
        return mode


def _install_fla_backend_hook() -> bool:
    """Wrap fla's FlashQLABackend.chunk_gated_delta_rule so the patch is applied exactly when fla's
    dispatcher actually selects FlashQLA for a call. Imports fla's backend module, never FlashQLA."""
    global _fla_hooked
    with _lock:
        if _fla_hooked:
            return True
        try:
            module = importlib.import_module(FLA_FLASHQLA_BACKEND_MODULE)
        except ImportError as exc:
            logger.warning("%s: fla has no FlashQLA backend (%r); nothing to patch on the fla path", ENV, exc)
            return False
        cls = module.FlashQLABackend
        original = cls.chunk_gated_delta_rule
        if not getattr(original, "_miles_flashqla_corr_sync", False):
            @functools.wraps(original)
            def chunk_gated_delta_rule(self, *args, **kwargs):
                maybe_apply(strict_arch=False)
                return original(self, *args, **kwargs)

            chunk_gated_delta_rule._miles_flashqla_corr_sync = True
            cls.chunk_gated_delta_rule = chunk_gated_delta_rule
        _fla_hooked = True
        return True


def apply_for_backend(backend: str) -> str | None:
    """Hook for ``gdn_kernel``. Returns the applied mode (None when nothing is applied yet)."""
    if os.environ.get("MILES_CORR_DET_PROBE", "").strip() == "1":  # test-only determinism probe (corr_det_probe)
        from miles_plugins.models import corr_det_probe

        corr_det_probe.install_for_backend(backend, importlib.import_module(__name__))
    mode = requested_mode()
    with _lock:
        if _applied is not None and mode != _applied:
            raise RuntimeError(f"{ENV} was applied as {_applied!r}; it cannot be changed or switched off")
    if mode is None:
        return None
    if backend == "flashqla":
        return maybe_apply(strict_arch=True)
    if backend == "fla":
        _install_fla_backend_hook()  # applied lazily on the first call fla dispatches to FlashQLA
        return _applied
    return None
