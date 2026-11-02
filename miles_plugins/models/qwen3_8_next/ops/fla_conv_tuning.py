"""One fixed launch config for fla's NB-keyed Triton kernels on the GDN path (``--qwen38-fla-conv-pinned``).

fla (0.5.2) autotunes ``causal_conv1d_fwd_kernel`` / ``causal_conv1d_bwd_kernel`` -- the GDN short
convolution -- over 16 configs keyed by ``['D', 'W', 'NB']`` with ``NB = cdiv(B * T, 1024)``, and ``NB``
is also a ``tl.constexpr``. Every new ``NB`` (every new row length, to 1024 tokens) therefore compiles
and benchmarks 16 kernels: about 15 s in the forward and 2 x 18 s in the backward on a B300, with the
whole job waiting at the next collective. The results go to the node's Triton cache
(``cache_results``), so a resume replaying the same rows is fast, and a CP1 run on a seeded cache never
sees it. Under context parallelism fla's ``causal_conv1d_cp`` passes an ``initial_state`` (the previous
rank's tail), which is another autotune key, so a CP2 run tunes every local ``NB`` again.

The kernel bodies never read ``NB`` and the config only tiles the channels (``BD``) and sets the warp
count, so this pins one config per kernel (the fastest or within 2.5 % of it at D = 2560, NB 12-256 on
B300, from the tuning records) and passes ``NB = 1``: one compile per kernel variant for the whole run,
no benchmarking, and every rank and every restart runs the same config. The forward output does not
depend on the config; the backward's per-chunk weight-gradient partials are an fp32 ``tl.sum`` over
the chunk's tokens whose order follows the layout, so they can differ from an autotuned config's in
the last fp32 bit -- as two autotuned runs that pick different configs already do.

The same flag pins the two other NB-keyed kernel families on the GDN path (``pin_fla_norm_autotune``):
``layer_norm_gated_{fwd,bwd}_kernel`` (``FusedRMSNormGated`` on the GDN output, 9 configs ``BT x
num_warps``) and ``l2norm_{fwd,bwd}_kernel`` (the in-kernel q/k L2 norm, 25 configs), both keyed by
``NB = cdiv(rows, 65536)`` (rows = tokens x local heads, head dim 128 <= 512). Their bodies do not read
``NB`` either; ``BT`` only tiles the rows. Under CP2 they see ~20-24 / ~8 local ``NB`` values, and each
new one costs ~6 s (gated norm, fwd + bwd) / ~16 s (l2norm) of compiling and benchmarking on a B300, a
few minutes per node in all. Many configs time within noise of each other, so ranks and runs pick
different ones (two cold runs did). On B300 the configs autotune picks give the same output bits as
the pinned ones (6 of 9 gated-norm backward configs, 8 of 9 forward and 22 of 25 l2norm configs agree
bitwise; the pinned configs are in those groups), so on the shapes we tested the pinned path gives the
same output bits as the autotuned one (bitwise identical on the tested shapes; other shapes are not
proven) and removes the cold cost.
"""

from __future__ import annotations

import importlib

# kernel name -> (BD, num_warps)
PINNED = {
    "causal_conv1d_fwd_kernel": (64, 4),
    "causal_conv1d_bwd_kernel": (32, 4),
}
PINNED_NB = 1
_MARK = "_qwen38_conv_pinned"

# (fla module, kernel name) -> (BT, num_warps), D = 128 on B300 (GBSMIG P2 microbench, NB 1-22 / 1-8):
# the gated-norm pair is what autotune picks at almost every NB (fwd within 0.7 %, bwd the fastest by 6.7 %);
# l2norm fwd (32, 8) is within 2.4 % of the per-NB best (mean 0.7 %), l2norm bwd (8, 4) the fastest at all NB.
PINNED_NORM = {
    ("fla.modules.fused_norm_gate", "layer_norm_gated_fwd_kernel"): (32, 8),
    ("fla.modules.fused_norm_gate", "layer_norm_gated_bwd_kernel"): (32, 16),
    ("fla.modules.l2norm", "l2norm_fwd_kernel"): (32, 8),
    ("fla.modules.l2norm", "l2norm_bwd_kernel"): (8, 4),
}


def _autotuner(kernel):
    """The autotuner under fla's ``triton.heuristics`` wrapper (the object holding ``configs``)."""
    obj = kernel
    for _ in range(4):
        if hasattr(obj, "configs") and hasattr(obj, "run"):
            return obj
        obj = getattr(obj, "fn", None)
    raise RuntimeError(f"no autotuner under {kernel!r}: fla's NB-keyed kernels changed shape")


def _pin(tuner, kwargs: dict, num_warps: int, nb: int) -> None:
    """Keep only the config with ``kwargs`` and ``num_warps``, and pass ``NB=nb``; idempotent."""
    if getattr(tuner, _MARK, False):
        return
    if "NB" not in getattr(tuner, "keys", ["NB"]):
        raise RuntimeError(f"NB is no longer an autotune key of {tuner!r}: fla changed, re-check the pin")
    match = [
        c for c in tuner.configs if all(c.kwargs.get(k) == v for k, v in kwargs.items()) and c.num_warps == num_warps
    ]
    if len(match) != 1:
        raise RuntimeError(
            f"expected one config {kwargs} num_warps={num_warps} among "
            f"{[(c.kwargs, c.num_warps) for c in tuner.configs]}"
        )
    tuner.configs = match
    run = tuner.run

    def pinned_run(*args, **kw):
        if "NB" in kw:
            kw["NB"] = nb
        return run(*args, **kw)

    tuner.run = pinned_run
    setattr(tuner, _MARK, True)


def pin_autotuner(tuner, block_d: int, num_warps: int, nb: int = PINNED_NB) -> None:
    """Keep only the config ``(BD=block_d, num_warps)`` and pass ``NB=nb``; idempotent."""
    _pin(tuner, {"BD": block_d}, num_warps, nb)


def pin_fla_conv_autotune(kernels_module=None) -> None:
    """Pin both causal_conv1d kernels of ``fla.modules.conv.triton.kernels`` (or ``kernels_module``).

    Mutates the kernel objects in place, so ``fla.modules.conv.triton.ops`` (which imported them by
    name) and fla's CP convolution see the pinned kernels too.
    """
    if kernels_module is None:
        from fla.modules.conv.triton import kernels as kernels_module
    for name, (block_d, num_warps) in PINNED.items():
        pin_autotuner(_autotuner(getattr(kernels_module, name)), block_d, num_warps)


def pin_fla_norm_autotune(modules=None) -> None:
    """Pin fla's gated RMSNorm and L2-norm kernels (``PINNED_NORM``) in place.

    ``modules`` maps a module name of ``PINNED_NORM`` to a stand-in module (tests); by default the fla
    modules are imported. In-place, so every caller that imported the kernels by name sees the pin.
    """
    for (mod_name, name), (block_t, num_warps) in PINNED_NORM.items():
        mod = (modules or {}).get(mod_name) or importlib.import_module(mod_name)
        _pin(_autotuner(getattr(mod, name)), {"BT": block_t}, num_warps, PINNED_NB)


def pin_fla_autotune() -> None:
    """Everything ``--qwen38-fla-conv-pinned`` pins: the causal_conv1d and the norm kernels."""
    pin_fla_conv_autotune()
    pin_fla_norm_autotune()
