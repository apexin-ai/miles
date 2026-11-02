"""One fixed launch config for fla's causal_conv1d Triton kernels (``--qwen38-fla-conv-pinned``).

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
"""

from __future__ import annotations

# kernel name -> (BD, num_warps)
PINNED = {
    "causal_conv1d_fwd_kernel": (64, 4),
    "causal_conv1d_bwd_kernel": (32, 4),
}
PINNED_NB = 1
_MARK = "_qwen38_conv_pinned"


def _autotuner(kernel):
    """The autotuner under fla's ``triton.heuristics`` wrapper (the object holding ``configs``)."""
    obj = kernel
    for _ in range(4):
        if hasattr(obj, "configs") and hasattr(obj, "run"):
            return obj
        obj = getattr(obj, "fn", None)
    raise RuntimeError(f"no autotuner under {kernel!r}: fla's causal_conv1d kernels changed shape")


def pin_autotuner(tuner, block_d: int, num_warps: int, nb: int = PINNED_NB) -> None:
    """Keep only the config ``(BD=block_d, num_warps)`` and pass ``NB=nb``; idempotent."""
    if getattr(tuner, _MARK, False):
        return
    match = [c for c in tuner.configs if c.kwargs.get("BD") == block_d and c.num_warps == num_warps]
    if len(match) != 1:
        raise RuntimeError(
            f"expected one config BD={block_d} num_warps={num_warps} among "
            f"{[(c.kwargs, c.num_warps) for c in tuner.configs]}"
        )
    tuner.configs = match
    run = tuner.run

    def pinned_run(*args, **kwargs):
        if "NB" in kwargs:
            kwargs["NB"] = nb
        return run(*args, **kwargs)

    tuner.run = pinned_run
    setattr(tuner, _MARK, True)


def pin_fla_conv_autotune(kernels_module=None) -> None:
    """Pin both causal_conv1d kernels of ``fla.modules.conv.triton.kernels`` (or ``kernels_module``).

    Mutates the kernel objects in place, so ``fla.modules.conv.triton.ops`` (which imported them by
    name) and fla's CP convolution see the pinned kernels too.
    """
    if kernels_module is None:
        from fla.modules.conv.triton import kernels as kernels_module
    for name, (block_d, num_warps) in PINNED.items():
        pin_autotuner(_autotuner(getattr(kernels_module, name)), block_d, num_warps)
