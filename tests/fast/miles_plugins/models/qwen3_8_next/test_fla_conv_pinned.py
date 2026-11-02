"""--qwen38-fla-conv-pinned: one config and a constant NB for fla's causal_conv1d kernels (CPU)."""

import types

import pytest

from miles_plugins.models.qwen3_8_next.ops import fla_conv_tuning as T


class _Config:
    def __init__(self, bd, nw):
        self.kwargs, self.num_warps = {"BD": bd}, nw


class _Tuner:
    def __init__(self):
        self.configs = [_Config(bd, nw) for bd in (16, 32, 64, 128) for nw in (4, 8, 16, 32)]
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "ran"


class _Heuristics:
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        return self.fn.run(*args, **kwargs)


def _module():
    return types.SimpleNamespace(
        causal_conv1d_fwd_kernel=_Heuristics(_Tuner()), causal_conv1d_bwd_kernel=_Heuristics(_Tuner())
    )


def test_pin_keeps_one_config_and_constant_nb():
    m = _module()
    T.pin_fla_conv_autotune(m)
    for name, (bd, nw) in T.PINNED.items():
        h = getattr(m, name)
        assert [(c.kwargs["BD"], c.num_warps) for c in h.fn.configs] == [(bd, nw)]
        # every NB (every row length) reaches the kernel as the same constexpr
        for nb in (1, 71, 128, 256):
            assert h.run(x=1, NB=nb, D=2560) == "ran"
        assert [c["NB"] for c in h.fn.calls] == [T.PINNED_NB] * 4
        assert all(c["D"] == 2560 for c in h.fn.calls)


def test_pin_is_idempotent():
    m = _module()
    T.pin_fla_conv_autotune(m)
    T.pin_fla_conv_autotune(m)
    h = m.causal_conv1d_fwd_kernel
    h.run(NB=5)
    assert len(h.fn.calls) == 1 and len(h.fn.configs) == 1


def test_missing_config_refused():
    m = _module()
    m.causal_conv1d_bwd_kernel.fn.configs = [_Config(16, 4)]
    with pytest.raises(RuntimeError, match="expected one config"):
        T.pin_fla_conv_autotune(m)


def test_real_fla_kernels_have_the_pinned_configs():
    kernels = pytest.importorskip("fla.modules.conv.triton.kernels")
    for name, (bd, nw) in T.PINNED.items():
        tuner = T._autotuner(getattr(kernels, name))
        assert "NB" in tuner.keys
        assert any(c.kwargs.get("BD") == bd and c.num_warps == nw for c in tuner.configs)
