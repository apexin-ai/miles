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


class _NormConfig:
    def __init__(self, bt, nw):
        self.kwargs, self.num_warps = {"BT": bt}, nw


def _norm_tuner(bts, warps):
    t = _Tuner()
    t.configs = [_NormConfig(bt, nw) for bt in bts for nw in warps]
    t.keys = ["D", "NB"]
    return t


def _norm_modules():
    lng = types.SimpleNamespace(
        layer_norm_gated_fwd_kernel=_Heuristics(_norm_tuner((16, 32, 64), (4, 8, 16))),
        layer_norm_gated_bwd_kernel=_Heuristics(_norm_tuner((16, 32, 64), (4, 8, 16))),
    )
    l2 = types.SimpleNamespace(
        l2norm_fwd_kernel=_norm_tuner((8, 16, 32, 64, 128), (1, 2, 4, 8, 16)),
        l2norm_bwd_kernel=_norm_tuner((8, 16, 32, 64, 128), (1, 2, 4, 8, 16)),
    )
    return {"fla.modules.fused_norm_gate": lng, "fla.modules.l2norm": l2}


def test_norm_pin_keeps_one_config_and_constant_nb():
    mods = _norm_modules()
    T.pin_fla_norm_autotune(mods)
    assert len(T.PINNED_NORM) == 4
    for (mod, name), (bt, nw) in T.PINNED_NORM.items():
        k = getattr(mods[mod], name)
        tuner = T._autotuner(k)
        assert [(c.kwargs["BT"], c.num_warps) for c in tuner.configs] == [(bt, nw)]
        # every local row count (every NB = cdiv(rows, 65536)) reaches the kernel as the same constexpr
        for nb in (1, 3, 8, 22, 44):
            assert k.run(x=1, NB=nb, D=128) == "ran"
        assert [c["NB"] for c in tuner.calls] == [T.PINNED_NB] * 5
    # idempotent
    T.pin_fla_norm_autotune(mods)
    assert len(T._autotuner(mods["fla.modules.l2norm"].l2norm_fwd_kernel).configs) == 1


def test_norm_pin_refuses_when_nb_left_the_key():
    mods = _norm_modules()
    mods["fla.modules.l2norm"].l2norm_bwd_kernel.keys = ["D"]
    with pytest.raises(RuntimeError, match="NB is no longer"):
        T.pin_fla_norm_autotune(mods)


def test_flag_pins_conv_and_norm(monkeypatch):
    seen = []
    monkeypatch.setattr(T, "pin_fla_conv_autotune", lambda: seen.append("conv"))
    monkeypatch.setattr(T, "pin_fla_norm_autotune", lambda: seen.append("norm"))
    T.pin_fla_autotune()
    assert seen == ["conv", "norm"]


def test_real_fla_norm_kernels_have_the_pinned_configs():
    import importlib

    pytest.importorskip("fla.modules.l2norm")
    for (mod, name), (bt, nw) in T.PINNED_NORM.items():
        tuner = T._autotuner(getattr(importlib.import_module(mod), name))
        assert "NB" in tuner.keys
        assert sum(c.kwargs.get("BT") == bt and c.num_warps == nw for c in tuner.configs) == 1
