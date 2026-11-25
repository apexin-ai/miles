"""``--qwen38-hc-fused``: the flag reaches hc_mix_inject through the TransformerConfig, default off.

With the flag off the HC modules call ``hc_mix_inject`` exactly as before (no ``fused`` keyword), so the default
path and every test double of hc_mix_inject stay unchanged. The fused kernels' numerics: tests/fast/kernels/
test_hc_fused.py (CPU, Triton interpreter) and tests/fast-gpu/kernels/hyper_connection/test_hc_fused_gpu.py (GPU).
"""

from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["miles-plugin"])

N, C, R = 4, 32, 8
EPS = 1e-6


def _hc_config(**extra):
    return SimpleNamespace(
        num_residual_streams=N,
        hidden_size=C,
        layernorm_epsilon=EPS,
        qwen3_8_next_hc_lowrank=R,
        params_dtype=torch.float32,
        sequence_parallel=False,
        **extra,
    )


@pytest.mark.parametrize("fused", [None, False, True])
def test_hc_modules_pass_fused_only_when_on(monkeypatch, fused):
    from miles_plugins.models.qwen3_8_next import hyper_connection

    extra = {} if fused is None else {"qwen3_8_next_hc_fused": fused}
    config = _hc_config(qwen3_8_next_hc_matmul_precision="tf32", **extra)
    calls = []

    def spy(x, weight, w_down, w_up, w_inject, n, eps, **kwargs):
        calls.append(kwargs)
        return x[..., :C], None if w_inject is None else x[..., :n]

    monkeypatch.setattr(hyper_connection, "hc_mix_inject", spy)
    x = torch.randn(3, 1, N * C)
    hyper_connection.Qwen38NextHyperConnection(config, layer_number=1)(x)
    hyper_connection.Qwen38NextHCHeadContraction(config)(x)
    want = {"matmul_precision": "tf32", **({"fused": True} if fused else {})}
    assert calls == [want, want]


@pytest.mark.parametrize("value", [True, False])
def test_spec_sets_hc_fused(monkeypatch, value):
    from miles_plugins.models.qwen3_8_next import qwen3_8_next

    config = SimpleNamespace(virtual_pipeline_model_parallel_size=2)
    args = SimpleNamespace(
        qwen38_hc_fused=value, hf_checkpoint="/x", backend=SimpleNamespace(position_embedding_type="rope")
    )
    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: None)
    monkeypatch.setattr(qwen3_8_next, "_get_text_config", lambda hf: None)
    monkeypatch.setattr(qwen3_8_next, "_apply_qwen3_8_next_config", lambda c, t: None)
    # the knobs are set before the VPP guard; the rest of the spec needs a checkpoint
    with pytest.raises(NotImplementedError, match="interleaved pipeline"):
        qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)
    assert config.qwen3_8_next_hc_fused is value


def test_spec_hc_fused_absent_is_off(monkeypatch):
    from miles_plugins.models.qwen3_8_next import qwen3_8_next

    config = SimpleNamespace(virtual_pipeline_model_parallel_size=2)
    args = SimpleNamespace(hf_checkpoint="/x", backend=SimpleNamespace(position_embedding_type="rope"))
    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: None)
    monkeypatch.setattr(qwen3_8_next, "_get_text_config", lambda hf: None)
    monkeypatch.setattr(qwen3_8_next, "_apply_qwen3_8_next_config", lambda c, t: None)
    with pytest.raises(NotImplementedError, match="interleaved pipeline"):
        qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)
    assert config.qwen3_8_next_hc_fused is False


def test_cli_default_off():
    from miles.utils.args.configs.custom_megatron_plugins import Qwen38VisionMegatronPluginsConfig

    field = Qwen38VisionMegatronPluginsConfig.model_fields["qwen38_hc_fused"]
    assert field.default is False
    assert field.annotation is bool


def test_hc_mix_inject_default_is_not_fused(monkeypatch):
    from miles.kernels.hyper_connection import hc

    used = []

    class _Spy:
        def __init__(self, name):
            self.name = name

        def apply(self, x2d, weight, w_down, w_up, w_inject, n, eps, precision):
            used.append(self.name)
            return x2d[:, : x2d.shape[1] // n], x2d.new_empty(0)

    monkeypatch.setattr(hc, "_HCMixInject", _Spy("default"))
    monkeypatch.setattr(hc, "_HCMixInjectFused", _Spy("fused"))
    x = torch.randn(5, N * C)
    w = torch.zeros(N * C)
    hc.hc_mix_inject(x, w, None, None, None, N, EPS)
    hc.hc_mix_inject(x, w, None, None, None, N, EPS, matmul_precision="tf32", fused=False)
    hc.hc_mix_inject(x, w, None, None, None, N, EPS, matmul_precision="tf32", fused=True)
    assert used == ["default", "default", "fused"]
