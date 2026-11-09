"""``--qwen38-hc-kernel-v2``: the flag reaches hc_mix_inject through the TransformerConfig, default off, and wins over
``--qwen38-hc-fused``. With both flags off the HC modules call ``hc_mix_inject`` exactly as before (no extra keyword).
Kernel numerics: tests/fast/kernels/test_hc_v2.py (CPU) and tests/fast-gpu/kernels/hyper_connection/test_hc_v2_gpu.py.
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


@pytest.mark.parametrize(
    "v2,fused,want_extra",
    [
        (None, None, {}),
        (False, None, {}),
        (False, True, {"fused": True}),
        (True, None, {"v2": True}),
        (True, True, {"v2": True}),
        (True, False, {"v2": True}),
    ],
)
def test_hc_modules_pass_v2_only_when_on(monkeypatch, v2, fused, want_extra):
    from miles_plugins.models.qwen3_8_next import hyper_connection

    extra = {}
    if v2 is not None:
        extra["qwen3_8_next_hc_kernel_v2"] = v2
    if fused is not None:
        extra["qwen3_8_next_hc_fused"] = fused
    config = _hc_config(qwen3_8_next_hc_matmul_precision="tf32", **extra)
    calls = []

    def spy(x, weight, w_down, w_up, w_inject, n, eps, **kwargs):
        calls.append(kwargs)
        return x[..., :C], None if w_inject is None else x[..., :n]

    monkeypatch.setattr(hyper_connection, "hc_mix_inject", spy)
    x = torch.randn(3, 1, N * C)
    hyper_connection.Qwen38NextHyperConnection(config, layer_number=1)(x)
    hyper_connection.Qwen38NextHCHeadContraction(config)(x)
    want = {"matmul_precision": "tf32", **want_extra}
    assert calls == [want, want]


@pytest.mark.parametrize("value", [True, False, None])
def test_spec_sets_hc_kernel_v2(monkeypatch, value):
    from miles_plugins.models.qwen3_8_next import qwen3_8_next

    config = SimpleNamespace(virtual_pipeline_model_parallel_size=2)
    args = SimpleNamespace(hf_checkpoint="/x", backend=SimpleNamespace(position_embedding_type="rope"))
    if value is not None:
        args.qwen38_hc_kernel_v2 = value
    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: None)
    monkeypatch.setattr(qwen3_8_next, "_get_text_config", lambda hf: None)
    monkeypatch.setattr(qwen3_8_next, "_apply_qwen3_8_next_config", lambda c, t: None)
    with pytest.raises(NotImplementedError, match="interleaved pipeline"):
        qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)
    assert config.qwen3_8_next_hc_kernel_v2 is bool(value)
