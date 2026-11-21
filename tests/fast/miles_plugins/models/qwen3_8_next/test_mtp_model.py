"""CPU contracts for the Qwen4Exp one-step draft."""

from types import SimpleNamespace

import torch
from torch import nn

from miles_plugins.models.qwen3_8_next.mtp import Qwen38MTPHiddenNorm, Qwen38MTPLayer


def test_hidden_norm_uses_full_hc_vector():
    cfg = SimpleNamespace(num_residual_streams=4, sequence_parallel=True, params_dtype=torch.float32)
    norm = Qwen38MTPHiddenNorm(cfg, hidden_size=2, eps=1e-6)
    x = torch.arange(1, 17, dtype=torch.float32).reshape(2, 1, 4, 2).requires_grad_()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.7, 0.5, 8))
    expect = torch.nn.functional.rms_norm(x.flatten(-2), (8,), 1 + norm.weight, 1e-6)
    torch.testing.assert_close(norm(x).flatten(-2), expect)
    assert norm.weight.sequence_parallel
    norm(x).sum().backward()
    assert x.grad is not None and norm.weight.grad is not None


def test_visual_pack_embedding_is_local_to_call(monkeypatch):
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    def capture(self, **kwargs):
        return kwargs["embedding"]

    monkeypatch.setattr(MultiTokenPredictionLayer, "_get_embeddings", capture)
    base, vision = object(), object()
    layer = object.__new__(Qwen38MTPLayer)
    kwargs = dict(input_ids=None, position_ids=None, hidden_states=None, embedding=base)
    assert layer._get_embeddings(**kwargs, packed_seq_params=SimpleNamespace(qwen38_mtp_embedding=vision)) is vision
    assert layer._get_embeddings(**kwargs, packed_seq_params=None) is base


def test_mtp_output_uses_qwen_mixer_without_additional_norm():
    layer = Qwen38MTPLayer.__new__(Qwen38MTPLayer)
    nn.Module.__init__(layer)
    layer.hc_head_contraction = nn.Linear(8, 2, bias=False)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    torch.testing.assert_close(layer._postprocess(x), layer.hc_head_contraction(x))
