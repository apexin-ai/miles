"""CPU contracts for the Qwen4Exp one-step draft."""

from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize("detach", [False, True])
def test_native_mtp_block_controls_trunk_stream_and_shared_embedding_gradients(detach):
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock

    class SharedEmbedding(nn.Embedding):
        add_position_embedding = False

        def forward(self, *, input_ids, position_ids):
            return super().forward(input_ids).transpose(0, 1)

    class TinyLayer(Qwen38MTPLayer):
        # Preserve MCore forward, roll, detach and aux-depth selection. Substitute
        # only the distributed transformer body to keep the gradient graph on CPU.
        def _proj_and_transformer_layer(self, hidden_states, decoder_input, **kwargs):
            return self.scale * (hidden_states + decoder_input.repeat(1, 1, 4))

    cfg = SimpleNamespace(
        pipeline_model_parallel_size=1, mtp_num_layers=1, mtp_detach_heads=detach, recompute_granularity=None
    )
    layer = TinyLayer.__new__(TinyLayer)
    nn.Module.__init__(layer)
    layer.config = cfg
    layer.cp_group = None
    layer.scale = nn.Parameter(torch.ones(()))
    layer.hc_head_contraction = nn.Linear(8, 2, bias=False)
    layer.hc_head_contraction.weight.data.fill_(0.25)

    block = MultiTokenPredictionBlock.__new__(MultiTokenPredictionBlock)
    nn.Module.__init__(block)
    block.config = cfg
    block.vp_stage = None
    block.mtp_use_repeated_layer = False
    block.layers = nn.ModuleList([layer])
    embedding = SharedEmbedding(7, 2)
    embedding.weight.data.fill_(0.125)
    trunk_hc = torch.ones(4, 1, 8, requires_grad=True)
    main = torch.ones(4, 1, 2, requires_grad=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    result = block(
        input_ids=tokens,
        position_ids=None,
        hidden_states=main,
        mhc_multistream=trunk_hc,
        attention_mask=None,
        embedding=embedding,
    )
    # Restrict the objective to supervised draft rows. No main LM loss contributes.
    result[4:7].sum().backward()
    assert layer.scale.grad.item() > 0
    assert layer.hc_head_contraction.weight.grad.abs().sum() > 0
    assert (trunk_hc.grad is None) == detach
    assert (embedding.weight.grad is None) == detach
    if not detach:
        assert trunk_hc.grad[:3].abs().min() > 0
        assert embedding.weight.grad[2:5].abs().min() > 0
        assert not embedding.weight.grad[[0, 1, 5, 6]].any()
