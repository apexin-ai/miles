"""Packed SFT's per-token denominator must match the pin's MTP auxiliary gradients."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import multi_token_prediction as mtp

from miles_plugins.models.qwen3_8_next.vision_mtp import mask_image_targets


@pytest.mark.parametrize("coefficient", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("all_masked", [False, True], ids=["packed-image", "zero-targets"])
def test_packed_mtp_loss_and_gradients_use_rolled_token_count(monkeypatch, coefficient, all_masked):
    # THD fragments [1,2,3,4] and [5,6,7]. Image placeholder=3. One-step MTP predicts t+2.
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    mask = mask_image_targets(torch.ones_like(tokens), tokens, image_token_id=3)
    mask[:, [0, 4]] = 0  # The first token of each packed sample is prompt context.
    if all_masked:
        mask.zero_()
    bounds = torch.tensor([0, 4, 7], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=bounds, cu_seqlens_kv=bounds, max_seqlen_q=4, max_seqlen_kv=4
    )
    cfg = SimpleNamespace(
        mtp_num_layers=1,
        mtp_detach_heads=False,
        calculate_per_token_loss=True,
        mtp_loss_scaling_factor=coefficient,
        cross_entropy_loss_fusion=False,
    )
    logs = []
    monkeypatch.setattr(mtp.MTPLossLoggingHelper, "should_collect_acceptance", lambda: False)
    monkeypatch.setattr(
        mtp.MTPLossLoggingHelper,
        "save_loss_to_tracker",
        lambda loss_sum, count, *args, **kwargs: logs.append((loss_sum.detach(), count.detach())),
    )
    monkeypatch.setattr(mtp.parallel_state, "get_data_parallel_group", lambda **kw: None)
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    gen = torch.Generator().manual_seed(273)
    weight = torch.randn(8, 3, dtype=torch.float64, generator=gen, requires_grad=True)
    states = torch.randn(14, 1, 3, dtype=torch.float64, generator=gen, requires_grad=True)

    def output_layer(hidden, *, weight, runtime_gather_output):
        return F.linear(hidden, weight), None

    def ce(labels, logits):
        return F.cross_entropy(logits[:, 0], labels[0], reduction="none").unsqueeze(0)

    visible = mtp.process_mtp_loss(
        hidden_states=states,
        labels=None,
        input_ids=tokens,
        loss_mask=mask,
        output_layer=output_layer,
        output_weight=weight,
        runtime_gather_output=False,
        is_training=True,
        compute_language_model_loss=ce,
        config=cfg,
        cp_group=None,
        tp_group=None,
        packed_seq_params=packed,
    )
    # The forward returns only the target model's states. A zero target-model loss isolates the MTP gradient.
    actual = torch.autograd.grad(visible.sum() * 0, (states, weight))
    assert len(logs) == 1
    if all_masked:
        assert logs[0][0].item() == logs[0][1].item() == 0
        for grad in actual:
            torch.testing.assert_close(grad, torch.zeros_like(grad))
        return

    # Main next-token labels surviving the first roll: [2,4,6,7] (4). Draft labels
    # surviving the second roll: [4,7] (2), produced at pooled positions 1 and 4.
    selected = states[7:, 0][[1, 4]]
    per_token = F.cross_entropy(F.linear(selected, weight), torch.tensor([4, 7]), reduction="none")
    torch.testing.assert_close(logs[0][0], per_token.sum())
    assert logs[0][1].item() == 2
    expected = torch.autograd.grad(coefficient * per_token.mean(), (states, weight))
    for raw_grad, after_global_divide in zip(actual, expected, strict=True):
        # finalize_model_grads divides the injected gradient by 4 supervised main tokens on this rank.
        torch.testing.assert_close(raw_grad / 4, after_global_divide)
