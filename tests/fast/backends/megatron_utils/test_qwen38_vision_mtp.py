"""Vision input and loss contracts for Qwen3.8's one-depth MTP head."""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.qwen3_8_next.vision_mtp import (
    MTPImageEmbedding,
    mask_image_targets,
    validate_vision_mtp,
)


def _args(*, spec_name="get_qwen3_8_next_spec", pp=1, data="mixed"):
    return SimpleNamespace(
        backend=SimpleNamespace(spec=["miles_plugins.models.qwen3_8_next.qwen3_8_next", spec_name]),
        pipeline_model_parallel_size=pp,
        qwen38_vision_data=data,
    )


def test_vision_mtp_only_uses_the_native_spec():
    validate_vision_mtp(_args())
    with pytest.raises(ValueError, match="get_qwen3_8_next_spec"):
        validate_vision_mtp(_args(spec_name="get_local_spec"))


def test_image_mtp_needs_the_tower_output_at_the_mtp_stage():
    validate_vision_mtp(_args(pp=2, data="text"))
    with pytest.raises(ValueError, match="pipeline"):
        validate_vision_mtp(_args(pp=2))


class _Words(torch.nn.Embedding):
    add_position_embedding = False

    def forward(self, input_ids, position_ids):
        return super().forward(input_ids).transpose(0, 1)


@pytest.mark.parametrize("detach", [False, True])
def test_megatron_rotates_visual_inputs_per_row_and_honors_detach(detach):
    pytest.importorskip("megatron.core.transformer.multi_token_prediction")
    from megatron.core.packed_seq_params import PackedSeqParams

    from miles_plugins.models.qwen3_8_next.mtp import Qwen38MTPLayer

    bounds = torch.tensor([0, 5, 8, 10], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=bounds, cu_seqlens_kv=bounds)
    ids = torch.tensor([[1, 8, 8, 2, 3, 4, 5, 6, 0, 0]])
    positions = torch.arange(10)[None, None, :].expand(3, 1, -1)
    words = _Words(9, 4)
    visual = torch.randn(2, 4, requires_grad=True)
    packed.qwen38_mtp_embedding = MTPImageEmbedding(
        words, torch.tensor([1, 2]), visual, tp_rank=0, sequence_parallel=False
    )
    # The CPU check exercises the production layer's embedding path independently of its GPU attention backend.
    layer = Qwen38MTPLayer.__new__(Qwen38MTPLayer)
    torch.nn.Module.__init__(layer)
    layer.cp_group = None
    layer.config = SimpleNamespace(mtp_detach_heads=detach)
    rolled, rope, _, result, _ = layer._get_embeddings(
        ids, positions, words, torch.randn(10, 1, 4, requires_grad=True), packed
    )
    expected_ids = torch.tensor([[8, 8, 2, 3, 0, 5, 6, 0, 0, 0]])
    torch.testing.assert_close(rolled, expected_ids)
    torch.testing.assert_close(rope, positions)
    expected = words(expected_ids, positions)
    expected[0:2, 0] = visual
    torch.testing.assert_close(result, expected)
    assert result.requires_grad is not detach
    if not detach:
        result.sum().backward()
        torch.testing.assert_close(visual.grad, torch.ones_like(visual))
        torch.testing.assert_close(words.weight.grad[8], torch.zeros(4))


def test_shift_can_move_a_visual_token_to_preceding_parallel_slice():
    # Rank-local slice of the rolled sequence. Global image tokens 4,5 now belong to positions 3,4.
    class LocalWords(_Words):
        def forward(self, input_ids, position_ids):
            return super().forward(input_ids, position_ids)[:2]

    words = LocalWords(9, 3)
    visual = torch.randn(2, 3)
    ids = torch.tensor([[0, 8, 8, 0]])
    options = dict(tp_rank=1, sequence_parallel=True, window_start=0)
    left = MTPImageEmbedding(words, torch.tensor([4, 5]), visual, **options)(input_ids=ids, position_ids=None)
    torch.testing.assert_close(left[1, 0], visual[0])
    right = MTPImageEmbedding(words, torch.tensor([4, 5]), visual, tp_rank=0, sequence_parallel=True, window_start=4)
    torch.testing.assert_close(right(input_ids=ids, position_ids=None)[0, 0], visual[1])


@pytest.mark.parametrize("weight", [0.05, 0.2])
@pytest.mark.parametrize("detach", [False, True])
def test_native_mtp_loss_masks_images_prompt_row_tails_and_padding(monkeypatch, weight, detach):
    mtp = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
    from megatron.core.packed_seq_params import PackedSeqParams

    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    # Row 1 supervises image placeholder 8 and tokens 3/4. Only row 2's final token is reachable at depth one.
    # First two row-2 positions deliberately have mask=1 to expose cross-example leakage at row 1's tail.
    ids = torch.tensor([[1, 2, 8, 3, 4, 5, 6, 7, 0, 0]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0]])
    aligned = mask_image_targets(mask, ids, image_token_id=8)
    assert mask[0, 2].item() == 1
    bounds = torch.tensor([0, 5, 8, 10], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=bounds, cu_seqlens_kv=bounds)
    main_and_draft = torch.ones(20, 1, 3, requires_grad=True)
    cfg = SimpleNamespace(
        mtp_num_layers=1,
        mtp_detach_heads=detach,
        mtp_loss_scaling_factor=weight,
        calculate_per_token_loss=False,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="linear",
    )
    output_weight = torch.randn(9, 3, requires_grad=True)
    seen = {}

    def linear_ce(hidden, *, labels, weight, **kwargs):
        # Real tied vocabulary weights participate in the autograd graph.
        logits = torch.nn.functional.linear(hidden[:, 0], weight)
        loss = torch.nn.functional.cross_entropy(logits, labels.flatten(), reduction="none").reshape_as(labels)
        seen.update(labels=labels, per_token=loss)
        loss.retain_grad()
        return loss

    hidden = mtp.process_mtp_loss(
        hidden_states=main_and_draft,
        input_ids=ids,
        labels=None,
        loss_mask=aligned,
        output_layer=linear_ce,
        output_weight=output_weight,
        runtime_gather_output=False,
        is_training=False,
        compute_language_model_loss=None,
        config=cfg,
        cp_group=None,
        tp_group=None,
        packed_seq_params=packed,
    )
    hidden.sum().backward()
    grad = seen["per_token"].grad
    assert seen["labels"][grad > 0].tolist() == [3, 4, 7]
    torch.testing.assert_close(grad[grad > 0], torch.full((3,), weight / 3))
    assert (output_weight.grad is None) == detach


def test_vision_forward_shares_this_packs_merged_embeddings_with_its_draft(monkeypatch):
    from megatron.core import parallel_state
    from tests.fast.backends.megatron_utils.test_qwen38_vision_text_fallback import _CONFIG, _image_batch, _Language

    from miles_plugins.models.qwen3_8_next.vision import wire_visual

    class Capture(_Language):
        def forward(self, **kwargs):
            return kwargs

    model = Capture()
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    wire_visual(
        model, _CONFIG, scope="full", recompute=False, freeze_language=False, data="mixed", enable_mtp_training=True
    )
    ids, inputs = _image_batch()
    packed = SimpleNamespace(cu_seqlens_host=(0, ids.shape[1]), sample_count=1)
    mask = torch.ones_like(ids)
    out = model(input_ids=ids, packed_seq_params=packed, loss_mask=mask, **inputs)
    assert not hasattr(packed, "qwen38_mtp_embedding")
    assert mask.all()
    torch.testing.assert_close(out["loss_mask"], ids.ne(_CONFIG["image_token_id"]).long())
    torch.testing.assert_close(out["position_ids"][:, 0].T, inputs["qwen38_position_ids"])
    rolled = ids.roll(-1, -1)
    rolled[:, -1] = 0
    embedding = out["packed_seq_params"].qwen38_mtp_embedding
    draft = embedding(rolled, position_ids=None)
    torch.testing.assert_close(draft[:-1], out["decoder_input"][1:])
    assert draft.requires_grad
    # The next text-only pack can reuse the caller's metadata safely.
    text_ids = torch.arange(1, ids.numel() + 1)[None]
    text = model(input_ids=text_ids, packed_seq_params=packed, loss_mask=mask)
    assert text["packed_seq_params"] is packed
    assert not hasattr(text["packed_seq_params"], "qwen38_mtp_embedding")
