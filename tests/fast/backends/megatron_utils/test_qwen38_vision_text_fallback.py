"""CPU contracts for running the Qwen3.8 vision model on text, image and mixed data.

A text microbatch skips the vision tower and gives the language model text RoPE positions on every mRoPE axis. With
text-only data the tower must be frozen before the optimizer is built: a tower that is held but never run would
otherwise get zero gradients and AdamW weight decay would shrink it.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from miles_plugins.models.qwen3_8_next import vision, vision_sft
from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions, text_rope_positions

_HIDDEN = 16
_CONFIG = {
    "image_token_id": 9001,
    "vision_start_token_id": 9002,
    "vision_end_token_id": 9003,
    "vision_config": {
        "depth": 2,
        "hidden_size": 16,
        "intermediate_size": 24,
        "out_hidden_size": _HIDDEN,
        "num_heads": 2,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [],
    },
}


class _Embedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(10000, _HIDDEN)

    def forward(self, input_ids, position_ids=None):
        return self.word_embeddings(input_ids).transpose(0, 1)  # Megatron layout [s, b, h]


class _Language(torch.nn.Module):
    """Stands in for the Megatron GPT model: records what the wrapper passes and returns a loss."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_process = True
        self.config = SimpleNamespace(params_dtype=torch.float32, sequence_parallel=False)
        self.embedding = _Embedding()
        self.head = torch.nn.Linear(_HIDDEN, _HIDDEN)
        self.calls = []

    def forward(self, input_ids=None, position_ids=None, decoder_input=None, **kwargs):
        self.calls.append({"position_ids": position_ids, "decoder_input": decoder_input})
        hidden = decoder_input if decoder_input is not None else self.embedding(input_ids)
        return self.head(hidden).square().mean()


def _model(*, scope, data, seed=0):
    torch.manual_seed(seed)
    model = _Language()
    vision.wire_visual(model, _CONFIG, scope=scope, recompute=False, freeze_language=False, data=data)
    return model


def _text_ids(length=12):
    return torch.arange(100, 100 + length)[None]


def _image_batch():
    """One image of grid [1, 4, 4]: 16 patches, 4 merged image tokens between the vision start / end tokens."""
    ids = torch.tensor([[100, 101, 9002, 9001, 9001, 9001, 9001, 9003, 102, 103]])
    grid = torch.tensor([[1, 4, 4]])
    positions = image_rope_positions(
        ids[0], grid, image_token_id=9001, vision_start_token_id=9002, vision_end_token_id=9003, merge_size=2
    )
    return ids, {"pixel_values": torch.randn(16, 24), "image_grid_thw": grid, "qwen38_position_ids": positions}


def _forbid_vision_forward(model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the vision tower must not run on a text microbatch")

    monkeypatch.setattr(model.vision_model, "forward", forbidden)


@pytest.fixture
def tp_rank_zero(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)


@pytest.mark.parametrize("data", ["text", "mixed"])
def test_a_text_microbatch_skips_the_vision_tower_and_gets_text_rope_positions(data, monkeypatch):
    model = _model(scope="none", data=data)
    _forbid_vision_forward(model, monkeypatch)
    ids = _text_ids()
    loss = model(input_ids=ids)
    assert torch.isfinite(loss)
    call = model.calls[-1]
    assert call["decoder_input"] is None
    assert call["position_ids"].shape == (3, 1, ids.shape[1])
    for axis in range(3):
        assert torch.equal(call["position_ids"][axis, 0], torch.arange(ids.shape[1]))


def test_image_mode_still_requires_images():
    model = _model(scope="full", data="image")
    with pytest.raises(ValueError, match="requires genuine image patches"):
        model(input_ids=_text_ids())


def test_text_mode_refuses_a_trainable_tower_and_image_inputs(tp_rank_zero):
    for scope in ("full", "projector"):
        with pytest.raises(ValueError, match="needs a frozen vision tower"):
            _model(scope=scope, data="text")
    model = _model(scope="none", data="text")
    ids, inputs = _image_batch()
    with pytest.raises(ValueError, match="--qwen38-vision-data text"):
        model(input_ids=ids, **inputs)


def test_partial_image_inputs_are_rejected():
    model = _model(scope="full", data="mixed")
    ids, inputs = _image_batch()
    del inputs["image_grid_thw"]
    with pytest.raises(ValueError, match="requires genuine image patches"):
        model(input_ids=ids, **inputs)


def test_unknown_data_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown Qwen vision data mode"):
        _model(scope="none", data="video")


def test_mixed_microbatches_train_the_tower_on_images_only(tp_rank_zero, monkeypatch):
    model = _model(scope="full", data="mixed")
    ids, inputs = _image_batch()
    model(input_ids=ids, **inputs).backward()
    image_call = model.calls[-1]
    assert image_call["decoder_input"] is not None
    assert torch.equal(image_call["position_ids"][:, 0].T, inputs["qwen38_position_ids"])
    grads = {name: p.grad.clone() for name, p in model.vision_model.named_parameters()}
    assert any(g.abs().sum() > 0 for g in grads.values())

    _forbid_vision_forward(model, monkeypatch)
    model(input_ids=_text_ids()).backward()
    for name, p in model.vision_model.named_parameters():
        assert torch.equal(p.grad, grads[name]), name  # the text microbatch adds nothing to the tower
    assert model.calls[-1]["decoder_input"] is None


def _grad_reports(model):
    """Counts gradient accumulations per vision parameter, as Megatron DDP's overlap hooks see them."""
    reports = {}
    for name, p in model.vision_model.named_parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(lambda p, name=name: reports.update({name: reports.get(name, 0) + 1}))
    return reports


@pytest.mark.parametrize("scope", ["full", "projector"])
def test_a_text_microbatch_reports_a_zero_grad_for_every_trainable_tower_parameter(scope, tp_rank_zero, monkeypatch):
    """--overlap-grad-reduce reduces a bucket only once all its parameters reported a grad in the step's last
    microbatch, and it expects the same reports every step. A text microbatch must report like an image one."""
    model = _model(scope=scope, data="mixed")
    trainable = {name for name, p in model.vision_model.named_parameters() if p.requires_grad}
    assert trainable and (scope == "full") == (trainable == {n for n, _ in model.vision_model.named_parameters()})
    reports = _grad_reports(model)
    ids, inputs = _image_batch()
    model(input_ids=ids, **inputs).backward()
    image_reports = dict(reports)
    assert set(image_reports) == trainable
    model.zero_grad(set_to_none=True)
    reports.clear()

    _forbid_vision_forward(model, monkeypatch)
    loss = model(input_ids=_text_ids())
    loss.backward()
    assert reports == image_reports
    for name, p in model.vision_model.named_parameters():
        if name in trainable:
            assert torch.equal(p.grad, torch.zeros_like(p)), name
        else:
            assert p.grad is None, name

    # The output and the language model's gradients are those of a run whose tower is frozen.
    reference = _model(scope="none", data="mixed")
    reference_loss = reference(input_ids=_text_ids())
    reference_loss.backward()
    assert torch.equal(loss, reference_loss)
    language = {n: p.grad for n, p in model.named_parameters() if not n.startswith("vision_model.")}
    for name, p in reference.named_parameters():
        if not name.startswith("vision_model."):
            assert torch.equal(language[name], p.grad), name


def test_a_text_microbatch_without_grad_or_trainable_tower_adds_no_backward_node(monkeypatch):
    model = _model(scope="full", data="mixed")
    _forbid_vision_forward(model, monkeypatch)
    with torch.no_grad():
        assert model(input_ids=_text_ids()).grad_fn is None
    frozen = _model(scope="none", data="mixed")
    assert "ZeroGradsFor" not in type(frozen(input_ids=_text_ids()).grad_fn).__name__


@pytest.mark.parametrize("interleaved_mrope", [False, True])
def test_text_positions_through_mrope_equal_text_rope(interleaved_mrope, monkeypatch):
    """The language model sees the same rotary embedding for a text microbatch as the text-only provider gives."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        MultimodalRotaryEmbedding,
        RotaryEmbedding,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    length, kv_channels = 37, 32
    text = RotaryEmbedding(kv_channels, 1.0, use_cpu_initialization=True, cp_group=None)(length)
    mrope = MultimodalRotaryEmbedding(kv_channels, 1.0, cp_group=None, interleaved_mrope=interleaved_mrope)
    positions = text_rope_positions(length).T.unsqueeze(1)
    multimodal = mrope(positions, mrope_section=[6, 5, 5])
    torch.testing.assert_close(multimodal, text, rtol=0, atol=0)


@pytest.fixture
def gloo_world_of_one(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _megatron_param_groups(model):
    from megatron.core.optimizer import OptimizerConfig, _get_param_groups

    return _get_param_groups([model], OptimizerConfig(lr=0.01, min_lr=0.0, weight_decay=0.1), None)


def _train_text_steps(model, steps=3):
    groups = _megatron_param_groups(model)
    optimizer = torch.optim.AdamW(
        [{"params": g["params"], "weight_decay": 0.1 * g["wd_mult"]} for g in groups if g["params"]], lr=0.01
    )
    for step in range(steps):
        optimizer.zero_grad()
        model(input_ids=_text_ids(8 + step)).backward()
        optimizer.step()
    return groups


def test_a_frozen_tower_is_not_in_the_optimizer_and_stays_bitwise_unchanged(gloo_world_of_one, monkeypatch):
    model = _model(scope="none", data="text")
    _forbid_vision_forward(model, monkeypatch)
    tower = {name: p.detach().clone() for name, p in model.vision_model.named_parameters()}
    language = {name: p.detach().clone() for name, p in model.head.named_parameters()}
    assert not any(p.requires_grad for p in model.vision_model.parameters())

    groups = _train_text_steps(model)

    in_optimizer = {id(p) for g in groups for p in g["params"]}
    assert not in_optimizer & {id(p) for p in model.vision_model.parameters()}
    assert {id(p) for p in model.head.parameters()} <= in_optimizer
    for name, p in model.vision_model.named_parameters():
        assert torch.equal(p, tower[name]), name
    assert any(not torch.equal(p, language[name]) for name, p in model.head.named_parameters())


def test_positive_control_a_trainable_tower_on_text_steps_is_shrunk_by_weight_decay(gloo_world_of_one, monkeypatch):
    """Why text data needs the freeze: in the optimizer but never run, the tower still moves."""
    model = _model(scope="full", data="mixed")
    _forbid_vision_forward(model, monkeypatch)
    tower = {name: p.detach().clone() for name, p in model.vision_model.named_parameters()}
    for p in model.vision_model.parameters():
        p.grad = torch.zeros_like(p)  # what Megatron's zeroed main_grad buffer hands the optimizer
    groups = _megatron_param_groups(model)
    assert {id(p) for p in model.vision_model.parameters()} <= {id(p) for g in groups for p in g["params"]}
    optimizer = torch.optim.AdamW([{"params": list(model.vision_model.parameters()), "weight_decay": 0.1}], lr=0.01)
    optimizer.step()
    assert any(not torch.equal(p, tower[name]) for name, p in model.vision_model.named_parameters())


class _Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join(f"[{m['role']}]{m['content']}[end]" for m in messages)
        return text + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)


class _Processor:
    tokenizer = _Tokenizer()

    def __call__(self, *args, **kwargs):
        raise AssertionError("a text sample must not go through the image processor")


_TEXT_MESSAGES = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]


def test_a_text_sample_encodes_without_image_inputs_when_text_is_allowed():
    tokens, mask, inputs = vision_sft.encode_sft_sample(_Processor(), _CONFIG, _TEXT_MESSAGES, [], allow_text=True)
    assert inputs is None
    assert len(tokens) == len(mask)
    assert "".join(chr(t) for t, m in zip(tokens, mask, strict=True) if m) == "A[end]"
    with pytest.raises(ValueError, match="needs real images"):
        vision_sft.encode_sft_sample(_Processor(), _CONFIG, _TEXT_MESSAGES, [])


@pytest.mark.parametrize(
    "data, images, expect",
    [
        ("text", [], "text"),
        ("mixed", [], "text"),
        ("text", ["img"], "reject"),
        ("image", [], "reject"),
    ],
)
def test_generate_rollout_follows_the_data_mode(data, images, expect, monkeypatch):
    monkeypatch.setattr(vision_sft, "_processor", lambda checkpoint: (_Processor(), _CONFIG))
    sample = SimpleNamespace(index=0, prompt=_TEXT_MESSAGES, multimodal_inputs={"images": images}, metadata={})
    buffer = SimpleNamespace(get_samples=lambda n: [(sample,)])
    args = SimpleNamespace(
        rollout_global_dataset=True,
        hf_checkpoint="/ckpt",
        qwen38_vision_data=data,
        rollout_batch_size=1,
        apply_chat_template_kwargs=None,
        seq_length=4096,
        qwen38_sft_final_answer_format="dataset",
    )
    if expect == "reject":
        with pytest.raises(ValueError, match="images but --qwen38-vision-data is text|needs real images"):
            vision_sft.generate_rollout(args, 0, buffer)
    else:
        [[out]] = vision_sft.generate_rollout(args, 0, buffer)
        assert out.multimodal_train_inputs is None
        assert out.response_length == sum(out.loss_mask) == len("A[end]")
