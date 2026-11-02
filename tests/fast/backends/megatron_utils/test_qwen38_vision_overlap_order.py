"""CPU contracts: with --overlap-grad-reduce, a text and an image microbatch complete Megatron DDP's buckets in one order.

Megatron DDP issues a bucket's reduce-scatter as soon as every parameter in it reported a gradient in the step's last
microbatch (register_grad_ready, from a hook on each parameter's gradient accumulator). Every DP rank must issue them in
one order on the DP communicator, or NCCL deadlocks. In mixed data one DP rank's last microbatch can be text and
another's an image one (F8, 2026-11-02: step 2 hung in the last microbatch's backward on 4-layer TP4 / EP8, DP 2).

The tests hook the gradient accumulators like Megatron DDP and lay out the buckets with Megatron's own layout function
(DistributedOptimizer._compute_per_buffer_param_layout) over the parameters in DDP's order, the real HF vision tower
included.
"""

from types import SimpleNamespace

import pytest
import torch

from miles_plugins.models.qwen3_8_next import vision, vision_scope_switch
from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions

_HIDDEN = 16
_CONFIG = {
    "image_token_id": 9001,
    "vision_start_token_id": 9002,
    "vision_end_token_id": 9003,
    "vision_config": {
        "depth": 3,
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
_DDP = SimpleNamespace(pad_buckets_for_high_nccl_busbw=False)


class _Embedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(10000, _HIDDEN)

    def forward(self, input_ids, position_ids=None):
        return self.word_embeddings(input_ids).transpose(0, 1)  # Megatron layout [s, b, h]


class _Language(torch.nn.Module):
    """Stands in for the Megatron GPT model: embedding, layers, final norm, output layer, in Megatron's module order."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_process = True
        self.config = SimpleNamespace(params_dtype=torch.float32, sequence_parallel=False)
        self.embedding = _Embedding()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(_HIDDEN, _HIDDEN) for _ in range(3))
        self.final_layernorm = torch.nn.LayerNorm(_HIDDEN)
        self.output_layer = torch.nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, input_ids=None, position_ids=None, decoder_input=None, **kwargs):
        hidden = decoder_input if decoder_input is not None else self.embedding(input_ids)
        for layer in self.layers:
            hidden = hidden + torch.tanh(layer(hidden))
        return self.output_layer(self.final_layernorm(hidden)).square().mean()


def _model(scope="full", recompute=False):
    torch.manual_seed(0)
    model = _Language()
    vision.wire_visual(model, _CONFIG, scope=scope, recompute=recompute, freeze_language=False, data="mixed")
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


@pytest.fixture(autouse=True)
def tp_rank_zero(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)


def _ddp_params(model):
    """What Megatron DDP hands its buffers: trainable params in named_parameters order."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _hook_like_ddp(model):
    """Records each parameter's gradient report in order, from a hook on its gradient accumulator (as Megatron DDP's
    _make_backward_post_hook is registered)."""
    reports, keep = [], []
    for name, param in _ddp_params(model):
        grad_acc = param.expand_as(param).grad_fn.next_functions[0][0]
        grad_acc.register_hook(lambda *unused, name=name: reports.append(name))
        keep.append(grad_acc)
    model._test_grad_accs = keep
    return reports


def _bucket_dispatch_order(model, reports, bucket_size, dp=2):
    """The buckets in the order register_grad_ready issues their reduce-scatters, given the reports of one microbatch."""
    named = _ddp_params(model)
    layout = vision_scope_switch.compute_layout([p for _, p in named], bucket_size, dp, _DDP)
    bucket_of = {n: layout.param_index_map[p][2] for n, p in named}
    missing = {b: {n for n in bucket_of if bucket_of[n] == b} for b in set(bucket_of.values())}
    order = []
    for name in reports:
        bucket = missing[bucket_of[name]]
        bucket.discard(name)
        if not bucket:
            order.append(bucket_of[name])
    assert sorted(order) == sorted(missing), "every bucket must be issued exactly once"
    return order


def _bucket_sizes(model):
    """Bucket sizes from the largest parameter of a tower block (a real bucket, >= 20M elements, holds more than a
    block's largest parameter, 1152 x 4304) up to one bucket for all but the word embeddings, and one bucket."""
    block = max(p.numel() for n, p in model.vision_model.named_parameters() if n.startswith("blocks."))
    rest = sum(p.numel() for n, p in _ddp_params(model) if not n.startswith("embedding."))
    return [None] + list(range(block, rest + block, block // 8))


def _image_and_text_reports(model, text_forward=None):
    reports = _hook_like_ddp(model)
    ids, inputs = _image_batch()
    model(input_ids=ids, **inputs).backward()
    image = list(reports)
    reports.clear()
    (text_forward or model)(input_ids=_text_ids()).backward()
    return image, list(reports)


def _old_text_forward(model):
    """75c9bbf0's text microbatch: the zero gradients hooked on the language model's output (the deadlock)."""

    def forward(input_ids):
        positions = torch.arange(input_ids.shape[1]).expand(3, 1, -1)
        output = _Language.forward(model, input_ids=input_ids, position_ids=positions)
        trainable = [p for p in model.vision_model.parameters() if p.requires_grad]
        return vision._ZeroGradsFor.apply(output, *trainable)

    return forward


@pytest.mark.parametrize("scope, recompute", [("full", False), ("full", True), ("projector", False)])
def test_text_and_image_microbatches_issue_the_buckets_in_one_order(scope, recompute):
    model = _model(scope, recompute)
    image, text = _image_and_text_reports(model)
    assert sorted(image) == sorted(text) and len(set(text)) == len(text)
    for bucket_size in _bucket_sizes(model):
        assert _bucket_dispatch_order(model, text, bucket_size) == _bucket_dispatch_order(
            model, image, bucket_size
        ), bucket_size


@pytest.mark.parametrize("scope", ["full", "projector"])
def test_a_text_microbatch_reports_the_tower_after_the_language_model_in_reverse_registration_order(scope):
    model = _model(scope)
    image, text = _image_and_text_reports(model)
    tower = [f"vision_model.{n}" for n, p in model.vision_model.named_parameters() if p.requires_grad]
    for reports in (image, text):
        positions = [reports.index(n) for n in tower]
        # after every language-model layer, before the word embeddings (the embedding is merged with the tower)
        assert min(positions) > max(reports.index(n) for n in reports if n.startswith("layers."))
        assert max(positions) < reports.index("embedding.word_embeddings.weight")
    assert [n for n in text if n in tower] == tower[::-1]
    # The image backward differs only inside a block: norm2 (registered before attn) reports before attn.
    image_tower = [n for n in image if n in tower]
    if scope == "projector":
        assert image_tower == tower[::-1]
    else:
        assert image_tower != tower[::-1]
        assert {n for i, (a, b) in enumerate(zip(image_tower, tower[::-1])) for n in (a, b) if a != b} <= {
            n for n in tower if ".norm2." in n or ".attn." in n
        }


def test_positive_control_the_old_text_hook_issues_the_buckets_in_another_order():
    model = _model("full")
    image, old_text = _image_and_text_reports(model, _old_text_forward(model))
    assert sorted(image) == sorted(old_text)
    differs = [
        bucket_size
        for bucket_size in _bucket_sizes(model)
        if _bucket_dispatch_order(model, old_text, bucket_size) != _bucket_dispatch_order(model, image, bucket_size)
    ]
    assert differs and differs[0] is None and differs[1] == _bucket_sizes(model)[1]


def test_the_text_microbatch_loss_and_language_gradients_equal_a_frozen_tower_run(monkeypatch):
    """With overlap off the fix changes nothing but where the zero gradients come from: bitwise."""
    model = _model("full")

    def forbidden(*args, **kwargs):
        raise AssertionError("the vision tower must not run on a text microbatch")

    monkeypatch.setattr(model.vision_model, "forward", forbidden)
    loss = model(input_ids=_text_ids())
    loss.backward()
    old = _model("full")
    old_loss = _old_text_forward(old)(_text_ids())
    old_loss.backward()
    assert torch.equal(loss, old_loss)
    for (name, p), (old_name, q) in zip(model.named_parameters(), old.named_parameters(), strict=True):
        assert name == old_name and torch.equal(p.grad, q.grad), name
        if name.startswith("vision_model."):
            assert torch.equal(p.grad, torch.zeros_like(p)), name


def test_a_text_microbatch_hands_the_embeddings_as_decoder_input_only_when_it_reports_zero_grads(monkeypatch):
    seen = []
    original = _Language.forward

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("decoder_input"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_Language, "forward", spy)  # before wiring: the wrapper keeps the bound forward
    model = _model("full")
    with torch.no_grad():
        assert model(input_ids=_text_ids()).grad_fn is None
    model(input_ids=_text_ids())
    _model("none")(input_ids=_text_ids())
    assert seen[0] is None and seen[2] is None
    assert torch.equal(seen[1], model.embedding(_text_ids())) and seen[1]._base is None


def test_a_text_microbatch_refuses_a_padding_mask():
    model = _model("full")
    with pytest.raises(ValueError, match="padding_mask"):
        model(input_ids=_text_ids(), padding_mask=torch.zeros(1, 12, dtype=torch.bool))
