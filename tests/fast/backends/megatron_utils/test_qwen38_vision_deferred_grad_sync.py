"""CPU contracts for vision_grad_sync: with --overlap-grad-reduce, the order in which a DP rank issues its bucket groups'
reduce-scatters does not depend on whether its last microbatch held images.

A simulated DistributedDataParallel step runs Megatron's own bucket-group methods (fdf82356
_ParamAndGradBucketGroup.register_grad_ready / finish_grad_sync / reset) on stand-in groups whose start_grad_sync records
the issue order of one communicator. The buckets come from Megatron's layout function
(DistributedOptimizer._compute_per_buffer_param_layout, DP 2) over the parameters in DDP's order, gradients come from
the real HF Qwen3-VL vision tower wired by vision.wire_visual, and the reports from hooks on each parameter's gradient
accumulator, as Megatron DDP registers them.

F8-GUARD (CX4 review, 2026-11-02) is the counterexample: a tiny tower (depth 2, hidden 8), bucket sizes 33 / 64 / 65,
where a block's norm2 and its attention sit in different buckets; the image backward reports norm2 first, the text
zero-grad chain attention first, and the ready-triggered issue orders differ (text 17,0,1,...; image ...4,7,5,6,...).
"""

import types
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from miles_plugins.models.qwen3_8_next import vision, vision_grad_sync, vision_scope_switch
from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions

_HIDDEN = 8
_IDS = {"image_token_id": 9001, "vision_start_token_id": 9002, "vision_end_token_id": 9003}
# CX4's counterexample tower: depth 2, hidden 8, intermediate 12, 2 heads, patch 2, temporal patch 1, merge 2
_TINY = {
    "depth": 2,
    "hidden_size": 8,
    "intermediate_size": 12,
    "out_hidden_size": _HIDDEN,
    "num_heads": 2,
    "patch_size": 2,
    "temporal_patch_size": 1,
    "spatial_merge_size": 2,
    "num_position_embeddings": 16,
    "deepstack_visual_indexes": [],
}
# Qwen3.8-Flash-Next's vision config (checkpoint config.json); built on the meta device for the layout only
_REAL = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 2560,
    "num_heads": 16,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "deepstack_visual_indexes": [],
    "hidden_act": "gelu_pytorch_tanh",
}
_DDP = SimpleNamespace(pad_buckets_for_high_nccl_busbw=False)
_KINDS = ("text", "image", "two_images", "image_off_this_tp_slice")


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
        self.layers = torch.nn.ModuleList(torch.nn.Linear(_HIDDEN, _HIDDEN) for _ in range(2))
        self.final_layernorm = torch.nn.LayerNorm(_HIDDEN)
        self.output_layer = torch.nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, input_ids=None, position_ids=None, decoder_input=None, **kwargs):
        hidden = decoder_input if decoder_input is not None else self.embedding(input_ids)
        for layer in self.layers:
            hidden = hidden + torch.tanh(layer(hidden))
        return self.output_layer(self.final_layernorm(hidden)).square().mean()


def _model(depth=2, recompute=False):
    torch.manual_seed(0)
    model = _Language()
    config = {**_IDS, "vision_config": {**_TINY, "depth": depth}}
    vision.wire_visual(model, config, scope="full", recompute=recompute, freeze_language=False, data="mixed")
    return model


def _batch(kind):
    """input_ids and image inputs: text; one image (grid 1x4x4, 4 tokens); two images (1x4x4 and 1x2x2, 5 tokens)."""
    if kind == "text":
        return torch.arange(100, 112)[None], {}
    grids = [[1, 4, 4]] if kind != "two_images" else [[1, 4, 4], [1, 2, 2]]
    ids = [100]
    for t, h, w in grids:
        ids += [9002] + [9001] * (t * h * w // 4) + [9003]
    ids = torch.tensor([ids + [101, 102]])
    grid = torch.tensor(grids)
    positions = image_rope_positions(ids[0], grid, merge_size=2, **_IDS)
    patches = int(grid.prod(-1).sum())
    pixels = torch.arange(patches * 12, dtype=torch.float32).reshape(patches, 12) / 200
    return ids, {"pixel_values": pixels, "image_grid_thw": grid, "qwen38_position_ids": positions}


def _forward(model, kind, monkeypatch):
    """One microbatch; image_off_this_tp_slice: TP rank 3 of 4 under SP, whose slice holds none of the image tokens."""
    from megatron.core import parallel_state

    off_slice = kind == "image_off_this_tp_slice"
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 3 if off_slice else 0)
    model.config.sequence_parallel = off_slice
    ids, inputs = _batch("image" if off_slice else kind)
    return model(input_ids=ids, **inputs)


def _layout(model, bucket_size, real_shapes=False):
    """{parameter name: bucket} from Megatron's layout function over DDP's parameter order (DP 2). real_shapes: the
    tower's parameters with Qwen3.8-Flash-Next's shapes (meta device, same names at depth 27)."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if real_shapes:
        with torch.device("meta"):
            tower = Qwen3VLVisionModel(Qwen3VLVisionConfig(**_REAL))
        real = {f"vision_model.{n}": p for n, p in tower.named_parameters()}
        assert set(real) == {n for n, _ in named if n.startswith("vision_model.")}
        named = [(n, real.get(n, p)) for n, p in named]
    layout = vision_scope_switch.compute_layout([p for _, p in named], bucket_size, 2, _DDP)
    return {n: layout.param_index_map[p][2] for n, p in named}


class _Chunk(torch.nn.Module):
    """Megatron DDP's surface: ddp_config, bucket_groups / expert_parallel_bucket_groups, finish_grad_sync, the module.

    Each group binds Megatron's own register_grad_ready / finish_grad_sync / reset; its start_grad_sync records the
    group's index in `issued` (one communicator) instead of a reduce-scatter."""

    def __init__(self, model, bucket_of, overlap=True):
        from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup as Group

        super().__init__()
        self.module = model
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap, num_distributed_optimizer_instances=1)
        self.issued = []
        params = dict(model.named_parameters())
        self.bucket_groups = []
        for index in sorted(set(bucket_of.values())):
            members = {params[n] for n, b in bucket_of.items() if b == index}
            group = SimpleNamespace(
                ddp_config=self.ddp_config,
                params=members,
                param_to_bucket={p: index for p in members},
                is_last_microbatch=True,
                is_first_batch=True,
                per_param_grad_ready_counts={},
                golden_per_param_grad_ready_counts={},
                grad_reduce_handle=None,
                grad_reduce_finished=False,
                param_gather_dispatched=False,
                _copy_back_extra_main_grads=lambda: None,
            )
            for name in ("register_grad_ready", "finish_grad_sync", "reset"):
                setattr(group, name, types.MethodType(getattr(Group, name), group))
            group.start_grad_sync = self._recorder(group, index)
            self.bucket_groups.append(group)
        self.expert_parallel_bucket_groups = []
        self.group_of = {p: g for g in self.bucket_groups for p in g.params}
        self._accs = []
        for p in self.group_of:  # Megatron DDP's backward post hook, registered on the gradient accumulator
            acc = p.expand_as(p).grad_fn.next_functions[0][0]
            acc.register_hook(lambda *unused, p=p: self.group_of[p].register_grad_ready(p))
            self._accs.append(acc)

    def _recorder(self, group, index):
        def start_grad_sync(force_all_reduce=False):
            if group.is_first_batch and group.grad_reduce_handle is not None:
                return  # Megatron: a no-op once the first step issued it
            assert group.grad_reduce_handle is None, "Should not have multiple communication calls outstanding at once"
            self.issued.append(index)
            group.grad_reduce_handle = SimpleNamespace(wait=lambda: None)

        return start_grad_sync

    def step(self, kind, monkeypatch):
        """One optimizer step with one (= last) microbatch: forward, backward, DDP finish_grad_sync, reset."""
        self.issued.clear()
        self.module.zero_grad(set_to_none=True)
        _forward(self.module, kind, monkeypatch).backward()
        for group in self.bucket_groups + self.expert_parallel_bucket_groups:  # DDP.finish_grad_sync
            group.finish_grad_sync()
        for group in self.bucket_groups:
            group.reset()
        return list(self.issued)


def _issue_orders(model, bucket_of, monkeypatch, defer=True):
    """{kind: issue order of the step} for every kind, each as a non-first step after an image first step."""
    chunk = _Chunk(model, bucket_of)
    if defer:
        assert vision_grad_sync.defer_vision_bucket_groups([chunk])
    chunk.step("image", monkeypatch)  # the first step: Megatron issues every group at finish_grad_sync
    return {kind: chunk.step(kind, monkeypatch) for kind in _KINDS}


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("bucket_size", [33, 64, 65])
def test_cx4_counterexample_issues_differently_without_the_deferral(bucket_size, recompute, monkeypatch):
    """Positive control: ready-triggered issue orders differ between text and image microbatches (F8-GUARD)."""
    model = _model(recompute=recompute)
    orders = _issue_orders(model, _layout(model, bucket_size), monkeypatch, defer=False)
    assert orders["text"] != orders["image"]


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("bucket_size", [33, 64, 65, 20_000_000, 320_000_000])
def test_every_microbatch_kind_issues_the_bucket_groups_in_one_order(bucket_size, recompute, monkeypatch):
    model = _model(recompute=recompute)
    bucket_of = _layout(model, bucket_size)
    orders = _issue_orders(model, bucket_of, monkeypatch)
    assert len({tuple(o) for o in orders.values()}) == 1, orders
    order = orders["text"]
    assert sorted(order) == sorted(set(bucket_of.values()))  # every group, exactly once
    vision_groups = {b for n, b in bucket_of.items() if n.startswith("vision_model.")}
    # language-model-only groups still go from the hook, during the backward; vision groups after, in index order
    deferred = [b for b in order if b in vision_groups]
    assert deferred == sorted(vision_groups) and order[-len(deferred) :] == deferred


@pytest.mark.parametrize("bucket_size", [20_000_000, 40_000_000, 320_000_000])
def test_real_tower_shapes_issue_the_bucket_groups_in_one_order(bucket_size, monkeypatch):
    """Qwen3.8-Flash-Next's tower shapes (depth 27) for the layout; a tiny-width tower of the same depth runs."""
    model = _model(depth=27)
    bucket_of = _layout(model, bucket_size, real_shapes=True)
    vision_groups = {b for n, b in bucket_of.items() if n.startswith("vision_model.")}
    assert len(vision_groups) > 1
    orders = _issue_orders(model, bucket_of, monkeypatch)
    assert len({tuple(o) for o in orders.values()}) == 1, orders


def test_a_deferred_group_still_requires_every_parameter_to_report(monkeypatch):
    """Like Megatron's hook path: a deferred group whose parameters did not all report fails instead of reducing."""
    model = _model()
    chunk = _Chunk(model, _layout(model, 64))
    vision_grad_sync.defer_vision_bucket_groups([chunk])
    chunk.step("image", monkeypatch)
    vision_group = next(g for g in chunk.bucket_groups if getattr(g, vision_grad_sync.MARKER, False))
    with pytest.raises(AssertionError, match="Deferred vision bucket group"):
        vision_group.finish_grad_sync()  # no backward ran this step: nothing reported


def test_deferral_marks_exactly_the_groups_with_vision_parameters_and_logs_them(capsys):
    model = _model()
    bucket_of = _layout(model, 64)
    chunk = _Chunk(model, bucket_of)
    tags = vision_grad_sync.defer_vision_bucket_groups([chunk])
    vision_groups = sorted({b for n, b in bucket_of.items() if n.startswith("vision_model.")})
    assert tags == [f"0:{b}" for b in vision_groups]
    marked = [i for i, g in enumerate(chunk.bucket_groups) if getattr(g, vision_grad_sync.MARKER, False)]
    assert marked == vision_groups and len(marked) < len(chunk.bucket_groups)
    assert f"Q38_F8_DEFERRED rank=0 groups=[{','.join(tags)}]" in capsys.readouterr().out
    assert vision_grad_sync.defer_vision_bucket_groups([chunk]) == tags  # idempotent


def test_an_undeferred_vision_group_fails_the_startup_assertion(monkeypatch):
    model = _model()
    chunk = _Chunk(model, _layout(model, 64))
    monkeypatch.setattr(vision_grad_sync, "_defer", lambda group: None)
    with pytest.raises(AssertionError, match="hold vision parameters but are not deferred"):
        vision_grad_sync.defer_vision_bucket_groups([chunk])


def test_nothing_is_deferred_without_overlap_or_ddp(capsys):
    model = _model()
    chunk = _Chunk(model, _layout(model, 64), overlap=False)
    assert vision_grad_sync.defer_vision_bucket_groups([chunk]) == []
    assert not any(getattr(g, vision_grad_sync.MARKER, False) for g in chunk.bucket_groups)
    assert "Q38_F8_DEFERRED rank=0 groups=[]" in capsys.readouterr().out
    assert vision_grad_sync.defer_vision_bucket_groups([model]) == []  # not a DDP chunk
