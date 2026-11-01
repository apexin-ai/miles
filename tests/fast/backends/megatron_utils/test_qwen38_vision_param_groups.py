"""CPU contracts for the Qwen3.8 vision scopes and the vision lr / wd param groups (TASK-VL F7).

Megatron's real _get_param_groups, OptimizerParamScheduler and param-group reload matcher run here; the model is a
small stand-in language model with the real transformers Qwen3-VL vision tower wired by vision.wire_visual.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from miles_plugins.models.qwen3_8_next import vision, vision_optimizer
from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions

_HIDDEN = 16
_LR, _MIN_LR, _WD = 1e-3, 1e-4, 0.1
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
        return self.word_embeddings(input_ids).transpose(0, 1)


class _Language(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre_process = True
        self.config = SimpleNamespace(params_dtype=torch.float32, sequence_parallel=False)
        self.embedding = _Embedding()
        self.head = torch.nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, input_ids=None, position_ids=None, decoder_input=None, **kwargs):
        hidden = decoder_input if decoder_input is not None else self.embedding(input_ids)
        return self.head(hidden).square().mean()


class _Wrapped(torch.nn.Module):
    """Gives parameter names the module. prefix DDP adds (module.vision_model.merger...)."""

    def __init__(self, model) -> None:
        super().__init__()
        self.module = model


def _model(scope, seed=0):
    torch.manual_seed(seed)
    model = _Language()
    vision.wire_visual(model, _CONFIG, scope=scope, recompute=False, freeze_language=False, data="mixed")
    return model


def _image_batch():
    ids = torch.tensor([[100, 101, 9002, 9001, 9001, 9001, 9001, 9003, 102, 103]])
    grid = torch.tensor([[1, 4, 4]])
    positions = image_rope_positions(
        ids[0], grid, image_token_id=9001, vision_start_token_id=9002, vision_end_token_id=9003, merge_size=2
    )
    return {"input_ids": ids, "pixel_values": torch.randn(16, 24), "image_grid_thw": grid, "qwen38_position_ids": positions}


@pytest.fixture
def tp_rank_zero(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)


@pytest.fixture
def gloo_world_of_one(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def _optimizer_config(**kwargs):
    from megatron.core.optimizer import OptimizerConfig

    return OptimizerConfig(lr=_LR, min_lr=_MIN_LR, weight_decay=_WD, **kwargs)


def _groups(chunks, config, overrides):
    from megatron.core.optimizer import _get_param_groups

    return _get_param_groups(chunks, config, overrides)


def _standard_groups(chunks, config):
    """What get_megatron_optimizer builds with config_overrides=None (_get_param_groups itself takes None as none)."""
    from megatron.core.optimizer import get_standard_config_overrides

    return _groups(chunks, config, get_standard_config_overrides(config))


def _group_of(groups, param):
    (group,) = [g for g in groups if any(p is param for p in g["params"])]
    return group


def _named(model):
    return dict(model.named_parameters())


# --- scopes: requires_grad and what reaches the optimizer ---------------------------------------------------------------


@pytest.mark.parametrize("scope", ["none", "projector", "full"])
def test_each_scope_trains_exactly_its_part_of_the_tower_and_only_that_reaches_the_optimizer(scope, gloo_world_of_one):
    model = _model(scope)
    chunks = [_Wrapped(model)]
    config = _optimizer_config()
    tower = dict(model.vision_model.named_parameters())
    assert len(tower) > 0 and any(n.startswith("merger.") for n in tower)
    want = {n for n in tower if scope == "full" or (scope == "projector" and n.startswith("merger."))}
    assert {n for n, p in tower.items() if p.requires_grad} == want
    in_optimizer = {id(p) for g in _standard_groups(chunks, config) for p in g["params"]}
    assert {n for n, p in tower.items() if id(p) in in_optimizer} == want
    assert {id(p) for p in model.head.parameters()} <= in_optimizer
    summary = vision_optimizer.check_vision_param_groups(
        chunks, config, vision_optimizer.vision_config_overrides(config), scope=scope
    )
    assert summary["vision_tensors"] == len(tower) and summary["vision_trainable"] == len(want)
    assert summary["vision_trainable_prefixes"] == sorted({n.split(".")[0] for n in want})


def test_the_check_refuses_a_tower_whose_requires_grad_does_not_follow_the_scope(gloo_world_of_one):
    model = _model("projector")
    model.vision_model.blocks[0].attn.qkv.weight.requires_grad_(True)  # a ViT tensor left trainable
    config = _optimizer_config()
    with pytest.raises(ValueError, match="blocks.0.attn.qkv.weight: requires_grad True, scope projector wants False"):
        vision_optimizer.check_vision_param_groups([model], config, vision_optimizer.vision_config_overrides(config), scope="projector")
    model = _model("full")
    model.vision_model.merger.linear_fc1.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="merger.linear_fc1.weight: requires_grad False, scope full wants True"):
        vision_optimizer.check_vision_param_groups([model], config, vision_optimizer.vision_config_overrides(config), scope="full")


# --- projector: the frozen ViT runs forward only ------------------------------------------------------------------------


def test_projector_builds_no_autograd_graph_in_the_frozen_vit_and_backprops_only_into_the_merger(tp_rank_zero):
    model = _model("projector")
    seen = {}

    def record(name):
        def hook(module, inputs, output):
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            seen[name] = (tensor.requires_grad, tensor.grad_fn is not None)

        return hook

    for i, block in enumerate(model.vision_model.blocks):
        block.register_forward_hook(record(f"blocks.{i}"))
    model.vision_model.patch_embed.register_forward_hook(record("patch_embed"))
    model.vision_model.merger.register_forward_pre_hook(
        lambda module, args: seen.__setitem__("merger_input", (args[0].requires_grad, args[0].grad_fn is not None))
    )
    model.vision_model.merger.register_forward_hook(record("merger"))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.numel()) or t, lambda t: t):
        loss = model(**_image_batch())
    # no ViT output carries a graph: autograd recorded nothing (and saved no activations) before the merger
    for name in ["patch_embed", "blocks.0", "blocks.1", "merger_input"]:
        assert seen[name] == (False, False), name
    assert seen["merger"] == (True, True)
    loss.backward()
    for name, p in model.vision_model.named_parameters():
        if name.startswith("merger."):
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name

    full = _model("full")
    full_saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: full_saved.append(t.numel()) or t, lambda t: t):
        full(**_image_batch())
    assert sum(saved) < sum(full_saved)


def test_none_runs_the_tower_without_grad(tp_rank_zero):
    model = _model("none")
    flags = []
    model.vision_model.merger.register_forward_hook(lambda m, i, o: flags.append(torch.is_grad_enabled()))
    model(**_image_batch()).backward()
    assert flags == [False]
    assert all(p.grad is None for p in model.vision_model.parameters())


# --- lr / wd param groups -----------------------------------------------------------------------------------------------


def _same_groups(left, right):
    def shape(groups):
        return sorted(
            (tuple(sorted((k, v) for k, v in g.items() if k != "params")), tuple(sorted(id(p) for p in g["params"])))
            for g in groups
        )

    return shape(left) == shape(right)


def test_default_multipliers_give_exactly_megatrons_standard_param_groups(gloo_world_of_one):
    from megatron.core.optimizer import get_standard_config_overrides

    model = _model("full")
    config = _optimizer_config()
    overrides = vision_optimizer.vision_config_overrides(config)
    assert set(overrides) == set(get_standard_config_overrides(config))
    assert _same_groups(_groups([model], config, overrides), _standard_groups([model], config))


@pytest.mark.parametrize("scope", ["projector", "full"])
def test_vision_lr_mult_scales_only_the_trainable_vision_parameters(scope, gloo_world_of_one):
    model = _model(scope)
    chunks = [_Wrapped(model)]
    config = _optimizer_config()
    overrides = vision_optimizer.vision_config_overrides(config, lr_mult=0.25)
    groups = _groups(chunks, config, overrides)
    for name, p in _named(model).items():
        if not p.requires_grad:
            continue
        group = _group_of(groups, p)
        decayed = not (p.ndim == 1 or name.endswith(".bias"))
        assert group["wd_mult"] == (1.0 if decayed else 0.0), name
        if name.startswith("vision_model."):
            assert (group["max_lr"], group["min_lr"], group["lr_mult"]) == (_LR * 0.25, _MIN_LR * 0.25, 0.25), name
        else:
            assert (group["max_lr"], group["min_lr"], group["lr_mult"]) == (_LR, _MIN_LR, 1.0), name
    summary = vision_optimizer.check_vision_param_groups(chunks, config, overrides, scope=scope, lr_mult=0.25)
    assert {(g["part"], g["lr_mult"]) for g in summary["groups"]} == {("vision", 0.25), ("language", 1.0)}


@pytest.mark.parametrize("wd_mult", [0.0, 0.3])
def test_vision_wd_mult_keeps_vision_biases_and_1d_parameters_undecayed(wd_mult, gloo_world_of_one):
    model = _model("full")
    config = _optimizer_config()
    overrides = vision_optimizer.vision_config_overrides(config, wd_mult=wd_mult)
    groups = _groups([model], config, overrides)
    kinds = set()
    for name, p in _named(model).items():
        if not p.requires_grad:
            continue
        group = _group_of(groups, p)
        one_d = p.ndim == 1 or name.endswith(".bias")
        is_vision = name.startswith("vision_model.")
        kinds.add((is_vision, one_d))
        assert group["wd_mult"] == (0.0 if one_d else (wd_mult if is_vision else 1.0)), name
        assert (group["max_lr"], group.get("lr_mult")) == (_LR, 1.0), name
    assert kinds == {(True, True), (True, False), (False, True), (False, False)}
    vision_optimizer.check_vision_param_groups([model], config, overrides, scope="full", wd_mult=wd_mult)


def test_the_qk_layernorm_weight_decay_rule_is_kept_too(gloo_world_of_one):
    """With apply_wd_to_qk_layernorm Megatron's no-decay key is a different predicate; the vision rule follows it."""
    model = _model("full")
    config = _optimizer_config(apply_wd_to_qk_layernorm=True)
    overrides = vision_optimizer.vision_config_overrides(config, lr_mult=2.0, wd_mult=0.5)
    groups = _groups([model], config, overrides)
    for name, p in model.vision_model.named_parameters():
        assert _group_of(groups, p)["wd_mult"] == (0.0 if p.ndim == 1 or name.endswith(".bias") else 0.5), name
    vision_optimizer.check_vision_param_groups([model], config, overrides, scope="full", lr_mult=2.0, wd_mult=0.5)


def test_the_check_refuses_a_rule_that_reaches_language_parameters(gloo_world_of_one):
    from megatron.core.optimizer import ParamKey

    model = _model("full")
    config = _optimizer_config()
    overrides = vision_optimizer.vision_config_overrides(config, lr_mult=0.5)
    overrides[ParamKey(name="head.weight")] = {"max_lr": _LR * 0.5, "min_lr": _MIN_LR * 0.5, "lr_mult": 0.5}
    with pytest.raises(ValueError, match="head.weight: override"):
        vision_optimizer.check_vision_param_groups([model], config, overrides, scope="full", lr_mult=0.5)
    # and one that silently dropped Megatron's no-decay default
    bare = {k: v for k, v in vision_optimizer.vision_config_overrides(config, lr_mult=0.5).items() if "wd_mult" not in v}
    with pytest.raises(ValueError, match="head.bias: override"):
        vision_optimizer.check_vision_param_groups([model], config, bare, scope="full", lr_mult=0.5)


@pytest.mark.parametrize("lr_mult,wd_mult", [(0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0), (float("inf"), 1.0), (1.0, -0.1), (1.0, float("nan"))])
def test_bad_multipliers_are_refused(lr_mult, wd_mult):
    with pytest.raises(ValueError, match="qwen38-vision"):
        vision_optimizer.check_multipliers(lr_mult, wd_mult)


# --- the schedule and a reload -------------------------------------------------------------------------------------------


def _identifier(group):
    from megatron.core.optimizer.optimizer import param_group_identifier_keys

    return tuple(group[k] for k in param_group_identifier_keys)


def _torch_optimizer(groups):
    return torch.optim.AdamW([dict(g) for g in groups if g["params"]], lr=_LR, betas=(0.9, 0.98))


def _scheduler(optimizer):
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    return OptimizerParamScheduler(
        optimizer, init_lr=0.0, max_lr=_LR, min_lr=_MIN_LR, lr_warmup_steps=2, lr_decay_steps=10,
        lr_decay_style="cosine", start_wd=_WD, end_wd=_WD, wd_incr_steps=10, wd_incr_style="constant",
    )


def _lr_wd_by_part(optimizer, model):
    vision_ids = {id(p) for p in model.vision_model.parameters()}
    out = {}
    for g in optimizer.param_groups:
        part = "vision" if id(g["params"][0]) in vision_ids else "language"
        decayed = g["wd_mult"] != 0.0
        out[(part, decayed)] = (g["lr"], g["weight_decay"])
    return out


def test_scheduler_gives_vision_lr_and_wd_times_their_multipliers_at_every_step(gloo_world_of_one):
    model = _model("full")
    config = _optimizer_config()
    overrides = vision_optimizer.vision_config_overrides(config, lr_mult=0.5, wd_mult=0.2)
    optimizer = _torch_optimizer(_groups([model], config, overrides))
    scheduler = _scheduler(optimizer)
    for step in range(12):  # warmup, cosine decay, then past lr_decay_steps (min_lr)
        got = _lr_wd_by_part(optimizer, model)
        lang_lr = got[("language", True)][0]
        assert got[("vision", True)][0] == pytest.approx(0.5 * lang_lr, rel=1e-12, abs=0), step
        assert got[("vision", False)][0] == pytest.approx(0.5 * lang_lr, rel=1e-12, abs=0), step
        assert got[("language", True)][1] == pytest.approx(_WD) and got[("language", False)][1] == 0.0
        assert got[("vision", True)][1] == pytest.approx(0.2 * _WD) and got[("vision", False)][1] == 0.0
        scheduler.step(1)
    assert _lr_wd_by_part(optimizer, model)[("vision", True)][0] == pytest.approx(0.5 * _MIN_LR)


def _reload_groups(current, saved):
    """What DistributedOptimizer.load_state_dict (Megatron fdf82356 distrib_optimizer.py:994-1021) does with the
    checkpointed param groups: index them by param_group_identifier_keys, then give each running group the saved
    group with its key (hyperparameters from the checkpoint, params from the running optimizer)."""
    by_key = {_identifier(g): g for g in saved}
    return [{**by_key[_identifier(g)], "params": g["params"]} for g in current]


def _save_and_reload(groups_fn, steps_before=3):
    """Uninterrupted vs saved after `steps_before` and reloaded into a fresh optimizer: per-group lr afterwards."""
    model = _model("full")
    first = _torch_optimizer(groups_fn(model))
    sched = _scheduler(first)
    sched.step(steps_before)
    saved_groups, saved_sched = first.state_dict()["param_groups"], sched.state_dict()

    fresh = _torch_optimizer(groups_fn(model))
    resumed = _scheduler(fresh)
    restored = _reload_groups(fresh.state_dict()["param_groups"], saved_groups)
    fresh.load_state_dict({"state": {}, "param_groups": restored})
    resumed.load_state_dict(saved_sched)
    sched.step(1)
    resumed.step(1)
    return _lr_wd_by_part(first, model), _lr_wd_by_part(fresh, model), restored


def test_a_reload_restores_each_vision_group_with_its_own_lr_and_wd(gloo_world_of_one):
    config = _optimizer_config()
    overrides = vision_optimizer.vision_config_overrides(config, lr_mult=0.5, wd_mult=0.2)
    groups = lambda model: _groups([model], config, overrides)
    keys = [_identifier(g) for g in groups(_model("full"))]
    assert len(keys) == len(set(keys)), "every param group needs its own checkpoint key"
    uninterrupted, resumed, _ = _save_and_reload(groups)
    assert resumed == uninterrupted
    assert resumed[("vision", True)][0] == pytest.approx(0.5 * resumed[("language", True)][0], rel=1e-12)

    # Megatron's own matcher for the non-distributed optimizers and sharded state dicts agrees
    from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer

    model = _model("full")
    current = groups(model)
    saved = [{k: v for k, v in g.items()} for g in current]
    matched = MixedPrecisionOptimizer._filter_and_reorder_param_groups(current, list(reversed(saved)))
    assert [g["max_lr"] for g in matched] == [g["max_lr"] for g in current]


def test_positive_control_a_vision_lr_given_by_max_lr_alone_is_lost_on_reload(gloo_world_of_one):
    """Why lr_mult is set: a group that differs only in max_lr / min_lr shares the language group's checkpoint key."""
    from megatron.core.optimizer import ParamKey, ParamWithNamePredicate, get_standard_config_overrides

    config = _optimizer_config()
    naive = dict(get_standard_config_overrides(config))
    naive[ParamKey(with_name_predicate=ParamWithNamePredicate(name="naive", fn=lambda p, n: n.startswith("vision_model.")))] = {
        "max_lr": _LR * 0.5,
        "min_lr": _MIN_LR * 0.5,
    }
    groups = lambda model: _groups([model], config, naive)
    keys = [_identifier(g) for g in groups(_model("full"))]
    assert len(keys) != len(set(keys))
    uninterrupted, resumed, _ = _save_and_reload(groups)
    assert uninterrupted[("vision", True)][0] == pytest.approx(0.5 * uninterrupted[("language", True)][0], rel=1e-12)
    # the vision and language groups share a key: after the reload both carry one checkpointed group's max_lr
    assert resumed[("vision", True)][0] == pytest.approx(resumed[("language", True)][0], rel=1e-12)
    assert resumed != uninterrupted


# --- zero-gradient (text-only) steps under AdamW ---------------------------------------------------------------------


@pytest.mark.parametrize("wd", [0.0, 0.1])
def test_text_only_steps_still_move_a_trainable_projector_but_never_the_frozen_vit(wd, tp_rank_zero, gloo_world_of_one):
    """Documents the semantics: zero grads on a text-only step do not freeze AdamW (momentum, decoupled decay)."""
    model = _model("projector")
    config = _optimizer_config()
    groups = _standard_groups([model], config)
    optimizer = torch.optim.AdamW(
        [{"params": g["params"], "weight_decay": wd * g["wd_mult"]} for g in groups if g["params"]], lr=_LR
    )
    vit = {n: p.detach().clone() for n, p in model.vision_model.named_parameters() if not n.startswith("merger.")}

    optimizer.zero_grad()
    model(**_image_batch()).backward()
    optimizer.step()
    merger = {n: p.detach().clone() for n, p in model.vision_model.named_parameters() if n.startswith("merger.")}

    optimizer.zero_grad()
    model(input_ids=torch.arange(100, 112)[None]).backward()  # a text microbatch: exact zero grads for the merger
    for n, p in model.vision_model.named_parameters():
        if n.startswith("merger."):
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0, n
    optimizer.step()
    changed = [n for n, p in model.vision_model.named_parameters() if n.startswith("merger.") and not torch.equal(p, merger[n])]
    assert changed  # momentum from the image step (and, with wd > 0, decay) still updates the merger
    for n, p in model.vision_model.named_parameters():
        if not n.startswith("merger."):
            assert torch.equal(p, vit[n]), n


# --- model.py wiring ------------------------------------------------------------------------------------------------------


def _args(provider="miles_plugins.models.qwen3_8_next.vision.get_qwen38_vision_model_provider", **kwargs):
    fields = {"qwen38_vision_lr_mult": 1.0, "qwen38_vision_wd_mult": 1.0, "qwen38_vision_trainable": "projector"}
    return SimpleNamespace(custom_model_provider_path=SimpleNamespace(path=provider) if provider else None, **(fields | kwargs))


@pytest.fixture
def megatron_model_module(monkeypatch):
    from miles.backends.megatron_utils import model as module

    for name in ["get_data_parallel_rank", "get_tensor_model_parallel_rank", "get_pipeline_model_parallel_rank"]:
        monkeypatch.setattr(module.mpu, name, lambda: 0)
    return module


def test_setup_passes_the_merged_overrides_only_when_a_multiplier_is_set(megatron_model_module, gloo_world_of_one, caplog):
    config = _optimizer_config()
    model = [_model("projector")]
    with caplog.at_level("INFO", logger=megatron_model_module.logger.name):
        assert megatron_model_module._qwen38_vision_config_overrides(_args(), config, model) is None
    assert "qwen38 vision param groups" in caplog.text and "'vision_trainable_prefixes': ['merger']" in caplog.text
    overrides = megatron_model_module._qwen38_vision_config_overrides(_args(qwen38_vision_lr_mult=0.5), config, model)
    assert set(overrides) >= set(vision_optimizer.vision_config_overrides(_optimizer_config()))
    assert len(overrides) == len(vision_optimizer.vision_config_overrides(config)) + 1
    assert megatron_model_module._qwen38_vision_config_overrides(_args(provider=None), config, model) is None


@pytest.mark.parametrize(
    "args,match",
    [
        (dict(provider=None, qwen38_vision_lr_mult=0.5), "need the Qwen3.8 vision provider"),
        (dict(provider="miles_plugins.models.qwen3_8_next.model_provider.x", qwen38_vision_wd_mult=0.0), "need the Qwen3.8 vision provider"),
        (dict(qwen38_vision_trainable="none", qwen38_vision_lr_mult=2.0), "frozen tower"),
        (dict(qwen38_vision_lr_mult=0.0), "must be a finite number > 0"),
    ],
)
def test_setup_refuses_multipliers_that_would_do_nothing_or_are_invalid(args, match, megatron_model_module, gloo_world_of_one):
    with pytest.raises(ValueError, match=match):
        megatron_model_module._qwen38_vision_config_overrides(_args(**args), _optimizer_config(), [_model("projector")])


def test_describe_optimizer_names_the_vision_parameters_a_built_optimizer_updates(gloo_world_of_one):
    model = _model("projector")
    config = _optimizer_config()
    groups = _standard_groups([model], config)
    inner = torch.optim.AdamW([dict(g) for g in groups if g["params"]], lr=_LR)
    wrapper = SimpleNamespace(optimizer=inner, param_groups=inner.param_groups)
    described = vision_optimizer.describe_optimizer([model], SimpleNamespace(chained_optimizers=[wrapper]))
    assert described["vision_in_optimizer_prefixes"] == ["merger"]
    assert described["vision_in_optimizer"] == sum(1 for n, _ in model.vision_model.named_parameters() if n.startswith("merger."))
    assert {g["wd_mult"] for g in described["param_groups"]} == {0.0, 1.0}
