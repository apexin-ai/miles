"""Strict typed-config contracts for the PR #5/#6 final-save reload checks.

The checks run through the real init-hook view (compute_custom_function_config), the
real ``save`` argument order used by ``MegatronTrainRayActor.save_model`` and a loader
stub that binds against the real ``load_checkpoint`` signature. Earlier tests used
SimpleNamespace args and ``*args/**kwargs`` loaders, which hid the contract changes.
"""

import importlib
import inspect
import json
import sys
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config, replace_config_values

from miles.backends.megatron_utils import checkpoint as checkpoint_module
from miles.backends.megatron_utils import model as backend
from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS
from miles.utils.args.custom_view import compute_custom_function_config
from miles.utils.args.trainer_utils import compute_trainer_config

QWEN_HOOK = "miles_plugins.models.qwen3_8_next.vision_audit.install_final_save_check"
DSV41_HOOK = "miles_plugins.models.deepseek_v4_1.checkpoint_audit.install_final_save_check"
NUM_ROLLOUT = 3


def _trainer_config(hook: str, tmp_path, *argv: str):
    parsed = parse_megatron_test_config(
        "--custom-megatron-init-path",
        hook,
        "--save",
        str(tmp_path / "ckpt"),
        "--save-interval",
        "1",
        "--num-rollout",
        str(NUM_ROLLOUT),
        *argv,
    )
    return compute_trainer_config(all_config=parsed, trainer=parsed.raw_megatron.trainers[0])


def _real_save_signature_stub(calls):
    real = inspect.signature(backend.save)

    def save(*positional, **keywords):
        bound = real.bind(*positional, **keywords)
        calls.append(bound.arguments)

    save.__signature__ = real
    return save


def _model(names):
    chunk = torch.nn.Module()
    for index in range(len(names)):
        chunk.register_parameter(f"p{index}", torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + index))
    named = list(zip(names, chunk.parameters(), strict=True))
    return [SimpleNamespace(named_parameters=lambda: iter(named))], {n: p.detach().clone() for n, p in named}


_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


@pytest.fixture(scope="module")
def actor() -> Iterator[ModuleType]:
    """Import the Megatron actor with its unavailable native memory dependency stubbed (as test_actor.py)."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")
    try:
        yield importlib.import_module(_ACTOR_MODULE_NAME)
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


@pytest.fixture
def harness(monkeypatch, actor):
    """Install stubs for save, load and torch.distributed; restore the real ones on teardown."""
    saves, loads = [], []
    save_stub = _real_save_signature_stub(saves)
    monkeypatch.setattr(backend, "save", save_stub)
    monkeypatch.setattr(actor, "save", save_stub)
    real_load = inspect.signature(checkpoint_module.load_checkpoint)
    state = SimpleNamespace(saves=saves, loads=loads, restore=None, actor=actor)

    def load_checkpoint(*positional, **keywords):
        bound = real_load.bind(*positional, **keywords)
        args = bound.arguments["args"]
        loads.append({name: vars(args.backend)[name] for name in CHECKPOINT_LOAD_FIELDS})
        assert all(torch.isnan(p.detach().flatten()[0]) for p in state.perturbed())
        if state.restore is not None:
            state.restore()
        return checkpoint_module._CheckpointLoadResult(iteration=0, restored_trained_iteration=False)

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", load_checkpoint)
    for module_name in ("qwen3_8_next.vision_audit", "deepseek_v4_1.checkpoint_audit"):
        module = __import__(f"miles_plugins.models.{module_name}", fromlist=["dist"])
        monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(module.dist, "get_world_size", lambda: 1)
        monkeypatch.setattr(module.dist, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return state


def _install(args):
    # Same call as initialize.py for --custom-megatron-init-path.
    from miles.utils.function_registry import load_function

    load_function(args.custom_megatron_init_path)(compute_custom_function_config(args, args.custom_megatron_init_path))


def _final_save(harness, args, model, rollout_id=NUM_ROLLOUT - 1):
    # Same argument order as MegatronTrainRayActor.save_model.
    harness.actor.save(args, rollout_id, model, None, None, snapshot_publisher=None)


class TestQwenFinalSaveCheck:
    def _args(self, tmp_path):
        return _trainer_config(QWEN_HOOK, tmp_path, "--qwen38-audit-dir", str(tmp_path / "audit"))

    def test_final_save_reloads_through_the_typed_contract(self, tmp_path, harness) -> None:
        args = self._args(tmp_path)
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args.backend))
        model, saved = _model(["decoder.w", "vision_model.blocks.0.w"])
        params = [p for chunk in model for _, p in chunk.named_parameters()]
        harness.perturbed = lambda: params
        harness.restore = lambda: [p.data.copy_(saved[n]) for chunk in model for n, p in chunk.named_parameters()]
        _install(args)

        _final_save(harness, args, model, rollout_id=0)
        assert len(harness.saves) == 1 and harness.loads == []

        _final_save(harness, args, model)
        assert harness.loads == [
            dict(
                load=args.backend.save, ckpt_step=NUM_ROLLOUT - 1, no_load_optim=True, no_load_rng=True, finetune=True
            )
        ]
        assert harness.saves[-1]["args"] is args and harness.saves[-1]["iteration"] == NUM_ROLLOUT - 1
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args.backend))
        receipt = json.loads((tmp_path / "audit/checkpoint-reload/rank-000.json").read_text())
        assert receipt["passed"] and receipt["checkpoint_iteration"] == NUM_ROLLOUT - 1

    def test_failed_reload_raises_and_leaves_no_nan_slices(self, tmp_path, harness) -> None:
        args = self._args(tmp_path)
        model, saved = _model(["decoder.w", "vision_model.blocks.0.w"])
        params = [p for chunk in model for _, p in chunk.named_parameters()]
        harness.perturbed = lambda: params
        _install(args)

        with pytest.raises(ValueError, match="did not restore"):
            _final_save(harness, args, model)
        assert all(torch.equal(p, saved[n]) for chunk in model for n, p in chunk.named_parameters())
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args.backend))
        assert not (tmp_path / "audit/checkpoint-reload/rank-000.json").exists()
        failure = json.loads((tmp_path / "audit/checkpoint-reload/rank-000.failed.json").read_text())
        assert failure["stage"] == "compare" and failure["loader_entered"] and failure["loader_returned"]
        assert {row["name"] for row in failure["mismatches"]} == {"decoder.w", "vision_model.blocks.0.w"}

    def test_hook_rejects_async_save(self, tmp_path, harness) -> None:
        # Megatron's parser drops --async-save for this toy setup, so set the parsed value directly.
        args = replace_config_values(self._args(tmp_path), async_save=True)
        with pytest.raises(ValueError, match="synchronous"):
            _install(args)


class TestDeepSeekFinalSaveCheck:
    NAMES = [f"module.visual.t{i}" for i in range(266)] + [
        "module.mlp.router.weight",
        "module.mlp.experts.linear_fc1.weight0",
        "module.self_attention.linear_q_up_proj.weight",
    ]

    def test_init_hook_accepts_the_framework_view_and_reloads_on_final_save(
        self, tmp_path, harness, monkeypatch
    ) -> None:
        monkeypatch.setenv("DSV41_AUDIT_DIR", str(tmp_path / "audit"))
        args = _trainer_config(DSV41_HOOK, tmp_path)
        model, saved = _model(self.NAMES)
        params = [p for chunk in model for _, p in chunk.named_parameters()]
        harness.perturbed = lambda: params
        harness.restore = lambda: [p.data.copy_(saved[n]) for chunk in model for n, p in chunk.named_parameters()]
        _install(args)

        _final_save(harness, args, model)

        assert harness.loads == [
            dict(
                load=args.backend.save, ckpt_step=NUM_ROLLOUT - 1, no_load_optim=True, no_load_rng=True, finetune=True
            )
        ]
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args.backend))
        assert json.loads((tmp_path / "audit/checkpoint-reload/rank-000.json").read_text())["passed"]

    def test_failed_reload_raises_and_leaves_no_nan_slices(self, tmp_path, harness, monkeypatch) -> None:
        monkeypatch.setenv("DSV41_AUDIT_DIR", str(tmp_path / "audit"))
        args = _trainer_config(DSV41_HOOK, tmp_path)
        model, saved = _model(self.NAMES)
        params = [p for chunk in model for _, p in chunk.named_parameters()]
        harness.perturbed = lambda: params
        _install(args)

        with pytest.raises(ValueError, match="failed to restore"):
            _final_save(harness, args, model)
        assert all(torch.equal(p, saved[n]) for chunk in model for n, p in chunk.named_parameters())
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args.backend))
