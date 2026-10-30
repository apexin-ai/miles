"""Resuming from a checkpoint must continue the LR / WD schedule exactly where it stopped.

Uses Megatron's real OptimizerParamScheduler (its file loaded under a private name, because the
module fixture below replaces megatron with stubs) and the real load_model_state; only the
checkpoint read is faked, by restoring the scheduler the way Megatron's load_checkpoint does.
"""

import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Import the real module before the stub fixture swaps megatron out; collected alone, its imports
# would otherwise resolve against the stubs.
import miles.backends.megatron_utils.model  # noqa: F401
from tests.fast.backends.megatron_utils.test_model_initialize import (  # noqa: F401  (autouse stub fixture)
    _FakeModelChunk,
    _mock_megatron_environment,
    _patch_initialize_side_effects,
)
from tests.fast.fixtures.args_fixtures import make_trainer_args

_SCHEDULER_FILE = Path(importlib.util.find_spec("megatron.core").submodule_search_locations[0]) / "optimizer_param_scheduler.py"

# Samples per optimizer update (dynamic batches); None = a skipped update (NaN grad), which does not step.
_INCREMENTS = [8, 5, 8, None, 7, 8, 8, 6, 3, 8]


def _scheduler_class():
    spec = importlib.util.spec_from_file_location("_real_optimizer_param_scheduler", _SCHEDULER_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["megatron.core.utils"].log_single_rank = lambda *args, **kwargs: None
    spec.loader.exec_module(module)
    return module.OptimizerParamScheduler


def _new_scheduler(use_checkpoint_opt_param_scheduler: bool):
    optimizer = SimpleNamespace(param_groups=[{"params": []}, {"params": [], "wd_mult": 0.0}])
    return _scheduler_class()(
        optimizer,
        init_lr=0.0,
        max_lr=1e-5,
        min_lr=1e-6,
        lr_warmup_steps=12,
        lr_decay_steps=64,
        lr_decay_style="cosine",
        start_wd=0.05,
        end_wd=0.1,
        wd_incr_steps=64,
        wd_incr_style="linear",
        use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=False,
    )


def _train(scheduler, increments) -> list[tuple[float, float, float]]:
    """(lr, weight_decay of both groups) after each update, as train_one_step steps the scheduler."""
    seen = []
    for increment in increments:
        if increment is not None:
            scheduler.step(increment=increment)
        groups = scheduler.optimizer.param_groups
        seen.append((groups[0]["lr"], groups[0]["weight_decay"], groups[1]["weight_decay"]))
    return seen


def _load(
    tmp_path: Path, scheduler, *, iteration: int, saved_state: dict | None, finetune: bool = False, tracker: str | None = None
):
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir(exist_ok=True)
    (load_dir / "latest_checkpointed_iteration.txt").write_text(tracker or str(iteration))

    def megatron_load(model, optimizer, opt_param_scheduler, **kwargs):
        if saved_state is not None:
            opt_param_scheduler.load_state_dict(saved_state)
        return iteration, not finetune

    with ExitStack() as stack:
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", side_effect=megatron_load))
        _patch_initialize_side_effects(stack)
        return load_model_state(
            make_trainer_args(
                use_checkpoint_opt_param_scheduler=scheduler.use_checkpoint_opt_param_scheduler,
                global_batch_size=8,
                finetune=finetune,
                load=str(load_dir),
            ),
            model=[_FakeModelChunk()],
            optimizer=None,
            opt_param_scheduler=scheduler,
            role="actor",
            checkpointing_context=None,
            resume_from_ckpt=True,
        )


class TestResumedScheduleMatchesUninterrupted:
    @pytest.mark.parametrize("use_checkpoint_opt_param_scheduler", [True, False])
    @pytest.mark.parametrize("stop_after", [1, 3, 4, 7])
    def test_every_lr_and_wd_after_resume_equals_the_uninterrupted_run(
        self, tmp_path: Path, use_checkpoint_opt_param_scheduler: bool, stop_after: int
    ) -> None:
        """Dynamic batch sizes and a skipped update included; compared with ==, not approx."""
        uninterrupted = _train(_new_scheduler(use_checkpoint_opt_param_scheduler), _INCREMENTS)

        first = _new_scheduler(use_checkpoint_opt_param_scheduler)
        before = _train(first, _INCREMENTS[:stop_after])
        saved = first.state_dict()

        resumed = _new_scheduler(use_checkpoint_opt_param_scheduler)
        output = _load(tmp_path, resumed, iteration=stop_after, saved_state=saved)
        after = _train(resumed, _INCREMENTS[stop_after:])

        assert output.start_rollout_id == stop_after + 1
        assert resumed.num_steps == sum(i for i in _INCREMENTS if i is not None)
        assert before + after == uninterrupted

    def test_a_scheduler_that_already_stepped_is_restored_to_the_saved_count(self, tmp_path: Path) -> None:
        """Megatron's load_state_dict adds the saved count to the current one; the load makes it absolute."""
        first = _new_scheduler(True)
        _train(first, _INCREMENTS[:4])
        resumed = _new_scheduler(True)
        _train(resumed, [8, 8])
        _load(tmp_path, resumed, iteration=4, saved_state=first.state_dict())
        assert resumed.num_steps == first.num_steps
        assert resumed.optimizer.param_groups[0]["lr"] == first.optimizer.param_groups[0]["lr"]

    def test_the_wrapper_is_removed_after_the_load(self, tmp_path: Path) -> None:
        """Later explicit load_state_dict calls behave like Megatron's own."""
        scheduler = _new_scheduler(True)
        original = scheduler.load_state_dict
        _load(tmp_path, scheduler, iteration=2, saved_state=_new_scheduler(True).state_dict())
        assert scheduler.load_state_dict == original


    def test_a_checkpoint_saved_after_rollout_0_resumes_the_schedule(self, tmp_path: Path) -> None:
        """iteration is the zero-based rollout id: rollout 0 has trained and its schedule continues."""
        uninterrupted = _train(_new_scheduler(False), _INCREMENTS)
        first = _new_scheduler(False)
        before = _train(first, _INCREMENTS[:2])
        resumed = _new_scheduler(False)
        output = _load(tmp_path, resumed, iteration=0, saved_state=first.state_dict())
        assert output.start_rollout_id == 1
        assert before + _train(resumed, _INCREMENTS[2:]) == uninterrupted


class TestLoadsThatDoNotRestoreTheScheduler:
    @pytest.mark.parametrize("iteration", [5, 0])
    def test_a_trained_checkpoint_without_scheduler_state_is_refused(self, tmp_path: Path, iteration: int) -> None:
        """--no-load-optim / --no-save-optim: the LR progress is unknown, so do not guess iteration * GBS.
        CRV-FIX-011: a checkpoint written after rollout 0 has trained too."""
        with pytest.raises(RuntimeError, match="no LR scheduler state"):
            _load(tmp_path, _new_scheduler(True), iteration=iteration, saved_state=None)

    def test_a_release_checkpoint_starts_the_schedule_at_step_zero(self, tmp_path: Path) -> None:
        """A converted (release) checkpoint is untrained: Megatron loads it at iteration 0 without optimizer state."""
        scheduler = _new_scheduler(True)
        _load(tmp_path, scheduler, iteration=0, saved_state=None, tracker="release")
        assert scheduler.num_steps == 0

    def test_finetune_starts_a_new_schedule_at_step_zero(self, tmp_path: Path) -> None:
        """--finetune is the explicit new-stage policy: iteration 0, nothing restored, no error."""
        scheduler = _new_scheduler(True)
        output = _load(tmp_path, scheduler, iteration=0, saved_state=None, finetune=True)
        assert output.start_rollout_id == 0
        assert scheduler.num_steps == 0
        assert _train(scheduler, _INCREMENTS) == _train(_new_scheduler(True), _INCREMENTS)

    def test_a_reload_without_restore_rewrites_the_stale_lr_in_the_param_groups(self, tmp_path: Path) -> None:
        """actor.load_state zeroes num_steps; the param groups must not keep the previous run's LR for the
        first update (the scheduler is stepped only after optimizer.step)."""
        scheduler = _new_scheduler(True)
        _train(scheduler, _INCREMENTS)
        scheduler.num_steps = 0
        _load(tmp_path, scheduler, iteration=0, saved_state=None, finetune=True)
        fresh = _new_scheduler(True).optimizer.param_groups
        assert [(g["lr"], g["weight_decay"]) for g in scheduler.optimizer.param_groups] == [
            (g["lr"], g["weight_decay"]) for g in fresh
        ]
