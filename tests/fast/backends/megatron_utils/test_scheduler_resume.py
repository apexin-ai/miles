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
    tmp_path: Path,
    scheduler,
    *,
    iteration: int,
    saved_state: dict | None,
    finetune: bool = False,
    tracker: str | None = None,
    dirname: str = "ckpt",
):
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / dirname
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

    def test_a_trained_checkpoint_in_a_directory_named_release_is_still_refused(self, tmp_path: Path) -> None:
        """CRV-FIX-016: Megatron follows the tracker's content, so must the release classification."""
        with pytest.raises(RuntimeError, match="no LR scheduler state"):
            _load(tmp_path, _new_scheduler(True), iteration=0, saved_state=None, tracker="0", dirname="release")

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


@pytest.mark.parametrize(
    ("dirname", "tracker", "ckpt_step", "release"),
    [
        ("ckpt", "release", None, True),
        ("ckpt", "7", None, False),
        ("release", "0", None, False),  # CRV-FIX-016: the tracker wins over the directory name
        ("release", None, None, True),
        ("ckpt", "release", 3, False),  # an explicit --ckpt-step loads that trained iteration
    ],
)
def test_release_classification_follows_what_megatron_reads(tmp_path: Path, dirname, tracker, ckpt_step, release) -> None:
    from miles.backends.megatron_utils.checkpoint import is_release_checkpoint

    load_dir = tmp_path / dirname
    load_dir.mkdir()
    if tracker is not None:
        (load_dir / "latest_checkpointed_iteration.txt").write_text(tracker)
    assert is_release_checkpoint(load_dir, ckpt_step) is release
    assert is_release_checkpoint(None, None) is False


def _groups(scheduler) -> list[tuple[float, float]]:
    return [(g["lr"], g["weight_decay"]) for g in scheduler.optimizer.param_groups]


class TestTheRestoreIsObservedNotInferred:
    """REBASE-plan (#3964 observer): the decision follows a real load_state_dict call, not a flag, the
    iteration or a change of num_steps."""

    def test_a_restore_that_leaves_num_steps_unchanged_is_still_a_restore(self, tmp_path: Path) -> None:
        """The scheduler already holds the saved count: a before/after comparison would see no restore."""
        first = _new_scheduler(True)
        _train(first, _INCREMENTS[:5])
        resumed = _new_scheduler(True)
        _train(resumed, _INCREMENTS[:5])
        assert resumed.num_steps == first.num_steps
        output = _load(tmp_path, resumed, iteration=4, saved_state=first.state_dict())
        assert output.start_rollout_id == 5
        assert resumed.num_steps == first.num_steps
        assert _groups(resumed) == _groups(first)

    def test_loading_the_same_checkpoint_twice_gives_the_same_lr_and_wd(self, tmp_path: Path) -> None:
        """Absolute restore: the second load must not add the saved count to the first one."""
        first = _new_scheduler(False)
        _train(first, _INCREMENTS[:6])
        saved = first.state_dict()
        resumed = _new_scheduler(False)
        _load(tmp_path, resumed, iteration=5, saved_state=saved)
        once = (resumed.num_steps, _groups(resumed))
        _load(tmp_path, resumed, iteration=5, saved_state=saved)
        assert (resumed.num_steps, _groups(resumed)) == once == (first.num_steps, _groups(first))


class TestTheObserverOnAFailedLoad:
    """A load that raises propagates; the scheduler keeps exactly the load_state_dict it had."""

    @pytest.mark.parametrize("restore_before_failing", [True, False])
    def test_the_class_method_comes_back(self, tmp_path: Path, restore_before_failing: bool) -> None:
        scheduler = _new_scheduler(True)
        saved = _new_scheduler(True).state_dict()

        def failing_load(model, optimizer, opt_param_scheduler, **kwargs):
            if restore_before_failing:
                opt_param_scheduler.load_state_dict(saved)
            raise OSError("checkpoint read failed")

        with pytest.raises(OSError, match="checkpoint read failed"):
            self._load_with(tmp_path, scheduler, failing_load)
        assert "load_state_dict" not in vars(scheduler)
        assert scheduler.load_state_dict.__func__ is type(scheduler).load_state_dict

    @pytest.mark.parametrize("load_fails", [True, False])
    def test_an_instance_override_is_put_back_not_deleted(self, tmp_path: Path, load_fails: bool) -> None:
        scheduler = _new_scheduler(True)
        calls = []
        class_method = scheduler.load_state_dict

        def override(state_dict):
            calls.append(state_dict["num_steps"])
            return class_method(state_dict)

        scheduler.load_state_dict = override
        first = _new_scheduler(True)
        _train(first, _INCREMENTS[:3])
        saved = first.state_dict()

        def load(model, optimizer, opt_param_scheduler, **kwargs):
            opt_param_scheduler.load_state_dict(saved)
            if load_fails:
                raise OSError("checkpoint read failed")
            return 2, True

        if load_fails:
            with pytest.raises(OSError, match="checkpoint read failed"):
                self._load_with(tmp_path, scheduler, load)
        else:
            self._load_with(tmp_path, scheduler, load)
            assert scheduler.num_steps == first.num_steps
        assert vars(scheduler)["load_state_dict"] is override
        assert calls == [first.num_steps]

    @staticmethod
    def _load_with(tmp_path: Path, scheduler, load_checkpoint):
        from miles.backends.megatron_utils.model import load_model_state

        load_dir = tmp_path / "ckpt"
        load_dir.mkdir(exist_ok=True)
        (load_dir / "latest_checkpointed_iteration.txt").write_text("2")
        with ExitStack() as stack:
            stack.enter_context(
                patch("miles.backends.megatron_utils.model.load_checkpoint", side_effect=load_checkpoint)
            )
            _patch_initialize_side_effects(stack)
            return load_model_state(
                make_trainer_args(use_checkpoint_opt_param_scheduler=True, global_batch_size=8, load=str(load_dir)),
                model=[_FakeModelChunk()],
                optimizer=None,
                opt_param_scheduler=scheduler,
                role="actor",
                checkpointing_context=None,
                resume_from_ckpt=True,
            )


class TestTheObserverCoversTheLoraRestore:
    """The observer scope includes restore_lora_adapter: a native LoRA training state restores the scheduler
    there, after the base load under --finetune restored nothing."""

    @staticmethod
    def _load_lora(tmp_path: Path, scheduler, *, adapter_iteration: int | None, saved_state: dict | None):
        from miles.backends.megatron_utils.model import load_model_state

        load_dir = tmp_path / "base"
        load_dir.mkdir(exist_ok=True)
        (load_dir / "latest_checkpointed_iteration.txt").write_text("release")

        def restore_lora_adapter(model, args, *, optimizer, opt_param_scheduler):
            if saved_state is not None:
                opt_param_scheduler.load_state_dict(saved_state)
            return adapter_iteration

        with ExitStack() as stack:
            stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(0, False)))
            stack.enter_context(
                patch("miles.backends.megatron_utils.model.restore_lora_adapter", side_effect=restore_lora_adapter)
            )
            _patch_initialize_side_effects(stack)
            return load_model_state(
                make_trainer_args(
                    use_checkpoint_opt_param_scheduler=True,
                    global_batch_size=8,
                    finetune=True,
                    lora_rank=8,
                    lora_adapter_path=str(tmp_path / "adapter"),
                    load=str(load_dir),
                ),
                model=[_FakeModelChunk()],
                optimizer=None,
                opt_param_scheduler=scheduler,
                role="actor",
                checkpointing_context=None,
                resume_from_ckpt=False,
            )

    @pytest.mark.parametrize("adapter_iteration", [0, 3])
    def test_a_native_training_state_with_scheduler_continues_the_schedule(
        self, tmp_path: Path, adapter_iteration: int
    ) -> None:
        uninterrupted = _train(_new_scheduler(True), _INCREMENTS)
        first = _new_scheduler(True)
        before = _train(first, _INCREMENTS[:4])
        resumed = _new_scheduler(True)
        output = self._load_lora(
            tmp_path, resumed, adapter_iteration=adapter_iteration, saved_state=first.state_dict()
        )
        assert output.start_rollout_id == adapter_iteration + 1
        assert before + _train(resumed, _INCREMENTS[4:]) == uninterrupted

    @pytest.mark.parametrize("adapter_iteration", [0, 3])
    def test_a_native_training_state_without_scheduler_is_refused(self, tmp_path: Path, adapter_iteration: int) -> None:
        """lora/utils.py returns the iteration even when training_state.opt_param_scheduler is None."""
        with pytest.raises(RuntimeError, match="no LR scheduler state"):
            self._load_lora(tmp_path, _new_scheduler(True), adapter_iteration=adapter_iteration, saved_state=None)

    def test_hf_adapter_weights_only_start_a_new_schedule(self, tmp_path: Path) -> None:
        """An HF/PEFT adapter has no iteration: a new stage at step 0, not a refusal."""
        scheduler = _new_scheduler(True)
        output = self._load_lora(tmp_path, scheduler, adapter_iteration=None, saved_state=None)
        assert output.start_rollout_id == 0
        assert scheduler.num_steps == 0
