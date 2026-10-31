import sys
import types
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from megatron.core.packed_seq_params import PackedSeqParams
from tests.fast.fixtures.args_fixtures import make_trainer_args, make_trainer_config

if TYPE_CHECKING:
    from miles.backends.megatron_utils.model import LoadCheckpointOutput

try:
    # Import the real scheduler before the module fixture below swaps megatron for stubs.
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler as _MegatronScheduler
except ImportError:
    _MegatronScheduler = None


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    if attrs is not None:
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


class _DummyDDP:
    pass


class _DummyModel:
    pass


class _DummyOptimizer:
    pass


class _DummyChainedOptimizer:
    pass


class _DummyDistributedOptimizer:
    pass


class _DummyScheduler:
    pass


class _DummyOptimizerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModelChunk:
    role: str | None = None


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_environment():
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron", is_package=True)
        core_module = _stub_module("megatron.core", is_package=True)
        core_module.mpu = types.SimpleNamespace()
        core_module.tensor_parallel = _stub_module(
            "megatron.core.tensor_parallel",
            {"model_parallel_cuda_manual_seed": MagicMock()},
            is_package=True,
        )
        _stub_module(
            "megatron.core.tensor_parallel.random",
            {"_get_all_rng_states": MagicMock(), "_set_all_rng_states": MagicMock()},
        )
        _stub_module(
            "megatron.core.distributed",
            {
                "DistributedDataParallel": _DummyDDP,
                "finalize_model_grads": MagicMock(),
            },
        )
        _stub_module(
            "megatron.core.enums",
            {"ModelType": types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")},
        )
        _stub_module("megatron.core.models", is_package=True)
        _stub_module("megatron.core.models.gpt", {"GPTModel": _DummyModel})
        _stub_module(
            "megatron.core.optimizer",
            {
                "OptimizerConfig": _DummyOptimizerConfig,
                "get_megatron_optimizer": MagicMock(),
                "Adam": _DummyOptimizer,
                "CPUAdam": _DummyOptimizer,
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.emerging_optimizers", {"TensorParallelMuon": _DummyOptimizer})
        _stub_module("megatron.core.optimizer.muon", {"get_megatron_muon_optimizer": MagicMock()})
        _stub_module("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": _DummyDistributedOptimizer})
        _stub_module(
            "megatron.core.optimizer.optimizer",
            {
                "ChainedOptimizer": _DummyChainedOptimizer,
                "MegatronOptimizer": _DummyOptimizer,
            },
        )
        _stub_module("megatron.core.optimizer_param_scheduler", {"OptimizerParamScheduler": _DummyScheduler})
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": PackedSeqParams})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock(), "unwrap_model": MagicMock()})
        _stub_module("megatron.core.config", {"set_experimental_flag": MagicMock()})
        _stub_module("megatron.core.num_microbatches_calculator", {"init_num_microbatches_calculator": MagicMock()})
        _stub_module("megatron.training", is_package=True)
        _stub_module(
            "megatron.training.global_vars",
            {
                "get_args": MagicMock(),
                "_build_tokenizer": MagicMock(),
                "set_args": MagicMock(),
            },
        )
        _stub_module("megatron.training.training", {"get_model": MagicMock()})
        _stub_module(
            "megatron.training.checkpointing",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
            },
        )
        _stub_module("sglang.srt.debug_utils", is_package=True)
        _stub_module(
            "sglang.srt.debug_utils.dumper",
            {
                "DumperConfig": MagicMock(),
                "_get_rank": MagicMock(return_value=0),
                "dumper": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.lora.bridge",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.model_provider",
            {
                "get_model_provider_func": MagicMock(),
                "LinearForLastLayer": _DummyModel,
            },
        )
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = make_trainer_config(
        use_checkpoint_opt_param_scheduler=True,
        global_batch_size=8,
        finetune=False,
        pretrained_checkpoint="/checkpoint",
    )
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.load_checkpoint", side_effect=_load_restoring_scheduler(100))
        )
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args, resume_from_ckpt=True)

    assert result == (
        model,
        optimizer,
        opt_param_scheduler,
        LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101),
    )
    opt_param_scheduler.load_state_dict.assert_called_once_with({"num_steps": 800})
    opt_param_scheduler.step.assert_not_called()


def _load_restoring_scheduler(iteration: int):
    def load(model, optimizer, opt_param_scheduler, **kwargs):
        opt_param_scheduler.load_state_dict({"num_steps": iteration * 8})
        return iteration, True

    return load


def test_initialize_does_not_step_a_restored_scheduler_even_without_use_checkpoint_flag():
    """use_checkpoint_opt_param_scheduler only decides which hyperparameters win; Megatron restores
    num_steps either way, so stepping again by iteration * GBS would double the LR progress."""
    from miles.backends.megatron_utils.model import LoadCheckpointOutput, initialize_model_and_optimizer

    args = make_trainer_config(
        use_checkpoint_opt_param_scheduler=False,
        global_batch_size=8,
        finetune=False,
        pretrained_checkpoint="/checkpoint",
    )
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.load_checkpoint", side_effect=_load_restoring_scheduler(100))
        )
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args, resume_from_ckpt=True)

    assert result[3] == LoadCheckpointOutput(loaded_rollout_id=100, start_rollout_id=101)
    opt_param_scheduler.step.assert_not_called()


def test_initialize_refuses_a_trained_iteration_without_scheduler_state():
    """--no-load-optim on a trained checkpoint leaves no record of the LR progress."""
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = make_trainer_config(
        use_checkpoint_opt_param_scheduler=False,
        global_batch_size=8,
        finetune=False,
        pretrained_checkpoint="/checkpoint",
    )
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, True)))
        _patch_initialize_side_effects(stack)
        with pytest.raises(RuntimeError, match="no LR scheduler state"):
            initialize_model_and_optimizer(args, resume_from_ckpt=True)

    opt_param_scheduler.step.assert_not_called()


def _load_model_state_with(
    *,
    tmp_path: Path,
    finetune: bool,
    iteration: int,
    restored_trained_iteration: bool | None = None,
    adapter_iteration: int | None = None,
    lora_adapter_path: str | None = None,
    write_tracker: bool = True,
    ckpt_step: int | None = None,
    exit_on_missing_checkpoint: bool = False,
    resume_from_ckpt: bool,
    megatron_to_hf_mode: str = "core",
) -> "LoadCheckpointOutput":
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir()
    if restored_trained_iteration is None:
        restored_trained_iteration = not finetune or iteration > 0
    if write_tracker:
        (load_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.load_checkpoint",
                return_value=(iteration, restored_trained_iteration),
            )
        )
        stack.enter_context(
            patch("miles.backends.megatron_utils.model.restore_lora_adapter", return_value=adapter_iteration)
        )
        _patch_initialize_side_effects(stack)
        return load_model_state(
            make_trainer_args(
                use_checkpoint_opt_param_scheduler=True,
                global_batch_size=8,
                finetune=finetune,
                lora_rank=0 if lora_adapter_path is None else 8,
                megatron_to_hf_mode=megatron_to_hf_mode,
                lora_adapter_path=lora_adapter_path,
                load=str(load_dir),
                ckpt_step=ckpt_step,
                exit_on_missing_checkpoint=exit_on_missing_checkpoint,
            ),
            model=[_FakeModelChunk()],
            optimizer=None,
            opt_param_scheduler=None,
            role="actor",
            checkpointing_context=None,
            resume_from_ckpt=resume_from_ckpt,
        )


class TestWhereALoadSaysTheRunStarts:
    def test_a_finetune_load_starts_the_run_at_rollout_zero(self, tmp_path: Path):
        """--finetune means there is no run to continue, so rollout 0 is still ahead rather than behind."""
        assert (
            _load_model_state_with(
                tmp_path=tmp_path, finetune=True, iteration=0, resume_from_ckpt=False
            ).start_rollout_id
            == 0
        )

    def test_a_resumed_load_starts_the_run_after_the_checkpoint_it_read(self, tmp_path: Path):
        """The checkpoint's own rollout is done, so the run continues at the next one."""
        assert (
            _load_model_state_with(
                tmp_path=tmp_path, finetune=False, iteration=100, resume_from_ckpt=True
            ).start_rollout_id
            == 101
        )

    def test_a_run_that_restored_the_iteration_zero_checkpoint_it_wrote_starts_at_one(self, tmp_path: Path):
        """A real resume from the very first checkpoint must not be read as a finetune that starts over."""
        output = _load_model_state_with(tmp_path=tmp_path, finetune=False, iteration=0, resume_from_ckpt=True)

        assert output.start_rollout_id == 1

    def test_reference_checkpoint_iteration_does_not_resume_a_new_run(self, tmp_path: Path) -> None:
        """Reference initialization retains the loaded iteration but starts new rollout state at zero."""
        output = _load_model_state_with(
            tmp_path=tmp_path, finetune=False, iteration=100, resume_from_ckpt=False, megatron_to_hf_mode="bridge"
        )

        assert output.loaded_rollout_id == 100
        assert output.start_rollout_id == 0

    def test_weight_initialization_with_a_trained_iteration_is_refused(self, tmp_path: Path):
        """Weight initialization cannot claim a nonzero trained iteration."""
        with pytest.raises(AssertionError, match="Weight initialization returned a trained iteration"):
            _load_model_state_with(
                tmp_path=tmp_path,
                finetune=True,
                iteration=100,
                restored_trained_iteration=False,
                resume_from_ckpt=False,
            )

    def test_explicit_reference_weights_still_start_at_zero(self, tmp_path: Path) -> None:
        """Selecting reference weights by step does not turn finetuning into resumed training."""
        output = _load_model_state_with(
            tmp_path=tmp_path,
            finetune=True,
            iteration=0,
            ckpt_step=7,
            write_tracker=False,
            resume_from_ckpt=False,
        )

        assert (output.loaded_rollout_id, output.start_rollout_id) == (0, 0)

    @pytest.mark.parametrize(
        "selector",
        [{"ckpt_step": 7}, {"exit_on_missing_checkpoint": True}],
        ids=["checkpoint-step", "exit-on-missing"],
    )
    def test_an_explicit_selection_enters_the_checkpoint_loader_without_a_tracker(self, tmp_path, selector):
        """Explicit selectors reach the loader even when no latest-checkpoint tracker exists."""
        output = _load_model_state_with(
            tmp_path=tmp_path,
            finetune=False,
            iteration=7,
            write_tracker=False,
            resume_from_ckpt=False,
            **selector,
        )

        assert (output.loaded_rollout_id, output.start_rollout_id) == (7, 8)


class TestALoraAdapterThatCarriesItsOwnIteration:
    """LoRA saves write no tracker, so a LoRA resume loads the base under --finetune (iteration 0, nothing
    trained) and the adapter checkpoint alone says whether, and where, training stopped."""

    def _start(self, tmp_path: Path, adapter_iteration: int | None) -> "LoadCheckpointOutput":
        return _load_model_state_with(
            tmp_path=tmp_path,
            finetune=True,
            iteration=0,
            adapter_iteration=adapter_iteration,
            lora_adapter_path=str(tmp_path / "adapter"),
            resume_from_ckpt=False,
        )

    def test_a_resume_at_iteration_zero_continues_after_it(self, tmp_path: Path) -> None:
        """An adapter's saved iteration zero is a trained iteration, not a missing training state."""
        assert self._start(tmp_path, adapter_iteration=0).start_rollout_id == 1

    def test_a_resume_continues_after_the_iteration_the_adapter_names(self, tmp_path: Path) -> None:
        output = self._start(tmp_path, adapter_iteration=100)

        assert (output.loaded_rollout_id, output.start_rollout_id) == (100, 101)

    def test_adapter_weights_without_training_state_start_at_rollout_zero(self, tmp_path: Path) -> None:
        """An HF adapter or shards saved without training state carry no iteration to continue from."""
        assert self._start(tmp_path, adapter_iteration=None).start_rollout_id == 0


class TestCheckpointSourceRequirement:
    def test_local_checkpoint_manager_allows_loading_without_disk_or_pretrained_path(self) -> None:
        """An in-memory checkpoint supplies a valid source even when no disk path was requested."""
        from miles.backends.megatron_utils.model import load_model_state

        args = make_trainer_config(load=None, pretrained_checkpoint=None, lora_rank=0)
        context = {"local_checkpoint_manager": object()}
        model = [_FakeModelChunk()]
        with ExitStack() as stack:
            load = stack.enter_context(
                patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(7, True))
            )
            _patch_initialize_side_effects(stack)
            result = load_model_state(
                args,
                model=model,
                optimizer=None,
                opt_param_scheduler=None,
                role="actor",
                checkpointing_context=context,
                resume_from_ckpt=True,
            )

        assert (result.loaded_rollout_id, result.start_rollout_id) == (7, 8)
        assert load.call_args.kwargs["checkpointing_context"] is context
        assert load.call_args.kwargs["args"] is args
        load.assert_called_once()

    @pytest.mark.parametrize("context", [None, {}, {"unrelated": True}])
    def test_loading_without_any_checkpoint_source_is_rejected(self, context: dict | None) -> None:
        """An empty or unrelated checkpoint context cannot manufacture a valid model-loading source."""
        from miles.backends.megatron_utils.model import load_model_state

        args = make_trainer_config(load=None, pretrained_checkpoint=None, lora_rank=0)
        with patch("miles.backends.megatron_utils.model.load_checkpoint") as load:
            with pytest.raises(AssertionError):
                load_model_state(
                    args,
                    model=[_FakeModelChunk()],
                    optimizer=None,
                    opt_param_scheduler=None,
                    role="actor",
                    checkpointing_context=context,
                    resume_from_ckpt=False,
                )

        load.assert_not_called()


_GLOBAL_BATCH_SIZE = 256
_OPTIMIZER_STEPS_PER_ROLLOUT = 2


def _megatron_scheduler(*, use_checkpoint_opt_param_scheduler: bool):
    optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.0, "default_config": True}])
    return _MegatronScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=1e-6,
        min_lr=0.0,
        lr_warmup_steps=10 * _GLOBAL_BATCH_SIZE,
        lr_decay_steps=400 * _GLOBAL_BATCH_SIZE,
        lr_decay_style="cosine",
        start_wd=0.0,
        end_wd=0.1,
        wd_incr_steps=400 * _GLOBAL_BATCH_SIZE,
        wd_incr_style="linear",
        use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=False,
    )


def _load_model_state_into(
    scheduler, *, tmp_path: Path, iteration: int, use_checkpoint_opt_param_scheduler: bool, load_checkpoint
) -> "LoadCheckpointOutput":
    from miles.backends.megatron_utils.model import load_model_state

    load_dir = tmp_path / "ckpt"
    load_dir.mkdir(exist_ok=True)
    (load_dir / "latest_checkpointed_iteration.txt").write_text(str(iteration))

    with ExitStack() as stack:
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", side_effect=load_checkpoint))
        _patch_initialize_side_effects(stack)
        return load_model_state(
            make_trainer_args(
                use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
                global_batch_size=_GLOBAL_BATCH_SIZE,
                finetune=False,
                lora_rank=0,
                megatron_to_hf_mode="core",
                lora_adapter_path=None,
                load=str(load_dir),
            ),
            model=[_FakeModelChunk()],
            optimizer=None,
            opt_param_scheduler=scheduler,
            role="actor",
            checkpointing_context=None,
            resume_from_ckpt=True,
        )


def _restoring_load_checkpoint(*, saved_state: dict, iteration: int):
    """A load that restores the scheduler as Megatron's load_checkpoint does without --no-load-optim."""

    def load_checkpoint(model, optimizer, opt_param_scheduler, **kwargs):
        opt_param_scheduler.load_state_dict(saved_state)
        return iteration, True

    return load_checkpoint


@pytest.mark.skipif(_MegatronScheduler is None, reason="needs Megatron-LM on PYTHONPATH, as the CPU CI has")
class TestTheSchedulerAfterALoad:
    @pytest.mark.parametrize("use_checkpoint_opt_param_scheduler", [False, True])
    @pytest.mark.parametrize("saved_rollout", [0, 4, 49, 149])
    def test_a_resume_continues_the_schedule_the_checkpoint_saved(
        self, tmp_path: Path, saved_rollout: int, use_checkpoint_opt_param_scheduler: bool
    ):
        """Megatron's load replays the saved num_steps whatever the flag says, so stepping again would skip ahead."""
        uninterrupted = _megatron_scheduler(use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler)
        for _ in range((saved_rollout + 1) * _OPTIMIZER_STEPS_PER_ROLLOUT):
            uninterrupted.step(increment=_GLOBAL_BATCH_SIZE)
        saved_state = uninterrupted.state_dict()

        resumed = _megatron_scheduler(use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler)
        output = _load_model_state_into(
            resumed,
            tmp_path=tmp_path,
            iteration=saved_rollout,
            use_checkpoint_opt_param_scheduler=use_checkpoint_opt_param_scheduler,
            load_checkpoint=_restoring_load_checkpoint(saved_state=saved_state, iteration=saved_rollout),
        )

        assert output.start_rollout_id == saved_rollout + 1
        assert resumed.num_steps == uninterrupted.num_steps
        assert resumed.optimizer.param_groups == uninterrupted.optimizer.param_groups

    def test_a_resume_from_a_checkpoint_saved_before_any_optimizer_step_stays_at_step_zero(self, tmp_path: Path):
        """fp16 loss scaling can skip every early optimizer step, so a later checkpoint may still hold num_steps 0."""
        resumed = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False)
        saved_state = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False).state_dict()

        _load_model_state_into(
            resumed,
            tmp_path=tmp_path,
            iteration=5,
            use_checkpoint_opt_param_scheduler=False,
            load_checkpoint=_restoring_load_checkpoint(saved_state=saved_state, iteration=5),
        )

        assert resumed.num_steps == 0

    def test_a_load_that_did_not_restore_the_scheduler_is_refused(self, tmp_path: Path):
        """Upstream #3964 rebuilt num_steps = iteration * GBS here. This branch keeps the strict contract
        (FIX-B-2 D): with --no-load-optim Megatron leaves the scheduler alone, the real LR progress (dynamic
        batches, skipped updates) is unknown, so the load is refused instead of guessed."""
        resumed = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False)

        with pytest.raises(RuntimeError, match="no LR scheduler state"):
            _load_model_state_into(
                resumed,
                tmp_path=tmp_path,
                iteration=99,
                use_checkpoint_opt_param_scheduler=False,
                load_checkpoint=lambda *args, **kwargs: (99, True),
            )

        assert resumed.num_steps == 0

    @pytest.mark.parametrize("load_fails", [False, True])
    def test_the_scheduler_gets_its_own_load_state_dict_back_after_the_load(self, tmp_path: Path, load_fails: bool):
        """Nothing of the restore record stays on the scheduler, also when the load raises."""
        scheduler = _megatron_scheduler(use_checkpoint_opt_param_scheduler=False)
        restoring_load_checkpoint = _restoring_load_checkpoint(saved_state=scheduler.state_dict(), iteration=0)

        def load_checkpoint(*args, **kwargs):
            result = restoring_load_checkpoint(*args, **kwargs)
            if load_fails:
                raise RuntimeError("the checkpoint read failed")
            return result

        with pytest.raises(RuntimeError, match="the checkpoint read failed") if load_fails else nullcontext():
            _load_model_state_into(
                scheduler,
                tmp_path=tmp_path,
                iteration=0,
                use_checkpoint_opt_param_scheduler=False,
                load_checkpoint=load_checkpoint,
            )

        assert "load_state_dict" not in vars(scheduler)
        assert scheduler.load_state_dict.__func__ is _MegatronScheduler.load_state_dict
