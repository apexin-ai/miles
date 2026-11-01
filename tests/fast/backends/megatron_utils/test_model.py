import logging
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from tests.fast.fixtures.args_fixtures import make_trainer_args, with_backend_values
from tests.fast.utils.test_utils.fault_injector.fakes import _arm_marker_hook

from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.model_companion import ModelCompanion
from miles.utils.test_utils.fault_injector.models import FaultHookName


class FakeModelChunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.zero_grad_buffer_count = 0
        self.add_module("model_companion", ModelCompanion(pipeline_rank=0, chunk_index=0, replica_id=(0, 0, 0)))

    def zero_grad_buffer(self) -> None:
        self.zero_grad_buffer_count += 1


class FakeMpu:
    def __init__(self, *, is_last_pipeline_stage: bool = False) -> None:
        self.is_last_pipeline_stage = is_last_pipeline_stage

    def is_pipeline_last_stage(self, ignore_virtual: bool = False) -> bool:
        return self.is_last_pipeline_stage


class FakeForwardBackwardEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return []


class FakeNonfiniteOptimizer:
    def zero_grad(self) -> None:
        pass

    def prepare_grads(self) -> bool:
        return True


class FakeFiniteOptimizer(FakeNonfiniteOptimizer):
    def __init__(self) -> None:
        self.step_count = 0

    def prepare_grads(self) -> bool:
        return False

    def get_grad_norm(self) -> float:
        return 1.0

    def step(self) -> tuple[bool, float, int]:
        self.step_count += 1
        return True, 1.0, 0


class FakeScheduler:
    def step(self, *, increment: int) -> None:
        pass


@dataclass
class FakeParallelGroup:
    size: int = 1
    rank: int = 0


@dataclass
class FakeParallelState:
    indep_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)
    effective_dp: FakeParallelGroup = field(default_factory=FakeParallelGroup)


@dataclass
class TrainOneStepEnv:
    args: Namespace
    model: list[FakeModelChunk]
    data_iterator: list[DataIterator]
    parallel_state: FakeParallelState
    forward_backward_engine: FakeForwardBackwardEngine


def make_train_one_step_args(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = dict(
        debug_disable_optimizer=False,
        multi_lora=False,
        enable_sample_ownership_checker=True,
        custom_megatron_before_train_step_hook_path=None,
        dumper_enable=False,
        dumper_fwd_bwd=[],
        seq_length=8,
        decoder_seq_length=8,
        micro_batch_size=1,
        check_for_nan_in_loss_and_grad=True,
        calculate_per_token_loss=False,
        ci_test=False,
        enable_mtp_training=False,
        rollout_max_response_len=512,
        use_sampling_support_replay=False,
        enable_witness=False,
        save_local_weight_checksum=False,
    )
    return make_trainer_args(**{**defaults, **overrides})


@pytest.fixture
def train_one_step_env(monkeypatch) -> TrainOneStepEnv:
    from miles.backends.megatron_utils import model as model_module

    env = TrainOneStepEnv(
        args=make_train_one_step_args(),
        model=[FakeModelChunk()],
        data_iterator=[
            DataIterator(
                {
                    "lineage_source_sample_indices": [],
                    "lineage_output_indices": [],
                    "lineage_output_counts": [],
                },
                micro_batch_size=1,
            )
        ],
        parallel_state=FakeParallelState(),
        forward_backward_engine=FakeForwardBackwardEngine(),
    )

    monkeypatch.setattr(model_module, "get_parallel_state", lambda: env.parallel_state)
    monkeypatch.setattr("miles.backends.training_utils.parallel.get_parallel_state", lambda: env.parallel_state)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: env.forward_backward_engine)
    monkeypatch.setattr(model_module, "mpu", FakeMpu())

    return env


class TestTrainOneStepStructuredLog:
    def test_train_one_step_emits_the_train_tag_in_its_structured_event(
        self, train_one_step_env: TrainOneStepEnv, caplog
    ):
        """Log consumers key train-step events off the train tag, so the caller must emit that tag with its fields."""
        from miles.backends.megatron_utils.model import train_one_step

        with caplog.at_level(logging.INFO, logger="miles.backends.megatron_utils.model"):
            train_one_step(
                args=train_one_step_env.args,
                rollout_id=7,
                step_id=3,
                data_iterator=train_one_step_env.data_iterator,
                model=train_one_step_env.model,
                optimizer=None,
                opt_param_scheduler=None,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=2,
            )

        assert "train op=train_step rollout=7 step=3 attempt=2 outcome=NORMAL valid_step=true" in caplog.messages


class TestMtpGradientGate:
    @pytest.mark.parametrize(
        ("detach", "response_length", "checked"), [(True, 128, True), (False, 128, False), (True, 129, False)]
    )
    def test_non_mtp_gradients_are_checked_only_for_detached_truncated_training(
        self, train_one_step_env: TrainOneStepEnv, detach: bool, response_length: int, checked: bool
    ) -> None:
        """The backend detach flag preserves the real gradient check and its truncation boundary."""
        from miles.backends.megatron_utils.model import train_one_step

        env = train_one_step_env
        env.args = make_train_one_step_args(
            ci_test=True, enable_mtp_training=True, mtp_detach_heads=detach, rollout_max_response_len=response_length
        )
        assert "mtp_detach_heads" not in vars(env.args)
        parameter = torch.nn.Parameter(torch.ones(1))
        parameter.grad = torch.ones(1)
        env.model[0].register_parameter("non_mtp_weight", parameter)

        with pytest.raises(AssertionError, match="Expected all non-MTP parameters") if checked else nullcontext():
            train_one_step(
                args=env.args,
                rollout_id=7,
                step_id=3,
                data_iterator=env.data_iterator,
                model=env.model,
                optimizer=None,
                opt_param_scheduler=None,
                num_microbatches=1,
                num_rollouts=1,
                witness_info=None,
                attempt=0,
            )


class TestTrainOneStepFaultHook:
    @pytest.fixture
    def allreduce_log(self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch) -> list[object]:
        from miles.backends.megatron_utils import model as model_module

        log: list[object] = []
        train_one_step_env.args.enable_sample_ownership_checker = False
        train_one_step_env.parallel_state.indep_dp = FakeParallelGroup(size=2)

        def allreduce(*args: Any, **kwargs: Any) -> tuple[bool, dict[str, float]]:
            log.append("allreduce")
            return True, {}

        monkeypatch.setattr(model_module, "allreduce_grads_and_losses_across_replicas", allreduce)
        return log

    def _train(self, env: TrainOneStepEnv) -> None:
        from miles.backends.megatron_utils.model import train_one_step

        train_one_step(
            args=env.args,
            rollout_id=7,
            step_id=0,
            data_iterator=env.data_iterator,
            model=env.model,
            optimizer=None,
            opt_param_scheduler=None,
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

    def test_the_hook_of_this_step_fires_before_the_cross_replica_allreduce(
        self, train_one_step_env: TrainOneStepEnv, allreduce_log: list[object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fault armed for this rollout and attempt must strike before replicas exchange gradients."""
        _arm_marker_hook(
            monkeypatch,
            log=allreduce_log,
            hook_name=FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE,
            rollout_id=7,
            attempt=2,
        )
        self._train(train_one_step_env)
        assert allreduce_log == [("hook", FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE.value), "allreduce"]

    @pytest.mark.parametrize("filters", [{"rollout_id": 6}, {"attempt": 1}])
    def test_a_hook_armed_for_another_step_or_attempt_does_not_fire(
        self,
        train_one_step_env: TrainOneStepEnv,
        allreduce_log: list[object],
        monkeypatch: pytest.MonkeyPatch,
        filters: dict[str, int],
    ) -> None:
        """The hook must be reached with this step's rollout and attempt so other steps are spared."""
        _arm_marker_hook(
            monkeypatch, log=allreduce_log, hook_name=FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE, **filters
        )
        self._train(train_one_step_env)
        assert allreduce_log == ["allreduce"]

    def test_a_failing_hook_stops_the_step_before_the_allreduce(
        self, train_one_step_env: TrainOneStepEnv, allreduce_log: list[object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hook failure must surface instead of letting the gradients be exchanged."""
        _arm_marker_hook(
            monkeypatch, log=allreduce_log, hook_name=FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE, fail=True
        )
        with pytest.raises(RuntimeError, match="failed"):
            self._train(train_one_step_env)
        assert allreduce_log == [("hook", FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE.value)]

    def test_a_single_replica_step_never_reaches_the_hook(
        self, train_one_step_env: TrainOneStepEnv, allreduce_log: list[object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without independent replicas there is no allreduce, so the hook must not fire."""
        train_one_step_env.parallel_state.indep_dp = FakeParallelGroup(size=1)
        _arm_marker_hook(monkeypatch, log=allreduce_log, hook_name=FaultHookName.TRAINER_STEP_BEFORE_ALLREDUCE)
        self._train(train_one_step_env)
        assert allreduce_log == []


def test_forward_only_omits_sampling_mask_for_callbacks_that_do_not_replay_sampling_support(monkeypatch):
    from miles.backends.megatron_utils import model as model_module

    def forward_backward(**kwargs):
        output, callback = kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])
        return [callback(output)]

    callback_kwargs = {}

    def critic_callback(_output, **kwargs):
        callback_kwargs.update(kwargs)
        return {}

    args = make_trainer_args(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        enable_witness=False,
        use_rollout_entropy=False,
        custom_megatron_before_log_prob_hook_path=None,
        seq_length=8,
        micro_batch_size=1,
    )
    batch = {
        "tokens": None,
        "input_loss_masks": None,
        "multimodal_train_inputs": None,
        "unconcat_tokens": [],
        "total_lengths": [],
        "response_lengths": [],
        "max_seq_lens": None,
    }
    dumper = Mock()
    dumper.wrap_forward_step.side_effect = lambda fn: fn

    monkeypatch.setattr(model_module, "DumperMegatronUtil", lambda *_args, **_kwargs: dumper)
    monkeypatch.setattr(model_module, "get_model_config", lambda _model: Mock(timers=None))
    monkeypatch.setattr(model_module, "get_batch", lambda *_args, **_kwargs: batch)
    monkeypatch.setattr(model_module, "get_packed_seq_params", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward)
    monkeypatch.setattr(model_module, "mpu", FakeMpu())

    model_module.forward_only(
        critic_callback,
        args,
        [Mock()],
        [Mock()],
        [1],
        rollout_id=0,
    )

    assert "rollout_sampling_mask" not in callback_kwargs


@pytest.mark.parametrize(
    ("loss_type", "fetched"),
    [("policy_loss", True), ("score_centering", True), ("custom_loss", True), ("sft_loss", False)],
)
def test_sampling_support_replay_fetches_sampling_masks_for_losses_that_score_the_support(
    monkeypatch, loss_type: str, fetched: bool
):
    """A custom loss reads the replayed sampling support like the built-in policy losses."""
    from miles.backends.megatron_utils import model as model_module

    class BatchFetched(Exception):
        pass

    def get_batch(_data_iterator, keys, *_args, **_kwargs):
        raise BatchFetched(keys)

    def forward_backward(**kwargs):
        kwargs["forward_step_func"](kwargs["data_iterator"][0], kwargs["model"][0])

    args = make_trainer_args(
        loss_type=loss_type,
        use_sampling_support_replay=True,
        data_pad_size_multiplier=1,
        qkv_format="thd",
        allgather_cp=False,
        seq_length=8,
        decoder_seq_length=8,
        micro_batch_size=1,
    )
    dumper = Mock()
    dumper.wrap_forward_step.side_effect = lambda fn: fn
    monkeypatch.setattr(model_module, "get_batch", get_batch)
    monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward)

    with pytest.raises(BatchFetched) as fetched_keys:
        model_module.run_forward_backward_pass(args, dumper, [Mock()], [Mock()], 1, 1)

    keys = fetched_keys.value.args[0]
    assert ("rollout_sampling_mask_ids" in keys) is fetched
    assert ("rollout_sampling_mask_offsets" in keys) is fetched


class TestTrainOneStepModelCompanion:
    def test_disabled_check_skips_witness_identity_collection(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordinary training needs neither a companion nor sample identity fields."""
        from miles.backends.megatron_utils import model as model_module

        train_one_step_env.args.enable_sample_ownership_checker = False
        del train_one_step_env.model[0].model_companion
        train_one_step_env.data_iterator[0].rollout_data.clear()
        optimizer = FakeFiniteOptimizer()

        def reject_witness_work(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Witness collection must be disabled")

        monkeypatch.setattr(
            model_module.SampleIdentityExtractor, "get_consumed_sample_identities", reject_witness_work
        )
        monkeypatch.setattr(model_module.SampleIdentityExtractor, "gather_sample_identities", reject_witness_work)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=3,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=optimizer,
            opt_param_scheduler=FakeScheduler(),
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert outcome is model_module.TrainStepOutcome.NORMAL
        assert optimizer.step_count == 1

    def test_successful_step_records_consumed_rows_as_trained(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful optimizer update records each consumed row as trained."""
        from miles.backends.megatron_utils import model as model_module

        identity_fields = train_one_step_env.data_iterator[0].rollout_data
        identity_fields["lineage_source_sample_indices"] = [7]
        identity_fields["lineage_output_indices"] = [0]
        identity_fields["lineage_output_counts"] = [1]
        train_one_step_env.args = with_backend_values(train_one_step_env.args, check_for_nan_in_loss_and_grad=False)

        def forward_backward_engine(**kwargs: Any) -> list[dict[str, Any]]:
            kwargs["data_iterator"][0].offset = 1
            return []

        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=3,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=FakeFiniteOptimizer(),
            opt_param_scheduler=FakeScheduler(),
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert outcome is model_module.TrainStepOutcome.NORMAL
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {
            model_module.SampleLineage(source_sample_index=7, output_index=0, output_count=1): 1
        }
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=True) == {}
        assert train_one_step_env.model[0].model_companion.weight_version.item() == 1

    def test_nonfinite_step_records_consumed_rows_as_skipped(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A nonfinite optimizer skip records each consumed row without marking it trained."""
        from miles.backends.megatron_utils import model as model_module

        identity_fields = train_one_step_env.data_iterator[0].rollout_data
        identity_fields["lineage_source_sample_indices"] = [7]
        identity_fields["lineage_output_indices"] = [1]
        identity_fields["lineage_output_counts"] = [2]
        train_one_step_env.args = with_backend_values(train_one_step_env.args, check_for_nan_in_loss_and_grad=False)
        train_one_step_env.forward_backward_engine = FakeForwardBackwardEngine()

        def forward_backward_engine(**kwargs: Any) -> list[dict[str, Any]]:
            kwargs["data_iterator"][0].offset = 1
            return []

        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=3,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=FakeNonfiniteOptimizer(),
            opt_param_scheduler=None,
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert train_one_step_env.model[0].model_companion.weight_version.item() == 0
        assert outcome is model_module.TrainStepOutcome.NORMAL
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {}
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=True) == {
            model_module.SampleLineage(source_sample_index=7, output_index=1, output_count=2): 1
        }

    def test_ft_discard_does_not_record_consumed_rows_as_skipped(
        self, train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A discarded FT attempt leaves both successful outcome counters unchanged."""
        from miles.backends.megatron_utils import model as model_module

        identity_fields = train_one_step_env.data_iterator[0].rollout_data
        identity_fields["lineage_source_sample_indices"] = [7]
        identity_fields["lineage_output_indices"] = [0]
        identity_fields["lineage_output_counts"] = [1]
        train_one_step_env.parallel_state.indep_dp.size = 2
        train_one_step_env.args = with_backend_values(train_one_step_env.args, check_for_nan_in_loss_and_grad=False)

        def forward_backward_engine(**kwargs: Any) -> list[dict[str, Any]]:
            kwargs["data_iterator"][0].offset = 1
            return []

        def discard_attempt(*_args: Any, collect_training_metadata, **_kwargs: Any) -> tuple[bool, dict]:
            collect_training_metadata()
            return False, {}

        optimizer = FakeFiniteOptimizer()
        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)
        monkeypatch.setattr(model_module, "allreduce_grads_and_losses_across_replicas", discard_attempt)

        _, _, outcome = model_module.train_one_step(
            args=train_one_step_env.args,
            rollout_id=7,
            step_id=0,
            data_iterator=train_one_step_env.data_iterator,
            model=train_one_step_env.model,
            optimizer=optimizer,
            opt_param_scheduler=None,
            num_microbatches=1,
            num_rollouts=1,
            witness_info=None,
            attempt=2,
        )

        assert outcome is model_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY
        assert optimizer.step_count == 0
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=False) == {}
        assert train_one_step_env.model[0].model_companion.snapshot_sample_consumptions(is_skipped=True) == {}


@pytest.mark.parametrize("tensor_norm", [False, True])
def test_ft_discard_stays_invalid_with_finite_gradient_norm(
    train_one_step_env: TrainOneStepEnv, monkeypatch: pytest.MonkeyPatch, tensor_norm: bool
) -> None:
    """A finite gradient norm cannot revive a failed collective and update weights."""
    from miles.backends.megatron_utils import model as model_module

    env = train_one_step_env
    env.args.enable_sample_ownership_checker = False
    env.args = with_backend_values(env.args, check_for_nan_in_loss_and_grad=False)
    env.args = with_backend_values(env.args, calculate_per_token_loss=False)
    env.parallel_state.indep_dp.size = 2

    def reject_optimizer_step() -> None:
        raise AssertionError("A discarded step must not update weights")

    optimizer = SimpleNamespace(
        zero_grad=lambda: None,
        prepare_grads=lambda: False,
        get_grad_norm=lambda: torch.tensor(1.0) if tensor_norm else 1.0,
        step=reject_optimizer_step,
    )
    monkeypatch.setattr(
        model_module, "allreduce_grads_and_losses_across_replicas", lambda *args, **kwargs: (False, {})
    )

    _, _, outcome = model_module.train_one_step(
        args=env.args,
        rollout_id=7,
        step_id=0,
        data_iterator=env.data_iterator,
        model=env.model,
        optimizer=optimizer,
        opt_param_scheduler=None,
        num_microbatches=1,
        num_rollouts=1,
        witness_info=None,
        attempt=2,
    )

    assert outcome is model_module.TrainStepOutcome.DISCARDED_SHOULD_RETRY


class TorchBackedOptimizer:
    """The MegatronOptimizer calls of train_one_step, backed by a real AdamW whose weight decay moves weights."""

    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.1)
        self.step_count = 0

    @property
    def param_groups(self) -> list[dict]:
        return self.optimizer.param_groups

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def prepare_grads(self) -> bool:
        return False

    def get_grad_norm(self) -> float:
        return self.parameter.grad.norm().item()

    def step(self) -> tuple[bool, float, int]:
        grad_norm = self.get_grad_norm()
        self.optimizer.step()
        self.step_count += 1
        return True, grad_norm, 0


class TestUnsupervisedStep:
    @pytest.mark.parametrize(
        ("per_token", "num_tokens", "grad", "updates"),
        [
            (True, 0, 0.0, False),
            # e.g. a MoE aux loss: gradients without any supervised token are skipped too
            (True, 0, 1.0, False),
            (True, 5, 1.0, True),
            # sample-mean losses carry no token count, so their steps are untouched
            (False, None, 0.0, True),
        ],
    )
    def test_a_step_without_supervised_tokens_skips_the_optimizer_and_the_scheduler(
        self,
        train_one_step_env: TrainOneStepEnv,
        monkeypatch: pytest.MonkeyPatch,
        per_token: bool,
        num_tokens: int | None,
        grad: float,
        updates: bool,
    ) -> None:
        """Megatron clamps a zero global token count to one; the step must still leave weights and LR untouched."""
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        from miles.backends.megatron_utils import model as model_module

        env = train_one_step_env
        env.args.enable_sample_ownership_checker = False
        env.args = with_backend_values(
            env.args, check_for_nan_in_loss_and_grad=False, calculate_per_token_loss=per_token
        )
        parameter = torch.nn.Parameter(torch.ones(3))
        env.model[0].register_parameter("weight", parameter)
        optimizer = TorchBackedOptimizer(parameter)
        scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=0.0,
            max_lr=0.1,
            min_lr=0.0,
            lr_warmup_steps=4,
            lr_decay_steps=16,
            lr_decay_style="linear",
            start_wd=0.1,
            end_wd=0.1,
            wd_incr_steps=16,
            wd_incr_style="constant",
        )
        scheduler.step(increment=2)
        weight_before = parameter.detach().clone()
        lr_before = optimizer.param_groups[0]["lr"]

        def finalize_model_grads(model, num_tokens=None, pg_collection=None, force_all_reduce=False):
            # one rank: the PP broadcast and DP-CP all-reduce leave the count as it is
            if num_tokens is not None:
                parameter.grad *= 1.0 / torch.clamp(num_tokens, min=1)

        def forward_backward_engine(**kwargs: Any) -> list[dict[str, Any]]:
            parameter.grad = torch.full_like(parameter, grad)
            count = torch.tensor(num_tokens, dtype=torch.int) if per_token else None
            # the wrapper's `finalize=` default binds Megatron's function at import, so inject the fake the way
            # setup_train_iteration_config does (functools.partial) instead of patching the module global
            model_module.finalize_model_grads_with_empty_cache(
                kwargs["model"], count, pg_collection=None, force_all_reduce=False, finalize=finalize_model_grads
            )
            return []

        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)
        monkeypatch.setattr(model_module, "mpu", FakeMpu(is_last_pipeline_stage=True))
        monkeypatch.setattr(model_module, "aggregate_train_losses", lambda losses_reduced, num_rollouts: {})
        monkeypatch.setattr(model_module, "is_first_replica_megatron_main_rank", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))

        loss_dict, _, outcome = model_module.train_one_step(
            args=env.args,
            rollout_id=7,
            step_id=0,
            data_iterator=env.data_iterator,
            model=env.model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            num_microbatches=1,
            num_rollouts=4,
            witness_info=None,
            attempt=0,
        )

        assert outcome is model_module.TrainStepOutcome.NORMAL
        assert parameter.grad is None
        if updates:
            assert optimizer.step_count == 1
            assert scheduler.num_steps == 6
            assert not torch.equal(parameter.detach(), weight_before)
            assert env.model[0].model_companion.weight_version.item() == 1
            assert "skipped_unsupervised_step" not in loss_dict
        else:
            assert optimizer.step_count == 0
            assert scheduler.num_steps == 2
            assert optimizer.param_groups[0]["lr"] == lr_before
            assert torch.equal(parameter.detach(), weight_before)
            assert env.model[0].model_companion.weight_version.item() == 0
            assert loss_dict == {"skipped_unsupervised_step": 1.0}


class TestFinalizeInjection:
    """setup_train_iteration_config injects the finalize (Megatron core, or Bridge's expert-adapter sync for
    Bridge LoRA) into finalize_model_grads_with_empty_cache; the step-skip decision needs the token count that
    finalize reduced in place."""

    @pytest.fixture
    def finalize_env(self, monkeypatch: pytest.MonkeyPatch):
        from miles.backends.megatron_utils import model as model_module

        monkeypatch.setattr(model_module, "_step_num_tokens", None)
        monkeypatch.setattr(model_module, "get_model_config", lambda chunk: SimpleNamespace())
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))
        return model_module

    @staticmethod
    def _recording_finalize(calls: list):
        def finalize(model, num_tokens=None, pg_collection=None, **kwargs):
            calls.append((model, num_tokens, pg_collection, kwargs))
            if num_tokens is not None:
                # what finalize_model_grads does: PP broadcast + DP-CP all-reduce, in place (here: two DP ranks)
                num_tokens.mul_(2)

        return finalize

    @pytest.mark.parametrize("variant", ["megatron_core", "bridge_lora", "multi_lora"])
    @pytest.mark.parametrize("local_tokens", [5, 0, None])
    def test_the_injected_finalize_gets_num_tokens_and_the_wrapper_records_the_global_count(
        self, finalize_env, monkeypatch: pytest.MonkeyPatch, variant: str, local_tokens: int | None
    ) -> None:
        model_module = finalize_env
        calls: list = []
        recorder = self._recording_finalize(calls)
        if variant == "megatron_core":
            monkeypatch.setattr(model_module, "finalize_model_grads", recorder)
        else:
            bridge_utils = pytest.importorskip("megatron.bridge.peft.utils")
            monkeypatch.setattr(bridge_utils, "finalize_model_grads_with_expert_adapter_sync", recorder)
        monkeypatch.setattr(model_module, "is_multi_lora_enabled", lambda args: variant == "multi_lora")
        monkeypatch.setattr(model_module, "is_lora_enabled", lambda args: variant == "bridge_lora")
        args = SimpleNamespace(
            megatron_to_hf_mode="bridge",
            backend=SimpleNamespace(overlap_grad_reduce=False, overlap_param_gather=False, align_param_gather=False),
        )
        model = [torch.nn.Linear(2, 2)]

        config = model_module.setup_train_iteration_config(args, model, Mock(), disable_optimizer=False)
        num_tokens = None if local_tokens is None else torch.tensor(local_tokens, dtype=torch.int)
        # the call megatron/core/pipeline_parallel/schedules.py makes
        config.finalize_model_grads_func(model, num_tokens, pg_collection=None, force_all_reduce=False)

        assert len(calls) == 1
        called_model, called_tokens, called_pg, called_kwargs = calls[0]
        assert called_model is model
        assert called_tokens is num_tokens
        assert called_pg is None and called_kwargs == {"force_all_reduce": False}
        if local_tokens is None:
            assert model_module._step_num_tokens is None
        else:
            assert model_module._step_num_tokens is num_tokens
            assert model_module._step_num_tokens.item() == 2 * local_tokens

    def test_bridge_expert_adapter_finalize_takes_num_tokens_positionally(self) -> None:
        """The wrapper calls finalize(model, num_tokens, ...); the installed Bridge function must bind that."""
        import inspect

        bridge_utils = pytest.importorskip("megatron.bridge.peft.utils")
        signature = inspect.signature(bridge_utils.finalize_model_grads_with_expert_adapter_sync)
        bound = signature.bind([torch.nn.Linear(2, 2)], torch.tensor(3), pg_collection=None, force_all_reduce=False)

        assert list(signature.parameters)[:2] == ["model", "num_tokens"]
        assert bound.arguments["num_tokens"].item() == 3


class TestLowMemoryClearLog:
    """The in-step clear under 10% free stays; with --no-clear-memory-every-step it must be visible in the log,
    because firing on most steps would spend the time the flag saves."""

    def test_clears_under_ten_percent_free_and_logs_the_1st_2nd_4th_clear(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from miles.backends.megatron_utils import model as model_module

        free = {"bytes": 5 * 2**30}
        clears: list[int] = []
        monkeypatch.setattr(model_module, "_low_memory_clears", 0)
        monkeypatch.setattr(model_module, "clear_memory", lambda: clears.append(free["bytes"]))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free["bytes"], 100 * 2**30))
        finalize = Mock()

        with caplog.at_level(logging.INFO, logger=model_module.logger.name):
            for _ in range(5):
                model_module.finalize_model_grads_with_empty_cache([], None, finalize=finalize)
            free["bytes"] = 50 * 2**30
            model_module.finalize_model_grads_with_empty_cache([], None, finalize=finalize)

        assert len(clears) == 5
        assert finalize.call_count == 6
        logged = [r.getMessage() for r in caplog.records if "low-memory clear" in r.getMessage()]
        assert [m.rsplit("#", 1)[1] for m in logged] == ["1 on this rank)", "2 on this rank)", "4 on this rank)"]
        assert "free 5.0 of 100.0 GiB" in logged[0]
