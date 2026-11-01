from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import train as train_driver
from tests.fast.fixtures.driver_fakes import (
    FakeInferenceController,
    FakeObjectStore,
    FakeRemoteMethod,
    FakeRolloutExecutor,
    FakeTrainingModel,
    FakeWorkerManager,
)

from miles.backends.megatron_utils.megatron_config import MegatronConfig, MegatronTrainerConfig
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.ray import placement_group, wiring
from miles.utils import object_store
from miles.utils.async_utils import with_disposer
from miles.utils.orchestration_utils import ArgvOrchestratorStartupInfo


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        clear_memory_every_step=True,
        colocate_memory_peak_device="cpu",
        debug_exit_after_rollout=None,
        eval_interval=None,
        eval_uses_snapshots=False,
        ft_components=[],
        fully_async=False,
        hf_checkpoint="/base/checkpoint",
        num_critic_only_steps=0,
        num_rollout=0,
        offload_rollout=False,
        offload_rollout_level="",
        offload_train=False,
        prefetch_rollout=0,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=False,
        start_rollout_id=0,
        use_critic=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    args.raw_megatron = MegatronConfig(
        trainers=[
            MegatronTrainerConfig(
                trainer_id=role,
                model_id=None,
                role=role,
                actor_index=0 if role == "actor" else None,
                overrides={},
            )
            for role in (["actor", "critic"] if args.use_critic else ["actor"])
        ],
        base_args={},
    )
    return args


def _startup_info(args: SimpleNamespace) -> ArgvOrchestratorStartupInfo:
    return ArgvOrchestratorStartupInfo(args=args, all_args=args)


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch,
    args: SimpleNamespace,
    events: list[str],
    *,
    num_rollout_per_epoch: int = 4,
) -> SimpleNamespace:
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
        cell_operations=object(),
    )
    capability = SimpleNamespace(cell_operations=lambda: components.cell_operations)

    handles = {"actor": components.actor_model}
    if args.use_critic:
        handles["critic"] = components.critic_model

    def create_trainer_handles(
        _args: SimpleNamespace, *, trainer_configs: list[MegatronTrainerConfig], capability: Any
    ) -> dict[str, FakeTrainingModel]:
        assert [config.trainer_id for config in trainer_configs] == list(handles)
        return handles

    async def create_rollout_components(
        _args: SimpleNamespace, *, capability: Any, trainer_handles: dict[str, FakeTrainingModel]
    ) -> tuple[Any, Any, int]:
        assert trainer_handles is handles
        return components.inference_controller, components.rollout_executor, num_rollout_per_epoch

    async def create_training_models(
        _args: SimpleNamespace, _executor: Any, *, handles: dict[str, FakeTrainingModel]
    ) -> tuple[Any, Any]:
        assert handles["actor"] is components.actor_model
        if args.use_critic:
            assert handles["critic"] is components.critic_model
        return components.actor_model, components.critic_model

    async def update_weights(
        _args: Any, _model: Any, _executor: Any, _inference_controller: Any, *, rollout_id: int | None = None
    ) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_driver, "init_orchestration_script", lambda _startup_info, *, disposer: capability)
    monkeypatch.setattr(train_driver, "create_trainer_handles", create_trainer_handles)
    monkeypatch.setattr(train_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    return components


class TestEvalOnlyRun:
    async def test_eval_only_prepares_inference_and_runs_exactly_one_eval(self, monkeypatch: pytest.MonkeyPatch):
        """A run with no rollouts but an eval interval evaluates once and generates or trains nothing."""
        events: list[str] = []
        args = _make_args(num_rollout=0, eval_interval=2)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert events.count("prepare_eval") == 1
        assert events.count("eval:0") == 1
        assert events.index("prepare_eval") < events.index("eval:0")
        assert components.actor_model.trained == []
        assert not [event for event in events if event.startswith(("prepare_rollout", "generate_start"))]


class TestEvalBeforeTrain:
    async def test_fresh_run_evaluates_the_initial_checkpoint_at_step_zero(self, monkeypatch: pytest.MonkeyPatch):
        events: list[str] = []
        args = _make_args(num_rollout=2, eval_interval=1)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert events.index("eval:0") < events.index("prepare_rollout:0")

    async def test_resumed_run_evaluates_the_completed_rollout_not_the_next_one(self, monkeypatch: pytest.MonkeyPatch):
        """start_rollout_id is the loaded checkpoint's rollout plus one, so dispatching it
        would ask for a checkpoint that does not exist yet and, in staging mode, export
        into the directory the first resumed iteration goes on to overwrite."""
        events: list[str] = []
        args = _make_args(num_rollout=5, eval_interval=1, start_rollout_id=3)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert events.index("eval:2") < events.index("prepare_rollout:3")
        assert "eval:3" not in events[: events.index("prepare_rollout:3")]


class TestFinalEval:
    async def test_the_last_rollout_is_always_evaluated_even_off_cadence(self, monkeypatch: pytest.MonkeyPatch):
        """The final point carries force=True because training is over and backpressure is
        free, but the cadence check has to reach it first: with num_rollout not a multiple
        of eval_interval, the final weights would otherwise never be measured."""
        events: list[str] = []
        args = _make_args(num_rollout=3, eval_interval=2)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert [event for event in events if event.startswith("eval:")] == ["eval:0", "eval:1", "eval:2"]

    async def test_an_off_cadence_final_eval_follows_the_last_handoff(self, monkeypatch: pytest.MonkeyPatch):
        """The last rollout skips its handoff only when no final eval follows; an off-cadence
        final eval must still see the engine onloaded with the trained weights."""
        events: list[str] = []
        args = _make_args(num_rollout=3, eval_interval=2, offload_rollout=True)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        final = events[events.index("actor_train:2") : events.index("eval:2")]
        assert final.index("onload_weights") < final.index("update_weights:2") < final.index("onload_kv")

    async def test_the_last_rollout_skips_the_handoff_without_an_eval(self, monkeypatch: pytest.MonkeyPatch):
        events: list[str] = []
        args = _make_args(num_rollout=3, offload_rollout=True)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert "update_weights:1" in events
        assert "update_weights:2" not in events
        assert "onload_weights" not in events[events.index("actor_train:2") :]


class TestWeightEqualityCheck:
    async def test_weight_equality_check_is_routed_to_the_inference_controller(self, monkeypatch: pytest.MonkeyPatch):
        """--check-weight-update-equal must reach the inference controller with every comparison option intact."""
        events: list[str] = []
        args = _make_args(
            check_weight_update_equal=True,
            check_weight_update_allow_quant_error=True,
            check_weight_update_selector="layers.0",
            check_weight_update_skip_list=["lm_head", "embed_tokens"],
        )
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert components.inference_controller.check_weights_calls == [
            dict(
                action="compare",
                allow_quant_error=True,
                selector="layers.0",
                skip_list=["lm_head", "embed_tokens"],
            )
        ]

    async def test_no_weight_comparison_without_the_flag(self, monkeypatch: pytest.MonkeyPatch):
        """The comparison reloads weights on every engine, so an ordinary run must never trigger it."""
        events: list[str] = []
        args = _make_args(check_weight_update_equal=False)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert components.inference_controller.check_weights_calls == []


class TestCriticValuesHandoff:
    async def test_critic_outputs_reach_the_actor_and_are_released_after_training(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critic value references reach actor training and remain live until it consumes them."""
        events: list[str] = []
        args = _make_args(num_rollout=1, use_critic=True)
        components = _install_driver_fakes(monkeypatch, args, events)
        store = FakeObjectStore()
        monkeypatch.setattr(object_store, "_INSTANCE", store)
        ref = store.put({"values": ["critic-values"]})
        values = [TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref)]
        components.critic_model.train_outputs[0] = values

        def consume_critic_values(external_data: list[TrainStepOutput]) -> None:
            assert external_data is values
            assert store.get(external_data[0].values).value == {"values": ["critic-values"]}

        components.actor_model.consume_external_data = consume_critic_values

        await with_disposer(train_driver.train, _startup_info(args))

        assert store.consumed == [ref]
        assert not store.contains(ref)


class TestApiServerWiring:
    async def test_api_server_receives_the_sync_driver_handles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only an enabled API server receives the live actor and inference controller handles."""
        started_with: list[dict[str, Any]] = []
        monkeypatch.setattr(placement_group, "start_api_server", lambda **kwargs: started_with.append(kwargs))
        disabled_events: list[str] = []
        disabled_args = _make_args(api_server_port=None)
        _install_driver_fakes(monkeypatch, disabled_args, disabled_events)
        await with_disposer(train_driver.train, _startup_info(disabled_args))
        assert started_with == []

        enabled_events: list[str] = []
        enabled_args = _make_args(api_server_port=8080)
        components = _install_driver_fakes(monkeypatch, enabled_args, enabled_events)
        await with_disposer(train_driver.train, _startup_info(enabled_args))

        assert len(started_with) == 1
        assert started_with[0]["trainer_models"] == {"actor": components.actor_model}
        assert started_with[0]["inference_controller"] is components.inference_controller
        assert started_with[0]["cell_operations"] is components.cell_operations


class TestCheckpointPublication:
    async def test_checkpoint_and_eval_share_the_completed_rollout_cadence(self, monkeypatch: pytest.MonkeyPatch):
        events: list[str] = []
        args = _make_args(num_rollout=40, save_interval=20, eval_interval=20, skip_eval_before_train=True)
        components = _install_driver_fakes(monkeypatch, args, events, num_rollout_per_epoch=100)

        await with_disposer(train_driver.train, _startup_info(args))

        assert components.actor_model.saved == [19, 39]
        assert [event for event in events if event.startswith("eval:")] == ["eval:19", "eval:39"]
        for rollout_id in components.actor_model.saved:
            assert (
                events.index(f"executor_save:{rollout_id}")
                < events.index(f"actor_save:{rollout_id}")
                < events.index(f"eval:{rollout_id}")
            )

    async def test_rollout_save_failure_does_not_publish_a_model_checkpoint(self, monkeypatch: pytest.MonkeyPatch):
        """A resumable model checkpoint must not precede its matching data progress."""
        events: list[str] = []
        args = _make_args(num_rollout=1, save_interval=20)
        components = _install_driver_fakes(monkeypatch, args, events)

        async def fail_rollout_save(rollout_id):
            raise OSError("rollout state write failed")

        components.rollout_executor.save = fail_rollout_save

        with pytest.raises(OSError, match="rollout state write failed"):
            await with_disposer(train_driver.train, _startup_info(args))

        assert components.actor_model.saved == []


class TestTerminalLifecycle:
    @pytest.mark.parametrize("audit_fails", [False, True])
    async def test_final_audit_runs_with_live_api_and_cells(
        self, monkeypatch: pytest.MonkeyPatch, audit_fails: bool
    ) -> None:
        """Final audits retain a working observation endpoint even when the audit raises."""
        events: list[str] = []
        args = _make_args()
        components = _install_driver_fakes(monkeypatch, args, events)

        def close_api() -> None:
            events.append("api_closed")

        async def final_audit() -> None:
            assert "api_closed" not in events
            assert "actor_dispose" not in events
            assert "inference_dispose" not in events
            events.append("audit")
            if audit_fails:
                raise RuntimeError("audit failed")

        monkeypatch.setattr(train_driver, "maybe_start_api_server", lambda *args, **kwargs: close_api)
        monkeypatch.setattr(components.rollout_executor, "_dispose", final_audit)

        if audit_fails:
            with pytest.raises(RuntimeError, match="audit failed"):
                await with_disposer(train_driver.train, _startup_info(args))
        else:
            await with_disposer(train_driver.train, _startup_info(args))

        assert events.count("audit") == 1
        assert events.index("audit") < events.index("api_closed") < events.index("actor_dispose")
        assert "inference_dispose" in events

    async def test_model_creation_failure_still_disposes_rollout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Startup failure before the API exists still releases the rollout components."""
        events: list[str] = []
        args = _make_args()
        _install_driver_fakes(monkeypatch, args, events)

        async def fail_creation(*args: Any, handles: dict[str, FakeTrainingModel]) -> None:
            raise RuntimeError("model creation failed")

        monkeypatch.setattr(train_driver, "create_training_models", fail_creation)
        with pytest.raises(RuntimeError, match="model creation failed"):
            await with_disposer(train_driver.train, _startup_info(args))

        assert events == ["executor_dispose", "inference_dispose"]

    async def test_train_disposes_all_created_component_controllers(self, monkeypatch: pytest.MonkeyPatch):
        """Every component the driver created must be disposed, or its watchers outlive the run."""
        events: list[str] = []
        args = _make_args(use_critic=True)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert sorted(event for event in events if event.endswith("_dispose")) == [
            "actor_dispose",
            "critic_dispose",
            "executor_dispose",
            "inference_dispose",
        ]

    async def test_train_releases_its_worker_manager_after_component_disposal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The driver releases its own manager only after every component has been disposed."""
        events: list[str] = []
        args = _make_args(use_critic=True)
        manager = FakeWorkerManager(events)
        _install_driver_fakes(monkeypatch, args, events)

        def init_orchestration_script(_startup_info: ArgvOrchestratorStartupInfo, *, disposer: Any) -> None:
            disposer.add(partial(wiring.shutdown_worker_manager, manager))

        monkeypatch.setattr(train_driver, "init_orchestration_script", init_orchestration_script)
        monkeypatch.setattr(wiring.ray, "kill", manager.kill)

        await with_disposer(train_driver.train, _startup_info(args))

        assert manager.killed == [manager]
        assert all(
            events.index(event) < events.index("manager_shutdown") for event in events if event.endswith("_dispose")
        )
        assert events.index("manager_shutdown") < events.index("manager_kill")


class TestOutputStoreRelease:
    async def test_each_rollout_is_released_once_it_has_trained(self, monkeypatch: pytest.MonkeyPatch):
        """The executor may delete a rollout's stored objects only after every trainer read them."""
        events: list[str] = []
        args = _make_args(num_rollout=3)
        components = _install_driver_fakes(monkeypatch, args, events)
        trained_at_release: list[tuple[int, list[int]]] = []

        async def release(rollout_id: int) -> None:
            trained_at_release.append((rollout_id, list(components.actor_model.trained)))

        components.rollout_executor.release = FakeRemoteMethod(release)

        await with_disposer(train_driver.train, _startup_info(args))

        assert trained_at_release == [(0, [0]), (1, [0, 1]), (2, [0, 1, 2])]


_LOOP_EVENTS = ("actor_train:", "executor_save:", "actor_save:", "eval:", "update_weights:")


def _loop_trace(events: list[str]) -> list[str]:
    """The training loop's events from the first trained rollout on, with the per-step memory work kept."""
    loop = events[events.index("actor_train:0") :] if "actor_train:0" in events else []
    return [e for e in loop if e.startswith(_LOOP_EVENTS) or e in ("actor_clear_memory", "actor_offload")]


class TestClearMemoryEveryStep:
    async def test_default_clears_after_every_rollout_but_the_last(self, monkeypatch: pytest.MonkeyPatch):
        """Today's behaviour stays the default: a clear between every two rollouts, none after the final one."""
        events: list[str] = []
        args = _make_args(num_rollout=4)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert _loop_trace(events) == [
            "actor_train:0",
            "actor_clear_memory",
            "update_weights:0",
            "actor_train:1",
            "actor_clear_memory",
            "update_weights:1",
            "actor_train:2",
            "actor_clear_memory",
            "update_weights:2",
            "actor_train:3",
        ]

    async def test_off_keeps_the_cache_between_plain_steps(self, monkeypatch: pytest.MonkeyPatch):
        events: list[str] = []
        args = _make_args(num_rollout=4, clear_memory_every_step=False)
        components = _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert components.actor_model.trained == [0, 1, 2, 3]
        assert "actor_clear_memory" not in events
        assert [e for e in events if e.startswith("update_weights:")] == [
            "update_weights:None",
            "update_weights:0",
            "update_weights:1",
            "update_weights:2",
        ]

    async def test_off_clears_before_and_after_each_save(self, monkeypatch: pytest.MonkeyPatch):
        """A save (and its HF export) starts from a clear cache and leaves one; the final save has no step after it."""
        events: list[str] = []
        args = _make_args(num_rollout=5, save_interval=2, clear_memory_every_step=False)
        components = _install_driver_fakes(monkeypatch, args, events, num_rollout_per_epoch=100)

        await with_disposer(train_driver.train, _startup_info(args))

        assert components.actor_model.saved == [1, 3, 4]
        assert _loop_trace(events) == [
            "actor_train:0",
            "update_weights:0",
            "actor_train:1",
            "actor_clear_memory",
            "executor_save:1",
            "actor_save:1",
            "actor_clear_memory",
            "update_weights:1",
            "actor_train:2",
            "update_weights:2",
            "actor_train:3",
            "actor_clear_memory",
            "executor_save:3",
            "actor_save:3",
            "actor_clear_memory",
            "update_weights:3",
            "actor_train:4",
            "actor_clear_memory",
            "executor_save:4",
            "actor_save:4",
        ]

    async def test_on_does_not_add_a_clear_before_the_save(self, monkeypatch: pytest.MonkeyPatch):
        """The pre-save clear belongs to the off mode only: the default keeps miles' exact call sequence."""
        events: list[str] = []
        args = _make_args(num_rollout=3, save_interval=2)
        _install_driver_fakes(monkeypatch, args, events, num_rollout_per_epoch=100)

        await with_disposer(train_driver.train, _startup_info(args))

        assert _loop_trace(events) == [
            "actor_train:0",
            "actor_clear_memory",
            "update_weights:0",
            "actor_train:1",
            "executor_save:1",
            "actor_save:1",
            "actor_clear_memory",
            "update_weights:1",
            "actor_train:2",
            "executor_save:2",
            "actor_save:2",
        ]

    async def test_off_clears_before_every_eval_including_the_final_one(self, monkeypatch: pytest.MonkeyPatch):
        events: list[str] = []
        args = _make_args(num_rollout=4, eval_interval=2, skip_eval_before_train=True, clear_memory_every_step=False)
        _install_driver_fakes(monkeypatch, args, events, num_rollout_per_epoch=100)

        await with_disposer(train_driver.train, _startup_info(args))

        assert _loop_trace(events) == [
            "actor_train:0",
            "update_weights:0",
            "actor_train:1",
            "actor_clear_memory",
            "update_weights:1",
            "eval:1",
            "actor_train:2",
            "update_weights:2",
            "actor_train:3",
            "actor_clear_memory",
            "update_weights:3",
            "eval:3",
        ]

    async def test_off_clears_around_a_sentinel_save(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        """An operator-triggered save gets the same clears as a scheduled one."""
        sentinel = tmp_path / "save-now"
        events: list[str] = []
        args = _make_args(num_rollout=3, save_trigger_sentinel=str(sentinel), clear_memory_every_step=False)
        components = _install_driver_fakes(monkeypatch, args, events)

        async def train_and_request_save(rollout_id, rollout_data, external_data=None):
            events.append(f"actor_train:{rollout_id}")
            components.actor_model.trained.append(rollout_id)
            if rollout_id == 0:
                sentinel.write_text("")

        components.actor_model.train = train_and_request_save

        await with_disposer(train_driver.train, _startup_info(args))

        assert not sentinel.exists()
        assert _loop_trace(events) == [
            "actor_train:0",
            "actor_clear_memory",
            "executor_save:0",
            "actor_save:0",
            "actor_clear_memory",
            "update_weights:0",
            "actor_train:1",
            "update_weights:1",
            "actor_train:2",
        ]

    async def test_offload_train_still_offloads_every_step(self, monkeypatch: pytest.MonkeyPatch):
        """The flag only drops the cache clear: an offloading run hands its GPU over after each step either way
        (argument validation refuses this combination; the driver must still not skip the offload)."""
        events: list[str] = []
        args = _make_args(num_rollout=3, offload_train=True, clear_memory_every_step=False)
        _install_driver_fakes(monkeypatch, args, events)

        await with_disposer(train_driver.train, _startup_info(args))

        assert events.count("actor_offload") == 2
        assert "actor_clear_memory" not in events
