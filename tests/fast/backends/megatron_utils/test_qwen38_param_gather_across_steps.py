"""CPU contracts for --qwen38-overlap-param-gather-across-steps (F8 tier B, PERF-F 2026-11-09).

train() runs with stand-in DDP chunks that record every forward pre-hook enable / disable (and whether the disable
all-gathered synchronously), a stub train_one_step and stub logging. Flag off: the call sequence of 38d88efa.
Flag on: the hooks stay on between train() calls and no synchronous gather happens at the end of train().
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import model as mm


class FakeDDP:
    def __init__(self, events, params=()):
        self.events = events
        self.remove_forward_pre_hook_handles = {"m": object()}  # Megatron DDP enables the hooks at init
        self._params = list(params)

    def enable_forward_pre_hook(self):
        assert len(self.remove_forward_pre_hook_handles) == 0
        self.remove_forward_pre_hook_handles = {"m": object()}
        self.events.append("enable")

    def disable_forward_pre_hook(self, param_sync=True):
        assert len(self.remove_forward_pre_hook_handles) > 0, "disable with the hooks already off (KeyError in DDP)"
        self.remove_forward_pre_hook_handles = {}
        self.events.append(f"disable(sync={param_sync})")

    def train(self):
        pass

    def start_param_sync(self, force_sync=False):
        self.events.append(f"sync(force={force_sync})")

    def named_parameters(self):
        return iter(self._params)


def _args(flag=True, **backend):
    b = dict(manual_gc=False, use_distributed_optimizer=True, overlap_param_gather=True, overlap_grad_reduce=True,
             align_param_gather=False, overlap_param_gather_with_optimizer_step=False, calculate_per_token_loss=False)
    b.update(backend)
    return SimpleNamespace(debug_disable_optimizer=True, reset_optimizer_states=False, backend=SimpleNamespace(**b),
                           enable_mtp_training=False, ci_test=False, debug_train_only=True,
                           save_local_weight_checksum=False, enable_witness=False, ci_save_model_hash=False,
                           custom_megatron_before_train_step_hook_path=None,
                           qwen38_before_step_hook_drains_param_gather=False, keep_old_actor=False, colocate=False,
                           offload_train=False, qwen38_overlap_param_gather_across_steps=flag)


@pytest.fixture
def env(monkeypatch):
    events = []
    monkeypatch.setattr(mm, "DDP", FakeDDP)
    monkeypatch.setattr(mm, "setup_train_iteration_config", lambda *a, **k: SimpleNamespace(param_sync_func=None))
    monkeypatch.setattr(mm, "get_parallel_state",
                        lambda: SimpleNamespace(indep_dp=SimpleNamespace(size=1), effective_dp=SimpleNamespace(rank=0)))

    def step(*a, **k):
        events.append("step")
        return {}, 1.0, mm.TrainStepOutcome.NORMAL
    monkeypatch.setattr(mm, "train_one_step", step)
    monkeypatch.setattr(mm, "log_train_step", lambda **k: {})
    monkeypatch.setattr(mm, "mpu", SimpleNamespace(is_pipeline_last_stage=lambda **k: True))
    monkeypatch.setattr(mm, "is_first_replica_megatron_main_rank", lambda: False)
    monkeypatch.setattr(mm, "install_param_gather_scopes", lambda model: [])   # covered by test_..._scopes.py
    return events


def _train(args, model, rollout_id=0):
    it = SimpleNamespace(reset=lambda: None)
    return mm.train(args, rollout_id, model, None, None, [it], [1], [1], None, 0)


def test_flag_default_off():
    from miles.utils.args.configs.custom_megatron_plugins import Qwen38VisionMegatronPluginsConfig
    field = Qwen38VisionMegatronPluginsConfig.model_fields["qwen38_overlap_param_gather_across_steps"]
    assert field.default is False


def test_flag_off_keeps_the_old_sequence(env):
    model = [FakeDDP(env)]
    _train(_args(flag=False), model)
    assert env == ["disable(sync=False)", "step", "enable", "disable(sync=True)"]


def test_flag_off_without_overlap_param_gather_touches_no_hook(env):
    model = [FakeDDP(env)]
    _train(_args(flag=False, overlap_param_gather=False), model)
    assert env == ["step"]


def test_flag_on_keeps_hooks_between_steps(env):
    model = [FakeDDP(env), FakeDDP(env)]
    for r in range(3):
        _train(_args(), model, r)
    # first train(): as before up to the enable; later train(): no hook change, no synchronous gather
    assert env == ["disable(sync=False)", "disable(sync=False)", "step", "enable", "enable", "step", "step"]
    assert mm.forward_pre_hooks_enabled(model)
    assert not any(e == "disable(sync=True)" for e in env)


def test_finish_pending_gathers_once_and_rearms(env):
    model = [FakeDDP(env)]
    args = _args()
    _train(args, model)
    assert mm.param_gather_pending(model)
    env.clear()
    assert mm.finish_pending_param_gather(args, model) is True
    assert env == ["disable(sync=True)", "enable"] and not mm.param_gather_pending(model)
    env.clear()
    assert mm.finish_pending_param_gather(args, model) is False   # nothing pending: no second gather (PERFFREV F7)
    assert env == []
    _train(args, model, 1)
    assert env == ["step"] and mm.param_gather_pending(model)


def test_pending_cleared_by_the_next_train(env):
    model = [FakeDDP(env)]
    args = _args()
    _train(args, model)
    _train(args, model, 1)            # its forward pre-hooks finished the gather; its own step left a new one
    assert mm.param_gather_pending(model)
    env.clear()
    assert mm.finish_pending_param_gather(args, model) is True and env == ["disable(sync=True)", "enable"]


def test_before_step_hook_sees_the_pending_gather(env, monkeypatch):
    seen = []

    def step(*a, **k):
        seen.append(mm.param_gather_pending(a[4]))   # what a before-train-step hook inside the step would see
        mm.drain_pending_param_gather(a[4])            # a draining hook (state digest)
        return {}, 1.0, mm.TrainStepOutcome.NORMAL
    monkeypatch.setattr(mm, "train_one_step", step)
    model = [FakeDDP(env)]
    args = _args()
    for r in range(3):
        _train(args, model, r)
    assert seen == [False, True, True]
    assert env.count("disable(sync=True)") == 2 and mm.param_gather_pending(model)


def test_pending_with_hooks_off_syncs_without_hooks(env):
    model = [FakeDDP(env)]
    args = _args()
    _train(args, model)
    model[0].remove_forward_pre_hook_handles = {}
    env.clear()
    assert mm.finish_pending_param_gather(args, model) is True
    assert env == ["sync(force=True)"] and not mm.forward_pre_hooks_enabled(model)


def test_finish_pending_is_a_no_op_when_off(env):
    model = [FakeDDP(env)]
    assert mm.finish_pending_param_gather(_args(flag=False), model) is False
    assert env == []


def test_warm_with_hooks_switched_off_rearms(env):
    model = [FakeDDP(env)]
    args = _args()
    _train(args, model)
    model[0].disable_forward_pre_hook(param_sync=True)  # e.g. an eval path that synced and left them off
    env.clear()
    _train(args, model, 1)
    assert env == ["enable", "step"]


@pytest.mark.parametrize("backend,flag_args", [
    ({"overlap_param_gather": False}, {}), ({"overlap_grad_reduce": False}, {}),
    ({"use_distributed_optimizer": False}, {}), ({"align_param_gather": True}, {}),
    ({"overlap_param_gather_with_optimizer_step": True}, {}),
    ({}, {"debug_train_only": False}), ({}, {"save_local_weight_checksum": True}), ({}, {"enable_witness": True}),
    ({}, {"ci_save_model_hash": True}), ({}, {"custom_megatron_before_train_step_hook_path": "x.hook"}),
    ({}, {"keep_old_actor": True}), ({}, {"colocate": True}), ({}, {"offload_train": True}),
])
def test_refused_combinations(backend, flag_args):
    args = _args(**backend)
    for k, v in flag_args.items():
        setattr(args, k, v)
    with pytest.raises(ValueError, match="across-steps"):
        mm.keep_param_gather_hooks(args)


def test_draining_before_step_hook_allowed():
    args = _args()
    args.custom_megatron_before_train_step_hook_path = "state_digest.before_train_step"
    args.qwen38_before_step_hook_drains_param_gather = True
    assert mm.keep_param_gather_hooks(args) is True


def test_save_syncs_before_reading_and_clears_pending():
    src = inspect.getsource(mm.save)
    i_dis = src.index("disable_forward_pre_hook(model)")
    assert src.index("_set_param_gather_pending(model, False)") > i_dis
    assert src.index("check_tp_replicas(") > i_dis and src.index("save_checkpoint(") > i_dis


def test_actor_refuses_weight_backup_with_the_flag():
    from miles.backends.megatron_utils import actor
    a = inspect.getsource(actor)
    assert "if keep_param_gather_hooks(self.args) and self._enable_weight_backup:" in a


def test_trainable_vision_refused(env):
    p = torch.nn.Parameter(torch.zeros(1))
    model = [FakeDDP(env, params=[("module.vision_model.blocks.0.w", p)])]
    with pytest.raises(ValueError, match="vision"):
        _train(_args(), model)


def test_frozen_vision_allowed(env):
    p = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    model = [FakeDDP(env, params=[("module.vision_model.blocks.0.w", p)])]
    _train(_args(), model)
    assert env[-1] == "enable"


def test_out_of_step_readers_finish_the_gather():
    src = inspect.getsource(mm.forward_only)
    assert "finish_pending_param_gather(args, model)" in src
    from miles.backends.megatron_utils import actor
    a = inspect.getsource(actor)
    assert a.count("finish_pending_param_gather(self.args, self.model)") == 2   # save_model HF export, export_hf
    # weight updates: the flag needs --debug-train-only, where update_weights returns before reading weights
    assert "if self.args.debug_train_only or self.args.debug_rollout_only:" in a
