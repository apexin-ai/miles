"""Merged SFT steps (apex/pr/qwen38-dynb-merge): per-row loss numerator scale, its data path and the
after-forward-backward hook (miles/backends/training_utils/loss/row_loss_scale.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tests.fast.backends.megatron_utils.test_model import (
    FakeMpu,
    TorchBackedOptimizer,
    train_one_step_env,
)
from tests.fast.fixtures.args_fixtures import with_backend_values
from tests.fast.fixtures.hook_consumer_fixtures import _hook

from miles.backends.training_utils.data.rollout import DataIterator, get_batch
from miles.backends.training_utils.loss import row_loss_scale
from miles.backends.training_utils.loss.hub import losses
from miles.ray.rollout import rollout_data_conversion as rdc
from miles.ray.rollout import train_data_conversion as tdc
from miles.utils.args.runtime import TrainerConfig
from miles.utils.function_registry import function_registry
from miles.utils.types import Sample

__all__ = ["train_one_step_env"]


@pytest.fixture
def cp1(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0), tp=SimpleNamespace(size=1, rank=0))
    monkeypatch.setattr("miles.backends.training_utils.parallel.get_parallel_state", lambda: state)
    monkeypatch.setattr("miles.backends.training_utils.data.context_parallel.get_parallel_state", lambda: state)
    return state


def _rows(seed=0, n=5):
    g = torch.Generator().manual_seed(seed)
    lens = [3, 1, 4, 2, 5][:n]
    lps = [-torch.rand(L, generator=g, dtype=torch.float32) for L in lens]
    masks = [torch.tensor([(i + j) % 3 != 0 for j in range(L)], dtype=torch.int32) for i, L in enumerate(lens)]
    return lens, lps, masks


def _args():
    return SimpleNamespace(backend=SimpleNamespace(calculate_per_token_loss=True))


class TestScaledSftLoss:
    def test_unit_scales_are_bitwise_the_original_per_token_sum(self, cp1):
        lens, lps, masks = _rows()
        batch = {"loss_masks": masks, "row_loss_scale": [1.0] * len(lens)}
        logits = torch.zeros(1, 2)
        loss, log = row_loss_scale.scaled_sft_loss(_args(), batch, lps, logits)
        original = -sum([(x * m).sum() for x, m in zip(lps, masks)])
        assert torch.equal(loss, original)
        assert torch.equal(log["loss"], original)

    def test_scales_weight_only_the_numerator(self, cp1):
        lens, lps, masks = _rows(1)
        scales = [2.5, 0.25, 1.0, 3.0, 0.5]
        batch = {"loss_masks": masks, "row_loss_scale": scales}
        loss, log = row_loss_scale.scaled_sft_loss(_args(), batch, lps, torch.zeros(1, 2))
        want = sum(-(x * m).sum().double() * a for x, m, a in zip(lps, masks, scales))
        assert abs(loss.double().item() - want.item()) < 1e-5
        # the logged loss stays the unweighted sum
        assert abs(log["loss"].item() + sum((x * m).sum().item() for x, m in zip(lps, masks))) < 1e-5

    def test_merge_identity_bucket_weights_kept(self, cp1):
        """(1/N_p) sum_k a_k sum_{t in k} l_t == sum_k (1/N_k) sum_{t in k} l_t with a_k = N_p / N_k (fp64)."""
        g = torch.Generator().manual_seed(3)
        buckets = [[torch.rand(int(n), generator=g, dtype=torch.float64) for n in rows] for rows in
                   ([3, 7, 2], [11], [1, 1, 5, 9])]
        n_k = [sum(len(r) for r in b) for b in buckets]
        n_p = sum(n_k)
        merged = sum((n_p / n) * sum(r.sum() for r in b) for b, n in zip(buckets, n_k)) / n_p
        separate = sum(sum(r.sum() for r in b) / n for b, n in zip(buckets, n_k))
        assert abs(merged.item() - separate.item()) < 1e-12

    def test_observer_gets_row_sums_and_mask_counts_only_for_train_micro_batches(self, cp1):
        lens, lps, masks = _rows(2)
        seen = []
        prev = row_loss_scale.set_observer(lambda kind, batch, **kw: seen.append((kind, batch[row_loss_scale.MB_ID_KEY], kw)))
        try:
            batch = {"loss_masks": masks, "row_loss_scale": [1.0] * 5}
            row_loss_scale.scaled_sft_loss(_args(), batch, lps, torch.zeros(1, 2))
            assert seen == []   # no q38_mb_id: not a train micro-batch
            batch[row_loss_scale.MB_ID_KEY] = (7, 0, 3)
            row_loss_scale.scaled_sft_loss(_args(), batch, lps, torch.zeros(1, 2))
        finally:
            row_loss_scale.set_observer(prev)
        (kind, mb, kw), = seen
        assert kind == "sft" and mb == (7, 0, 3)
        assert kw["row_counts"].tolist() == [int(m.sum()) for m in masks]
        assert torch.allclose(kw["row_sums"], torch.stack([-(x * m).sum() for x, m in zip(lps, masks)]))

    def test_refuses_sample_mean_loss_and_cp(self, cp1):
        lens, lps, masks = _rows()
        batch = {"loss_masks": masks, "row_loss_scale": [1.0] * 5}
        with pytest.raises(AssertionError, match="per-token"):
            row_loss_scale.scaled_sft_loss(SimpleNamespace(backend=SimpleNamespace(calculate_per_token_loss=False)),
                                           batch, lps, torch.zeros(1, 2))
        cp1.cp.size = 2
        with pytest.raises(AssertionError, match="context parallel"):
            row_loss_scale.scaled_sft_loss(_args(), batch, lps, torch.zeros(1, 2))

    def test_sft_loss_function_routes_on_the_fields(self, cp1, monkeypatch):
        lens, lps, masks = _rows(4)
        monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda *a, **k: {"log_probs": [x.clone() for x in lps]})
        base = {"loss_masks": masks, "response_lengths": lens, "total_lengths": lens, "unconcat_tokens": None}
        from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean

        ssm = get_sum_of_sample_mean(lens, lens, masks, True)
        off, _ = losses.sft_loss_function(_args(), {**base, "row_loss_scale": None}, torch.zeros(1, 2), ssm)
        on, _ = losses.sft_loss_function(_args(), {**base, "row_loss_scale": [1.0] * 5}, torch.zeros(1, 2), ssm)
        assert torch.equal(off, on)
        on2, _ = losses.sft_loss_function(_args(), {**base, "row_loss_scale": [2.0] * 5}, torch.zeros(1, 2), ssm)
        assert torch.allclose(on2, 2 * off)


def _sample(i, meta=None):
    s = Sample(prompt=[1, 2], index=i)
    s.tokens, s.response_length, s.loss_mask, s.reward = [1, 2, 3], 2, [1, 1], 0
    s.metadata = dict(meta or {})
    return s


def _meta(i, bucket=0):
    return {"row_loss_scale": 1.5, "logical_bucket_id": bucket, "row_instance_id": 100 + i}


class TestDataPath:
    def _args(self, **kw):
        base = dict(disable_rollout_trim_samples=False, use_dynamic_global_batch_size=True, global_batch_size=4,
                    enable_sample_ownership_checker=False)
        return SimpleNamespace(**{**base, **kw})

    def test_postprocess_keeps_every_row_of_a_merged_step(self):
        data = [_sample(i, _meta(i)) for i in range(8)]
        out, meta = rdc.postprocess_rollout_data(self._args(), data, SimpleNamespace(dp_size=4))
        assert meta["dynamic_global_batch_size"] == 8 and [s.index for s in out] == list(range(8))

    def test_postprocess_refuses_a_trim_and_the_bypasses(self):
        data = [_sample(i, _meta(i)) for i in range(6)]
        with pytest.raises(ValueError, match="trim changed the step"):
            rdc.postprocess_rollout_data(self._args(), data, SimpleNamespace(dp_size=4))
        for kw in ({"disable_rollout_trim_samples": True}, {"use_dynamic_global_batch_size": False}):
            with pytest.raises(ValueError, match="use-dynamic-global-batch-size"):
                rdc.postprocess_rollout_data(self._args(**kw), [_sample(i, _meta(i)) for i in range(8)],
                                             SimpleNamespace(dp_size=4))
        compact = [_sample(i, _meta(i)) for i in range(8)]
        for s in compact:
            s.rollout_id = 3
        with pytest.raises(ValueError, match="compact"):
            rdc.postprocess_rollout_data(self._args(), compact, SimpleNamespace(dp_size=4))

    def test_postprocess_without_fields_is_unchanged(self):
        data = [_sample(i) for i in range(6)]
        out, meta = rdc.postprocess_rollout_data(self._args(), data, SimpleNamespace(dp_size=4))
        assert len(out) == 4 and meta["dynamic_global_batch_size"] == 4

    def test_convert_all_or_none(self):
        ok = [_sample(i, _meta(i, bucket=i % 2)) for i in range(4)]
        td = {"dynamic_global_batch_size": 4}
        tdc._add_row_loss_scale_fields(None, ok, td)
        assert td["row_loss_scale"] == [1.5] * 4 and td["logical_bucket_id"] == [0, 1, 0, 1]
        assert td["row_instance_id"] == [100, 101, 102, 103]
        td = {"dynamic_global_batch_size": 4}
        tdc._add_row_loss_scale_fields(None, [_sample(i) for i in range(4)], td)
        assert td == {"dynamic_global_batch_size": 4}
        bad = [_sample(i, _meta(i)) for i in range(4)]
        del bad[2].metadata["logical_bucket_id"]
        with pytest.raises(ValueError, match="every row"):
            tdc._add_row_loss_scale_fields(None, bad, {"dynamic_global_batch_size": 4})
        with pytest.raises(ValueError, match="one dynamic global batch"):
            tdc._add_row_loss_scale_fields(None, ok, {"dynamic_global_batch_size": 2})

    def test_dp_shards_carry_the_fields_with_their_rows(self):
        data = {"tokens": [[i] * (i + 1) for i in range(8)], "row_loss_scale": [float(i) for i in range(8)],
                "logical_bucket_id": [i // 4 for i in range(8)], "row_instance_id": [10 + i for i in range(8)]}
        shards = tdc._package_shards(SimpleNamespace(multi_lora_n_adapters=None), data, [[7, 2, 4], [0, 1, 3, 5, 6]])
        for shard in shards:
            for j, row in enumerate(shard["partition"]):
                assert shard["tokens"][j] == data["tokens"][row]
                assert shard["row_loss_scale"][j] == float(row)
                assert shard["row_instance_id"][j] == 10 + row
                assert shard["logical_bucket_id"][j] == row // 4
        assert set(tdc.ROLLOUT_DATA_VALUE_SPEC) >= set(row_loss_scale.ROW_FIELDS)


class _PS:
    def __init__(self):
        self.tp = SimpleNamespace(size=1, rank=0)
        self.cp = SimpleNamespace(size=1, rank=0)


class TestMicroBatchIndex:
    @pytest.fixture(autouse=True)
    def _state(self, monkeypatch):
        monkeypatch.setattr("miles.backends.training_utils.data.rollout.get_parallel_state", lambda: _PS())
        monkeypatch.setattr("miles.backends.training_utils.data.context_parallel.get_parallel_state", lambda: _PS())
        monkeypatch.setattr("miles.backends.training_utils.data.rollout.current_platform",
                            SimpleNamespace(current_device=lambda: "cpu"))

    def _data(self, with_fields=True):
        d = {"tokens": [torch.tensor([1, 2, 3]) for _ in range(4)], "loss_masks": [torch.tensor([1]) for _ in range(4)],
             "total_lengths": [3] * 4, "response_lengths": [1] * 4}
        if with_fields:
            d["row_instance_id"] = [5, 6, 7, 8]
        return d

    def test_index_is_the_position_in_the_index_schedule(self):
        it = DataIterator(self._data(), micro_batch_indices=[[3, 0], [1], [2]])
        got = [get_batch(it, ["tokens", "loss_masks", "total_lengths", "response_lengths", "row_instance_id"])
               for _ in range(3)]
        assert [b["q38_mb_index"] for b in got] == [0, 1, 2]
        assert [b["row_instance_id"] for b in got] == [[8, 5], [6], [7]]

    def test_index_is_the_slice_number_with_fixed_micro_batches(self):
        it = DataIterator(self._data(), micro_batch_size=2)
        got = [get_batch(it, ["tokens", "loss_masks", "total_lengths", "response_lengths"]) for _ in range(2)]
        assert [b["q38_mb_index"] for b in got] == [0, 1]

    def test_no_index_without_the_fields(self):
        it = DataIterator(self._data(False), micro_batch_size=2)
        assert "q38_mb_index" not in get_batch(it, ["tokens", "loss_masks", "total_lengths", "response_lengths"])


class _Reached(Exception):
    pass


class TestAfterForwardBackwardHook:
    def _run(self, env, monkeypatch, hook_fn, *, num_tokens, optimizer, scheduler, num_microbatches=3):
        from miles.backends.megatron_utils import model as model_module

        hook = _hook("test.after_fb")
        env.args.enable_sample_ownership_checker = False
        env.args = with_backend_values(env.args, check_for_nan_in_loss_and_grad=False, calculate_per_token_loss=True)
        args = TrainerConfig.model_construct(**vars(env.args)).model_copy(
            update={"custom_megatron_after_forward_backward_hook_path": hook}
        )
        order = []

        def forward_backward_engine(**kwargs: Any):
            order.append(("fwd_bwd", model_module._train_step_context))
            for p in kwargs["model"][0].parameters():
                p.grad = torch.ones_like(p)
            count = torch.tensor(num_tokens, dtype=torch.int)
            model_module.finalize_model_grads_with_empty_cache(
                kwargs["model"], count, finalize=lambda model, num_tokens=None, *a, **k: None
            )
            return []

        monkeypatch.setattr(model_module, "get_forward_backward_func", lambda: forward_backward_engine)
        monkeypatch.setattr(model_module, "mpu", FakeMpu(is_last_pipeline_stage=True))
        monkeypatch.setattr(model_module, "aggregate_train_losses", lambda losses_reduced, num_rollouts: {})
        monkeypatch.setattr(model_module, "is_first_replica_megatron_main_rank", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))

        def callback(view, rollout_id, step_id, models, opt, *, step_num_tokens, num_microbatches):
            order.append(("hook", rollout_id, step_id, int(step_num_tokens), num_microbatches, opt is optimizer))
            assert view.hook_marker == 41
            hook_fn()

        with function_registry.temporary(hook.path, callback):
            out = model_module.train_one_step(
                args, 7, 0, env.data_iterator, env.model, optimizer, scheduler, num_microbatches, 4,
                witness_info=None, attempt=0,
            )
        return order, out

    def _opt(self, env):
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        parameter = torch.nn.Parameter(torch.ones(3))
        env.model[0].register_parameter("weight", parameter)
        opt = TorchBackedOptimizer(parameter)
        sched = OptimizerParamScheduler(opt, init_lr=0.0, max_lr=0.1, min_lr=0.0, lr_warmup_steps=4,
                                        lr_decay_steps=16, lr_decay_style="linear", start_wd=0.1, end_wd=0.1,
                                        wd_incr_steps=16, wd_incr_style="constant")
        return opt, sched

    def test_runs_after_finalize_with_the_step_token_count_and_mb_count(self, train_one_step_env, monkeypatch):
        opt, sched = self._opt(train_one_step_env)
        order, (loss, _, outcome) = self._run(train_one_step_env, monkeypatch, lambda: None, num_tokens=5,
                                              optimizer=opt, scheduler=sched)
        assert order == [("fwd_bwd", (7, 0)), ("hook", 7, 0, 5, 3, True)]
        assert opt.step_count == 1

    @pytest.mark.parametrize("num_tokens", [0, 5])
    def test_a_hook_failure_stops_before_the_skip_and_the_optimizer(self, train_one_step_env, monkeypatch,
                                                                     num_tokens):
        """Planned tokens > 0 but the step's masks are all zero: the hook raises before the unsupervised skip."""
        opt, sched = self._opt(train_one_step_env)
        before = sched.num_steps

        def boom():
            raise _Reached

        with pytest.raises(_Reached):
            self._run(train_one_step_env, monkeypatch, boom, num_tokens=num_tokens, optimizer=opt, scheduler=sched)
        assert opt.step_count == 0 and sched.num_steps == before

    def test_unsupervised_step_still_reaches_the_hook(self, train_one_step_env, monkeypatch):
        opt, sched = self._opt(train_one_step_env)
        order, (loss, _, outcome) = self._run(train_one_step_env, monkeypatch, lambda: None, num_tokens=0,
                                              optimizer=opt, scheduler=sched)
        assert ("hook", 7, 0, 0, 3, True) in order and loss == {"skipped_unsupervised_step": 1.0}
        assert opt.step_count == 0

    def test_runs_without_an_optimizer(self, train_one_step_env, monkeypatch):
        order, _ = self._run(train_one_step_env, monkeypatch, lambda: None, num_tokens=5, optimizer=None,
                             scheduler=None)
        assert ("hook", 7, 0, 5, 3, True) in order

    def test_context_is_cleared_after_the_pass(self, train_one_step_env, monkeypatch):
        from miles.backends.megatron_utils import model as model_module

        opt, sched = self._opt(train_one_step_env)
        self._run(train_one_step_env, monkeypatch, lambda: None, num_tokens=5, optimizer=opt, scheduler=sched)
        assert model_module._train_step_context is None
