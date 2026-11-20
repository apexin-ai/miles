"""Exercise the Megatron update decision across two pipeline stages and two replicas."""

import logging
import time
from datetime import timedelta
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cpu_ci
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles.backends.training_utils.data.rollout import DataIterator
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.utils.external_utils.command_utils.common import encode_pseudo_file

from .test_model import (  # noqa: F401 (train_one_step_env is a fixture)
    FakeFiniteOptimizer,
    FakeModelChunk,
    FakeNonfiniteOptimizer,
    FakeScheduler,
    make_train_one_step_args,
    train_one_step_env,
)

# 10 spawned Gloo cases (4 and 2 ranks) plus in-process checks; about 340 s on a CPU-only pod
register_cpu_ci(est_time=360, suite="stage-b-cpu", labels=["megatron"])

_GROUP_TIMEOUT = timedelta(seconds=30)
_FLAG = "--skip-empty-supervision-steps"


class _Chunk(FakeModelChunk):
    def __init__(self, dp_group):
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor(1.0))
        self.right = torch.nn.Parameter(torch.tensor(2.0))
        self.config = TransformerConfig(num_layers=2, hidden_size=4, num_attention_heads=1)
        self.dp_group = dp_group

    def forward(self, **kwargs):
        return torch.stack((self.left, self.right))

    def finish_grad_sync(self, force_all_reduce=False):
        for p in (self.left, self.right):
            dist.all_reduce(p.grad, group=self.dp_group)

    def scale_gradients(self, scale):
        for p in (self.left, self.right):
            p.grad.mul_(scale)


class _AdamW(torch.optim.AdamW):
    def scale_loss(self, loss):
        return loss

    def step(self):
        super().step()
        return True, 0.0, 0


class _SampleScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.samples = 0

    def step(self, *, increment):
        self.samples += increment
        for group in self.optimizer.param_groups:
            group["lr"] *= 0.9


class _State(NamedTuple):
    steps: tuple
    weights: tuple
    first_moments: tuple
    second_moments: tuple
    samples: int
    lrs: tuple
    version: int


def _cpu_batch(iterator, *args, **kwargs):
    batch = iterator.get_next(["tokens", "loss_masks", "total_lengths", "response_lengths"])
    batch["unconcat_tokens"] = batch["tokens"]
    batch["tokens"] = torch.stack(batch["tokens"])
    batch["input_loss_masks"] = torch.nn.functional.pad(batch["loss_masks"][0], (0, 1)).unsqueeze(0)
    batch["multimodal_train_inputs"] = None
    return batch


def _cpu_log_probs(logits, *, unconcat_tokens, **kwargs):
    target = unconcat_tokens[0][-1:]
    return {"log_probs": [logits.log_softmax(0)[target]]}


def _exercise(groups, masks, *, enabled):
    from miles.backends.megatron_utils import model as trainer
    from miles.backends.training_utils.loss.hub import losses

    last = groups.pp.rank() == 1
    options = {"skip_empty_supervision_steps": True} if enabled else {}
    args = make_train_one_step_args(
        **options,
        loss_type="sft_loss",
        calculate_per_token_loss=True,
        enable_sample_ownership_checker=False,
        qkv_format="bshd",
        log_probs_backend="torch",
        overlap_param_gather=False,
    )
    chunk = _Chunk(groups.dp_cp)
    params = (chunk.left, chunk.right)
    optimizer = _AdamW([{"params": [params[0]], "lr": 0.1}, {"params": [params[1]], "lr": 0.04}], weight_decay=0.1)
    scheduler = _SampleScheduler(optimizer)
    config = trainer.setup_train_iteration_config(args, [chunk], optimizer, disable_optimizer=False)
    iterator = DataIterator(
        {
            "tokens": [torch.tensor([0, 1])] * len(masks),
            "total_lengths": [2] * len(masks),
            "response_lengths": [1] * len(masks),
            "loss_masks": [torch.tensor([m]) for m in masks],
        },
        micro_batch_size=1,
    )
    grads = []

    def schedule(*, forward_step_func, model, data_iterator, num_microbatches, **kwargs):
        logs = []
        tokens = torch.zeros((), dtype=torch.int64)
        for _ in range(num_microbatches):
            output, compute_loss = forward_step_func(data_iterator[0], model[0])
            if last:
                loss, count, metrics = compute_loss(output)
                loss.backward()
                tokens += count
                logs.append(metrics)
        # Copy the tail stage's backward signal to a toy chunk with the same two parameters.
        for p in params:
            if not last:
                p.grad = torch.zeros_like(p)
            dist.broadcast(p.grad, group=groups.pp, group_src=1)
        with patch.object(torch.cuda, "current_device", return_value=0), patch.object(
            torch.cuda, "mem_get_info", return_value=(100, 100)
        ):
            config.finalize_model_grads_func(model, tokens, pg_collection=groups)
        grads.append(tuple(p.grad.item() for p in params))
        return logs

    states = []
    with (
        patch.object(trainer, "get_batch", _cpu_batch),
        patch.object(trainer, "get_forward_backward_func", return_value=schedule),
        patch.object(trainer.mpu, "is_pipeline_last_stage", return_value=last),
        patch.object(losses, "get_log_probs_and_entropy", _cpu_log_probs),
    ):
        for step in range(len(masks) // 2):
            trainer.train_one_step(args, 0, step, [iterator], [chunk], optimizer, scheduler, 2, 4, None, 0)
            assert iterator.offset == (step + 1) * 2
            assert all(p.grad is None or not p.grad.any() for p in params)
            states.append(
                _State(
                    tuple(optimizer.state[p]["step"].item() for p in params),
                    tuple(p.item() for p in params),
                    tuple(optimizer.state[p]["exp_avg"].item() for p in params),
                    tuple(optimizer.state[p]["exp_avg_sq"].item() for p in params),
                    scheduler.samples,
                    tuple(g["lr"] for g in optimizer.param_groups),
                    chunk.model_companion.weight_version.item(),
                )
            )
    return states, grads


def _group(ranks):
    return dist.new_group(ranks, timeout=_GROUP_TIMEOUT)


def _info(group):
    return GroupInfo(group.rank(), group.size(), group, gloo_group=group, src_rank=dist.get_global_rank(group, 0))


def _worker(rank, store, enabled):
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=4, timeout=_GROUP_TIMEOUT)
    try:
        tp = [_group([r]) for r in range(4)][rank]
        pp = [_group(ranks) for ranks in ([0, 1], [2, 3])][rank // 2]
        dp = [_group(ranks) for ranks in ([0, 2], [1, 3])][rank % 2]
        groups = SimpleNamespace(tp=tp, pp=pp, dp_cp=dp, embd=None, pos_embd=None)
        single = _info(tp)
        set_parallel_state(
            ParallelState(
                intra_dp=_info(dp),
                intra_dp_cp=_info(dp),
                cp=single,
                tp=single,
                pp=_info(pp),
                ep=single,
                etp=single,
                indep_dp=single,
                is_pp_last_stage=pp.rank() == 1,
            )
        )
        # Only replica 1 supplies a label on the last microbatch. Only PP tails see losses.
        warmup, empty, sparse = [1, 1], [0, 0], [0, dp.rank()]
        states, grads = _exercise(groups, warmup + empty + sparse, enabled=enabled)
        reference, ref_grads = _exercise(groups, warmup + sparse, enabled=False)
        results = [None] * 4
        dist.all_gather_object(results, (states, grads))
        assert all(result == (states, grads) for result in results), "ranks diverged"
        first, middle, last = states
        assert first == reference[0]
        assert grads[0] == ref_grads[0] and all(grads[0])
        assert grads[1] == (0.0, 0.0) and all(grads[2])
        expected = [1, 1, 2] if enabled else [1, 2, 3]
        observed = [s.steps for s in states]
        assert observed == [(n, n) for n in expected], f"rank {rank}: AdamW steps {observed}, expected {expected}"
        assert [s.samples for s in states] == [4 * n for n in expected]
        assert [s.version for s in states] == expected
        assert all(first.first_moments) and all(first.second_moments)
        if enabled:
            assert middle == first
            assert grads[2] == ref_grads[1]
            assert last == reference[1]
        transitions = [(middle, last)] if enabled else [(first, middle), (middle, last)]
        for before, after in transitions:
            for field in ("weights", "first_moments", "second_moments", "lrs"):
                assert all(a != b for a, b in zip(getattr(before, field), getattr(after, field), strict=True)), field
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("enabled", [False, True], ids=["default", "enabled"])
def test_empty_supervision_update(tmp_path, enabled):
    ctx = mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(), enabled), nprocs=4, join=False)
    deadline = time.monotonic() + 180
    try:
        while not ctx.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("four-rank train step did not finish within 180 seconds")
    finally:
        for proc in ctx.processes:
            if proc.is_alive():
                proc.terminate()
        for proc in ctx.processes:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        (["--enable-mtp-training", "--mtp-num-layers", "1"], "mtp|multi.token"),
        (["--num-experts", "2", "--moe-aux-loss-coeff", "0.01"], "moe|expert|aux|dense"),
        (["--indep-dp", "--use-dynamic-batch-size", "--max-tokens-per-gpu", "512"], "indep|independent"),
        (
            [
                "--experimental-attention-variant",
                "dsa",
                "--multi-latent-attention",
                "--disable-bias-linear",
                "--dsa-kernel-backend",
                "none",
                "--dsa-indexer-loss-coeff",
                "0.1",
            ],
            "dsa|indexer",
        ),
        (["--overlap-grad-reduce", "--overlap-param-gather"], "overlap"),
        (
            [
                "--custom-model-provider-path",
                "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider",
            ],
            "provider",
        ),
        (["--optimizer", "sgd"], "adam"),
    ],
    ids=[
        "mtp",
        "moe-aux",
        "independent-dp",
        "dsa-indexer-loss",
        "overlap-param-gather",
        "custom-model-provider",
        "sgd",
    ],
)
def test_rejects_incompatible_supervision_skip(extra, reason):
    common = ["--loss-type", "sft_loss", "--calculate-per-token-loss", "--position-embedding-type", "rope"]
    parse_megatron_test_config(*common, _FLAG)
    parse_megatron_test_config(*common, *extra)
    with pytest.raises((ValueError, AssertionError), match=f"(?i){reason}"):
        parse_megatron_test_config(*common, _FLAG, *extra)


class _MegatronOptimizer:
    """What the save guard reads from a Megatron optimizer: the torch optimizer it wraps."""

    def __init__(self, optimizer):
        self.optimizer = optimizer


def _adamw(stepped):
    param = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([param], lr=0.1, weight_decay=0.1)
    if stepped:
        param.grad = torch.ones(2)
        optimizer.step()
    return _MegatronOptimizer(optimizer)


@pytest.mark.parametrize(
    ("enabled", "stepped", "no_save_optim", "refused"),
    [
        (True, False, False, True),
        (False, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
    ],
    ids=["never-stepped", "flag-off", "stepped", "no-save-optim"],
)
def test_save_refuses_an_optimizer_that_never_stepped(enabled, stepped, no_save_optim, refused):
    """Megatron cannot save and restore an optimizer before its first update, so save() refuses instead."""
    from miles.backends.megatron_utils import model as trainer

    options = {"skip_empty_supervision_steps": True} if enabled else {}
    args = make_train_one_step_args(
        **options, no_save_optim=no_save_optim, use_distributed_optimizer=False, overlap_param_gather=False
    )
    saved = []
    with (
        patch.object(trainer, "save_checkpoint", lambda *a, **k: saved.append(a[0])),
        patch.object(trainer, "is_lora_model", return_value=False),
    ):
        if refused:
            with pytest.raises(RuntimeError, match="has not made an update yet"):
                trainer.save(args, 3, [FakeModelChunk()], _adamw(stepped), None)
        else:
            trainer.save(args, 3, [FakeModelChunk()], _adamw(stepped), None)
    assert saved == ([] if refused else [3])


def _guard_worker(rank, store, stepped_by_rank, refused):
    from miles.backends.megatron_utils.model import check_optimizer_stepped_before_save

    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2, timeout=_GROUP_TIMEOUT)
    try:
        state = stepped_by_rank[rank]
        if state is None:  # this rank holds no parameter, which leaves its optimizer without state after a step too
            optimizer = _MegatronOptimizer(torch.optim.AdamW([{"params": []}]))
        elif state == "stub":  # MCore's stub DistributedOptimizer for a rank without trainable parameters
            optimizer = SimpleNamespace(is_stub_optimizer=True, optimizer=None)
        elif state == "empty-chain":
            from megatron.core.optimizer.optimizer import ChainedOptimizer

            optimizer = ChainedOptimizer([])
        else:
            optimizer = _adamw(state)
        if refused:
            with pytest.raises(RuntimeError, match="has not made an update yet"):
                check_optimizer_stepped_before_save(optimizer, 3)
        else:
            check_optimizer_stepped_before_save(optimizer, 3)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("stepped_by_rank", "refused"),
    [
        ((True, False), True),
        ((False, True), True),
        ((True, True), False),
        ((True, None), False),
        ((True, "stub"), False),
        ((False, "stub"), True),
        ((True, "empty-chain"), False),
        ((False, "empty-chain"), True),
    ],
    ids=[
        "rank1-never-stepped",
        "rank0-never-stepped",
        "both-stepped",
        "rank1-holds-no-param",
        "rank1-stub-rank0-stepped",
        "rank1-stub-rank0-never-stepped",
        "rank1-empty-chain-rank0-stepped",
        "rank1-empty-chain-rank0-never-stepped",
    ],
)
def test_save_guard_agrees_across_ranks(tmp_path, stepped_by_rank, refused):
    """Every rank raises when any rank's optimizer has never stepped, so no rank waits in a save collective."""
    mp.spawn(_guard_worker, args=((tmp_path / "gloo").as_uri(), stepped_by_rank, refused), nprocs=2, join=True)


def _final_trainer_configs(*argv):
    from miles.utils.args.trainer_utils import compute_trainer_config

    args = parse_megatron_test_config("--loss-type", "sft_loss", "--calculate-per-token-loss", *argv)
    return [compute_trainer_config(args, trainer) for trainer in args.raw_megatron.trainers]


@pytest.mark.parametrize(
    ("argv", "reason"),
    [
        (
            [
                "--megatron-config",
                encode_pseudo_file(
                    yaml.dump(
                        {
                            "trainers": [
                                {"model_id": "policy", "overrides": {"num_experts": 2, "moe_aux_loss_coeff": 0.01}}
                            ]
                        }
                    )
                ),
            ],
            "moe|expert",
        ),
        (["--advantage-estimator", "ppo"], "sft_loss"),
    ],
    ids=["megatron-config-moe-override", "ppo-critic"],
)
def test_rejects_supervision_skip_on_a_final_trainer_config(argv, reason):
    """A per-trainer override or the derived critic can bring back what the global check rejects."""
    _final_trainer_configs(*argv)
    with pytest.raises(ValueError, match=f"(?i){reason}"):
        _final_trainer_configs(_FLAG, *argv)


def test_accepts_supervision_skip_on_a_dense_sft_trainer():
    """Positive control for the final-config check: plain dense per-token SFT."""
    (config,) = _final_trainer_configs(_FLAG)
    assert config.skip_empty_supervision_steps


@pytest.mark.parametrize(
    ("optimizer_cls", "skipped"),
    [(FakeFiniteOptimizer, True), (FakeNonfiniteOptimizer, False)],
    ids=["finite-grads", "nonfinite-grads"],
)
def test_only_a_step_that_would_have_updated_counts_as_skipped(request, monkeypatch, caplog, optimizer_cls, skipped):
    """A step without supervision whose grads are not finite is a failed step, not a successful skip."""
    from miles.backends.megatron_utils import model as trainer

    env = request.getfixturevalue("train_one_step_env")
    env.args = make_train_one_step_args(
        skip_empty_supervision_steps=True, check_for_nan_in_loss_and_grad=False, enable_sample_ownership_checker=False
    )
    monkeypatch.setattr(trainer, "_step_has_supervision", lambda supervision: False)
    optimizer = optimizer_cls()
    with caplog.at_level(logging.INFO, logger="miles.backends.megatron_utils.model"):
        trainer.train_one_step(env.args, 7, 3, env.data_iterator, env.model, optimizer, FakeScheduler(), 1, 1, None, 2)
    flag = "true" if skipped else "false"
    valid = "true" if skipped else "false"
    assert (
        f"train op=train_step rollout=7 step=3 attempt=2 outcome=NORMAL valid_step={valid} skipped_no_supervision={flag}"
        in caplog.messages
    )
    assert any("has no supervised token" in m for m in caplog.messages) == skipped
    if skipped:
        assert optimizer.step_count == 0
