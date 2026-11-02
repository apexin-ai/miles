"""CPU contracts for f8_guard: (a) per-bucket local norms, (b) written-after-issue check, (c) alerts.

Stand-in bucket groups run Megatron's own check_grads (fdf82356 _ParamAndGradBucketGroup.check_grads, with the real
rerun state machine) and a start_grad_sync shaped like Megatron's: NaN check, in-place gradient scaling, then
param_and_grad_buffer.dist_reduce_scatter_func(this rank's shard, the bucket) per bucket. The collective is a stand-in
that writes only this rank's shard, as a reduce-scatter does.
"""

import json
import time
import types
from types import SimpleNamespace

import pytest
import torch
from megatron.core.distributed import param_and_grad_buffer as pgb

from miles.backends.megatron_utils import f8_guard


def _fake_reduce_scatter(output, input, *args, **kwargs):
    output.mul_(2.0)  # the "reduced" shard: only this rank's part of the bucket is written
    return SimpleNamespace(wait=lambda: None)


class _Group(SimpleNamespace):
    def start_grad_sync(self, force_all_reduce=False):
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return
        if self.ddp_config.check_for_nan_in_grad:
            self.check_grads(check_for_nan_or_inf=True, check_for_large=False)
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor
        for bucket in self.buckets:
            shard = bucket.grad_data.numel() // self.intra_distributed_optimizer_instance_size
            local = bucket.grad_data[self.intra_distributed_optimizer_instance_rank * shard :][:shard]
            self.grad_reduce_handle = pgb.dist_reduce_scatter_func(local, bucket.grad_data, op=None, async_op=True)

    def finish_grad_sync(self, force_all_reduce=False):
        if self.grad_reduce_handle is None:
            self.start_grad_sync()
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None


def _chunk(n_groups=4, buckets_per_group=2, numel=64, nan_check=True, overlap=True, scale=1.0, rank=1):
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup as Group

    ddp = SimpleNamespace(
        use_distributed_optimizer=True,
        overlap_grad_reduce=overlap,
        check_for_nan_in_grad=nan_check,
        check_for_large_grads=False,
    )
    groups = []
    for _ in range(n_groups):
        buckets = [
            SimpleNamespace(grad_data=torch.zeros(numel), gradient_scaling_factor=scale)
            for _ in range(buckets_per_group)
        ]
        g = _Group(
            ddp_config=ddp,
            buckets=buckets,
            is_first_batch=False,
            grad_reduce_handle=None,
            intra_distributed_optimizer_instance_size=4,
            intra_distributed_optimizer_instance_rank=rank,
        )
        g.check_grads = types.MethodType(Group.check_grads, g)
        groups.append(g)
    return SimpleNamespace(ddp_config=ddp, bucket_groups=groups, expert_parallel_bucket_groups=[])


@pytest.fixture
def guard_env(monkeypatch, tmp_path):
    monkeypatch.setattr(pgb, "dist_reduce_scatter_func", _fake_reduce_scatter)
    for k in [k for k in __import__("os").environ if k.startswith("Q38_F8_GUARD")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("Q38_F8_GUARD_DIR", str(tmp_path / "f8-guard"))
    return tmp_path / "f8-guard"


def _step(chunk, rollout_id, grad_norm=1.0, between=None, seed=0):
    """begin, fill the grads, issue every group (as the hooks would), optionally act after the issue, finish, end."""
    f8_guard.begin_step(rollout_id)
    gen = torch.Generator().manual_seed(seed + rollout_id)
    for g in chunk.bucket_groups:
        for b in g.buckets:
            b.grad_data.copy_(torch.randn(b.grad_data.shape, generator=gen))
    for g in chunk.bucket_groups:
        g.start_grad_sync()
    if between is not None:
        between(chunk)
    for g in chunk.bucket_groups:
        g.finish_grad_sync()
    return f8_guard._GUARD.end_step(grad_norm)


def test_a_reuses_the_nan_checks_norm_without_an_extra_kernel(guard_env, monkeypatch):
    chunk = _chunk()
    assert f8_guard.install([chunk]) is not None
    calls = []
    real_norm = torch.Tensor.norm
    monkeypatch.setattr(torch.Tensor, "norm", lambda self, *a, **k: calls.append(1) or real_norm(self, *a, **k))
    f8_guard.begin_step(0)
    expected = {}
    for gi, g in enumerate(chunk.bucket_groups):
        for bi, b in enumerate(g.buckets):
            b.grad_data.copy_(torch.arange(64.0) * (gi + 1) + bi)
            expected[f"0:{gi}:{bi}"] = b.grad_data.norm().item()
    calls.clear()
    for g in chunk.bucket_groups:
        g.start_grad_sync()
    assert len(calls) == 8  # Megatron's check_grads, one per bucket; nothing added
    got = {t: v.item() for t, v in f8_guard._GUARD.local_norms.items()}
    assert got == pytest.approx(expected, rel=1e-6)


def test_a_without_the_nan_check_computes_its_own_norm_before_the_issue(guard_env):
    chunk = _chunk(nan_check=False, scale=0.5)
    f8_guard.install([chunk])
    f8_guard.begin_step(0)
    for b in chunk.bucket_groups[0].buckets:
        b.grad_data.fill_(2.0)
    chunk.bucket_groups[0].start_grad_sync()
    assert f8_guard._GUARD.local_norms["0:0:0"].item() == pytest.approx(16.0)  # 2 x 8, before the 0.5 scaling


@pytest.mark.parametrize("scale", [1.0, 0.25])
def test_b_no_alert_when_nothing_writes_after_the_issue(guard_env, scale, capsys):
    chunk = _chunk(scale=scale)
    f8_guard.install([chunk])
    seen = set()
    for step in range(8):  # every bucket is checked once in 8 steps
        assert _step(chunk, step) == []
        seen |= set(f8_guard._GUARD.pre)
        assert set(f8_guard._GUARD.pre) == set(f8_guard._GUARD.post)
    assert len(seen) == 8
    assert "Q38_F8_ALERT" not in capsys.readouterr().out and not guard_env.exists()


def test_b_negative_control_a_write_after_the_issue_is_caught(guard_env, capsys):
    """+1 on one element outside this rank's shard of a checked bucket after its reduce-scatter was issued."""
    chunk = _chunk()
    f8_guard.install([chunk])
    step = 3  # checks bucket position 2 (group 1, bucket 0): 2 % 8 == 3 % 8 is false; position 3 % 8 == 3
    target = chunk.bucket_groups[1].buckets[1]  # position 3

    def perturb(_):
        target.grad_data[0] += 1.0  # rank 1's shard is [16, 32): element 0 is not this rank's

    assert _step(chunk, step - 1, between=perturb) == ["write_after_issue"]
    out = capsys.readouterr().out
    assert f"Q38_F8_ALERT step={step} rank=0 reason=write_after_issue buckets=['0:1:1']" in out
    record = json.loads((guard_env / f"step-{step}-rank-00000.json").read_text())
    assert record["write_check"]["mismatched"][0]["bucket"] == "0:1:1"
    assert len(record["local_bucket_norms"]) == 8


def test_b_a_write_into_this_ranks_shard_or_an_unchecked_bucket_is_not_flagged(guard_env):
    chunk = _chunk()
    f8_guard.install([chunk])

    def own_shard(c):
        c.bucket_groups[1].buckets[1].grad_data[20] += 1.0  # inside rank 1's shard: the collective's output

    def unchecked(c):
        c.bucket_groups[0].buckets[0].grad_data[0] += 1.0  # position 0, not checked on step 3

    assert _step(chunk, 2, between=own_shard) == []
    assert _step(chunk, 2, between=unchecked) == []


def test_c_grad_norm_alert_after_the_warm_up(guard_env, monkeypatch, capsys):
    monkeypatch.setenv("Q38_F8_GUARD_WARMUP", "3")
    chunk = _chunk()
    f8_guard.install([chunk])
    for step, g in enumerate([1.0, 1.2, 0.9, 1.9, 1.0]):
        assert _step(chunk, step, grad_norm=g) == []  # 1.9 < 2 x median
    assert _step(chunk, 5, grad_norm=2.5) == ["grad_norm"]
    assert "Q38_F8_ALERT step=6 rank=0 reason=grad_norm grad_norm=2.5 median=1" in capsys.readouterr().out
    record = json.loads((guard_env / "step-6-rank-00000.json").read_text())
    assert record["reasons"] == ["grad_norm"] and len(record["local_bucket_norms"]) == 8
    assert _step(chunk, 6, grad_norm=None) == []  # a skipped step: no decision, not in the history


def test_off_switches(guard_env, monkeypatch):
    real = pgb.dist_reduce_scatter_func
    monkeypatch.setenv("Q38_F8_GUARD", "0")
    assert f8_guard.install([_chunk()]) is None and pgb.dist_reduce_scatter_func is real
    monkeypatch.delenv("Q38_F8_GUARD")
    assert f8_guard.install([_chunk(overlap=False)]) is None  # F8 off: not installed
    monkeypatch.setenv("Q38_F8_GUARD_FORCE", "1")
    assert f8_guard.install([_chunk(overlap=False)]) is not None  # an overlap-off control arm
    assert f8_guard.install([_chunk()]) is not None and pgb.dist_reduce_scatter_func._f8_guard_real is real
    f8_guard.begin_step(0)
    f8_guard.end_step(1.0)


def test_cpu_overhead_of_the_python_side(guard_env, capsys):
    """Host-side cost per step with 400 buckets (about the formal model's dense + expert buckets at 320M): wrappers,
    the per-step comparison and bookkeeping. Printed for the report; bounded loosely."""
    chunk = _chunk(n_groups=200, buckets_per_group=2, numel=8)
    f8_guard.install([chunk])
    t0 = time.perf_counter()
    for step in range(20):
        _step(chunk, step)
    per_step = (time.perf_counter() - t0) / 20
    baseline = _chunk(n_groups=200, buckets_per_group=2, numel=8)
    f8_guard._GUARD = None
    t0 = time.perf_counter()
    for step in range(20):
        gen = torch.Generator().manual_seed(step)
        for g in baseline.bucket_groups:
            for b in g.buckets:
                b.grad_data.copy_(torch.randn(b.grad_data.shape, generator=gen))
        for g in baseline.bucket_groups:
            g.start_grad_sync()
        for g in baseline.bucket_groups:
            g.finish_grad_sync()
    base = (time.perf_counter() - t0) / 20
    print(f"F8_GUARD_CPU_OVERHEAD per_step_guarded={per_step * 1e3:.2f} ms baseline={base * 1e3:.2f} ms")
    assert per_step - base < 0.5  # seconds; the real number is in the printed line
