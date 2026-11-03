"""CPU contracts for issue_order_check (CX3 F8-RC): ranks that issue their bucket groups in different orders fail.

Two real processes (gloo, the all-reduce the check runs) issue stand-in groups' reductions in chosen orders; and the
simulated DDP of test_qwen38_vision_deferred_grad_sync (Megatron's own bucket-group methods and layout, the real HF
vision tower, gradient-accumulator hooks) shows what the check records: step 1's groups at finish, hook-issued groups,
the deferred vision groups, no_sync microbatches.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.backends.megatron_utils.test_ddp_trace import _with_buckets
from tests.fast.backends.megatron_utils.test_qwen38_vision_deferred_grad_sync import _Chunk, _forward, _layout, _model

from miles.backends.megatron_utils import issue_order_check
from miles_plugins.models.qwen3_8_next import vision_grad_sync

# ---- two processes, real all-reduce --------------------------------------------------------------------------------


class _Group(SimpleNamespace):
    def start_grad_sync(self, force_all_reduce=False):
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return
        self.grad_reduce_handle = SimpleNamespace(wait=lambda: None)

    def finish_grad_sync(self, force_all_reduce=False):
        if self.grad_reduce_handle is None:
            self.start_grad_sync()
        self.grad_reduce_handle = None


class _DDP(SimpleNamespace):
    def finish_grad_sync(self):
        for g in self.bucket_groups + self.expert_parallel_bucket_groups:
            g.finish_grad_sync()


def _ddp(n_dense=3, n_expert=2):
    ddp_config = SimpleNamespace(overlap_grad_reduce=True, use_distributed_optimizer=True)

    def group():
        return _Group(
            ddp_config=ddp_config,
            buckets=[SimpleNamespace(grad_data=torch.zeros(4))],
            intra_distributed_optimizer_instance_group=dist.group.WORLD,
            is_first_batch=True,
            grad_reduce_handle=None,
        )

    return _DDP(
        ddp_config=ddp_config,
        bucket_groups=[group() for _ in range(n_dense)],
        expert_parallel_bucket_groups=[group() for _ in range(n_expert)],
    )


def _worker(rank, path, plans):
    dist.init_process_group("gloo", init_method=f"file://{path}/store", rank=rank, world_size=2)
    chunk = _ddp()
    issue_order_check.install([chunk])
    results = []
    groups = chunk.bucket_groups + chunk.expert_parallel_bucket_groups
    for step, plan in enumerate(plans[rank]):
        try:
            for i in plan:  # the hook-issued groups of this step, in this rank's order
                groups[i].start_grad_sync()
            chunk.finish_grad_sync()  # the rest at finish, in index order
            results.append("ok")
        except issue_order_check.IssueOrderError as e:
            results.append(f"raised: {e}")
            for g in groups:
                g.grad_reduce_handle = None
        if step == 0:
            for g in groups:
                g.is_first_batch = False
    (path / f"rank{rank}.json").write_text(json.dumps(results))
    dist.destroy_process_group()


def _two_ranks(tmp_path, plan0, plan1):
    mp.spawn(_worker, args=(tmp_path, [plan0, plan1]), nprocs=2, join=True)
    return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]


def test_two_ranks_with_one_order_pass_and_swapped_orders_raise(tmp_path):
    step1 = []  # step 1: every group at finish
    same = [2, 1, 0, 4, 3]  # hook-issued, dense then expert
    swapped_dense = [1, 2, 0, 4, 3]
    swapped_expert = [2, 1, 0, 3, 4]
    partial = [2, 1]  # the rest (0, 3, 4) at finish
    r0, r1 = _two_ranks(
        tmp_path,
        [step1, same, same, same, partial],
        [step1, same, swapped_dense, swapped_expert, partial],
    )
    assert r0[:2] == r1[:2] == ["ok", "ok"]
    assert r0[4] == r1[4] == "ok"
    for results in (r0, r1):  # both ranks fail closed, naming the communicator
        assert results[2].startswith("raised:") and "['dense']" in results[2]
        assert results[3].startswith("raised:") and "['expert']" in results[3]
    assert "dense=[1, 2, 0]" in r1[2] and "dense=[2, 1, 0]" in r0[2]


def test_the_hash_depends_on_order_and_length():
    h = issue_order_check.sequence_hash
    assert h([0, 1, 2]) == h([0, 1, 2]) and h([0, 1, 2]) != h([1, 0, 2]) and h([]) != h([0])
    assert all(-(2**63) <= h(s) < 2**63 for s in ([], [0], list(range(400))))


# ---- the simulated DDP: what one rank records ----------------------------------------------------------------------


class _SimDDP(_Chunk):
    def finish_grad_sync(self):
        for g in self.bucket_groups + self.expert_parallel_bucket_groups:
            g.finish_grad_sync()

    def run(self, kind, monkeypatch, microbatches=1):
        self.issued.clear()
        self.module.zero_grad(set_to_none=True)
        for mb in range(microbatches):
            for g in self.bucket_groups:
                g.is_last_microbatch = mb == microbatches - 1  # DDP.no_sync for all but the last
            _forward(self.module, kind, monkeypatch).backward()
        self.finish_grad_sync()  # the order check runs here
        for g in self.bucket_groups:
            g.reset()
        return list(self.issued)


def _sim(bucket_size, defer=True):
    model = _model()
    chunk = _with_buckets(_SimDDP(model, _layout(model, bucket_size)))
    if defer:
        vision_grad_sync.defer_vision_bucket_groups([chunk])
    (check,) = issue_order_check.install([chunk])
    recorded = []
    verify = check.verify

    def spy():
        recorded.append({k: list(v) for k, v in check.issued.items()})
        verify()

    check.verify = spy
    return chunk, recorded


@pytest.mark.parametrize("microbatches", [1, 3])
def test_it_records_step_one_hook_issued_and_deferred_groups_once_each(monkeypatch, microbatches):
    chunk, recorded = _sim(64)
    n = len(chunk.bucket_groups)
    deferred = [i for i, g in enumerate(chunk.bucket_groups) if getattr(g, vision_grad_sync.MARKER, False)]
    for kind in ("image", "text", "image", "two_images"):
        issued = chunk.run(kind, monkeypatch, microbatches)
        assert recorded[-1]["dense"] == issued  # exactly what Megatron issued, in that order
    assert recorded[0]["dense"] == list(range(n))  # step 1: all at finish, index order
    for seq in recorded[1:]:
        assert sorted(seq["dense"]) == list(range(n)) and seq["dense"][-len(deferred) :] == deferred
    assert len({json.dumps(s) for s in recorded[1:]}) == 1  # text / image / two images: one sequence


def test_without_the_deferral_text_and_image_record_different_sequences(monkeypatch):
    """What the all-reduce would catch between a text-last and an image-last rank (CX4's layout, bucket 65)."""
    chunk, recorded = _sim(65, defer=False)
    chunk.run("image", monkeypatch)
    chunk.run("text", monkeypatch)
    chunk.run("image", monkeypatch)
    h = issue_order_check.sequence_hash
    assert h(recorded[1]["dense"]) != h(recorded[2]["dense"])


def test_only_overlap_chunks():
    chunk = _ddp()
    chunk.ddp_config.overlap_grad_reduce = False
    assert issue_order_check.install([chunk]) == [] and issue_order_check.install([torch.nn.Linear(2, 2)]) == []


def test_cpu_cost_of_one_verify_with_400_groups(capsys):
    """Host cost of the per-step hash + tensor + read for 400 issued groups (one process: no all-reduce)."""
    import time

    chunk = _ddp(n_dense=100, n_expert=300)
    (check,) = issue_order_check.install([chunk])
    groups = chunk.bucket_groups + chunk.expert_parallel_bucket_groups
    t0 = time.perf_counter()
    for _ in range(50):
        for g in groups:
            g.start_grad_sync()
        chunk.finish_grad_sync()
    per_step = (time.perf_counter() - t0) / 50
    print(f"ISSUE_ORDER_CHECK_CPU per_step={per_step * 1e3:.3f} ms (400 groups, incl. the stand-in issue / finish)")
    assert per_step < 0.1


@pytest.fixture(autouse=True)
def _no_leftover_env():
    os.environ.pop("MASTER_ADDR", None)
    yield
