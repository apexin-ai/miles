"""CPU contracts for issue_order_check (CX3 F8-RC): ranks that issue their bucket groups in different orders, or hold
different bucket layouts, all stop before the optimizer step.

Real processes (gloo: the all-reduces the check runs) issue stand-in groups' reductions in chosen orders, like the
F8STRESS swap injection; with 4 ranks, two dense communicators ({0, 1}, {2, 3}) and two expert ones ({0, 2}, {1, 3})
show that a mismatch on one communicator stops every rank. The simulated DDP of test_qwen38_vision_deferred_grad_sync
(Megatron's own bucket-group methods and layout, the real HF vision tower, gradient-accumulator hooks) shows what the
check records: step 1's groups at finish, hook-issued groups, the deferred vision groups, no_sync microbatches.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.backends.megatron_utils.test_ddp_trace import _with_buckets
from tests.fast.backends.megatron_utils.test_qwen38_vision_deferred_grad_sync import _Chunk, _forward, _layout, _model

from miles.backends.megatron_utils import issue_order_check
from miles_plugins.models.qwen3_8_next import vision_grad_sync

# ---- real processes, real all-reduces ------------------------------------------------------------------------------


class _Group(SimpleNamespace):
    def start_grad_sync(self, force_all_reduce=False):
        if self.is_first_batch and self.grad_reduce_handle is not None:
            return
        self.grad_reduce_handle = SimpleNamespace(wait=lambda: None)

    def finish_grad_sync(self, force_all_reduce=False):
        if self.grad_reduce_handle is None:
            self.start_grad_sync()
        self.grad_reduce_handle = None


class _DDP(torch.nn.Module):
    """DDP's surface: ddp_config, bucket_groups / expert_parallel_bucket_groups, named_parameters, finish_grad_sync."""

    def __init__(self, dense_pg=None, expert_pg=None, n_dense=3, n_expert=2, numels=None, overlap=True):
        super().__init__()
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=overlap, use_distributed_optimizer=True)
        numels = numels or [8] * (n_dense + n_expert)
        self.weights = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(n)) for n in numels)
        self.bucket_groups, self.expert_parallel_bucket_groups = [], []
        for i, p in enumerate(self.weights):
            group = _Group(
                ddp_config=self.ddp_config,
                buckets=[
                    SimpleNamespace(params_list=[p], gradient_scaling_factor=1.0, grad_data=torch.zeros(p.numel()))
                ],
                intra_distributed_optimizer_instance_group=dense_pg if i < n_dense else expert_pg,
                is_first_batch=True,
                grad_reduce_handle=None,
            )
            (self.bucket_groups if i < n_dense else self.expert_parallel_bucket_groups).append(group)

    def finish_grad_sync(self):
        for g in self.bucket_groups + self.expert_parallel_bucket_groups:
            g.finish_grad_sync()


def _worker(rank, path, world, plans, numels, checked):
    dist.init_process_group("gloo", init_method=f"file://{path}/store", rank=rank, world_size=world)
    if world == 2:
        dense_pg = expert_pg = dist.new_group([0, 1])
    else:  # every rank creates every group, in one order
        dense = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        expert = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        dense_pg, expert_pg = dense[rank // 2], expert[rank % 2]
    results = []
    try:
        chunk = _DDP(dense_pg, expert_pg, numels=numels[rank])
        if checked:
            issue_order_check.install([chunk])
    except issue_order_check.IssueOrderError as e:
        results.append(f"install raised: {e}")
        plans = [[] for _ in plans]
    groups = [] if results else chunk.bucket_groups + chunk.expert_parallel_bucket_groups
    for step, plan in enumerate(plans[rank]):
        for i in plan:  # this step's hook-issued groups, in this rank's order
            groups[i].start_grad_sync()
        chunk.finish_grad_sync()  # the rest at finish, in index order (DDP.finish_grad_sync)
        try:
            issue_order_check.verify_step()  # once per step, after every group finished
            results.append("ok")
        except issue_order_check.IssueOrderError as e:
            results.append(f"raised: {e}")
        if step == 0:
            for g in groups:
                g.is_first_batch = False
    (path / f"rank{rank}.json").write_text(json.dumps(results))
    dist.destroy_process_group()


def _ranks(tmp_path, plans, numels=None, checked=True):
    world = len(plans)
    numels = numels or [None] * world
    mp.spawn(_worker, args=(tmp_path, world, plans, numels, checked), nprocs=world, join=True)
    return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(world)]


STEP1 = []  # step 1: every group at finish
SAME = [2, 1, 0, 4, 3]  # hook-issued, dense then expert
SWAP_DENSE = [1, 2, 0, 4, 3]
SWAP_EXPERT = [2, 1, 0, 3, 4]
PARTIAL = [2, 1]  # the rest (0, 3, 4) at finish


def test_two_ranks_one_order_passes_and_a_swapped_order_raises(tmp_path):
    r0, r1 = _ranks(tmp_path, [[STEP1, SAME, SAME, SAME, PARTIAL], [STEP1, SAME, SWAP_DENSE, SWAP_EXPERT, PARTIAL]])
    assert r0[:2] == r1[:2] == ["ok", "ok"] and r0[4] == r1[4] == "ok"
    for results in (r0, r1):
        assert results[2].startswith("raised:") and "0:dense" in results[2]
        assert results[3].startswith("raised:") and "0:expert" in results[3]


def test_red_without_the_check_the_same_swap_goes_through_silently(tmp_path):
    """The F8STRESS-style swap with the check not installed (the base 09a5adcf): every step completes."""
    r0, r1 = _ranks(tmp_path, [[STEP1, SAME, SAME], [STEP1, SAME, SWAP_DENSE]], checked=False)
    assert r0 == r1 == ["ok", "ok", "ok"]


def test_four_ranks_a_mismatch_on_one_communicator_stops_every_rank(tmp_path):
    """Rank 1 swaps its dense order: only communicator {0, 1} disagrees; ranks 2 and 3 stop too (world flag)."""
    results = _ranks(
        tmp_path, [[STEP1, SAME, SAME], [STEP1, SAME, SWAP_DENSE], [STEP1, SAME, SAME], [STEP1, SAME, SAME]]
    )
    for r, res in enumerate(results):
        assert res[:2] == ["ok", "ok"], r
        assert res[2].startswith("raised:"), r
    assert "0:dense" in results[0][2] and "0:dense" in results[1][2]
    assert "another rank's communicator" in results[2][2] and "another rank's communicator" in results[3][2]


def test_a_different_bucket_layout_stops_every_rank_at_install(tmp_path):
    """Same group count, rank 1's second dense group has another numel: the layout check fails before training."""
    results = _ranks(tmp_path, [[STEP1], [STEP1]], numels=[[8] * 5, [8, 16, 8, 8, 8]])
    for res in results:
        assert len(res) == 1 and res[0].startswith("install raised:") and "layout" in res[0]


def test_the_hash_depends_on_order_and_length():
    h = issue_order_check.sequence_hash
    assert h([0, 1, 2]) == h([0, 1, 2]) and h([0, 1, 2]) != h([1, 0, 2]) and h([]) != h([0])
    assert all(-(2**63) <= h(s) < 2**63 for s in ([], [0], list(range(400)), [2**63 - 1, -(2**63)]))


# ---- the simulated DDP: what one rank records ----------------------------------------------------------------------


class _SimDDP(_Chunk):
    def finish_grad_sync(self):
        for g in self.bucket_groups + self.expert_parallel_bucket_groups:
            g.finish_grad_sync()

    def run(self, kind, monkeypatch, check, microbatches=1):
        self.issued.clear()
        self.module.zero_grad(set_to_none=True)
        for mb in range(microbatches):
            for g in self.bucket_groups:
                g.is_last_microbatch = mb == microbatches - 1  # DDP.no_sync for all but the last
            _forward(self.module, kind, monkeypatch).backward()
        self.finish_grad_sync()
        recorded = list(check.issued["dense"])
        issue_order_check.verify_step()
        for g in self.bucket_groups:
            g.reset()
        assert recorded == self.issued  # the check saw exactly what Megatron issued, in that order
        return recorded


def _sim(bucket_size, defer=True):
    model = _model()
    chunk = _with_buckets(_SimDDP(model, _layout(model, bucket_size)))
    for g in chunk.bucket_groups:
        g.buckets[0].gradient_scaling_factor = 1.0
    if defer:
        vision_grad_sync.defer_vision_bucket_groups([chunk])
    (check,) = issue_order_check.install([chunk])
    return chunk, check


@pytest.mark.parametrize("microbatches", [1, 3])
def test_it_records_step_one_hook_issued_and_deferred_groups_once_each(monkeypatch, microbatches):
    chunk, check = _sim(64)
    n = len(chunk.bucket_groups)
    deferred = [i for i, g in enumerate(chunk.bucket_groups) if getattr(g, vision_grad_sync.MARKER, False)]
    recorded = [chunk.run(k, monkeypatch, check, microbatches) for k in ("image", "text", "image", "two_images")]
    assert recorded[0] == list(range(n))  # step 1: all at finish, index order
    for seq in recorded[1:]:
        assert sorted(seq) == list(range(n)) and seq[-len(deferred) :] == deferred
    assert len({tuple(s) for s in recorded[1:]}) == 1  # text / image / two images: one sequence


def test_without_the_deferral_text_and_image_hash_differently(monkeypatch):
    """What the all-reduce would catch between a text-last and an image-last rank (CX4's layout, bucket 65)."""
    chunk, check = _sim(65, defer=False)
    sigs = check.signatures["dense"]
    seqs = [chunk.run(k, monkeypatch, check) for k in ("image", "text", "image")]
    h = issue_order_check.sequence_hash
    assert h([sigs[i] for i in seqs[1]]) != h([sigs[i] for i in seqs[2]])


def test_signatures_cover_names_numels_dtypes_and_scaling():
    base = _DDP()
    names = {p: n for n, p in base.named_parameters()}
    sig = issue_order_check.group_signature
    g = base.bucket_groups[0]
    s0 = sig(g, names)
    assert sig(g, {p: n + "x" for p, n in names.items()}) != s0
    g.buckets[0].gradient_scaling_factor = 0.5
    assert sig(g, names) != s0
    other = _DDP(numels=[9, 8, 8, 8, 8])
    assert sig(other.bucket_groups[0], {p: n for n, p in other.named_parameters()}) != s0


def test_without_overlap_it_is_a_no_op_with_no_collective(monkeypatch):
    """Overlap off (the launch config): nothing is wrapped and no collective runs, so the code can sit in a launch."""

    def forbidden(*args, **kwargs):
        raise AssertionError("no collective without --overlap-grad-reduce")

    monkeypatch.setattr(dist, "all_reduce", forbidden)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)  # as in a real run: only overlap may trigger it
    chunk = _DDP(overlap=False)
    groups = chunk.bucket_groups + chunk.expert_parallel_bucket_groups
    before = [(g.start_grad_sync, g.finish_grad_sync) for g in groups]
    assert issue_order_check.install([chunk]) == [] and issue_order_check.install([torch.nn.Linear(2, 2)]) == []
    assert [(g.start_grad_sync, g.finish_grad_sync) for g in groups] == before
    assert not any(getattr(g, issue_order_check.MARKER, False) for g in groups)
    for _ in range(3):
        for g in groups[:2]:
            g.start_grad_sync()
        chunk.finish_grad_sync()
        issue_order_check.verify_step()


def test_cpu_cost_of_one_verify_with_400_groups(capsys):
    """Host cost of the per-step hashing, tensors and read for 400 issued groups (one process: no all-reduce)."""
    import time

    chunk = _DDP(n_dense=100, n_expert=300)
    issue_order_check.install([chunk])
    groups = chunk.bucket_groups + chunk.expert_parallel_bucket_groups
    for g in groups:
        g.is_first_batch = False
    t0 = time.perf_counter()
    for _ in range(50):
        for g in groups:
            g.start_grad_sync()
        chunk.finish_grad_sync()
        issue_order_check.verify_step()
    per_step = (time.perf_counter() - t0) / 50
    print(f"ISSUE_ORDER_CHECK_CPU per_step={per_step * 1e3:.3f} ms (400 groups, incl. the stand-in issue / finish)")
    assert per_step < 0.1
