"""CPU contracts for grad_ready_check (CX3-F8-02): terminal checks of Megatron DDP's gradient-ready counts.

Stand-in bucket groups run Megatron's own register_grad_ready / finish_grad_sync / reset (fdf82356
_ParamAndGradBucketGroup), driven by hooks on each parameter's gradient accumulator as Megatron DDP registers them;
start_grad_sync records the issue instead of a reduce-scatter.

CX3's positive control: one weight shared by two reentrant checkpoints. The first step uses it once (golden 1); a later
step uses it in both checkpoints, so it reports twice, the second time after its group was issued. Megatron alone
reaches the optimizer step with the extra contribution added after the issue; with the check the backward raises.
"""

import types
from types import SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint as checkpoint
from tests.fast.backends.megatron_utils.test_qwen38_vision_deferred_grad_sync import _Chunk as _VisionChunk
from tests.fast.backends.megatron_utils.test_qwen38_vision_deferred_grad_sync import _issue_orders, _layout
from tests.fast.backends.megatron_utils.test_qwen38_vision_deferred_grad_sync import _model as _vision_model

from miles.backends.megatron_utils import grad_ready_check
from miles_plugins.models.qwen3_8_next import vision_grad_sync


class _Shared(torch.nn.Module):
    """shared: the weight CX3's control reuses; other: a second group's weight."""

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(4, 4)
        self.other = torch.nn.Linear(4, 4)

    def block(self, x):
        return torch.tanh(self.shared(x))

    def forward(self, x, twice):
        h = checkpoint.checkpoint(self.block, self.other(x), use_reentrant=True)
        if twice:
            h = checkpoint.checkpoint(self.block, h, use_reentrant=True)
        return h.square().mean()


class _Groups:
    """Megatron's bucket-group methods on stand-ins: one group per module of _Shared (index 0 shared, 1 other)."""

    def __init__(self, model, checked):
        from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup as Group

        self.model, self.issued = model, []
        self.ddp_config = SimpleNamespace(overlap_grad_reduce=True, num_distributed_optimizer_instances=1)
        self.bucket_groups, self.expert_parallel_bucket_groups = [], []
        for index, module in enumerate((model.shared, model.other)):
            params = set(module.parameters())
            group = SimpleNamespace(
                ddp_config=self.ddp_config,
                params=params,
                param_to_bucket={p: index for p in params},
                is_last_microbatch=True,
                is_first_batch=True,
                per_param_grad_ready_counts={},
                golden_per_param_grad_ready_counts={},
                grad_reduce_handle=None,
                grad_reduce_finished=False,
                param_gather_dispatched=False,
                _copy_back_extra_main_grads=lambda: None,
            )
            for name in ("register_grad_ready", "finish_grad_sync", "reset"):
                setattr(group, name, types.MethodType(getattr(Group, name), group))
            group.start_grad_sync = self._recorder(group, index)
            self.bucket_groups.append(group)
        if checked:
            assert grad_ready_check.install([self]) == 2
        self.group_of = {p: g for g in self.bucket_groups for p in g.params}
        self._accs = []
        for p in self.group_of:
            acc = p.expand_as(p).grad_fn.next_functions[0][0]
            acc.register_hook(lambda *unused, p=p: self.group_of[p].register_grad_ready(p))
            self._accs.append(acc)

    def named_parameters(self):
        return self.model.named_parameters()

    def _recorder(self, group, index):
        def start_grad_sync(force_all_reduce=False):
            if group.is_first_batch and group.grad_reduce_handle is not None:
                return
            self.issued.append(index)
            group.grad_reduce_handle = SimpleNamespace(wait=lambda: None)

        return start_grad_sync

    def step(self, twice, microbatches=1):
        """forward / backward per microbatch (all but the last under no_sync), finish_grad_sync, reset."""
        self.issued.clear()
        self.model.zero_grad(set_to_none=True)
        for mb in range(microbatches):
            last = mb == microbatches - 1
            for g in self.bucket_groups:  # DDP.no_sync
                g.is_last_microbatch = last
            self.model(torch.ones(2, 4) * (mb + 1), twice).backward()
        for g in self.bucket_groups:
            g.finish_grad_sync()
        counts = {n: g.per_param_grad_ready_counts.get(p) for g in self.bucket_groups for n, p in self._named(g)}
        for g in self.bucket_groups:
            g.reset()
        return counts

    def _named(self, group):
        return [(n, p) for n, p in self.model.named_parameters() if p in group.params]


def test_positive_control_megatron_alone_lets_a_late_report_through():
    """Red without the check: the second use reports after the issue and the step completes (count 2, golden 1)."""
    torch.manual_seed(0)
    groups = _Groups(_Shared(), checked=False)
    groups.step(twice=False)
    counts = groups.step(twice=True)
    assert counts["shared.weight"] == 2 and groups.issued.count(0) == 1  # issued once, at count 1


def test_a_report_after_the_issue_raises_before_the_optimizer_step():
    torch.manual_seed(0)
    groups = _Groups(_Shared(), checked=True)
    groups.step(twice=False)
    with pytest.raises(Exception, match="reported a gradient after the group's reduce-scatter was issued"):
        groups.step(twice=True)


def test_a_count_that_differs_from_golden_raises_at_finish():
    """A parameter that reports less than golden: the group is never issued; finish names the parameter."""
    torch.manual_seed(0)
    groups = _Groups(_Shared(), checked=True)
    groups.step(twice=False)
    groups.issued.clear()
    groups.model.zero_grad(set_to_none=True)
    groups.model(torch.ones(2, 4), False).backward()
    shared = groups.bucket_groups[0]
    shared.per_param_grad_ready_counts.pop(groups.model.shared.bias)  # as if the bias had not reported
    with pytest.raises(grad_ready_check.GradReadyCountError, match=r"shared.bias: count 0, golden 1"):
        for g in groups.bucket_groups:
            g.finish_grad_sync()


@pytest.mark.parametrize("microbatches", [1, 3])
def test_no_false_alarm_on_the_first_step_no_sync_or_steady_steps(microbatches):
    torch.manual_seed(0)
    groups = _Groups(_Shared(), checked=True)
    first = groups.step(twice=True, microbatches=microbatches)  # golden: shared reports 2
    assert first["shared.weight"] == 2 and sorted(groups.issued) == [0, 1]
    for _ in range(3):
        assert groups.step(twice=True, microbatches=microbatches)["shared.weight"] == 2
        assert sorted(groups.issued) == [0, 1]


@pytest.mark.parametrize("bucket_size", [33, 65, 20_000_000])
def test_the_vision_deferral_and_the_check_together_keep_one_issue_order(bucket_size, monkeypatch):
    """The deferred vision groups are checked too (installed after vision_grad_sync); every microbatch kind issues
    alike and nothing false-alarms."""
    model = _vision_model()
    chunk = _VisionChunk(model, _layout(model, bucket_size))
    vision_grad_sync.defer_vision_bucket_groups([chunk])
    assert grad_ready_check.install([chunk]) == len(chunk.bucket_groups)
    chunk.step("image", monkeypatch)
    orders = [chunk.step(kind, monkeypatch) for kind in ("text", "image", "two_images", "image_off_this_tp_slice")]
    assert len({tuple(o) for o in orders}) == 1
    assert orders == list(_issue_orders(_vision_model(), _layout(model, bucket_size), monkeypatch).values())


def test_a_late_report_on_a_deferred_vision_group_after_finish_raises(monkeypatch):
    model = _vision_model()
    chunk = _VisionChunk(model, _layout(model, 64))
    vision_grad_sync.defer_vision_bucket_groups([chunk])
    grad_ready_check.install([chunk])
    chunk.step("image", monkeypatch)
    chunk.step("text", monkeypatch)  # reset() ran; now emulate a finished vision group getting one more report
    group = next(g for g in chunk.bucket_groups if getattr(g, vision_grad_sync.MARKER, False))
    param = next(iter(group.params))
    group.grad_reduce_finished = True
    with pytest.raises(grad_ready_check.GradReadyCountError, match="after the group's reduce-scatter was issued"):
        group.register_grad_ready(param)


def test_only_overlap_chunks_are_wrapped():
    groups = _Groups(_Shared(), checked=False)
    groups.ddp_config.overlap_grad_reduce = False
    assert grad_ready_check.install([groups]) == 0
    assert grad_ready_check.install([torch.nn.Linear(2, 2)]) == 0
