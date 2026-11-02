"""CPU contracts for ddp_trace (Q38_DDP_TRACE, the F8 gate G9's trace; format: ddp_trace's docstring and INT's
ci/INT-gates-final/checkers/g9_check.py).

The simulated DDP of test_qwen38_vision_deferred_grad_sync (Megatron's own bucket-group methods, Megatron's layout, the
real HF vision tower, gradient-accumulator hooks) gets buckets with a flat grad_data holding the step's gradients, as
Megatron's buffers do; two such "ranks" run a text and an image last microbatch.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.megatron_utils.test_qwen38_vision_deferred_grad_sync import _Chunk, _forward, _layout, _model

from miles.backends.megatron_utils import ddp_trace
from miles_plugins.models.qwen3_8_next import vision_grad_sync


def _with_buckets(chunk):
    """Megatron bucket surface per group: params_list (bucket order), param_to_index, flat grad_data."""
    order = [p for _, p in chunk.module.named_parameters()][::-1]  # DDP lays buckets over the reversed list
    chunk.ddp_config.use_distributed_optimizer = False
    for group in chunk.bucket_groups:
        params = [p for p in order if p in group.params]
        index, start = {}, 0
        for p in params:
            index[p] = (start, start + p.numel())
            start += p.numel()
        group.buckets = [SimpleNamespace(params_list=params, param_to_index=index, grad_data=torch.zeros(start))]
        group.data_parallel_group = None
    return chunk


def _fill_grads(chunk):
    for group in chunk.bucket_groups:
        (bucket,) = group.buckets
        for p in bucket.params_list:
            start, end = bucket.param_to_index[p]
            bucket.grad_data[start:end] = p.grad.reshape(-1) if p.grad is not None else 0


def _run(chunk, kinds, monkeypatch, first_index=0):
    """Steps of one microbatch each, as train_one_step drives them: begin_step, forward / backward, DDP
    finish_grad_sync, end_step, reset."""
    for step, kind in enumerate(kinds):
        ddp_trace.begin_step()
        chunk.module.zero_grad(set_to_none=True)
        ddp_trace.on_microbatch({"sample_indices": [first_index + step]})
        _forward(chunk.module, kind, monkeypatch).backward()
        _fill_grads(chunk)
        for group in chunk.bucket_groups:
            group.finish_grad_sync()
        ddp_trace.end_step(step)
        for group in chunk.bucket_groups:
            group.reset()


def _traced_rank(tmp_path, monkeypatch, kinds, rank, *, stats=False, defer=True, bucket_size=64):
    monkeypatch.setenv(ddp_trace.ENV_DIR, str(tmp_path))
    if stats:
        monkeypatch.setenv(ddp_trace.ENV_GRAD_STATS, "1")
    monkeypatch.setattr(ddp_trace, "_rank", lambda: rank)
    model = _model()
    chunk = _with_buckets(_Chunk(model, _layout(model, bucket_size)))
    if defer:
        vision_grad_sync.defer_vision_bucket_groups([chunk])
    assert ddp_trace.install([chunk]) is not None
    _run(chunk, kinds, monkeypatch, first_index=10 * rank)
    return [json.loads(x) for x in (tmp_path / f"rank-{rank:05d}.jsonl").read_text().splitlines()], chunk


def _bucket_lines(lines):
    return {(x["step"], x["comm"], x["group"]): x["buckets"] for x in lines if "buckets" in x}


def test_trace_lines_match_the_g9_format(tmp_path, monkeypatch):
    lines, chunk = _traced_rank(tmp_path, monkeypatch, ["image", "text", "image"], 0, stats=True)
    buckets = _bucket_lines(lines)
    assert sorted(buckets) == [(s, "dp", "local") for s in (1, 2, 3)]
    for (step, _, _), entries in buckets.items():
        assert [e["seq"] for e in entries] == list(range(len(chunk.bucket_groups)))
        assert all(e["count"] == 1 and len(e["sig"]) == 64 for e in entries)
        deferred = {i for i, g in enumerate(chunk.bucket_groups) if getattr(g, vision_grad_sync.MARKER, False)}
        for e in entries:  # step 1: Megatron issues every group at finish; then only the deferred vision groups
            assert e["source"] == ("finish" if step == 1 or e["index"] in deferred else "hook"), (step, e)
    mbs = {x["step"]: x["mb_sample_indices"] for x in lines if "mb_sample_indices" in x}
    assert mbs == {1: [[0]], 2: [[1]], 3: [[2]]}
    stats = {x["step"]: x["grad_stats"] for x in lines if "grad_stats" in x}
    assert sorted(stats) == [1, 2, 3]
    names = {p: f"module.{n}" for n, p in chunk.module.named_parameters()}
    for p, name in names.items():  # step 3's reduced gradient (here: the one rank's)
        s = stats[3][name]
        assert s["numel"] == p.numel()
        assert s["l2"] == pytest.approx(p.grad.double().norm().item(), rel=1e-12)
        assert s["max_abs"] == pytest.approx(p.grad.abs().max().item(), rel=1e-12)
    vision = [n for n in stats[2] if ".vision_model." in n]
    assert vision and all(stats[2][n]["l2"] == 0 for n in vision)  # step 2 is text: exact zero tower grads


def test_two_ranks_with_text_and_image_last_microbatches_trace_the_same_buckets(tmp_path, monkeypatch):
    """G9 check 1 on the simulation: a text-last and an image-last rank, every step the same (seq, sig, count)."""
    a, _ = _traced_rank(tmp_path, monkeypatch, ["image", "text", "image", "two_images"], 0)
    b, _ = _traced_rank(tmp_path, monkeypatch, ["image", "image", "text", "image_off_this_tp_slice"], 1)
    key = lambda lines: {k: [(e["seq"], e["sig"], e["count"]) for e in v] for k, v in _bucket_lines(lines).items()}
    assert key(a) == key(b) and len(key(a)) == 4


def test_without_the_deferral_the_trace_shows_the_mismatch(tmp_path, monkeypatch):
    """Positive control for the gate: CX4's layout without vision_grad_sync traces different issue orders."""
    a, _ = _traced_rank(tmp_path, monkeypatch, ["image", "text"], 0, defer=False, bucket_size=65)
    b, _ = _traced_rank(tmp_path, monkeypatch, ["image", "image"], 1, defer=False, bucket_size=65)
    key = lambda lines: [(e["seq"], e["sig"]) for e in _bucket_lines(lines)[(2, "dp", "local")]]
    assert key(a) != key(b)


def test_record_ids_come_from_data_jsonl(tmp_path, monkeypatch):
    trace = tmp_path / "trace"
    _traced_rank(trace, monkeypatch, ["image", "text"], 0)
    _traced_rank(trace, monkeypatch, ["text", "image"], 1)
    data = tmp_path / "data.jsonl"  # rank 0 traced samples 0, 1 and rank 1 10, 11: not one step's two records
    data.write_text(json.dumps({"step": 1, "record_ids": ["a", "b"]}) + "\n")
    with pytest.raises(ValueError, match="traced sample indices"):
        ddp_trace.add_record_ids(trace, data)

    trace2 = tmp_path / "trace2"
    for rank in (0, 1):
        lines = [
            {"step": 1, "rank": rank, "mb_sample_indices": [[2 * rank]]},
            {"step": 2, "rank": rank, "mb_sample_indices": [[4 + rank], None]},
        ]
        trace2.mkdir(exist_ok=True)
        (trace2 / f"rank-{rank:05d}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    data.write_text(
        json.dumps({"step": 1, "record_ids": ["r0", "r1", "r2"]})
        + "\n"
        + json.dumps({"step": 2, "record_ids": ["s0", "s1"]})
        + "\n"
    )
    with pytest.raises(ValueError):  # step 1: indices 0, 2 for three records
        ddp_trace.add_record_ids(trace2, data)
    data.write_text(
        json.dumps({"step": 1, "record_ids": ["r0", "r1"]})
        + "\n"
        + json.dumps({"step": 2, "record_ids": ["s0", "s1"]})
    )
    (trace2 / "rank-00001.jsonl").write_text(
        json.dumps({"step": 1, "rank": 1, "mb_sample_indices": [[1]]})
        + "\n"
        + json.dumps({"step": 2, "rank": 1, "mb_sample_indices": [[5], None]})
        + "\n"
    )
    assert ddp_trace.add_record_ids(trace2, data) == 4
    assert ddp_trace.add_record_ids(trace2, data) == 0  # idempotent
    got = {}
    for rank in (0, 1):
        for x in map(json.loads, (trace2 / f"rank-{rank:05d}.jsonl").read_text().splitlines()):
            if "mb_record_ids" in x:
                got[(x["step"], rank)] = x["mb_record_ids"]
    assert got == {(1, 0): ["r0"], (1, 1): ["r1"], (2, 0): ["s0", None], (2, 1): ["s1", None]}


def test_sharded_grad_stats_combine_to_the_whole_parameter(tmp_path, monkeypatch):
    """With the distributed optimizer each rank holds its reduce-scatter shard; sum of squares / max over the shards
    give the whole parameter's (the all-reduce over the communicator, done by hand here)."""
    _, chunk = _traced_rank(tmp_path, monkeypatch, ["image"], 0, stats=True)
    chunk.ddp_config.use_distributed_optimizer = True
    parts = []
    for shard_rank in (0, 1):
        for group in chunk.bucket_groups:
            n = group.buckets[0].grad_data.numel()
            pad = (-n) % 2  # Megatron pads buckets to the DP size
            group.buckets[0].grad_data = torch.cat([group.buckets[0].grad_data[:n], torch.zeros(pad)])
            group.intra_distributed_optimizer_instance_group = None
            group.intra_distributed_optimizer_instance_size = 2
            group.intra_distributed_optimizer_instance_rank = shard_rank
        parts.append(ddp_trace._TRACE._grad_stats())
    for name, p in ((f"module.{n}", p) for n, p in chunk.module.named_parameters()):
        l2 = (parts[0][name]["l2"] ** 2 + parts[1][name]["l2"] ** 2) ** 0.5
        assert l2 == pytest.approx(p.grad.double().norm().item(), rel=1e-12)
        assert max(parts[0][name]["max_abs"], parts[1][name]["max_abs"]) == p.grad.abs().max().item()


def test_off_by_default_wraps_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(ddp_trace.ENV_DIR, raising=False)
    model = _model()
    chunk = _with_buckets(_Chunk(model, _layout(model, 64)))
    before = [(g.start_grad_sync, g.finish_grad_sync) for g in chunk.bucket_groups]
    assert ddp_trace.install([chunk]) is None
    assert [(g.start_grad_sync, g.finish_grad_sync) for g in chunk.bucket_groups] == before
    ddp_trace.begin_step()
    ddp_trace.on_microbatch({"sample_indices": [0]})
    ddp_trace.end_step(0)
    assert not list(tmp_path.iterdir())
