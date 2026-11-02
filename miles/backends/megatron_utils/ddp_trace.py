"""Opt-in trace of Megatron DDP's gradient reductions, for the F8 (--overlap-grad-reduce) acceptance gate G9.

Off unless the environment sets Q38_DDP_TRACE=<directory>; then every rank appends JSON lines to
<directory>/rank-<global rank, 5 digits>.jsonl (the gate uses <run>/audit/ddp-trace). Off, nothing is wrapped and the
per-microbatch call is one global lookup.

Per optimizer step (step = rollout_id + 1, as progress/train.jsonl), after the backward and finalize_model_grads:
  {"step", "rank", "comm", "group", "buckets": [...]}   one line per communicator this rank reduces on
      comm: "dp" (dense grads, CP 1), "dp_cp" (dense grads over DP x CP), "expert_dp" (expert grads)
      group: the communicator's global ranks, "r0,r1,..." (the same string on every member rank)
      buckets: one entry per DDP bucket group on that communicator, in issue order:
        {"seq": issue order in this step (-1: not issued), "sig": sha256 of the group's parameter names (bucket
         order), "count": issues this step, "source": "hook" (from register_grad_ready during the backward) or
         "finish" (from finish_grad_sync), "index": the group's index in DDP, "numel": its elements}
  {"step", "rank", "mb_sample_indices": [[sample index, ...] per training microbatch, in order]}
  with Q38_DDP_TRACE_GRAD_STATS=1, before the optimizer step:
  {"step", "rank", "grad_stats": {parameter name: {"l2", "max_abs", "numel"}}}
      of the reduced gradient: each rank's reduce-scatter shard, combined over the communicator (sum of squares /
      max), so every rank of a communicator holds the whole parameter's values; comparable between bucket layouts
      (overlap on: many buckets; off: one bucket per buffer). Extra collectives: debugging only.

Record ids: miles' trainer has the sample indices, not the data's record ids. After the run,
  python3 -m miles.backends.megatron_utils.ddp_trace add-record-ids <trace dir> <run>/progress/data.jsonl
appends {"step", "rank", "mb_record_ids": [record id per microbatch (rows of one microbatch joined by "+")]} per rank
and step: a step's sample indices are drawn in data.jsonl's record order (miles/rollout/data_source.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

ENV_DIR = "Q38_DDP_TRACE"
ENV_GRAD_STATS = "Q38_DDP_TRACE_GRAD_STATS"

_TRACE = None


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _communicator(group):
    """The process group the bucket group reduces on (Megatron's start_grad_sync choice)."""
    if group.ddp_config.use_distributed_optimizer:
        return (
            group.intra_distributed_optimizer_instance_group,
            group.intra_distributed_optimizer_instance_size,
            group.intra_distributed_optimizer_instance_rank,
        )
    pg = group.data_parallel_group
    return pg, None, None


def _group_id(pg) -> str:
    if pg is None or not (dist.is_available() and dist.is_initialized()):
        return "local"
    return ",".join(str(r) for r in dist.get_process_group_ranks(pg))


def _cp_size() -> int:
    from megatron.core import parallel_state

    if parallel_state.get_context_parallel_group(check_initialized=False) is None:
        return 1
    return parallel_state.get_context_parallel_world_size()


class _Trace:
    def __init__(self, directory: Path, grad_stats: bool):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rank-{_rank():05d}.jsonl"
        self.grad_stats = grad_stats
        self.groups = []  # (chunk, comm, group id, index, bucket group, sig, numel, names)
        self.issues = defaultdict(list)  # id(bucket group) -> [(seq, source)]
        self.seq = defaultdict(int)  # (comm, group id) -> next seq
        self.in_finish = set()
        self.microbatches = []
        self.names = {}  # parameter -> name in its chunk

    def write(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def wrap(self, chunk_index: int, chunk) -> None:
        names = {p: n for n, p in chunk.named_parameters()}
        self.names.update(names)
        dense = [("dense", g) for g in chunk.bucket_groups]
        expert = [("expert", g) for g in chunk.expert_parallel_bucket_groups]
        cp = _cp_size() > 1
        for index, (kind, group) in enumerate(dense + expert):
            pg, _, _ = _communicator(group)
            comm = "expert_dp" if kind == "expert" else ("dp_cp" if cp else "dp")
            params = [(b, p) for b in group.buckets for p in b.params_list]
            pnames = [names.get(p, "?") for _, p in params]
            sig = hashlib.sha256("\n".join(pnames).encode()).hexdigest()
            numel = sum(b.grad_data.numel() for b in group.buckets)
            self.groups.append((chunk_index, comm, _group_id(pg), index, group, sig, numel, pnames))
            self._wrap_group(group, comm, _group_id(pg))

    def _wrap_group(self, group, comm: str, gid: str) -> None:
        start, finish = group.start_grad_sync, group.finish_grad_sync

        def start_grad_sync(*args, **kwargs):
            # Megatron's start_grad_sync returns at once when the first step already issued this group
            if not (group.is_first_batch and group.grad_reduce_handle is not None):
                key = (comm, gid)
                self.issues[id(group)].append((self.seq[key], "finish" if id(group) in self.in_finish else "hook"))
                self.seq[key] += 1
            return start(*args, **kwargs)

        def finish_grad_sync(*args, **kwargs):
            self.in_finish.add(id(group))
            try:
                return finish(*args, **kwargs)
            finally:
                self.in_finish.discard(id(group))

        group.start_grad_sync = start_grad_sync
        group.finish_grad_sync = finish_grad_sync

    def begin_step(self) -> None:
        self.issues.clear()
        self.seq.clear()
        self.microbatches = []

    def end_step(self, step: int) -> None:
        rank = _rank()
        by_comm = defaultdict(list)
        for _, comm, gid, index, group, sig, numel, _ in self.groups:
            issued = self.issues.get(id(group), [])
            entry = {
                "seq": issued[0][0] if issued else -1,
                "sig": sig,
                "count": len(issued),
                "source": issued[0][1] if issued else None,
                "index": index,
                "numel": numel,
            }
            by_comm[(comm, gid)].append(entry)
        for (comm, gid), entries in sorted(by_comm.items()):
            entries.sort(key=lambda e: (e["seq"] < 0, e["seq"], e["index"]))
            self.write({"step": step, "rank": rank, "comm": comm, "group": gid, "buckets": entries})
        self.write({"step": step, "rank": rank, "mb_sample_indices": self.microbatches})
        if self.grad_stats:
            self.write({"step": step, "rank": rank, "grad_stats": self._grad_stats()})

    def _grad_stats(self) -> dict:
        """Per parameter: L2 norm and max |g| of the reduced gradient, from each rank's shard, over the communicator."""
        stats = {}
        for _, _, _, _, group, _, _, _ in self.groups:
            pg, size, shard_rank = _communicator(group)
            sharded = size is not None
            entries = []
            for bucket in group.buckets:
                n = bucket.grad_data.numel()
                lo, hi = (shard_rank * (n // size), (shard_rank + 1) * (n // size)) if sharded else (0, n)
                for p in bucket.params_list:
                    start, end = bucket.param_to_index[p]
                    a, b = max(start, lo), min(end, hi)
                    piece = bucket.grad_data[a:b].double() if a < b else bucket.grad_data.new_zeros(0).double()
                    entries.append((p, piece))
            if not entries:
                continue
            sq = torch.stack([x.square().sum() if x.numel() else x.new_zeros(()) for _, x in entries])
            mx = torch.stack([x.abs().max() if x.numel() else x.new_zeros(()) for _, x in entries])
            if sharded and pg is not None and dist.is_available() and dist.is_initialized():
                dist.all_reduce(sq, group=pg)
                dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=pg)
            for (p, _), s, m in zip(entries, sq.tolist(), mx.tolist(), strict=True):
                stats[self._name(p)] = {"l2": math.sqrt(s), "max_abs": m, "numel": p.numel()}
        return stats

    def _name(self, param) -> str:
        return self.names[param]


def install(model_chunks):
    """Wrap every DDP bucket group of the chunks when Q38_DDP_TRACE is set; call after the groups' final wrapping
    (vision_grad_sync). Returns the trace, or None when off."""
    global _TRACE
    directory = os.environ.get(ENV_DIR)
    if not directory:
        _TRACE = None
        return None
    trace = _Trace(Path(directory), os.environ.get(ENV_GRAD_STATS) == "1")
    for index, chunk in enumerate(model_chunks):
        if getattr(chunk, "ddp_config", None) is not None:  # config-access-exempt: only Megatron DDP chunks
            trace.wrap(index, chunk)
    _TRACE = trace
    return trace


def begin_step() -> None:
    if _TRACE is not None:
        _TRACE.begin_step()


def on_microbatch(batch) -> None:
    """Record a training microbatch's sample indices (forward-only passes are skipped)."""
    if _TRACE is None or not torch.is_grad_enabled():
        return
    indices = batch.get("sample_indices")
    _TRACE.microbatches.append([int(i) for i in indices] if indices is not None else None)


def end_step(rollout_id: int) -> None:
    if _TRACE is not None:
        _TRACE.end_step(rollout_id + 1)


def add_record_ids(trace_dir, data_jsonl) -> int:
    """Append each rank's per-microbatch record ids (from data.jsonl) to its trace file; returns lines written."""
    records = {}
    for line in Path(data_jsonl).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            records[row["step"]] = row["record_ids"]  # the last attempt's row wins
    written = 0
    for path in sorted(Path(trace_dir).glob("rank-*.jsonl")):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        done = {r["step"] for r in rows if "mb_record_ids" in r}
        steps = {}
        for r in rows:
            if "mb_sample_indices" in r:
                steps[r["step"]] = r  # a re-run step: its last record
        out = []
        for step, r in sorted(steps.items()):
            if step in done or step not in records:
                continue
            out.append((step, r))
        if not out:
            continue
        with open(path, "a") as f:
            for step, r in out:
                index_to_record = _index_to_record(trace_dir, step, records[step])
                ids = [
                    None if mb is None else "+".join(str(index_to_record[i]) for i in mb)
                    for mb in r["mb_sample_indices"]
                ]
                f.write(json.dumps({"step": step, "rank": r["rank"], "mb_record_ids": ids}) + "\n")
                written += 1
    return written


def _index_to_record(trace_dir, step, record_ids) -> dict:
    """The step's sample indices over all ranks, ascending, are data.jsonl's record_ids in order."""
    indices = set()
    for path in Path(trace_dir).glob("rank-*.jsonl"):
        for x in path.read_text().splitlines():
            if x.strip() and '"mb_sample_indices"' in x:
                r = json.loads(x)
                if r["step"] == step:
                    indices.update(i for mb in r["mb_sample_indices"] if mb is not None for i in mb)
    ordered = sorted(indices)
    if len(ordered) != len(record_ids) or (ordered and ordered[-1] - ordered[0] + 1 != len(ordered)):
        raise ValueError(
            f"step {step}: {len(ordered)} traced sample indices, data.jsonl has {len(record_ids)} records"
        )
    return dict(zip(ordered, record_ids, strict=True))


def main(argv) -> int:
    if len(argv) == 3 and argv[0] == "add-record-ids":
        print(f"wrote {add_record_ids(argv[1], argv[2])} mb_record_ids lines")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
