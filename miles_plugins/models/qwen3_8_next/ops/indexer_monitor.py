"""Sampled recall of the frozen QSA indexer against dense attention, for training logs.

The indexer is frozen during SFT while the backbone moves, so its picks can drift away
from where the attention actually looks. Every ``MILES_QWEN38_INDEXER_MONITOR_EVERY``
calls of a QSA layer, a few query rows are scored densely over their whole document:

- ``recall``: attention mass (mean over the local heads) on the keys the sparse kernel
  attends, i.e. the indexer blocks plus the query's own partial block;
- ``overlap``: share of the indexer's complete blocks that are also among the dense
  teacher's top blocks, where a block's teacher score is the max over its tokens of
  the head-summed attention (the Qwen indexer-distillation target).

Rows are drawn per position bucket, only where the indexer has to choose (more complete
blocks than its budget), with a seed shared by every rank so TP ranks score the same
rows. Query heads are split over TP, so each row's per-key attention mass is summed over
the TP group before the teacher's top blocks are taken: a top-k of each rank's heads
averaged afterwards is not the teacher over all heads. Each rank appends JSON lines to
``$MILES_QWEN38_INDEXER_MONITOR_DIR/rank<global rank>.jsonl``; ``python -m
miles_plugins.models.qwen3_8_next.ops.indexer_monitor <dir>`` summarizes them. The
alarm threshold is a starting point to calibrate, not a proven trigger for training
the indexer.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

BUCKETS = ((2048, 8192, "2k-8k"), (8192, 32768, "8k-32k"), (32768, 131072, "32k-128k"), (131072, 1 << 62, "128k+"))


class IndexerRecallMonitor:
    def __init__(self, every: int, out_dir: str | None, rows: int = 64):
        self.every = every
        self.out_dir = out_dir
        self.rows = rows
        self._calls: dict[int, int] = defaultdict(int)

    @classmethod
    def from_env(cls) -> "IndexerRecallMonitor":
        return cls(
            every=int(os.environ.get("MILES_QWEN38_INDEXER_MONITOR_EVERY", "0")),
            out_dir=os.environ.get("MILES_QWEN38_INDEXER_MONITOR_DIR"),
            rows=int(os.environ.get("MILES_QWEN38_INDEXER_MONITOR_ROWS", "64")),
        )

    @property
    def enabled(self) -> bool:
        return self.every > 0 and bool(self.out_dir)

    def maybe_record(self, layer_number: int, query: Tensor, key: Tensor, block_form, scale: float, block_topk: int):
        """Count one call of ``layer_number``; on every ``every``-th call, measure and append a record."""
        call = self._calls[layer_number]
        self._calls[layer_number] = call + 1
        if call % self.every:
            return None
        sel_bitmap, lo, hi, blk_base, _, _, _, ratio = block_form
        rows = sample_rows(lo, hi, ratio, block_topk, self.rows, seed=layer_number * 1_000_003 + call)
        if rows.numel() == 0:
            return None
        stats = measure_rows(
            query, key, sel_bitmap, lo, hi, blk_base, ratio, scale, block_topk, rows, tp_group=_tp_group()
        )
        record = {"layer": layer_number, "call": call, **_ranks(), **summarize_rows(stats)}
        path = Path(self.out_dir) / f"rank{record['rank']}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record


def _tp_group():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_group() if get_tensor_model_parallel_world_size() > 1 else None


def _ranks() -> dict:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return {"rank": 0, "tp_rank": 0}
    from megatron.core.parallel_state import get_tensor_model_parallel_rank

    return {"rank": torch.distributed.get_rank(), "tp_rank": get_tensor_model_parallel_rank()}


def sample_rows(lo: Tensor, hi: Tensor, ratio: int, block_topk: int, n: int, seed: int) -> Tensor:
    """Up to ``n`` query rows, split evenly over the position buckets that have candidates."""
    positions = (hi - lo).long()
    eligible = (positions + 1) // ratio > block_topk
    gen = torch.Generator().manual_seed(seed)
    buckets = [
        torch.nonzero(eligible & (positions >= a) & (positions < b)).flatten().cpu() for a, b, _ in BUCKETS
    ]
    buckets = [b for b in buckets if b.numel()]
    if not buckets:
        return torch.zeros(0, dtype=torch.long)
    per = max(1, n // len(buckets))
    picked = [b[torch.randperm(b.numel(), generator=gen)[:per]] for b in buckets]
    return torch.cat(picked).sort().values.to(lo.device)


@torch.no_grad()
def measure_rows(
    query: Tensor,
    key: Tensor,
    sel_bitmap: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    ratio: int,
    scale: float,
    block_topk: int,
    rows: Tensor,
    tp_group=None,
) -> list[dict]:
    """Per row: position, recall and teacher overlap. ``query`` [T, Hq, D], ``key`` [T, Hkv, D].

    With ``tp_group``, ``query`` holds this rank's heads only; per-key mass is summed over the
    group (one all-reduce per row, same rows on every rank), so both metrics cover all heads.
    """
    hq, hkv = query.shape[1], key.shape[1]
    group = hq // hkv
    total_heads = hq * (torch.distributed.get_world_size(tp_group) if tp_group is not None else 1)
    out = []
    for row in rows.tolist():
        start, end = int(lo[row]), int(hi[row]) + 1
        n_keys = end - start
        q = query[row].float().view(hkv, group, -1)
        k = key[start:end].float()
        probs = torch.softmax(torch.einsum("hgd,lhd->hgl", q, k) * scale, dim=-1).reshape(hq, n_keys)
        key_block = int(blk_base[row]) + torch.arange(n_keys, device=query.device) // ratio
        chosen = sel_bitmap[row, key_block] != 0
        head_mass = probs.sum(dim=0)
        if tp_group is not None:
            torch.distributed.all_reduce(head_mass, group=tp_group)
        recall = (head_mass * chosen).sum() / total_heads

        complete = n_keys // ratio
        mass = head_mass[: complete * ratio].view(complete, ratio).amax(dim=-1)
        teacher = torch.topk(mass, min(block_topk, complete)).indices
        picked = chosen[: complete * ratio].view(complete, ratio)[:, 0]
        overlap = picked[teacher].float().sum() / max(1, int(picked.sum()))
        out.append({"pos": n_keys - 1, "recall": float(recall), "overlap": float(overlap)})
    return out


def summarize_rows(stats: list[dict]) -> dict:
    result = {"n": len(stats)}
    if stats:
        result["recall_mean"] = sum(s["recall"] for s in stats) / len(stats)
        result["recall_min"] = min(s["recall"] for s in stats)
        result["overlap_mean"] = sum(s["overlap"] for s in stats) / len(stats)
    buckets = {}
    for a, b, name in BUCKETS:
        sub = [s for s in stats if a <= s["pos"] < b]
        if sub:
            buckets[name] = {
                "n": len(sub),
                "recall_mean": sum(s["recall"] for s in sub) / len(sub),
                "overlap_mean": sum(s["overlap"] for s in sub) / len(sub),
            }
    result["buckets"] = buckets
    return result


def summarize_dir(out_dir: str, baseline_records: int = 3, alarm_drop: float = 0.01) -> list[str]:
    """Per layer: baseline (first records) vs latest recall; flag drops above ``alarm_drop``."""
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(Path(out_dir).glob("rank*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("n"):
                    by_layer[rec["layer"]].append(rec)
    lines = []
    for layer in sorted(by_layer):
        recs = sorted(by_layer[layer], key=lambda r: r["call"])
        calls = sorted({r["call"] for r in recs})
        mean_at = lambda c, k: sum(r[k] for r in recs if r["call"] == c) / sum(1 for r in recs if r["call"] == c)  # noqa: E731
        base_calls = calls[:baseline_records]
        base = sum(mean_at(c, "recall_mean") for c in base_calls) / len(base_calls)
        last = mean_at(calls[-1], "recall_mean")
        overlap = mean_at(calls[-1], "overlap_mean")
        flag = "  ALARM" if base - last > alarm_drop else ""
        lines.append(
            f"layer {layer}: recall base {base:.4f} -> last {last:.4f} (call {calls[-1]}), "
            f"overlap last {overlap:.4f}, records {len(recs)}{flag}"
        )
    return lines


_MONITOR: IndexerRecallMonitor | None = None


def get_monitor() -> IndexerRecallMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = IndexerRecallMonitor.from_env()
    return _MONITOR


if __name__ == "__main__":
    print("\n".join(summarize_dir(sys.argv[1])))
