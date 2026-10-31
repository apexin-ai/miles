"""The indexer recall monitor must agree with a per-head, per-token dense reference.

The monitor reads the packed kernel's input, per query its selected whole-block ids. The
reference here rebuilds the old one-byte-per-block bitmap (selected blocks plus the query's
own partial block, as the kernel attends) for the sampled rows only, test-side.
"""

import json
import math

import pytest

torch = pytest.importorskip("torch")

from miles_plugins.models.qwen3_8_next.ops import indexer_monitor  # noqa: E402
from miles_plugins.models.qwen3_8_next.ops.indexer_monitor import (  # noqa: E402
    IndexerRecallMonitor,
    measure_rows,
    row_geometry,
    sample_rows,
    summarize_dir,
)

RATIO = 4
TOPK = 3
HQ, HKV, D = 4, 2, 8


def _packed(lens, width=12, p=0.3):
    """lo / hi / blk_base per row, cu_seqlens, and per row a random set of its whole blocks.

    ``blocks`` is ``[T, width]`` int32 packed block ids in random order, ``-1`` pad, as
    ``qsa_selected_blocks`` publishes them (distinct, never the query's partial block).
    """
    lo, hi, blk_base, rows = [], [], [], []
    tok = blk = 0
    g = torch.Generator().manual_seed(0)
    for length in lens:
        for pos in range(length):
            lo.append(tok)
            hi.append(tok + pos)
            blk_base.append(blk)
            whole = (pos + 1) // RATIO
            keep = [b for b in torch.randperm(whole, generator=g).tolist() if torch.rand(1, generator=g) < p]
            ids = [blk + b for b in keep[:width]]
            rows.append(ids + [-1] * (width - len(ids)))
        tok += length
        blk += -(-length // RATIO)
    cu = torch.tensor([0, *lens]).cumsum(0).to(torch.int32)
    blocks = torch.tensor(rows, dtype=torch.int32).reshape(tok, width)
    return torch.tensor(lo), torch.tensor(hi), torch.tensor(blk_base), cu, blocks


def _bitmap(blocks, lo, hi, blk_base, n_blocks):
    """The kernel's attended blocks as ``[T, blocks]`` uint8: selected ids plus the own partial block."""
    bitmap = torch.zeros(blocks.shape[0], n_blocks, dtype=torch.uint8)
    for row in range(blocks.shape[0]):
        for b in blocks[row].tolist():
            if b >= 0:
                bitmap[row, b] = 1
        pos = int(hi[row] - lo[row])
        if (pos + 1) % RATIO:
            bitmap[row, int(blk_base[row]) + pos // RATIO] = 1
    return bitmap


def _reference(query, key, bitmap, lo, hi, blk_base, scale, row):
    start, end = int(lo[row]), int(hi[row])
    keys = list(range(start, end + 1))
    group = HQ // HKV
    probs = []
    for h in range(HQ):
        logits = [float(query[row, h].double() @ key[t, h // group].double()) * scale for t in keys]
        m = max(logits)
        e = [math.exp(x - m) for x in logits]
        probs.append([x / sum(e) for x in e])
    block_of = {t: int(blk_base[row]) + (t - start) // RATIO for t in keys}
    chosen = {t for t in keys if bitmap[row, block_of[t]]}
    recall = sum(sum(p[i] for i, t in enumerate(keys) if t in chosen) for p in probs) / HQ
    complete = len(keys) // RATIO
    head_sum = [sum(p[i] for p in probs) for i in range(len(keys))]
    teacher_score = [max(head_sum[b * RATIO : (b + 1) * RATIO]) for b in range(complete)]
    teacher = sorted(range(complete), key=lambda b: -teacher_score[b])[:TOPK]
    picked = [b for b in range(complete) if bitmap[row, int(blk_base[row]) + b]]
    overlap = len(set(teacher) & set(picked)) / max(1, len(picked))
    return recall, overlap


def test_row_geometry_matches_the_packed_grid():
    lens = [23, 9, 0, 40]  # an empty document repeats a boundary
    lo, hi, blk_base, cu, _ = _packed(lens)
    got = row_geometry(cu, sum(lens) + 5, RATIO)  # 5 rows past the last document
    n = sum(lens)
    for a, b in zip(got, (lo, hi, blk_base), strict=True):
        assert torch.equal(a[:n], b)
    pad_lo, pad_hi, _ = (t[n:] for t in got)
    assert torch.equal(pad_lo, pad_hi), "a row past the batch must have position 0"


def test_measure_matches_dense_reference_on_packed_rows():
    lens = [23, 9, 40]
    lo, hi, blk_base, cu, blocks = _packed(lens)
    bitmap = _bitmap(blocks, lo, hi, blk_base, int(-(-torch.tensor(lens) // RATIO).sum()))
    torch.manual_seed(1)
    query = torch.randn(sum(lens), HQ, D)
    key = torch.randn(sum(lens), HKV, D)
    scale = D**-0.5
    rows = torch.tensor([0, 5, 22, 23, 31, 32, 50, 71])
    stats = measure_rows(query, key, blocks, lo, hi, blk_base, RATIO, scale, TOPK, rows)
    for row, s in zip(rows.tolist(), stats, strict=True):
        recall, overlap = _reference(query, key, bitmap, lo, hi, blk_base, scale, row)
        assert s["pos"] == int(hi[row] - lo[row])
        assert s["recall"] == pytest.approx(recall, abs=1e-6)
        assert s["overlap"] == pytest.approx(overlap, abs=1e-6)


def test_full_selection_has_recall_one():
    lo, hi, blk_base, _, blocks = _packed([30], p=1.0)
    query, key = torch.randn(30, HQ, D), torch.randn(30, HKV, D)
    for row in (27, 29):  # a whole block ends at the row; a partial tail follows the last one
        stats = measure_rows(query, key, blocks, lo, hi, blk_base, RATIO, 1.0, TOPK, torch.tensor([row]))
        assert stats[0]["recall"] == pytest.approx(1.0, abs=1e-6)


def test_ids_the_kernel_ignores_do_not_count():
    """Another document's block, the query's own partial block and a later block are not attended."""
    lens = [20, 30]
    lo, hi, blk_base, cu, blocks = _packed(lens, width=4, p=0.5)
    torch.manual_seed(3)
    query, key = torch.randn(sum(lens), HQ, D), torch.randn(sum(lens), HKV, D)
    row = 20 + 13  # position 13 of the second document: 3 whole blocks, partial block 3
    rows = torch.tensor([row])
    clean = blocks.clone()
    clean[row] = torch.tensor([5, 7, -1, -1], dtype=torch.int32)
    dirty = clean.clone()
    dirty[row, 2:] = torch.tensor([0, 5 + 3], dtype=torch.int32)
    a = measure_rows(query, key, clean, lo, hi, blk_base, RATIO, D**-0.5, TOPK, rows)
    b = measure_rows(query, key, dirty, lo, hi, blk_base, RATIO, D**-0.5, TOPK, rows)
    assert a == b
    dirty[row, 3] = 5 + 4  # a block after the query
    assert measure_rows(query, key, dirty, lo, hi, blk_base, RATIO, D**-0.5, TOPK, rows) == a


def test_rows_are_drawn_per_bucket_only_where_the_indexer_must_choose():
    lo = torch.zeros(40000, dtype=torch.long)
    hi = torch.arange(40000)
    rows = sample_rows(lo, hi, RATIO, block_topk=512, n=12, seed=3)
    pos = (hi - lo)[rows]
    assert bool(((pos + 1) // RATIO > 512).all())
    assert [int(((pos >= a) & (pos < b)).sum()) for a, b, _ in indexer_monitor.BUCKETS[:3]] == [4, 4, 4]
    assert torch.equal(rows, sample_rows(lo, hi, RATIO, 512, 12, seed=3))
    assert sample_rows(lo[:2000], hi[:2000], RATIO, 512, 12, seed=3).numel() == 0


def test_records_every_nth_call_and_summary_flags_a_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(indexer_monitor, "BUCKETS", ((8, 1 << 62, "all"),))
    _, _, _, cu, blocks = _packed([64])
    torch.manual_seed(2)
    query, key = torch.randn(64, HQ, D), torch.randn(64, HKV, D)
    monitor = IndexerRecallMonitor(every=2, out_dir=str(tmp_path), rows=4)
    records = [monitor.maybe_record(5, query, key, blocks, cu, RATIO, D**-0.5, TOPK) for _ in range(6)]
    assert [r is not None for r in records] == [True, False, True, False, True, False]
    lines = (tmp_path / "rank0.jsonl").read_text().splitlines()
    assert [json.loads(x)["call"] for x in lines] == [0, 2, 4]

    good = summarize_dir(str(tmp_path), baseline_records=1)
    assert "ALARM" not in good[0]
    worse = dict(json.loads(lines[-1]), call=6, recall_mean=json.loads(lines[0])["recall_mean"] - 0.05)
    with (tmp_path / "rank0.jsonl").open("a") as f:
        f.write(json.dumps(worse) + "\n")
    assert "ALARM" in summarize_dir(str(tmp_path), baseline_records=1)[0]


def test_disabled_without_both_settings(monkeypatch):
    monkeypatch.delenv("MILES_QWEN38_INDEXER_MONITOR_DIR", raising=False)
    monkeypatch.setenv("MILES_QWEN38_INDEXER_MONITOR_EVERY", "10")
    assert not IndexerRecallMonitor.from_env().enabled


TP_HQ, TP_HKV, TP_D, TP_TOPK = 8, 2, 4, 3


def _tp_case():
    lens = [57, 70]
    lo, hi, blk_base, _, blocks = _packed(lens, width=17)
    g = torch.Generator().manual_seed(4)
    query = torch.randn(sum(lens), TP_HQ, TP_D, generator=g) * 3
    key = torch.randn(sum(lens), TP_HKV, TP_D, generator=g) * 3
    rows = torch.arange(20, sum(lens), 3)
    return query, key, blocks, lo, hi, blk_base, rows


def _tp_worker(rank, world, init_file, out_file):
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    query, key, blocks, lo, hi, blk_base, rows = _tp_case()
    # Megatron's split: each rank owns a contiguous slice of query heads and the kv heads they read
    hq, hkv = TP_HQ // world, TP_HKV // world
    q = query[:, rank * hq : (rank + 1) * hq]
    k = key[:, rank * hkv : (rank + 1) * hkv]
    stats = measure_rows(
        q, k, blocks, lo, hi, blk_base, RATIO, TP_D**-0.5, TP_TOPK, rows, tp_group=torch.distributed.group.WORLD
    )
    if rank == 0:
        torch.save(stats, out_file)
    torch.distributed.destroy_process_group()


def test_tp_split_heads_match_the_all_heads_teacher(tmp_path):
    """Per-rank heads summed over the TP group give the all-heads recall and teacher overlap."""
    query, key, blocks, lo, hi, blk_base, rows = _tp_case()
    want = measure_rows(query, key, blocks, lo, hi, blk_base, RATIO, TP_D**-0.5, TP_TOPK, rows)
    out = tmp_path / "stats.pt"
    torch.multiprocessing.spawn(_tp_worker, args=(2, str(tmp_path / "init"), str(out)), nprocs=2, join=True)
    got = torch.load(out)
    for g, w in zip(got, want, strict=True):
        assert g["pos"] == w["pos"]
        assert g["recall"] == pytest.approx(w["recall"], abs=1e-6)
        assert g["overlap"] == pytest.approx(w["overlap"], abs=1e-6)
    # the case is sensitive: one rank's heads alone pick a different teacher somewhere
    local = measure_rows(
        query[:, : TP_HQ // 2], key[:, :1], blocks, lo, hi, blk_base, RATIO, TP_D**-0.5, TP_TOPK, rows
    )
    assert any(abs(a["overlap"] - b["overlap"]) > 1e-6 for a, b in zip(local, want, strict=True))
