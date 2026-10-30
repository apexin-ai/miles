"""Cross-TP-rank consistency check for TP-replicated tensors.

Every rank of a TP group holds its own copy of each TP-replicated parameter (norms, routers, the
Qwen3.8 GDN, ...) and of the optimizer state for its shard of it. A checkpoint stores the TP-rank-0
copy only, so a copy that drifted on another rank changes silently on reload. This check compares,
bit for bit against TP rank 0, every replicated parameter and everything the optimizer keeps for it
(the fp32 master shard, Adam moments, the step count, an offloaded optimizer's CPU master). It fails
on any difference and names the tensors, their dtype and the max |diff| per TP rank.

``MILES_TP_REPLICA_CHECK=0`` turns the check off. ``MILES_TENSOR_DIGEST_DIR=<dir>`` also writes the
sha256 of every local tensor (sharded ones included) at every check, so a save and the reload that
follows can be compared tensor by tensor:
``python -m miles.backends.megatron_utils.tp_replica_check <dir> <tag before save> <tag after load>``.
"""

import json
import logging
import numbers
import os
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.weight_update.checksum_utils import hash_tensor_sha256

logger = logging.getLogger(__name__)

_CHUNK = 1 << 28  # bytes per broadcast
_MAX_REPORTED = 8


def is_tp_replicated(param: torch.Tensor) -> bool:
    # Megatron marks TP-sharded params with tensor_model_parallel and expert params with allreduce=False;
    # both legitimately differ across the ranks of a TP group.
    sharded = getattr(
        param, "tensor_model_parallel", False
    )  # config-access-exempt: Megatron sets this tensor attribute only on TP-sharded params
    expert = not getattr(
        param, "allreduce", True
    )  # config-access-exempt: Megatron sets allreduce=False only on expert-parallel params
    return not sharded and not expert


def _sub_optimizers(optimizer) -> Iterator:
    chained = getattr(
        optimizer, "chained_optimizers", None
    )  # config-access-exempt: only ChainedOptimizer wraps sub-optimizers
    if chained is None:
        yield optimizer
    else:
        for sub in chained:
            yield from _sub_optimizers(sub)


def _optimizer_param_owners(sub) -> dict[int, torch.Tensor]:
    """id(tensor the torch optimizer steps) -> model param, for the Megatron optimizer layouts."""
    owners = {}
    pairs = [
        # DistributedOptimizer: fp32 model params step through a view of their shard; bf16/fp16 ones through
        # an fp32 master shard, or through the bf16 shard itself with the precision-aware optimizer.
        ("model_fp32_groups", "shard_fp32_groups"),
        ("model_float16_groups", "shard_fp32_from_float16_groups"),
        ("model_float16_groups", "shard_float16_groups"),
        # Float16OptimizerWithFloat16Params
        ("float16_groups", "fp32_from_float16_groups"),
    ]
    for model_attr, shard_attr in pairs:
        model_groups = getattr(sub, model_attr, None)  # config-access-exempt: attribute set depends on optimizer class
        shard_groups = getattr(sub, shard_attr, None)  # config-access-exempt: attribute set depends on optimizer class
        if model_groups is None or shard_groups is None:
            continue
        for model_group, shard_group in zip(model_groups, shard_groups, strict=True):
            for model_param, shard in zip(model_group, shard_group, strict=True):
                if shard is not None:
                    owners.setdefault(id(shard), model_param)
    return owners


def _as_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, bool | numbers.Integral):
        return torch.tensor(int(value), dtype=torch.int64)
    if isinstance(value, numbers.Real):
        return torch.tensor(float(value), dtype=torch.float64)
    return None


def named_tensors(
    model: Sequence[torch.nn.Module], optimizer=None, *, replicated_only: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    """Parameters, then each optimizer-stepped tensor and every state entry (tensors and numbers) per param.

    Raises if the optimizer steps a tensor that cannot be traced back to a model param: a check that
    silently skipped optimizer state would report agreement it never verified.
    """
    names = {}
    for chunk_idx, chunk in enumerate(model):
        for name, param in chunk.named_parameters():
            names[id(param)] = (f"pp{chunk_idx}.{name}", param)
            if not replicated_only or is_tp_replicated(param):
                yield f"pp{chunk_idx}.{name}", param
    if optimizer is None:
        return
    for sub_idx, sub in enumerate(_sub_optimizers(optimizer)):
        inner = (
            sub if isinstance(sub, torch.optim.Optimizer) else getattr(sub, "optimizer", None)
        )  # config-access-exempt: Megatron optimizers wrap a torch optimizer, stub optimizers wrap none
        if not isinstance(inner, torch.optim.Optimizer):
            continue
        owners = _optimizer_param_owners(sub)
        for group in inner.param_groups:
            for stepped in group["params"]:
                owner = owners.get(id(stepped), stepped)
                if id(owner) not in names:
                    raise RuntimeError(
                        f"optimizer {sub_idx} ({type(inner).__name__}) steps a {tuple(stepped.shape)} tensor that "
                        "is not traceable to a model parameter; the TP replica check does not support this layout"
                    )
                name, param = names[id(owner)]
                if replicated_only and not is_tp_replicated(param):
                    continue
                if stepped is not param:
                    yield f"{name}#main", stepped
                for key, value in sorted(inner.state.get(stepped, {}).items()):
                    tensor = _as_tensor(value)
                    if tensor is None:
                        raise RuntimeError(f"{name}: optimizer state {key!r} has unsupported type {type(value).__name__}")
                    yield f"{name}#{key}", tensor


def _collective_device(group) -> torch.device:
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    # raw bytes: NaN payloads and -0.0 compare bitwise, and gloo and NCCL both broadcast uint8
    return tensor.detach().reshape(-1).contiguous().view(torch.uint8)


def _differs_from(tensor: torch.Tensor, *, src: int, group, device) -> bool:
    """Bitwise comparison with the copy on ``src``; every rank issues the same broadcasts."""
    raw = _bytes(tensor)
    differs = False
    for start in range(0, max(raw.numel(), 1), _CHUNK):
        local = raw[start : start + _CHUNK].to(device)
        reference = local.clone()
        dist.broadcast(reference, src=src, group=group)
        differs |= not torch.equal(local, reference)
    return differs


def _max_abs_diff(tensor: torch.Tensor, *, src: int, group, device) -> float:
    local = tensor.detach().reshape(-1).to(device=device, dtype=torch.float64)
    reference = local.clone()
    dist.broadcast(reference, src=src, group=group)
    return float((local - reference).abs().max()) if local.numel() else 0.0


def check_tp_replicas(model: Sequence[torch.nn.Module], optimizer=None, *, where: str, group=None) -> int:
    """Raise if any TP-replicated tensor differs across the TP group; return how many tensors were compared."""
    _maybe_dump_digests(model, optimizer, where=where)
    if os.environ.get("MILES_TP_REPLICA_CHECK", "1") == "0" or not dist.is_initialized():
        return 0
    if group is None:
        from megatron.core import mpu

        group = mpu.get_tensor_model_parallel_group()
    world = dist.get_world_size(group)
    if world == 1:
        return 0
    entries = list(named_tensors(model, optimizer, replicated_only=True))
    device = _collective_device(group)

    layout = [f"{name}|{t.dtype}|{tuple(t.shape)}" for name, t in entries]
    layouts = [None] * world
    dist.all_gather_object(layouts, layout, group=group)
    if any(other != layouts[0] for other in layouts):
        mismatch = [
            next((f"{a} vs {b}" for a, b in zip(layouts[0], other) if a != b), f"length {len(other)}")
            for other in layouts
        ]
        raise RuntimeError(
            f"[TP replica check @ {where}] TP ranks hold different replicated tensor lists "
            f"(counts {[len(x) for x in layouts]}); first difference from TP rank 0 by rank: {mismatch}"
        )

    src = dist.get_global_rank(group, 0)
    differs = torch.tensor(
        [_differs_from(t, src=src, group=group, device=device) for _, t in entries], dtype=torch.int32, device=device
    )
    dist.all_reduce(differs, op=dist.ReduceOp.MAX, group=group)
    bad = differs.nonzero().flatten().tolist()
    if not bad:
        logger.info(f"[TP replica check @ {where}] {len(entries)} replicated tensors bitwise identical on {world} TP ranks")
        return len(entries)

    lines = []
    for idx in bad[:_MAX_REPORTED]:
        name, tensor = entries[idx]
        max_abs = [None] * world
        dist.all_gather_object(max_abs, _max_abs_diff(tensor, src=src, group=group, device=device), group=group)
        lines.append(
            f"{name} dtype={tensor.dtype} shape={tuple(tensor.shape)} differs from TP rank 0; "
            f"max|diff| by TP rank={max_abs}"
        )
    raise RuntimeError(
        f"[TP replica check @ {where}] {len(bad)} of {len(entries)} TP-replicated tensors differ across TP ranks "
        f"(first {len(lines)} listed):\n  " + "\n  ".join(lines)
    )


def _maybe_dump_digests(model, optimizer, *, where: str) -> None:
    root = os.environ.get("MILES_TENSOR_DIGEST_DIR")
    if not root:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    out = Path(root) / re.sub(r"[^A-Za-z0-9_.-]+", "_", where)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {name: hash_tensor_sha256(t) for name, t in named_tensors(model, optimizer, replicated_only=False)}
    record = {"rank": rank, "world_size": world, "where": where, "tensors": tensors}
    tmp = out / f".rank{rank:05d}.json.tmp"
    tmp.write_text(json.dumps(record, sort_keys=True))
    tmp.replace(out / f"rank{rank:05d}.json")


def _load_dump(directory: Path) -> tuple[dict[int, dict], list[str]]:
    if not directory.is_dir():
        return {}, [f"{directory} does not exist"]
    records = {}
    for path in sorted(directory.glob("rank*.json")):
        record = json.loads(path.read_text())
        records[record["rank"]] = record
    if not records:
        return {}, [f"{directory} holds no rank dumps"]
    problems = []
    worlds = {r["world_size"] for r in records.values()}
    if len(worlds) != 1:
        problems.append(f"{directory}: rank dumps disagree on world size {sorted(worlds)}")
    elif set(records) != set(range(worlds.pop())):
        problems.append(f"{directory}: incomplete dump, ranks present {sorted(records)}")
    return records, problems


def compare_digest_dirs(before: Path, after: Path) -> list[str]:
    """Per-rank, per-tensor differences between two complete digest dumps; empty means identical."""
    records_before, problems = _load_dump(Path(before))
    records_after, problems_after = _load_dump(Path(after))
    problems += problems_after
    if problems:
        return problems
    if set(records_before) != set(records_after):
        return [f"rank sets differ: {sorted(records_before)} vs {sorted(records_after)}"]
    for rank in sorted(records_before):
        a, b = records_before[rank]["tensors"], records_after[rank]["tensors"]
        if not a:
            problems.append(f"rank {rank}: empty dump")
        for name in sorted(a.keys() | b.keys()):
            if a.get(name) != b.get(name):
                problems.append(f"rank {rank}: {name}: {a.get(name, 'missing')} -> {b.get(name, 'missing')}")
    return problems


if __name__ == "__main__":
    root, tag_before, tag_after = sys.argv[1:4]
    found = compare_digest_dirs(Path(root) / tag_before, Path(root) / tag_after)
    print("\n".join(found) if found else "identical")
    sys.exit(1 if found else 0)
