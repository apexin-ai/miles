"""Cross-TP-rank consistency check for TP-replicated tensors.

Every rank of a TP group holds its own copy of each TP-replicated parameter (norms, routers, the
Qwen3.8 GDN, ...). A checkpoint stores the TP-rank-0 copy only, so a copy that drifted on another
rank changes silently on reload. This check compares a bitwise digest of every replicated
parameter, and of the fp32 master shard and optimizer moments the optimizer keeps for it, across
the TP group. It fails on any difference and names the tensors, their dtype and max |diff|.

``MILES_TP_REPLICA_CHECK=0`` turns the check off. ``MILES_TENSOR_DIGEST_DIR=<dir>`` also writes the
per-rank digests of all local tensors (sharded ones included) at every check, so a save and the
reload that follows can be compared tensor by tensor:
``python -m miles.backends.megatron_utils.tp_replica_check <dir> <tag before save> <tag after load>``.
"""

import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_CHUNK = 1 << 24
_WEIGHT_MOD = 1_000_003
_WORD_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_MAX_REPORTED = 8


def tensor_digest(tensor: torch.Tensor) -> torch.Tensor:
    """[sum, position-weighted sum, numel] of the raw bits as int64.

    Integer sums wrap the same way in any reduction order, so the digest is deterministic on GPU,
    and any single bit flip changes the first sum.
    """
    flat = tensor.detach().contiguous().view(-1)
    words = flat.view(_WORD_DTYPES[flat.element_size()])
    total = torch.zeros((), dtype=torch.int64, device=flat.device)
    weighted = torch.zeros((), dtype=torch.int64, device=flat.device)
    for start in range(0, words.numel(), _CHUNK):
        chunk = words[start : start + _CHUNK].to(torch.int64)
        position = torch.arange(start, start + chunk.numel(), dtype=torch.int64, device=flat.device)
        total += chunk.sum()
        weighted += (chunk * (position % _WEIGHT_MOD + 1)).sum()
    return torch.stack([total, weighted, torch.tensor(words.numel(), dtype=torch.int64, device=flat.device)])


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


def _optimizer_states(optimizer) -> list[dict]:
    states = []
    for sub in _sub_optimizers(optimizer):
        inner = (
            sub if isinstance(sub, torch.optim.Optimizer) else getattr(sub, "optimizer", None)
        )  # config-access-exempt: Megatron optimizers wrap a torch optimizer, stub optimizers wrap none
        if isinstance(inner, torch.optim.Optimizer):
            states.append(inner.state)
    return states


def named_tensors(model: Sequence[torch.nn.Module], optimizer=None, *, replicated_only: bool) -> Iterator[tuple[str, torch.Tensor]]:
    """Parameters, plus the fp32 master shard and optimizer moments the optimizer holds for them."""
    states = _optimizer_states(optimizer) if optimizer is not None else []
    for chunk_idx, chunk in enumerate(model):
        for name, param in chunk.named_parameters():
            if replicated_only and not is_tp_replicated(param):
                continue
            key = f"pp{chunk_idx}.{name}"
            yield key, param
            main = getattr(
                param, "main_param", None
            )  # config-access-exempt: the distributed optimizer attaches main_param only to params whose shard this DP rank owns
            if main is None:
                continue
            yield f"{key}#main", main
            for state in states:
                for state_name, value in sorted(state.get(main, {}).items()):
                    if isinstance(value, torch.Tensor) and value.numel() > 1:
                        yield f"{key}#{state_name}", value


def _collective_device(group) -> torch.device:
    if dist.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _fingerprint(entries: list[tuple[str, torch.Tensor]]) -> int:
    text = "\n".join(f"{name}|{t.dtype}|{tuple(t.shape)}" for name, t in entries)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little", signed=True)


def check_tp_replicas(model: Sequence[torch.nn.Module], optimizer=None, *, where: str, group=None) -> int:
    """Raise if any TP-replicated tensor differs across the TP group; return how many tensors were compared."""
    _maybe_dump_digests(model, optimizer, where=where)
    if os.environ.get("MILES_TP_REPLICA_CHECK", "1") == "0" or not dist.is_initialized():
        return 0
    if group is None:
        from megatron.core import mpu

        group = mpu.get_tensor_model_parallel_group()
    if dist.get_world_size(group) == 1:
        return 0
    entries = list(named_tensors(model, optimizer, replicated_only=True))
    device = _collective_device(group)
    world = dist.get_world_size(group)

    header = torch.tensor([len(entries), _fingerprint(entries)], dtype=torch.int64, device=device)
    headers = [torch.empty_like(header) for _ in range(world)]
    dist.all_gather(headers, header, group=group)
    if any(not torch.equal(h, headers[0]) for h in headers):
        counts = [int(h[0]) for h in headers]
        raise RuntimeError(f"[TP replica check @ {where}] TP ranks hold different replicated tensor lists: counts={counts}")
    if not entries:
        return 0

    digests = torch.stack([tensor_digest(t).to(device) for _, t in entries])
    gathered = [torch.empty_like(digests) for _ in range(world)]
    dist.all_gather(gathered, digests, group=group)
    differs = torch.zeros(len(entries), dtype=torch.bool, device=device)
    for other in gathered[1:]:
        differs |= (other != gathered[0]).any(dim=1)
    bad = differs.nonzero().flatten().tolist()
    if not bad:
        logger.info(f"[TP replica check @ {where}] {len(entries)} replicated tensors identical on {world} TP ranks")
        return len(entries)

    src = dist.get_global_rank(group, 0)
    lines = []
    for idx in bad[:_MAX_REPORTED]:
        name, tensor = entries[idx]
        local = tensor.detach().to(device)
        reference = local.clone()
        dist.broadcast(reference, src=src, group=group)
        max_abs = (local.float() - reference.float()).abs().max().reshape(1)
        max_abs_by_rank = [torch.empty_like(max_abs) for _ in range(world)]
        dist.all_gather(max_abs_by_rank, max_abs, group=group)
        ranks = [r for r in range(world) if not torch.equal(gathered[r][idx], gathered[0][idx])]
        lines.append(
            f"{name} dtype={tensor.dtype} shape={tuple(tensor.shape)} differs from TP rank 0 on TP ranks {ranks}; "
            f"max|diff| by TP rank={[float(m) for m in max_abs_by_rank]}"
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
    out = Path(root) / re.sub(r"[^A-Za-z0-9_.-]+", "_", where)
    out.mkdir(parents=True, exist_ok=True)
    digests = {name: tensor_digest(t).tolist() for name, t in named_tensors(model, optimizer, replicated_only=False)}
    (out / f"rank{rank:05d}.json").write_text(json.dumps(digests, sort_keys=True))


def compare_digest_dirs(before: Path, after: Path) -> list[str]:
    """Per-rank, per-tensor differences between two digest dumps; empty means identical."""
    problems = []
    ranks_before = sorted(p.name for p in before.glob("rank*.json"))
    ranks_after = sorted(p.name for p in after.glob("rank*.json"))
    if ranks_before != ranks_after:
        problems.append(f"rank files differ: {ranks_before} vs {ranks_after}")
    for rank_file in sorted(set(ranks_before) & set(ranks_after)):
        a = json.loads((before / rank_file).read_text())
        b = json.loads((after / rank_file).read_text())
        for name in sorted(a.keys() | b.keys()):
            if a.get(name) != b.get(name):
                problems.append(f"{rank_file}: {name}: {a.get(name)} -> {b.get(name)}")
    return problems


if __name__ == "__main__":
    root, tag_before, tag_after = sys.argv[1:4]
    found = compare_digest_dirs(Path(root) / tag_before, Path(root) / tag_after)
    print("\n".join(found) if found else "identical")
    sys.exit(1 if found else 0)
