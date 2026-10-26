"""Reduced-gradient and real-weight witnesses for native image SFT qualification."""

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def record_inventory(model):
    directory = os.environ.get("DSV41_AUDIT_DIR")
    if directory is None:
        return
    root = Path(directory) / "parameters"
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(
            name=name,
            shape=list(p.shape),
            dtype=str(p.dtype),
            trainable=p.requires_grad,
            numel=p.numel(),
            sequence_parallel=getattr(p, "sequence_parallel", False),
        )
        for name, p in model.named_parameters()
    ]
    (root / f"rank-{dist.get_rank():03d}.json").write_text(json.dumps(rows) + "\n")


def _gradient_buffers(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        for child in children:
            yield from _gradient_buffers(child)
    else:
        yield from getattr(optimizer, "buffers", ())


def _reduced_gradient(optimizer, parameter, indices):
    buffers = [buffer for buffer in _gradient_buffers(optimizer) if parameter in buffer.param_index_map]
    if len(buffers) != 1:
        raise ValueError("Image SFT witness needs one distributed gradient buffer per parameter")
    buffer = buffers[0]
    start, end, bucket_index = buffer.param_index_map[parameter]
    bucket = buffer.buckets[bucket_index]
    group = buffer.data_parallel_group
    if end - start != parameter.numel() or bucket.grad_data.numel() % group.size():
        raise ValueError("Unsupported image SFT gradient bucket layout")
    shard_size = bucket.grad_data.numel() // group.size()
    own_start = bucket.offset + group.rank() * shard_size
    absolute = indices + start
    owned = (absolute >= own_start) & (absolute < own_start + shard_size)
    values = torch.zeros(len(indices), dtype=torch.float32, device=parameter.device)
    counts = owned.to(torch.int32)
    values[owned] = bucket.grad_data.flatten()[absolute[owned] - bucket.offset].float()
    dist.all_reduce(values, group=group)
    dist.all_reduce(counts, group=group)
    if not torch.all(counts == 1):
        raise ValueError("Reduced image SFT gradient samples do not have exactly one owner")
    return values


def _parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        flat = parameter.detach().flatten()
        for start in range(0, len(flat), 1048576):
            digest.update(flat[start : start + 1048576].cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def before_step(args, rollout_id, step_id, model, optimizer, scheduler):
    root = Path(os.environ["DSV41_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    parameters = [(name, parameter) for chunk in model for name, parameter in chunk.named_parameters()]
    frozen_language = [(name, p) for name, p in parameters if ".visual." not in name]
    frozen_hash = _parameter_hash(frozen_language) if args.dsv41_freeze_language else None
    selected = [(name, p) for name, p in parameters if p.requires_grad and ".visual." in name]
    for selector in ("mlp.router.weight", "mlp.experts.linear_fc1", "self_attention.linear_q_up_proj"):
        match = next(((name, p) for name, p in parameters if selector in name and p.requires_grad), None)
        if match is not None:
            selected.append(match)
    original = optimizer.step
    # A discarded/nonfinite step may never call the previous witness wrapper.
    # Do not nest stale wrappers (and stale before-update hashes) on retry.
    original = getattr(original, "_dsv41_original_step", original)

    def step(*positional, **keywords):
        probes = []
        for name, parameter in selected:
            count = min(parameter.numel(), 1024)
            indices = torch.arange(count, device=parameter.device, dtype=torch.long)
            indices = indices * (parameter.numel() - 1) // max(count - 1, 1)
            gradient = _reduced_gradient(optimizer, parameter, indices)
            probes.append((name, parameter, indices, parameter.detach().flatten()[indices].clone(), gradient))
        result = original(*positional, **keywords)
        torch.cuda.synchronize()
        rows = []
        for name, parameter, indices, initial, gradient in probes:
            final = parameter.detach().flatten()[indices]
            rows.append(
                dict(
                    name=name,
                    shape=list(parameter.shape),
                    gradient_finite=bool(torch.isfinite(gradient).all()),
                    gradient_nonzero=int(torch.count_nonzero(gradient)),
                    gradient_max=float(gradient.abs().max()),
                    delta_nonzero=int(torch.count_nonzero(final != initial)),
                    delta_max=float((final.float() - initial.float()).abs().max()),
                    weight_finite=bool(torch.isfinite(final).all()),
                )
            )
        language_after = _parameter_hash(frozen_language) if frozen_hash else None
        visual = [(name, p) for name, p in parameters if ".visual." in name]
        record = dict(
            rank=dist.get_rank(),
            rollout_id=rollout_id,
            step_id=step_id,
            optimizer_result=str(result),
            parameters=rows,
            frozen_language_before=frozen_hash,
            frozen_language_after=language_after,
            visual_sha256=_parameter_hash(visual) if visual else None,
        )
        (root / f"rank-{dist.get_rank():03d}-update-{rollout_id}-{step_id}.json").write_text(json.dumps(record) + "\n")
        optimizer.step = original
        if frozen_hash and language_after != frozen_hash:
            raise ValueError("Frozen language weights changed during the vision-only control")
        if any(not row["gradient_finite"] or not row["weight_finite"] for row in rows):
            raise ValueError("Nonfinite image SFT gradient or weight witness")
        snapshot = os.environ.get("DSV41_VISUAL_SNAPSHOT")
        if snapshot and rollout_id == args.num_rollout - 1 and step_id == args.num_steps_per_rollout - 1:
            from miles_plugins.models.deepseek_v4_1.checkpoint_audit import save_visual_snapshot

            save_visual_snapshot(model, snapshot)
        print(
            json.dumps(
                dict(
                    event="dsv41-image-update",
                    rank=dist.get_rank(),
                    rollout=rollout_id,
                    probes=len(rows),
                    changed=sum(row["delta_nonzero"] for row in rows),
                )
            ),
            flush=True,
        )
        return result

    step._dsv41_original_step = original
    optimizer.step = step
