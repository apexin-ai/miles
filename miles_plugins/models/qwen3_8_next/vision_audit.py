"""Opt-in visual and language gradient, weight-change and replica witnesses."""

import hashlib
import importlib.metadata
import json
import sys
import time
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file


def _runtime_identity():
    selected = {
        "mbridge.core.bridge",
        "megatron.core.models.gpt.gpt_model",
        "megatron.core.optimizer.optimizer",
        "miles.backends.megatron_utils.model",
        "miles_plugins.mbridge.qwen3_8_next",
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
    }
    sources = {}
    for name, module in sorted(sys.modules.copy().items()):
        if name not in selected and not name.startswith("miles_plugins.models.qwen3_8_next."):
            continue
        if filename := getattr(module, "__file__", None):
            path = Path(filename).resolve()
            sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "python": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "ray")},
        "sources": sources,
    }


def parameter_hash(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
        flat = parameter.detach().flatten()
        for start in range(0, flat.numel(), 1048576):
            digest.update(flat[start : start + 1048576].cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _buffers(optimizer):
    if (children := getattr(optimizer, "chained_optimizers", None)) is not None:
        for child in children:
            yield from _buffers(child)
    else:
        yield from getattr(optimizer, "buffers", ())


def _gradient_samples(optimizer, parameter, indices):
    matches = [buffer for buffer in _buffers(optimizer) if parameter in buffer.param_index_map]
    if len(matches) != 1:
        raise ValueError("Qwen gradient audit requires one native distributed gradient buffer per parameter")
    buffer = matches[0]
    start, end, bucket_id = buffer.param_index_map[parameter]
    bucket, group = buffer.buckets[bucket_id], buffer.data_parallel_group
    if end - start != parameter.numel() or bucket.grad_data.numel() % group.size():
        raise ValueError("Unsupported distributed gradient layout")
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
        raise ValueError("Reduced gradient probes must have exactly one data-parallel owner")
    return values


def _visual_parameters(parameters):
    return [(name.split("vision_model.", 1)[1], p) for name, p in parameters if "vision_model." in name]


def _language_probe_names(parameters, *, limit=16):
    names = [name for name, p in parameters if "vision_model." not in name and p.requires_grad]
    count = min(limit, len(names))
    return [names[i * (len(names) - 1) // max(count - 1, 1)] for i in range(count)]


def _capture_probes(optimizer, parameters):
    probes = []
    for name, parameter in parameters:
        count = min(parameter.numel(), 1024)
        indices = torch.arange(count, device=parameter.device) * (parameter.numel() - 1) // max(count - 1, 1)
        gradient = _gradient_samples(optimizer, parameter, indices)
        probes.append((name, parameter, indices, parameter.detach().flatten()[indices].clone(), gradient))
    return probes


def _probe_results(probes):
    rows = []
    for name, parameter, indices, initial, gradient in probes:
        final = parameter.detach().flatten()[indices]
        rows.append(
            {
                "name": name,
                "gradient_finite": bool(torch.isfinite(gradient).all()),
                "gradient_nonzero": int(torch.count_nonzero(gradient)),
                "gradient_max": float(gradient.abs().max()),
                "delta_nonzero": int(torch.count_nonzero(final != initial)),
                "delta_max": float((final.float() - initial.float()).abs().max()),
                "weight_finite": bool(torch.isfinite(final).all()),
            }
        )
    if any(not row["gradient_finite"] or not row["weight_finite"] for row in rows):
        raise ValueError("Nonfinite audited gradient or parameter")
    return rows


def save_visual_snapshot(parameters, directory, *, update, frozen_language, source_identity):
    from megatron.core import parallel_state

    visual = _visual_parameters(parameters)
    if not visual:
        return
    digest = parameter_hash(visual)
    group = parallel_state.get_tensor_and_data_parallel_group()
    hashes = [None] * group.size()
    dist.all_gather_object(hashes, digest, group=group)
    if len(set(hashes)) != 1:
        raise ValueError("Qwen visual replicas disagree at snapshot time")
    if dist.get_rank() != 0:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    state = {"model.visual." + name: p.detach().cpu().contiguous() for name, p in visual}
    target = directory / "visual.safetensors"
    save_file(state, str(target))
    restored = load_file(str(target))
    if set(restored) != set(state) or any(not torch.equal(restored[name], value) for name, value in state.items()):
        raise ValueError("The visual snapshot did not round-trip exactly")
    receipt = {
        "complete": True,
        "resumable": False,
        "update": update,
        "frozen_language": frozen_language,
        "source_identity": source_identity,
        "tensors": len(state),
        "parameters": sum(p.numel() for _, p in visual),
        "replicas": len(hashes),
        "parameter_sha256": digest,
        "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    (directory / "visual.json").write_text(json.dumps(receipt, indent=2) + "\n")


def _initial_state(args, model, parameters):
    root = Path(args.qwen38_audit_dir)
    root.mkdir(parents=True, exist_ok=True)
    visual = _visual_parameters(parameters)
    language = [(name, p) for name, p in parameters if "vision_model." not in name]
    if args.qwen38_freeze_language and any(p.requires_grad for _, p in language):
        raise ValueError("Frozen-language SFT has a trainable decoder parameter")
    language_probes = [] if args.qwen38_freeze_language else _language_probe_names(parameters)
    if not args.qwen38_freeze_language and not language_probes:
        raise ValueError("Joint SFT has no trainable decoder parameters")
    index = json.loads((Path(args.hf_checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {name.removeprefix("model.visual.") for name in index if name.startswith("model.visual.")}
    if visual and {name for name, _ in visual} != expected:
        raise ValueError("The visual parameter inventory does not match the complete native checkpoint")
    state = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "runtime": _runtime_identity(),
        "num_updates": args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size,
        "language_sha256": parameter_hash(language),
        "language_frozen": args.qwen38_freeze_language,
        "language_probe_names": language_probes,
        "visual_sha256": parameter_hash(visual) if visual else None,
        "steps_per_rollout": args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size,
        "source_identity": {name: hashlib.sha256((Path(args.hf_checkpoint) / name).read_bytes()).hexdigest() for name in ("config.json", "model.safetensors.index.json")},
    }
    inventory = [{"name": name, "shape": list(p.shape), "dtype": str(p.dtype), "trainable": p.requires_grad} for name, p in parameters]
    (root / f"rank-{dist.get_rank():03d}-initial.json").write_text(json.dumps({**state, "parameters": inventory}) + "\n")
    if args.qwen38_visual_snapshot_interval:
        save_visual_snapshot(
            parameters,
            root / "snapshots/update-000000",
            update=0,
            frozen_language=args.qwen38_freeze_language,
            source_identity=state["source_identity"],
        )
    return state


def before_step(args, rollout_id, step_id, model, optimizer, scheduler):
    if not args.qwen38_audit_dir:
        raise ValueError("Qwen's audit hook requires --qwen38-audit-dir")
    parameters = [(name, p) for chunk in model for name, p in chunk.named_parameters()]
    state = getattr(model[0], "_qwen38_audit_state", None)
    if state is None:
        state = _initial_state(args, model, parameters)
        model[0]._qwen38_audit_state = state
    selected = [(name, p) for name, p in parameters if "vision_model." in name and p.requires_grad]
    language_names = set(state["language_probe_names"])
    language_selected = [(name, p) for name, p in parameters if name in language_names]
    original = getattr(optimizer.step, "_qwen38_original_step", optimizer.step)

    def audited_step(*positional, **keywords):
        probes = _capture_probes(optimizer, selected)
        language_probes = _capture_probes(optimizer, language_selected)
        result = original(*positional, **keywords)
        torch.cuda.synchronize()
        rows = _probe_results(probes)
        language_rows = _probe_results(language_probes)
        if rows and not any(row["gradient_nonzero"] for row in rows):
            raise ValueError("No visual gradient probe received a nonzero gradient")
        if language_rows and not any(row["gradient_nonzero"] for row in language_rows):
            raise ValueError("No language gradient probe received a nonzero gradient")
        update = rollout_id * state["steps_per_rollout"] + step_id + 1
        is_final = rollout_id == args.num_rollout - 1 and step_id == state["steps_per_rollout"] - 1
        language_after = None
        if is_final:
            language_after = parameter_hash([(name, p) for name, p in parameters if "vision_model." not in name])
            if args.qwen38_freeze_language and language_after != state["language_sha256"]:
                raise ValueError("Frozen decoder weights changed during visual training")
            if not args.qwen38_freeze_language and language_after == state["language_sha256"]:
                raise ValueError("The trainable decoder did not change during joint SFT")
        visual = _visual_parameters(parameters)
        visual_hash = parameter_hash(visual) if visual else None
        if is_final and selected and visual_hash == state["visual_sha256"]:
            raise ValueError("The trainable visual tower did not change during SFT")
        record = {
            "rank": dist.get_rank(),
            "update": update,
            "optimizer_result": str(result),
            "parameters": rows,
            "language_parameters": language_rows,
            "visual_sha256": visual_hash,
            "initial_language_sha256": state["language_sha256"],
            "final_language_sha256": language_after,
        }
        root = Path(args.qwen38_audit_dir)
        (root / f"rank-{dist.get_rank():03d}-update-{update:06d}.json").write_text(json.dumps(record) + "\n")
        interval = args.qwen38_visual_snapshot_interval
        if interval and (update % interval == 0 or is_final):
            save_visual_snapshot(
                parameters,
                root / f"snapshots/update-{update:06d}",
                update=update,
                frozen_language=args.qwen38_freeze_language,
                source_identity=state["source_identity"],
            )
        optimizer.step = original
        print(
            json.dumps(
                {
                    "event": "qwen38-visual-update",
                    "rank": dist.get_rank(),
                    "update": update,
                    "visual_tensors": len(rows),
                }
            ),
            flush=True,
        )
        return result

    audited_step._qwen38_original_step = original
    optimizer.step = audited_step


def _verify_saved_model(args, iteration, model):
    from miles.backends.megatron_utils.checkpoint import load_checkpoint

    parameters = [(name, parameter) for chunk in model for name, parameter in chunk.named_parameters()]
    visual = _visual_parameters(parameters)
    before = parameter_hash(parameters)
    previous = {key: getattr(args, key) for key in ("load", "ckpt_step", "no_load_optim", "no_load_rng", "finetune")}
    started = time.monotonic()
    try:
        # Only the live model is corrupted. The closed checkpoint and source HF files are read-only.
        with torch.no_grad():
            for _, parameter in parameters:
                parameter.flatten()[: min(32, parameter.numel())].fill_(float("nan"))
        args.load = args.save
        args.ckpt_step = iteration
        args.no_load_optim = True
        args.no_load_rng = True
        args.finetune = True
        result = load_checkpoint(model, None, None, checkpointing_context=None, skip_load_to_model_and_opt=False)
        torch.cuda.synchronize()
        if result[0] != 0 or parameter_hash(parameters) != before:
            raise ValueError("The native checkpoint did not restore every model parameter exactly")
    finally:
        for key, value in previous.items():
            setattr(args, key, value)
    root = Path(args.qwen38_audit_dir) / "checkpoint-reload"
    root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "checkpoint_iteration": iteration,
        "passed": True,
        "model_only": True,
        "optimizer_resume_qualified": False,
        "tensors_perturbed": len(parameters),
        "complete_parameter_sha256": before,
        "complete_visual_sha256": parameter_hash(visual) if visual else None,
        "seconds": time.monotonic() - started,
    }
    (root / f"rank-{dist.get_rank():03d}.json").write_text(json.dumps(receipt) + "\n")
    print(json.dumps({"event": "qwen38-native-checkpoint-reload", **receipt}), flush=True)
    dist.barrier()


def install_final_save_check(args):
    """Use the native final-save lifecycle for a synchronous model-only roundtrip."""
    from miles.backends.megatron_utils import actor
    from miles.backends.megatron_utils import model as backend

    if args.async_save or not args.qwen38_audit_dir or not args.save:
        raise ValueError("Qwen checkpoint qualification requires synchronous saving and an audit directory")
    if getattr(backend.save, "_qwen38_final_save_check", False):
        return
    original = backend.save

    @wraps(original)
    def save(iteration, model, optimizer, scheduler, *positional, **keywords):
        result = original(iteration, model, optimizer, scheduler, *positional, **keywords)
        if iteration == args.num_rollout - 1:
            _verify_saved_model(args, iteration, model)
        return result

    save._qwen38_final_save_check = True
    backend.save = save
    actor.save = save
