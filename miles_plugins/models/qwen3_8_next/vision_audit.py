"""Opt-in visual and language gradient, weight-change and replica witnesses."""

import hashlib
import importlib.metadata
import inspect
import json
import sys
import time
import traceback
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
        if filename := getattr(
            module, "__file__", None
        ):  # config-access-exempt: namespace and builtin modules have no __file__
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
    if (
        children := getattr(optimizer, "chained_optimizers", None)
    ) is not None:  # config-access-exempt: only ChainedOptimizer has children
        for child in children:
            yield from _buffers(child)
    else:
        yield from getattr(
            optimizer, "buffers", ()
        )  # config-access-exempt: only distributed optimizers own gradient buffers


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


def _is_companion(name):
    from miles.backends.training_utils.model_companion import ModelCompanionInstallationUtils

    return ModelCompanionInstallationUtils.is_companion_parameter(name)


def _language_parameters(parameters):
    # The model companion (weight-version counter, sample-consumption rows) is bookkeeping, not a decoder weight:
    # it changes on every update, frozen or not (M-034).
    return [(name, p) for name, p in parameters if "vision_model." not in name and not _is_companion(name)]


def _frozen_visual(parameters):
    return [(name, p) for name, p in _visual_parameters(parameters) if not p.requires_grad]


def _count_images(module, positional, keywords):
    # The tower runs only on image microbatches; a text microbatch skips it (vision.py). One call per microbatch,
    # grid_thw has one row per image.
    grid = keywords.get("grid_thw")
    module._qwen38_audit_images += int(grid.shape[0]) if isinstance(grid, torch.Tensor) else 1


def _install_image_counters(model):
    towers = [
        module for chunk in model for name, module in chunk.named_modules() if name.split(".")[-1] == "vision_model"
    ]
    for tower in towers:
        tower._qwen38_audit_images = 0
        tower.register_forward_pre_hook(_count_images, with_kwargs=True)
    return towers


def _counts_images_here():
    # TP ranks run the same microbatches; count them once per data-parallel replica.
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        return True
    # Every CP rank runs the tower on all images of the microbatch (vision.py): count them on CP rank 0 only.
    cp_rank = parallel_state.get_context_parallel_rank() if parallel_state.get_context_parallel_world_size() > 1 else 0
    return parallel_state.get_tensor_model_parallel_rank() == 0 and cp_rank == 0


def _images_in_step(towers, device):
    """Images the vision tower ran on in this step, summed over every rank (each DP replica counted once)."""
    local = sum(tower._qwen38_audit_images for tower in towers) if _counts_images_here() else 0
    if dist.get_world_size() == 1:
        return local
    total = torch.tensor([local], dtype=torch.int64, device=device)
    dist.all_reduce(total)
    return int(total.item())


def _collective_device(parameters):
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


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
    language = _language_parameters(parameters)
    if args.qwen38_audit_interval < 1:
        raise ValueError("--qwen38-audit-interval must be at least 1")
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
        "num_updates": args.num_rollout
        * args.rollout_batch_size
        * args.n_samples_per_prompt
        // args.global_batch_size,
        "language_sha256": parameter_hash(language),
        "language_frozen": args.qwen38_freeze_language,
        "language_probe_names": language_probes,
        "visual_sha256": parameter_hash(visual) if visual else None,
        "frozen_visual_sha256": parameter_hash(_frozen_visual(parameters)) if _frozen_visual(parameters) else None,
        "audit_interval": args.qwen38_audit_interval,
        "steps_per_rollout": args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size,
        "source_identity": {
            name: hashlib.sha256((Path(args.hf_checkpoint) / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")
        },
    }
    inventory = [
        {
            "name": name,
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "trainable": p.requires_grad,
            "companion": _is_companion(name),
        }
        for name, p in parameters
    ]
    (root / f"rank-{dist.get_rank():03d}-initial.json").write_text(
        json.dumps({**state, "parameters": inventory}) + "\n"
    )
    if args.qwen38_visual_snapshot_interval:
        save_visual_snapshot(
            parameters,
            root / "snapshots/update-000000",
            update=0,
            frozen_language=args.qwen38_freeze_language,
            source_identity=state["source_identity"],
        )
    state["images_total"] = 0
    return state


def before_step(args, rollout_id, step_id, model, optimizer, scheduler):
    """Audit the update this train step is about to make; runs inside the wrapped ``optimizer.step``.

    Per audited update (every ``--qwen38-audit-interval`` updates and the final one):
    - visual gradients follow the images the step actually ran, over all ranks: a step with an image must give some
      trainable visual parameter a nonzero gradient, a step without one (a legal text-only step of a mixed run) must
      give them all exactly zero;
    - joint SFT: some language gradient probe is nonzero.
    At the final update:
    - decoder weights (the model companion excluded) are unchanged when language is frozen, changed when joint;
    - frozen visual parameters are unchanged; the trainable tower changed if any image ran during the run.
    """
    if not args.qwen38_audit_dir:
        raise ValueError("Qwen's audit hook requires --qwen38-audit-dir")
    parameters = [(name, p) for chunk in model for name, p in chunk.named_parameters()]
    state = getattr(
        model[0], "_qwen38_audit_state", None
    )  # config-access-exempt: audit state cached on the first model chunk
    if state is None:
        state = _initial_state(args, model, parameters)
        state["towers"] = _install_image_counters(model)
        model[0]._qwen38_audit_state = state
    for tower in state["towers"]:
        tower._qwen38_audit_images = 0
    selected = [(name, p) for name, p in parameters if "vision_model." in name and p.requires_grad]
    language_names = set(state["language_probe_names"])
    language_selected = [(name, p) for name, p in parameters if name in language_names]
    update = rollout_id * state["steps_per_rollout"] + step_id + 1
    is_final = rollout_id == args.num_rollout - 1 and step_id == state["steps_per_rollout"] - 1
    audited = is_final or update % state["audit_interval"] == 0
    original = getattr(
        optimizer.step, "_qwen38_original_step", optimizer.step
    )  # config-access-exempt: marker set on our own step wrapper

    def snapshot():
        interval = args.qwen38_visual_snapshot_interval
        if interval and (update % interval == 0 or is_final):
            save_visual_snapshot(
                parameters,
                Path(args.qwen38_audit_dir) / f"snapshots/update-{update:06d}",
                update=update,
                frozen_language=args.qwen38_freeze_language,
                source_identity=state["source_identity"],
            )

    def audited_step(*positional, **keywords):
        # Every rank reaches optimizer.step together (a skipped update skips it on every rank), so this sum is safe.
        images = _images_in_step(
            state["towers"], _collective_device(parameters) if dist.get_world_size() > 1 else None
        )
        probes = _capture_probes(optimizer, selected) if audited else []
        language_probes = _capture_probes(optimizer, language_selected) if audited else []
        lazy = getattr(optimizer, "_qwen38_lazy", None)  # config-access-exempt: vision_lazy's marker
        lazy_before = [(p, p.detach().clone()) for p in lazy.params] if lazy is not None and audited else []
        result = original(*positional, **keywords)
        torch.cuda.synchronize()
        optimizer.step = original
        state["images_total"] += images
        lazy_moved = None
        if lazy is not None:
            if lazy.last_skipped != (images == 0):
                raise ValueError(
                    f"The lazy text-step decision {lazy.last_skipped} disagrees with {images} audited images"
                )
            if lazy.last_skipped and lazy_before:
                lazy_moved = sum(not torch.equal(p, before) for p, before in lazy_before)
                if lazy_moved:
                    raise ValueError(f"A lazy text-only step moved {lazy_moved} lazy vision parameters")
        if not audited:
            snapshot()
            return result
        rows = _probe_results(probes)
        language_rows = _probe_results(language_probes)
        if rows and images and not any(row["gradient_nonzero"] for row in rows):
            raise ValueError(
                f"An image step gave no visual gradient: no visual gradient probe received a nonzero gradient "
                f"({images} images ran in update {update})"
            )
        if rows and not images and any(row["gradient_nonzero"] for row in rows):
            leaked = [row["name"] for row in rows if row["gradient_nonzero"]]
            raise ValueError(f"A visual gradient without an image in update {update}: {leaked[:8]}")
        if language_rows and not any(row["gradient_nonzero"] for row in language_rows):
            raise ValueError("No language gradient probe received a nonzero gradient")
        language_after = None
        frozen_visual_after = None
        if is_final:
            language_after = parameter_hash(_language_parameters(parameters))
            if args.qwen38_freeze_language and language_after != state["language_sha256"]:
                raise ValueError("Frozen decoder weights changed during visual training")
            if not args.qwen38_freeze_language and language_after == state["language_sha256"]:
                raise ValueError("The trainable decoder did not change during joint SFT")
            frozen_visual = _frozen_visual(parameters)
            frozen_visual_after = parameter_hash(frozen_visual) if frozen_visual else None
            if frozen_visual_after != state["frozen_visual_sha256"]:
                raise ValueError("Frozen visual weights changed during SFT")
        visual = _visual_parameters(parameters)
        visual_hash = parameter_hash(visual) if visual else None
        if is_final and selected and state["images_total"] and visual_hash == state["visual_sha256"]:
            raise ValueError("The trainable visual tower did not change during SFT")
        record = {
            "rank": dist.get_rank(),
            "update": update,
            "optimizer_result": str(result),
            "images_in_step": images,
            "images_total": state["images_total"],
            "lazy_skipped": None if lazy is None else lazy.last_skipped,
            "lazy_tensors_checked": len(lazy_before),
            "lazy_moved": lazy_moved,
            "parameters": rows,
            "language_parameters": language_rows,
            "visual_sha256": visual_hash,
            "initial_language_sha256": state["language_sha256"],
            "final_language_sha256": language_after,
            "initial_frozen_visual_sha256": state["frozen_visual_sha256"],
            "final_frozen_visual_sha256": frozen_visual_after,
        }
        root = Path(args.qwen38_audit_dir)
        (root / f"rank-{dist.get_rank():03d}-update-{update:06d}.json").write_text(json.dumps(record) + "\n")
        snapshot()
        print(
            json.dumps(
                {
                    "event": "qwen38-visual-update",
                    "rank": dist.get_rank(),
                    "update": update,
                    "visual_tensors": len(rows),
                    "images_in_step": images,
                }
            ),
            flush=True,
        )
        return result

    audited_step._qwen38_original_step = original
    optimizer.step = audited_step


_RELOAD_HEAD = 32
_MAX_REPORTED_MISMATCHES = 32


def _head(parameter):
    # view(-1) refuses a non-contiguous parameter instead of silently perturbing a copy
    return parameter.detach().view(-1)[: min(_RELOAD_HEAD, parameter.numel())]


def _perturb_head(parameter):
    head = _head(parameter)
    if head.is_floating_point() or head.is_complex():
        head.fill_(float("nan"))
    elif head.dtype == torch.bool:
        head.logical_not_()
    else:
        head.bitwise_not_()  # x -> -x - 1: every integer element changes, nothing overflows


def _tensor_digest(parameter):
    return parameter_hash([("", parameter)])


def _tp_rank():
    from megatron.core import parallel_state

    return parallel_state.get_tensor_model_parallel_rank() if parallel_state.model_parallel_is_initialized() else None


def _weight_version(companion):
    versions = [int(p.item()) for name, p in companion if name.split(".")[-1] == "weight_version"]
    return versions[0] if len(set(versions)) == 1 else versions or None


def _reload_receipt_dir(args, iteration, *, final):
    """``checkpoint-reload/`` for the final save (as before); ``checkpoint-reload/iter_NNNNNNN/`` for a listed one."""
    root = Path(args.qwen38_audit_dir) / "checkpoint-reload"
    return root if final else root / f"iter_{iteration:07d}"


def _verify_saved_model(args, iteration, model, *, receipt_dir=None):
    """Run the reload check on this rank, then agree on the outcome over all ranks so none waits on a failed one."""
    try:
        receipt = _reload_and_compare(args, iteration, model, receipt_dir=receipt_dir)
        local_error = None
    except Exception as error:
        receipt, local_error = None, error
    if dist.get_world_size() > 1:
        parameters = [(name, p) for chunk in model for name, p in chunk.named_parameters()]
        failed = torch.tensor([int(local_error is not None)], dtype=torch.int64, device=_collective_device(parameters))
        dist.all_reduce(failed)
        failed = int(failed.item())
    else:
        failed = int(local_error is not None)
    if local_error is not None:
        raise local_error
    if failed:
        raise ValueError(f"The native checkpoint reload check failed on {failed} other rank(s); this rank passed")
    print(json.dumps({"event": "qwen38-native-checkpoint-reload", **receipt}), flush=True)


def _reload_and_compare(args, iteration, model, *, receipt_dir=None):
    """Perturb the live model, reload the just-saved checkpoint and require every model tensor back bit for bit.

    ``args`` is the TrainerConfig that ``save`` received. Checkpoint load fields are
    runtime-only on ``args.backend``, so they are applied through MegatronCkptLoadConfig.

    Model tensors are every parameter except the model companion. Floating-point heads are set to NaN and integer
    heads bit-inverted, so an integer tensor the loader skips is caught too. The companion (weight-version counter,
    sample-consumption rows) is bookkeeping the finetune load may reset: it is not perturbed, its loaded weight
    version must equal the live one, and its live values are put back afterwards.

    On any failure every perturbed head is put back, ``checkpoint-reload/rank-NNN.failed.json`` records how far the
    check got (``loader_entered`` / ``loader_returned``), the exception and each differing tensor, and the error is
    re-raised. ``receipt_dir`` replaces ``checkpoint-reload/`` (a check before the final save writes its own folder).

    On success the live model is bit for bit what it was (every model tensor compared, the companion put back), and
    the finetune load skips the optimizer, the scheduler, RNG, rerun state and consumed samples (Megatron
    ``load_checkpoint``), so training can go on after a check before the final save.
    """
    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig
    from miles.backends.megatron_utils.tp_replica_check import is_tp_replicated

    parameters = [(name, parameter) for chunk in model for name, parameter in chunk.named_parameters()]
    weights = [(name, p) for name, p in parameters if not _is_companion(name)]
    companion = [(name, p) for name, p in parameters if _is_companion(name)]
    visual = _visual_parameters(weights)
    root = Path(receipt_dir) if receipt_dir is not None else Path(args.qwen38_audit_dir) / "checkpoint-reload"
    progress = {"stage": "prepare", "loader_entered": False, "loader_returned": False, "mismatches": []}
    started = time.monotonic()
    originals, companion_live = {}, []
    try:
        before = [_tensor_digest(p) for _, p in weights]
        # One pass over the model: the complete hash is taken over the per-tensor digests.
        complete_before = hashlib.sha256(
            "".join(f"{name}\0{digest}\n" for (name, _), digest in zip(weights, before, strict=True)).encode()
        ).hexdigest()
        request = MegatronCkptLoadConfig(
            load=args.backend.save,
            resume_from_ckpt=False,
            no_load_optim=True,
            no_load_rng=True,
            finetune=True,
            ckpt_step=iteration,
        )
        load_kwargs = dict(checkpointing_context=None, skip_load_to_model_and_opt=False, args=args)
        # Fail on a call-contract mismatch before the live model is touched.
        inspect.signature(load_checkpoint).bind(model, None, None, **load_kwargs)
        companion_live = [(p, p.detach().clone()) for _, p in companion]
        live_version = _weight_version(companion)
        progress["stage"] = "perturb"
        with torch.no_grad():
            for _, parameter in weights:
                originals[id(parameter)] = (parameter, _head(parameter).clone())
                _perturb_head(parameter)
        progress["stage"] = "load"
        progress["loader_entered"] = True
        with request.apply(args.backend):
            result = load_checkpoint(model, None, None, **load_kwargs)
        torch.cuda.synchronize()
        progress["loader_returned"] = True
        progress["stage"] = "compare"
        loaded_version = _weight_version(companion)
        for (name, parameter), digest in zip(weights, before, strict=True):
            if _tensor_digest(parameter) != digest:
                original = originals[id(parameter)][1]
                progress["mismatches"].append(
                    {
                        "name": name,
                        "dtype": str(parameter.dtype),
                        "shape": list(parameter.shape),
                        "tp_replicated": is_tp_replicated(parameter),
                        "head_restored": bool(torch.equal(_head(parameter), original)),
                    }
                )
        if progress["mismatches"]:
            names = [row["name"] for row in progress["mismatches"]]
            replicated = sum(row["tp_replicated"] for row in progress["mismatches"])
            raise ValueError(
                f"The native checkpoint did not restore every model parameter exactly: {len(names)} of "
                f"{len(weights)} model tensors differ after the reload: {', '.join(names[:8])}"
                f" ({replicated} TP-replicated; tp_rank={_tp_rank()})"
            )
        if result[0] != 0:
            raise ValueError(f"The model-only reload returned iteration {result[0]}, not 0")
        if loaded_version != live_version:
            raise ValueError(f"The checkpoint holds weight version {loaded_version}, the live model {live_version}")
    except BaseException as error:
        # Leave no perturbed heads behind; the caller still sees the original failure.
        with torch.no_grad():
            for parameter, original in originals.values():
                _head(parameter).copy_(original)
            for parameter, value in companion_live:
                parameter.data = value
        root.mkdir(parents=True, exist_ok=True)
        progress["mismatches"] = progress["mismatches"][:_MAX_REPORTED_MISMATCHES]
        failure = {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "tp_rank": _tp_rank(),
            "checkpoint_iteration": iteration,
            "passed": False,
            **progress,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "seconds": time.monotonic() - started,
        }
        (root / f"rank-{dist.get_rank():03d}.failed.json").write_text(json.dumps(failure) + "\n")
        raise
    with torch.no_grad():
        for parameter, value in companion_live:
            parameter.data = value
    root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "tp_rank": _tp_rank(),
        "checkpoint_iteration": iteration,
        "passed": True,
        "model_only": True,
        "optimizer_resume_qualified": False,
        "tensors_perturbed": len(weights),
        "integer_tensors_perturbed": sum(not p.is_floating_point() for _, p in weights),
        "companion_tensors_excluded": len(companion),
        "companion_weight_version": {"live": live_version, "checkpoint": loaded_version},
        "complete_parameter_sha256": complete_before,
        "complete_visual_sha256": parameter_hash(visual) if visual else None,
        "seconds": time.monotonic() - started,
    }
    (root / f"rank-{dist.get_rank():03d}.json").write_text(json.dumps(receipt) + "\n")
    return receipt


def _listed_iterations(args):
    listed = args.qwen38_reload_check_iterations
    if listed is None:
        return frozenset()
    if (
        not listed
        or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in listed)
        or len(set(listed)) != len(listed)
    ):
        raise ValueError(f"--qwen38-reload-check-iterations must list distinct iterations >= 0, got {listed!r}")
    return frozenset(listed)


def install_final_save_check(args):
    """Use the native save lifecycle for a synchronous model-only roundtrip.

    The final save (``iteration == num_rollout - 1``) is always checked. ``--qwen38-reload-check-iterations`` adds
    saves before it (e.g. an early save made through ``--save-trigger-sentinel``); each of those writes its receipts
    to ``checkpoint-reload/iter_NNNNNNN/``, and a failure stops the run on every rank like a final-save failure. A
    listed iteration past the final one is refused at the first save; a listed iteration that is never saved is
    never checked.

    ``args`` is the frozen flat view passed to ``--custom-megatron-init-path`` hooks.
    The wrapped ``save`` receives the live TrainerConfig as its first argument.
    """
    from miles.backends.megatron_utils import actor
    from miles.backends.megatron_utils import model as backend

    if args.async_save or not args.qwen38_audit_dir or not args.save:
        raise ValueError("Qwen checkpoint qualification requires synchronous saving and an audit directory")
    if getattr(
        backend.save, "_qwen38_final_save_check", False
    ):  # config-access-exempt: marker attribute on a function, not a config field
        return
    listed = _listed_iterations(args)
    original = backend.save
    if next(iter(inspect.signature(original).parameters)) != "args":
        raise TypeError("Qwen checkpoint qualification expects save(args, iteration, model, ...)")

    @wraps(original)
    def save(trainer_args, iteration, model, optimizer, scheduler, *positional, **keywords):
        final = iteration == trainer_args.num_rollout - 1
        if listed and max(listed) > trainer_args.num_rollout - 1:
            raise ValueError(
                f"--qwen38-reload-check-iterations {sorted(listed)}: the final save is iteration "
                f"{trainer_args.num_rollout - 1}"
            )
        result = original(trainer_args, iteration, model, optimizer, scheduler, *positional, **keywords)
        if final or iteration in listed:
            _verify_saved_model(
                trainer_args, iteration, model, receipt_dir=_reload_receipt_dir(trainer_args, iteration, final=final)
            )
        return result

    save._qwen38_final_save_check = True
    backend.save = save
    actor.save = save
