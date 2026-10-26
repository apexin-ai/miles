"""Opt-in destructive-in-memory, read-only-on-disk native checkpoint checks.

Install during the custom Megatron initialization hook, before training starts.
The final synchronous save is followed by corruption and model-only reload;
optimizer/data-cursor resume is a separate qualification, not implied here.
"""

import json
import os
import time
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist

from miles_plugins.models.deepseek_v4_1.audit import _parameter_hash


def verify_saved_model(args, iteration, model):
    from miles.backends.megatron_utils.checkpoint import load_checkpoint

    if args.async_save:
        raise ValueError("Image checkpoint qualification requires a completed synchronous save")
    visual = [(name, p) for chunk in model for name, p in chunk.named_parameters() if ".visual." in name]
    language = [(name, p) for chunk in model for name, p in chunk.named_parameters() if ".visual." not in name]
    selected = []
    for selector in ("mlp.router.weight", "mlp.experts.linear_fc1", "self_attention.linear_q_up_proj"):
        found = next(((name, p) for name, p in language if selector in name), None)
        if found is not None:
            selected.append(found)
    if len(selected) != 3 or (visual and len(visual) != 266):
        raise ValueError("Incomplete image checkpoint qualification inventory")
    samples = [(name, p, p.detach().flatten()[:65536].cpu().clone()) for name, p in visual + selected]
    visual_before = _parameter_hash(visual) if visual else None
    previous = {key: getattr(args, key) for key in ("load", "ckpt_step", "no_load_optim", "no_load_rng", "finetune")}
    started = time.monotonic()
    try:
        with torch.no_grad():
            for _, parameter, _ in samples:
                parameter.flatten()[:65536].fill_(float("nan"))
        args.load = args.save
        args.ckpt_step = iteration
        args.no_load_optim = True
        args.no_load_rng = True
        args.finetune = True
        result = load_checkpoint(model, None, None, checkpointing_context=None, skip_load_to_model_and_opt=False)
        if result[0] != 0:
            raise ValueError("Model-only checkpoint check unexpectedly restored training counters")
        torch.cuda.synchronize()
        for name, parameter, expected in samples:
            actual = parameter.detach().flatten()[:65536].cpu()
            if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                raise ValueError("Native checkpoint failed to restore " + name)
        visual_after = _parameter_hash(visual) if visual else None
        if visual_before != visual_after:
            raise ValueError("Native checkpoint did not restore the complete visual state")
    finally:
        for key, value in previous.items():
            setattr(args, key, value)
    root = Path(os.environ["DSV41_AUDIT_DIR"]) / "checkpoint-reload"
    root.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        rank=dist.get_rank(),
        iteration=iteration,
        passed=True,
        model_only=True,
        optimizer_resume_qualified=False,
        perturbed_then_restored=[name for name, _, _ in samples],
        complete_visual_sha256=visual_after,
        seconds=time.monotonic() - started,
    )
    (root / f"rank-{dist.get_rank():03d}.json").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(dict(event="dsv41-checkpoint-reload", **receipt)), flush=True)
    dist.barrier()


def install_final_save_check():
    """Attach once to the native final-save lifecycle, without changing core files."""
    from megatron.training import get_args

    from miles.backends.megatron_utils import actor
    from miles.backends.megatron_utils import model as backend

    if getattr(backend.save, "_dsv41_final_save_check", False):
        return
    original = backend.save

    @wraps(original)
    def save(iteration, model, optimizer, scheduler, *positional, **keywords):
        result = original(iteration, model, optimizer, scheduler, *positional, **keywords)
        args = get_args()
        if iteration == args.num_rollout - 1:
            verify_saved_model(args, iteration, model)
        return result

    save._dsv41_final_save_check = True
    backend.save = save
    actor.save = save


def save_visual_snapshot(model, root):
    """Export every trained visual tensor with a native-name, complete-state hash."""
    from megatron.core import parallel_state
    from safetensors.torch import load_file, save_file

    visual = [(name, p) for chunk in model for name, p in chunk.named_parameters() if ".visual." in name]
    if not visual:
        return
    if len(visual) != 266:
        raise ValueError("A visual snapshot must include all 266 native tensors")
    expected = _parameter_hash(visual)
    group = parallel_state.get_tensor_and_data_parallel_group()
    hashes = [None] * group.size()
    dist.all_gather_object(hashes, expected, group=group)
    if len(set(hashes)) != 1:
        raise ValueError("Visual replicas disagree before snapshot export")
    if dist.get_rank() == 0:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        state = {name.split(".visual.", 1)[1]: p.detach().cpu().contiguous() for name, p in visual}
        target = root / "visual.safetensors"
        save_file(state, str(target))
        read = load_file(str(target))
        if set(read) != set(state) or any(not torch.equal(read[name], p) for name, p in state.items()):
            raise ValueError("Native visual export did not round-trip bitwise")
        (root / "visual.json").write_text(
            json.dumps(dict(complete=True, resumable=False, tensors=266, parameter_sha256=expected)) + "\n"
        )
