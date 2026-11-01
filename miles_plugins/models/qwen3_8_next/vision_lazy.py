"""Lazy Adam for trainable vision parameters on text-only steps (--qwen38-vision-lazy; user 2026-11-01 15:10 UTC).

On a step whose global batch has no image, the trainable vision parameters get exact zero gradients
(vision._ZeroGradsFor), yet AdamW would still move them: momentum from earlier image steps, and decoupled weight decay
lr * wd * p. With --qwen38-vision-lazy merger (or vision: the whole trainable tower) such a step leaves those
parameters, their fp32 master copies and their exp_avg / exp_avg_sq exactly as they were, with no weight decay, like
TensorFlow's LazyAdam. Their Adam step still advances with every other parameter's: Megatron checkpoints one step per
optimizer (DistributedOptimizer.state_dict asserts it for a HybridDeviceOptimizer and a resume writes it back to every
parameter), so a step that stayed behind would fail the save or be silently overwritten on resume. Bias correction
therefore uses the global step, and the decay these parameters get happens on image steps only.

"Has an image" is the number of images the vision tower ran on in this step's gradient-enabled forwards, counted on
TP rank 0 (TP ranks run the same microbatches) and summed over every rank with one all-reduce when optimizer.step
starts: every rank reaches optimizer.step together, and a skipped (unsupervised / non-finite) update skips it on every
rank. The count is reset by optimizer.zero_grad, which train_one_step calls at the start and end of every step.

Mechanism, for Megatron's DistributedOptimizer over a HybridDeviceOptimizer (HDO, --optimizer-cpu-offload; the
production path) and nothing else:
  - HDO steps a CPU fp32 copy of each parameter shard with torch AdamW, which skips every parameter whose .grad is None
    (no state change, no step count, no decay). After HDO copies the gradients to the CPU (_set_sub_optimizer_grads),
    a skipped step sets the lazy parameters' CPU .grad to None. HDO attaches that .grad only once, so a normal step
    attaches it again.
  - After the step, a skipped parameter's Adam step is set to the global one (its state is created with zero moments
    if it had none yet), and HDO's state view is synced again.
  - HDO copies every CPU parameter back to the GPU after the step: the lazy ones are unchanged, so are their bf16
    model copies.
HDO rebuilds its CPU copies and optimizers on load_state_dict, so the shard -> CPU copy mapping is looked up on every
step, never cached. Install after vision_scope_switch.install (it only changes how a checkpoint loads).
"""

from __future__ import annotations

import json
import logging

import torch
import torch.distributed as dist

from miles_plugins.models.qwen3_8_next.vision_optimizer import (
    PROJECTOR_PREFIX,
    VISION_MODULE,
    is_vision_parameter_name,
    vision_relative_name,
)

logger = logging.getLogger(__name__)

MODES = ("off", "merger", "vision")
_MARKER = "_qwen38_lazy"


def lazy_parameters(model_chunks, mode: str) -> dict:
    """{param: name}: the trainable vision parameters the mode makes lazy."""
    if mode not in MODES or mode == "off":
        raise ValueError(f"--qwen38-vision-lazy mode must be one of {MODES[1:]}, got {mode!r}")
    found = {}
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if not (param.requires_grad and is_vision_parameter_name(name)):
                continue
            if mode == "vision" or vision_relative_name(name).startswith(PROJECTOR_PREFIX):
                found[param] = name
    return found


def _counts_images_here() -> bool:
    from megatron.core import parallel_state

    return not parallel_state.model_parallel_is_initialized() or parallel_state.get_tensor_model_parallel_rank() == 0


def global_images(towers, device) -> int:
    """Images the towers ran on in this step (TP rank 0 only), summed over every rank."""
    local = sum(tower._qwen38_lazy_images for tower in towers) if _counts_images_here() else 0
    if dist.get_world_size() == 1:
        return int(local)
    total = torch.tensor([local], dtype=torch.int64, device=device)
    dist.all_reduce(total)
    return int(total.item())


def _count_images(module, positional, keywords):
    if torch.is_grad_enabled():  # training forwards; not log-prob or evaluation passes
        grid = keywords.get("grid_thw")
        module._qwen38_lazy_images += int(grid.shape[0]) if isinstance(grid, torch.Tensor) else 1


def _distributed_optimizers(optimizer):
    return list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


class LazyTextSteps:
    def __init__(self, model_chunks, optimizer, params: dict, mode: str):
        self.mode = mode
        self.params = params
        self.optimizer = optimizer
        self.skip = False
        self.last_skipped = False
        self.updates = 0
        self.skipped_updates = 0
        self.towers = [
            module
            for chunk in model_chunks
            for name, module in chunk.named_modules()
            if name.split(".")[-1] == VISION_MODULE
        ]
        # (HDO, shard) per lazy parameter this rank's optimizer shards hold
        self.shards = []
        self.hdos = []
        for inner in _distributed_optimizers(optimizer):
            index_map = getattr(inner, "model_param_group_index_map", None)
            if index_map is None:
                raise ValueError("--qwen38-vision-lazy needs Megatron's DistributedOptimizer")
            held = [p for p in params if p in index_map]
            if not held:
                continue
            hdo = inner.optimizer
            if not _is_hdo(hdo):
                raise ValueError(
                    "--qwen38-vision-lazy needs a HybridDeviceOptimizer (--optimizer-cpu-offload), got "
                    f"{type(hdo).__name__}"
                )
            for p in held:
                group, order = index_map[p]
                self.shards.append((hdo, hdo.param_groups[group]["params"][order]))
            if hdo not in self.hdos:
                self.hdos.append(hdo)

    def local_images(self) -> int:
        return sum(tower._qwen38_lazy_images for tower in self.towers)

    def _reset(self):
        for tower in self.towers:
            tower._qwen38_lazy_images = 0

    def _inner(self, hdo):
        """This HDO's CPU parameters (what AdamW steps) of the lazy shards; looked up now, HDO rebuilds them."""
        inners = []
        for owner, shard in self.shards:
            if owner is not hdo:
                continue
            inner = hdo.param_to_inner_param.get(shard)
            if inner is None or (hdo.gpu_optimizer is not None and _holds(hdo.gpu_optimizer, inner)):
                raise ValueError(
                    "--qwen38-vision-lazy needs every lazy parameter in HDO's CPU optimizer "
                    "(--optimizer-offload-fraction 1.0)"
                )
            inners.append(inner)
        return inners

    def install(self):
        for tower in self.towers:
            tower._qwen38_lazy_images = 0
            tower.register_forward_pre_hook(_count_images, with_kwargs=True)
        for hdo in self.hdos:
            self._wrap_hdo(hdo)
        self._wrap_optimizer()
        setattr(self.optimizer, _MARKER, self)
        return self

    def _wrap_optimizer(self):
        optimizer = self.optimizer
        original_zero_grad, original_step = optimizer.zero_grad, optimizer.step

        def zero_grad(*positional, **keywords):
            self._reset()
            return original_zero_grad(*positional, **keywords)

        def step(*positional, **keywords):
            images = global_images(self.towers, _collective_device())
            self.skip = images == 0
            self.updates += 1
            self.skipped_updates += self.skip
            if dist.get_rank() == 0:
                print(
                    json.dumps(
                        {"event": "qwen38-vision-lazy", "update": self.updates, "images": images, "skip": self.skip}
                    ),
                    flush=True,
                )
            try:
                return original_step(*positional, **keywords)
            finally:
                self.last_skipped, self.skip = self.skip, False

        optimizer.zero_grad, optimizer.step = zero_grad, step

    def _wrap_hdo(self, hdo):
        original_set_grads, original_step = hdo._set_sub_optimizer_grads, hdo.step

        def set_sub_optimizer_grads():
            original_set_grads()
            for inner in self._inner(hdo):
                if self.skip:
                    inner.grad = None  # torch AdamW skips it entirely
                elif inner in hdo.cpu_copy_map_grad:
                    inner.grad = hdo.cpu_copy_map_grad[inner]  # HDO attaches it once; put it back after a skip

        def step(closure=None):
            if not self.skip:
                return original_step(closure)
            inners = self._inner(hdo)
            owners = _owners(hdo)
            before = {inner: _step_value(owners[inner].state.get(inner)) for inner in inners}
            result = original_step(closure)
            lazy = set(inners)
            reference = next(
                (
                    state["step"]
                    for optimizer in hdo.cpu_optimizers
                    for param, state in optimizer.state.items()
                    if param not in lazy and "step" in state
                ),
                None,
            )
            if reference is None:
                raise ValueError("--qwen38-vision-lazy: no non-lazy parameter stepped to take the global step from")
            for inner in inners:
                state = owners[inner].state[inner]
                if before[inner] is None:
                    state["step"] = reference.detach().clone()
                    state["exp_avg"] = torch.zeros_like(inner, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(inner, memory_format=torch.preserve_format)
                else:
                    if before[inner] + 1 != reference.item() or _step_value(state) != before[inner]:
                        raise ValueError("--qwen38-vision-lazy: a lazy parameter's Adam step is out of line")
                    state["step"].copy_(reference)
            hdo._sync_sub_optimizers_state_to_hdo()
            return result

        hdo._set_sub_optimizer_grads, hdo.step = set_sub_optimizer_grads, step


def _is_hdo(optimizer) -> bool:
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    return isinstance(optimizer, HybridDeviceOptimizer)


def _holds(optimizer, param) -> bool:
    return any(p is param for group in optimizer.param_groups for p in group["params"])


def _owners(hdo) -> dict:
    return {p: optimizer for optimizer in hdo.cpu_optimizers for group in optimizer.param_groups for p in group["params"]}


def _step_value(state):
    if not state or "step" not in state:
        return None
    step = state["step"]
    return float(step.item() if isinstance(step, torch.Tensor) else step)


def _collective_device():
    if dist.get_world_size() > 1 and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def install(model_chunks, optimizer, *, mode: str):
    """Make the mode's trainable vision parameters lazy on text-only steps; returns the controller."""
    if getattr(optimizer, _MARKER, None) is not None:  # config-access-exempt: our own marker
        raise ValueError("--qwen38-vision-lazy is already installed on this optimizer")
    params = lazy_parameters(model_chunks, mode)
    has_tower = any(
        name.split(".")[-1] == VISION_MODULE for chunk in model_chunks for name, _ in chunk.named_modules()
    )
    if has_tower and not params:
        what = "merger" if mode == "merger" else "vision"
        raise ValueError(f"--qwen38-vision-lazy {mode}: the vision tower has no trainable {what} parameter")
    # a pipeline stage without the tower installs too: it joins every step's image all-reduce
    controller = LazyTextSteps(model_chunks, optimizer, params, mode).install()
    names = sorted(vision_relative_name(n) for n in params.values())
    logger.info("qwen38 vision lazy text steps: mode %s, %d parameters (%s ...)", mode, len(names), names[:4])
    return controller
