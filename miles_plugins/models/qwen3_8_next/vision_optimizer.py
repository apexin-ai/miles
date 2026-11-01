"""Optimizer parameter groups for Qwen3.8's native vision tower.

--qwen38-vision-trainable picks what of the tower trains (set_visual_scope, before DDP and the optimizer are built):
  none       nothing: the tower is frozen, not in DDP's grad buffers and not in the optimizer;
  projector  the merger (merger.*) only; the ViT (patch_embed, pos_embed, blocks) is frozen;
  full       the whole tower.

--qwen38-vision-lr-mult / --qwen38-vision-wd-mult scale the language schedule for the trainable vision parameters:
their lr is the language lr times lr_mult at every step (max_lr and min_lr are both scaled; the warmup starts from the
shared --lr-warmup-init), and their weight decay is the language weight decay schedule times wd_mult. With both 1.0
(the default) the param groups are exactly Megatron's standard ones (get_standard_config_overrides).

The vision rules are merged into get_standard_config_overrides, not given instead of it: Megatron applies a non-None
config_overrides in place of its defaults, which would drop the no-decay rule for biases and 1-D parameters. Vision
biases and 1-D parameters stay undecayed whatever wd_mult is.

Every vision group with its own lr also carries lr_mult. Megatron matches checkpointed param groups to the running ones
by (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) only (DistributedOptimizer.load_state_dict,
MixedPrecisionOptimizer._filter_and_reorder_param_groups). A group that differs from the language group only in
max_lr / min_lr would share its key, and on a resume both would take one checkpointed group's max_lr: the vision lr
would silently become the language lr (or the reverse). lr_mult != 1.0 makes the key distinct.

Zero gradients still move parameters under AdamW. On a step whose global batch has no image, the trainable vision
parameters get exact zero gradients (vision._ZeroGradsFor); Adam then still applies its momentum from earlier image
steps, and decoupled weight decay shrinks every decayed vision parameter by lr * wd at every step, image or not. Only
none (not in the optimizer) leaves the tower bitwise unchanged.
"""

from __future__ import annotations

import logging
import math
from collections import Counter

import torch

logger = logging.getLogger(__name__)

VISION_MODULE = "vision_model"
PROJECTOR_PREFIX = "merger."
SCOPES = ("none", "projector", "full")


def is_vision_parameter_name(name: str) -> bool:
    """A parameter of the vision tower, under any wrapper prefix (module., module.module., ...)."""
    return VISION_MODULE in name.split(".")


def vision_relative_name(name: str) -> str:
    """The name inside the tower: module.module.vision_model.merger.norm.weight -> merger.norm.weight."""
    parts = name.split(".")
    return ".".join(parts[parts.index(VISION_MODULE) + 1 :])


def expected_trainable(scope: str, relative_name: str) -> bool:
    if scope not in SCOPES:
        raise ValueError(f"Unknown Qwen visual scope: {scope}")
    return scope == "full" or (scope == "projector" and relative_name.startswith(PROJECTOR_PREFIX))


def check_multipliers(lr_mult: float, wd_mult: float) -> None:
    if not (isinstance(lr_mult, (int, float)) and math.isfinite(lr_mult) and lr_mult > 0):
        raise ValueError(f"--qwen38-vision-lr-mult must be a finite number > 0, got {lr_mult!r}")
    if not (isinstance(wd_mult, (int, float)) and math.isfinite(wd_mult) and wd_mult >= 0):
        raise ValueError(f"--qwen38-vision-wd-mult must be a finite number >= 0, got {wd_mult!r}")


def is_default(lr_mult: float, wd_mult: float) -> bool:
    return float(lr_mult) == 1.0 and float(wd_mult) == 1.0


def _vision_predicate(_param, name):
    return is_vision_parameter_name(name)


def vision_config_overrides(config, *, lr_mult: float = 1.0, wd_mult: float = 1.0) -> dict:
    """Megatron's standard config overrides plus the vision lr / wd rules (for get_megatron_optimizer)."""
    from megatron.core.optimizer import ParamKey, ParamWithNamePredicate, get_standard_config_overrides

    check_multipliers(lr_mult, wd_mult)
    overrides = dict(get_standard_config_overrides(config))
    if is_default(lr_mult, wd_mult):
        return overrides
    # the standard keys that turn weight decay off (biases / 1-D parameters): vision wd_mult must not collide with them
    no_decay = [key for key, value in overrides.items() if value.get("wd_mult") == 0.0]
    if lr_mult != 1.0:
        if config.lr is None or config.min_lr is None:
            raise ValueError("--qwen38-vision-lr-mult needs --lr and --min-lr")
        key = ParamKey(with_name_predicate=ParamWithNamePredicate(name="qwen38_vision_lr", fn=_vision_predicate))
        # lr_mult is not read by the scheduler (it uses max_lr / min_lr); it keeps the group's checkpoint key distinct
        overrides[key] = {"max_lr": config.lr * lr_mult, "min_lr": config.min_lr * lr_mult, "lr_mult": float(lr_mult)}
    if wd_mult != 1.0:

        def decayed_vision(param, name):
            return is_vision_parameter_name(name) and not any(k.matches(param, name) for k in no_decay)

        key = ParamKey(with_name_predicate=ParamWithNamePredicate(name="qwen38_vision_wd", fn=decayed_vision))
        overrides[key] = {"wd_mult": float(wd_mult)}
    return overrides


def resolve_overrides(model_chunks, overrides) -> list[tuple[str, torch.nn.Parameter, dict]]:
    """Per trainable parameter, the merged override Megatron's _get_param_groups gives it (raises on conflicts)."""
    return [
        (name, param, resolve_overrides_for(overrides or {}, param, name))
        for chunk in model_chunks
        for name, param in chunk.named_parameters()
        if param.requires_grad
    ]


def resolve_overrides_for(overrides, param, name) -> dict:
    from megatron.core.optimizer_param_scheduler import combine_param_group_overrides

    matched = [value for key, value in overrides.items() if key.matches(param, name)]
    return dict(combine_param_group_overrides(matched)) if matched else {}


def check_vision_param_groups(model_chunks, config, overrides, *, scope, lr_mult=1.0, wd_mult=1.0) -> dict:
    """Checks the trainable vision set and each trainable parameter's lr / wd rule; returns a summary to log.

    Raises ValueError when a vision parameter's requires_grad does not follow the scope, or when any trainable
    parameter (vision or language) does not get the lr / wd the flags ask for.
    """
    from megatron.core.optimizer import get_standard_config_overrides

    check_multipliers(lr_mult, wd_mult)
    standard = get_standard_config_overrides(config)
    vision = {"tensors": 0, "trainable": 0, "frozen_names": [], "trainable_names": []}
    errors = []
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if not is_vision_parameter_name(name):
                continue
            vision["tensors"] += 1
            relative = vision_relative_name(name)
            want = expected_trainable(scope, relative)
            if param.requires_grad != want:
                errors.append(f"{name}: requires_grad {param.requires_grad}, scope {scope} wants {want}")
            if param.requires_grad:
                vision["trainable"] += 1
                vision["trainable_names"].append(relative)
            else:
                vision["frozen_names"].append(relative)

    groups = Counter()
    numel = Counter()
    for name, param, got in resolve_overrides(model_chunks, overrides):
        base = resolve_overrides_for(standard, param, name)
        is_vision = is_vision_parameter_name(name)
        want = dict(base)
        if is_vision and lr_mult != 1.0:
            want.update(max_lr=config.lr * lr_mult, min_lr=config.min_lr * lr_mult, lr_mult=float(lr_mult))
        if is_vision and wd_mult != 1.0 and base.get("wd_mult") != 0.0:
            want["wd_mult"] = float(wd_mult)
        if got != want:
            errors.append(f"{name}: override {got}, expected {want}")
        key = (
            "vision" if is_vision else "language",
            got.get("lr_mult", 1.0),
            got.get("wd_mult", 1.0),
            got.get("max_lr", config.lr),
            got.get("min_lr", config.min_lr),
        )
        groups[key] += 1
        numel[key] += param.numel()
    if errors:
        raise ValueError("Qwen vision param groups are wrong:\n  " + "\n  ".join(errors[:20]))
    return {
        "scope": scope,
        "lr_mult": float(lr_mult),
        "wd_mult": float(wd_mult),
        "vision_tensors": vision["tensors"],
        "vision_trainable": vision["trainable"],
        "vision_trainable_prefixes": sorted({n.split(".")[0] for n in vision["trainable_names"]}),
        "groups": [
            {
                "part": k[0],
                "lr_mult": k[1],
                "wd_mult": k[2],
                "max_lr": k[3],
                "min_lr": k[4],
                "tensors": groups[k],
                "numel": numel[k],
            }
            for k in sorted(groups, key=str)
        ],
    }


def optimizer_model_params(optimizer):
    """The model parameters this rank's optimizer updates (its DP shard), or None for an unknown optimizer type."""
    found = []
    for inner in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        index_map = getattr(inner, "model_param_group_index_map", None)
        if index_map is not None:  # DistributedOptimizer
            found.extend(index_map)
            continue
        fp16 = getattr(inner, "float16_groups", None)
        fp32 = getattr(inner, "fp32_from_fp32_groups", None)
        if fp16 is not None and fp32 is not None:  # Float16OptimizerWithFloat16Params: model params are float16_groups
            found.extend(p for group in fp16 + fp32 for p in group)
            continue
        groups = getattr(getattr(inner, "optimizer", None), "param_groups", None)
        if groups is None:
            return None
        found.extend(p for group in groups for p in group["params"])
    return found


GROUP_LOG_KEYS = ("lr", "weight_decay", "max_lr", "min_lr", "lr_mult", "wd_mult", "is_expert_parallel")


def describe_optimizer(model_chunks, optimizer) -> dict:
    """What the built optimizer holds: its param groups and, by name, which vision parameters it updates."""
    groups = []
    for inner in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        for group in getattr(inner, "param_groups", None) or []:
            groups.append({k: _plain(group.get(k)) for k in GROUP_LOG_KEYS} | {"params": len(group["params"])})
    params = optimizer_model_params(optimizer)
    result = {"param_groups": groups}
    if params is None:
        result["vision_in_optimizer"] = None
        return result
    ids = {id(p) for p in params}
    names = [
        vision_relative_name(n)
        for chunk in model_chunks
        for n, p in chunk.named_parameters()
        if is_vision_parameter_name(n) and id(p) in ids
    ]
    result["vision_in_optimizer"] = len(names)
    result["vision_in_optimizer_prefixes"] = sorted({n.split(".")[0] for n in names})
    return result


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    return value
