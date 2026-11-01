"""Resume a checkpoint saved with a smaller vision scope into a run that trains more of the tower.

    --qwen38-vision-trainable full --qwen38-vision-load-from-scope projector   (once, on the resume that switches)

The checkpoint was saved by a run with --qwen38-vision-trainable projector (or none): its ViT weights are in the model
state (frozen tensors are saved too), but the ViT is in neither the DDP grad buffers nor the optimizer. Megatron's
dp_reshardable optimizer format stores each DDP bucket as one flat tensor, laid out by the bucket's parameter list, and
its loader refuses a run whose buckets have a different number of elements (load_parameter_state_from_dp_reshardable).
The new parameters change that layout, so a plain resume fails loudly.

This module loads the optimizer state across the layout change. It does not loosen any checkpoint check:
  1. The old layout is computed with Megatron's own DistributedOptimizer._compute_per_buffer_param_layout on each
     buffer's parameter list without the new parameters (after checking that the same call reproduces the running
     layout exactly). The checkpoint's per_bucket_numel / per_bucket_numel_unpadded must equal it.
  2. For every parameter slice this rank owns in the new layout, the request reads the same elements from the old
     bucket (old bucket start + offset inside the parameter); old padding is requested on DP rank 0 so the old buckets
     are covered completely. The loaded slices are then handed, in the new layout's order, to Megatron's unchanged
     load_parameter_state_from_dp_reshardable, which handles the HDO / precision-aware storage as on any resume.
  3. The new parameters start fresh: fp32 main param = the loaded model weight (the bf16 tensor is exact in fp32),
     exp_avg = exp_avg_sq = 0. Their step is the global step every other parameter has (HybridDeviceOptimizer
     restores one step for all groups and asserts it is unique), so Adam's bias correction is the steady-state one:
     with zero moments the first updates are about sqrt(1 - beta2) / (1 - beta1) of a fresh Adam's (0.71 at
     beta1 0.9 / beta2 0.98) and reach full size within ~1 / (1 - beta1) steps.
  4. Param groups that exist only because of the new parameters (e.g. the ViT's own group with --qwen38-vit-lr-mult)
     are not in the checkpoint; they are added from the running config, with the checkpoint's step. Any other missing
     group is an error.
Everything else (language / merger parameters and Adam state, scheduler, RNG, data cursor) loads exactly as on a plain
resume. The next save is in the new layout; later resumes are plain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from miles_plugins.models.qwen3_8_next.vision_optimizer import (
    expected_trainable,
    is_vision_parameter_name,
    vision_relative_name,
)

logger = logging.getLogger(__name__)

SCOPE_ORDER = {"none": 0, "projector": 1, "full": 2}
SWITCH_MARKER = "qwen38_vision_scope_switch"


def check_switch(from_scope: str, to_scope: str) -> None:
    if from_scope not in SCOPE_ORDER or to_scope not in SCOPE_ORDER:
        raise ValueError(f"Unknown vision scopes {from_scope!r} -> {to_scope!r}")
    if SCOPE_ORDER[from_scope] >= SCOPE_ORDER[to_scope]:
        raise ValueError(
            f"--qwen38-vision-load-from-scope {from_scope} must train less than --qwen38-vision-trainable {to_scope}"
        )


def new_trainable_parameters(model_chunks, from_scope: str) -> dict:
    """{param: name}: trainable now, frozen (so not in the optimizer) in a run with from_scope."""
    found = {}
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if param.requires_grad and is_vision_parameter_name(name):
                if not expected_trainable(from_scope, vision_relative_name(name)):
                    found[param] = name
    return found


# --- layouts -----------------------------------------------------------------------------------------------------------


def compute_layout(params, bucket_size, data_parallel_world_size, ddp_config, param_indices=None):
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    return DistributedOptimizer._compute_per_buffer_param_layout(
        list(params), bucket_size, data_parallel_world_size, ddp_config, param_indices
    )


def same_layout(layout, param_index_map, per_bucket_numel_unpadded) -> bool:
    return dict(layout.param_index_map) == dict(param_index_map) and list(layout.per_bucket_numel_unpadded) == list(
        per_bucket_numel_unpadded
    )


def old_layout_for(params, new_params, *, bucket_sizes, data_parallel_world_size, ddp_config, running):
    """The layout of `params` without `new_params`, with the bucket size that reproduces the running layout.

    running = (param_index_map, per_bucket_numel_unpadded) of the live buffer.
    """
    for bucket_size in bucket_sizes:
        if same_layout(compute_layout(params, bucket_size, data_parallel_world_size, ddp_config), *running):
            old = [p for p in params if p not in new_params]
            return compute_layout(old, bucket_size, data_parallel_world_size, ddp_config)
    raise ValueError("Megatron's layout function does not reproduce the running DDP buffer layout; refusing to switch")


@dataclass
class Slice:
    """Elements [offset, offset + numel) of an old bucket's flat tensor (global shape (bucket_numel,))."""

    old_bucket: int
    offset: int
    numel: int
    bucket_numel: int


@dataclass
class BufferPlan:
    old_layout: object
    # per new bucket, per param in the new param_map order: a Slice, or None for a new parameter
    slices: list = field(default_factory=list)
    # per old bucket: [(start, end)] of elements no old parameter covers (requested on DP rank 0 only)
    gaps: dict = field(default_factory=dict)


def plan_buffer(old_layout, new_bucket_param_maps, new_params) -> BufferPlan:
    """Where each slice this rank owns in the new layout sits in the old buckets.

    new_bucket_param_maps: per new bucket, the DistributedOptimizer gbuf_range_map["param_map"] (param -> ranges,
    "param" = the owned slice inside the parameter).
    """
    plan = BufferPlan(old_layout=old_layout)
    for param_map in new_bucket_param_maps:
        row = []
        for param, ranges in param_map.items():
            if param in new_params:
                row.append(None)
                continue
            start, end, bucket = old_layout.param_index_map[param]
            bucket_start = old_layout.bucket_indices[bucket][0]
            inside = ranges["param"]
            if inside.end > end - start:
                raise ValueError("a parameter's owned slice is larger than the parameter in the old layout")
            row.append(
                Slice(
                    old_bucket=bucket,
                    offset=start - bucket_start + inside.start,
                    numel=inside.end - inside.start,
                    bucket_numel=old_layout.per_bucket_numel_unpadded[bucket],
                )
            )
        plan.slices.append(row)
    for bucket, (bucket_start, _) in enumerate(old_layout.bucket_indices):
        size = old_layout.per_bucket_numel_unpadded[bucket]
        covered = sorted(
            (s - bucket_start, e - bucket_start) for s, e, b in old_layout.param_index_map.values() if b == bucket
        )
        gaps, cursor = [], 0
        for s, e in covered:
            if s > cursor:
                gaps.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < size:
            gaps.append((cursor, size))
        plan.gaps[bucket] = gaps
    return plan


# --- param groups -----------------------------------------------------------------------------------------------------


def _group_key(group):
    from megatron.core.optimizer.optimizer import param_group_identifier_keys

    return tuple(group[k] if k in group else group[f"pre_{k}"] for k in param_group_identifier_keys)


def add_missing_param_groups(saved_groups, running_groups, allowed_indices):
    """Appends a saved group for each running group the checkpoint does not have; returns their indices.

    allowed_indices: running group indices that hold new parameters here; a missing group must be one of them and hold
    only new parameters, or be empty on this rank.
    The added group carries the running hyperparameters and the checkpoint's step.
    """
    saved_keys = {_group_key(g) for g in saved_groups}
    steps = {g["step"] for g in saved_groups if "step" in g}
    if len(steps) > 1:
        raise ValueError(f"checkpointed param groups have different steps {sorted(steps)}")
    added = []
    for index, group in enumerate(running_groups):
        if _group_key(group) in saved_keys:
            continue
        if index not in allowed_indices and group["params"]:
            raise ValueError(f"param group {_group_key(group)} is not in the checkpoint and holds old parameters")
        copy = {k: v for k, v in group.items() if k != "params"}
        copy["params"] = []
        if steps:
            copy["step"] = next(iter(steps))
        saved_groups.append(copy)
        added.append(index)
    return added


# --- the DistributedOptimizer hooks -------------------------------------------------------------------------------


def install(model_chunks, optimizer, *, from_scope: str, to_scope: str) -> dict:
    """Prepares every DistributedOptimizer in `optimizer` to load a `from_scope` checkpoint. Call before the load."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    check_switch(from_scope, to_scope)
    new_params = new_trainable_parameters(model_chunks, from_scope)
    summary = {"from": from_scope, "to": to_scope, "new_tensors": len(new_params), "optimizers": []}
    bucket_sizes = [None] + sorted({s for c in model_chunks if (s := getattr(c, "bucket_size", None)) is not None})
    for inner in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        if not isinstance(inner, DistributedOptimizer):
            raise ValueError(f"the vision scope switch needs Megatron's DistributedOptimizer, got {type(inner).__name__}")
        if getattr(inner, "_optimizer_state_offloader", None) is not None:
            raise ValueError("the vision scope switch does not support the optimizer state offloader")
        _Switch(inner, new_params, bucket_sizes).bind()
        summary["optimizers"].append(inner._qwen38_scope_switch.summary)
    logger.info("qwen38 vision scope switch armed: %s", summary)
    return summary


class _Switch:
    def __init__(self, opt, new_params, bucket_sizes):
        self.opt = opt
        self.new_params = {p: n for p, n in new_params.items() if p in opt.model_param_group_index_map}
        # Buffers without a new parameter (expert buffers, pipeline stages without the tower) keep their layout: their
        # optimizer state loads through Megatron's own path. Their param groups may still miss the ViT's (empty) group.
        self.translate = any(p in new_params for buffer in opt.buffers for p in buffer.params)
        self.plans = []
        for gbuf_idx, buffer in enumerate(opt.buffers if self.translate else []):
            running = (buffer.param_index_map, [b.numel_unpadded for b in buffer.buckets])
            old = old_layout_for(
                buffer.params,
                new_params,
                bucket_sizes=bucket_sizes,
                data_parallel_world_size=buffer.data_parallel_world_size,
                ddp_config=buffer.ddp_config,
                running=running,
            )
            (gbuf_range_map_for_all_buckets,) = opt.gbuf_ranges[gbuf_idx].values()
            self.plans.append(plan_buffer(old, [m["param_map"] for m in gbuf_range_map_for_all_buckets], new_params))
        self.old_per_bucket_numel_unpadded = [
            {dtype: list(plan.old_layout.per_bucket_numel_unpadded) for dtype in opt.gbuf_ranges[i]}
            for i, plan in enumerate(self.plans)
        ]
        self.allowed_groups = {opt.model_param_group_index_map[p][0] for p in self.new_params}
        self.summary = {
            "translate": self.translate,
            "new_tensors_here": len(self.new_params),
            "new_numel_here": sum(p.numel() for p in self.new_params),
            "buffers": [
                {"old": list(p.old_layout.per_bucket_numel_unpadded), "new": [b.numel_unpadded for b in buf.buckets]}
                for p, buf in zip(self.plans, opt.buffers)
            ],
        }
        self.loaded = False

    def bind(self):
        opt = self.opt
        self.original_sharded = opt.sharded_param_state_dp_reshardable
        self.original_load_param_state = opt.load_parameter_state_from_dp_reshardable
        self.original_load_state_dict = opt.load_state_dict
        opt.sharded_param_state_dp_reshardable = self.sharded_param_state
        opt.load_parameter_state_from_dp_reshardable = self.load_param_state
        opt.load_state_dict = self.load_state_dict
        opt._qwen38_scope_switch = self

    def _bucket_key(self, gbuf_idx, dtype, bucket):
        return (
            f"optimizer.distributed.dp_group_idx_{self.opt.data_parallel_group_idx}"
            f".gbuf_idx_{gbuf_idx}.dtype_{dtype}.bucket_idx_{bucket}"
        )

    def sharded_param_state(self, model_sharded_state_dict, is_loading=False, metadata=None):
        if not is_loading or self.loaded or not self.translate:
            return self.original_sharded(model_sharded_state_dict, is_loading, metadata)
        from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject, ShardedTensor

        opt = self.opt
        dp_rank = opt.data_parallel_group.rank()
        replica = (opt.distributed_optimizer_instance_id, 0, 0)
        state = {SWITCH_MARKER: LocalNonpersistentObject(True), "slices": {}, "gaps": {}}
        old_per_bucket_numel = [
            {dtype: [end - start for start, end in plan.old_layout.bucket_indices] for dtype in opt.gbuf_ranges[i]}
            for i, plan in enumerate(self.plans)
        ]
        for name, value in (
            ("per_bucket_numel", old_per_bucket_numel),
            ("per_bucket_numel_unpadded", self.old_per_bucket_numel_unpadded),
        ):
            key = f"optimizer.distributed.dp_group_idx_{opt.data_parallel_group_idx}.{name}"
            state[name] = ShardedObject(key, value, (1,), (0,), replica_id=(opt.distributed_optimizer_instance_id, 0, dp_rank))
        for gbuf_idx, plan in enumerate(self.plans):
            ((dtype, buckets),) = opt.gbuf_ranges[gbuf_idx].items()
            templates = {}
            for bucket_idx, (bucket_map, row) in enumerate(zip(buckets, plan.slices)):
                for i, (param, piece) in enumerate(zip(bucket_map["param_map"], row)):
                    tensors = opt._get_main_param_and_optimizer_states(param)
                    templates = templates or tensors
                    if piece is None:
                        continue
                    for k, v in tensors.items():
                        if not isinstance(v, torch.Tensor) or v.dim() == 0:
                            continue  # the scalar step is restored from the param groups
                        state["slices"][f"{gbuf_idx}|{bucket_idx}|{i}|{k}"] = ShardedTensor(
                            f"{self._bucket_key(gbuf_idx, dtype, piece.old_bucket)}.{k}",
                            torch.empty((piece.numel,), dtype=v.dtype, device=v.device),
                            v.dtype,
                            (piece.numel,),
                            (piece.bucket_numel,),
                            (piece.offset,),
                            axis_fragmentations=None,
                            flattened_range=None,
                            allow_shape_mismatch=False,
                            replica_id=replica,
                        )
            if dp_rank == 0 and templates:
                for old_bucket, gaps in plan.gaps.items():
                    for j, (start, end) in enumerate(gaps):
                        for k, v in templates.items():
                            if not isinstance(v, torch.Tensor) or v.dim() == 0:
                                continue
                            state["gaps"][f"{gbuf_idx}|{old_bucket}|{j}|{k}"] = ShardedTensor(
                                f"{self._bucket_key(gbuf_idx, dtype, old_bucket)}.{k}",
                                torch.empty((end - start,), dtype=v.dtype, device=v.device),
                                v.dtype,
                                (end - start,),
                                (plan.old_layout.per_bucket_numel_unpadded[old_bucket],),
                                (start,),
                                axis_fragmentations=None,
                                flattened_range=None,
                                allow_shape_mismatch=False,
                                replica_id=replica,
                            )
        return state

    def load_param_state(self, state_dict):
        if not isinstance(state_dict, dict) or SWITCH_MARKER not in state_dict:
            if not self.translate and "per_bucket_numel_unpadded" in (state_dict or {}):
                self.loaded = True
            return self.original_load_param_state(state_dict)
        opt = self.opt
        if state_dict["per_bucket_numel_unpadded"] != self.old_per_bucket_numel_unpadded:
            raise ValueError(
                "the checkpoint's buckets do not match the layout without the new vision parameters: "
                f"{state_dict['per_bucket_numel_unpadded']} vs {self.old_per_bucket_numel_unpadded}"
            )
        translated = {"per_bucket_numel_unpadded": opt.per_bucket_numel_unpadded}
        fresh = 0
        for gbuf_idx, plan in enumerate(self.plans):
            ((dtype, buckets),) = opt.gbuf_ranges[gbuf_idx].items()
            rows = []
            for bucket_idx, (bucket_map, row) in enumerate(zip(buckets, plan.slices)):
                out = []
                for i, (param, piece) in enumerate(zip(bucket_map["param_map"], row)):
                    tensors = opt._get_main_param_and_optimizer_states(param)
                    if piece is None:
                        inside = bucket_map["param_map"][param]["param"]
                        source = param.detach().view(-1)[inside.start : inside.end]
                        built = {}
                        for k, v in tensors.items():
                            if not isinstance(v, torch.Tensor) or v.dim() == 0:
                                built[k] = v
                            elif k == "param":
                                built[k] = source.to(dtype=v.dtype, device=v.device)
                            else:
                                built[k] = torch.zeros_like(v)
                        fresh += 1
                    else:
                        built = {
                            k: (state_dict["slices"][f"{gbuf_idx}|{bucket_idx}|{i}|{k}"] if isinstance(v, torch.Tensor) and v.dim() else v)
                            for k, v in tensors.items()
                        }
                    built["padding"] = False
                    out.append(built)
                rows.append(out)
            translated[gbuf_idx] = {dtype: rows}
        self.original_load_param_state(translated)
        self.loaded = True
        self.summary["fresh_slices_here"] = fresh
        logger.info("qwen38 vision scope switch loaded: %s", self.summary)

    def load_state_dict(self, state_dict):
        if not self.loaded and "param_state" in state_dict:
            added = add_missing_param_groups(
                state_dict["optimizer"]["param_groups"], self.opt.optimizer.param_groups, self.allowed_groups
            )
            self.summary["added_param_groups"] = added
        return self.original_load_state_dict(state_dict)
