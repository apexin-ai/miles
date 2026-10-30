"""Export a complete native Qwen checkpoint, including frozen PLE/MTP assets.

Unlike visual snapshots, this includes every decoder tensor in the native
checkpoint. --plan-only validates mapping and shapes without loading weights.
"""

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf

_EXPERT = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.(?:experts\.)?linear_fc([12])\.weight")
_FROZEN_PLE = re.compile(
    r"model\.language_model\.layers\.\d+\.ple\.ple_embedding\."
    r"(?:layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes|ngram_embedding\.shard_\d+\.weight)"
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_args(base):
    config = json.loads((base / "config.json").read_text())
    text = config.get("text_config", config)
    return SimpleNamespace(
        num_layers=text["num_hidden_layers"],
        num_experts=text["num_experts"],
        hidden_size=text["hidden_size"],
        num_attention_heads=text["num_attention_heads"],
        num_query_groups=text["num_key_value_heads"],
        kv_channels=text.get("head_dim"),
        vocab_size=text["vocab_size"],
        hf_checkpoint=str(base),
        sglang_enable_ep_moe=False,
    )


def _convert(args, tensors):
    converted = {}
    for name, tensor in tensors.items():
        match = _EXPERT.fullmatch(name)
        if match:
            layer, projection = match.groups()
            if tensor.ndim != 3 or tensor.shape[0] != args.num_experts:
                raise ValueError(f"Unexpected complete expert layout: {name}, {tensor.shape}")
            suffix = "gate_up_proj" if projection == "1" else "down_proj"
            outputs = [(f"model.language_model.layers.{layer}.mlp.experts.{suffix}", tensor)]
        else:
            outputs = convert_to_hf(args, "qwen4_exp", "module.module." + name, tensor)
        for target, value in outputs:
            if target in converted:
                raise ValueError(f"Duplicate exported tensor: {target}")
            converted[target] = value
    return converted


def _check_inventory(base, converted):
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    missing, extra = set(weight_map) - set(converted), set(converted) - set(weight_map)
    unsupported = {name for name in missing if not (_FROZEN_PLE.fullmatch(name) or name.startswith("mtp."))}
    if extra or unsupported:
        raise ValueError(f"Incomplete model conversion: extra={sorted(extra)}, missing={sorted(unsupported)}")
    by_file = defaultdict(list)
    for name, filename in weight_map.items():
        by_file[filename].append(name)
    for filename, names in by_file.items():
        with safe_open(str(base / filename), framework="pt", device="cpu") as source:
            if set(source.keys()) != set(names):
                raise ValueError(f"Source shard and index disagree: {filename}")
            for name in set(names) & set(converted):
                if list(converted[name].shape) != source.get_slice(name).get_shape():
                    raise ValueError(f"Export shape differs from original native HF tensor: {name}")
    return index, by_file, missing


def _native_tensors(checkpoint, *, meta):
    reader = dcp.FileSystemReader(str(checkpoint))
    metadata = reader.read_metadata()
    state = {name: torch.empty(entry.size, dtype=entry.properties.dtype, device="meta" if meta else "cpu") for name, entry in metadata.state_dict_metadata.items() if isinstance(entry, dcp.metadata.TensorStorageMetadata) and "optimizer" not in name and "_state" not in name}
    if not meta:
        dcp.load(state, storage_reader=reader, planner=dcp.DefaultLoadPlanner(flatten_state_dict=False))
    return state


def _closed_audit(directory, iteration):
    initial = [json.loads(path.read_text()) for path in directory.glob("rank-*-initial.json")]
    world = initial[0]["world_size"] if initial else 0
    if not world or {row["rank"] for row in initial} != set(range(world)) or len(initial) != world:
        raise ValueError("Initial training audits do not cover every rank")
    records = []
    for row in initial:
        rank = row["rank"]
        final = json.loads((directory / f"rank-{rank:03d}-update-{iteration + 1:06d}.json").read_text())
        reload = json.loads((directory / f"checkpoint-reload/rank-{rank:03d}.json").read_text())
        if final["rank"] != rank or final["update"] != iteration + 1:
            raise ValueError("Final training audit has the wrong rank or update")
        if row["language_frozen"] or final["initial_language_sha256"] == final["final_language_sha256"]:
            raise ValueError("Joint export requires completed decoder weight changes")
        if not final["final_language_sha256"] or final["visual_sha256"] == row["visual_sha256"]:
            raise ValueError("Joint export requires completed visual and language audits")
        if not reload["passed"] or reload["checkpoint_iteration"] != iteration or reload["rank"] != rank:
            raise ValueError("The native model-only checkpoint has no qualified complete reload")
        records.append({"rank": rank, "complete_parameter_sha256": reload["complete_parameter_sha256"]})
    return records


def _write_shards(base, output, converted, index, by_file, missing):
    total_size = 0
    shared = []
    for filename, names in by_file.items():
        target = output / filename
        if set(names) <= missing:
            target.hardlink_to((base / filename).resolve())
            shared.append(filename)
            with safe_open(str(target), framework="pt", device="cpu") as source:
                total_size += sum(source.get_tensor(name).numel() * source.get_tensor(name).element_size() for name in names)
            continue
        with safe_open(str(base / filename), framework="pt", device="cpu") as source:
            tensors = {name: (converted[name] if name in converted else source.get_tensor(name)).contiguous() for name in names}
            save_file(tensors, str(target), metadata={"format": "pt"})
            with safe_open(str(target), framework="pt", device="cpu") as restored:
                if set(restored.keys()) != set(names) or any(not torch.equal(restored.get_tensor(name), value) for name, value in tensors.items()):
                    raise ValueError(f"Full-model export did not round-trip exactly: {filename}")
            total_size += sum(value.numel() * value.element_size() for value in tensors.values())
        print(json.dumps({"saved": filename, "tensors": len(names)}), flush=True)
    index.setdefault("metadata", {})["total_size"] = total_size
    (output / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for source in base.iterdir():
        if source.is_file() and not source.name.endswith((".safetensors", ".bin", ".pt")) and source.name != "model.safetensors.index.json":
            shutil.copy2(source, output / source.name)
    return shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace an existing export or plan")
    if not args.plan_only and args.audit_dir is None:
        parser.error("A complete joint export requires --audit-dir")
    converted = _convert(_model_args(args.base_model), _native_tensors(args.checkpoint, meta=True))
    index, by_file, missing = _check_inventory(args.base_model, converted)
    receipt = {
        "complete": args.plan_only,
        "plan_only": args.plan_only,
        "checkpoint": str(args.checkpoint),
        "base_model": str(args.base_model),
        "native_metadata_sha256": _sha(args.checkpoint / ".metadata"),
        "source_index_sha256": _sha(args.base_model / "model.safetensors.index.json"),
        "source_config_sha256": _sha(args.base_model / "config.json"),
        "exporter_sha256": _sha(__file__),
        "native_export_tensors": len(converted),
        "unchanged_passthrough_keys": sorted(missing),
        "tensor_shapes": {name: list(value.shape) for name, value in converted.items()},
    }
    if args.plan_only:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({key: value for key, value in receipt.items() if key != "tensor_shapes"}))
        return
    iteration = int(args.checkpoint.name.removeprefix("iter_"))
    receipt["closed_rank_audits"] = _closed_audit(args.audit_dir, iteration)
    stage = args.output.with_name(args.output.name + ".incomplete")
    stage.mkdir()
    converted = _convert(_model_args(args.base_model), _native_tensors(args.checkpoint, meta=False))
    receipt["shared_immutable_source_shards"] = _write_shards(args.base_model, stage, converted, index, by_file, missing)
    receipt.update(complete=True, model_only=True, optimizer_resume_qualified=False, hf_index_sha256=_sha(stage / "model.safetensors.index.json"), update=iteration + 1)
    (stage / "joint-export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    stage.rename(args.output)
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
