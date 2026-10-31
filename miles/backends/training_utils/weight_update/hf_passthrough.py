"""Complete an HF export with the base checkpoint tensors the training model does not hold, and verify it key by key.

A raw HF export only covers the live model's parameters. For Qwen3.8-Flash-Next text SFT the base checkpoint
also has the frozen PLE n-gram table (131 keys, never a parameter), the MTP head (31 keys, not built unless MTP
is trained) and the vision tower (333 keys, not built in text mode). Without them the exported config still
declares those parts and the checkpoint cannot be served.

``complete_hf_export`` (rank 0, right after the raw export) copies exactly those keys from the base checkpoint:

- the passthrough set is explicit: the PLE table always; ``mtp.*`` unless MTP is trained; ``model.visual.*``
  when the model exported none of it. A group is either exported completely or passed through completely;
- the export must not have keys the base lacks, every base key must end up in the export, and every exported
  tensor keeps the base dtype and shape;
- base shards holding only passthrough keys are copied whole (``MILES_HF_PASSTHROUGH=hardlink`` links them
  instead, for an immutable base on the same filesystem); passthrough keys of mixed shards are copied byte
  for byte into ``passthrough-NNNNN.safetensors``;
- every shard read or written must hold exactly the bytes its header declares (no truncation, gap or overlap);
- every passthrough tensor's raw bytes are hashed in the export and in the base and must match.

``normalize_shard_names`` (rank 0, after completion, also for raw exports without a policy) then names every shard
``model-{i:05d}-of-{N:05d}.safetensors``: the exporter's shards first in write order, then the whole base shards
in base order, then the passthrough shards. Without it one directory mixed model-NNNNN, the base's
model-NNNNN-of-00131 and passthrough-NNNNN, and a tool counting shards by -of- saw 31 of 131 (SMOKE REPO-3).

``python -m miles.backends.training_utils.weight_update.hf_passthrough <export> <base> [--expect-unchanged RE]``
compares all keys of a finished export with the base: passthrough and ``--expect-unchanged`` keys must be byte
identical, and at least one exported key must have changed. It prints how many keys of each group changed.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

INDEX = "model.safetensors.index.json"
RECEIPT = "passthrough.json"
_READ = 64 << 20
_THREADS = 8

QWEN38_GROUPS = {
    "ple_table": re.compile(
        r"model\.language_model\.layers\.\d+\.ple\.ple_embedding\."
        r"(?:layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes|ngram_embedding\.shard_\d+\.weight)"
    ),
    "mtp": re.compile(r"mtp\..+"),
    "visual": re.compile(r"model\.visual\..+"),
}
# HF model_type -> key groups that may be passed through from the base checkpoint
POLICIES = {"qwen4_exp": QWEN38_GROUPS}
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}  # fmt: skip


class ExportIncomplete(RuntimeError):
    pass


def read_header(path: Path) -> tuple[int, dict]:
    """(byte offset of the data section, tensor entries) of a safetensors file."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return 8 + length, header


def _index(directory: Path) -> dict[str, str]:
    return json.loads((directory / INDEX).read_text())["weight_map"]


def _headers(directory: Path, files) -> dict[str, tuple[int, dict]]:
    return {name: read_header(directory / name) for name in sorted(set(files))}


def _tensor_hash(path: Path, start: int, end: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining:
            chunk = f.read(min(_READ, remaining))
            if not chunk:
                raise ExportIncomplete(f"{path} is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def tensor_hashes(directory: Path, weight_map: dict[str, str], keys) -> dict[str, str]:
    """sha256 of each tensor's raw bytes, read in parallel."""
    headers = _headers(directory, (weight_map[k] for k in keys))

    def one(key):
        data_start, header = headers[weight_map[key]]
        begin, end = header[key]["data_offsets"]
        return key, _tensor_hash(directory / weight_map[key], data_start + begin, data_start + end)

    with ThreadPoolExecutor(_THREADS) as pool:
        return dict(pool.map(one, sorted(keys)))


def passthrough_groups(groups: dict[str, re.Pattern], exported, *, mtp_trained: bool) -> set[str]:
    names = {"ple_table"}
    if not mtp_trained:
        names.add("mtp")
    if "visual" in groups and not any(groups["visual"].fullmatch(k) for k in exported):
        names.add("visual")
    return names & set(groups)


def plan(base_map: dict[str, str], export_map: dict[str, str], groups: dict[str, re.Pattern], passthrough: set[str]):
    """Passthrough keys, or ExportIncomplete naming what is wrong."""
    problems = []
    extra = sorted(set(export_map) - set(base_map))
    if extra:
        problems.append(f"{len(extra)} exported keys are not in the base checkpoint: {extra[:5]}")
    selected = set()
    for name, pattern in groups.items():
        members = {k for k in base_map if pattern.fullmatch(k)}
        exported = members & set(export_map)
        if name in passthrough:
            if exported:
                problems.append(f"group {name} is passed through from the base but {len(exported)} of its keys were exported: {sorted(exported)[:5]}")
            selected |= members
        elif exported != members:
            problems.append(f"group {name} is trained but {len(members - exported)} of its keys are missing: {sorted(members - exported)[:5]}")
    unexplained = sorted(set(base_map) - set(export_map) - selected)
    if unexplained:
        problems.append(f"{len(unexplained)} base keys are neither exported nor passed through: {unexplained[:5]}")
    if problems:
        raise ExportIncomplete("HF export does not match the base checkpoint:\n  " + "\n  ".join(problems))
    return selected


def check_payloads(directory: Path, files) -> None:
    """Each shard's tensors tile its data section exactly, each with the size its dtype and shape declare."""
    problems = []
    for name, (data_start, header) in _headers(directory, files).items():
        size = (directory / name).stat().st_size - data_start
        spans = sorted((entry["data_offsets"], key, entry) for key, entry in header.items())
        end = 0
        for (begin, stop), key, entry in spans:
            itemsize = _DTYPE_BYTES.get(entry["dtype"])
            if begin != end:
                problems.append(f"{name}: {key} starts at {begin}, previous tensor ends at {end}")
            elif itemsize is not None and stop - begin != itemsize * math.prod(entry["shape"]):
                expected = itemsize * math.prod(entry["shape"])
                problems.append(f"{name}: {key} spans {stop - begin} bytes, {entry['dtype']}{entry['shape']} is {expected}")
            end = max(end, stop)
        if end != size:
            problems.append(f"{name}: header declares {end} data bytes, the file holds {size} (truncated or padded)")
    if problems:
        raise ExportIncomplete(f"{len(problems)} safetensors payload problems in {directory}: {problems[:5]}")


def _check_dtype_and_shape(base_dir, base_map, export_dir, export_map) -> None:
    base_headers = _headers(base_dir, base_map.values())
    export_headers = _headers(export_dir, export_map.values())
    wrong = []
    for key, filename in export_map.items():
        got = export_headers[filename][1][key]
        want = base_headers[base_map[key]][1][key]
        if (got["dtype"], got["shape"]) != (want["dtype"], want["shape"]):
            wrong.append(f"{key}: {got['dtype']}{got['shape']} vs base {want['dtype']}{want['shape']}")
    if wrong:
        raise ExportIncomplete(f"{len(wrong)} exported tensors differ from the base in dtype or shape: {wrong[:5]}")


def _write_raw_shard(target: Path, source: Path, keys) -> None:
    data_start, header = read_header(source)
    entries, offset = {}, 0
    for key in sorted(keys):
        begin, end = header[key]["data_offsets"]
        entries[key] = {"dtype": header[key]["dtype"], "shape": header[key]["shape"], "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    blob = json.dumps({"__metadata__": {"format": "pt"}, **entries}, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(source, "rb") as src, open(target, "xb") as dst:
        dst.write(struct.pack("<Q", len(blob)) + blob)
        for key in sorted(keys):
            begin, end = header[key]["data_offsets"]
            src.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                chunk = src.read(min(_READ, remaining))
                if not chunk:
                    raise ExportIncomplete(f"{source} ends {remaining} bytes before the end of {key}")
                dst.write(chunk)
                remaining -= len(chunk)


def _place_whole_shard(target: Path, source: Path, mode: str) -> None:
    if target.exists():
        raise ExportIncomplete(f"{target.name} already exists in the export")
    if mode == "hardlink":
        os.link(source.resolve(), target)
    else:
        shutil.copyfile(source, target)


def _policy(base_dir) -> dict[str, re.Pattern] | None:
    config = Path(base_dir) / "config.json"
    if not config.is_file():
        return None
    return POLICIES.get(json.loads(config.read_text()).get("model_type"))


def applies(base_dir) -> bool:
    """Whether exports from this base checkpoint get completed (a local checkpoint with a policy)."""
    return base_dir is not None and _policy(base_dir) is not None


def complete_hf_export(
    export_dir, base_dir, *, mtp_trained: bool, mode: str | None = None, required: bool = False
) -> dict | None:
    """Add the passthrough keys to a raw HF export and verify it; returns the receipt (None: no policy).

    ``required``: the run is known to train a model with a policy, so a base that no longer has one (config.json
    gone or changed, the directory gone) is an error rather than a plain export.
    """
    groups = _policy(base_dir) if base_dir is not None else None
    if groups is None:
        if required:
            raise ExportIncomplete(
                f"this run trained from a base whose export must be completed, but {base_dir} no longer has a "
                "config.json with a passthrough policy; restore the base checkpoint the run started from"
            )
        return None
    export_dir, base_dir = Path(export_dir), Path(base_dir)
    mode = mode or os.environ.get("MILES_HF_PASSTHROUGH", "copy")
    if mode not in ("copy", "hardlink"):
        raise ValueError(f"MILES_HF_PASSTHROUGH must be copy or hardlink, got {mode!r}")

    base_map, export_map = _index(base_dir), _index(export_dir)
    passthrough = passthrough_groups(groups, export_map, mtp_trained=mtp_trained)
    selected = plan(base_map, export_map, groups, passthrough)
    _check_dtype_and_shape(base_dir, base_map, export_dir, export_map)
    check_payloads(base_dir, base_map.values())
    check_payloads(export_dir, export_map.values())

    by_file = defaultdict(set)
    for key, filename in base_map.items():
        by_file[filename].add(key)
    new_map = dict(export_map)
    whole, partial, partial_files = [], 0, []
    for filename in sorted(by_file):
        keys = by_file[filename] & selected
        if not keys:
            continue
        if keys == by_file[filename]:
            _place_whole_shard(export_dir / filename, base_dir / filename, mode)
            whole.append(filename)
            target = filename
        else:
            partial += 1
            target = f"passthrough-{partial:05d}.safetensors"
            _write_raw_shard(export_dir / target, base_dir / filename, keys)
            partial_files.append(target)
        new_map.update({key: target for key in keys})

    if set(new_map) != set(base_map):
        raise ExportIncomplete("completed export keys differ from the base keys")
    check_payloads(export_dir, new_map.values())
    headers = _headers(export_dir, new_map.values())
    total_size = sum(
        entry["data_offsets"][1] - entry["data_offsets"][0] for _, header in headers.values() for entry in header.values()
    )
    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(new_map.items()))}
    tmp = export_dir / (INDEX + ".tmp")
    tmp.write_text(json.dumps(index, indent=2))
    tmp.replace(export_dir / INDEX)

    exported_hashes = tensor_hashes(export_dir, new_map, selected)
    base_hashes = tensor_hashes(base_dir, base_map, selected)
    differ = sorted(k for k in selected if exported_hashes[k] != base_hashes[k])
    if differ:
        raise ExportIncomplete(f"{len(differ)} passthrough tensors differ from the base bytes: {differ[:5]}")

    receipt = {
        "base": str(base_dir),
        "mode": mode,
        "passthrough_groups": sorted(passthrough),
        "exported_keys": len(export_map),
        "passthrough_keys": len(selected),
        "total_keys": len(new_map),
        "exported_shards": sorted(set(export_map.values())),
        "whole_base_shards": whole,
        "passthrough_shards": partial,
        "passthrough_files": partial_files,
        "passthrough_sha256": {k: exported_hashes[k] for k in sorted(selected)},
    }
    (export_dir / RECEIPT).write_text(json.dumps(receipt, indent=1))
    return receipt


SHARD_NAME = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")


def normalize_shard_names(export_dir, receipt: dict | None = None) -> dict[str, str]:
    """Rename every indexed shard to model-i-of-N and rewrite the index; returns {old: new} for the renamed ones.

    Order: ``receipt``'s exported_shards, whole_base_shards, passthrough_files (a completed export), or all indexed
    shards by name (a raw export without a policy: the exporter's zero-padded write order). A ``*.safetensors``
    file the index does not name fails the export. Renames go through temporary names, since a final name may be
    held by another shard (a whole base shard keeps the base's name until now). Tensor bytes are not touched;
    hardlinked base shards are renamed in the export only. With a receipt, ``shard_renames`` is added to it.
    Runs before ``.complete``: a failure leaves an export that write_checkpoint_dir removes.
    """
    export_dir = Path(export_dir)
    index = json.loads((export_dir / INDEX).read_text())
    indexed = set(index["weight_map"].values())
    stray = sorted(p.name for p in export_dir.glob("*.safetensors") if p.name not in indexed)
    if stray:
        raise ExportIncomplete(f"{export_dir}: safetensors files the index does not name: {stray[:5]}")
    if receipt is not None:
        order = [*receipt["exported_shards"], *receipt["whole_base_shards"], *receipt["passthrough_files"]]
        if len(order) != len(set(order)) or set(order) != indexed:
            raise ExportIncomplete(
                f"{export_dir}: the receipt's shards {sorted(set(order) ^ indexed)[:5]} do not match the index"
            )
    else:
        order = sorted(indexed)
    total = len(order)
    target = {old: f"model-{i:05d}-of-{total:05d}.safetensors" for i, old in enumerate(order, 1)}
    renames = {old: new for old, new in target.items() if old != new}
    # step 1 frees every name that is changing; step 2 takes the final names (no final name is held any more:
    # a shard keeping its name is the only one with that target)
    temporary = {old: f".rename-{i:05d}.safetensors.tmp" for i, old in enumerate(renames, 1)}
    for old, tmp in temporary.items():
        os.rename(export_dir / old, export_dir / tmp)
    for old, tmp in temporary.items():
        if (export_dir / renames[old]).exists():
            raise ExportIncomplete(f"{export_dir}: {renames[old]} exists while renaming {old}")
        os.rename(export_dir / tmp, export_dir / renames[old])
    if renames:
        index["weight_map"] = {k: target[f] for k, f in sorted(index["weight_map"].items())}
        tmp = export_dir / (INDEX + ".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(export_dir / INDEX)
    if receipt is not None:
        receipt["shard_renames"] = renames
        (export_dir / RECEIPT).write_text(json.dumps(receipt, indent=1))
    return renames


def verify_export(export_dir, base_dir, *, expect_unchanged=()) -> dict:
    """Key-by-key comparison of a finished export with its base; raises ExportIncomplete on a violation."""
    export_dir, base_dir = Path(export_dir), Path(base_dir)
    groups = POLICIES[json.loads((base_dir / "config.json").read_text())["model_type"]]
    base_map, export_map = _index(base_dir), _index(export_dir)
    if set(base_map) != set(export_map):
        raise ExportIncomplete(
            f"keys differ: missing {sorted(set(base_map) - set(export_map))[:5]}, extra {sorted(set(export_map) - set(base_map))[:5]}"
        )
    _check_dtype_and_shape(base_dir, base_map, export_dir, export_map)
    check_payloads(base_dir, base_map.values())
    check_payloads(export_dir, export_map.values())
    receipt = json.loads((export_dir / RECEIPT).read_text())
    passthrough = set(receipt["passthrough_sha256"])
    exported_hashes = tensor_hashes(export_dir, export_map, export_map)
    base_hashes = tensor_hashes(base_dir, base_map, base_map)
    changed = {k for k in base_map if exported_hashes[k] != base_hashes[k]}
    must_match = passthrough | {k for k in base_map for p in expect_unchanged if re.search(p, k)}
    report = {
        "keys": len(base_map),
        "passthrough": len(passthrough),
        "changed": len(changed),
        "changed_by_group": {},
        "unchanged_trained_keys": sorted(set(base_map) - changed - must_match),
    }
    for key in base_map:
        group = next((name for name, pattern in groups.items() if pattern.fullmatch(key)), "other")
        counts = report["changed_by_group"].setdefault(group, [0, 0])
        counts[0] += key in changed
        counts[1] += 1
    bad = sorted(changed & must_match)
    if bad:
        raise ExportIncomplete(f"{len(bad)} keys that must equal the base changed: {bad[:5]}")
    if not changed - passthrough:
        raise ExportIncomplete("no exported key differs from the base: the export holds untrained weights")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Key-by-key check of a completed HF export against its base checkpoint")
    parser.add_argument("export")
    parser.add_argument("base")
    parser.add_argument("--expect-unchanged", action="append", default=[], help="regex of keys that must equal the base")
    args = parser.parse_args(argv)
    try:
        report = verify_export(args.export, args.base, expect_unchanged=args.expect_unchanged)
    except ExportIncomplete as exc:
        print(f"FAILED: {exc}")
        return 1
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
