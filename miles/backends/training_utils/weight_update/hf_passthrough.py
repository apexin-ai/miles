"""Complete a raw HF export with the base checkpoint tensors the training model does not hold.

A raw HF export only covers the training model's parameters. Some base checkpoints hold more: frozen
buffers that are never parameters (e.g. an n-gram embedding table), or a head or tower the run does not
build. The export still copies the base ``config.json``, which declares those parts, so without them the
checkpoint cannot be loaded as the model it claims to be.

A model registers which key groups of its base checkpoint an export may take from the base
(``register_hf_export_passthrough``). ``complete_hf_export`` then, on the writer rank and before the
export is marked complete:

- checks the key set: every exported key exists in the base with the same dtype and shape, every base
  key is exported or passed through, and a group is either exported completely or passed through
  completely;
- checks that every shard read or written holds exactly the bytes its header declares;
- copies the passthrough tensors byte for byte (whole base shards when a shard holds only passthrough
  keys) and checks their bytes against the base.

Any violation raises ``ExportIncomplete``.

``normalize_shard_names`` (writer rank, after completion, also for raw exports without a passthrough) then names
every shard ``model-{i:05d}-of-{N:05d}.safetensors``: the exporter's shards first in write order, then the whole
base shards in base order, then the passthrough shards. Without it one directory mixed model-NNNNN, the base's
model-NNNNN-of-00131 and passthrough-NNNNN, and a tool counting shards by -of- saw 31 of 131 (SMOKE REPO-3).

``python -m miles.backends.training_utils.weight_update.hf_passthrough <export> <base> [--expect-unchanged RE]``
compares all keys of a finished export with the base (``verify_export``): passthrough and ``--expect-unchanged``
keys must be byte identical, and at least one exported key must have changed. It prints how many keys of each
registered group changed and exits 1 on a violation.
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
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

INDEX = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
RECEIPT = "passthrough.json"
_READ = 64 << 20
_THREADS = 8
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}  # fmt: skip


class ExportIncomplete(RuntimeError):
    pass


@dataclass(frozen=True)
class PassthroughGroup:
    """Base checkpoint keys (full-matched by ``pattern``) an export may take from the base.

    ``only_if_absent=False``: the training model never exports the group; it always comes from the base.
    ``only_if_absent=True``: it comes from the base only when the export holds none of it (a part the run
    did not build); an export that holds all of it keeps its own copy.
    """

    pattern: str
    only_if_absent: bool = False


@dataclass(frozen=True)
class ExportPassthrough:
    model_type: str
    groups: Mapping[str, PassthroughGroup]


_REGISTRY: dict[str, Callable[[Namespace], Mapping[str, PassthroughGroup]]] = {}


def register_hf_export_passthrough(
    model_type: str, groups: Callable[[Namespace], Mapping[str, PassthroughGroup]]
) -> None:
    """Register the passthrough groups of an HF ``model_type``; ``groups`` gets the training args."""
    _REGISTRY[model_type] = groups


def hf_export_passthrough(model_type: str | None, args: Namespace) -> ExportPassthrough | None:
    """The passthrough of the model a run trains, fixed when the run starts; None: plain raw export."""
    if model_type not in _REGISTRY:
        return None
    return ExportPassthrough(model_type, dict(_REGISTRY[model_type](args)))


def read_header(path: Path) -> tuple[int, dict]:
    """(byte offset of the data section, tensor entries) of a safetensors file."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return 8 + length, header


def weight_map(directory: Path) -> dict[str, str]:
    """``{key: shard}`` from the index, or from a single ``model.safetensors``."""
    if (directory / INDEX).is_file():
        return json.loads((directory / INDEX).read_text())["weight_map"]
    if (directory / SINGLE_FILE).is_file():
        return dict.fromkeys(read_header(directory / SINGLE_FILE)[1], SINGLE_FILE)
    raise ExportIncomplete(f"{directory} has neither {INDEX} nor {SINGLE_FILE}")


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


def tensor_hashes(directory: Path, files: Mapping[str, str], keys) -> dict[str, str]:
    """sha256 of each tensor's raw bytes."""
    headers = _headers(directory, (files[k] for k in keys))

    def one(key):
        data_start, header = headers[files[key]]
        begin, end = header[key]["data_offsets"]
        return key, _tensor_hash(directory / files[key], data_start + begin, data_start + end)

    with ThreadPoolExecutor(_THREADS) as pool:
        return dict(pool.map(one, sorted(keys)))


def plan(base_map: Mapping[str, str], export_map: Mapping[str, str], groups: Mapping[str, PassthroughGroup]):
    """(names of the groups passed through, their keys), or ExportIncomplete naming what is wrong."""
    problems = []
    extra = sorted(set(export_map) - set(base_map))
    if extra:
        problems.append(f"{len(extra)} exported keys are not in the base checkpoint: {extra[:5]}")
    passed, selected = [], set()
    for name, group in groups.items():
        pattern = re.compile(group.pattern)
        members = {k for k in base_map if pattern.fullmatch(k)}
        if not members:
            continue
        exported = members & set(export_map)
        if group.only_if_absent and exported:
            if exported != members:
                missing = sorted(members - exported)
                problems.append(f"group {name} is exported but {len(missing)} of its keys are missing: {missing[:5]}")
            continue
        if exported:
            problems.append(
                f"group {name} comes from the base but {len(exported)} keys were exported: {sorted(exported)[:5]}"
            )
        passed.append(name)
        selected |= members
    unexplained = sorted(set(base_map) - set(export_map) - selected)
    if unexplained:
        problems.append(f"{len(unexplained)} base keys are neither exported nor passed through: {unexplained[:5]}")
    if problems:
        raise ExportIncomplete("HF export does not match the base checkpoint:\n  " + "\n  ".join(problems))
    return sorted(passed), selected


def check_payloads(directory: Path, files) -> None:
    """Each shard's tensors tile its data section exactly, each with the size its dtype and shape declare."""
    problems = []
    for name, (data_start, header) in _headers(directory, files).items():
        size = (directory / name).stat().st_size - data_start
        end = 0
        for (begin, stop), key, entry in sorted((entry["data_offsets"], key, entry) for key, entry in header.items()):
            itemsize = _DTYPE_BYTES.get(entry["dtype"])
            expected = None if itemsize is None else itemsize * math.prod(entry["shape"])
            if begin != end:
                problems.append(f"{name}: {key} starts at {begin}, previous tensor ends at {end}")
            elif expected is not None and stop - begin != expected:
                problems.append(
                    f"{name}: {key} spans {stop - begin} bytes, {entry['dtype']}{entry['shape']} is {expected}"
                )
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


def _check_base_config(base_dir: Path, model_type: str) -> None:
    config = base_dir / "config.json"
    if not config.is_file():
        raise ExportIncomplete(f"base checkpoint {base_dir} has no config.json")
    found = json.loads(config.read_text()).get("model_type")
    if found != model_type:
        raise ExportIncomplete(
            f"base checkpoint {base_dir} is now model_type {found!r}, the run trains {model_type!r}"
        )


def _write_raw_shard(target: Path, source: Path, keys) -> None:
    data_start, header = read_header(source)
    entries, offset = {}, 0
    for key in sorted(keys):
        begin, end = header[key]["data_offsets"]
        entries[key] = {
            "dtype": header[key]["dtype"],
            "shape": header[key]["shape"],
            "data_offsets": [offset, offset + end - begin],
        }
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


def complete_hf_export(export_dir, base_dir, passthrough: ExportPassthrough) -> dict:
    """Add the passthrough tensors of ``base_dir`` to the raw export in ``export_dir``; returns the receipt.

    The export must have a ``model.safetensors.index.json``. The base must still be the checkpoint the run
    started from: a missing directory, index or config, or another ``model_type``, is an error.
    """
    export_dir, base_dir = Path(export_dir), Path(base_dir)
    if not base_dir.is_dir():
        raise ExportIncomplete(f"base checkpoint {base_dir} is not a local directory")
    _check_base_config(base_dir, passthrough.model_type)
    base_map = weight_map(base_dir)
    export_map = json.loads((export_dir / INDEX).read_text())["weight_map"]
    passed, selected = plan(base_map, export_map, passthrough.groups)
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
        if keys == by_file[filename] and not (export_dir / filename).exists():
            shutil.copyfile(base_dir / filename, export_dir / filename)
            whole.append(filename)
            target = filename
        else:
            partial += 1
            target = f"passthrough-{partial:05d}.safetensors"
            _write_raw_shard(export_dir / target, base_dir / filename, keys)
            partial_files.append(target)
        new_map.update(dict.fromkeys(keys, target))

    check_payloads(export_dir, new_map.values())
    total_size = sum(
        entry["data_offsets"][1] - entry["data_offsets"][0]
        for _, header in _headers(export_dir, new_map.values()).values()
        for entry in header.values()
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
        "passthrough_groups": passed,
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
    shards by name (a raw export without a passthrough: the exporter's zero-padded write order). A ``*.safetensors``
    file the index does not name fails the export. Renames go through temporary names, since a final name may be
    held by another shard (a whole base shard keeps the base's name until now). Tensor bytes are not touched.
    With a receipt, ``shard_renames`` is added to it. Runs on the writer rank after every collective and before
    ``.complete``: a failure leaves an export that write_checkpoint_dir removes.
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


def _report_groups(base_dir: Path) -> Mapping[str, PassthroughGroup]:
    """The registered groups of the base's model_type, for reporting only (as a run without MTP training)."""
    config = base_dir / "config.json"
    model_type = json.loads(config.read_text()).get("model_type") if config.is_file() else None
    passthrough = hf_export_passthrough(model_type, Namespace(enable_mtp_training=False))
    return {} if passthrough is None else passthrough.groups


def verify_export(export_dir, base_dir, *, expect_unchanged=()) -> dict:
    """Key-by-key comparison of a completed export with its base; raises ExportIncomplete on a violation.

    The keys that must equal the base are the receipt's passthrough keys plus ``expect_unchanged`` (regexes,
    searched); the export must hold every base key, with the base dtype and shape, and change at least one.
    """
    export_dir, base_dir = Path(export_dir), Path(base_dir)
    base_map, export_map = weight_map(base_dir), weight_map(export_dir)
    if set(base_map) != set(export_map):
        raise ExportIncomplete(
            f"keys differ: missing {sorted(set(base_map) - set(export_map))[:5]}, "
            f"extra {sorted(set(export_map) - set(base_map))[:5]}"
        )
    _check_dtype_and_shape(base_dir, base_map, export_dir, export_map)
    check_payloads(base_dir, base_map.values())
    check_payloads(export_dir, export_map.values())
    passthrough = set(json.loads((export_dir / RECEIPT).read_text())["passthrough_sha256"])
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
    patterns = {name: re.compile(group.pattern) for name, group in _report_groups(base_dir).items()}
    for key in base_map:
        group = next((name for name, pattern in patterns.items() if pattern.fullmatch(key)), "other")
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
    # Under ``python -m`` this file runs as __main__; the models register into the imported module, so the
    # check goes through that one.
    import miles.backends.megatron_utils.megatron_to_hf  # noqa: F401  (registers each model's passthrough)
    from miles.backends.training_utils.weight_update import hf_passthrough

    parser = argparse.ArgumentParser(
        description="Key-by-key check of a completed HF export against its base checkpoint"
    )
    parser.add_argument("export")
    parser.add_argument("base")
    parser.add_argument(
        "--expect-unchanged", action="append", default=[], help="regex of keys that must equal the base"
    )
    args = parser.parse_args(argv)
    try:
        report = hf_passthrough.verify_export(args.export, args.base, expect_unchanged=args.expect_unchanged)
    except hf_passthrough.ExportIncomplete as exc:
        print(f"FAILED: {exc}")
        return 1
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
