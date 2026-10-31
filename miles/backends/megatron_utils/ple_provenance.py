"""Bind a run and its checkpoints to the bytes of the frozen PLE table it trains with.

The Qwen3.8-Next PLE n-gram table and its hash parameters are not parameters: every start reads them again
from ``--hf-checkpoint`` and no Megatron checkpoint holds them. A resume against a base whose table changed,
under the same path or another, would silently train a different model.

- A new binding (HF weights, a release checkpoint, an empty ``--load``, ``--finetune``) records a manifest of
  the base: dtype, shape and raw-byte sha256 of each PLE tensor, the config / index sha256 and the HF revision.
  ``MILES_PLE_MANIFEST=<file>`` makes a new binding verify the base against that manifest first.
- Every save writes the run's manifest to ``<save>/iter_NNNNNNN/ple_provenance.json`` *before* Megatron's
  ``save_checkpoint`` runs, so neither a sync save (tracker written before it returns) nor an async one
  (tracker written by a later finalize) can publish an iteration without its manifest.
- A resume from a trained checkpoint reads that checkpoint's manifest and re-hashes the current base against
  it. PLE bytes, dtype, shape or the config differing is an error; only the path, the index or the revision
  differing is logged. A trained checkpoint without a manifest is an error unless
  ``MILES_PLE_PROVENANCE_ADOPT=1``, which adopts the current base (checkpoints written before this check).
- A completed HF export must carry the PLE bytes of the run's manifest (``check_export``).

Bases without PLE tensors are not touched: every hook is a no-op for them.
The hashing is split over all ranks.
"""

import hashlib
import json
import logging
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch.distributed as dist

logger = logging.getLogger(__name__)

MANIFEST_NAME = "ple_provenance.json"
ADOPT_ENV = "MILES_PLE_PROVENANCE_ADOPT"
MANIFEST_ENV = "MILES_PLE_MANIFEST"
INDEX = "model.safetensors.index.json"
PLE_KEY = re.compile(
    r"model\.language_model\.layers\.\d+\.ple\.ple_embedding\."
    r"(?:layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes|ngram_embedding\.shard_\d+\.weight)"
)
_READ = 64 << 20
_THREADS = 8
_run_manifest: dict | None = None


class PleProvenanceError(RuntimeError):
    pass


def _ple_keys(hf_checkpoint) -> tuple[dict[str, str], list[str]]:
    """(weight map, sorted PLE keys) of a base; no keys when it has no safetensors index."""
    index = Path(hf_checkpoint) / INDEX
    if not index.is_file():
        return {}, []
    weight_map = json.loads(index.read_text())["weight_map"]
    return weight_map, sorted(k for k in weight_map if PLE_KEY.fullmatch(k))


def _read_header(path: Path) -> tuple[int, dict]:
    """(byte offset of the data section, tensor entries) of a safetensors file."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return 8 + length, header


def _tensor_sha256(path: Path, start: int, end: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining:
            chunk = f.read(min(_READ, remaining))
            if not chunk:
                raise PleProvenanceError(f"{path} is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hf_revision(hf_checkpoint) -> str | None:
    """The commit huggingface_hub recorded for the download, if every file names the same one."""
    metadata = (Path(hf_checkpoint) / ".cache" / "huggingface" / "download").glob("*.metadata")
    commits = {lines[0].strip() for path in metadata if (lines := path.read_text().splitlines())}
    return commits.pop() if len(commits) == 1 else None


def compute_manifest(hf_checkpoint) -> dict:
    """Manifest of the base's PLE tensors; collective when torch.distributed is initialized."""
    base = Path(hf_checkpoint)
    weight_map, keys = _ple_keys(base)
    if not keys:
        raise PleProvenanceError(f"{base} has no PLE tensors")
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    mine = keys[rank::world]
    headers = {name: _read_header(base / name) for name in {weight_map[k] for k in mine}}

    def entry(key):
        data_start, header = headers[weight_map[key]]
        begin, end = header[key]["data_offsets"]
        digest = _tensor_sha256(base / weight_map[key], data_start + begin, data_start + end)
        return key, {"dtype": header[key]["dtype"], "shape": header[key]["shape"], "sha256": digest}

    with ThreadPoolExecutor(_THREADS) as pool:
        local = dict(pool.map(entry, mine))
    gathered = [local]
    if distributed:
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
    return {
        "hf_checkpoint": str(base.resolve()),
        "revision": _hf_revision(base),
        "config_sha256": _sha256(base / "config.json"),
        "index_sha256": _sha256(base / INDEX),
        "ple": dict(sorted((key, e) for part in gathered for key, e in part.items())),
    }


def compare(expected: dict, actual: dict) -> tuple[list[str], list[str]]:
    """(errors, notes): errors for PLE bytes / layout or the config, notes for path, index or revision."""
    errors, notes = [], []
    missing = sorted(expected["ple"].keys() - actual["ple"].keys())
    extra = sorted(actual["ple"].keys() - expected["ple"].keys())
    if missing or extra:
        errors.append(f"PLE keys differ: missing {missing[:3]}, extra {extra[:3]}")
    changed = sorted(
        k for k in expected["ple"].keys() & actual["ple"].keys() if expected["ple"][k] != actual["ple"][k]
    )
    if changed:
        errors.append(f"{len(changed)} PLE tensors differ in bytes, dtype or shape: {changed[:3]}")
    if expected["config_sha256"] != actual["config_sha256"]:
        errors.append("config.json differs")
    for field in ("hf_checkpoint", "index_sha256", "revision"):
        if expected.get(field) != actual.get(field):
            notes.append(f"{field}: {expected.get(field)} -> {actual.get(field)}")
    return errors, notes


def manifest_path(checkpoint_root, iteration: int) -> Path:
    return Path(checkpoint_root) / f"iter_{iteration:07d}" / MANIFEST_NAME


def run_manifest() -> dict | None:
    return _run_manifest


def on_load(hf_checkpoint, *, trained_checkpoint_dir) -> dict | None:
    """Start the run's binding, or verify the base against the trained checkpoint it resumes.

    ``trained_checkpoint_dir`` is the iteration directory of a checkpoint written by training, or None when
    the load starts a new binding. Collective when torch.distributed is initialized and the base has PLE.
    """
    global _run_manifest
    expected_path = None if trained_checkpoint_dir is None else Path(trained_checkpoint_dir) / MANIFEST_NAME
    if hf_checkpoint is None or not _ple_keys(hf_checkpoint)[1]:
        if expected_path is not None and expected_path.is_file():
            raise PleProvenanceError(
                f"{expected_path} binds this run to a PLE table, but --hf-checkpoint {hf_checkpoint} "
                "is missing or has no PLE tensors"
            )
        _run_manifest = None
        return None

    current = compute_manifest(hf_checkpoint)
    expected, source = current, None
    if expected_path is not None:
        if expected_path.is_file():
            expected, source = json.loads(expected_path.read_text()), str(expected_path)
        elif os.environ.get(ADOPT_ENV) == "1":
            logger.warning(f"{expected_path} is missing; {ADOPT_ENV}=1 adopts the current base {hf_checkpoint}")
        else:
            raise PleProvenanceError(
                f"resuming {trained_checkpoint_dir} but {expected_path} is missing, so the PLE table this run "
                f"trained with is unknown. Set {ADOPT_ENV}=1 only if the base is known to be unchanged."
            )
    elif os.environ.get(MANIFEST_ENV):
        # a new binding pinned to a known manifest, e.g. the one recorded in the run's lock file
        source = os.environ[MANIFEST_ENV]
        expected = json.loads(Path(source).read_text())
    if source is not None:
        errors, notes = compare(expected, current)
        if errors:
            raise PleProvenanceError(f"base {hf_checkpoint} does not match {source}: " + "; ".join(errors))
        for note in notes:
            logger.warning(f"PLE provenance {source}: {note} (PLE bytes and config unchanged)")
    _run_manifest = expected | {"hf_checkpoint": current["hf_checkpoint"]}
    action = "verified" if source else "recorded"
    logger.info(f"PLE provenance: {len(current['ple'])} PLE tensors of {hf_checkpoint} {action}")
    return _run_manifest


def _write_durably(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def before_save(save_dir, iteration: int) -> None:
    """Write the run's manifest into the iteration directory before Megatron's save can publish it.

    Rank 0 (the rank that writes Megatron's tracker) writes it; a failure is raised on every rank, so no rank
    enters the save's collectives alone. Collective when torch.distributed is initialized and the run is bound.
    """
    if _run_manifest is None:
        return
    error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            _write_durably(manifest_path(save_dir, iteration), _run_manifest)
        except OSError as e:
            error = f"cannot write {manifest_path(save_dir, iteration)}: {e!r}"
    if dist.is_initialized():
        box = [error]
        dist.broadcast_object_list(box, src=0)
        error = box[0]
    if error is not None:
        raise PleProvenanceError(f"not saving iteration {iteration}: {error}")


def check_export(passthrough_sha256: dict[str, str]) -> None:
    """The PLE tensors a completed HF export copied from the base must be the ones the run is bound to."""
    if _run_manifest is None:
        return
    differ = sorted(k for k, entry in _run_manifest["ple"].items() if passthrough_sha256.get(k) != entry["sha256"])
    if differ:
        raise PleProvenanceError(
            f"{len(differ)} exported PLE tensors are not the ones this run trained with: {differ[:3]}"
        )
