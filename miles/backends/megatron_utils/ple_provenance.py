"""Bind a Qwen3.8 run to the bytes of the frozen PLE table it trains with.

The PLE n-gram table (131 HF keys: 128 shards and 3 hash parameters) is never a parameter: every
start reads it again from ``--hf-checkpoint`` and no Megatron checkpoint holds it. A resume against a
base whose table changed, under the same path or another, would silently train a different model.

- A fresh start (HF weights, empty ``--load``, ``--finetune``) records a manifest of the base: the
  dtype, shape and raw-byte sha256 of each PLE key, the config / index sha256 and the HF revision.
  ``MILES_PLE_MANIFEST=<file>`` makes the fresh start verify the base against that manifest first.
- Every save writes the run's manifest to ``<save>/iter_NNNNNNN/ple_provenance.json``.
- A resume reads the manifest of the checkpoint it loads and re-hashes the current base against it,
  never a manifest rebuilt from the new base. PLE bytes or the config differing is an error; only the
  path, the index or the revision differing is logged. A trained checkpoint without a manifest is an
  error unless ``MILES_PLE_PROVENANCE_ADOPT=1`` (then the current base is adopted and recorded).
- A completed HF export must carry the manifest's PLE bytes.

The hashing is split over all ranks (each reads about 102 GB / world size).
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import torch.distributed as dist

from miles.backends.training_utils.weight_update.hf_passthrough import read_header, tensor_hashes

logger = logging.getLogger(__name__)

MANIFEST_NAME = "ple_provenance.json"
MODEL_TYPE = "qwen4_exp"
PLE_KEY = re.compile(
    r"model\.language_model\.layers\.\d+\.ple\.ple_embedding\."
    r"(?:layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes|ngram_embedding\.shard_\d+\.weight)"
)
_run_manifest: dict | None = None


class PleProvenanceError(RuntimeError):
    pass


def applies(hf_checkpoint) -> bool:
    if hf_checkpoint is None:
        return False
    config = Path(hf_checkpoint) / "config.json"
    return config.is_file() and json.loads(config.read_text()).get("model_type") == MODEL_TYPE


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hf_revision(hf_checkpoint) -> str | None:
    """The commit huggingface_hub recorded for the download, if every file names the same one."""
    metadata = sorted((Path(hf_checkpoint) / ".cache" / "huggingface" / "download").glob("*.metadata"))
    commits = {path.read_text().splitlines()[0].strip() for path in metadata if path.read_text().strip()}
    return commits.pop() if len(commits) == 1 else None


def compute_manifest(hf_checkpoint) -> dict:
    """Manifest of the base's PLE table; collective when torch.distributed is initialized."""
    base = Path(hf_checkpoint)
    weight_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    keys = sorted(k for k in weight_map if PLE_KEY.fullmatch(k))
    if not keys:
        raise PleProvenanceError(f"{base} has no PLE table keys")
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    mine = keys[rank::world]
    headers = {name: read_header(base / name)[1] for name in {weight_map[k] for k in mine}}
    local = {
        key: {
            "dtype": headers[weight_map[key]][key]["dtype"],
            "shape": headers[weight_map[key]][key]["shape"],
            "sha256": digest,
        }
        for key, digest in tensor_hashes(base, weight_map, mine).items()
    }
    gathered = [None] * world
    if distributed:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    entries = {key: entry for part in gathered for key, entry in part.items()}
    return {
        "hf_checkpoint": str(base.resolve()),
        "revision": hf_revision(base),
        "config_sha256": _sha256(base / "config.json"),
        "index_sha256": _sha256(base / "model.safetensors.index.json"),
        "ple": dict(sorted(entries.items())),
    }


def compare(expected: dict, actual: dict) -> tuple[list[str], list[str]]:
    """(errors, notes): errors for PLE bytes / layout or config, notes for path, index or revision."""
    errors, notes = [], []
    missing = sorted(expected["ple"].keys() - actual["ple"].keys())
    extra = sorted(actual["ple"].keys() - expected["ple"].keys())
    if missing or extra:
        errors.append(f"PLE keys differ: missing {missing[:3]}, extra {extra[:3]}")
    changed = sorted(k for k in expected["ple"].keys() & actual["ple"].keys() if expected["ple"][k] != actual["ple"][k])
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


def on_load(hf_checkpoint, *, load_dir, iteration: int, resumed: bool) -> dict | None:
    """Establish (fresh start) or verify (resume) the run's PLE provenance."""
    global _run_manifest
    if not applies(hf_checkpoint):
        if resumed and load_dir is not None and manifest_path(load_dir, iteration).is_file():
            raise PleProvenanceError(
                f"{manifest_path(load_dir, iteration)} binds this run to a PLE table, but --hf-checkpoint "
                f"{hf_checkpoint} is missing or not a Qwen3.8 checkpoint"
            )
        _run_manifest = None
        return None
    current = compute_manifest(hf_checkpoint)
    if resumed:
        path = manifest_path(load_dir, iteration)
        if path.is_file():
            expected, source = json.loads(path.read_text()), str(path)
        elif os.environ.get("MILES_PLE_PROVENANCE_ADOPT") == "1":
            logger.warning(f"{path} is missing; MILES_PLE_PROVENANCE_ADOPT=1 adopts the current base {hf_checkpoint}")
            expected, source = current, None
        else:
            raise PleProvenanceError(
                f"resuming iteration {iteration} but {path} is missing, so the PLE table this run trained with "
                "is unknown. Set MILES_PLE_PROVENANCE_ADOPT=1 only if the base is known to be unchanged."
            )
    elif os.environ.get("MILES_PLE_MANIFEST"):
        source = os.environ["MILES_PLE_MANIFEST"]
        expected = json.loads(Path(source).read_text())
    else:
        expected, source = current, None
    if source is not None:
        errors, notes = compare(expected, current)
        if errors:
            raise PleProvenanceError(f"base {hf_checkpoint} does not match {source}: " + "; ".join(errors))
        for note in notes:
            logger.warning(f"PLE provenance {source}: {note} (PLE bytes and config unchanged)")
    _run_manifest = expected | {"hf_checkpoint": current["hf_checkpoint"]}
    logger.info(f"PLE provenance: {len(current['ple'])} PLE tensors of {hf_checkpoint} {'verified' if source else 'recorded'}")
    return _run_manifest


def on_save(save_dir, iteration: int) -> None:
    """Rank 0 writes the run's manifest next to the checkpoint just saved."""
    if _run_manifest is None or (dist.is_initialized() and dist.get_rank() != 0):
        return
    path = manifest_path(save_dir, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_run_manifest, indent=1))
    tmp.replace(path)


def check_export(passthrough_sha256: dict[str, str]) -> None:
    """The PLE tensors a completed HF export copied must be the run's."""
    if _run_manifest is None:
        return
    differ = sorted(k for k, entry in _run_manifest["ple"].items() if passthrough_sha256.get(k) != entry["sha256"])
    if differ:
        raise PleProvenanceError(f"{len(differ)} exported PLE tensors are not the ones this run trained with: {differ[:3]}")
