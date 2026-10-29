"""Combine disjoint checkpoint evaluations only after checking their exact protocol."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from examples.qwen3_8_next_vision_sft.curves import collect_points
from examples.qwen3_8_next_vision_sft.score import PROTOCOL


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(manifest):
    return {
        **{
            key: manifest[key]
            for key in ("protocol", "model_files_sha256", "data_files_sha256", "evaluator_sha256", "packages")
        },
        "args": {key: value for key, value in manifest["args"].items() if key not in ("output", "skip_baseline")},
    }


def merge_evaluations(evaluations, output, *, data):
    if not evaluations:
        raise ValueError("No checkpoint evaluations supplied")
    manifests = [json.loads((path / "evaluation.json").read_text()) for path in evaluations]
    protocol = _protocol(manifests[0])
    points, files, components = {}, [], {}
    for path, manifest in zip(evaluations, manifests, strict=True):
        if not manifest["complete"] or manifest["args"]["limit"] or manifest["protocol"] != PROTOCOL:
            raise ValueError("Every component must be a complete, unlimited native evaluation")
        if _protocol(manifest) != protocol:
            raise ValueError("Checkpoint workers used different evaluation protocols or sources")
        components[str(path)] = _sha(path / "evaluation.json")
        for result in manifest["results"]:
            update = result["update"]
            if update in points:
                raise ValueError(f"Duplicate measured checkpoint update: {update}")
            points[update] = result
            for split in manifest["args"]["splits"]:
                source = path / f"update-{update:06d}-{split}.jsonl"
                files.append((source, _sha(source)))
    if not points or min(points) != 0 or len(points) < 2:
        raise ValueError("A merged curve requires a measured baseline and trained checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    combined = {
        **manifests[0],
        "complete": False,
        "args": {**manifests[0]["args"], "output": str(output), "skip_baseline": False},
        "results": [points[update] for update in sorted(points)],
        "visual_snapshots": [snapshot for manifest in manifests for snapshot in manifest["visual_snapshots"]],
        "components_sha256": components,
        "seconds": sum(manifest["seconds"] for manifest in manifests),
    }
    target = output / "evaluation.json"
    target.write_text(json.dumps(combined, indent=2) + "\n")
    for source, expected in files:
        destination = output / source.name
        shutil.copyfile(source, destination)
        if _sha(destination) != expected:
            raise ValueError("Raw prediction copy differs from the measured component")
    combined["complete"] = True
    target.write_text(json.dumps(combined, indent=2) + "\n")
    try:
        for split in combined["args"]["splits"]:
            collect_points(output, data, split=split)
    except Exception:
        combined["complete"] = False
        target.write_text(json.dumps(combined, indent=2) + "\n")
        raise
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge_evaluations(args.evaluations, args.output, data=args.data)


if __name__ == "__main__":
    main()
