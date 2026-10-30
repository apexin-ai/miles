"""Re-score retained native predictions without running generation again."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from examples.qwen3_8_next_vision_sft.score import LEGACY_PROTOCOL, PROTOCOL, answer_of, read_rows


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rescore_evaluation(evaluation, output, *, data):
    started = time.monotonic()
    original = json.loads((evaluation / "evaluation.json").read_text())
    previous = original["protocol"]
    if not original["complete"] or original["args"]["limit"] or previous not in (LEGACY_PROTOCOL, PROTOCOL):
        raise ValueError("Re-scoring requires complete, unlimited predictions with a known scoring protocol")
    controls = original["args"]["controls"]
    if not controls or len(set(controls)) != len(controls):
        raise ValueError("Evaluation controls must be nonempty and distinct")
    measured, source_hashes = {}, {}
    for split in original["args"]["splits"]:
        source_path = data / f"{split}.jsonl"
        if _sha(source_path) != original["data_files_sha256"][split]:
            raise ValueError("Re-scoring cannot change the evaluation dataset")
        sources = read_rows(source_path)
        if not sources or len({row["id"] for row in sources}) != len(sources):
            raise ValueError("Source scenes must be nonempty and unique")
        for point in original["results"]:
            update = point["update"]
            name = f"update-{update:06d}-{split}.jsonl"
            path = evaluation / name
            rows = read_rows(path)
            if len(rows) != len(sources) * len(controls) or any(row["control"] not in controls for row in rows):
                raise ValueError("Raw predictions must cover every scene and control exactly once")
            for control in controls:
                selected = [row for row in rows if row["control"] == control]
                if len(selected) != len(sources):
                    raise ValueError("Incomplete prediction control")
                for source, row in zip(sources, selected, strict=True):
                    if (
                        row["id"] != source["id"]
                        or row["question"] != source["messages"][0]["content"]
                        or row["answer"] != source["answer"]
                        or row["update"] != update
                        or row["dataset_role"] != split
                    ):
                        raise ValueError("Re-scoring cannot change a measured request or target")
                    if (
                        "generated_text" in row
                        and row["prediction"] != row["assistant_prefill"] + row["generated_text"]
                    ):
                        raise ValueError("Raw generated text and assistant prefix disagree")
                    extracted = answer_of(row["prediction"], protocol=previous)
                    if "extracted_answer" in row and row["extracted_answer"] != extracted:
                        raise ValueError("Original extracted answer disagrees with its scoring protocol")
                old_correct = sum(
                    answer_of(row["prediction"], protocol=previous) == int(row["answer"]) for row in selected
                )
                if old_correct != point["metrics"][split][control]["correct"]:
                    raise ValueError("Original aggregate disagrees with raw predictions")
            measured[split, update] = rows
            source_hashes[name] = _sha(path)
    manifest = copy.deepcopy(original)
    manifest.update(
        complete=False,
        protocol=PROTOCOL,
        scorer_sha256=_sha(Path(__file__).with_name("score.py")),
        rescore={
            "generation_unchanged": True,
            "source_protocol": previous,
            "source_manifest_sha256": _sha(evaluation / "evaluation.json"),
            "source_predictions_sha256": source_hashes,
        },
    )
    manifest["args"]["output"] = str(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "evaluation.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    for point in manifest["results"]:
        update = point["update"]
        for split in manifest["args"]["splits"]:
            rows = copy.deepcopy(measured[split, update])
            for row in rows:
                row["original_extracted_answer"] = answer_of(row["prediction"], protocol=previous)
                row["extracted_answer"] = answer_of(row["prediction"])
            with (output / f"update-{update:06d}-{split}.jsonl").open("x") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=True) + "\n")
            for control in controls:
                selected = [row for row in rows if row["control"] == control]
                correct = sum(row["extracted_answer"] == int(row["answer"]) for row in selected)
                invalid = sum(row["extracted_answer"] is None for row in selected)
                point["metrics"][split][control] = {
                    "n": len(selected),
                    "correct": correct,
                    "invalid": invalid,
                    "accuracy": correct / len(selected),
                }
    manifest.update(complete=True, rescore_seconds=time.monotonic() - started)
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rescore_evaluation(args.evaluation, args.output, data=args.data)


if __name__ == "__main__":
    main()
