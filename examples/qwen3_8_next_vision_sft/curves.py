"""Plot only completed, fixed-protocol evaluation points; never interpolate scores."""

import argparse
import hashlib
import json
from pathlib import Path

from examples.qwen3_8_next_vision_sft.score import PROTOCOL, answer_of, metrics_from_answers, paired_answers, read_rows


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_points(evaluation, data, *, split):
    manifest = json.loads((evaluation / "evaluation.json").read_text())
    if not manifest["complete"] or manifest["args"]["limit"] or manifest["protocol"] != PROTOCOL:
        raise ValueError("A curve requires a complete, untruncated fixed-protocol evaluation")
    if manifest["data_files_sha256"][split] != _sha(data / f"{split}.jsonl"):
        raise ValueError("The evaluation used a different dataset")
    updates = [point["update"] for point in manifest["results"]]
    if len(updates) < 2 or updates[0] != 0 or updates != sorted(set(updates)):
        raise ValueError("Expected a baseline followed by distinct increasing checkpoint updates")
    sources = read_rows(data / f"{split}.jsonl")
    baseline = read_rows(evaluation / f"update-000000-{split}.jsonl")
    controls = manifest["args"]["controls"]
    if "image" not in controls or len(set(controls)) != len(controls):
        raise ValueError("A visual SFT curve requires an image arm and distinct controls")
    points = []
    hashes = {}
    for update, result in zip(updates, manifest["results"], strict=True):
        path = evaluation / f"update-{update:06d}-{split}.jsonl"
        records = read_rows(path)
        if len(records) != len(sources) * len(controls):
            raise ValueError("Evaluation is missing prediction records or has duplicates")
        metrics = metrics_from_answers(paired_answers(sources, baseline, records))
        accuracy = {}
        for control in controls:
            selected = [row for row in records if row["control"] == control]
            old = [row for row in baseline if row["control"] == control]
            if len(selected) != len(sources) or len(old) != len(sources):
                raise ValueError("A negative control has incomplete coverage")
            for source, before, after in zip(sources, old, selected, strict=True):
                if (
                    before["id"] != source["id"]
                    or after["id"] != source["id"]
                    or after["dataset_role"] != split
                    or after["update"] != update
                    or after["question"] != source["messages"][0]["content"]
                    or after["answer"] != source["answer"]
                    or before["image_id"] != after["image_id"]
                    or before["image_pixel_sha256"] != after["image_pixel_sha256"]
                    or before["input_ids_sha256"] != after["input_ids_sha256"]
                ):
                    raise ValueError("A checkpoint changed the fixed evaluation request")
            # Score raw text again rather than trusting the precomputed extracted answer.
            correct = sum(answer_of(row["prediction"]) == int(row["answer"]) for row in selected)
            if result["metrics"][split][control]["correct"] != correct:
                raise ValueError("Recorded aggregate disagrees with raw predictions")
            accuracy[control] = 100 * correct / len(sources)
        points.append({"update": update, "accuracy_pct": accuracy, "image_paired_metrics": metrics})
        hashes[path.name] = _sha(path)
    return {
        "protocol": PROTOCOL,
        "split": split,
        "n": len(sources),
        "points": points,
        "input_sha256": {"evaluation.json": _sha(evaluation / "evaluation.json"), **hashes},
        "observed_endpoint_improvement": points[-1]["image_paired_metrics"]["delta_percentage_points"] > 0,
        "all_adjacent_points_nondecreasing": all(
            later["accuracy_pct"]["image"] >= earlier["accuracy_pct"]["image"]
            for earlier, later in zip(points[:-1], points[1:], strict=True)
        ),
    }


def plot_points(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = report["points"]
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = {"image": "#15656c", "no_image": "#9a542f", "swapped_image": "#7a7185"}
    for control in points[0]["accuracy_pct"]:
        axis.plot(
            [point["update"] for point in points],
            [point["accuracy_pct"][control] for point in points],
            marker="o",
            color=colors[control],
            label=control.replace("_", " "),
        )
    for point in points:
        correct = point["image_paired_metrics"]["trained_correct"]
        axis.annotate(
            f"{correct}/{report['n']}",
            (point["update"], point["accuracy_pct"]["image"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.axhline(
        points[0]["image_paired_metrics"]["always_one_accuracy_pct"],
        color="#888888",
        linestyle=":",
        label="always one (reference)",
    )
    axis.set(
        title=f"Qwen3.8-Flash-Next visual SFT | {report['split']}",
        xlabel="Completed optimizer updates",
        ylabel="Strict integer accuracy (%)",
        xticks=[point["update"] for point in points],
        ylim=(0, 100),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="best")
    figure.text(
        0.5, -0.015, "Dots are measured checkpoints; lines only connect observations.", ha="center", fontsize=9
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect_points(args.evaluation, args.data, split=args.split)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "curve.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_points(report, args.output / "eval.png")


if __name__ == "__main__":
    main()
