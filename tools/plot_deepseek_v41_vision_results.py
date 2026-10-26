"""Plot recorded 1017 training loss and final evaluation without interpolation.

Requires matplotlib and the sanitized recorded JSON, not model weights or GPUs.
Example:
    python tools/plot_deepseek_v41_vision_results.py \
        --data docs/assets/images/deepseek-v4-1-1017/curve_data.json \
        --output-dir docs/assets/images/deepseek-v4-1-1017
"""

import argparse
import json
import math
from pathlib import Path


def _validate(data):
    updates = data["recipe"]["updates"]
    for run in data["training"]:
        points = run["points"]
        if [point["step"] for point in points] != list(range(updates)):
            raise ValueError("Training curve must contain every recorded step exactly once")
        if any(not math.isfinite(point["loss"]) or point["loss"] <= 0 for point in points):
            raise ValueError("Log-scale loss requires finite positive recorded values")
    evaluation = data["evaluation"]
    for run in evaluation["runs"]:
        if any(not 0 <= value <= evaluation["n"] for value in run["correct"].values()):
            raise ValueError("Evaluation correct counts are out of range")


def _training_plot(data, plt, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for axis, scale in zip(axes, ("linear", "log"), strict=True):
        for run in data["training"]:
            steps = [point["step"] for point in run["points"]]
            losses = [point["loss"] for point in run["points"]]
            axis.plot(
                steps,
                losses,
                color=run["color"],
                label=run["label"],
                linewidth=2,
                marker="o",
                markersize=3.8,
            )
            axis.annotate(
                f"{losses[-1]:.4f}",
                xy=(steps[-1], losses[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                color=run["color"],
                va="center",
                fontsize=10,
            )
        axis.set_yscale(scale)
        axis.set_title(f"{scale.capitalize()} scale", loc="left", fontsize=12)
        axis.set_xlabel("Logged training step (zero-based)")
        axis.set_ylabel("Logged train/loss")
        axis.set_xlim(-0.5, data["recipe"]["updates"] + 3)
        axis.set_xticks([0, 8, 16, 24, data["recipe"]["updates"] - 1])
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        if scale == "linear":
            axis.set_ylim(bottom=0)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(f"{data['model']} | Recorded training loss", fontsize=17, fontweight="bold")
    recipe = data["recipe"]
    figure.text(
        0.5,
        0.88,
        f"{recipe['updates']} updates | batch {recipe['global_batch_size']} | "
        f"LR {recipe['learning_rate']:g} | weight decay {recipe['weight_decay']:g}",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    figure.text(
        0.5,
        0.035,
        "All raw batch losses. No smoothing or missing-step interpolation. "
        "Different batches each step; not validation loss.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    figure.subplots_adjust(top=0.78, bottom=0.17, left=0.065, right=0.965, wspace=0.23)
    figure.savefig(output_dir / "training_curves.png", dpi=160, facecolor="white")
    plt.close(figure)


def _evaluation_plot(data, plt, output_dir):
    evaluation = data["evaluation"]
    controls = ("image", "swap", "none")
    figure, axis = plt.subplots(figsize=(12.8, 5.8))
    width = 0.23
    for run_index, run in enumerate(evaluation["runs"]):
        positions = [index + (run_index - 1) * width for index in range(len(controls))]
        counts = [run["correct"][control] for control in controls]
        values = [100 * value / evaluation["n"] for value in counts]
        bars = axis.bar(positions, values, width=width, color=run["color"], label=run["label"])
        axis.bar_label(
            bars,
            labels=[f"{count}/{evaluation['n']}\n{value:.2f}%" for count, value in zip(counts, values, strict=True)],
            padding=5,
            fontsize=10,
        )
    majority = 100 * evaluation["majority_correct"] / evaluation["n"]
    axis.axhline(majority, color="#64748b", linestyle="--", linewidth=1.2)
    axis.text(2.43, majority + 0.6, f"Always answer 1: {majority:.2f}%", ha="right", color="#64748b", fontsize=10)
    axis.set_xticks(range(len(controls)), ("Real image", "Swapped image", "No image"))
    axis.set_ylabel("Strict accuracy (%)")
    axis.set_ylim(0, max(45, majority + 6))
    axis.set_xlim(-0.55, 2.55)
    axis.grid(axis="y", alpha=0.15)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", frameon=False, ncol=3)
    figure.suptitle(f"{data['model']} | Held-out evaluation", fontsize=17, fontweight="bold")
    figure.text(
        0.5,
        0.035,
        f"Same {evaluation['n']} scenes and exact scorer. Greedy 4096-token budget. "
        "Baseline/final results only; no intermediate checkpoint evaluations.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    figure.subplots_adjust(top=0.86, bottom=0.17, left=0.075, right=0.96)
    figure.savefig(output_dir / "eval_gain.png", dpi=160, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    _validate(data)

    # Plotting is optional; validation and importing this tool need no matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": "#334155"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _training_plot(data, plt, args.output_dir)
    _evaluation_plot(data, plt, args.output_dir)


if __name__ == "__main__":
    main()
