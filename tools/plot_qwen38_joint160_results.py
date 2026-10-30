"""Plot measured final joint SFT results with the original and visual-only models."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ASSETS = Path(__file__).resolve().parents[1] / "docs/assets/images/qwen3-8-flash-next-joint-sft"
_COLORS = ["#718096", "#007f78", "#3366b3"]


def _style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def _save(figure, output, footer):
    figure.text(0.5, 0.015, footer, ha="center", fontsize=9, color="#465466")
    figure.tight_layout(rect=(0, 0.08, 1, 0.91))
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_scores(report, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    figure.suptitle("Qwen3.8-Flash-Next: final SFT model comparison", fontsize=17, fontweight="bold")
    labels = ["Original base", "Visual SFT\nstep 160", "Joint SFT\nsaved step 160"]
    models = ["baseline", "visual_sft160", "joint_sft160"]
    for axis, (dataset, title) in zip(axes, [("training1024", "Training set (1024 scenes)"), ("validation256", "Validation (256 scenes)")], strict=True):
        points = [report[model][dataset] for model in models]
        values = [point["accuracy_pct"] for point in points]
        bars = axis.bar(labels, values, color=_COLORS, width=0.6)
        for bar, point in zip(bars, points, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{point['correct']}/{point['n']}\n{point['accuracy_pct']:.2f}%", ha="center", fontsize=11)
        axis.set(title=title, ylabel="Exact integer accuracy (%)", ylim=(0, 100))
        _style(axis)
    _save(
        figure,
        output / "joint160-evaluation.png",
        "Identical evaluation inputs/settings; training schedules differ. Thinking enabled (low), 16,384 output tokens.\nJoint scores evaluate the validated saved checkpoint; exact live-model restoration remains unqualified.",
    )


def _plot_loss(report, output):
    points = report["training_loss"]["points"]
    figure, axis = plt.subplots(figsize=(12, 5.4))
    figure.suptitle("Joint language and visual SFT: measured training loss", fontsize=17, fontweight="bold")
    axis.plot([point["update"] for point in points], [point["loss"] for point in points], color=_COLORS[2], linewidth=1.1, marker=".", markersize=3)
    for epoch, mean in enumerate(report["training_loss"]["epoch_means"], 1):
        left, right = (epoch - 1) * 32 + 0.5, epoch * 32 + 0.5
        if epoch % 2:
            axis.axvspan(left, right, color="#f0f5f7", zorder=0)
        axis.text((left + right) / 2, 0.95, f"Epoch {epoch}\nmean {mean:.4f}", transform=axis.get_xaxis_transform(), ha="center", fontsize=10, va="top")
    axis.set(xlabel="Optimizer updates", ylabel="Supervised-token training loss", xlim=(0, 161), ylim=(0, max(point["loss"] for point in points) * 1.18))
    _style(axis)
    _save(figure, output / "joint160-training-loss.png", "160 actual logged points; no smoothing | Original base, fresh Adam | LR 1e-6 | Global batch 32 | Five epochs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_ASSETS / "joint160-results.json")
    parser.add_argument("--output", type=Path, default=_ASSETS)
    args = parser.parse_args()
    report = json.loads(args.data.read_text())
    if not report["complete"] or report["model_update"] != 160:
        raise ValueError("Expected a complete final step-160 joint-model evaluation")
    if [point["update"] for point in report["training_loss"]["points"]] != list(range(1, 161)):
        raise ValueError("Expected all 160 measured training loss values")
    for dataset, n in [("training1024", 1024), ("validation256", 256)]:
        for model in ["baseline", "visual_sft160", "joint_sft160"]:
            point = report[model][dataset]
            if point["n"] != n or abs(point["accuracy_pct"] - 100 * point["correct"] / n) > 1e-9:
                raise ValueError("The score and sample count disagree")
    args.output.mkdir(parents=True, exist_ok=True)
    _plot_scores(report, args.output)
    _plot_loss(report, args.output)


if __name__ == "__main__":
    main()
