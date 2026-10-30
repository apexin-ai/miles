"""Rebuild the measured Qwen visual-only result figures from committed receipts.

Example:
    python -m tools.plot_qwen38_vision160_results
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ASSETS = Path(__file__).resolve().parents[1] / "docs/assets/images/qwen3-8-flash-next-vision-sft"
_COLORS = {"baseline": "#718096", "trained": "#007f78", "lost": "#c05448", "format": "#bba37a"}


def _style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def _save(figure, path, footer):
    figure.text(0.5, 0.01, footer, ha="center", fontsize=9, color="#465466")
    figure.tight_layout(rect=(0, 0.07, 1, 0.91))
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_scores(report, output):
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    figure.suptitle("Qwen3.8-Flash-Next: vision-only SFT", fontsize=17, fontweight="bold")
    for axis, (name, metrics) in zip(axes, report["datasets"].items(), strict=True):
        correct = [metrics["baseline_correct"], metrics["trained_correct"]]
        accuracy = [100 * count / metrics["n"] for count in correct]
        bars = axis.bar(["Original base", "Step 160"], accuracy, color=[_COLORS["baseline"], _COLORS["trained"]], width=0.55)
        for bar, count, value in zip(bars, correct, accuracy, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{count}/{metrics['n']}\n{value:.2f}%", ha="center", fontsize=12)
        axis.set(title=report["dataset_labels"][name], ylim=(0, 100), ylabel="Exact integer accuracy (%)")
        axis.text(0.5, 0.82, f"+{metrics['delta_percentage_points']:.2f} percentage points", transform=axis.transAxes, ha="center", color=_COLORS["trained"], fontsize=12)
        _style(axis)
    _save(figure, output / "u160-eval-gain.png", "Two measured endpoints only | Frozen language | Thinking enabled (low) | 16,384 output tokens")


def _plot_pairs(report, output):
    metrics = list(report["datasets"].values())
    labels = [report["dataset_labels"][name] for name in report["datasets"]]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    figure.suptitle("Paired predictions: gains, regressions and answer validity", fontsize=16, fontweight="bold")
    positions = [0, 1]
    width = 0.34
    for offset, field, label, color in [
        (-width / 2, "wrong_to_right", "Wrong to right", _COLORS["trained"]),
        (width / 2, "right_to_wrong", "Right to wrong", _COLORS["lost"]),
    ]:
        values = [item[field] for item in metrics]
        bars = axes[0].bar([x + offset for x in positions], values, width, label=label, color=color)
        axes[0].bar_label(bars, padding=4)
    axes[0].set(xticks=positions, xticklabels=labels, ylabel="Number of scenes", ylim=(0, 270))
    axes[0].legend(frameon=False)
    valid = [item["net_correct_change_both_valid"] for item in metrics]
    transitions = [item["net_correct_change_validity_transitions"] for item in metrics]
    axes[1].bar(positions, valid, width=0.55, label="Both answers valid", color=_COLORS["trained"])
    axes[1].bar(positions, transitions, bottom=valid, width=0.55, label="Validity transitions", color=_COLORS["format"])
    for x, a, b in zip(positions, valid, transitions, strict=True):
        axes[1].text(x, a / 2, str(a), ha="center", color="white", fontsize=12)
        axes[1].text(x, a + b + 2, f"Net +{a + b}", ha="center", fontsize=12)
    axes[1].set(xticks=positions, xticklabels=labels, ylabel="Net additional correct scenes", ylim=(0, 85))
    axes[1].legend(frameon=False)
    for axis in axes:
        _style(axis)
    _save(figure, output / "u160-paired-changes.png", "Same scene IDs, questions, labels, image pixels and expanded input tokens at both endpoints")


def _plot_loss(loss, output):
    points = loss["points"]
    figure, axis = plt.subplots(figsize=(11, 5.2))
    figure.suptitle("Actual training loss: continuation from update 64 to 160", fontsize=16, fontweight="bold")
    axis.plot([point["update"] for point in points], [point["loss"] for point in points], color=_COLORS["trained"], linewidth=1.2, marker=".", markersize=3)
    for left, right, epoch, color in [(64.5, 96.5, 3, "#f0f5f7"), (96.5, 128.5, 4, "#ffffff"), (128.5, 160.5, 5, "#f0f5f7")]:
        axis.axvspan(left, right, color=color, zorder=0)
        selected = [point["loss"] for point in points if left < point["update"] < right]
        axis.text((left + right) / 2, 0.94, f"Epoch {epoch}: mean {sum(selected) / len(selected):.3f}", transform=axis.get_xaxis_transform(), ha="center", fontsize=10)
    axis.set(xlabel="Cumulative optimizer updates", ylabel="Supervised-token training loss", xlim=(64, 161), ylim=(0, max(point["loss"] for point in points) * 1.15))
    _style(axis)
    _save(figure, output / "u160-training-loss.png", "96 actual logged points; no smoothing | LR 1e-6 | Adam reset at update 64 | Different batch at each point")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_ASSETS / "u160-results.json")
    parser.add_argument("--loss", type=Path, default=_ASSETS / "u160-training-loss.json")
    parser.add_argument("--output", type=Path, default=_ASSETS)
    args = parser.parse_args()
    report = json.loads(args.data.read_text())
    loss = json.loads(args.loss.read_text())
    if not report["complete"] or [point["update"] for point in loss["points"]] != list(range(65, 161)):
        raise ValueError("Expected complete endpoint results and all 96 continuation loss points")
    args.output.mkdir(parents=True, exist_ok=True)
    _plot_scores(report, args.output)
    _plot_pairs(report, args.output)
    _plot_loss(loss, args.output)


if __name__ == "__main__":
    main()
