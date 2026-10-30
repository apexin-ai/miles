"""Frozen integer scoring and paired comparison for geometric_stack_count."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PROTOCOL = "integer_last_final_line_v2"
LEGACY_PROTOCOL = "integer_first_final_line_v1"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_of(text: str, *, protocol: str = PROTOCOL) -> int | None:
    if protocol not in (PROTOCOL, LEGACY_PROTOCOL):
        raise ValueError(f"Unknown answer-scoring protocol: {protocol}")
    if text.count("</think>") > 1:
        return None
    if "<think>" in text and "</think>" not in text:
        return None
    body = text.rsplit("</think>", 1)[-1]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[-1 if protocol == PROTOCOL else 0].replace("*", "").replace("`", "").strip()
    matched = re.fullmatch(r"([+-]?\d+)\.?", line)
    return int(matched.group(1)) if matched else None


def paired_answers(sources: list[dict], before: list[dict], after: list[dict]) -> list[dict]:
    before = [row for row in before if row.get("control", "image") == "image"]
    after = [row for row in after if row.get("control", "image") == "image"]
    if not sources or len(sources) != len(before) or len(sources) != len(after):
        raise ValueError("Both prediction files must cover every held-out image exactly once")
    if len({row["id"] for row in sources}) != len(sources):
        raise ValueError("Held-out sample IDs must be unique")
    answers = []
    for source, baseline, trained in zip(sources, before, after, strict=True):
        for prediction in (baseline, trained):
            if (
                prediction["id"] != source["id"]
                or prediction["question"] != source["messages"][0]["content"]
                or prediction["answer"] != source["answer"]
                or prediction.get("control", "image") != "image"
            ):
                raise ValueError(f"Prediction/source identity differs: {source['id']}")
        answers.append(
            {
                "id": source["id"],
                "answer": int(source["answer"]),
                "sheet_kind": source["generator_metadata"]["variant"]["target_kind"],
                "baseline_answer": answer_of(baseline["prediction"]),
                "trained_answer": answer_of(trained["prediction"]),
            }
        )
    return answers


def _counts(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        a, b, gold = row["baseline_answer"], row["trained_answer"], row["answer"]
        ac, bc = a == gold, b == gold
        both_valid = a is not None and b is not None
        counts.update(
            {
                "n": 1,
                "baseline_correct": int(ac),
                "trained_correct": int(bc),
                "wrong_to_right": int(not ac and bc),
                "right_to_wrong": int(ac and not bc),
                "baseline_invalid": int(a is None),
                "trained_invalid": int(b is None),
                "both_valid_n": int(both_valid),
                "both_valid_baseline_correct": int(both_valid and ac),
                "both_valid_trained_correct": int(both_valid and bc),
                "baseline_invalid_to_trained_correct": int(a is None and bc),
                "baseline_correct_to_trained_invalid": int(ac and b is None),
            }
        )
    n = counts["n"]
    if not n:
        raise ValueError("Cannot score an empty sample set")
    return {
        **dict(counts),
        "baseline_accuracy_pct": 100 * counts["baseline_correct"] / n,
        "trained_accuracy_pct": 100 * counts["trained_correct"] / n,
        "delta_percentage_points": 100 * (counts["trained_correct"] - counts["baseline_correct"]) / n,
        "net_correct_change_both_valid": counts["both_valid_trained_correct"] - counts["both_valid_baseline_correct"],
        "net_correct_change_validity_transitions": (
            counts["baseline_invalid_to_trained_correct"] - counts["baseline_correct_to_trained_invalid"]
        ),
    }


def metrics_from_answers(rows: list[dict]) -> dict:
    result = _counts(rows)
    result["by_sheet_kind"] = {
        kind: _counts([row for row in rows if row["sheet_kind"] == kind])
        for kind in sorted({row["sheet_kind"] for row in rows})
    }
    result["by_answer"] = {
        str(answer): _counts([row for row in rows if row["answer"] == answer])
        for answer in sorted({row["answer"] for row in rows})
    }
    for model in ("baseline", "trained"):
        result[f"{model}_answer_macro_accuracy_pct"] = (
            100
            * sum(group[f"{model}_correct"] / group["n"] for group in result["by_answer"].values())
            / len(result["by_answer"])
        )
    result["always_one_accuracy_pct"] = 100 * sum(row["answer"] == 1 for row in rows) / len(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    answers = paired_answers(read_rows(args.data), read_rows(args.baseline), read_rows(args.trained))
    result = {
        "protocol": PROTOCOL,
        "metrics": metrics_from_answers(answers),
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("data", args.data), ("baseline", args.baseline), ("trained", args.trained))
        },
    }
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
