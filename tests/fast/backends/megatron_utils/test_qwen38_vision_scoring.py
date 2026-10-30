"""Last-line scoring and lossless migration of retained native predictions."""

import hashlib
import json

import pytest
from examples.qwen3_8_next_vision_sft.rescore import rescore_evaluation
from examples.qwen3_8_next_vision_sft.score import LEGACY_PROTOCOL, PROTOCOL, answer_of


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>Count.</think>\nExplanation mentions 7.\n\n**3**\n", 3),
        ("3\n2", 2),
        ("3\nExplanation", None),
        ("<think>Unfinished: 3", None),
        ("<think>A</think>1</think>2", None),
        ("\n -2.\n", -2),
        ("Answer: 3", None),
        ("", None),
    ],
)
def test_answer_uses_last_nonempty_final_line_and_requires_complete_thinking(text, expected):
    assert answer_of(text) == expected


def test_legacy_first_line_scoring_is_explicit():
    assert answer_of("3\n2", protocol=LEGACY_PROTOCOL) == 3
    assert answer_of("Explanation\n2", protocol=LEGACY_PROTOCOL) is None
    with pytest.raises(ValueError, match="Unknown"):
        answer_of("3", protocol="different")


@pytest.fixture
def original_run(tmp_path):
    data, original = tmp_path / "data", tmp_path / "original"
    data.mkdir()
    original.mkdir()
    sources = [{"id": str(i), "messages": [{"content": "<image>Count."}], "answer": "2"} for i in range(2)]
    dataset = data / "validation.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in sources))
    rows = [
        {
            "id": source["id"],
            "question": source["messages"][0]["content"],
            "answer": source["answer"],
            "control": "image",
            "dataset_role": "validation",
            "update": 0,
            "assistant_prefill": "<think>\n",
            "generated_text": "Done.</think>\nExplanation\n2" if i == 0 else "Still thinking: 2",
            "extracted_answer": None,
        }
        for i, source in enumerate(sources)
    ]
    for row in rows:
        row["prediction"] = row["assistant_prefill"] + row["generated_text"]
    raw = original / "update-000000-validation.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "complete": True,
        "protocol": LEGACY_PROTOCOL,
        "args": {"limit": 0, "splits": ["validation"], "controls": ["image"], "output": str(original)},
        "data_files_sha256": {"validation": hashlib.sha256(dataset.read_bytes()).hexdigest()},
        "results": [{"update": 0, "metrics": {"validation": {"image": {"correct": 0}}}}],
    }
    (original / "evaluation.json").write_text(json.dumps(manifest))
    return data, original, tmp_path / "rescored", rows


def test_rescore_retains_generation_and_original_evidence(original_run):
    data, original, output, before = original_run
    raw = original / "update-000000-validation.jsonl"
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    result = rescore_evaluation(original, output, data=data)
    assert result["complete"] and result["protocol"] == PROTOCOL
    assert result["results"][0]["metrics"]["validation"]["image"] == {
        "n": 2,
        "correct": 1,
        "invalid": 1,
        "accuracy": 0.5,
    }
    assert result["rescore"]["source_predictions_sha256"][raw.name] == digest
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest
    after = [json.loads(line) for line in (output / raw.name).read_text().splitlines()]
    for x, y in zip(before, after, strict=True):
        assert {k: v for k, v in y.items() if k not in ("extracted_answer", "original_extracted_answer")} == {
            k: v for k, v in x.items() if k != "extracted_answer"
        }
    assert [row["extracted_answer"] for row in after] == [2, None]


@pytest.mark.parametrize("failure", ["dataset", "aggregate", "raw_prefix"])
def test_rescore_rejects_changed_or_inconsistent_evidence(original_run, failure):
    data, original, output, _ = original_run
    if failure == "dataset":
        (data / "validation.jsonl").write_text("changed")
    elif failure == "aggregate":
        path = original / "evaluation.json"
        manifest = json.loads(path.read_text())
        manifest["results"][0]["metrics"]["validation"]["image"]["correct"] = 1
        path.write_text(json.dumps(manifest))
    else:
        path = original / "update-000000-validation.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["generated_text"] = "changed"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        rescore_evaluation(original, output, data=data)
    assert not output.exists()
