"""Evidence validation, independent of any successful GPU quality run."""

import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest
import torch
from examples.qwen3_8_next_vision_sft.curves import collect_points
from examples.qwen3_8_next_vision_sft.merge import merge_evaluations
from examples.qwen3_8_next_vision_sft.score import PROTOCOL, paired_answers
from safetensors.torch import save_file


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def evidence(tmp_path):
    data, evaluation = tmp_path / "data", tmp_path / "evaluation"
    data.mkdir()
    evaluation.mkdir()
    sources = [
        {
            "id": str(index),
            "messages": [{"content": "<image>How many?"}],
            "answer": str(index + 1),
            "generator_metadata": {"variant": {"target_kind": "square"}},
        }
        for index in range(3)
    ]
    _write_jsonl(data / "validation.jsonl", sources)
    results, predictions = [], []
    for update, answers in ((0, [1, 0, 0]), (8, [1, 2, 0]), (16, [0, 2, 0]), (24, [1, 2, 3])):
        rows = [
            {
                "id": source["id"],
                "question": source["messages"][0]["content"],
                "answer": source["answer"],
                "dataset_role": "validation",
                "control": control,
                "update": update,
                "image_id": source["id"] if control == "image" else None,
                "image_pixel_sha256": source["id"] if control == "image" else None,
                "input_ids_sha256": source["id"] + control,
                "prediction": str(answers[index]) if control == "image" else "1",
            }
            for control in ("image", "no_image")
            for index, source in enumerate(sources)
        ]
        _write_jsonl(evaluation / f"update-{update:06d}-validation.jsonl", rows)
        predictions.append(rows)
        results.append(
            {
                "update": update,
                "metrics": {
                    "validation": {
                        "image": {"correct": sum(answer == index + 1 for index, answer in enumerate(answers))},
                        "no_image": {"correct": 1},
                    }
                },
            }
        )
    manifest = {
        "complete": True,
        "protocol": PROTOCOL,
        "args": {"limit": 0, "controls": ["image", "no_image"]},
        "data_files_sha256": {"validation": hashlib.sha256((data / "validation.jsonl").read_bytes()).hexdigest()},
        "results": results,
    }
    (evaluation / "evaluation.json").write_text(json.dumps(manifest))
    return data, evaluation, sources, predictions


def test_paired_scoring_filters_controls_on_both_sides(evidence):
    _, _, sources, predictions = evidence
    pairs = paired_answers(sources, predictions[0], predictions[-1])
    assert len(pairs) == 3
    assert [row["trained_answer"] for row in pairs] == [1, 2, 3]


def test_curve_preserves_a_real_dip_instead_of_inventing_monotonicity(evidence):
    data, evaluation, _, _ = evidence
    curve = collect_points(evaluation, data, split="validation")
    assert curve["observed_endpoint_improvement"]
    assert not curve["all_adjacent_points_nondecreasing"]
    assert [p["image_paired_metrics"]["trained_correct"] for p in curve["points"]] == [1, 2, 1, 3]


@pytest.mark.parametrize("key,value", [("complete", False), ("protocol", "different")])
def test_curve_rejects_incomplete_or_changed_protocol(evidence, key, value):
    data, evaluation, _, _ = evidence
    path = evaluation / "evaluation.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete, untruncated"):
        collect_points(evaluation, data, split="validation")


@pytest.mark.parametrize("key,value", [("input_ids_sha256", "different"), ("image_id", "different")])
def test_curve_rejects_changed_checkpoint_requests(evidence, key, value):
    data, evaluation, _, predictions = evidence
    predictions[-1][0][key] = value
    _write_jsonl(evaluation / "update-000024-validation.jsonl", predictions[-1])
    with pytest.raises(ValueError, match="fixed evaluation request"):
        collect_points(evaluation, data, split="validation")


def test_curve_recomputes_accuracy_from_raw_text(evidence):
    data, evaluation, _, predictions = evidence
    predictions[-1][0]["prediction"] = "Not an integer"
    _write_jsonl(evaluation / "update-000024-validation.jsonl", predictions[-1])
    with pytest.raises(ValueError, match="aggregate disagrees"):
        collect_points(evaluation, data, split="validation")


@pytest.fixture
def visual_snapshot(tmp_path):
    model, audit = tmp_path / "model", tmp_path / "audit"
    model.mkdir()
    identity = {}
    for name in ("config.json", "model.safetensors.index.json"):
        (model / name).write_text("{}")
        identity[name] = hashlib.sha256((model / name).read_bytes()).hexdigest()
    snapshot = audit / "snapshots/update-000008"
    snapshot.mkdir(parents=True)
    weights = snapshot / "visual.safetensors"
    save_file({"model.visual.test": torch.ones(2)}, str(weights))
    (snapshot / "visual.json").write_text(
        json.dumps(
            {
                "complete": True,
                "frozen_language": True,
                "source_identity": identity,
                "update": 8,
                "file_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            }
        )
    )
    for rank in range(2):
        (audit / f"rank-{rank:03d}-initial.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": 2,
                    "num_updates": 8,
                    "source_identity": identity,
                    "language_sha256": "unchanged",
                }
            )
        )
        (audit / f"rank-{rank:03d}-update-000008.json").write_text(
            json.dumps({"rank": rank, "update": 8, "final_language_sha256": "unchanged"})
        )
    return snapshot, SimpleNamespace(model=model)


def test_visual_snapshot_requires_all_frozen_language_ranks(visual_snapshot):
    from examples.qwen3_8_next_vision_sft.evaluate import _load_snapshot

    snapshot, args = visual_snapshot
    assert _load_snapshot(snapshot, args)[0] == 8
    (snapshot.parent.parent / "rank-001-initial.json").unlink()
    with pytest.raises(ValueError, match="every training rank"):
        _load_snapshot(snapshot, args)


def test_visual_snapshot_requires_final_hash_not_just_an_intermediate_audit(visual_snapshot):
    from examples.qwen3_8_next_vision_sft.evaluate import _load_snapshot

    snapshot, args = visual_snapshot
    final = snapshot.parent.parent / "rank-001-update-000008.json"
    final.rename(final.with_name("rank-001-update-000007.json"))
    with pytest.raises(ValueError, match="no completed update"):
        _load_snapshot(snapshot, args)


@pytest.mark.parametrize("failure", [None, "begin", "load", "end", "exception"])
def test_visual_updates_obey_native_session_lifecycle(failure):
    from examples.qwen3_8_next_vision_sft.evaluate import _LOADER, _update_visual

    calls = []
    tensors = {"model.visual.weight": torch.ones(2)}

    def begin():
        calls.append("begin")
        return failure != "begin", "begin failed"

    def load(weights, *, load_format, flush_cache):
        calls.append("load")
        assert weights == list(tensors.items()) and load_format == _LOADER and flush_cache
        if failure == "exception":
            raise RuntimeError("load exception")
        return failure != "load", "load failed"

    def end():
        calls.append("end")
        return failure != "end", "end failed"

    engine = SimpleNamespace(begin_weight_update=begin, update_weights_from_tensor=load, end_weight_update=end)
    if failure is None:
        _update_visual(engine, tensors)
    else:
        with pytest.raises(
            RuntimeError if failure == "exception" else ValueError,
            match=failure.replace("exception", "load exception"),
        ):
            _update_visual(engine, tensors)
    assert calls == (["begin"] if failure == "begin" else ["begin", "load", "end"])


def test_visual_update_discards_native_cached_image_embeddings(monkeypatch):
    from examples.qwen3_8_next_vision_sft.evaluate import _clear_visual_embedding_cache
    from sglang.srt.managers import mm_schedule
    from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache

    cache = MultiModalStaticCache(1024)
    assert cache.set(17, EmbeddingResult(embedding=torch.ones(2)))
    assert cache.get_single(17) is not None
    monkeypatch.setattr(mm_schedule, "embedding_cache", cache)
    _clear_visual_embedding_cache()
    assert cache.get_single(17) is None
    assert cache.current_size == 0
    # Reusing the same image identity must accept features from the new visual state.
    assert cache.set(17, EmbeddingResult(embedding=torch.full((2,), 2.0)))
    assert torch.equal(cache.get_single(17).embedding, torch.full((2,), 2.0))


@pytest.mark.parametrize("thinking", [False, True])
def test_empty_thinking_prefill_preserves_native_system_reasoning_options(thinking):
    from examples.qwen3_8_next_vision_sft.evaluate import _prepare

    captured = {}
    native = "<think>\n" if thinking else "<think>\n\n</think>\n\n"

    def template(messages, **kwargs):
        captured.update(messages=messages, options=kwargs)
        return "<|im_start|>system\nnative reasoning instructions<|im_end|>\n<|im_start|>assistant\n" + native

    def encode(text, **kwargs):
        captured["text"] = text
        return [1, 2]

    row = {"messages": [{"content": "How many?"}]}
    result = _prepare(
        SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=template, encode=encode)),
        {},
        row,
        row,
        control="no_image",
        enable_thinking=thinking,
        reasoning_effort="xhigh",
        empty_thinking_prefill=True,
    )
    assert result[3] == "<think>\n\n</think>\n\n"
    assert captured["text"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert captured["text"].count("</think>") == 1
    assert captured["options"]["enable_thinking"] is thinking
    assert captured["options"]["reasoning_effort"] == "xhigh"
    assert captured["messages"] == [{"role": "user", "content": "How many?"}]


@pytest.mark.parametrize(
    "thinking,generated,correct,invalid",
    [(True, "Counted two.\n</think>\n\n2", 1, 0), (True, "Still counting: 2", 0, 1), (False, "2", 1, 0)],
)
def test_scoring_preserves_prompt_thinking_prefix(tmp_path, thinking, generated, correct, invalid):
    from examples.qwen3_8_next_vision_sft.evaluate import _evaluate_split

    prefill = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **k: "<|im_start|>user\nCount.<|im_end|>\n<|im_start|>assistant\n" + prefill,
        encode=lambda *a, **k: [1, 2],
    )

    def generate(**kwargs):
        assert kwargs["input_ids"] == [[1, 2]] and "image_data" not in kwargs
        assert kwargs["sampling_params"]["max_new_tokens"] == 4096
        return [{"text": generated, "meta_info": {"prompt_tokens": 2}}]

    path = tmp_path / "predictions.jsonl"
    metrics = _evaluate_split(
        SimpleNamespace(generate=generate),
        SimpleNamespace(tokenizer=tokenizer),
        {},
        [{"id": "scene", "messages": [{"content": "Count."}], "answer": "2"}],
        path,
        args=SimpleNamespace(
            controls=["no_image"],
            batch_size=1,
            max_new_tokens=4096,
            context_length=8192,
            enable_thinking=thinking,
            reasoning_effort="xhigh",
            final_answer_format="dataset",
        ),
        split="development",
        update=0,
    )
    assert metrics["no_image"]["correct"] == correct
    assert metrics["no_image"]["invalid"] == invalid
    record = json.loads(path.read_text())
    assert record["generated_text"] == generated
    assert record["assistant_prefill"] == prefill
    assert record["prediction"] == prefill + generated


@pytest.mark.parametrize("answer_format", ["dataset", "integer-only"])
def test_native_reasoning_and_final_format_preserve_question_and_thinking(answer_format):
    from examples.qwen3_8_next_vision_sft.evaluate import _prepare

    captured = {}

    def template(messages, **kwargs):
        captured.update(messages=messages, options=kwargs)
        return "<|im_start|>assistant\n<think>\n"

    tokenizer = SimpleNamespace(apply_chat_template=template, encode=lambda *a, **k: [1, 2])
    row = {"messages": [{"content": "Count the black squares."}]}
    result = _prepare(
        SimpleNamespace(tokenizer=tokenizer),
        {},
        row,
        row,
        control="no_image",
        enable_thinking=True,
        reasoning_effort="low",
        final_answer_format=answer_format,
    )
    assert result == ([1, 2], None, None, "<think>\n")
    assert captured["options"]["reasoning_effort"] == "low"
    assert captured["options"]["enable_thinking"] is True
    assert captured["messages"][-1] == {"role": "user", "content": "Count the black squares."}
    if answer_format == "integer-only":
        assert captured["messages"][0]["role"] == "system"
        assert "exactly one integer" in captured["messages"][0]["content"]
    else:
        assert len(captured["messages"]) == 1


@pytest.fixture
def checkpoint_workers(evidence, tmp_path):
    data, evaluation, _, _ = evidence
    original = json.loads((evaluation / "evaluation.json").read_text())
    original.update(
        model_files_sha256={"config.json": "model"},
        evaluator_sha256="evaluator",
        packages={"torch": "pinned"},
        visual_snapshots=[],
        seconds=2,
    )
    original["args"].update(splits=["validation"], max_new_tokens=4096, data=str(data))
    workers = []
    for index, updates in enumerate(([0], [8, 16], [24])):
        path = tmp_path / f"worker-{index}"
        path.mkdir()
        manifest = {**original, "results": [item for item in original["results"] if item["update"] in updates]}
        manifest["args"] = {**original["args"], "output": str(path), "skip_baseline": index != 0}
        (path / "evaluation.json").write_text(json.dumps(manifest))
        for update in updates:
            name = f"update-{update:06d}-validation.jsonl"
            shutil.copyfile(evaluation / name, path / name)
        workers.append(path)
    return data, workers, tmp_path / "merged"


def test_parallel_checkpoint_merge_preserves_raw_predictions_and_real_dip(checkpoint_workers):
    data, workers, output = checkpoint_workers
    result = merge_evaluations(list(reversed(workers)), output, data=data)
    assert result["complete"] and len(result["components_sha256"]) == 3
    assert [item["update"] for item in result["results"]] == [0, 8, 16, 24]
    assert (output / "update-000008-validation.jsonl").read_bytes() == (
        workers[1] / "update-000008-validation.jsonl"
    ).read_bytes()
    assert not collect_points(output, data, split="validation")["all_adjacent_points_nondecreasing"]


@pytest.mark.parametrize("change", ["tokens", "evaluator", "packages", "incomplete", "duplicate"])
def test_parallel_checkpoint_merge_rejects_different_or_repeated_measurements(checkpoint_workers, change):
    data, workers, output = checkpoint_workers
    path = workers[-1] / "evaluation.json"
    manifest = json.loads(path.read_text())
    if change == "tokens":
        manifest["args"]["max_new_tokens"] = 128
    elif change == "evaluator":
        manifest["evaluator_sha256"] = "different"
    elif change == "packages":
        manifest["packages"]["torch"] = "different"
    elif change == "incomplete":
        manifest["complete"] = False
    else:
        workers.append(workers[0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        merge_evaluations(workers, output, data=data)
    assert not output.exists()


def test_parallel_checkpoint_merge_rescores_text_before_accepting_result(checkpoint_workers):
    data, workers, output = checkpoint_workers
    path = workers[-1] / "update-000024-validation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["prediction"] = "invalid"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="aggregate disagrees"):
        merge_evaluations(workers, output, data=data)
    assert not json.loads((output / "evaluation.json").read_text())["complete"]
