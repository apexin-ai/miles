"""Evaluate fixed Qwen visual checkpoints with genuine images and negative controls.

Runs on an allocated GPU node, not the control host. Each checkpoint sees the same
ordered dev/test rows, native processor/template, greedy decoding and scorer.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from examples.qwen3_8_next_vision_sft.score import answer_of, read_rows
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from miles_plugins.models.qwen3_8_next.vision_inputs import image_rope_positions, normalize_image_messages
from miles_plugins.models.qwen3_8_next.vision_sft import _processor

_LOADER = "examples.qwen3_8_next_vision_sft.evaluate.load_visual_and_verify"
_INTEGER_FINAL_INSTRUCTION = (
    "Keep all reasoning in the thinking block. Your final answer after thinking must consist of "
    "exactly one integer and nothing else. Do not include explanations, headings, lists, units, "
    "or Markdown in the final answer."
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_name(name):
    return name.replace("model.visual.", "visual.").replace("attn.qkv.", "attn.qkv_proj.")


def load_visual_and_verify(model, named_tensors):
    """SGLang's opt-in weight-loader hook; verify every native TP shard after load."""
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    tensors = dict(named_tensors)
    parameters = {"visual." + name: param for name, param in model.visual.named_parameters()}
    mapped = {_native_name(name): tensor for name, tensor in tensors.items()}
    if any(not name.startswith("model.visual.") for name in tensors) or set(mapped) != set(parameters):
        raise ValueError("A visual update must cover precisely the entire native visual tower")
    loaded = model.load_weights(list(tensors.items()))
    if loaded != set(parameters):
        raise ValueError("The inference engine did not load every visual tensor")
    with torch.no_grad():
        for name, parameter in parameters.items():
            expected = torch.nn.Parameter(torch.empty_like(parameter), requires_grad=False)
            for key, value in vars(parameter).items():
                setattr(expected, key, value)
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(expected, mapped[name])
            if not torch.equal(parameter, expected) or not torch.isfinite(parameter).all():
                raise ValueError(f"Native visual shard failed exact readback: {name}")
    print(json.dumps({"event": "visual-load-verified", "tensors": len(parameters)}), flush=True)


def _update_visual(engine, tensors):
    success, message = engine.begin_weight_update()
    if not success:
        raise ValueError(f"Cannot begin inference weight update: {message}")
    try:
        success, message = engine.update_weights_from_tensor(
            list(tensors.items()), load_format=_LOADER, flush_cache=True
        )
        if not success:
            raise ValueError(f"Inference visual update failed: {message}")
    finally:
        success, message = engine.end_weight_update()
        if not success:
            raise ValueError(f"Cannot finalize inference weight update: {message}")


def _source_visual(model):
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {name: shard for name, shard in mapping.items() if name.startswith("model.visual.")}
    tensors = {}
    for shard in sorted(set(selected.values())):
        with safe_open(model / shard, framework="pt", device="cpu") as source:
            for name, filename in selected.items():
                if filename == shard:
                    tensors[name] = source.get_tensor(name)
    if not tensors:
        raise ValueError("The baseline has no native visual weights")
    return tensors


def _prepare(
    processor,
    config,
    row,
    image_row,
    *,
    control,
    enable_thinking,
    reasoning_effort="xhigh",
    final_answer_format="dataset",
):
    question = row["messages"][0]["content"].replace("<image>", "")
    images, pixel_hash = [], None
    if control != "no_image":
        if len(image_row["images"]) != 1:
            raise ValueError("The pinned geometric protocol requires exactly one image per scene")
        with Image.open(image_row["images"][0]) as source:
            image = source.convert("RGB")
        pixel_hash = hashlib.sha256(json.dumps(image.size).encode() + image.tobytes()).hexdigest()
        if pixel_hash != image_row["image_pixel_sha256"]:
            raise ValueError("Evaluation image pixels differ from the pinned scene")
        images.append(image)
    messages = normalize_image_messages(
        [{"role": "user", "content": ("<image>" * len(images)) + question}], len(images)
    )
    if final_answer_format == "integer-only":
        messages.insert(0, {"role": "system", "content": _INTEGER_FINAL_INSTRUCTION})
    elif final_answer_format != "dataset":
        raise ValueError("Unknown final-answer format")
    text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    _, header, prefill = text.rpartition("<|im_start|>assistant\n")
    if not header or prefill not in ("<think>\n", "<think>\n\n</think>\n\n"):
        raise ValueError("Unexpected native Qwen assistant generation prefix")
    if not images:
        ids = processor.tokenizer.encode(text, add_special_tokens=False)
        return ids, None, pixel_hash, prefill
    encoded = processor(text=[text], images=images, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"][0].tolist()
    positions = image_rope_positions(
        ids,
        encoded["image_grid_thw"],
        image_token_id=config["image_token_id"],
        vision_start_token_id=config["vision_start_token_id"],
        vision_end_token_id=config["vision_end_token_id"],
        merge_size=config["vision_config"]["spatial_merge_size"],
    )
    payload = {
        "format": "processor_output",
        "modality": "IMAGE",
        "pixel_values": encoded["pixel_values"],
        "image_grid_thw": encoded["image_grid_thw"],
        "mrope_positions": positions.T.contiguous(),
        "mrope_position_delta": torch.tensor([[int(positions.max()) + 1 - len(ids)]]),
    }
    return ids, [payload], pixel_hash, prefill


def _evaluate_split(engine, processor, config, rows, output, *, args, split, update):
    totals = {}
    with output.open("x") as stream:
        for control in args.controls:
            correct = invalid = 0
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                image_rows = [rows[(start + i + 1) % len(rows)] for i in range(len(batch))]
                prepared = [
                    _prepare(
                        processor,
                        config,
                        row,
                        other if control == "swapped_image" else row,
                        control=control,
                        enable_thinking=args.enable_thinking,
                        reasoning_effort=args.reasoning_effort,
                        final_answer_format=args.final_answer_format,
                    )
                    for row, other in zip(batch, image_rows, strict=True)
                ]
                ids = [item[0] for item in prepared]
                if max(map(len, ids)) + args.max_new_tokens > args.context_length:
                    raise ValueError("Expanded image prompt plus generation exceeds the fixed context budget")
                kwargs = {} if control == "no_image" else {"image_data": [item[1] for item in prepared]}
                outputs = engine.generate(
                    input_ids=ids,
                    sampling_params={"temperature": 0, "top_p": 1, "max_new_tokens": args.max_new_tokens},
                    **kwargs,
                )
                if len(outputs) != len(batch):
                    raise ValueError("Incomplete evaluation batch")
                for row, other, encoded, result in zip(batch, image_rows, prepared, outputs, strict=True):
                    meta = result["meta_info"]
                    if meta["prompt_tokens"] != len(encoded[0]):
                        raise ValueError("The inference engine changed the expanded image prompt length")
                    # The native template opens thinking inside the prompt, outside generated text.
                    prediction = encoded[3] + result["text"]
                    extracted = answer_of(prediction)
                    correct += extracted == int(row["answer"])
                    invalid += extracted is None
                    record = {
                        "id": row["id"],
                        "dataset_role": split,
                        "control": control,
                        "update": update,
                        "question": row["messages"][0]["content"],
                        "answer": row["answer"],
                        "image_id": (
                            None if control == "no_image" else (other if control == "swapped_image" else row)["id"]
                        ),
                        "input_ids_sha256": hashlib.sha256(json.dumps(encoded[0]).encode()).hexdigest(),
                        "image_pixel_sha256": encoded[2],
                        "prompt_tokens": len(encoded[0]),
                        "prediction": prediction,
                        "generated_text": result["text"],
                        "assistant_prefill": encoded[3],
                        "extracted_answer": extracted,
                        "finish_reason": meta.get("finish_reason"),
                        "completion_tokens": meta.get("completion_tokens"),
                    }
                    stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                stream.flush()
                print(
                    json.dumps({"split": split, "control": control, "update": update, "done": start + len(batch)}),
                    flush=True,
                )
            totals[control] = {"n": len(rows), "correct": correct, "invalid": invalid, "accuracy": correct / len(rows)}
    return totals


def _load_snapshot(path, args):
    receipt = json.loads((path / "visual.json").read_text())
    if not receipt.get("complete") or not receipt.get("frozen_language"):
        raise ValueError("A visual-only evaluation update requires a complete frozen-language snapshot")
    identity = {name: _sha(args.model / name) for name in ("config.json", "model.safetensors.index.json")}
    if receipt.get("source_identity") != identity:
        raise ValueError("The snapshot was trained from a different base model")
    audit = path.parent.parent
    initial = sorted(audit.glob("rank-*-initial.json"))
    if not initial:
        raise ValueError("A visual-only evaluation update requires frozen-language rank audits")
    states = [json.loads(path.read_text()) for path in initial]
    world = states[0]["world_size"]
    if {state["rank"] for state in states} != set(range(world)) or len(states) != world:
        raise ValueError("Frozen-language audits do not cover every training rank exactly once")
    for before in states:
        if before["world_size"] != world or before["source_identity"] != identity:
            raise ValueError("Training rank audits disagree on the source or world size")
        if receipt["update"] > before["num_updates"]:
            raise ValueError("The visual snapshot update exceeds the audited training length")
        completed = audit / f"rank-{before['rank']:03d}-update-{before['num_updates']:06d}.json"
        if not completed.is_file():
            raise ValueError("A training rank has no completed update audit")
        after = json.loads(completed.read_text())
        if after["rank"] != before["rank"] or after["update"] != before["num_updates"]:
            raise ValueError("A final training audit has the wrong rank or update")
        if not before["language_sha256"] or before["language_sha256"] != after["final_language_sha256"]:
            raise ValueError("Training has not closed all frozen-language hashes")
    target = path / "visual.safetensors"
    if receipt["file_sha256"] != _sha(target):
        raise ValueError("Visual snapshot file hash differs from its receipt")
    return receipt["update"], load_file(str(target)), receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["development", "validation"])
    parser.add_argument(
        "--controls",
        nargs="+",
        choices=["image", "no_image", "swapped_image"],
        default=["image", "no_image", "swapped_image"],
    )
    parser.add_argument("--visual-snapshots", nargs="*", type=Path, default=[])
    parser.add_argument("--skip-baseline", action="store_true", help="Evaluate only supplied snapshots on this worker")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=["xhigh", "medium", "low"], default="xhigh")
    parser.add_argument("--final-answer-format", choices=["dataset", "integer-only"], default="dataset")
    parser.add_argument("--enable-cuda-graph", action="store_true", help="Opt in to native decode graph capture")
    parser.add_argument("--mem-fraction-static", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=0, help="Bring-up only; not a held-out quality result")
    args = parser.parse_args()
    if min(args.tp_size, args.batch_size, args.max_new_tokens) < 1 or args.limit < 0:
        parser.error("Invalid device, batch or token counts")
    if args.skip_baseline and not args.visual_snapshots:
        parser.error("--skip-baseline requires at least one visual snapshot")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    processor, config = _processor(str(args.model))
    rows = {split: read_rows(args.data / f"{split}.jsonl") for split in args.splits}
    if args.limit:
        rows = {split: items[: args.limit] for split, items in rows.items()}
    if any(not items for items in rows.values()):
        raise ValueError("Cannot evaluate an empty split")
    manifest = {
        "complete": False,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "visual_snapshots"
        },
        "visual_snapshots": list(map(str, args.visual_snapshots)),
        "model_files_sha256": {
            name: _sha(args.model / name)
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "chat_template.jinja",
                *sorted(path.name for path in args.model.glob("*processor*json")),
            )
        },
        "data_files_sha256": {split: _sha(args.data / f"{split}.jsonl") for split in args.splits},
        "evaluator_sha256": _sha(__file__),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "sglang-kernel", "flashinfer-python")
        },
        "protocol": "integer_first_final_line_v1",
        "results": [],
    }
    target = args.output / "evaluation.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    import sglang as sgl

    engine = sgl.Engine(
        model_path=str(args.model),
        tp_size=args.tp_size,
        ep_size=args.tp_size,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.batch_size,
        chunked_prefill_size=4096,
        disable_cuda_graph=not args.enable_cuda_graph,
        cuda_graph_max_bs_decode=args.batch_size if args.enable_cuda_graph else None,
        disable_radix_cache=True,
        linear_attn_prefill_backend="flashinfer",
        moe_runner_backend="triton",
        trust_remote_code=True,
        custom_weight_loader=[_LOADER],
        log_level="info",
        decode_log_interval=200,
    )
    try:
        checkpoints = ([] if args.skip_baseline else [(0, None)]) + [(None, path) for path in args.visual_snapshots]
        for update, path in checkpoints:
            if path is None:
                tensors, receipt = _source_visual(args.model), None
            else:
                update, tensors, receipt = _load_snapshot(path, args)
            _update_visual(engine, tensors)
            del tensors
            metrics = {
                split: _evaluate_split(
                    engine,
                    processor,
                    config,
                    items,
                    args.output / f"update-{update:06d}-{split}.jsonl",
                    args=args,
                    split=split,
                    update=update,
                )
                for split, items in rows.items()
            }
            manifest["results"].append({"update": update, "metrics": metrics, "snapshot": receipt})
            target.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        engine.shutdown()
    manifest.update(complete=True, seconds=time.monotonic() - started)
    target.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
