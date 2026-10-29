"""Re-render the pinned experiment scenes; do not resample their targets or labels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

from examples.qwen3_8_next_vision_sft.score import read_rows
from PIL import ImageDraw

_ASSETS = Path(__file__).parent / "assets"


def _load_renderer(checkout: Path, provenance: dict):
    source = checkout / "src/sphinx/tasks/geometric/geometric_stack_count.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["generator_module_sha256"]:
        raise ValueError("Use the exact SPHINX source revision recorded with the experiment")
    sys.path.insert(0, str(checkout / "src"))
    module = importlib.import_module("sphinx.tasks.geometric.geometric_stack_count")
    if Path(module.__file__).resolve() != source.resolve():
        raise ValueError("The documentation Sphinx package shadowed the dataset generator")
    return module


def _render(module, metadata: dict):
    width, height = metadata["dims"]
    image = module.graph_paper_rgb(width, height).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = dict(module.COLOR_POOL)
    for sheet in metadata["stack_sheets"]:
        color = module._hex_to_rgb(colors[sheet["color"]])
        if sheet["kind"] == "rectangle":
            x0, y0, x1, y1 = sheet["bbox"]
            module.SheetRect(sheet["color"], color, x0, y0, x1 - x0, y1 - y0).draw(draw)
        elif sheet["kind"] == "circle":
            module.SheetCircle(sheet["color"], color, sheet["cx"], sheet["cy"], sheet["R"]).draw(draw)
        elif sheet["kind"] == "triangle":
            # Metadata records the actual vertices, not a lossy inferred orientation.
            vertices = [tuple(point) for point in sheet["verts"]]
            draw.polygon(vertices, fill=color, outline=module.BIG_OUTLINE_RGB)
            draw.line(vertices + [vertices[0]], fill=module.BIG_OUTLINE_RGB, width=module.SHEET_OUTLINE_WIDTH)
        else:
            raise ValueError(f"Unknown sheet kind: {sheet['kind']}")
    shapes = [
        module.SmallShape(shape["kind"], *shape["center"], shape["r"], shape["rot_deg"])
        for shape in metadata["shapes"]
    ]
    return module._draw_small_shapes_supersampled(image, shapes).convert("RGB")


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")


def _validate_scenes(rows: list[dict], result: dict) -> None:
    expected = result["dataset"]["role_counts"]
    if dict(Counter(row["dataset_role"] for row in rows)) != expected:
        raise ValueError("Pinned scene coverage differs from the recorded result")
    for key in ("id", "sample_seed", "image_pixel_sha256", "structure_sha256"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate scene identity across roles: {key}")
    for row in rows:
        if (
            row["task"] != result["task"]
            or int(row["answer"]) != int(row["generator_metadata"]["answer"])
            or row["question"] != row["generator_metadata"]["question"] + result["dataset"]["answer_suffix"]
        ):
            raise ValueError(f"Pinned question or label differs: {row['id']}")


def materialize(checkout: Path, output: Path, assets: Path = _ASSETS) -> dict:
    result = json.loads((assets / "scenes.json").read_text())
    rows = []
    for filename, digest in result["provenance"]["scene_files_sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("Unsafe scene archive path")
        compressed = (assets / filename).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != digest:
            raise ValueError(f"Pinned scene archive was modified: {filename}")
        rows.extend(
            json.loads(line) for line in gzip.decompress(compressed).decode("utf-8").splitlines() if line.strip()
        )
    _validate_scenes(rows, result)
    renderer = _load_renderer(checkout, result["provenance"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{role}.jsonl").exists() for role in result["dataset"]["role_counts"]):
        raise FileExistsError("Inspect the existing dataset before rebuilding it")
    groups = {role: [] for role in result["dataset"]["role_counts"]}
    for index, original in enumerate(rows):
        image = _render(renderer, original["generator_metadata"])
        digest = hashlib.sha256(json.dumps(image.size).encode() + image.tobytes()).hexdigest()
        if digest != original["image_pixel_sha256"]:
            raise ValueError(f"Rendered pixels differ: {original['id']}")
        path = output / "images" / original["dataset_role"] / f"{original['id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        groups[original["dataset_role"]].append(
            {
                **{key: value for key, value in original.items() if key != "question"},
                "images": [str(path.resolve())],
                "messages": [{"role": "user", "content": original["question"]}],
            }
        )
        if (index + 1) % 128 == 0:
            print(json.dumps({"rendered": index + 1, "total": len(rows)}), flush=True)
    for role, sources in groups.items():
        _write_rows(output / f"{role}.jsonl", sources)
    training = [
        {
            "id": row["id"],
            "task": row["task"],
            "source_split": "train",
            "images": row["images"],
            "answer": row["answer"],
            "messages": [
                {"role": "user", "content": "<image>" + row["messages"][0]["content"]},
                {"role": "assistant", "content": str(int(row["answer"]))},
            ],
        }
        for row in groups["distillation"]
    ]
    _write_rows(output / "sft_exact.jsonl", training)
    report = {
        "role_counts": {role: len(sources) for role, sources in groups.items()},
        "training_rows": len(read_rows(output / "sft_exact.jsonl")),
        "all_pixels_match": True,
        "cross_role_identity_overlap": False,
        "exact_targets_only": True,
        "oracle_in_model_input": False,
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.jsonl"))
        },
    }
    (output / "materialization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphinx-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=_ASSETS)
    args = parser.parse_args()
    print(json.dumps(materialize(args.sphinx_checkout, args.output, args.assets), indent=2))


if __name__ == "__main__":
    main()
