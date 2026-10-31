"""Offline key-by-key check of a completed Qwen3.8 HF export (``verify_export`` and its ``python -m`` CLI).

The check runs after training, against the base checkpoint, on a small fake checkpoint with the real
Qwen3.8 key layout: passthrough keys and ``--expect-unchanged`` keys must equal the base bytes, and the
export must hold trained weights.
"""

import json
import os
import subprocess
import sys
import zlib
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from miles.backends.megatron_utils.megatron_to_hf import qwen3_8_next  # noqa: F401  (registers the passthrough)
from miles.backends.training_utils.weight_update import hf_passthrough
from miles.backends.training_utils.weight_update.hf_passthrough import (
    ExportIncomplete,
    complete_hf_export,
    hf_export_passthrough,
    main,
    verify_export,
)

LM = "model.language_model"
PLE = f"{LM}.layers.1.ple.ple_embedding"
# base shard -> keys, laid out like the real checkpoint: a PLE-only shard, shards mixing trained keys with
# PLE metadata / MTP / vision keys, and a trained-only shard
BASE_LAYOUT = {
    "model-00001-of-00004.safetensors": [
        f"{LM}.embed_tokens.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.merger.linear.weight",
    ],
    "model-00002-of-00004.safetensors": [
        f"{PLE}.ngram_embedding.shard_0.weight",
        f"{PLE}.ngram_embedding.shard_1.weight",
    ],
    "model-00003-of-00004.safetensors": [
        f"{LM}.layers.1.ple.key_proj.weight",
        f"{PLE}.layer_multipliers",
        f"{PLE}.ngram_heads_offsets",
        f"{PLE}.ngram_heads_vocab_sizes",
        "mtp.fc_hidden.weight",
        "mtp.layers.0.mlp.experts.down_proj",
    ],
    "model-00004-of-00004.safetensors": [
        f"{LM}.layers.0.linear_attn.out_proj.weight",
        f"{LM}.norm.weight",
        "lm_head.weight",
    ],
}
ALL = [k for keys in BASE_LAYOUT.values() for k in keys]
TRAINED = [k for k in ALL if not (k.startswith(("model.visual.", "mtp.")) or ".ple_embedding." in k)]


def _tensor(key: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(zlib.crc32(f"{key}/{seed}".encode()))
    if key.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")):
        return torch.randint(0, 1 << 40, (3,), generator=generator, dtype=torch.int64)
    return torch.randn(4, 6, generator=generator).to(torch.bfloat16)


def _write(directory: Path, layout: dict[str, list[str]], tensors: dict[str, torch.Tensor]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for filename, keys in layout.items():
        save_file({k: tensors[k] for k in keys}, str(directory / filename), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(keys, filename))
    (directory / hf_passthrough.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    directory = tmp_path / "base"
    _write(directory, BASE_LAYOUT, {k: _tensor(k, 0) for k in ALL})
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    return directory


def _completed_export(tmp_path: Path, base: Path, *, seed: int = 1, override: dict | None = None) -> Path:
    """A raw export of the trained keys, completed from the base as a text-only run without MTP training."""
    export = tmp_path / "export"
    tensors = {k: _tensor(k, seed) for k in TRAINED} | (override or {})
    _write(export, {"model-00001.safetensors": sorted(TRAINED)}, tensors)
    complete_hf_export(export, base, hf_export_passthrough("qwen4_exp", Namespace(enable_mtp_training=False)))
    return export


class TestVerifyExport:
    def test_a_completed_export_verifies_and_reports_changes_by_group(self, tmp_path, base) -> None:
        """Trained keys changed, the PLE table, MTP and vision byte-identical to the base."""
        export = _completed_export(tmp_path, base)
        report = verify_export(export, base)
        assert report["keys"] == len(ALL)
        assert report["passthrough"] == len(ALL) - len(TRAINED)
        assert report["changed"] == len(TRAINED)
        assert report["changed_by_group"] == {
            "ple_table": [0, 5],
            "mtp": [0, 2],
            "visual": [0, 2],
            "other": [len(TRAINED), len(TRAINED)],
        }
        assert report["unchanged_trained_keys"] == []
        assert main([str(export), str(base)]) == 0

    def test_a_key_expected_unchanged_that_changed_fails(self, tmp_path, base) -> None:
        """--expect-unchanged pins frozen-but-exported params (e.g. the indexer) to the base bytes."""
        export = _completed_export(tmp_path, base)
        with pytest.raises(ExportIncomplete, match="must equal the base changed"):
            verify_export(export, base, expect_unchanged=[r"\.norm\.weight$"])
        assert main([str(export), str(base), "--expect-unchanged", r"\.norm\.weight$"]) == 1

    def test_an_export_of_untrained_weights_fails(self, tmp_path, base) -> None:
        """Exporting the base back unchanged means the trained weights never reached the export."""
        export = _completed_export(tmp_path, base, seed=0)
        with pytest.raises(ExportIncomplete, match="untrained weights"):
            verify_export(export, base)

    def test_unchanged_trained_keys_are_listed_not_failed(self, tmp_path, base) -> None:
        """A few steps in bf16 may leave some trained tensor bitwise equal; that is reported for review."""
        key = f"{LM}.norm.weight"
        export = _completed_export(tmp_path, base, override={key: _tensor(key, 0)})
        assert verify_export(export, base)["unchanged_trained_keys"] == [key]

    def test_an_export_that_lost_a_key_fails(self, tmp_path, base) -> None:
        export = _completed_export(tmp_path, base)
        index = json.loads((export / hf_passthrough.INDEX).read_text())
        del index["weight_map"][f"{LM}.norm.weight"]
        (export / hf_passthrough.INDEX).write_text(json.dumps(index))
        with pytest.raises(ExportIncomplete, match="keys differ"):
            verify_export(export, base)
        assert main([str(export), str(base)]) == 1


def test_the_module_cli_sees_the_qwen38_groups(tmp_path, base) -> None:
    """``python -m ...hf_passthrough`` runs the module as __main__; the groups must still be the registered ones."""
    export = _completed_export(tmp_path, base)
    repo = Path(hf_passthrough.__file__).resolve().parents[4]
    env = os.environ | {"PYTHONPATH": str(repo), "CUDA_VISIBLE_DEVICES": ""}
    command = [sys.executable, "-B", "-m", "miles.backends.training_utils.weight_update.hf_passthrough"]
    result = subprocess.run(command + [str(export), str(base)], capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout[result.stdout.index("{") :])
    assert report["changed_by_group"]["ple_table"] == [0, 5]
    assert report["changed_by_group"]["other"] == [len(TRAINED), len(TRAINED)]

    failing = subprocess.run(
        command + [str(export), str(base), "--expect-unchanged", r"\.norm\.weight$"],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert failing.returncode == 1
    assert "FAILED: 1 keys that must equal the base changed" in failing.stdout
