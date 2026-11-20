import argparse
import json
import pickle
import runpy
import sys
from pathlib import Path

import pytest
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace


def _run_converter(monkeypatch: pytest.MonkeyPatch, input_dir: Path, output_dir: Path, *options: str):
    argv = ["convert_torch_dist_to_hf.py", "--input-dir", str(input_dir), "--output-dir", str(output_dir), *options]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(pickle, "Unpickler", pickle.Unpickler)
    return runpy.run_module("tools.convert_torch_dist_to_hf", run_name="__main__")


class TestSerialCheckpointExport:
    @pytest.mark.parametrize("namespace_type", [argparse.Namespace, MegatronArgsNamespace])
    @pytest.mark.parametrize("override_origin", [False, True])
    def test_frozen_checkpoint_uses_selected_hf_assets_and_refreezes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        override_origin: bool,
        namespace_type: type[argparse.Namespace],
    ) -> None:
        """Checkpoint export overrides HF assets temporarily and restores the mutation boundary."""
        input_dir = tmp_path / "checkpoint"
        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        (hf_dir / "config.json").write_text(json.dumps({"intermediate_size": 2}))
        tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        dist_cp.save({"decoder.layers.0.mlp.experts.linear_fc1.weight0": tensor}, checkpoint_id=str(input_dir))
        backend = namespace_type(
            num_layers=1,
            num_experts=1,
            vocab_size=None,
            hf_checkpoint=str(tmp_path / "stale") if override_origin else str(hf_dir),
        )
        torch.save({"args": backend}, input_dir / "common.pt")
        output_dir = tmp_path / "output"
        options = ["--model-name", "inkling"]
        if override_origin:
            options += ["--origin-hf-dir", str(hf_dir)]
        namespace = _run_converter(monkeypatch, input_dir, output_dir, *options)

        assert vars(namespace["megatron_args"]) == vars(backend) | {"hf_checkpoint": str(hf_dir)}
        if namespace_type is MegatronArgsNamespace:
            with pytest.raises(AttributeError, match="immutable"):
                namespace["megatron_args"].hf_checkpoint = "forbidden"
        [shard] = output_dir.glob("*.safetensors")
        tensors = safetensors.torch.load_file(shard)
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.gate_proj.weight"], tensor[:2])
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.up_proj.weight"], tensor[2:])

    @pytest.mark.parametrize("model_name", [None, "qwen3_8_next"], ids=["from-config", "explicit"])
    def test_qwen38_head_contraction_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str | None
    ) -> None:
        input_dir = tmp_path / "checkpoint"
        weight = torch.tensor([0.25, -1.5, 2.0])
        dist_cp.save({"decoder.hc_head_contraction.hc_norm_weight": weight}, checkpoint_id=str(input_dir))
        torch.save({"args": argparse.Namespace(vocab_size=None)}, input_dir / "common.pt")

        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        # Keep AutoConfig loading local even on Transformers versions without the native Qwen4ExpConfig.
        (hf_dir / "configuration_qwen4_exp.py").write_text(
            "from transformers import PretrainedConfig\n\n"
            "class Qwen4ExpConfig(PretrainedConfig):\n"
            "    model_type = 'qwen4_exp'\n"
        )
        config = {"model_type": "qwen4_exp", "auto_map": {"AutoConfig": "configuration_qwen4_exp.Qwen4ExpConfig"}}
        (hf_dir / "config.json").write_text(json.dumps(config))
        output_dir = tmp_path / "output"
        options = ["--origin-hf-dir", str(hf_dir)]
        if model_name is not None:
            options += ["--model-name", model_name]
        _run_converter(monkeypatch, input_dir, output_dir, *options)

        hf_key = "model.language_model.hyper_connection_mixer.hc_norm.weight"
        weight_map = json.loads((output_dir / "model.safetensors.index.json").read_text())["weight_map"]
        assert set(weight_map) == {hf_key}
        tensors = safetensors.torch.load_file(output_dir / weight_map[hf_key])
        torch.testing.assert_close(tensors[hf_key], weight, rtol=0, atol=0)
