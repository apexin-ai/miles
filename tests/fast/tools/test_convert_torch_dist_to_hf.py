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
        argv = [
            "convert_torch_dist_to_hf.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "inkling",
        ]
        if override_origin:
            argv += ["--origin-hf-dir", str(hf_dir)]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(pickle, "Unpickler", pickle.Unpickler)

        namespace = runpy.run_module("tools.convert_torch_dist_to_hf", run_name="__main__")

        assert vars(namespace["megatron_args"]) == vars(backend) | {"hf_checkpoint": str(hf_dir)}
        if namespace_type is MegatronArgsNamespace:
            with pytest.raises(AttributeError, match="immutable"):
                namespace["megatron_args"].hf_checkpoint = "forbidden"
        [shard] = output_dir.glob("*.safetensors")
        tensors = safetensors.torch.load_file(shard)
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.gate_proj.weight"], tensor[:2])
        torch.testing.assert_close(tensors["model.llm.layers.0.mlp.experts.0.up_proj.weight"], tensor[2:])


class TestLinearAttentionLayoutMarker:
    @pytest.mark.parametrize(
        ("model_name", "hf_name"),
        [
            ("qwen3_5", "model.language_model.layers.0.linear_attn.A_log"),
            ("kimi_k3", "language_model.model.layers.0.self_attn.A_log"),
        ],
    )
    def test_layout_marker_is_skipped_and_weights_are_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_name: str, hf_name: str
    ) -> None:
        """The linear-attention layout marker has no HF tensor: export drops it and still writes every weight."""
        input_dir = tmp_path / "checkpoint"
        layer = "decoder.layers.0.self_attention.linear_attn"
        a_log = torch.arange(4, dtype=torch.float32)
        dist_cp.save(
            {f"{layer}.A_log": a_log, f"{layer}.weight_layout_version": torch.tensor([1], dtype=torch.int32)},
            checkpoint_id=str(input_dir),
        )
        backend = argparse.Namespace(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=4,
            vocab_size=None,
            hf_checkpoint=None,
        )
        torch.save({"args": backend}, input_dir / "common.pt")
        output_dir = tmp_path / "output"
        argv = [
            "convert_torch_dist_to_hf.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--model-name",
            model_name,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(pickle, "Unpickler", pickle.Unpickler)

        runpy.run_module("tools.convert_torch_dist_to_hf", run_name="__main__")

        [shard] = output_dir.glob("*.safetensors")
        tensors = safetensors.torch.load_file(shard)
        assert set(tensors) == {hf_name}
        torch.testing.assert_close(tensors[hf_name], a_log)
