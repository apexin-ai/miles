"""Raw HF exports must store MoE experts in the base checkpoint's fused layout (CRV P1-INT-4).

Qwen3.8 base checkpoints keep each MoE layer as ``mlp.experts.gate_up_proj`` [E, 2I, H] and
``mlp.experts.down_proj`` [E, H, I]; the raw converter emits per-expert gate/up/down tensors.
The end-to-end test runs the real converter, SnapshotPublisher, save_hf_model and the
key-set completion check; only parameter enumeration and the weight gather are replaced.
"""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils import hf_export, ple_provenance
from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf
from miles.backends.training_utils.weight_update import hf_passthrough
from miles.backends.training_utils.weight_update.hf_expert_packing import ExpertPacker, ExpertPackingError
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

NUM_EXPERTS, HIDDEN, INTER = 3, 4, 2
PREFIX = "model.language_model.layers.0.mlp.experts"
EMBED = "model.language_model.embed_tokens.weight"


def _megatron_experts(seed: int = 0) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Distinct values per expert: linear_fc1 [2I, H] (gate rows then up rows), linear_fc2 [H, I]."""
    gen = torch.Generator().manual_seed(seed)
    return {
        e: (torch.randn(2 * INTER, HIDDEN, generator=gen), torch.randn(HIDDEN, INTER, generator=gen))
        for e in range(NUM_EXPERTS)
    }


def _base_layout(experts) -> dict[str, torch.Tensor]:
    """What mbridge reads back: gate_up_proj[e] == linear_fc1 of expert e, down_proj[e] == linear_fc2."""
    return {
        f"{PREFIX}.gate_up_proj": torch.stack([experts[e][0] for e in range(NUM_EXPERTS)]),
        f"{PREFIX}.down_proj": torch.stack([experts[e][1] for e in range(NUM_EXPERTS)]),
    }


def _write_base(directory: Path, tensors: dict[str, torch.Tensor]) -> Path:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    save_file(tensors, str(directory / "model-00001.safetensors"))
    (directory / hf_passthrough.INDEX).write_text(
        json.dumps({"metadata": {}, "weight_map": {name: "model-00001.safetensors" for name in tensors}})
    )
    return directory


def _converted(experts) -> list[tuple[str, torch.Tensor]]:
    backend = SimpleNamespace(kv_channels=1, hidden_size=HIDDEN, num_attention_heads=1, num_query_groups=1)
    args = SimpleNamespace(backend=backend)
    out = []
    for e, (fc1, fc2) in experts.items():
        out += convert_qwen3_8_next_to_hf(
            args, f"module.module.decoder.layers.0.mlp.experts.linear_fc1.weight{e}", fc1
        )
        out += convert_qwen3_8_next_to_hf(
            args, f"module.module.decoder.layers.0.mlp.experts.linear_fc2.weight{e}", fc2
        )
    return out


def _read(directory: Path) -> dict[str, torch.Tensor]:
    weight_map = json.loads((directory / hf_passthrough.INDEX).read_text())["weight_map"]
    out = {}
    for name, file in weight_map.items():
        with safe_open(str(directory / file), framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out


class TestExpertPacker:
    def test_packs_any_arrival_order_into_the_base_layout(self, tmp_path) -> None:
        experts = _megatron_experts()
        base = _write_base(tmp_path / "base", _base_layout(experts))
        converted = _converted(experts)
        random.Random(0).shuffle(converted)
        packer = ExpertPacker.for_base(base)

        emitted = [item for name, tensor in converted for item in packer.feed(name, tensor)]
        packer.finish()

        assert [name for name, _ in emitted] == [f"{PREFIX}.gate_up_proj", f"{PREFIX}.down_proj"]
        for name, tensor in emitted:
            assert torch.equal(tensor, _base_layout(experts)[name])

    def test_other_tensors_pass_through_unchanged(self, tmp_path) -> None:
        base = _write_base(tmp_path / "base", _base_layout(_megatron_experts()))
        tensor = torch.ones(2)
        assert ExpertPacker.for_base(base).feed(EMBED, tensor) == [(EMBED, tensor)]

    def test_a_base_with_split_experts_keeps_split_keys(self, tmp_path) -> None:
        split = dict(_converted(_megatron_experts()))
        base = _write_base(tmp_path / "base", split)
        packer = ExpertPacker.for_base(base)
        name, tensor = next(iter(split.items()))
        assert packer.feed(name, tensor) == [(name, tensor)]
        packer.finish()

    def test_a_missing_expert_fails_at_finish(self, tmp_path) -> None:
        experts = _megatron_experts()
        base = _write_base(tmp_path / "base", _base_layout(experts))
        packer = ExpertPacker.for_base(base)
        for name, tensor in _converted(experts):
            if ".experts.2." not in name:
                assert packer.feed(name, tensor) == []
        with pytest.raises(ExpertPackingError, match="left incomplete"):
            packer.finish()

    def test_a_duplicate_or_out_of_range_expert_is_rejected(self, tmp_path) -> None:
        base = _write_base(tmp_path / "base", _base_layout(_megatron_experts()))
        packer = ExpertPacker.for_base(base)
        packer.feed(f"{PREFIX}.0.down_proj.weight", torch.zeros(HIDDEN, INTER))
        with pytest.raises(ExpertPackingError, match="twice"):
            packer.feed(f"{PREFIX}.0.down_proj.weight", torch.zeros(HIDDEN, INTER))
        with pytest.raises(ExpertPackingError, match="outside"):
            packer.feed(f"{PREFIX}.{NUM_EXPERTS}.down_proj.weight", torch.zeros(HIDDEN, INTER))

    def test_a_shape_that_differs_from_the_base_is_rejected(self, tmp_path) -> None:
        experts = _megatron_experts()
        base = _write_base(tmp_path / "base", _base_layout(experts))
        packer = ExpertPacker.for_base(base)
        wide = {e: (torch.cat([fc1, fc1], dim=1), fc2) for e, (fc1, fc2) in experts.items()}
        with pytest.raises(ExpertPackingError, match="base shape"):
            for name, tensor in _converted(wide):
                packer.feed(name, tensor)


class TestRawExportEndToEnd:
    @pytest.fixture(autouse=True)
    def _single_process(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(ple_provenance, "_run_manifest", None)
        monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *a, **kw: [])
        monkeypatch.setattr(hf_export, "is_lora_model", lambda model: False)
        zero = SimpleNamespace(rank=0)
        monkeypatch.setattr(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=zero, tp=zero))

    def _export(self, tmp_path, buckets):
        experts = _megatron_experts()
        base = _write_base(tmp_path / "base", _base_layout(experts) | {EMBED: torch.zeros(2)})
        args = SimpleNamespace(
            megatron_to_hf_mode="raw", enable_mtp_training=False, hf_checkpoint=str(base), save_hf=None
        )

        class Iterator:
            placement = SimpleNamespace(is_full_gather=True)

            def iter_hf_weights(self, weights):
                yield from buckets(_converted(experts) + [(EMBED, torch.ones(2))])

        target = tmp_path / "export"
        hf_export.save_hf_model(args, 0, [], publisher=SnapshotPublisher(Iterator()), path=target, raise_on_error=True)
        return experts, target

    @pytest.mark.parametrize("bucket_size", [1, 4, 100])
    def test_split_expert_converter_output_is_published_in_the_fused_base_layout(self, tmp_path, bucket_size) -> None:
        def buckets(items):
            for i in range(0, len(items), bucket_size):
                yield items[i : i + bucket_size]

        experts, target = self._export(tmp_path, buckets)

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        exported = _read(target)
        assert set(exported) == {f"{PREFIX}.gate_up_proj", f"{PREFIX}.down_proj", EMBED}
        for name, tensor in _base_layout(experts).items():
            assert torch.equal(exported[name], tensor)
        # gate rows then up rows, as the converter split them
        for e, (fc1, _) in experts.items():
            gate, up = fc1.chunk(2, dim=0)
            assert torch.equal(exported[f"{PREFIX}.gate_up_proj"][e][:INTER], gate)
            assert torch.equal(exported[f"{PREFIX}.gate_up_proj"][e][INTER:], up)

    def test_a_missing_expert_still_fails_the_export(self, tmp_path) -> None:
        def buckets(items):
            yield [(name, tensor) for name, tensor in items if ".experts.1." not in name]

        with pytest.raises(ExpertPackingError, match="left incomplete"):
            self._export(tmp_path, buckets)
        assert not (tmp_path / "export").exists()
