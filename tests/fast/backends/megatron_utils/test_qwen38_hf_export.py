"""Qwen3.8 raw HF export: how the export fixes work together on a qwen4_exp base.

The single fixes have their own tests (test_hf_export_base_dtype, test_hf_export_fused_experts,
test_hf_export_complete). These tests cover what only the combination decides on a qwen4_exp base: the
Qwen3.8 converter's per-expert output, packed into the fused base layout, passes the completion check; a
dtype difference off the A_log allow-list is still refused by it; the Qwen3.8 passthrough registration
(PLE table, MTP and vision tower from the base; an exported untrained MTP head is refused); and the PLE
bytes of a completed export are checked against the run's PLE provenance.
"""

import json
import shutil
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils import hf_export, ple_provenance
from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf
from miles.backends.training_utils.weight_update import hf_passthrough, snapshot_publisher
from miles.backends.training_utils.weight_update.hf_passthrough import hf_export_passthrough
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

INDEX = "model.safetensors.index.json"
A_LOG = "model.language_model.layers.0.linear_attn.A_log"
EMBED = "model.language_model.embed_tokens.weight"
EXPERTS = "model.language_model.layers.0.mlp.experts"
PLE_SHARD = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
PLE_MULTIPLIERS = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
MTP = ["mtp.fc_hidden.weight", "mtp.layers.0.input_layernorm.weight"]
VISUAL = "model.visual.merger.linear.weight"
NUM_EXPERTS, HIDDEN, INTER = 3, 4, 2


class _Iterator:
    placement = SimpleNamespace(is_full_gather=True)

    def __init__(self, buckets) -> None:
        self._buckets = buckets

    def iter_hf_weights(self, weights):
        yield from self._buckets


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot_publisher.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(hf_export.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(ple_provenance, "_run_manifest", None)
    monkeypatch.setattr(hf_export, "named_params_and_buffers", lambda *args, **kwargs: [])
    monkeypatch.setattr(hf_export, "is_lora_model", lambda model: False)
    rank0 = SimpleNamespace(rank=0)
    monkeypatch.setattr(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=rank0, tp=rank0))


def _megatron_experts() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """linear_fc1 [2I, H] (gate rows, then up rows) and linear_fc2 [H, I] of each expert, distinct values."""
    gen = torch.Generator().manual_seed(0)
    return {
        e: (torch.randn(2 * INTER, HIDDEN, generator=gen).bfloat16(), torch.randn(HIDDEN, INTER, generator=gen).bfloat16())
        for e in range(NUM_EXPERTS)
    }


def _fused(experts) -> dict[str, torch.Tensor]:
    return {
        f"{EXPERTS}.gate_up_proj": torch.stack([experts[e][0] for e in range(NUM_EXPERTS)]),
        f"{EXPERTS}.down_proj": torch.stack([experts[e][1] for e in range(NUM_EXPERTS)]),
    }


def _qwen38_converted(experts) -> list[tuple[str, torch.Tensor]]:
    backend = SimpleNamespace(kv_channels=1, hidden_size=HIDDEN, num_attention_heads=1, num_query_groups=1)
    args = SimpleNamespace(backend=backend)
    out = []
    for e, (fc1, fc2) in experts.items():
        layer = "module.module.decoder.layers.0.mlp.experts"
        out += convert_qwen3_8_next_to_hf(args, f"{layer}.linear_fc1.weight{e}", fc1)
        out += convert_qwen3_8_next_to_hf(args, f"{layer}.linear_fc2.weight{e}", fc2)
    return out


def _write_base(directory: Path, extra: dict[str, torch.Tensor] | None = None) -> Path:
    directory.mkdir(exist_ok=True)
    tensors = {A_LOG: torch.zeros(48, dtype=torch.bfloat16), EMBED: torch.zeros(2, dtype=torch.bfloat16)}
    tensors |= extra or {}
    save_file(tensors, str(directory / "model-00001.safetensors"))
    (directory / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": dict.fromkeys(tensors, "model-00001.safetensors")}))
    (directory / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    return directory


def _trained_a_log() -> torch.Tensor:
    # values that bf16 cannot hold exactly, so a missing cast or a wrong rounding shows
    return torch.linspace(-3.0, 1.0, 48, dtype=torch.float32) + 1e-4


def _publisher(args, buckets) -> SnapshotPublisher:
    # what the Megatron actor builds when the run starts
    return SnapshotPublisher(_Iterator(buckets), export_passthrough=hf_export_passthrough("qwen4_exp", args))


def _args(base: Path, *, mtp: bool = False) -> Namespace:
    return Namespace(megatron_to_hf_mode="raw", enable_mtp_training=mtp, hf_checkpoint=str(base), save_hf=None)


def _export(tmp_path: Path, bucket) -> Path:
    args = _args(_write_base(tmp_path / "base"))
    target = tmp_path / "export"
    hf_export.save_hf_model(args, 0, [], publisher=_publisher(args, [bucket]), path=target, raise_on_error=True)
    return target


class TestBaseDtypeWithCompletion:
    def test_the_restored_a_log_passes_the_completion_check(self, tmp_path) -> None:
        a_log = _trained_a_log()
        target = _export(tmp_path, [(A_LOG, a_log), (EMBED, torch.ones(2, dtype=torch.bfloat16))])

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        assert (target / hf_passthrough.RECEIPT).is_file()
        weight_map = json.loads((target / INDEX).read_text())["weight_map"]
        with safe_open(str(target / weight_map[A_LOG]), framework="pt") as f:
            assert torch.equal(f.get_tensor(A_LOG), a_log.to(torch.bfloat16))

    def test_a_dtype_difference_outside_the_allow_list_still_fails(self, tmp_path) -> None:
        """The caster leaves EMBED in fp32 (base bf16); the completion check refuses it."""
        with pytest.raises(hf_passthrough.ExportIncomplete, match="differ from the base in dtype or shape"):
            _export(tmp_path, [(A_LOG, _trained_a_log()), (EMBED, torch.ones(2, dtype=torch.float32))])
        assert not (tmp_path / "export").exists()


class TestFusedExpertsWithCompletion:
    @pytest.mark.parametrize("bucket_size", [1, 4, 100])
    def test_the_qwen38_converter_output_is_packed_and_passes_the_completion_check(self, tmp_path, bucket_size):
        """Before packing, the per-expert keys were not in the base and the completion check refused them."""
        experts = _megatron_experts()
        items = _qwen38_converted(experts) + [(A_LOG, _trained_a_log()), (EMBED, torch.ones(2, dtype=torch.bfloat16))]
        buckets = [items[i : i + bucket_size] for i in range(0, len(items), bucket_size)]
        args = _args(_write_base(tmp_path / "base", _fused(experts)))
        target = tmp_path / "export"

        hf_export.save_hf_model(args, 0, [], publisher=_publisher(args, buckets), path=target, raise_on_error=True)

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        weight_map = json.loads((target / INDEX).read_text())["weight_map"]
        assert set(weight_map) == {A_LOG, EMBED, *_fused(experts)}
        for name, tensor in _fused(experts).items():
            with safe_open(str(target / weight_map[name]), framework="pt") as f:
                assert torch.equal(f.get_tensor(name), tensor), name


def _base_tensor(key: str) -> torch.Tensor:
    if key == PLE_MULTIPLIERS:
        return torch.arange(3, dtype=torch.int64)
    return torch.full((2, 4), float(len(key)), dtype=torch.bfloat16)


def _qwen38_base(directory: Path, *, ple_shift: int = 0) -> Path:
    """A_log, embed, the PLE table, an MTP head and a vision tower; ``ple_shift`` changes the PLE shard bytes."""
    extra = {k: _base_tensor(k) for k in (PLE_SHARD, PLE_MULTIPLIERS, *MTP, VISUAL)}
    extra[PLE_SHARD] = extra[PLE_SHARD] + ple_shift
    return _write_base(directory, extra)


def _trained(*keys: str) -> list[tuple[str, torch.Tensor]]:
    out = [(A_LOG, _trained_a_log()), (EMBED, torch.ones(2, dtype=torch.bfloat16))]
    return out + [(k, _base_tensor(k) + 1) for k in keys]


def _read(target: Path) -> dict[str, torch.Tensor]:
    out = {}
    for name, file in json.loads((target / INDEX).read_text())["weight_map"].items():
        with safe_open(str(target / file), framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out


class TestQwen38Passthrough:
    def test_a_text_only_run_takes_the_ple_table_mtp_and_vision_from_the_base(self, tmp_path) -> None:
        """FIX-B-3 counts: passthrough = PLE + MTP + vision, exported = the rest."""
        base = _qwen38_base(tmp_path / "base")
        args, target = _args(base), tmp_path / "export"
        hf_export.save_hf_model(args, 0, [], publisher=_publisher(args, [_trained()]), path=target, raise_on_error=True)

        receipt = json.loads((target / hf_passthrough.RECEIPT).read_text())
        assert receipt["passthrough_groups"] == ["mtp", "ple_table", "visual"]
        assert (receipt["exported_keys"], receipt["passthrough_keys"], receipt["total_keys"]) == (2, 2 + len(MTP) + 1, 5 + len(MTP))
        exported = _read(target)
        for key in (PLE_SHARD, PLE_MULTIPLIERS, *MTP, VISUAL):
            assert torch.equal(exported[key], _base_tensor(key)), key

    @pytest.mark.parametrize("mtp_keys", [MTP, MTP[:1]], ids=["whole-head", "part-of-the-head"])
    def test_an_untrained_mtp_head_in_the_export_is_refused(self, tmp_path, mtp_keys) -> None:
        """Without --enable-mtp-training the head always comes from the base (only_if_absent=False)."""
        base = _qwen38_base(tmp_path / "base")
        args, target = _args(base), tmp_path / "export"
        with pytest.raises(hf_passthrough.ExportIncomplete, match="group mtp comes from the base"):
            hf_export.save_hf_model(
                args, 0, [], publisher=_publisher(args, [_trained(*mtp_keys)]), path=target, raise_on_error=True
            )
        assert not target.exists()

    def test_a_trained_mtp_head_is_kept_from_the_export(self, tmp_path) -> None:
        base = _qwen38_base(tmp_path / "base")
        args, target = _args(base, mtp=True), tmp_path / "export"
        hf_export.save_hf_model(
            args, 0, [], publisher=_publisher(args, [_trained(*MTP)]), path=target, raise_on_error=True
        )
        assert json.loads((target / hf_passthrough.RECEIPT).read_text())["passthrough_groups"] == ["ple_table", "visual"]
        exported = _read(target)
        for key in MTP:
            assert torch.equal(exported[key], _base_tensor(key) + 1), key


class TestPleProvenanceOfTheExport:
    """The completion copies the PLE table from --hf-checkpoint at export time; it must be the run's table."""

    def _bind(self, base: Path) -> None:
        assert ple_provenance.on_load(str(base), trained_checkpoint_dir=None) is not None

    def test_the_unchanged_base_completes(self, tmp_path) -> None:
        base = _qwen38_base(tmp_path / "base")
        self._bind(base)
        args, target = _args(base), tmp_path / "export"
        hf_export.save_hf_model(args, 0, [], publisher=_publisher(args, [_trained()]), path=target, raise_on_error=True)
        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()

    def test_a_ple_table_replaced_after_the_run_started_fails_the_export(self, tmp_path) -> None:
        base = _qwen38_base(tmp_path / "base")
        self._bind(base)
        shutil.rmtree(base)
        _qwen38_base(base, ple_shift=1)
        args, target = _args(base), tmp_path / "export"
        with pytest.raises(ple_provenance.PleProvenanceError, match="not the ones this run trained with"):
            hf_export.save_hf_model(
                args, 0, [], publisher=_publisher(args, [_trained()]), path=target, raise_on_error=True
            )
        assert not target.exists()
