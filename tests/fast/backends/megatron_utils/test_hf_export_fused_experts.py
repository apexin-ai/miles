"""Raw HF exports keep the fused MoE expert layout of the base checkpoint.

Runs the real Qwen3.5 converter, ``SnapshotPublisher`` and ``save_hf_model``; only parameter
enumeration and the weight gather are replaced.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.backends.megatron_utils import hf_export
from miles.backends.megatron_utils.megatron_to_hf.qwen3_5 import convert_qwen3_5_to_hf
from miles.backends.training_utils.weight_update import snapshot_publisher
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.hf_utils.config import HF_EXPORT_COMPLETE_MARKER

INDEX = "model.safetensors.index.json"
NUM_EXPERTS, HIDDEN, INTER = 3, 4, 2
PREFIX = "model.language_model.layers.0.mlp.experts"
EMBED = "model.language_model.embed_tokens.weight"


class _Iterator:
    placement = SimpleNamespace(is_full_gather=True)

    def __init__(self, buckets) -> None:
        self._buckets = buckets

    def iter_hf_weights(self, weights):
        yield from self._buckets


def _megatron_experts() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """linear_fc1 [2I, H] (gate rows, then up rows) and linear_fc2 [H, I] of each expert, distinct values."""
    gen = torch.Generator().manual_seed(0)
    return {
        e: (torch.randn(2 * INTER, HIDDEN, generator=gen), torch.randn(HIDDEN, INTER, generator=gen))
        for e in range(NUM_EXPERTS)
    }


def _converted(experts) -> list[tuple[str, torch.Tensor]]:
    backend = SimpleNamespace(kv_channels=1, hidden_size=HIDDEN, num_attention_heads=1, num_query_groups=1)
    args = SimpleNamespace(backend=backend)
    out = []
    for e, (fc1, fc2) in experts.items():
        layer = "module.module.decoder.layers.0.mlp.experts"
        out += convert_qwen3_5_to_hf(args, f"{layer}.linear_fc1.weight{e}", fc1)
        out += convert_qwen3_5_to_hf(args, f"{layer}.linear_fc2.weight{e}", fc2)
    return out + [(EMBED, torch.ones(2))]


def _fused(experts) -> dict[str, torch.Tensor]:
    """The fused layout: gate_up_proj[e] is expert e's linear_fc1, down_proj[e] its linear_fc2."""
    return {
        f"{PREFIX}.gate_up_proj": torch.stack([experts[e][0] for e in range(NUM_EXPERTS)]),
        f"{PREFIX}.down_proj": torch.stack([experts[e][1] for e in range(NUM_EXPERTS)]),
    }


def _write_base(directory: Path, tensors: dict[str, torch.Tensor]) -> Path:
    directory.mkdir()
    save_file(tensors, str(directory / "model-00001-of-00001.safetensors"))
    weight_map = dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
    (directory / INDEX).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (directory / "config.json").write_text("{}")
    return directory


def _read(target: Path) -> dict[str, torch.Tensor]:
    out = {}
    for name, file in json.loads((target / INDEX).read_text())["weight_map"].items():
        with safe_open(str(target / file), framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out


def _buckets(items, size: int):
    return [items[i : i + size] for i in range(0, len(items), size)]


def _patch_single_process(patch) -> None:
    rank0 = SimpleNamespace(rank=0)
    patch(hf_export, "named_params_and_buffers", lambda *args, **kwargs: [])
    patch(hf_export, "is_lora_model", lambda model: False)
    patch(hf_export, "get_parallel_state", lambda: SimpleNamespace(effective_dp_cp=rank0, tp=rank0))


class TestSingleRank:
    @pytest.fixture(autouse=True)
    def _single_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(snapshot_publisher.dist, "get_rank", lambda: 0)
        _patch_single_process(monkeypatch.setattr)

    def _export(self, tmp_path: Path, base: Path, buckets) -> Path:
        args = SimpleNamespace(megatron_to_hf_mode="raw", hf_checkpoint=str(base), save_hf=None)
        target = tmp_path / "export"
        publisher = SnapshotPublisher(_Iterator(buckets))
        hf_export.save_hf_model(args, 0, [], publisher=publisher, path=target, raise_on_error=True)
        return target

    @pytest.mark.parametrize("bucket_size", [1, 4, 100])
    def test_split_converter_output_is_written_in_the_fused_base_layout(self, tmp_path, bucket_size):
        experts = _megatron_experts()
        base = _write_base(tmp_path / "base", _fused(experts) | {EMBED: torch.zeros(2)})

        target = self._export(tmp_path, base, _buckets(_converted(experts), bucket_size))

        assert (target / HF_EXPORT_COMPLETE_MARKER).exists()
        exported = _read(target)
        assert set(exported) == {f"{PREFIX}.gate_up_proj", f"{PREFIX}.down_proj", EMBED}
        for name, tensor in _fused(experts).items():
            assert torch.equal(exported[name], tensor)
        for e, (fc1, _) in experts.items():
            gate, up = fc1.chunk(2, dim=0)
            assert torch.equal(exported[f"{PREFIX}.gate_up_proj"][e][:INTER], gate)
            assert torch.equal(exported[f"{PREFIX}.gate_up_proj"][e][INTER:], up)

    def test_a_base_with_split_experts_keeps_split_keys(self, tmp_path):
        # gate and up are views of one linear_fc1 on CPU; the GPU -> CPU copy separates them in training
        converted = [(name, tensor.clone()) for name, tensor in _converted(_megatron_experts())]
        base = _write_base(tmp_path / "base", dict(converted))

        target = self._export(tmp_path, base, _buckets(converted, 4))

        assert set(_read(target)) == {name for name, _ in converted}

    def test_a_missing_expert_fails_the_export_without_a_marker(self, tmp_path):
        experts = _megatron_experts()
        base = _write_base(tmp_path / "base", _fused(experts) | {EMBED: torch.zeros(2)})
        converted = [item for item in _converted(experts) if ".experts.1." not in item[0]]

        with pytest.raises(Exception, match="left incomplete"):
            self._export(tmp_path, base, [converted])
        assert not (tmp_path / "export").exists()


def _gather_worker(rank: int, world_size: int, port: int, base: str, out: str) -> None:
    from tests.fast.dist_utils import init_gloo

    init_gloo(rank, world_size, port=port)
    gathered = []

    class Iterator:
        placement = SimpleNamespace(is_full_gather=True)

        def iter_hf_weights(self, weights):
            for i in range(3):
                t = torch.ones(1)
                torch.distributed.all_reduce(t)  # stands in for the TP/EP weight gather
                gathered.append(int(t.item()))
                yield [(f"{EMBED}{i}", torch.ones(2))]

    target = Path(out) / f"rank{rank}"
    target.mkdir()
    try:
        SnapshotPublisher(Iterator()).write_model(target, weights={}, hf_checkpoint=base)
        error = None
    except Exception as exc:  # noqa: BLE001 - the test records which rank failed and why
        error = type(exc).__name__
    after = torch.ones(1)
    torch.distributed.all_reduce(after)
    result = {"error": error, "gathers": gathered, "next_collective": int(after.item())}
    (Path(out) / f"rank{rank}.json").write_text(json.dumps(result))
    torch.distributed.destroy_process_group()


def test_a_broken_base_index_fails_the_writer_only_after_every_rank_finished_the_gathers(tmp_path):
    from tests.fast.dist_utils import find_free_port

    base = tmp_path / "base"
    base.mkdir()
    (base / INDEX).write_text("{not json")
    torch.multiprocessing.spawn(
        _gather_worker, args=(2, find_free_port(), str(base), str(tmp_path)), nprocs=2, join=True
    )

    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert results[0]["error"] == "JSONDecodeError"
    assert results[1]["error"] is None
    assert all(r["gathers"] == [2, 2, 2] and r["next_collective"] == 2 for r in results)
