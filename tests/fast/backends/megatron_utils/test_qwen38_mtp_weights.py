"""Exercise the same bridge, converter and expert packer used by Qwen3.8 training."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf
from miles.backends.training_utils.weight_update.hf_expert_packing import ExpertPacker
from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge


def _tiny_base(real_base_dir=None):
    prefix = "mtp."
    block = prefix + "layers.0."
    shapes = {
        prefix + "pre_fc_norm_embedding.weight": (8,),
        prefix + "pre_fc_norm_hidden.weight": (32,),
        prefix + "fc_embedding.weight": (8, 8),
        prefix + "fc_hidden.weight": (8, 8),
    }
    for hc in ("hyper_connection_mixer", "layers.0.attn_hyper_connection", "layers.0.mlp_hyper_connection"):
        shapes[prefix + hc + ".hc_norm.weight"] = (32,)
        shapes[prefix + hc + ".input_mix_weight_down.weight"] = (4, 32)
        shapes[prefix + hc + ".input_mix_weight_up.weight"] = (32, 4)
        if hc.startswith("layers."):
            shapes[prefix + hc + ".block_inject_weight.weight"] = (4, 32)
    shapes.update(
        {
            block + name: shape
            for name, shape in {
                "self_attn.q_proj.weight": (16, 8),
                "self_attn.k_proj.weight": (4, 8),
                "self_attn.v_proj.weight": (4, 8),
                "self_attn.o_proj.weight": (8, 8),
                "self_attn.q_norm.weight": (2,),
                "self_attn.k_norm.weight": (2,),
                "self_attn.indexer.index_qk_proj.weight": (4, 8),
                "self_attn.indexer.k_layernorm.weight": (2,),
                "self_attn.indexer.q_layernorm.weight": (2,),
                "mlp.shared_expert.gate_proj.weight": (4, 8),
                "mlp.shared_expert.up_proj.weight": (4, 8),
                "mlp.shared_expert.down_proj.weight": (8, 4),
                "mlp.shared_expert_gate.weight": (1, 8),
                "mlp.gate.weight": (4, 8),
                "mlp.experts.gate_up_proj": (4, 8, 8),
                "mlp.experts.down_proj": (4, 8, 4),
            }.items()
        }
    )
    assert len(shapes) == 31
    if real_base_dir is not None:
        index = json.loads((real_base_dir / "model.safetensors.index.json").read_text())["weight_map"]
        assert {name for name in index if name.startswith("mtp.")} == shapes.keys()
        tensors = {}
        for key, shape in shapes.items():
            with safe_open(real_base_dir / index[key], framework="pt") as shard:
                view = shard.get_slice(key)
                if key.endswith("experts.gate_up_proj"):
                    # Preserve both halves of the SwiGLU projection when cropping the intermediate width.
                    half = view.get_shape()[1] // 2
                    tensors[key] = torch.cat([view[:4, :4, :8], view[:4, half : half + 4, :8]], dim=1)
                else:
                    tensors[key] = view[tuple(slice(n) for n in shape)].contiguous()
        return tensors
    generator = torch.Generator().manual_seed(81)
    return {name: torch.randn(shape, generator=generator) for name, shape in shapes.items()}


def _local_names():
    prefix = "mtp.layers.0."
    for key in ("enorm.weight", "hnorm.weight", "e_proj.weight", "h_proj.weight"):
        yield prefix + key
    for hc in ("hc_norm_weight", "input_mix_weight_down", "input_mix_weight_up"):
        yield prefix + "hc_head_contraction." + hc
    block = prefix + "mtp_model_layer."
    for slot in ("self_attention_hyper_connection", "mlp_hyper_connection"):
        for key in ("hc_norm_weight", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
            yield block + slot + "." + key
    for key in (
        "self_attention.linear_qkv.weight",
        "self_attention.linear_proj.weight",
        "self_attention.q_layernorm.weight",
        "self_attention.k_layernorm.weight",
        "self_attention.indexer.index_qk_proj.weight",
        "self_attention.indexer.k_layernorm",
        "self_attention.indexer.q_layernorm",
        "mlp.router.weight",
        "mlp.shared_experts.gate_weight",
        "mlp.shared_experts.linear_fc1.weight",
        "mlp.shared_experts.linear_fc2.weight",
    ):
        yield block + key


def _bridge(ep_rank):
    bridge = Qwen38NextBridge.__new__(Qwen38NextBridge)
    bridge.hf_config = SimpleNamespace(
        hidden_size=8, head_dim=2, num_attention_heads=4, num_key_value_heads=2, num_experts=4
    )
    bridge.mpu = SimpleNamespace(ep_rank=ep_rank, ep_size=2)
    bridge.config = SimpleNamespace(gated_linear_unit=True)
    bridge.make_vocab_size_divisible_by = None
    return bridge


@pytest.mark.parametrize("real_weights", [False, True], ids=["tiny", "base-slices"])
def test_native_head_cpu_checkpoint_roundtrip(tmp_path, real_weights):
    path = os.environ.get("QWEN38_TEST_BASE_DIR")
    if real_weights and not path:
        pytest.skip("set QWEN38_TEST_BASE_DIR to exercise the release safetensors")
    base = _tiny_base(Path(path) if real_weights else None)
    base_path = tmp_path / "model.safetensors"
    save_file(base, str(base_path))
    source = load_file(str(base_path))
    global_state = {}
    for rank in range(2):
        bridge = _bridge(rank)
        expert_prefix = "mtp.layers.0.mtp_model_layer.mlp.experts."
        names = list(_local_names()) if rank == 0 else []
        names.extend(expert_prefix + f"linear_fc{fc}.weight{local}" for fc in (1, 2) for local in range(2))
        for name in names:
            hf_names = bridge._weight_name_mapping_mcore_to_hf(name)
            weight = bridge._weight_to_mcore_format(name, [source[k] for k in hf_names])
            if name.endswith(("e_proj.weight", "h_proj.weight", "linear_qkv.weight")):
                # Production has TP=4 but only two KV groups; storage uses four contiguous QGKV chunks.
                shards = bridge._weight_split_across_tp(name, weight, weight[: weight.size(0) // 4], 4)
                weight = torch.cat(shards)
            if name.startswith(expert_prefix):
                from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import _gather_with_stride

                dim, stride = (0, 2) if "linear_fc1" in name else (1, 1)
                local = weight.narrow(dim, 0, weight.shape[dim] // 2)
                etp_shards = bridge._weight_split_across_tp(name, weight, local, 2)
                weight = _gather_with_stride(etp_shards, dim, stride)
                # named_weights translates each EP rank's two local expert IDs to global IDs.
                name = name[:-1] + str(rank * 2 + int(name[-1]))
            global_state[name] = weight + 1
    import torch.distributed.checkpoint as dcp

    dcp.save(global_state, checkpoint_id=tmp_path / "native", no_dist=True)
    restored = {key: torch.empty_like(value) for key, value in global_state.items()}
    dcp.load(restored, checkpoint_id=tmp_path / "native", no_dist=True)

    args = SimpleNamespace(
        backend=SimpleNamespace(hidden_size=8, kv_channels=2, num_attention_heads=4, num_query_groups=2)
    )
    converted = []
    for name, value in restored.items():
        converted.extend(convert_qwen3_8_next_to_hf(args, "module.module." + name, value))
    packer = ExpertPacker(tmp_path)
    packed = dict(packer.pack(converted))
    packer.finish()
    from examples.qwen3_8_next_vision_sft import export_full

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(source, base_path.name)})
    )
    index, by_file, missing = export_full._check_inventory(tmp_path, packed, mtp_trained=True)
    assert not missing
    destination = tmp_path / "export"
    destination.mkdir()
    export_full._write_shards(tmp_path, destination, packed, index, by_file, missing)
    exported = load_file(str(destination / base_path.name))
    assert set(exported) == set(source)
    for key in source:
        torch.testing.assert_close(exported[key], source[key] + 1, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["mtp.layers.0.eh_proj.weight", "mtp.layers.1.e_proj.weight"])
def test_reject_foreign_mtp_layout(bad):
    with pytest.raises((ValueError, NotImplementedError)):
        _bridge(0)._weight_name_mapping_mcore_to_hf(bad)


def test_offline_export_checks_trained_head_even_if_all_mtp_keys_are_missing(tmp_path):
    from examples.qwen3_8_next_vision_sft.export_full import _check_inventory

    tensors = _tiny_base()
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    for missing in (set(tensors), {"mtp.fc_hidden.weight"}):
        with pytest.raises(ValueError, match="missing="):
            _check_inventory(tmp_path, {k: v for k, v in tensors.items() if k not in missing}, mtp_trained=True)
    assert _check_inventory(tmp_path, {}, mtp_trained=False)[2] == set(tensors)


def test_dcp_stacked_experts_match_raw_and_hf_inventory(tmp_path):
    from examples.qwen3_8_next_vision_sft.export_full import _convert, _model_args

    config = dict(
        num_hidden_layers=2,
        num_experts=4,
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        vocab_size=32,
    )
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config}))
    args = _model_args(tmp_path)
    tensors = _tiny_base()
    restored = {
        "mtp.layers.0.transformer_layer.mlp.experts.experts.linear_fc1.weight": tensors[
            "mtp.layers.0.mlp.experts.gate_up_proj"
        ],
        "mtp.layers.0.transformer_layer.mlp.experts.experts.linear_fc2.weight": tensors[
            "mtp.layers.0.mlp.experts.down_proj"
        ],
        "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight": torch.randn(24, 8),
    }
    converted = _convert(args, restored)
    for k in ("mtp.layers.0.mlp.experts.gate_up_proj", "mtp.layers.0.mlp.experts.down_proj"):
        torch.testing.assert_close(converted[k], tensors[k])
    assert "mtp.layers.0.self_attn.q_proj.weight" in converted


@pytest.mark.parametrize("missing", [False, True])
def test_public_bridge_load_does_not_initialize_a_missing_base_key(tmp_path, missing):
    # One-rank CPU group uses mbridge's real scatter/load path. Only one projection is needed
    # to distinguish a missing initial weight from a successful load.
    import torch.distributed as dist

    base = _tiny_base()
    expected = base["mtp.fc_hidden.weight"]
    if missing:
        del base["mtp.fc_hidden.weight"]
    save_file(base, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text("{}")
    model = torch.nn.Module()
    model.mtp = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.mtp.layers[0].h_proj = torch.nn.Linear(8, 8, bias=False)
    bridge = _bridge(0)
    dist.init_process_group("gloo", init_method=(tmp_path / "gloo").as_uri(), rank=0, world_size=1)
    try:
        bridge.mpu = SimpleNamespace(
            tp_rank=0,
            etp_rank=0,
            tp_size=1,
            etp_size=1,
            ep_size=1,
            tp_group=dist.group.WORLD,
            etp_group=dist.group.WORLD,
        )
        if missing:
            with pytest.raises((KeyError, ValueError), match="mtp.fc_hidden.weight"):
                bridge.load_weights([model], str(tmp_path), memory_efficient=True)
        else:
            bridge.load_weights([model], str(tmp_path), memory_efficient=True)
            torch.testing.assert_close(model.mtp.layers[0].h_proj.weight, expected)
    finally:
        dist.destroy_process_group()
