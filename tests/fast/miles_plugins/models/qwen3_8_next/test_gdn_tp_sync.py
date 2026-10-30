"""The replicated Qwen3.8 GDN must stay bitwise identical on every TP rank.

Each TP rank computes the full GDN; fla's autotuned kernels make the per-rank gradients differ in
the last bits, and a checkpoint keeps TP rank 0's copy only. The workers below run Megatron's real
finalize step over a gloo TP2 group with the real GDN layer built on CPU.
"""

import importlib
import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from tests.fast.dist_utils import init_gloo, run_multiprocess

_ROOT = Path(__file__).resolve().parents[5]


class _CpuSdkModule(torch.nn.Module):
    """Stands in for fla's ShortConvolution / FusedRMSNormGated: one weight, no CUDA."""

    def __init__(self, *args: Any, hidden_size: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size))


class _NoTransformerEngine(torch.nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the GDN layer must not construct Transformer Engine modules")


def _load_gdn(mp: pytest.MonkeyPatch) -> ModuleType:
    importlib.import_module("megatron.core.models.gpt.gpt_layer_specs")
    engine = ModuleType("transformer_engine")
    engine.pytorch = ModuleType("transformer_engine.pytorch")
    engine.pytorch.RMSNorm = _NoTransformerEngine
    mp.setitem(sys.modules, "transformer_engine", engine)
    mp.setitem(sys.modules, "transformer_engine.pytorch", engine.pytorch)
    fla = ModuleType("fla.ops.gated_delta_rule")
    fla.chunk_gated_delta_rule = lambda *a, **k: None
    mp.setitem(sys.modules, "fla.ops.gated_delta_rule", fla)
    module = None
    for name, relative_path in (
        ("miles_plugins.models.qwen3_5", "miles_plugins/models/qwen3_5.py"),
        ("_test_qwen3_8_gdn_tp", "miles_plugins/models/qwen3_8_next/gdn.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    mp.setattr(module, "ShortConvolution", _CpuSdkModule, raising=False)
    mp.setattr(module, "FusedRMSNormGated", _CpuSdkModule, raising=False)
    mp.setattr(torch.cuda, "current_device", lambda: "cpu")

    def light_base_init(self, args, config, layer_number, *rest: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.hf_layer_idx = layer_number - 1
        self.hf_config = SimpleNamespace(
            hidden_size=8,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=4,
            hidden_act="silu",
            output_gate_type="sigmoid",
            rms_norm_eps=1e-6,
            dtype=torch.bfloat16,
        )

    mp.setattr(module.HuggingfaceAttention, "__init__", light_base_init)
    return module


def _build_layer(gdn: ModuleType) -> torch.nn.Module:
    torch.manual_seed(0)
    layer = gdn.ReplicatedGDNAttention(
        args=SimpleNamespace(linear_attention_backend="fla"), config=SimpleNamespace(), layer_number=1
    )
    return layer.to(torch.bfloat16)


class _Chunk(torch.nn.Module):
    """What finalize_model_grads reads from a DDP model chunk."""

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=False)


def _train_steps_worker(rank: int, world_size: int, port: int, *, keep_tp_average: bool) -> None:
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads

    from miles.backends.megatron_utils.tp_replica_check import check_tp_replicas

    init_gloo(rank, world_size, port=port)
    try:
        chunk = _Chunk(_build_layer(_load_gdn(pytest.MonkeyPatch())))
        if not keep_tp_average:
            for param in chunk.parameters():
                del param.average_gradients_across_tp_domain
        assert check_tp_replicas([chunk], None, group=dist.group.WORLD, where="init") > 0

        config = SimpleNamespace(sequence_parallel=True, qk_layernorm=False)
        for step in range(3):
            for param in chunk.parameters():
                generator = torch.Generator().manual_seed(step * 1000 + param.numel())
                # Same gradient on every rank up to a last-bit difference, as from an autotuned kernel.
                grad = torch.randn(param.shape, generator=generator) * (1 + 1e-3 * rank)
                param.main_grad = grad.float()
            _allreduce_non_tensor_model_parallel_grads([chunk], config, dist.group.WORLD)
            with torch.no_grad():
                for param in chunk.parameters():
                    param.sub_(0.1 * param.main_grad.to(param.dtype))

        if keep_tp_average:
            check_tp_replicas([chunk], None, group=dist.group.WORLD, where="after 3 steps")
        else:
            with pytest.raises(RuntimeError, match="TP-replicated tensors differ across TP ranks"):
                check_tp_replicas([chunk], None, group=dist.group.WORLD, where="after 3 steps")
    finally:
        dist.destroy_process_group()


class TestGdnTpSync:
    def test_every_gdn_parameter_is_averaged_across_tp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Projections, conv, A_log, dt_bias, the gated norm and the input norm all carry the flag."""
        layer = _build_layer(_load_gdn(monkeypatch))
        names = [name for name, _ in layer.named_parameters()]
        assert {"linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.out_proj.weight"} <= set(names)
        assert any(name.startswith("input_layernorm.") for name in names)
        assert any(name.startswith("linear_attn.conv1d.") for name in names)
        assert any(name.startswith("linear_attn.norm.") for name in names)
        unmarked = [n for n, p in layer.named_parameters() if not p.average_gradients_across_tp_domain]
        assert unmarked == []

    def test_a_tp_sharded_parameter_cannot_be_marked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Averaging a TP shard would mix different slices; the marker refuses."""
        gdn = _load_gdn(monkeypatch)
        module = torch.nn.Linear(2, 2)
        module.weight.tensor_model_parallel = True
        with pytest.raises(AssertionError, match="TP-sharded"):
            gdn.mark_tp_replicated(module)

    def test_tp2_replicas_stay_identical_through_finalize_and_steps(self) -> None:
        """With the flag, Megatron's finalize averages the drifting grads and the copies stay equal."""
        run_multiprocess(partial(_train_steps_worker, keep_tp_average=True), world_size=2)

    def test_without_the_flag_the_replicas_drift_and_the_check_fails(self) -> None:
        """Positive control: the same run without TP averaging is caught by the replica check."""
        run_multiprocess(partial(_train_steps_worker, keep_tp_average=False), world_size=2)
