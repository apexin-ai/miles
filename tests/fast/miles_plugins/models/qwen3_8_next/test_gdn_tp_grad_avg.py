"""CPU test: the replicated Qwen3.8-Next GDN must average its grads across TP.

``ReplicatedGDNAttention`` is not TP-sharded: every TP rank runs all of it on the same input (the
whole sequence, all-gathered under SP) with the same output grad, so every rank computes the full
grad. Each worker runs the real layer forward and backward (Megatron's real SP all-gather and
scatter) over a gloo TP group, then Megatron's real ``_allreduce_non_tensor_model_parallel_grads``.
The fla kernels cannot run on CPU, so ``chunk_gated_delta_rule``, ``ShortConvolution`` and
``FusedRMSNormGated`` are replaced by fp64 torch references. First, every rank's grad must equal
the single-rank grad G. Then rank r's grad is scaled by 1 + JITTER * (r + 1), a stand-in for copies
whose grads differ: after the reduction every rank must hold the same grad, equal to the arithmetic
mean of the perturbed grads, G * (1 + JITTER * (TP + 1) / 2) (not their sum, not one rank's copy),
and the same params after an optimizer step. The base class keeps an input RMSNorm (the Qwen3.8
subclass swaps it for an Identity); it is covered too.
"""

import importlib
import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

# 4 spawned gloo cases; about 90 s on an 8-thread CPU
register_cpu_ci(est_time=100, suite="stage-b-cpu", labels=["miles-plugin"])

_ROOT = Path(__file__).resolve().parents[5]
SEQ = 24
CU_SEQLENS = [0, 7, 15, SEQ]
TOL = 1e-12
# The relative change one rank's grad gets, as from a kernel that is not bitwise deterministic
JITTER = 1e-9
_AVG_FLAG = "average_gradients_across_tp_domain"  # what Megatron's finalize_model_grads reads


class _UnavailableRMSNorm(torch.nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("the GDN layer must not construct Transformer Engine layers")


class _ShortConvolution(torch.nn.Module):
    """fla ``ShortConvolution``: causal depthwise conv + silu, per packed sequence."""

    def __init__(self, hidden_size, kernel_size, bias=False, **kwargs):
        super().__init__()
        assert not bias
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, 1, kernel_size))

    def forward(self, x, cu_seqlens=None, cp_context=None):
        assert cp_context is None
        channels, _, kernel = self.weight.shape
        bounds = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        out = []
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            seq = F.pad(x[:, lo:hi].transpose(1, 2), (kernel - 1, 0))
            out.append(F.conv1d(seq, self.weight, groups=channels).transpose(1, 2))
        return F.silu(torch.cat(out, dim=1)), None


class _FusedRMSNormGated(torch.nn.Module):
    """fla ``FusedRMSNormGated``: rmsnorm(x) * weight * act(z)."""

    def __init__(self, hidden_size, eps, activation, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype))
        self.eps = eps
        self.act = torch.sigmoid if activation == "sigmoid" else F.silu

    def forward(self, x, z):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight * self.act(z)


def _chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=None, use_qk_l2norm_in_kernel=True, **kwargs):
    """The recurrent gated delta rule (fla's naive reference), per packed sequence."""
    assert use_qk_l2norm_in_kernel and q.shape[0] == 1
    q, k = F.normalize(q, dim=-1) * q.shape[-1] ** -0.5, F.normalize(k, dim=-1)
    bounds = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        state = v.new_zeros(q.shape[2], q.shape[3], v.shape[3])
        for t in range(lo, hi):
            state = state * g[0, t].exp()[:, None, None]
            delta = (v[0, t] - torch.einsum("hk,hkv->hv", k[0, t], state)) * beta[0, t][:, None]
            state = state + torch.einsum("hk,hv->hkv", k[0, t], delta)
            out.append(torch.einsum("hk,hkv->hv", q[0, t], state))
    return torch.stack(out).unsqueeze(0), None


def _hf_config():
    return SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        output_gate_type="sigmoid",
        rms_norm_eps=1e-6,
        dtype=torch.float64,
    )


def _use_cpu_references(gdn: ModuleType, setattr_=setattr) -> None:
    """Swap the fla kernels and the HF config load for CPU stand-ins."""
    from miles_plugins.models import hf_attention

    setattr_(hf_attention, "load_hf_config", lambda path: _hf_config())
    setattr_(gdn, "ShortConvolution", _ShortConvolution)
    setattr_(gdn, "FusedRMSNormGated", _FusedRMSNormGated)
    setattr_(gdn, "gdn_kernel", lambda backend: _chunk_gated_delta_rule)
    setattr_(torch.cuda, "current_device", lambda: "cpu")


def _load_gdn() -> ModuleType:
    """In a spawned worker: import gdn.py the way this directory's conftest.py ``gdn`` fixture does."""
    importlib.import_module("megatron.core.models.gpt.gpt_layer_specs")
    engine = ModuleType("transformer_engine")
    engine.pytorch = ModuleType("transformer_engine.pytorch")
    engine.pytorch.RMSNorm = _UnavailableRMSNorm
    sys.modules["transformer_engine"] = engine
    sys.modules["transformer_engine.pytorch"] = engine.pytorch
    for name, relative_path in (
        ("miles_plugins.models.qwen3_5.model", "miles_plugins/models/qwen3_5/model.py"),
        ("_test_qwen3_8_gdn", "miles_plugins/models/qwen3_8_next/gdn.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    _use_cpu_references(module)
    return module


def _build_layer(gdn: ModuleType, sequence_parallel: bool) -> torch.nn.Module:
    args = SimpleNamespace(
        hf_checkpoint=None,
        linear_attention_backend="fla",
        allgather_cp=False,
        backend=SimpleNamespace(sequence_parallel=sequence_parallel),
    )
    layer = gdn.ReplicatedGDNAttention(args, config=SimpleNamespace(), layer_number=1).to(torch.float64)
    layer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _, param in sorted(layer.named_parameters()):
            param.copy_(0.5 * torch.randn(param.shape, dtype=param.dtype, generator=gen))
    return layer


def _forward_backward(layer, x, upstream):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu = torch.tensor(CU_SEQLENS, dtype=torch.int32)
    out, _ = layer(x.clone().requires_grad_(), None, packed_seq_params=PackedSeqParams(cu_seqlens_q=cu))
    out.backward(upstream)


def _gather(tensor, group):
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return gathered


def _worker(rank, world_size, port, sequence_parallel):
    init_gloo(rank, world_size, port=port)
    try:
        from megatron.core import parallel_state
        from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        gdn = _load_gdn()
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(SEQ, 1, 16, dtype=torch.float64, generator=gen)
        upstream = torch.randn(SEQ, 1, 16, dtype=torch.float64, generator=gen)

        # single-rank reference: the same layer, no TP split
        ref_layer = _build_layer(gdn, sequence_parallel=False)
        _forward_backward(ref_layer, x, upstream)
        ref = {name: param.grad for name, param in ref_layer.named_parameters()}
        assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in ref.values())

        layer = _build_layer(gdn, sequence_parallel)
        names = [name for name, _ in layer.named_parameters()]
        assert {"linear_attn.A_log", "linear_attn.conv1d.weight", "linear_attn.norm.weight"} <= set(names)
        assert any(name.startswith("input_layernorm.") for name in names)
        if sequence_parallel:
            local = SEQ // world_size
            own = slice(rank * local, (rank + 1) * local)
            _forward_backward(layer, x[own], upstream[own])
        else:
            _forward_backward(layer, x, upstream)
        for name, param in layer.named_parameters():
            # every TP rank holds the full grad, not a partial sum, so averaging is the right reduction
            err = (param.grad - ref[name]).abs().max().item()
            assert err <= TOL * max(1.0, ref[name].abs().max().item()), f"rank {rank} {name} grad err {err:.3e}"
            param.main_grad = param.grad * (1 + JITTER * (rank + 1))
            param.grad = None

        config = SimpleNamespace(sequence_parallel=sequence_parallel, qk_layernorm=False)
        _allreduce_non_tensor_model_parallel_grads([layer], config, tp_group)

        for name, param in layer.named_parameters():
            copies = _gather(param.main_grad, tp_group)
            assert all(torch.equal(c, copies[0]) for c in copies), f"{name} grad differs across TP ranks"
            # mean over ranks r of G * (1 + JITTER * (r + 1)); TP rank 0's copy alone is off by (TP - 1) * JITTER / 2
            expected = ref[name] * (1 + JITTER * (world_size + 1) / 2)
            err = (param.main_grad - expected).abs().max().item()
            assert err <= TOL * ref[name].abs().max().item(), f"{name} grad is not the TP average ({err:.3e})"
        with torch.no_grad():
            for param in layer.parameters():
                param.sub_(0.1 * param.main_grad)
        for name, param in layer.named_parameters():
            copies = _gather(param.detach(), tp_group)
            assert all(torch.equal(c, copies[0]) for c in copies), f"{name} differs across TP ranks after a step"
    finally:
        dist.destroy_process_group()


def test_every_replicated_gdn_param_is_tp_averaged(monkeypatch: pytest.MonkeyPatch, gdn: ModuleType) -> None:
    """All GDN params, the fp32 A_log and the input norm included, carry Megatron's TP average flag."""
    _use_cpu_references(gdn, monkeypatch.setattr)
    layer = _build_layer(gdn, sequence_parallel=True)
    unmarked = [name for name, param in layer.named_parameters() if vars(param).get(_AVG_FLAG) is not True]
    assert unmarked == []
    assert not any("tensor_model_parallel" in vars(param) for param in layer.parameters())


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("sequence_parallel", [True, False], ids=["sp", "no-sp"])
def test_tp_replicas_get_the_same_averaged_grad(tp, sequence_parallel):
    run_multiprocess(partial(_worker, sequence_parallel=sequence_parallel), world_size=tp)
