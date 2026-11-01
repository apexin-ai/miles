"""CPU tests: Qwen3.8-Next's head-sharded GDN (``--qwen38-gdn-head-sharded``) against the replicated GDN.

* HF -> Megatron -> HF: the bridge's real load conversion and TP split, then the gather and both exports (the raw
  converter and the bridge's own), give back every HF tensor bitwise, at TP 1 / 2 / 8.
* Forward / backward: the real ``HeadShardedGDNAttention`` (Qwen3.5's ``LinearAttentionLayer``) loaded through the
  bridge, and the real ``ReplicatedGDNAttention`` (HF-attention shim) holding the same HF weights, run over a gloo TP
  group with and without sequence parallelism. Output, input gradient and every parameter gradient (gathered and
  exported to HF layout) must agree to 1e-10 in fp64. fla's Triton kernels cannot run on CPU, so the short conv,
  the gated RMSNorm and the gated delta rule are fp64 torch references, the same ones for both layers: the test
  covers the wiring (group-major rows, head sharding, TP collectives, gate activation, input norm), not fla.
* The spec builds the head-sharded layer exactly when the flag is set.

Shapes keep Qwen3.8-Next's ratio of 3 value heads per key head, so the group-major permutation is not the identity.
"""

import sys
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

# 8 spawned gloo cases (up to 8 ranks each) plus single-process cases
register_cpu_ci(est_time=240, suite="stage-b-cpu", labels=["miles-plugin"])

HIDDEN, NUM_K, NUM_V, HEAD_K, HEAD_V, CONV, EPS = 16, 8, 24, 4, 4, 4, 1e-6
SEQ = 24
CU_BOUNDS = [0, 7, SEQ]
TOL = 1e-10
LAYER = "decoder.layers.0.self_attention.linear_attn"
HF_LAYER = "model.language_model.layers.0.linear_attn"
NAMES = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_b.weight",
    "in_proj_a.weight",
    "conv1d.weight",
    "dt_bias",
    "A_log",
    "norm.weight",
    "out_proj.weight",
)


def _text_config(output_gate_type="sigmoid"):
    return SimpleNamespace(
        hidden_size=HIDDEN,
        linear_num_key_heads=NUM_K,
        linear_num_value_heads=NUM_V,
        linear_key_head_dim=HEAD_K,
        linear_value_head_dim=HEAD_V,
        linear_conv_kernel_dim=CONV,
        hidden_act="silu",
        output_gate_type=output_gate_type,
        rms_norm_eps=EPS,
        dtype=torch.float64,
    )


def _hf_weights(dtype=torch.float64):
    """One GDN layer's HF tensors. A_log values are fp32-exact: the head-sharded layer holds it in fp32."""
    gen = torch.Generator().manual_seed(1234)
    key_dim, value_dim = NUM_K * HEAD_K, NUM_V * HEAD_V
    shapes = {
        "in_proj_qkv.weight": (2 * key_dim + value_dim, HIDDEN),
        "in_proj_z.weight": (value_dim, HIDDEN),
        "in_proj_b.weight": (NUM_V, HIDDEN),
        "in_proj_a.weight": (NUM_V, HIDDEN),
        "conv1d.weight": (2 * key_dim + value_dim, 1, CONV),
        "dt_bias": (NUM_V,),
        "norm.weight": (HEAD_V,),
        "out_proj.weight": (HIDDEN, value_dim),
    }
    weights = {
        name: (0.3 * torch.randn(shape, generator=gen, dtype=torch.float64)).to(dtype)
        for name, shape in shapes.items()
    }
    weights["norm.weight"] = (1 + weights["norm.weight"].double()).to(dtype)
    weights["A_log"] = torch.empty(NUM_V, dtype=torch.float32).uniform_(1, 16, generator=gen).log().to(dtype)
    return weights


# ---------------------------------------------------------------- fp64 torch references of the fla kernels


def _ref_conv(x, weight, cu_seqlens):
    """Causal depthwise conv + SiLU of ``x`` ``[b, s, C]`` with ``weight`` ``[C, K]``, reset at sequence starts."""
    channels, kernel = weight.shape

    def conv(seq):
        h = F.pad(seq.transpose(1, 2), (kernel - 1, 0))
        return F.silu(F.conv1d(h, weight[:, None, :], groups=channels)).transpose(1, 2)

    if cu_seqlens is None:
        return conv(x)
    bounds = cu_seqlens.tolist()
    return torch.cat([conv(x[:, lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)], dim=1)


_GATE_ACT = {"silu": F.silu, "swish": F.silu, "sigmoid": torch.sigmoid}


def _ref_gated_norm(x, gate, weight, eps, activation):
    x32 = x.double()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight * _GATE_ACT[activation](gate.double())).to(x.dtype)


def _ref_gdn(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    cp_context=None,
    scale=None,
):
    """Gated delta rule, token by token. q / k may hold fewer heads than v: value head h reads key head
    ``h // (H // G)``, as fla groups them."""
    assert initial_state is None and cp_context is None
    batch, seq_len, groups, head_k = q.shape
    heads = v.shape[2]
    if groups != heads:
        q, k = (t.repeat_interleave(heads // groups, dim=2) for t in (q, k))
    if use_qk_l2norm_in_kernel:
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    scale = head_k**-0.5 if scale is None else scale
    bounds = [0, seq_len] if cu_seqlens is None else cu_seqlens.tolist()
    outs = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        state = v.new_zeros(batch, heads, head_k, v.shape[-1])
        for t in range(lo, hi):
            state = state * g[:, t].exp()[..., None, None]
            pred = torch.einsum("bhkv,bhk->bhv", state, k[:, t])
            state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], beta[:, t][..., None] * (v[:, t] - pred))
            outs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t] * scale))
    return torch.stack(outs, dim=1), None


class _RefShortConv(nn.Conv1d):
    """fla's ShortConvolution (an nn.Conv1d with weight ``[C, 1, K]``) on the torch reference."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", **kwargs):
        super().__init__(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False, padding=kernel_size - 1
        )

    def forward(self, x, cu_seqlens=None, cp_context=None, **kwargs):
        return _ref_conv(x, self.weight[:, 0], cu_seqlens), None


class _RefGatedNorm(nn.Module):
    """fla's FusedRMSNormGated on the torch reference."""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.bias = None
        self.eps = eps
        self.activation = activation

    def forward(self, x, gate):
        return _ref_gated_norm(x, gate, self.weight, self.eps, self.activation)


def _install_cpu_stubs():
    """Process-local: fla on the torch references, no Transformer Engine, no CUDA."""
    import megatron.core.tensor_parallel.random  # noqa: F401  (Megatron probes TE at import; let it see none)

    engine = ModuleType("transformer_engine")
    engine.pytorch = ModuleType("transformer_engine.pytorch")
    engine.pytorch.RMSNorm = nn.Module
    sys.modules["transformer_engine"] = engine
    sys.modules["transformer_engine.pytorch"] = engine.pytorch
    fla = ModuleType("fla")
    fla.modules = ModuleType("fla.modules")
    fla.modules.FusedRMSNormGated = _RefGatedNorm
    fla.modules.ShortConvolution = _RefShortConv
    fused = ModuleType("fla.modules.fused_norm_gate")
    fused.rms_norm_gated = _ref_rms_norm_gated
    ops = ModuleType("fla.ops.gated_delta_rule")
    ops.chunk_gated_delta_rule = _ref_gdn
    for name, module in {
        "fla": fla,
        "fla.modules": fla.modules,
        "fla.modules.fused_norm_gate": fused,
        "fla.ops": ModuleType("fla.ops"),
        "fla.ops.gated_delta_rule": ops,
    }.items():
        sys.modules[name] = module
    torch.cuda.current_device = lambda: "cpu"

    import miles_plugins.mbridge  # noqa: F401
    from miles_plugins.models import linear_attn
    from miles_plugins.models.qwen3_8_next import gdn

    # in case a module bound the names before the stubs were in place
    linear_attn.FusedRMSNormGated = _RefGatedNorm
    linear_attn.rms_norm_gated = _ref_rms_norm_gated
    gdn.FusedRMSNormGated = _RefGatedNorm
    gdn.ShortConvolution = _RefShortConv
    linear_attn.causal_short_conv = _ref_causal_short_conv
    linear_attn.get_cuda_rng_tracker = lambda: SimpleNamespace(fork=lambda *a, **k: _NullContext())


def _ref_rms_norm_gated(x, g, weight, bias, activation, eps):
    assert bias is None
    return _ref_gated_norm(x, g, weight, eps, activation)


def _ref_causal_short_conv(x, weight, activation, backend, cu_seqlens=None, cp_context=None):
    assert activation == "silu" and cp_context is None
    return _ref_conv(x, weight, cu_seqlens)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _bridge():
    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    bridge = Qwen38NextBridge.__new__(Qwen38NextBridge)
    bridge.hf_config = SimpleNamespace(text_config=_text_config())
    bridge.dtype = None
    bridge.make_vocab_size_divisible_by = None
    return bridge


def _export_args(sharded: bool):
    """What the raw export reads (the offline torch_dist converters pass such a namespace too)."""
    backend = SimpleNamespace(kv_channels=HEAD_K, hidden_size=HIDDEN, num_attention_heads=2, num_query_groups=1)
    return SimpleNamespace(qwen38_gdn_head_sharded=sharded, hf_checkpoint="/qwen38-next", backend=backend)


def _raw_export(full_mcore: dict, sharded: bool) -> dict:
    """The raw-mode converter on the gathered Megatron tensors."""
    from miles.backends.megatron_utils.megatron_to_hf import linear_attn_layout
    from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf

    out = {}
    original = linear_attn_layout.gdn_heads_of
    linear_attn_layout.gdn_heads_of = lambda path: linear_attn_layout.gdn_heads(_text_config())
    from miles.backends.megatron_utils.megatron_to_hf import qwen3_5 as qwen3_5_export

    qwen3_5_export.gdn_heads_of = linear_attn_layout.gdn_heads_of
    try:
        for name, tensor in full_mcore.items():
            for hf_name, hf_tensor in convert_qwen3_8_next_to_hf(
                _export_args(sharded), f"module.module.{LAYER}.{name}", tensor
            ):
                assert hf_name.startswith(HF_LAYER + "."), hf_name
                out[hf_name[len(HF_LAYER) + 1 :]] = hf_tensor
    finally:
        linear_attn_layout.gdn_heads_of = original
        qwen3_5_export.gdn_heads_of = original
    return out


# Megatron partition dim of each head-sharded tensor (norm.weight is replicated)
PARTITION_DIM = {name: 0 for name in NAMES} | {"out_proj.weight": 1, "norm.weight": None}


def _local_shape(name, full_shape, tp):
    dim = PARTITION_DIM[name]
    if dim is None:
        return full_shape
    return tuple(s // tp if i == dim else s for i, s in enumerate(full_shape))


# ---------------------------------------------------------------- (a) HF -> Megatron -> HF, emulated TP ranks


def _round_trip(tp: int, hf_dtype) -> None:
    from miles.backends.megatron_utils.megatron_to_hf.linear_attn_layout import LINEAR_ATTN_LAYOUT_KEY

    bridge = _bridge()

    holder = nn.Module()
    holder.linear_attn = nn.Module()
    holder.linear_attn.register_buffer(LINEAR_ATTN_LAYOUT_KEY, torch.ones(1))
    bridge._select_gdn_layout([holder])
    assert bridge._GDN_GROUP_MAJOR, "the bridge did not see the head-sharded layout marker"

    hf = _hf_weights(hf_dtype)
    full_mcore = {}
    bridge_hf = {}
    for name in NAMES:
        mcore_name = f"{LAYER}.{name}"
        assert bridge._weight_name_mapping_mcore_to_hf(mcore_name) == [f"{HF_LAYER}.{name}"]
        mcore = bridge._weight_to_mcore_format(mcore_name, [hf[name]])
        param = torch.empty(_local_shape(name, tuple(mcore.shape), tp), dtype=mcore.dtype)
        shards = bridge._weight_split_across_tp(mcore_name, mcore, param, tp)
        assert len(shards) == tp and all(s.shape == param.shape for s in shards), name
        dim = PARTITION_DIM[name]
        gathered = shards[0] if dim is None or tp == 1 else torch.cat(list(shards), dim=dim)
        full_mcore[name] = gathered
        names, tensors = bridge._weight_to_hf_format(mcore_name, gathered)
        assert names == [f"{HF_LAYER}.{name}"]
        bridge_hf[name] = tensors[0]

    raw_hf = _raw_export(full_mcore, sharded=True)
    for name in NAMES:
        for exported, path in ((raw_hf[name], "raw"), (bridge_hf[name], "bridge")):
            exported = exported.to(hf[name].dtype)
            assert exported.shape == hf[name].shape, (path, name)
            assert torch.equal(
                _bytes(exported), _bytes(hf[name])
            ), f"{path} export of {name} is not bitwise the HF tensor at TP={tp}"
    assert full_mcore["A_log"].dtype == torch.float32

    # positive control: q/k/v and conv are permuted in Megatron, so the replicated (flat) export would scramble them
    scrambled = _raw_export(full_mcore, sharded=False)
    for name in ("in_proj_qkv.weight", "conv1d.weight"):
        assert not torch.equal(scrambled[name].to(hf[name].dtype), hf[name]), name


def _bytes(t):
    return t.contiguous().reshape(-1).view(torch.uint8)


def _round_trip_worker(rank, world_size, port, tp, hf_dtype):
    _install_cpu_stubs()
    _round_trip(tp, hf_dtype)


@pytest.mark.parametrize("tp", [1, 2, 8])
@pytest.mark.parametrize("hf_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_hf_to_megatron_to_hf_round_trip_is_bitwise(tp, hf_dtype):
    """One spawned process (the CPU stubs replace modules process-wide) emulating ``tp`` ranks."""
    run_multiprocess(partial(_round_trip_worker, tp=tp, hf_dtype=hf_dtype), world_size=1)


# ---------------------------------------------------------------- (b) forward / backward, gloo TP group


def _args(sequence_parallel):
    return SimpleNamespace(
        hf_checkpoint="/qwen38-next",
        linear_attention_backend="fla",
        linear_attention_input_norm="hf",
        allgather_cp=False,
        backend=SimpleNamespace(sequence_parallel=sequence_parallel),
    )


def _megatron_config(sequence_parallel):
    return SimpleNamespace(
        hidden_size=HIDDEN,
        params_dtype=torch.float64,
        sequence_parallel=sequence_parallel,
        init_method=nn.init.xavier_uniform_,
        output_layer_init_method=nn.init.xavier_uniform_,
    )


def _patch_configs(output_gate_type):
    from miles_plugins.models import hf_attention, qwen3_5

    qwen3_5.load_hf_config = lambda path: _text_config(output_gate_type)
    hf_attention.load_hf_config = lambda path: _text_config("sigmoid")


def _build_replicated(sequence_parallel, hf):
    from miles_plugins.models.qwen3_8_next import gdn

    _patch_configs("sigmoid")
    layer = gdn.ReplicatedGDNAttention(_args(sequence_parallel), _megatron_config(sequence_parallel), layer_number=1)
    # Qwen38NextLinearAttention: the attention HC's hc_norm stands in for the input layernorm
    layer.input_layernorm = nn.Identity()
    layer.double()
    with torch.no_grad():
        for name, param in layer.linear_attn.named_parameters():
            param.copy_(hf[name])
    return layer


def _build_sharded(sequence_parallel, output_gate_type, hf):
    """Loaded like the bridge loads it: its real HF -> Megatron conversion and TP split."""
    from miles_plugins.models.qwen3_8_next import gdn

    _patch_configs(output_gate_type)
    layer = gdn.HeadShardedGDNAttention(_args(sequence_parallel), _megatron_config(sequence_parallel), layer_number=1)
    bridge = _bridge()
    bridge._select_gdn_layout([layer])
    assert bridge._GDN_GROUP_MAJOR
    tp = dist.get_world_size()
    params = dict(layer.linear_attn.named_parameters())
    assert set(params) == set(NAMES), sorted(params)
    with torch.no_grad():
        for name in NAMES:
            mcore_name = f"{LAYER}.{name}"
            mcore = bridge._weight_to_mcore_format(mcore_name, [hf[name]])
            params[name].copy_(bridge._weight_split_across_tp(mcore_name, mcore, params[name], tp)[dist.get_rank()])
    return layer


def _run(layer, x, dy, sequence_parallel):
    rank, tp = dist.get_rank(), dist.get_world_size()
    own = slice(rank * SEQ // tp, (rank + 1) * SEQ // tp) if sequence_parallel else slice(None)
    x_local = x[own].clone().requires_grad_()
    out, bias = layer(
        x_local, None, packed_seq_params=SimpleNamespace(cu_seqlens_q=torch.tensor(CU_BOUNDS, dtype=torch.int32))
    )
    assert bias is None
    out.backward(dy[own])
    if not sequence_parallel:
        return out.detach(), x_local.grad
    full_out, full_dx = torch.empty_like(x), torch.empty_like(x)
    dist.all_gather_into_tensor(full_out, out.detach().contiguous())
    dist.all_gather_into_tensor(full_dx, x_local.grad.contiguous())
    return full_out, full_dx


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _equivalence_worker(rank, world_size, port, sequence_parallel, sharded_gate):
    init_gloo(rank, world_size, port=port)
    try:
        _install_cpu_stubs()
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
        hf = _hf_weights()
        replicated = _build_replicated(sequence_parallel, hf)
        sharded = _build_sharded(sequence_parallel, sharded_gate, hf)
        bridge = _bridge()
        bridge._select_gdn_layout([replicated])
        assert bridge._GDN_GROUP_MAJOR == (), "the replicated GDN must keep HF's flat rows"
        assert isinstance(sharded.input_layernorm, nn.Identity)

        gen = torch.Generator().manual_seed(7)
        x = torch.randn(SEQ, 1, HIDDEN, dtype=torch.float64, generator=gen)
        dy = torch.randn(SEQ, 1, HIDDEN, dtype=torch.float64, generator=gen)
        out_ref, dx_ref = _run(replicated, x, dy, sequence_parallel)
        out_new, dx_new = _run(sharded, x, dy, sequence_parallel)

        if sharded_gate != "sigmoid":
            # positive control: the wrong gate activation is visible at this tolerance
            assert _rel(out_new, out_ref) > 1e-3
            return
        errors = {"out": _rel(out_new, out_ref), "dx": _rel(dx_new, dx_ref)}

        # gather the sharded grads to the full Megatron layout, export them to HF layout like weights
        full = {}
        for name, param in sharded.linear_attn.named_parameters():
            dim = PARTITION_DIM[name]
            grad = param.grad
            if dim is not None and world_size > 1:
                parts = [torch.empty_like(grad) for _ in range(world_size)]
                dist.all_gather(parts, grad.contiguous())
                grad = torch.cat(parts, dim=dim)
            full[name] = grad
        hf_grads = _raw_export(full, sharded=True)
        ref_grads = {name: p.grad for name, p in replicated.linear_attn.named_parameters()}
        for name in NAMES:
            errors[name] = _rel(hf_grads[name].double(), ref_grads[name].double())
        print(
            f"[gdn-head-sharded] TP={world_size} SP={sequence_parallel} rank={rank} max rel err {max(errors.values()):.2e}"
        )
        bad = {k: v for k, v in errors.items() if not v <= TOL}
        assert not bad, f"rank {rank} TP={world_size} SP={sequence_parallel}: {bad} (all: {errors})"
        assert ref_grads["in_proj_qkv.weight"].abs().max() > 0 and ref_grads["A_log"].abs().max() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [1, 2, 8])
@pytest.mark.parametrize("sequence_parallel", [False, True], ids=["no-sp", "sp"])
def test_head_sharded_matches_replicated_forward_and_backward(tp, sequence_parallel):
    if tp == 1 and sequence_parallel:
        pytest.skip("sequence parallelism needs TP > 1")
    run_multiprocess(partial(_equivalence_worker, sequence_parallel=sequence_parallel, sharded_gate="sigmoid"), tp)


def test_the_gate_activation_is_checked():
    """Positive control: the head-sharded layer built with Qwen3.5's silu gate differs from the replicated one."""
    run_multiprocess(partial(_equivalence_worker, sequence_parallel=True, sharded_gate=None), 2)


# ---------------------------------------------------------------- spec switch


@pytest.mark.parametrize("flag", [False, True])
def test_spec_builds_the_head_sharded_layer_only_with_the_flag(flag, monkeypatch, gdn):
    import importlib

    from megatron.core.transformer.spec_utils import ModuleSpec

    qwen3_8_next = importlib.import_module("miles_plugins.models.qwen3_8_next.qwen3_8_next")
    text = SimpleNamespace(
        layer_types=["linear_attention", "indexed_attention"],
        hc_count=4,
        hc_lowrank=2,
        ple_layer_ids=[],
        ple_embed_dim=8,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=17,
        split_ngram_parts=1,
        ple_conv_kernel_size=3,
        eos_token_id=7,
        indexer_budget=11,
        indexer_compress_ratio=2,
        indexer_n_heads=2,
        indexer_head_dim=4,
        indexer_kv_heads=1,
    )

    def layer_spec():
        attention = ModuleSpec(module=nn.Module, params={}, submodules=SimpleNamespace(linear_qkv=None))
        return ModuleSpec(module=nn.Module, submodules=SimpleNamespace(self_attention=attention))

    block = SimpleNamespace(layer_specs=[layer_spec(), layer_spec()])
    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: text)
    monkeypatch.setattr(qwen3_8_next, "get_gpt_decoder_block_spec", lambda *a, **k: block)
    monkeypatch.setattr(qwen3_8_next, "get_num_layers_to_build", lambda *a, **k: 2)
    monkeypatch.setattr(qwen3_8_next, "get_transformer_layer_offset", lambda *a, **k: 0)
    args = SimpleNamespace(
        hf_checkpoint="/qwen38-next",
        qwen38_gdn_head_sharded=flag,
        backend=SimpleNamespace(num_experts=4, position_embedding_type="rope"),
    )
    config = SimpleNamespace(num_layers=2, moe_layer_freq=[1, 1], pipeline_model_parallel_layout=None)
    spec = qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)

    gdn_module = spec.layer_specs[0].submodules.self_attention.module
    expected = qwen3_8_next.HeadShardedGDNAttention if flag else qwen3_8_next.Qwen38NextLinearAttention
    assert gdn_module is expected
    assert spec.layer_specs[1].submodules.self_attention.module is qwen3_8_next.Qwen38NextAttention
