"""CPU test: the PLE layer under sequence parallelism must give the same gradients as one rank.

The PLE causal conv looks back (kernel - 1) * dilation tokens, so the loss on rank r reaches the
last tokens of rank r - 1's slice. Each TP rank runs the real ``Qwen38NextPLEHyperConnection``
forward (Megatron's real all-gather and its backward) over a gloo group, and the PLE params are
reduced by Megatron's real ``_allreduce_non_tensor_model_parallel_grads``. The Triton kernels
cannot run on CPU, so ``ple_gate_conv`` and ``hc_mix_inject`` are replaced by the fp64 torch
references that tests/fast-gpu/test_qwen3_8_next_ops.py checks the kernels against; TELinear
(duplicated) and the host n-gram table are replaced by plain torch modules. Everything is fp64,
so every token's input gradient and every param gradient must match the unsplit run to 1e-10.
"""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

# 8 spawned gloo cases; about 150 s on an 8-thread CPU
register_cpu_ci(est_time=150, suite="stage-b-cpu", labels=["miles-plugin"])

N_STREAMS, HIDDEN, LOWRANK = 2, 8, 4
NGRAM_SIZE, HEADS_PER_NGRAM, EMBED_DIM, CONV_KERNEL = 3, 2, 8, 4  # conv dilation = ngram size = 3
LOOKBACK = (CONV_KERNEL - 1) * NGRAM_SIZE
TABLE_ROWS = 64
EPS = 1e-6
SEQ = 48
TOL = 1e-10


def _grouped_rmsnorm(x, weight, n, eps):
    xg = x.unflatten(-1, (n, x.shape[-1] // n))
    xn = (xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + eps)).flatten(-2)
    return xn * (1.0 + weight)


def _causal_depthwise_conv(x, weight, dilation, cu_seqlens):
    channels, _, kernel = weight.shape
    pad = (kernel - 1) * dilation

    def _conv(seq):
        h = seq.transpose(0, 1).unsqueeze(0)
        h = F.conv1d(F.pad(h, (pad, 0)), weight, groups=channels, dilation=dilation)
        return h.squeeze(0).transpose(0, 1)

    if cu_seqlens is None:
        return _conv(x)
    bounds = cu_seqlens.tolist()
    return torch.cat([_conv(x[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True) if hi > lo])


def _ple_gate_conv_reference(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu, deterministic_conv_grad=False):
    """``ple_reference`` of tests/fast-gpu/test_qwen3_8_next_ops.py, in the input dtype (autograd: no atomics)."""
    T, C = hc.shape[0], hc.shape[1] // n
    kn = _grouped_rmsnorm(key, wk, n, eps).reshape(T, n, C)
    qn = _grouped_rmsnorm(hc, wq, n, eps).reshape(T, n, C)
    score = (kn * qn).sum(dim=-1, keepdim=True) / math.sqrt(C)
    gate = torch.sigmoid(score.abs().clamp_min(1e-6).sqrt() * score.sign())
    gated = (gate * value.unsqueeze(-2)).flatten(-2)
    conv = F.silu(_causal_depthwise_conv(_grouped_rmsnorm(gated, wc, n, eps), convw, dil, cu))
    return gated + conv


def _hc_mix_inject_reference(x, weight, w_down, w_up, w_inject, n, eps, matmul_precision="fp32"):
    """``hc_mix`` + ``hc_inject_gate`` of tests/fast-gpu/test_qwen3_8_next_ops.py."""
    assert matmul_precision == "fp32"
    normed = _grouped_rmsnorm(x, weight, n, eps)
    hidden = x.shape[-1] // n
    gate = torch.sigmoid(F.linear(F.silu(F.linear(normed, w_down) / n), w_up))
    mixed = (gate.unflatten(-1, (n, hidden)) * normed.unflatten(-1, (n, hidden))).mean(dim=-2)
    return mixed, 2 * torch.sigmoid(F.linear(normed, w_inject) / n)


class _DuplicatedLinear(torch.nn.Linear):
    """TELinear(parallel_mode="duplicated"): marks its weight like Megatron's TELinear does."""

    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)
        for param in self.parameters():
            param.sequence_parallel = config.sequence_parallel
            param.tensor_model_parallel = False

    def forward(self, x):
        return super().forward(x), None


class _HostTable(torch.nn.Module):
    """The frozen n-gram table, unsharded: what every TP rank holds after its all-reduce."""

    def __init__(self, config, layer_number, tp_group=None):
        super().__init__()
        heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
        gen = torch.Generator().manual_seed(1234)
        self.register_buffer("table", torch.randn(TABLE_ROWS, EMBED_DIM // heads, dtype=torch.float64, generator=gen))

    def forward(self, ids):
        return self.table[ids].flatten(start_dim=-2)


def _patch_for_cpu():
    """Process-local patches (each test runs in its own spawned processes)."""
    import megatron.core.tensor_parallel.random as mp_random

    from miles_plugins.models.qwen3_8_next import hyper_connection
    from miles_plugins.models.qwen3_8_next.ops import ple

    torch.cuda.current_device = lambda: "cpu"
    ple.ple_gate_conv = _ple_gate_conv_reference
    ple.TELinear = _DuplicatedLinear
    ple.Qwen38NextFrozenNGramEmbedding = _HostTable
    hyper_connection.hc_mix_inject = _hc_mix_inject_reference
    # CheckpointFunction saves and restores CUDA RNG states; there are none on CPU
    mp_random._get_all_rng_states = lambda: ()
    mp_random._set_all_rng_states = lambda *states: None


def _config():
    return SimpleNamespace(
        num_residual_streams=N_STREAMS,
        hidden_size=HIDDEN,
        layernorm_epsilon=EPS,
        qwen3_8_next_hc_lowrank=LOWRANK,
        params_dtype=torch.float64,
        sequence_parallel=True,
        qk_layernorm=False,
        qwen3_8_next_ple_embed_dim=EMBED_DIM,
        qwen3_8_next_ngram_size=NGRAM_SIZE,
        qwen3_8_next_ple_conv_kernel_size=CONV_KERNEL,
        init_method=torch.nn.init.xavier_uniform_,
    )


def _build_layer():
    from miles_plugins.models.qwen3_8_next.hyper_connection import Qwen38NextPLEHyperConnection

    layer = Qwen38NextPLEHyperConnection(_config(), layer_number=1)
    layer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        # the norm/conv params start at zero; a zero conv would hide the cross-token path
        for _, param in sorted(layer.named_parameters()):
            param.copy_(0.3 * torch.randn(param.shape, dtype=param.dtype, generator=gen))
    return layer


def _inputs(cu_bounds):
    gen = torch.Generator().manual_seed(7)
    width = N_STREAMS * HIDDEN
    heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
    x = torch.randn(SEQ, 1, width, dtype=torch.float64, generator=gen)
    ngram_ids = torch.randint(0, TABLE_ROWS, (SEQ, heads), generator=gen)
    upstream = (
        torch.randn(SEQ, 1, HIDDEN, dtype=torch.float64, generator=gen),
        torch.randn(SEQ, 1, N_STREAMS, dtype=torch.float64, generator=gen),
        torch.randn(SEQ, 1, width, dtype=torch.float64, generator=gen),
    )
    cu = None if cu_bounds is None else torch.tensor(cu_bounds, dtype=torch.int32)
    return x, ngram_ids, cu, upstream


def _run(layer, x, ngram_ids, cu, upstream, recompute):
    """Forward + backward of the layer on ``x``; returns d loss / d x."""
    from megatron.core import tensor_parallel

    from miles_plugins.models.qwen3_8_next.ops.ple import clear_ple_batch, publish_ple_batch

    x = x.clone().requires_grad_()
    publish_ple_batch(ngram_ids, cu)

    def body(hidden_states):
        aggregated, h_res, h_post, residual = layer(hidden_states)
        assert h_res is None
        return aggregated, h_post, residual

    outputs = tensor_parallel.checkpoint(body, False, x) if recompute else body(x)
    # the recompute pass must find its batch in the FIFO, not in the side channel
    clear_ple_batch()
    torch.autograd.backward(outputs, upstream)
    return x.grad


def _boundary_tokens(tp):
    local = SEQ // tp
    return sorted({t for r in range(1, tp) for t in range(r * local - LOOKBACK, r * local)})


def _worker(rank, world_size, port, cu_bounds, recompute):
    init_gloo(rank, world_size, port=port)
    try:
        _patch_for_cpu()
        from megatron.core import parallel_state
        from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        x, ngram_ids, cu, upstream = _inputs(cu_bounds)
        layer = _build_layer()
        names = [name for name, param in layer.named_parameters() if param.requires_grad]

        # unsplit reference on every rank
        layer.config.sequence_parallel = False
        ref_dx = _run(layer, x, ngram_ids, cu, upstream, recompute)
        ref_grads = {name: param.grad.clone() for name, param in layer.named_parameters()}
        layer.zero_grad(set_to_none=True)

        # positive control: the loss past a slice boundary does reach the tokens before it. The
        # conv input is RMSNorm(gate * value), so the gate's scale cancels and this path is only
        # eps-sized (~1e-6 here); still far above the fp64 tolerance.
        local = SEQ // world_size
        later_only = tuple(torch.cat([torch.zeros_like(u[:local]), u[local:]]) for u in upstream)
        cross = _run(layer, x, ngram_ids, cu, later_only, recompute)[local - LOOKBACK : local]
        assert cross.abs().amax() > 1e3 * TOL, "the conv does not reach across the slice boundary"
        layer.zero_grad(set_to_none=True)

        # the same layer split over TP ranks
        layer.config.sequence_parallel = True
        own = slice(rank * local, (rank + 1) * local)
        local_upstream = tuple(u[own] for u in upstream)
        local_dx = _run(layer, x[own], ngram_ids, cu, local_upstream, recompute)
        _allreduce_non_tensor_model_parallel_grads([layer], layer.config, tp_group)

        dx = torch.empty_like(x)
        dist.all_gather_into_tensor(dx, local_dx.contiguous(), group=tp_group)
        token_err = (dx - ref_dx).abs().amax(dim=(1, 2))
        bad = [t for t in range(SEQ) if token_err[t] > TOL]
        assert not bad, (
            f"input grad differs from the unsplit run at tokens {bad} "
            f"(slice-boundary tails: {_boundary_tokens(world_size)}); max err {token_err.max():.3e}"
        )
        for name in names:
            got = dict(layer.named_parameters())[name].grad
            err = (got - ref_grads[name]).abs().max().item()
            assert err <= TOL * max(1.0, ref_grads[name].abs().max().item()), f"{name} grad err {err:.3e}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("cu_bounds", [None, [0, 5, 31, SEQ]], ids=["one-sequence", "packed"])
@pytest.mark.parametrize("recompute", [False, True], ids=["no-recompute", "full-recompute"])
def test_ple_sequence_parallel_grads_match_unsplit(tp, cu_bounds, recompute):
    run_multiprocess(partial(_worker, cu_bounds=cu_bounds, recompute=recompute), world_size=tp)
