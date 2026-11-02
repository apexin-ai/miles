"""CPU test: the PLE layer's CP x SP composite shard must give the same gradients as one rank.

New-baseline architecture (``_ple_shard`` / ``_gather_tails`` / ``_apply_ple_local_window``):
CP slices the global packed stream into contiguous pieces; SP then slices the piece a CP rank
holds. A rank's position in the composed shard is ``cp_rank * sp_size + sp_rank`` out of
``cp_size * sp_size`` -- ``_gather_tails`` gathers the halo first over the TP group (inner),
then over the CP group (outer), relying on that composed numbering to land each rank's tail at
the right index. That ordering claim was independently verified against real gloo collectives
in ``check_gather_order.py`` (TP2/CP2 and TP4/CP2, zero mismatches) -- this test is the next
layer up: does the actual ``Qwen38NextPLEHyperConnection.forward`` under that composition
produce the same gradients as computing PLE on the whole sequence unsplit.

Why this can't reuse the old (712aebd8) CPU PLE test: that test's SP-only grad check has since
moved to ``tests/fast-gpu/test_qwen3_8_next_ple_sp_grads.py`` with real Triton/TE/NCCL -- no CPU
fallback remains in the repository for the full HC-layer level. This rebuilds the same fp64
reference-stub technique (plain duplicated Linear for TELinear, a toy lookup table for the
frozen n-gram embedding, an fp64 torch reference for ``ple_gate_conv``) against the *current*
class, which did not exist when the old test was written.

Why manual process groups, not ``parallel_state.initialize_model_parallel``: that call requires
a CUDA device and raises ``ValueError: Expected a cuda device`` with none present. Two orthogonal
2-rank axes are built directly with ``dist.new_group`` (global rank = cp_rank * tp_size +
tp_rank), and every name ``hyper_connection.py`` imported *by reference* at module load time
(``get_context_parallel_group``, the three ``get_tensor_model_parallel_*``, ``cp_enabled``,
``cp_rank_of``, ``cp_world_size``) is overwritten in its own namespace -- patching the defining
module (``parallel_state`` / ``ops.cp``) would miss it, since the importing module already holds
its own bound copy of the original function object.
"""

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import find_free_port, init_gloo

# measured on the qwen38 pod's CPU for the 4-rank (TP2xCP2) + 2-rank (CP2, no SP) cases.
register_cpu_ci(est_time=180, suite="stage-b-cpu", labels=["miles-plugin"])

N_STREAMS, HIDDEN, LOWRANK = 2, 8, 4
NGRAM_SIZE, HEADS_PER_NGRAM, EMBED_DIM, CONV_KERNEL = 3, 2, 8, 4  # conv dilation = ngram size = 3
LOOKBACK = (CONV_KERNEL - 1) * NGRAM_SIZE
TABLE_ROWS = 64
EPS = 1e-6
SEQ = 48  # must be divisible by every (tp_size * cp_size) this file tests
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


def _ple_gate_conv_reference(hc, key, value, wk, wq, wc, convw, n, eps, dil, cu, deterministic_conv_grad=True):
    T, C = hc.shape[0], hc.shape[1] // n
    kn = _grouped_rmsnorm(key, wk, n, eps).reshape(T, n, C)
    qn = _grouped_rmsnorm(hc, wq, n, eps).reshape(T, n, C)
    score = (kn * qn).sum(dim=-1, keepdim=True) / math.sqrt(C)
    gate = torch.sigmoid(score.abs().clamp_min(1e-6).sqrt() * score.sign())
    gated = (gate * value.unsqueeze(-2)).flatten(-2)
    conv = F.silu(_causal_depthwise_conv(_grouped_rmsnorm(gated, wc, n, eps), convw, dil, cu))
    return gated + conv


def _hc_mix_inject_reference(x, weight, w_down, w_up, w_inject, n, eps, matmul_precision=None):
    normed = _grouped_rmsnorm(x, weight, n, eps)
    hidden = x.shape[-1] // n
    gate = torch.sigmoid(F.linear(F.silu(F.linear(normed, w_down)) / n, w_up))
    mixed = (gate.unflatten(-1, (n, hidden)) * normed.unflatten(-1, (n, hidden))).mean(dim=-2)
    return mixed, 2 * torch.sigmoid(F.linear(normed, w_inject) / n)


class _DuplicatedLinear(torch.nn.Linear):
    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)
        for param in self.parameters():
            param.sequence_parallel = config.sequence_parallel
            param.tensor_model_parallel = False

    def forward(self, x):
        return super().forward(x), None


class _HostTable(torch.nn.Module):
    """The frozen n-gram table, unsharded: what every rank holds after its (no-op, TP=1) all-reduce."""

    def __init__(self, config, layer_number, tp_group=None):
        super().__init__()
        heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
        gen = torch.Generator().manual_seed(1234)
        self.register_buffer("table", torch.randn(TABLE_ROWS, EMBED_DIM // heads, dtype=torch.float64, generator=gen))

    def forward(self, ids):
        return self.table[ids].flatten(start_dim=-2)


def _patch_for_cpu(hyper_connection, ple, groups, mutate=None):
    """Process-local patches. ``groups`` is ``{"tp": (group, rank, size), "cp": (group, rank, size)}``.

    ``mutate`` selectively breaks one specific claim, for the must-fail test:
      "order"  -- un-reverses ``_gather_tails``' group order (TP/CP swapped)
      "index"  -- swaps ``_ple_shard``'s composed-index formula (sp_rank*cp_size+cp_rank)
    """
    import megatron.core.tensor_parallel.random as mp_random

    torch.cuda.current_device = lambda: "cpu"
    ple.ple_gate_conv = _ple_gate_conv_reference
    ple.TELinear = _DuplicatedLinear
    ple.Qwen38NextFrozenNGramEmbedding = _HostTable
    hyper_connection.hc_mix_inject = _hc_mix_inject_reference
    mp_random._get_all_rng_states = lambda: ()
    mp_random._set_all_rng_states = lambda *states: None

    tp_group, tp_rank, tp_size = groups["tp"]
    cp_group, cp_rank, cp_size = groups["cp"]
    hyper_connection.get_tensor_model_parallel_group = lambda: tp_group
    hyper_connection.get_tensor_model_parallel_rank = lambda: tp_rank
    hyper_connection.get_tensor_model_parallel_world_size = lambda: tp_size
    hyper_connection.get_context_parallel_group = lambda: cp_group
    hyper_connection.cp_world_size = lambda: cp_size
    hyper_connection.cp_rank_of = lambda: cp_rank
    hyper_connection.cp_enabled = lambda config: cp_size > 1

    if mutate == "order":
        # the un-mutated code reverses groups (TP first, then CP); this drops the reversal
        def _gather_tails_unreversed(tail, groups_arg):
            for group in groups_arg:  # no reversed()
                tail = hyper_connection.gather_from_sequence_parallel_region(
                    tail, tensor_parallel_output_grad=True, group=group
                )
            return tail

        hyper_connection.Qwen38NextPLEHyperConnection._gather_tails = staticmethod(_gather_tails_unreversed)
    elif mutate == "index":
        orig_ple_shard = hyper_connection.Qwen38NextPLEHyperConnection._ple_shard

        def _ple_shard_swapped(self, hidden_states):
            rank, size = orig_ple_shard(self, hidden_states)
            # orig composes as cp_rank*sp_size+sp_rank; recover the two indices and swap them
            if size <= 1:
                return rank, size
            sp_size_local = tp_size if cp_size > 1 else size
            sp_r, cp_r = rank % sp_size_local, rank // sp_size_local
            return sp_r * cp_size + cp_r, size

        hyper_connection.Qwen38NextPLEHyperConnection._ple_shard = _ple_shard_swapped
    elif mutate is not None:
        raise ValueError(f"unknown mutate={mutate!r}")


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


def _build_layer(hyper_connection):
    layer = hyper_connection.Qwen38NextPLEHyperConnection(_config(), layer_number=1)
    layer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
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


def _run(ple_module, layer, x, ngram_ids, cu, upstream):
    x = x.clone().requires_grad_()
    ple_module.publish_ple_batch(ngram_ids, cu)
    aggregated, h_res, h_post, residual = layer(x)
    assert h_res is None
    torch.autograd.backward((aggregated, h_post, residual), upstream)
    ple_module.clear_ple_batch()
    return x.grad


def _worker(rank, world_size, port, tp_size, cp_size, cu_bounds, mutate):
    init_gloo(rank, world_size, port=port)
    try:
        from miles_plugins.models.qwen3_8_next import hyper_connection
        from miles_plugins.models.qwen3_8_next.ops import ple

        # global rank = cp_rank * tp_size + tp_rank (TP varies fastest)
        tp_rank, cp_rank = rank % tp_size, rank // tp_size

        # dist.new_group() is collective: EVERY rank in the default group must call it with
        # the same args, even ranks that are not members of the resulting group -- a rank
        # that only calls new_group() for the ONE subset it belongs to, skipping the others,
        # leaves those other calls permanently unmatched, and every rank waiting on one of
        # them hangs forever (no error, no shape mismatch, just silence). This bit a 2-rank
        # topology by luck: with trivial singleton groups ([0], [1]) nothing ever actually
        # communicates on them, so the malformed construction protocol went unnoticed; the
        # 4-rank case (two real 2-member subsets) deadlocked on exactly this.
        tp_group = cp_group = None
        for base in range(0, world_size, tp_size):
            members = list(range(base, base + tp_size))
            g = dist.new_group(members)
            if rank in members:
                tp_group = g
        for base in range(tp_size):
            members = list(range(base, world_size, tp_size))
            g = dist.new_group(members)
            if rank in members:
                cp_group = g
        groups = {"tp": (tp_group, tp_rank, tp_size), "cp": (cp_group, cp_rank, cp_size)}

        _patch_for_cpu(hyper_connection, ple, groups, mutate=mutate)

        x, ngram_ids, cu, upstream = _inputs(cu_bounds)
        layer = hyper_connection.Qwen38NextPLEHyperConnection(_config(), layer_number=1)
        layer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for _, param in sorted(layer.named_parameters()):
                param.copy_(0.3 * torch.randn(param.shape, dtype=param.dtype, generator=gen))

        # unsplit reference on every rank: force BOTH axes off, every rank holds the whole seq.
        # cp_enabled=False alone is not enough -- _ple_shard still reads the real tp_size via
        # get_tensor_model_parallel_world_size() whenever config.sequence_parallel is set
        # (always, in _config()), so the reference pass would otherwise treat the whole SEQ-row
        # tensor as if it were one rank's shard of a larger (SEQ * tp_size)-token sequence.
        real_tp_world_size = hyper_connection.get_tensor_model_parallel_world_size
        hyper_connection.cp_enabled = lambda config: False
        hyper_connection.get_tensor_model_parallel_world_size = lambda: 1
        ref_dx = _run(ple, layer, x, ngram_ids, cu, upstream)
        ref_grads = {name: param.grad.clone() for name, param in layer.named_parameters()}
        layer.zero_grad(set_to_none=True)

        total = tp_size * cp_size
        local = SEQ // total
        shard_rank = cp_rank * tp_size + tp_rank

        # positive control (still unsharded semantics -- tp forced to 1): zero the loss on
        # everything up to and including THIS rank's own shard start, so the only surviving
        # gradient the halo rows [own.start-LOOKBACK, own.start) can get is the conv reading
        # backward from own.start itself. Must be keyed to THIS rank's own boundary, not a
        # single global midpoint -- with 4 shards (tp2-cp2) only one rank's boundary happens
        # to sit at the sequence midpoint, so a shared cutoff silently zeroes the probe region
        # for the other three and makes the control pass for the wrong reason (nothing to find,
        # not "found and it's zero").
        own = slice(shard_rank * local, (shard_rank + 1) * local)
        if own.start > 0:
            later_only = tuple(torch.cat([torch.zeros_like(u[: own.start]), u[own.start :]]) for u in upstream)
            cross = _run(ple, layer, x, ngram_ids, cu, later_only)[own.start - LOOKBACK : own.start]
            assert cross.abs().amax() > 1e3 * TOL, "the conv does not reach across the shard boundary"
        layer.zero_grad(set_to_none=True)
        hyper_connection.get_tensor_model_parallel_world_size = real_tp_world_size

        # the real sharded path
        hyper_connection.cp_enabled = lambda config: cp_size > 1
        local_upstream = tuple(u[own] for u in upstream)
        local_dx = _run(ple, layer, x[own], ngram_ids, cu, local_upstream)

        dx = torch.empty_like(x)
        dist.all_gather_into_tensor(dx.view(SEQ, -1), local_dx.contiguous().view(local, -1), group=dist.group.WORLD)
        token_err = (dx - ref_dx).abs().amax(dim=(1, 2))
        bad = [t for t in range(SEQ) if token_err[t] > TOL]
        assert not bad, f"rank {rank}: {len(bad)} tokens differ, worst {token_err.amax().item():.3e} at {bad[:6]}"

        for name, param in layer.named_parameters():
            if param.grad is None:
                continue
            # A replicated param's grad needs summing over the FULL composite domain, not
            # just one axis -- picking cp_group XOR tp_group is exactly right for the two
            # single-axis cases (there, WORLD already equals that one group) but silently
            # undercounts for tp2-cp2, where the param is replicated across both and only
            # one axis' worth of ranks' contributions got summed.
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            err = (param.grad - ref_grads[name]).abs().amax().item()
            assert err <= TOL, f"rank {rank}: param {name} grad differs by {err:.3e}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("cu_bounds", [None, [0, SEQ], [0, 20, SEQ]], ids=["unpacked", "one-sequence", "packed"])
@pytest.mark.parametrize("tp_size,cp_size", [(1, 2), (2, 1), (2, 2)], ids=["cp2-nosp", "sp2-nocp", "tp2-cp2"])
def test_ple_composite_shard_grads_match_unsplit(tp_size, cp_size, cu_bounds) -> None:
    world_size = tp_size * cp_size
    port = find_free_port()
    torch.multiprocessing.spawn(
        partial(_worker, world_size=world_size, port=port, tp_size=tp_size, cp_size=cp_size, cu_bounds=cu_bounds, mutate=None),
        args=(),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("mutate", ["order", "index"])
def test_mutated_composition_is_detected(mutate) -> None:
    """Without the real reversed-group-order / composed-index logic, the test above must fail.

    This is the falsification check the equivalence test itself needs: a passing test that
    cannot fail under a wrong implementation has no teeth. Two outcomes both count as
    "detected": a rank's assertion firing (caught as an exception from join()), or every rank
    hanging in a collective because the broken order/index made different ranks call it with
    incompatible shapes or counts -- a deadlock is exactly as much evidence that correctness
    depends on the real logic as a clean failure is, and `ctx.join(timeout=...)` returning
    False (no exception, just stuck) is how that case is observed. Reaching the end with every
    rank having exited cleanly is the one outcome that means the mutation went undetected.
    """
    world_size = 4  # tp=2, cp=2: the only topology where order/index mutations are observable
    port = find_free_port()
    ctx = torch.multiprocessing.spawn(
        partial(_worker, world_size=world_size, port=port, tp_size=2, cp_size=2, cu_bounds=None, mutate=mutate),
        args=(),
        nprocs=world_size,
        join=False,
    )
    try:
        finished = ctx.join(timeout=30)
    except Exception as exc:  # noqa: BLE001 -- any rank's AssertionError surfaces as a spawn exception
        print(f"mutate={mutate!r}: correctly detected -- {type(exc).__name__}: {exc}")
        return
    if not finished:
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
        print(f"mutate={mutate!r}: correctly detected -- ranks deadlocked (30s timeout)")
        return
    pytest.fail(f"mutate={mutate!r}: all ranks exited cleanly -- the test did not detect the broken composition")

