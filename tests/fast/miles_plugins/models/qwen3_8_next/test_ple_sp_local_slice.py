"""CPU tests: under SP the PLE layer computes only its own slice plus the conv's look-back halo.

``Qwen38NextPLEHyperConnection._apply_ple_local_window`` runs the PLE on the window
``[rank * local - lookback, (rank + 1) * local)`` instead of on the all-gathered sequence. These
tests check it against the full-sequence path and the unsplit layer with packed rows that end
inside a halo, rows shorter than the halo, rows that span whole slices, and a slice exactly as long
as the halo:

* ``window_cu_seqlens``: the window's segment bounds, by hand.
* Real Triton kernels (``TRITON_INTERPRET=1``), one process emulating every rank: the window's
  output, per-token key / value grads and assembled input grad equal the full-sequence ones
  bitwise; the param grads stay within the full-sequence path's error against fp64.
* The real layer over a gloo group (TP 8) with the fp64 torch references: input and param grads
  match the unsplit run to 1e-10, with and without full recompute.
* The real layer and the real kernels over gloo (TP 8, bf16): outputs and input grads of the window
  path equal the full-sequence path bitwise. The duplicated linears are row-wise here, so the GEMM
  shape cannot change a row's result; on GPU that is cuBLAS's choice (see the GPU smoke script).
"""

import math
import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess
from tests.fast.miles_plugins.models.qwen3_8_next import test_ple_sequence_parallel_grads as m001

# 1 + 4 spawned cases (8 gloo ranks each) under the Triton interpreter; about 200 s on 2 CPUs
register_cpu_ci(est_time=240, suite="stage-b-cpu", labels=["miles-plugin"])

LOOKBACK = m001.LOOKBACK  # (kernel 4 - 1) * dilation 3 = 9
TP = 8
# 8 slices of 12; halos are [12r - 9, 12r). Rows [5, 7) and [7, 8) are shorter than the halo and
# end inside rank 1's halo, 22 ends a row inside rank 2's halo, 24 is a slice boundary, [42, 90)
# spans four whole slices.
PACKED_96 = [0, 5, 7, 8, 14, 22, 24, 33, 41, 42, 90, 96]
# 8 slices of 9 = the halo: each halo is the whole previous slice
PACKED_72 = [0, 3, 9, 10, 17, 18, 40, 72]
CASES = [(96, PACKED_96), (96, None), (72, PACKED_72)]
CASE_IDS = ["packed-96", "one-sequence-96", "packed-72-slice-equals-halo"]


# ----------------------------------------------------------------------------------------------
# window_cu_seqlens


def test_window_cu_seqlens_shifts_and_clips_rows():
    from miles_plugins.models.qwen3_8_next.ops.ple import window_cu_seqlens

    cu = torch.tensor(PACKED_96, dtype=torch.int32)
    # rank 2 of 8: window [15, 36)
    got = window_cu_seqlens(cu, 96, 15, 21).tolist()
    assert got == [0, 0, 0, 0, 0, 0, 7, 9, 18, 21, 21, 21, 21, 21]
    # rank 0: window [-9, 12), the 9 filler rows are their own segment
    got = window_cu_seqlens(cu, 96, -9, 21).tolist()
    assert got == [0, 9, 14, 16, 17, 21, 21, 21, 21, 21, 21, 21, 21, 21]
    # no packing: one row
    assert window_cu_seqlens(None, 96, 15, 21).tolist() == [0, 0, 21, 21]
    assert window_cu_seqlens(None, 96, -9, 21).tolist() == [0, 9, 21, 21]
    assert window_cu_seqlens(cu, 96, 15, 21).dtype == torch.int32


# ----------------------------------------------------------------------------------------------
# real kernels, every rank emulated in one process


def _windows(seq, tp):
    local = seq // tp
    for rank in range(tp):
        start = rank * local - LOOKBACK
        yield rank, slice(rank * local, (rank + 1) * local), start, torch.arange(start, start + local + LOOKBACK) % seq


def _kernel_worker(_rank, _world, _port, seq, cu_bounds, dtype):
    os.environ["TRITON_INTERPRET"] = "1"
    from miles.kernels.embedding.ple_gate_conv import ple_gate_conv

    from miles_plugins.models.qwen3_8_next.ops.ple import window_cu_seqlens

    n, C, K, dil, eps = 4, 16, m001.CONV_KERNEL, m001.NGRAM_SIZE, 1e-6
    W = n * C
    gen = torch.Generator().manual_seed(11)
    hc, key = (torch.randn(seq, W, generator=gen).to(dtype) for _ in range(2))
    value = torch.randn(seq, C, generator=gen).to(dtype)
    params = [(0.3 * torch.randn(W, generator=gen)).to(dtype) for _ in range(3)]
    params.append((2.0 * torch.randn(W, 1, K, generator=gen)).to(dtype))
    dout = torch.randn(seq, W, generator=gen).to(dtype)
    cu = None if cu_bounds is None else torch.tensor(cu_bounds, dtype=torch.int32)

    def layer(state, k, v, cu_seqlens):
        leaves = [t.detach().requires_grad_() for t in (state, k, v, *params)]
        updated = leaves[0] + ple_gate_conv(*leaves[:3], *leaves[3:], n, eps, dil, cu_seqlens)
        return leaves, updated

    full_leaves, full_updated = layer(hc, key, value, cu)
    old_dx = torch.zeros_like(hc)
    new_dx = torch.zeros_like(hc)
    old_pgrads = [torch.zeros_like(p) for p in params]
    new_pgrads = [torch.zeros_like(p) for p in params]
    halo_grads = {}
    for rank, own, start, rows in _windows(seq, TP):
        upstream = torch.zeros_like(dout)
        upstream[own] = dout[own]
        old = torch.autograd.grad(full_updated, full_leaves, upstream, retain_graph=True)

        leaves, updated = layer(hc[rows], key[rows], value[rows], window_cu_seqlens(cu, seq, start, len(rows)))
        new = torch.autograd.grad(updated[LOOKBACK:], leaves, dout[own])

        assert torch.equal(updated[LOOKBACK:], full_updated[own]), f"rank {rank}: forward differs"
        real = slice(LOOKBACK if start < 0 else 0, None)
        for name, o, w in (("key", old[1], new[1]), ("value", old[2], new[2]), ("state", old[0], new[0])):
            assert torch.equal(w[real], o[rows[real]]), f"rank {rank}: per-token {name} grad differs"
            outside = torch.ones(seq, dtype=torch.bool)
            outside[rows[real]] = False
            assert not o[outside].any(), f"rank {rank}: full-sequence {name} grad outside the window"
            if start < 0:
                assert not w[:LOOKBACK].any(), f"rank 0: filler rows got a {name} grad"

        old_dx = old_dx + old[0]  # the full-sequence path's reduce-scatter (SUM over ranks)
        new_dx[own] = new[0][LOOKBACK:]
        halo_grads[rank] = (rows[:LOOKBACK], new[0][:LOOKBACK])
        old_pgrads = [a + g for a, g in zip(old_pgrads, old[3:], strict=True)]
        new_pgrads = [a + g for a, g in zip(new_pgrads, new[3:], strict=True)]
    for rows, grad in halo_grads.values():  # the halo's reduce-scatter, added after the own grad
        new_dx[rows] = new_dx[rows] + grad
    assert torch.equal(new_dx, old_dx), "assembled input grad differs from the full-sequence path"

    # param grads: sums over different token sets; against fp64 the window path must be no worse
    ref_leaves = [t.detach().double().requires_grad_() for t in (hc, key, value, *params)]
    ref_out = m001._ple_gate_conv_reference(*ref_leaves[:3], *ref_leaves[3:], n, eps, dil, cu)
    ref = torch.autograd.grad(ref_out, ref_leaves[3:], dout.double())
    tol = 1e-6 if dtype == torch.float32 else 1e-2
    for name, r, o, w in zip(
        ("norm_key", "norm_query", "norm_conv", "conv"), ref, old_pgrads, new_pgrads, strict=True
    ):
        scale = r.abs().max().item()
        err_old = (o.double() - r).abs().max().item() / scale
        err_new = (w.double() - r).abs().max().item() / scale
        assert err_new <= err_old * (1 + 1e-3) + tol, f"{name}: window err {err_new:.3e} vs full {err_old:.3e}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_window_kernels_match_full_sequence(dtype):
    for seq, cu_bounds in CASES:
        run_multiprocess(partial(_kernel_worker, seq=seq, cu_bounds=cu_bounds, dtype=dtype), world_size=1)


# ----------------------------------------------------------------------------------------------
# the real layer over gloo


def _inputs(seq, cu_bounds, dtype):
    gen = torch.Generator().manual_seed(7)
    width = m001.N_STREAMS * m001.HIDDEN
    heads = (m001.NGRAM_SIZE - 1) * m001.HEADS_PER_NGRAM
    x = torch.randn(seq, 1, width, dtype=torch.float64, generator=gen).to(dtype)
    ngram_ids = torch.randint(0, m001.TABLE_ROWS, (seq, heads), generator=gen)
    upstream = tuple(
        torch.randn(seq, 1, w, dtype=torch.float64, generator=gen).to(dtype)
        for w in (m001.HIDDEN, m001.N_STREAMS, width)
    )
    cu = None if cu_bounds is None else torch.tensor(cu_bounds, dtype=torch.int32)
    return x, ngram_ids, cu, upstream


def _run(layer, x, ngram_ids, cu, upstream, recompute):
    from megatron.core import tensor_parallel

    from miles_plugins.models.qwen3_8_next.ops.ple import clear_ple_batch, publish_ple_batch

    x = x.clone().requires_grad_()
    publish_ple_batch(ngram_ids, cu)

    def body(hidden_states):
        aggregated, _, h_post, residual = layer(hidden_states)
        return aggregated, h_post, residual

    outputs = tensor_parallel.checkpoint(body, False, x) if recompute else body(x)
    clear_ple_batch()
    torch.autograd.backward(outputs, upstream)
    return tuple(o.detach() for o in outputs), x.grad


def _split_run(layer, x, ngram_ids, cu, upstream, recompute, rank, tp_group, local_slice):
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads

    local = x.shape[0] // TP
    own = slice(rank * local, (rank + 1) * local)
    layer.config.sequence_parallel = True
    layer.ple_sp_local_slice = local_slice
    outputs, local_dx = _run(layer, x[own], ngram_ids, cu, tuple(u[own] for u in upstream), recompute)
    _allreduce_non_tensor_model_parallel_grads([layer], layer.config, tp_group)
    dx = torch.empty_like(x)
    dist.all_gather_into_tensor(dx, local_dx.contiguous(), group=tp_group)
    grads = {name: p.grad.clone() for name, p in layer.named_parameters()}
    layer.zero_grad(set_to_none=True)
    return outputs, dx, grads


def _spy_window_calls(layer):
    calls = []
    original = layer._apply_ple_local_window

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    layer._apply_ple_local_window = spy
    return calls


def _fp64_worker(rank, world_size, port, seq, cu_bounds, recompute):
    init_gloo(rank, world_size, port=port)
    try:
        m001._patch_for_cpu()
        from megatron.core import parallel_state

        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        x, ngram_ids, cu, upstream = _inputs(seq, cu_bounds, torch.float64)
        layer = m001._build_layer()
        calls = _spy_window_calls(layer)

        layer.config.sequence_parallel = False
        _, ref_dx = _run(layer, x, ngram_ids, cu, upstream, recompute)
        ref_grads = {name: p.grad.clone() for name, p in layer.named_parameters()}
        layer.zero_grad(set_to_none=True)

        # positive control: the loss past each slice boundary reaches the halo before it
        local = seq // world_size
        for b in range(local, seq, local):
            later = tuple(torch.cat([torch.zeros_like(u[:b]), u[b:]]) for u in upstream)
            cross = _run(layer, x, ngram_ids, cu, later, recompute)[1][b - LOOKBACK : b]
            same_row = cu_bounds is None or b not in cu_bounds
            if same_row:
                assert cross.abs().amax() > 1e3 * m001.TOL, f"no gradient crosses the slice boundary {b}"
            layer.zero_grad(set_to_none=True)

        assert not calls
        _, dx, grads = _split_run(layer, x, ngram_ids, cu, upstream, recompute, rank, tp_group, True)
        assert calls, "the local-window path did not run"
        token_err = (dx - ref_dx).abs().amax(dim=(1, 2))
        bad = [t for t in range(seq) if token_err[t] > m001.TOL]
        assert not bad, f"input grad differs from the unsplit run at tokens {bad}; max err {token_err.max():.3e}"
        for name, ref in ref_grads.items():
            err = (grads[name] - ref).abs().max().item()
            assert err <= m001.TOL * max(1.0, ref.abs().max().item()), f"{name} grad err {err:.3e}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("seq,cu_bounds", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("recompute", [False, True], ids=["no-recompute", "full-recompute"])
def test_local_window_layer_grads_match_unsplit(seq, cu_bounds, recompute):
    if recompute and cu_bounds is None:
        pytest.skip("recompute is covered by the packed cases")
    run_multiprocess(partial(_fp64_worker, seq=seq, cu_bounds=cu_bounds, recompute=recompute), world_size=TP)


class _RowwiseLinear(torch.nn.Module):
    """TELinear(duplicated) whose every output row is the same reduction whatever the row count."""

    def __init__(self, input_size, output_size, *, config, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        self.weight.sequence_parallel = config.sequence_parallel
        self.weight.tensor_model_parallel = False

    def forward(self, x):
        return (x.unsqueeze(-2).float() * self.weight.float()).sum(dim=-1).to(x.dtype), None


class _HostTableBf16(m001._HostTable):
    def forward(self, ids):
        return super().forward(ids).to(torch.bfloat16)


def _kernel_layer_worker(rank, world_size, port, seq, cu_bounds):
    os.environ["TRITON_INTERPRET"] = "1"
    init_gloo(rank, world_size, port=port)
    try:
        m001._patch_for_cpu()
        from megatron.core import parallel_state

        from miles.kernels.embedding.ple_gate_conv import ple_gate_conv
        from miles_plugins.models.qwen3_8_next.ops import ple

        ple.ple_gate_conv = ple_gate_conv  # the real kernels, interpreted
        ple.TELinear = _RowwiseLinear
        ple.Qwen38NextFrozenNGramEmbedding = _HostTableBf16
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
        tp_group = parallel_state.get_tensor_model_parallel_group()

        from miles_plugins.models.qwen3_8_next.hyper_connection import Qwen38NextPLEHyperConnection

        config = SimpleNamespace(**vars(m001._config()))
        config.params_dtype = torch.bfloat16
        # the conv input is RMSNorm(gate * value): with eps 1e-6 the cross-slice gradient is
        # below bf16 resolution and a dropped halo gradient would still compare equal
        config.layernorm_epsilon = 2.0
        layer =Qwen38NextPLEHyperConnection(config, layer_number=1)
        layer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for _, param in sorted(layer.named_parameters()):
                scale = 2.0 if param.dim() == 3 else 0.3
                param.copy_((scale * torch.randn(param.shape, dtype=torch.float64, generator=gen)).to(param.dtype))
        calls = _spy_window_calls(layer)
        x, ngram_ids, cu, upstream = _inputs(seq, cu_bounds, torch.bfloat16)

        old_out, old_dx, old_grads = _split_run(layer, x, ngram_ids, cu, upstream, False, rank, tp_group, False)
        assert not calls
        new_out, new_dx, new_grads = _split_run(layer, x, ngram_ids, cu, upstream, False, rank, tp_group, True)
        assert calls, "the local-window path did not run"
        for i, (o, w) in enumerate(zip(old_out, new_out, strict=True)):
            assert torch.equal(o, w), f"rank {rank}: output {i} differs from the full-sequence path"
        assert torch.equal(old_dx, new_dx), "input grad differs from the full-sequence path"
        for name, o in old_grads.items():
            w = new_grads[name]
            err = ((w.float() - o.float()).abs().max() / o.float().abs().max().clamp_min(1e-30)).item()
            assert err <= 2e-2, f"{name}: window vs full-sequence param grad rel diff {err:.3e}"
    finally:
        dist.destroy_process_group()


def test_local_window_layer_bitwise_with_real_kernels():
    run_multiprocess(partial(_kernel_layer_worker, seq=96, cu_bounds=PACKED_96), world_size=TP)
