"""The Qwen3.8-Next PLE layer at TP = world size with sequence parallelism, against the same layer
unsplit: every token's input gradient and every param gradient after Megatron's TP reduction.
Real Triton kernels, TE linears, host n-gram table (row-sharded) and NCCL collectives.

The conv input is RMSNorm(gate * value), so the cross-token path from the hidden state only
exists through the norm's eps. Production eps (1e-6) makes it invisible in bf16; this test sets
eps = 2 and a large conv weight, so the cross-boundary gradient is ~0.2 of the largest gradient
(fp64 CPU estimate) and a dropped one is far above the bf16 noise.

    torchrun --nproc_per_node=2 tests/fast-gpu/test_qwen3_8_next_ple_sp_grads.py
    torchrun --nproc_per_node=4 tests/fast-gpu/test_qwen3_8_next_ple_sp_grads.py
"""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.ci.ci_register import register_cuda_ci

from miles_plugins.models.qwen3_8_next.hyper_connection import Qwen38NextPLEHyperConnection
from miles_plugins.models.qwen3_8_next.ops.ple import clear_ple_batch, publish_ple_batch

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"], num_gpus=2
)

N_STREAMS, HIDDEN, LOWRANK = 4, 256, 32
NGRAM_SIZE, HEADS_PER_NGRAM, EMBED_DIM, CONV_KERNEL = 3, 4, 256, 4
LOOKBACK = (CONV_KERNEL - 1) * NGRAM_SIZE
SPLIT_PARTS, ROWS_PER_SHARD = 8, 128
EPS = 2.0
SEQ = 1024
TOL = 2e-2  # bf16, relative to the largest gradient entry
# Unfixed code drops, at each slice boundary, exactly the gradient later tokens send back
# across it; the largest of those must clear TOL by a margin for the test to tell the two apart.
SIGNAL = 2.5 * TOL


def build_config(world):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        layernorm_epsilon=EPS,
        params_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=world,
        sequence_parallel=world > 1,
    )
    config.num_residual_streams = N_STREAMS
    config.qwen3_8_next_hc_lowrank = LOWRANK
    config.qwen3_8_next_ple_embed_dim = EMBED_DIM
    config.qwen3_8_next_ngram_size = NGRAM_SIZE
    config.qwen3_8_next_heads_per_ngram = HEADS_PER_NGRAM
    config.qwen3_8_next_split_ngram_parts = SPLIT_PARTS
    config.qwen3_8_next_ngram_rows_per_shard = ROWS_PER_SHARD
    config.qwen3_8_next_ple_conv_kernel_size = CONV_KERNEL
    config.qwen3_8_next_hf_checkpoint = None
    return config


def build_layer(config):
    layer = Qwen38NextPLEHyperConnection(config, layer_number=1).cuda()
    layer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        # identical params on every rank; the norm/conv params start at zero
        for _, param in sorted(layer.named_parameters()):
            scale = 2.0 if param.dim() == 3 else 1.0 / param.shape[-1] ** 0.5 if param.dim() == 2 else 0.3
            param.copy_(scale * torch.randn(param.shape, device="cuda", generator=gen))
        table = layer.ple.ple_embedding
        full = torch.randn(SPLIT_PARTS * ROWS_PER_SHARD, table.embedding_dim, device="cuda", generator=gen)
        table.table.copy_(full[table.row_start : table.row_end].to(torch.bfloat16).cpu())
        table._loaded = True
    return layer


def inputs():
    gen = torch.Generator(device="cuda").manual_seed(7)
    heads = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
    x = torch.randn(SEQ, 1, N_STREAMS * HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    ngram_ids = torch.randint(0, SPLIT_PARTS * ROWS_PER_SHARD, (SEQ, heads), device="cuda", generator=gen)
    cu = torch.tensor([0, 100, 600, SEQ], device="cuda", dtype=torch.int32)
    # d aggregated, d h_post, d residual; cast to each output's dtype in run()
    upstream = tuple(
        torch.randn(SEQ, 1, width, device="cuda", generator=gen) for width in (HIDDEN, N_STREAMS, N_STREAMS * HIDDEN)
    )
    return x, ngram_ids, cu, upstream


def run(layer, x, ngram_ids, cu, upstream, recompute):
    x = x.clone().requires_grad_()
    publish_ple_batch(ngram_ids, cu)

    def body(hidden_states):
        aggregated, _, h_post, residual = layer(hidden_states)
        return aggregated, h_post, residual

    outputs = tensor_parallel.checkpoint(body, False, x) if recompute else body(x)
    clear_ple_batch()
    torch.autograd.backward(outputs, [u.to(o.dtype) for u, o in zip(upstream, outputs, strict=True)])
    return x.grad.float()


def check(layer, tp_group, rank, world, recompute):
    x, ngram_ids, cu, upstream = inputs()
    local = SEQ // world

    layer.config.sequence_parallel = False
    ref_dx = run(layer, x, ngram_ids, cu, upstream, recompute)
    ref_grads = {n: p.grad.float().clone() for n, p in layer.named_parameters()}
    layer.zero_grad(set_to_none=True)
    scale = ref_dx.abs().max()

    signal = 0.0
    for b in range(local, SEQ, local):
        later_only = tuple(torch.cat([torch.zeros_like(u[:b]), u[b:]]) for u in upstream)
        crossing = run(layer, x, ngram_ids, cu, later_only, recompute)[b - LOOKBACK : b].abs().max() / scale
        signal = max(signal, crossing.item())
        layer.zero_grad(set_to_none=True)

    layer.config.sequence_parallel = True
    own = slice(rank * local, (rank + 1) * local)
    local_dx = run(layer, x[own], ngram_ids, cu, tuple(u[own] for u in upstream), recompute)
    _allreduce_non_tensor_model_parallel_grads([layer], layer.config, tp_group)
    dx = torch.empty_like(ref_dx)
    dist.all_gather_into_tensor(dx, local_dx.contiguous(), group=tp_group)

    token_err = (dx - ref_dx).abs().amax(dim=(1, 2)) / scale
    errors = {"signal": signal, "dx": token_err.max().item()}
    tails = [t for r in range(1, world) for t in range(r * local - LOOKBACK, r * local)]
    errors["dx_boundary"] = token_err[tails].max().item()
    for name, param in layer.named_parameters():
        ref = ref_grads[name]
        errors[name] = ((param.grad.float() - ref).abs().max() / ref.abs().max()).item()
    layer.zero_grad(set_to_none=True)
    return errors


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world > 1, "run with torchrun --nproc_per_node >= 2"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(1234)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    layer = build_layer(build_config(world))

    failures = {}
    for recompute in (False, True):
        errors = check(layer, tp_group, rank, world, recompute)
        tag = f"TP={world} recompute={recompute}"
        if rank == 0:
            print(f"[{tag}] " + "  ".join(f"{k}={v:.2e}" for k, v in errors.items()))
        bad = {k: v for k, v in errors.items() if k != "signal" and v > TOL}
        if errors["signal"] < SIGNAL:
            bad["signal"] = errors["signal"]
        if bad:
            failures[tag] = bad
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    if rank == 0:
        print(f"TP={world} PLE layer under SP matches the unsplit layer")


if __name__ == "__main__":
    if "RANK" not in os.environ:
        os.execvp("torchrun", ["torchrun", "--nproc_per_node=2", __file__])
    main()
