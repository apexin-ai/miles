"""Qwen3.8-Next GDN on one node: head-sharded (--qwen38-gdn-head-sharded) vs replicated, real fla kernels.

Both layers hold the checkpoint's layer-``--layer`` GDN weights (the head-sharded one loaded through the bridge's
HF -> Megatron conversion and TP split) and run under TP = world size with sequence parallelism on packed rows.
Each bf16 layer is compared with an fp32 replicated reference on the same input: forward output, input gradient and
every parameter gradient (head-sharded grads gathered and exported to HF layout). Then bf16 forward + backward time
and peak memory of each layer.

    torchrun --nproc_per_node=8 tests/manual/qwen38_gdn_head_sharded_check.py --hf-checkpoint <dir> \\
        --tokens 32768 131072 262144
"""

import argparse
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from safetensors import safe_open

from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf
from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge
from miles_plugins.models.qwen3_5 import _get_text_config
from miles_plugins.models.qwen3_8_next.gdn import HeadShardedGDNAttention, ReplicatedGDNAttention

PARTITION_DIM = {"out_proj.weight": 1, "norm.weight": None}


def hf_layer_weights(path, layer):
    import json

    prefix = f"model.language_model.layers.{layer}.linear_attn."
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for key, shard in index.items():
        if key.startswith(prefix):
            with safe_open(os.path.join(path, shard), "pt") as f:
                out[key[len(prefix) :]] = f.get_tensor(key)
    return out


def build(args, sharded, dtype, hf, sequence_parallel=True):
    world = dist.get_world_size()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hf["out_proj.weight"].shape[0],
        num_attention_heads=8,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        tensor_model_parallel_size=world,
        sequence_parallel=sequence_parallel,
    )
    layer_args = SimpleNamespace(
        hf_checkpoint=args.hf_checkpoint,
        linear_attention_backend="fla",
        linear_attention_input_norm="te",
        allgather_cp=False,
        backend=SimpleNamespace(sequence_parallel=sequence_parallel),
    )
    if not sharded:
        layer = ReplicatedGDNAttention(layer_args, config, layer_number=args.layer + 1).cuda()
        layer.input_layernorm = nn.Identity()  # Qwen38NextLinearAttention
        layer.to(dtype)
        layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
        with torch.no_grad():
            for name, param in layer.linear_attn.named_parameters():
                param.copy_(hf[name])
        return layer
    layer = HeadShardedGDNAttention(layer_args, config, layer_number=args.layer + 1)
    bridge = Qwen38NextBridge.__new__(Qwen38NextBridge)
    bridge.hf_config = load_hf_config(args.hf_checkpoint)
    bridge.dtype = None
    bridge.make_vocab_size_divisible_by = None
    bridge._select_gdn_layout([layer])
    assert bridge._GDN_GROUP_MAJOR
    params = dict(layer.linear_attn.named_parameters())
    with torch.no_grad():
        for name, param in params.items():
            mcore_name = f"decoder.layers.{args.layer}.self_attention.linear_attn.{name}"
            mcore = bridge._weight_to_mcore_format(mcore_name, [hf[name].cuda()])
            param.copy_(bridge._weight_split_across_tp(mcore_name, mcore, param, world)[dist.get_rank()])
    return layer


def hf_grads(layer, sharded, args):
    """Parameter gradients in HF layout (head-sharded: gathered over TP, then the raw export)."""
    out = {}
    for name, param in layer.linear_attn.named_parameters():
        grad = param.grad
        if sharded:
            dim = PARTITION_DIM.get(name, 0)
            if dim is not None:
                parts = [torch.empty_like(grad) for _ in range(dist.get_world_size())]
                dist.all_gather(parts, grad.contiguous())
                grad = torch.cat(parts, dim=dim)
            export_args = SimpleNamespace(
                qwen38_gdn_head_sharded=True,
                hf_checkpoint=args.hf_checkpoint,
                backend=SimpleNamespace(kv_channels=None, hidden_size=1, num_attention_heads=1, num_query_groups=1),
            )
            [(_, grad)] = convert_qwen3_8_next_to_hf(
                export_args, f"module.module.decoder.layers.{args.layer}.self_attention.linear_attn.{name}", grad
            )
        out[name] = grad.float()
    return out


def run(layer, x, dy, cu_seqlens):
    """SP: this rank's slice in, this rank's slice out; returns the gathered output and input gradient."""
    world, rank = dist.get_world_size(), dist.get_rank()
    local = x.shape[0] // world
    own = slice(rank * local, (rank + 1) * local)
    x_local = x[own].detach().clone().requires_grad_()
    out, _ = layer(x_local, None, packed_seq_params=SimpleNamespace(cu_seqlens_q=cu_seqlens))
    out.backward(dy[own])
    full_out = torch.empty(x.shape, dtype=out.dtype, device=x.device)
    full_dx = torch.empty(x.shape, dtype=x_local.grad.dtype, device=x.device)
    dist.all_gather_into_tensor(full_out, out.detach().contiguous())
    dist.all_gather_into_tensor(full_dx, x_local.grad.contiguous())
    return full_out.float(), full_dx.float()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def timed(layer, x, dy, cu_seqlens, iters):
    world, rank = dist.get_world_size(), dist.get_rank()
    local = x.shape[0] // world
    x_local = x[rank * local : (rank + 1) * local].detach().clone().requires_grad_()
    dy_local = dy[rank * local : (rank + 1) * local]
    packed = SimpleNamespace(cu_seqlens_q=cu_seqlens)
    for _ in range(2):
        layer(x_local, None, packed_seq_params=packed)[0].backward(dy_local)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(iters):
        layer(x_local, None, packed_seq_params=packed)[0].backward(dy_local)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed, (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32768])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--no-fp32-reference", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    model_parallel_cuda_manual_seed(1234)
    rank = dist.get_rank()
    text = _get_text_config(load_hf_config(args.hf_checkpoint))
    assert text.layer_types[args.layer] == "linear_attention", text.layer_types[args.layer]
    hf = {k: v.cuda() for k, v in hf_layer_weights(args.hf_checkpoint, args.layer).items()}
    if rank == 0:
        print({k: (tuple(v.shape), str(v.dtype)) for k, v in hf.items()}, flush=True)

    for tokens in args.tokens:
        gen = torch.Generator(device="cuda").manual_seed(tokens)
        # packed rows: one long row, a short one, the rest (lengths not multiples of the 64-token fla chunk)
        bounds = [0, tokens // 2 + 77, tokens // 2 + 77 + 1000, tokens]
        cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        x = torch.randn(tokens, 1, text.hidden_size, device="cuda", generator=gen)
        dy = torch.randn(tokens, 1, text.hidden_size, device="cuda", generator=gen) * 1e-2
        results = {}
        if not args.no_fp32_reference:
            ref = build(args, sharded=False, dtype=torch.float32, hf=hf)
            out_ref, dx_ref = run(ref, x, dy, cu_seqlens)
            grads_ref = hf_grads(ref, False, args)
            del ref
            torch.cuda.empty_cache()
        for sharded in (False, True):
            layer = build(args, sharded=sharded, dtype=torch.bfloat16, hf=hf)
            out, dx = run(layer, x.bfloat16(), dy.bfloat16(), cu_seqlens)
            grads = hf_grads(layer, sharded, args)
            tag = "sharded" if sharded else "replicated"
            if not args.no_fp32_reference:
                errors = {"out": rel(out, out_ref), "dx": rel(dx, dx_ref)}
                errors.update({k: rel(grads[k], grads_ref[k]) for k in grads_ref})
                results[tag] = errors
            layer.zero_grad(set_to_none=True)
            elapsed, peak = timed(layer, x.bfloat16(), dy.bfloat16(), cu_seqlens, args.iters)
            if rank == 0:
                print(f"[T={tokens}] {tag}: fwd+bwd {elapsed * 1e3:.1f} ms, peak +{peak:.2f} GiB", flush=True)
            del layer
            torch.cuda.empty_cache()
        if rank == 0 and results:
            for key in results["replicated"]:
                r, s = results["replicated"][key], results["sharded"][key]
                print(
                    f"[T={tokens}] vs fp32 replicated  {key:20s} bf16-replicated {r:.3e}  bf16-sharded {s:.3e}"
                    f"  ratio {s / max(r, 1e-30):.2f}",
                    flush=True,
                )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
