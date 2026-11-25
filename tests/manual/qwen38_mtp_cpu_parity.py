"""Single native MTP depth, CPU/Gloo, short causal prefix, sliced real HF weights.

Run under TRITON_INTERPRET=1 TORCH_COMPILE_DISABLE=1 with the matching MCore and
SGLang on PYTHONPATH. The MTP class and HC kernels are real; the inner transformer
uses eager Q/K/V and expert projections in place of TE and SGLang's serving kernels.
There are no GPU allocations or requests to a running inference service.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from torch import nn

from miles.kernels.attention.qsa import qsa_sparse_attention
from miles_plugins.models.qwen3_8_next.hyper_connection import Qwen38NextHCHeadContraction, Qwen38NextHyperConnection
from miles_plugins.models.qwen3_8_next.mtp import Qwen38MTPHiddenNorm, Qwen38MTPLayer

WIDTH, STREAMS, VOCAB, EXPERTS, FFN = 8, 4, 17, 3, 8
EPS = 1e-6


class BaseSlices:
    def __init__(self, path):
        self.root = path
        self.index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]

    def __call__(self, key, *axes):
        with safe_open(self.root / self.index[key], framework="pt", device="cpu") as f:
            return f.get_slice(key)[tuple(axes)].float().contiguous()


def linear(weight):
    mod = nn.Linear(weight.size(1), weight.size(0), bias=False)
    mod.weight = nn.Parameter(weight.clone())
    return mod


class ParallelLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


def native_norm(weight):
    mod = GemmaRMSNorm(weight.numel(), eps=EPS)
    mod.weight = nn.Parameter(weight.clone())
    # Use SGLang's device-independent inference implementation on CPU.
    mod.forward = mod.forward_native
    return mod


def model_pair(base, cfg, name, *, output=False):
    training = Qwen38NextHCHeadContraction(cfg) if output else Qwen38NextHyperConnection(cfg, layer_number=1)
    ref_cfg = HyperConnectionConfig(
        hc_count=STREAMS,
        hidden_size=WIDTH,
        hc_lowrank=3,
        params_dtype=torch.float32,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    # Older Qwen4Exp builds query a CUDA device only to choose nn.Linear placement.
    with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
        reference = GatedResidual(ref_cfg, use_combine=not output)
    for local, hf in (
        ("hc_norm_weight", "hc_norm"),
        ("input_mix_weight_up", "input_mix_weight_up"),
        ("input_mix_weight_down", "input_mix_weight_down"),
        ("block_inject_weight", "block_inject_weight"),
    ):
        if output and local == "block_inject_weight":
            continue
        param = getattr(training, local)
        value = base(f"{name}.{hf}.weight", *(slice(0, size) for size in param.shape))
        param.data.copy_(value)
        getattr(reference, hf).weight.data.copy_(value)
    return training, reference


class TinyAttention:
    """For a prefix within the QSA tail block every causal key is eligible."""

    def __init__(self, base):
        name = "mtp.layers.0.self_attn."
        w = slice(WIDTH)
        self.q = base(name + "q_proj.weight", w, w)
        self.gate = base(name + "q_proj.weight", slice(256, 256 + WIDTH), w)
        self.k = base(name + "k_proj.weight", w, w)
        self.v = base(name + "v_proj.weight", w, w)
        self.out = base(name + "o_proj.weight", w, w)
        self.q_norm = native_norm(base(name + "q_norm.weight", w))
        self.k_norm = native_norm(base(name + "k_norm.weight", w))

    @staticmethod
    def rotate(x):
        # Partial RoPE with one pair, using the real checkpoint's 1/4 rotary fraction.
        angle = torch.arange(len(x), dtype=x.dtype).unsqueeze(-1)
        a, b = x[:, :1], x[:, 1:2]
        return torch.cat([a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos(), x[:, 2:]], dim=-1)

    def __call__(self, x, *, reference):
        q = self.rotate(self.q_norm(F.linear(x, self.q)))
        k = self.rotate(self.k_norm(F.linear(x, self.k)))
        v = F.linear(x, self.v)
        if reference:
            out = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None], is_causal=True)[0, 0]
        else:
            rows = torch.arange(len(x))
            ids = rows.expand(len(x), -1).masked_fill(rows[None, :] > rows[:, None], -1).contiguous().int()
            out = qsa_sparse_attention(q[:, None], k[:, None], v[:, None], ids, WIDTH**-0.5)[:, 0]
        return F.linear(out * torch.sigmoid(F.linear(x, self.gate)), self.out)


class TinyMoE:
    def __init__(self, base):
        name = "mtp.layers.0.mlp."
        w, f, e = slice(WIDTH), slice(FFN), slice(EXPERTS)
        self.router = base(name + "gate.weight", e, w)
        self.shared_gate = base(name + "shared_expert_gate.weight", slice(1), w)
        self.shared = [
            base(name + f"shared_expert.{k}_proj.weight", *shape)
            for k, shape in (("gate", (f, w)), ("up", (f, w)), ("down", (w, f)))
        ]
        self.gate = base(name + "experts.gate_up_proj", e, f, w)
        self.up = base(name + "experts.gate_up_proj", e, slice(640, 640 + FFN), w)
        self.down = base(name + "experts.down_proj", e, w, f)

    @staticmethod
    def expert(x, gate, up, down):
        return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    def __call__(self, x, *, reference):
        probabilities, ids = F.linear(x, self.router).topk(2)
        coefficients = probabilities.softmax(-1)
        if reference:
            routed = torch.stack(
                [
                    sum(
                        p * self.expert(row, self.gate[e], self.up[e], self.down[e])
                        for e, p in zip(experts, probs, strict=True)
                    )
                    for row, experts, probs in zip(x, ids, coefficients, strict=True)
                ]
            )
        else:
            all_experts = torch.stack(
                [self.expert(x, *weights) for weights in zip(self.gate, self.up, self.down, strict=True)], 1
            )
            routed = (all_experts.gather(1, ids[..., None].expand(-1, -1, WIDTH)) * coefficients[..., None]).sum(1)
        return routed + F.linear(x, self.shared_gate).sigmoid() * self.expert(x, *self.shared)


class EagerInner(nn.Module):
    def __init__(self, hc1, hc2, attention, moe, *, reference):
        super().__init__()
        self.hc1, self.hc2, self.attention, self.moe, self.reference = hc1, hc2, attention, moe, reference

    def forward(self, hidden_states, **kwargs):
        x = hidden_states.squeeze(1)
        for site, transform in ((self.hc1, self.attention), (self.hc2, self.moe)):
            if self.reference:
                mixed, residual = site.mix(x)
                x = site.combine(transform(mixed, reference=True), residual)
            else:
                mixed, res_gate, inject, residual = site(x)
                x = site.fused_h_res_h_post_bda(
                    res_gate,
                    residual,
                    inject,
                    transform(mixed, reference=False),
                    dropout_prob=0,
                    training=False,
                    fused=False,
                )
        return x.unsqueeze(1), None


def compare(path):
    base = BaseSlices(path)
    cfg = SimpleNamespace(
        num_residual_streams=STREAMS,
        hidden_size=WIDTH,
        sequence_parallel=False,
        qwen3_8_next_hc_lowrank=3,
        layernorm_epsilon=EPS,
        params_dtype=torch.float32,
        fp8=False,
    )
    sgl = Qwen4ExpForCausalLMMTP.__new__(Qwen4ExpForCausalLMMTP)
    nn.Module.__init__(sgl)
    sgl.hc_count, sgl.hidden_size = STREAMS, WIDTH
    sgl._mtp_input_fusion = sgl._init_mtp_input_fusion(SimpleNamespace(hidden_size=WIDTH, rms_norm_eps=EPS))
    train = Qwen38MTPLayer.__new__(Qwen38MTPLayer)
    nn.Module.__init__(train)
    train.config = cfg
    train.tp_group = dist.group.WORLD
    train.mhc_enabled = True
    train.sequence_parallel = False
    train.mtp_layer_pattern = None
    for norm, name in (("enorm", "pre_fc_norm_embedding"), ("hnorm", "pre_fc_norm_hidden")):
        weight = base(f"mtp.{name}.weight", slice(WIDTH if norm == "enorm" else WIDTH * STREAMS))
        setattr(sgl, name, native_norm(weight))
        train_norm = native_norm(weight) if norm == "enorm" else Qwen38MTPHiddenNorm(cfg, WIDTH, EPS)
        train_norm.weight.data.copy_(weight)
        setattr(train, norm, train_norm)
    for proj, name in (("e_proj", "fc_embedding"), ("h_proj", "fc_hidden")):
        weight = base(f"mtp.{name}.weight", slice(WIDTH), slice(WIDTH))
        setattr(sgl, name, linear(weight))
        train_proj = ParallelLinear(WIDTH, WIDTH, bias=False)
        train_proj.weight.data.copy_(weight)
        setattr(train, proj, train_proj)
    attn = TinyAttention(base)
    mlp = TinyMoE(base)
    t1, r1 = model_pair(base, cfg, "mtp.layers.0.attn_hyper_connection")
    t2, r2 = model_pair(base, cfg, "mtp.layers.0.mlp_hyper_connection")
    train.hc_head_contraction, readout = model_pair(base, cfg, "mtp.hyper_connection_mixer", output=True)
    train.mtp_model_layer = EagerInner(t1, t2, attn, mlp, reference=False)

    embedding = nn.Embedding.from_pretrained(
        base("model.language_model.embed_tokens.weight", slice(VOCAB), slice(WIDTH))
    )
    head = linear(base("lm_head.weight", slice(VOCAB), slice(WIDTH)))
    ref_inner = EagerInner(r1, r2, attn, mlp, reference=True)

    class ReferenceBody(nn.Module):
        def forward(self, input_ids, positions, batch, states):
            streams, _ = ref_inner(states.unsqueeze(1))
            streams = streams.squeeze(1)
            return readout.mix(streams)[0], streams

    sgl.model = ReferenceBody()
    sgl.model.embed_tokens = embedding
    sgl.lm_head = head
    sgl.logits_processor = lambda ids, x, out, batch: SimpleNamespace(next_token_logits=out(x))
    ids = torch.tensor([2, 3, 5, 7, 11])
    pos = torch.arange(len(ids))
    streams = torch.randn(len(ids), WIDTH * STREAMS, generator=torch.Generator().manual_seed(219))
    mode = SimpleNamespace(is_idle=lambda: False, is_extend=lambda: False, is_draft_extend_v2=lambda: False)
    batch = SimpleNamespace(
        forward_mode=mode, spec_info=SimpleNamespace(hidden_states=streams), extend_seq_lens=None, mm_input_embeds=None
    )
    with torch.no_grad():
        reference = sgl(ids, pos, batch)
        wide = train._proj_and_transformer_layer(streams[:, None], embedding(ids)[:, None], input_ids=ids[None])
        actual = head(train._postprocess(wide).squeeze(1))
    torch.testing.assert_close(wide.squeeze(1), reference.hidden_states, rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(actual, reference.next_token_logits, rtol=2e-5, atol=1e-7)
    src = Path(inspect.getfile(Qwen4ExpForCausalLMMTP))
    print(
        json.dumps(
            dict(
                max_abs_logits=(actual - reference.next_token_logits).abs().max().item(),
                max_abs_streams=(wide.squeeze(1) - reference.hidden_states).abs().max().item(),
                positions=len(ids),
                vocab=VOCAB,
                sglang_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                sglang_hc_sha256=hashlib.sha256(Path(inspect.getfile(GatedResidual)).read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as tmp:
        dist.init_process_group("gloo", init_method=f"file://{tmp}/store", rank=0, world_size=1)
        try:
            compare(args.base)
        finally:
            dist.destroy_process_group()
