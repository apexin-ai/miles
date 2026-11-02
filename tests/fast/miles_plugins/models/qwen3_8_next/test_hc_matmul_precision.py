"""``--qwen38-hc-matmul-precision``: the hyper-connection gate chain's matmul precision.

- fp32 (the default) runs exactly the matmuls of the code before the flag: bitwise equal to a verbatim copy of it.
- bf16 rounds the matmul operands to bf16 and accumulates in fp32: it equals an fp64 reference that rounds the
  same operands at the same points, and stays close to the unrounded fp64 reference.
- tf32 sets TF32 only around the HC matmuls and restores the previous setting (CPU matmuls ignore it, so on CPU
  it equals fp32; its numerics and speed are GPU-only, see the PR's GPU check).
- The flag reaches the HC modules through the TransformerConfig.

The Triton kernels (grouped rmsnorm, gated mean) cannot run on CPU; they are replaced by torch versions of the
same math, which the old copy and the new code share.
"""

from types import SimpleNamespace

import pytest
from tests.ci.ci_register import register_cpu_ci

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from miles.kernels.hyper_connection import hc  # noqa: E402

register_cpu_ci(est_time=20, suite="stage-b-cpu", labels=["miles-plugin"])

T, N, C, R = 37, 4, 32, 8
EPS = 1e-6


class _Launch:
    """``kernel[grid](...)`` on CPU."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


def _rmsnorm_fwd(x2d, weight, n, eps):
    xg = x2d.float().unflatten(-1, (n, -1))
    rstd = torch.rsqrt(xg.pow(2).mean(dim=-1) + eps)
    normed = (xg * rstd.unsqueeze(-1)).flatten(-2) * (1.0 + weight.float())
    return normed, rstd


def _rmsnorm_bwd(x2d, weight, rstd, dnormed, dx, T, N, C, BLOCK_C):
    x_hat = x2d.float().unflatten(-1, (N, C)) * rstd.unsqueeze(-1)
    g = (dnormed * (1.0 + weight.float())).unflatten(-1, (N, C))
    d = rstd.unsqueeze(-1) * (g - x_hat * (g * x_hat).mean(dim=-1, keepdim=True))
    dx.copy_(d.flatten(-2))


def _mix_fwd(gate, normed, out, T, N, C, BLOCK_C):
    out.copy_((gate * normed).unflatten(-1, (N, C)).sum(dim=1) / N)


def _mix_bwd(dmixed, gate, normed, dgate, dnormed, T, N, C, BLOCK_C):
    dm = (dmixed / N).repeat(1, N)
    dgate.copy_(dm * normed)
    dnormed.copy_(dm * gate)


@pytest.fixture
def cpu_kernels(monkeypatch):
    monkeypatch.setattr(hc, "grouped_rmsnorm_fwd", _rmsnorm_fwd)
    monkeypatch.setattr(hc, "grouped_rmsnorm_bwd_kernel", _Launch(_rmsnorm_bwd))
    monkeypatch.setattr(hc, "_gate_mul_mean_fwd_kernel", _Launch(_mix_fwd))
    monkeypatch.setattr(hc, "_gate_mul_mean_bwd_kernel", _Launch(_mix_bwd))


class _OldHCMixInject(torch.autograd.Function):
    """Verbatim copy of ``_HCMixInject`` at miles b5c72b4d (before the flag); kernels looked up on ``hc``."""

    @staticmethod
    def forward(ctx, x2d, weight, w_down, w_up, w_inject, n, eps):
        T, W = x2d.shape
        C = W // n
        normed, rstd = hc.grouped_rmsnorm_fwd(x2d, weight, n, eps)
        z1 = F.linear(normed, w_down.float()) / n
        gate = torch.sigmoid(F.linear(F.silu(z1), w_up.float()))
        mixed = torch.empty(T, C, dtype=x2d.dtype, device=x2d.device)
        if T > 0:
            hc._gate_mul_mean_fwd_kernel[(T,)](gate, normed, mixed, T, N=n, C=C, BLOCK_C=hc.block_c(C))
        if w_inject is not None:
            h_post = 2.0 * torch.sigmoid(F.linear(normed, w_inject.float()) / n)
        else:
            h_post = x2d.new_empty(0, dtype=torch.float32)
        ctx.save_for_backward(x2d, weight, w_down, w_up, w_inject, rstd)
        ctx.n, ctx.eps = n, eps
        return mixed, h_post

    @staticmethod
    def backward(ctx, dmixed, dh_post):
        x2d, weight, w_down, w_up, w_inject, rstd = ctx.saved_tensors
        n, eps = ctx.n, ctx.eps
        T, W = x2d.shape
        C = W // n
        normed, _ = hc.grouped_rmsnorm_fwd(x2d, weight, n, eps)
        z1 = F.linear(normed, w_down.float()) / n
        gate = torch.sigmoid(F.linear(F.silu(z1), w_up.float()))
        s1 = F.silu(z1)

        dmixed = dmixed.float()
        dgate = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        dnormed = torch.empty(T, W, dtype=torch.float32, device=x2d.device)
        if T > 0:
            hc._gate_mul_mean_bwd_kernel[(T,)](
                dmixed, gate, normed, dgate, dnormed, T, N=n, C=C, BLOCK_C=hc.block_c(C)
            )

        dz2 = dgate * gate * (1.0 - gate)
        dw_up = dz2.t() @ s1
        ds1 = dz2 @ w_up.float()
        sig_z1 = torch.sigmoid(z1)
        dz1 = ds1 * sig_z1 * (1.0 + z1 * (1.0 - sig_z1))
        dz1 = dz1 / n
        dw_down = dz1.t() @ normed
        dnormed += dz1 @ w_down.float()

        dw_inject = None
        if w_inject is not None and dh_post is not None and dh_post.numel() > 0:
            u = F.linear(normed, w_inject.float()) / n
            sig_u = torch.sigmoid(u)
            du = dh_post.float() * 2.0 * sig_u * (1.0 - sig_u)
            du = du / n
            dw_inject = (du.t() @ normed).to(w_inject.dtype)
            dnormed += du @ w_inject.float()

        x_hat = (x2d.float().view(T, n, C) * rstd.unsqueeze(-1)).view(T, W)
        dweight = (dnormed * x_hat).sum(dim=0).to(weight.dtype)

        dx = torch.empty_like(x2d)
        if T > 0:
            hc.grouped_rmsnorm_bwd_kernel[(T * n,)](
                x2d, weight, rstd, dnormed, dx, T, N=n, C=C, BLOCK_C=hc.block_c(C)
            )
        return dx, dweight, dw_down.to(w_down.dtype), dw_up.to(w_up.dtype), dw_inject, None, None


def _inputs(seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, N * C, generator=g).to(dtype)
    weight = (0.1 * torch.randn(N * C, generator=g)).to(dtype)
    w_down = (torch.randn(R, N * C, generator=g) / (N * C) ** 0.5).to(dtype)
    w_up = (torch.randn(N * C, R, generator=g) / R**0.5).to(dtype)
    w_inj = (torch.randn(N, N * C, generator=g) / (N * C) ** 0.5).to(dtype)
    dmixed = torch.randn(T, C, generator=g).to(dtype)
    dh_post = torch.randn(T, N, generator=g)
    return [x, weight, w_down, w_up, w_inj], dmixed, dh_post


def _run(fn, tensors, dmixed, dh_post, with_inject=True):
    leaves = [t.clone().requires_grad_(True) for t in tensors]
    if not with_inject:
        leaves[4] = None
    mixed, h_post = fn(*leaves)
    loss = (mixed.float() * dmixed.float()).sum()
    if with_inject:
        loss = loss + (h_post * dh_post).sum()
    loss.backward()
    grads = [leaf.grad for leaf in leaves if leaf is not None]
    return [mixed, h_post] + grads


def _new(precision=None):
    def fn(x, weight, w_down, w_up, w_inj):
        kwargs = {} if precision is None else {"matmul_precision": precision}
        return hc.hc_mix_inject(x, weight, w_down, w_up, w_inj, N, EPS, **kwargs)

    return fn


def _old(x, weight, w_down, w_up, w_inj):
    return _OldHCMixInject.apply(x, weight, w_down, w_up, w_inj, N, EPS)


@pytest.mark.parametrize("with_inject", [True, False])
@pytest.mark.parametrize("precision", [None, "fp32", "tf32"])
def test_fp32_default_and_cpu_tf32_bitwise_equal_old_code(cpu_kernels, with_inject, precision):
    tensors, dmixed, dh_post = _inputs()
    want = _run(_old, tensors, dmixed, dh_post, with_inject)
    got = _run(_new(precision), tensors, dmixed, dh_post, with_inject)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == w.dtype and torch.equal(g, w)


def _reference(x, weight, w_down, w_up, w_inj, round_operands):
    """fp64 math; ``round_operands`` rounds each matmul's activation operand to bf16 where the bf16 mode does."""
    rnd = (lambda t: t.bfloat16().double()) if round_operands else (lambda t: t)
    x, weight, w_down, w_up, w_inj = (t.double() for t in (x, weight, w_down, w_up, w_inj))
    xg = x.unflatten(-1, (N, C))
    normed = (xg * torch.rsqrt(xg.pow(2).mean(dim=-1, keepdim=True) + EPS)).flatten(-2) * (1.0 + weight)
    z1 = F.linear(rnd(normed), w_down) / N
    gate = torch.sigmoid(F.linear(rnd(F.silu(z1)), w_up))
    mixed = (gate * normed).unflatten(-1, (N, C)).mean(dim=1)
    h_post = 2.0 * torch.sigmoid(F.linear(rnd(normed), w_inj) / N)
    return mixed, h_post


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def test_bf16_forward_matches_operand_rounded_reference(cpu_kernels):
    tensors, _, _ = _inputs(1)
    mixed, h_post = hc.hc_mix_inject(*tensors, N, EPS, matmul_precision="bf16")
    ref_mixed, ref_h_post = _reference(*tensors, round_operands=True)
    # h_post is fp32: only fp32 accumulation order differs from the operand-rounded fp64 reference
    assert h_post.dtype == torch.float32 and _rel(h_post, ref_h_post) < 1e-6
    # mixed is bf16 (the residual dtype): one output rounding
    assert mixed.dtype == torch.bfloat16 and _rel(mixed, ref_mixed) < 4e-3


def test_bf16_close_to_fp64_and_fp32(cpu_kernels):
    """Every output and gradient of bf16 mode is within 1% of the fp64 reference, and its error is a small
    multiple of fp32 mode's (whose bf16 output / bf16 grads dominate its error)."""
    tensors, dmixed, dh_post = _inputs(2)

    def ref(x, weight, w_down, w_up, w_inj):
        return _reference(x, weight, w_down, w_up, w_inj, round_operands=False)

    tensors64 = [t.double() for t in tensors]
    want = _run(ref, tensors64, dmixed.double(), dh_post.double())
    fp32 = _run(_new("fp32"), tensors, dmixed, dh_post)
    bf16 = _run(_new("bf16"), tensors, dmixed, dh_post)
    names = ["mixed", "h_post", "dx", "dweight", "dw_down", "dw_up", "dw_inject"]
    for name, w, a, b in zip(names, want, fp32, bf16, strict=True):
        assert a.dtype == b.dtype, name
        err32, err16 = _rel(a, w), _rel(b, w)
        assert err16 < 1e-2, (name, err16)
        assert err16 < max(10 * err32, 1e-4), (name, err16, err32)


def test_tf32_scope_sets_and_restores(cpu_kernels, monkeypatch):
    matmul = torch.backends.cuda.matmul
    before = matmul.fp32_precision
    seen = []
    real_linear, real_mm = hc._linear, hc._mm

    def spy_linear(x, w, precision):
        seen.append(matmul.fp32_precision)
        return real_linear(x, w, precision)

    def spy_mm(a, b, precision):
        seen.append(matmul.fp32_precision)
        return real_mm(a, b, precision)

    monkeypatch.setattr(hc, "_linear", spy_linear)
    monkeypatch.setattr(hc, "_mm", spy_mm)
    tensors, dmixed, dh_post = _inputs(3)
    _run(_new("tf32"), tensors, dmixed, dh_post)
    assert seen and set(seen) == {"tf32"}
    assert matmul.fp32_precision == before
    seen.clear()
    _run(_new("fp32"), tensors, dmixed, dh_post)
    assert seen and set(seen) == {before}


def test_rejects_unknown_precision():
    tensors, _, _ = _inputs()
    with pytest.raises(ValueError, match="matmul precision"):
        hc.hc_mix_inject(*tensors, N, EPS, matmul_precision="fp16")


def _hc_config(**extra):
    return SimpleNamespace(
        num_residual_streams=N,
        hidden_size=C,
        layernorm_epsilon=EPS,
        qwen3_8_next_hc_lowrank=R,
        params_dtype=torch.float32,
        sequence_parallel=False,
        **extra,
    )


@pytest.mark.parametrize("precision", [None, "fp32", "tf32", "bf16"])
def test_hc_modules_pass_config_precision(monkeypatch, precision):
    from miles_plugins.models.qwen3_8_next import hyper_connection

    extra = {} if precision is None else {"qwen3_8_next_hc_matmul_precision": precision}
    config = _hc_config(**extra)
    calls = []

    def spy(x, weight, w_down, w_up, w_inject, n, eps, matmul_precision="fp32"):
        calls.append(matmul_precision)
        return x[..., :C], None if w_inject is None else x[..., :n]

    monkeypatch.setattr(hyper_connection, "hc_mix_inject", spy)
    x = torch.randn(3, 1, N * C)
    hyper_connection.Qwen38NextHyperConnection(config, layer_number=1)(x)
    hyper_connection.Qwen38NextHCHeadContraction(config)(x)
    assert calls == [precision or "fp32"] * 2


def test_hc_module_rejects_unknown_precision():
    from miles_plugins.models.qwen3_8_next import hyper_connection

    with pytest.raises(ValueError, match="matmul precision"):
        hyper_connection.Qwen38NextHCHeadContraction(_hc_config(qwen3_8_next_hc_matmul_precision="fp16"))


def test_spec_sets_hc_matmul_precision(monkeypatch):
    from miles_plugins.models.qwen3_8_next import qwen3_8_next

    config = SimpleNamespace()
    args = SimpleNamespace(qwen38_hc_matmul_precision="bf16")
    monkeypatch.setattr(qwen3_8_next, "load_hf_config", lambda path: None)
    monkeypatch.setattr(qwen3_8_next, "_get_text_config", lambda hf: None)
    monkeypatch.setattr(qwen3_8_next, "_apply_qwen3_8_next_config", lambda c, t: None)
    args.hf_checkpoint = "/x"
    args.backend = SimpleNamespace(position_embedding_type="rope")
    # the knobs are set before the VPP guard; the rest of the spec needs a checkpoint
    config.virtual_pipeline_model_parallel_size = 2
    with pytest.raises(NotImplementedError, match="interleaved pipeline"):
        qwen3_8_next.get_qwen3_8_next_spec(args=args, config=config)
    assert config.qwen3_8_next_hc_matmul_precision == "bf16"


def test_cli_default_and_choices():
    from miles.utils.args.configs.custom_megatron_plugins import Qwen38VisionMegatronPluginsConfig

    field = Qwen38VisionMegatronPluginsConfig.model_fields["qwen38_hc_matmul_precision"]
    assert field.default == "fp32"
    assert hc.MATMUL_PRECISIONS == ("fp32", "tf32", "bf16")
