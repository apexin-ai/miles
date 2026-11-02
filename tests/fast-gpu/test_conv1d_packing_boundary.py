"""
test_conv1d_packing_boundary.py

Proves that conv1d kernels correctly reset at cu_seqlens boundaries
when sequences are packed together.  Two sequences packed must produce
identical per-token outputs and gradients as running each separately.

Covers:
  1. ple_gate_conv  (PLE dilated conv with boundary awareness)
  2. short_conv_fp32 (residual conv with per-seq bos/eos)
  3. FLA ShortConvolution (GDN's actual conv module)
"""
import torch
import pytest


def _rel_err(a, b):
    return (a - b).abs().max().item() / max(b.abs().max().item(), 1e-12)


# ---------- ple_gate_conv ----------

@pytest.mark.parametrize("T1,T2,C,n,K,dil", [
    (20, 30, 64, 4, 4, 3),
    (5,  8,  32, 2, 3, 1),
    (1,  1,  64, 4, 4, 3),
])
def test_ple_gate_conv_packing_boundary(T1, T2, C, n, K, dil):
    """Packed (cu_seqlens) must equal separate runs, token-by-token."""
    from miles.kernels.embedding.ple_gate_conv import ple_gate_conv

    torch.manual_seed(42)
    dev = "cuda"
    W = n * C
    eps = 1e-5
    T = T1 + T2

    nk  = torch.randn(W, device=dev)
    nq  = torch.randn(W, device=dev)
    nc  = torch.randn(W, device=dev)
    cw  = torch.randn(W, 1, K, device=dev)

    hc_data  = torch.randn(T, W, device=dev)
    key_data = torch.randn(T, W, device=dev)
    val_data = torch.randn(T, C, device=dev)

    # -- packed --
    hc_p  = hc_data.clone().requires_grad_(True)
    key_p = key_data.clone().requires_grad_(True)
    val_p = val_data.clone().requires_grad_(True)
    cu = torch.tensor([0, T1, T], device=dev, dtype=torch.int32)
    out_p = ple_gate_conv(hc_p, key_p, val_p, nk, nq, nc, cw, n, eps, dil, cu)
    out_p.sum().backward()

    # -- separate --
    outs, g_hc, g_key, g_val = [], [], [], []
    for lo, hi in [(0, T1), (T1, T)]:
        h = hc_data[lo:hi].clone().requires_grad_(True)
        k = key_data[lo:hi].clone().requires_grad_(True)
        v = val_data[lo:hi].clone().requires_grad_(True)
        o = ple_gate_conv(h, k, v, nk, nq, nc, cw, n, eps, dil, cu_seqlens=None)
        o.sum().backward()
        outs.append(o.detach()); g_hc.append(h.grad); g_key.append(k.grad); g_val.append(v.grad)

    ref_out = torch.cat(outs)
    tol = 1e-4
    assert _rel_err(out_p.detach(), ref_out) < tol, f"fwd {_rel_err(out_p.detach(), ref_out)}"
    assert _rel_err(hc_p.grad,  torch.cat(g_hc))  < tol, "grad hc"
    assert _rel_err(key_p.grad, torch.cat(g_key)) < tol, "grad key"
    assert _rel_err(val_p.grad, torch.cat(g_val)) < tol, "grad val"


# ---------- short_conv_fp32 ----------

@pytest.mark.parametrize("T1,T2,D,W", [
    (20, 30, 256, 4),
    (5,  8,  64,  3),
])
def test_short_conv_fp32_packing_boundary(T1, T2, D, W):
    """short_conv_fp32 with seqlens must equal separate runs."""
    from miles.kernels.activation.short_conv_fp32 import short_conv_fp32

    torch.manual_seed(42)
    dev = "cuda"
    T = T1 + T2
    weight = torch.randn(D, 1, W, device=dev)
    x_data = torch.randn(T, D, device=dev)

    # -- packed --
    x_p = x_data.clone().requires_grad_(True)
    out_p = short_conv_fp32(x_p, weight, seqlens=[T1, T2])
    out_p.sum().backward()

    # -- separate --
    x1 = x_data[:T1].clone().requires_grad_(True)
    o1 = short_conv_fp32(x1, weight, seqlens=None); o1.sum().backward()
    x2 = x_data[T1:].clone().requires_grad_(True)
    o2 = short_conv_fp32(x2, weight, seqlens=None); o2.sum().backward()

    ref = torch.cat([o1.detach(), o2.detach()])
    ref_g = torch.cat([x1.grad, x2.grad])
    tol = 1e-4
    assert _rel_err(out_p.detach(), ref) < tol, f"fwd {_rel_err(out_p.detach(), ref)}"
    assert _rel_err(x_p.grad, ref_g) < tol, f"grad {_rel_err(x_p.grad, ref_g)}"


# ---------- FLA ShortConvolution (GDN) ----------

@pytest.mark.parametrize("T1,T2,D,K", [
    (20, 30, 256, 4),
    (5,  8,  64,  3),
])
def test_fla_short_conv_packing_boundary(T1, T2, D, K):
    """FLA ShortConvolution with cu_seqlens must equal separate runs."""
    from fla.modules import ShortConvolution

    torch.manual_seed(42)
    dev = "cuda"
    T = T1 + T2
    conv = ShortConvolution(hidden_size=D, kernel_size=K, bias=False,
                            activation="silu").to(dev)
    x_data = torch.randn(1, T, D, device=dev)

    # -- packed --
    x_p = x_data.clone().requires_grad_(True)
    cu = torch.tensor([0, T1, T], device=dev, dtype=torch.long)
    out_p, _ = conv(x_p, cu_seqlens=cu)
    out_p.sum().backward()

    # -- separate --
    x1 = x_data[:, :T1].clone().requires_grad_(True)
    o1, _ = conv(x1); o1.sum().backward()
    x2 = x_data[:, T1:].clone().requires_grad_(True)
    o2, _ = conv(x2); o2.sum().backward()

    ref = torch.cat([o1.detach(), o2.detach()], dim=1)
    ref_g = torch.cat([x1.grad, x2.grad], dim=1)
    tol = 1e-4
    assert _rel_err(out_p.detach(), ref) < tol, f"fwd {_rel_err(out_p.detach(), ref)}"
    assert _rel_err(x_p.grad, ref_g) < tol, f"grad {_rel_err(x_p.grad, ref_g)}"
