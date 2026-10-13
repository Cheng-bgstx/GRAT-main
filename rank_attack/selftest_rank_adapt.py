"""Small CPU tests for the core linear algebra. No SAM repository required."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rank_adapt_utils import matrix_q_from_io, worst_case_lowrank_update


def test_q_formula():
    torch.manual_seed(0)
    n, din, dout = 11, 5, 7
    x = torch.randn(n, din)
    w = torch.randn(dout, din)
    dw = torch.zeros_like(w, requires_grad=True)
    target = torch.randn(n, dout)
    y = F.linear(x, w + dw)
    loss = (y * target).sum()
    q_auto = torch.autograd.grad(loss, dw)[0]
    q_formula = matrix_q_from_io(x, target)
    err = (q_auto - q_formula).abs().max().item()
    assert err < 1e-6, err
    print(f"[ok] Q formula max error={err:.3e}")



def test_shared_clean_adv_q():
    torch.manual_seed(3)
    n, din, dout = 13, 4, 6
    xc = torch.randn(n, din)
    xa = torch.randn(n, din)
    w = torch.randn(dout, din)
    dw = torch.zeros_like(w, requires_grad=True)
    yc = F.linear(xc, w + dw)
    ya = F.linear(xa, w + dw)
    loss = F.l1_loss(ya, yc)
    q_auto = torch.autograd.grad(loss, dw)[0]

    # Rebuild the two branch output gradients explicitly at DeltaW=0.
    yc0 = F.linear(xc, w).detach().requires_grad_(True)
    ya0 = F.linear(xa, w).detach().requires_grad_(True)
    loss0 = F.l1_loss(ya0, yc0)
    gyc, gya = torch.autograd.grad(loss0, [yc0, ya0])
    q_formula = matrix_q_from_io(xc, gyc) + matrix_q_from_io(xa, gya)
    err = (q_auto - q_formula).abs().max().item()
    assert err < 1e-6, err
    print(f"[ok] shared clean/adv Q max error={err:.3e}")


def test_worst_rank_r():
    torch.manual_seed(1)
    q = torch.randn(7, 5)
    r, rho = 2, 0.3
    dw, sr = worst_case_lowrank_update(q, r, rho, oversample=0, niter=8)
    s = torch.linalg.svdvals(q)
    sr_true = torch.sqrt((s[:r] ** 2).sum()).item()
    inner = (q * dw).sum().item()
    frob = dw.norm().item()
    rank_num = int(torch.linalg.matrix_rank(dw, tol=1e-5).item())
    assert abs(sr - sr_true) < 1e-5, (sr, sr_true)
    assert abs(frob - rho) < 1e-5, frob
    assert rank_num <= r, rank_num
    assert abs(inner - (-rho * sr_true)) < 1e-5, (inner, -rho * sr_true)
    print(f"[ok] worst rank-r: rank={rank_num}, ||dw||F={frob:.6f}, inner={inner:.6f}")


def test_linear_hook_identity():
    torch.manual_seed(2)
    lin = nn.Linear(5, 7, bias=True)
    x = torch.randn(3, 5)
    dw = torch.randn(7, 5) * 0.01

    def hook(module, inp, out):
        return out + F.linear(inp[0], dw, None)

    h = lin.register_forward_hook(hook)
    y_hook = lin(x)
    h.remove()
    y_ref = F.linear(x, lin.weight + dw, lin.bias)
    err = (y_hook - y_ref).abs().max().item()
    assert err < 1e-6, err
    print(f"[ok] linear injection max error={err:.3e}")


if __name__ == "__main__":
    test_q_formula()
    test_shared_clean_adv_q()
    test_worst_rank_r()
    test_linear_hook_identity()
    print("all rank-adaptation self-tests passed")
