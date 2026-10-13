"""
LoRA ΔW ≈ B@A singular values and Frobenius norms; MSA Adapter linear SVD.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _lora_delta_qv(linear_a: nn.Linear, linear_b: nn.Linear) -> torch.Tensor:
    """Linear: y = x @ A.T -> shapes (out_dim, in_dim) for weight."""
    # A: (r, in), B: (out, r) -> Delta = B @ A -> (out, in)
    return linear_b.weight @ linear_a.weight


def analyze_samed_lora(net: Any) -> Dict[str, List[float]]:
    """
    net: LoRA_Sam with .w_As, .w_Bs interleaved q then v per layer.
    Returns per-layer singular value mean and Frobenius norm of ΔW for q/v.
    """
    out: Dict[str, List[float]] = {
        "layer_idx": [],
        "fro_q": [],
        "fro_v": [],
        "smax_q": [],
        "smax_v": [],
        "smean_q": [],
        "smean_v": [],
    }
    n = len(net.w_As)
    if n % 2 != 0:
        raise ValueError("Expected paired w_As/w_Bs")
    for li in range(0, n, 2):
        layer_i = li // 2
        aq, bq = net.w_As[li], net.w_Bs[li]
        av, bv = net.w_As[li + 1], net.w_Bs[li + 1]
        dq = _lora_delta_qv(aq, bq)
        dv = _lora_delta_qv(av, bv)
        out["layer_idx"].append(layer_i)
        out["fro_q"].append(float(torch.linalg.norm(dq, ord="fro").item()))
        out["fro_v"].append(float(torch.linalg.norm(dv, ord="fro").item()))
        sq = torch.linalg.svdvals(dq)
        sv = torch.linalg.svdvals(dv)
        out["smax_q"].append(float(sq[0].item()))
        out["smax_v"].append(float(sv[0].item()))
        out["smean_q"].append(float(sq.mean().item()))
        out["smean_v"].append(float(sv.mean().item()))
    return out


def analyze_msa_adapters(net: nn.Module) -> Dict[str, List[float]]:
    """
    Walk image_encoder.blocks for AdapterBlock; collect D_fc1/D_fc2 product Frobenius.
    """
    out: Dict[str, List[float]] = {
        "block_idx": [],
        "space_adapter_fro": [],
        "mlp_adapter_fro": [],
        "depth_adapter_fro": [],
    }
    enc = net.image_encoder
    for i, blk in enumerate(enc.blocks):
        if not hasattr(blk, "Space_Adapter"):
            continue
        def _two_layer_fro(m):
            if m is None:
                return 0.0
            w1 = m.D_fc1.weight
            w2 = m.D_fc2.weight
            # bottleneck: R^{D x d} @ R^{d x D} -> rank-d
            prod = w2 @ w1
            return float(torch.linalg.norm(prod, ord="fro").item())

        out["block_idx"].append(i)
        out["space_adapter_fro"].append(_two_layer_fro(getattr(blk, "Space_Adapter", None)))
        out["mlp_adapter_fro"].append(_two_layer_fro(getattr(blk, "MLP_Adapter", None)))
        out["depth_adapter_fro"].append(_two_layer_fro(getattr(blk, "Depth_Adapter", None)))
    return out


def collect_lora_singular_values(net: Any) -> Tuple[np.ndarray, np.ndarray]:
    """All singular values of ΔW_q and ΔW_v across LoRA layers."""
    vals_q = []
    vals_v = []
    n = len(net.w_As)
    for li in range(0, n, 2):
        dq = _lora_delta_qv(net.w_As[li], net.w_Bs[li])
        dv = _lora_delta_qv(net.w_As[li + 1], net.w_Bs[li + 1])
        vals_q.append(torch.linalg.svdvals(dq).detach().cpu().numpy())
        vals_v.append(torch.linalg.svdvals(dv).detach().cpu().numpy())
    return np.concatenate(vals_q), np.concatenate(vals_v)


def random_baseline_singular_values(dim: int, rank: int, n_matrices: int, seed: int = 0) -> np.ndarray:
    g = torch.Generator()
    g.manual_seed(seed)
    parts = []
    for _ in range(n_matrices):
        a = torch.randn(rank, dim, generator=g)
        b = torch.randn(dim, rank, generator=g)
        parts.append(torch.linalg.svdvals(b @ a).numpy())
    return np.concatenate(parts)


def plot_lora_singular_histograms(
    sq: np.ndarray,
    sv: np.ndarray,
    s_rand: np.ndarray,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.hist(sq, bins=50, alpha=0.45, label="LoRA ΔW_q σ", density=True)
    plt.hist(sv, bins=50, alpha=0.45, label="LoRA ΔW_v σ", density=True)
    plt.hist(s_rand, bins=50, alpha=0.35, label="Random rank-r B@A σ", density=True)
    plt.xlabel("Singular value")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.savefig(out_path, dpi=150)
    plt.close()


def random_baseline_lora_fro(dim: int, rank: int, n_layers: int, seed: int = 0) -> List[float]:
    """Same-rank random B@A Frobenius norms for comparison."""
    g = torch.Generator()
    g.manual_seed(seed)
    fro = []
    for _ in range(n_layers):
        a = torch.randn(rank, dim, generator=g)
        b = torch.randn(dim, rank, generator=g)
        fro.append(float(torch.linalg.norm(b @ a, ord="fro").item()))
    return fro
