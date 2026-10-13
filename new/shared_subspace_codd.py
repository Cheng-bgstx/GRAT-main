"""
Shared radial subspace utilities for DIIA-CODD.

The goal is deliberately conservative:
- do NOT replace the strong online Gaussian-CODD objective;
- only provide optional warm-start / gradient bonus in image space.

Saved bundle format:
{
    "mean_direction": Tensor [1,C,H,W],
    "basis": Tensor [K, D],
    "shape": [1,C,H,W],
    "meta": {...}
}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SharedSubspaceBundle:
    mean_direction: torch.Tensor  # [1,C,H,W]
    basis: torch.Tensor           # [K, D]
    shape: Tuple[int, int, int, int]
    meta: Dict

    def to(self, device: torch.device) -> "SharedSubspaceBundle":
        self.mean_direction = self.mean_direction.to(device)
        self.basis = self.basis.to(device)
        return self


def _flatten_direction(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"expect [1,C,H,W], got {tuple(x.shape)}")
    return x.reshape(1, -1)


def normalize_l1(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.abs().sum() + eps)


def vector_list_to_matrix(vectors: List[torch.Tensor]) -> torch.Tensor:
    rows = []
    for v in vectors:
        rows.append(normalize_l1(_flatten_direction(v)).squeeze(0))
    return torch.stack(rows, dim=0)


def fit_shared_subspace(
    vectors: List[torch.Tensor],
    rank: int = 8,
) -> SharedSubspaceBundle:
    if len(vectors) < 2:
        raise ValueError("Need at least 2 vectors to fit a subspace.")
    shape = tuple(vectors[0].shape)
    X = vector_list_to_matrix(vectors)  # [N, D]
    mean = normalize_l1(X.mean(dim=0, keepdim=True))
    Xc = X - mean
    q = min(rank, Xc.shape[0], Xc.shape[1])
    # torch.pca_lowrank returns V [D,q]
    U, S, V = torch.pca_lowrank(Xc, q=q, center=False)
    basis = V[:, :q].T.contiguous()  # [q, D]
    mean_direction = mean.reshape(shape)
    meta = {
        "num_vectors": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "rank": int(q),
        "singular_values": S[:q].cpu(),
    }
    return SharedSubspaceBundle(
        mean_direction=mean_direction,
        basis=basis,
        shape=shape,
        meta=meta,
    )


def save_shared_subspace(bundle: SharedSubspaceBundle, path: str) -> None:
    payload = {
        "mean_direction": bundle.mean_direction.detach().cpu(),
        "basis": bundle.basis.detach().cpu(),
        "shape": list(bundle.shape),
        "meta": bundle.meta,
    }
    torch.save(payload, path)


def load_shared_subspace(path: str, device: Optional[torch.device] = None) -> SharedSubspaceBundle:
    payload = torch.load(path, map_location="cpu")
    bundle = SharedSubspaceBundle(
        mean_direction=payload["mean_direction"],
        basis=payload["basis"],
        shape=tuple(payload["shape"]),
        meta=payload.get("meta", {}),
    )
    if device is not None:
        bundle = bundle.to(device)
    return bundle


def project_to_shared_subspace(
    grad: torch.Tensor,
    bundle: SharedSubspaceBundle,
    coeff_clip: Optional[float] = None,
) -> torch.Tensor:
    """
    grad: [1,C,H,W]
    Returns projected image-space direction with same shape.
    """
    if tuple(grad.shape) != tuple(bundle.shape):
        raise ValueError(f"shape mismatch: grad={tuple(grad.shape)} bundle={bundle.shape}")
    g = _flatten_direction(grad)  # [1,D]
    B = bundle.basis  # [K,D]
    coeff = g @ B.T  # [1,K]
    if coeff_clip is not None:
        coeff = coeff.clamp(min=-coeff_clip, max=coeff_clip)
    proj = coeff @ B  # [1,D]
    proj = proj.reshape(bundle.shape)
    return normalize_l1(proj)


def blend_gradient_with_subspace(
    grad: torch.Tensor,
    bundle: Optional[SharedSubspaceBundle],
    proj_weight: float = 0.15,
    mean_weight: float = 0.0,
) -> torch.Tensor:
    if bundle is None:
        return grad
    g = normalize_l1(grad)
    g_proj = project_to_shared_subspace(g, bundle)
    g_mix = g + proj_weight * g_proj + mean_weight * normalize_l1(bundle.mean_direction)
    return normalize_l1(g_mix)


__all__ = [
    "SharedSubspaceBundle",
    "normalize_l1",
    "fit_shared_subspace",
    "save_shared_subspace",
    "load_shared_subspace",
    "project_to_shared_subspace",
    "blend_gradient_with_subspace",
]
