"""
CODD (Carrier-Oriented Drift Decomposition) — core theory functions.

Key claims to validate:
  C1. Fine-tuning drift Delta_F = F_t - F_s can be decomposed in the complex
      spectral plane of the source carrier e^{j*phi_s} into:
        Delta_G = alpha * e^{j*phi_s} + beta * j * e^{j*phi_s}
      where alpha = carrier-aligned radial, beta = carrier-orthogonal tangential.

  C2. Across different fine-tuned medical SAMs (SAMed, MedSA), the radial
      component dominates the drift energy (radial_ratio >> tangential_ratio).

  C3. Attack perturbations crafted along the radial direction transfer more
      effectively (degrade victim segmentation more) than tangential or
      isotropic perturbations of equal energy.

  C4. A shared radial adversarial subspace distilled from proxy data provides
      a beneficial warm-start for online attacks across unseen targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Drift decomposition
# ---------------------------------------------------------------------------

@dataclass
class DriftStats:
    radial_energy: float
    tangential_energy: float
    total_energy: float
    radial_ratio: float
    tangential_ratio: float
    angle_mean: float       # mean angle of drift in local complex plane
    angle_std: float        # std of drift angle (high = phase not preserved)
    layer_name: str = ""

    def to_dict(self) -> Dict:
        return {
            "radial_energy": self.radial_energy,
            "tangential_energy": self.tangential_energy,
            "total_energy": self.total_energy,
            "radial_ratio": self.radial_ratio,
            "tangential_ratio": self.tangential_ratio,
            "angle_mean": self.angle_mean,
            "angle_std": self.angle_std,
            "layer_name": self.layer_name,
        }


def decompose_drift(
    feat_source: torch.Tensor,   # [B, C, H, W]
    feat_target: torch.Tensor,   # [B, C, H, W]
    eps: float = 1e-8,
    aggregate: str = "mean",     # "mean" or "per_channel"
) -> DriftStats:
    """
    Decompose Delta_G = G_target - G_source in the local complex basis
    defined by the source carrier e^{j*phi_s}.

    Returns energy ratios that directly test C1/C2:
      - radial_ratio ~ 1.0  => drift is carrier-aligned (supports CODD)
      - radial_ratio ~ 0.5  => drift is isotropic (rejects CODD)
      - tangential_ratio ~ 1.0 => drift is orthogonal (strongly rejects)
    """
    if aggregate == "mean":
        fs = feat_source.mean(dim=1, keepdim=True)  # [B,1,H,W]
        ft = feat_target.mean(dim=1, keepdim=True)
    else:
        fs, ft = feat_source, feat_target

    Gs = torch.fft.fft2(fs, norm="ortho")
    Gt = torch.fft.fft2(ft, norm="ortho")

    Delta = Gt - Gs
    phase_s = torch.angle(Gs)
    carrier = torch.exp(1j * phase_s)

    # Project Delta onto the local complex basis: carrier (radial) and j*carrier (tangential)
    local = Delta * torch.conj(carrier)
    alpha = torch.real(local)   # radial coefficient
    beta = torch.imag(local)    # tangential coefficient

    radial_energy = float(alpha.square().sum().item())
    tangential_energy = float(beta.square().sum().item())
    total = radial_energy + tangential_energy

    # Drift angle in local complex plane
    angle = torch.atan2(beta, alpha + eps)

    return DriftStats(
        radial_energy=radial_energy,
        tangential_energy=tangential_energy,
        total_energy=total,
        radial_ratio=radial_energy / (total + eps),
        tangential_ratio=tangential_energy / (total + eps),
        angle_mean=float(angle.mean().item()),
        angle_std=float(angle.std().item() if angle.numel() > 1 else 0.0),
    )


def decompose_drift_per_layer(
    source_features: List[torch.Tensor],
    target_features: List[torch.Tensor],
    layer_names: Optional[List[str]] = None,
) -> List[DriftStats]:
    """Decompose drift at each layer to test where radial dominance holds."""
    results = []
    names = layer_names or [f"layer_{i}" for i in range(len(source_features))]
    for fs, ft, name in zip(source_features, target_features, names):
        stats = decompose_drift(fs, ft)
        stats.layer_name = name
        results.append(stats)
    return results


# ---------------------------------------------------------------------------
# CODD-based frequency-domain probe synthesis
# ---------------------------------------------------------------------------

def synthesize_codd_probe(
    F_clean: torch.Tensor,          # [1, C, H, W]
    F_adv: torch.Tensor,            # [1, C, H, W] (current perturbed features)
    direction: str = "radial",       # "radial" | "tangential" | "isotropic" | "misaligned"
    ads_rho: float = 1.0,
    ads_gamma: float = 1e-4,
    seed: Optional[int] = None,
    use_random_noise: bool = True,   # True = original SE-CGR style, False = deterministic drift decomposition
    detach_adv: bool = True,         # True = detach F_adv before FFT (no grad), False = allow gradients
) -> torch.Tensor:
    """
    Synthesize a structured frequency-domain drift probe.

    Two modes:
    1. use_random_noise=True (original SE-CGR): Gaussian noise modulated by
       source amplitude along the source carrier direction. Proven effective
       but ALL directions produce identical probes (random noise dominates).

    2. use_random_noise=False (CODD deterministic): Extract the ACTUAL
       radial/tangential component of the current drift Delta_G = F_adv - F_clean
       using the source carrier basis. This produces DIRECTIONALLY DISTINCT probes.

    Frequency-domain decomposition:
      Delta_G in basis {e^{j*phi_s}, j*e^{j*phi_s}}:
        radial = Re(Delta_G * conj(carrier)) * carrier
        tangential = Im(Delta_G * conj(carrier)) * (j * carrier)
    """
    if seed is not None:
        torch.manual_seed(seed)

    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    F_adv_for_fft = F_adv.detach() if detach_adv else F_adv
    G_adv = torch.fft.fftn(F_adv_for_fft, dim=(-2, -1), norm="ortho")
    A = torch.abs(G_clean)
    Phase_s = torch.angle(G_clean)

    if use_random_noise:
        # Original SE-CGR style: random noise along carrier
        current_diff = F_adv.detach() * 0.999 - F_clean.detach()
        diff_energy = torch.mean(torch.abs(current_diff))
        spectrum_energy = torch.mean(A)
        lambda_val = diff_energy / (spectrum_energy + 1e-6)
        Sigma_freq = lambda_val * (A + 1e-12).pow(ads_rho) + ads_gamma
        N_white = torch.randn_like(A, device=A.device)

        if direction == "radial":
            fft_c = (N_white * Sigma_freq) * torch.exp(1j * Phase_s)
        elif direction == "tangential":
            fft_c = (N_white * Sigma_freq) * (1j * torch.exp(1j * Phase_s))
        elif direction == "isotropic":
            N2 = torch.randn_like(A, device=A.device)
            scale = 1.0 / np.sqrt(2.0)
            fft_c = scale * (N_white * Sigma_freq) * torch.exp(1j * Phase_s) + \
                    scale * (N2 * Sigma_freq) * (1j * torch.exp(1j * Phase_s))
        elif direction == "misaligned":
            phase_offset = (torch.rand_like(Phase_s) * 2.0 - 1.0) * np.pi
            fft_c = (N_white * Sigma_freq) * torch.exp(1j * (Phase_s + phase_offset))
        else:
            raise ValueError(f"Unknown direction: {direction}")

    else:
        # CODD deterministic: actual drift decomposition
        Delta_G = G_adv - G_clean
        carrier = torch.exp(1j * Phase_s)
        local = Delta_G * torch.conj(carrier)

        if direction == "radial":
            # Radial component: Re(Delta * conj(carrier)) * carrier
            fft_c = torch.real(local) * carrier
        elif direction == "tangential":
            # Tangential component: Im(Delta * conj(carrier)) * (j * carrier)
            fft_c = torch.imag(local) * (1j * carrier)
        elif direction == "isotropic":
            # Full drift (both components)
            fft_c = Delta_G
        elif direction == "misaligned":
            # Random phase — magnitude-preserving but direction-randomized
            mag = torch.abs(Delta_G)
            rand_phase = torch.rand_like(Phase_s) * 2 * np.pi
            fft_c = mag * torch.exp(1j * rand_phase)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real


# ---------------------------------------------------------------------------
# Shared Radial Subspace (SRS)
# ---------------------------------------------------------------------------

@dataclass
class SharedSubspace:
    mean_direction: torch.Tensor    # [1, C, H, W]
    basis: torch.Tensor             # [K, D] where D = C*H*W
    shape: Tuple[int, ...]
    singular_values: torch.Tensor   # [K]
    explained_variance: float

    def to(self, device: torch.device) -> "SharedSubspace":
        self.mean_direction = self.mean_direction.to(device)
        self.basis = self.basis.to(device)
        self.singular_values = self.singular_values.to(device)
        return self

    def save(self, path: str) -> None:
        torch.save({
            "mean_direction": self.mean_direction.cpu(),
            "basis": self.basis.cpu(),
            "shape": list(self.shape),
            "singular_values": self.singular_values.cpu(),
            "explained_variance": self.explained_variance,
        }, path)

    @staticmethod
    def load(path: str, device: torch.device = torch.device("cpu")) -> "SharedSubspace":
        d = torch.load(path, map_location="cpu")
        return SharedSubspace(
            mean_direction=d["mean_direction"],
            basis=d["basis"],
            shape=tuple(d["shape"]),
            singular_values=d["singular_values"],
            explained_variance=d["explained_variance"],
        ).to(device)


def fit_shared_subspace(
    gradient_list: List[torch.Tensor],  # list of [1,C,H,W] normalized gradients
    rank: int = 8,
) -> SharedSubspace:
    """
    Fit a low-rank Shared Radial Subspace from attack gradients collected
    on proxy medical data.

    Each gradient g_i is the attack direction that successfully disrupts
    source features on a proxy image. The subspace captures directions that
    appear consistently across images = transferable attack directions.
    """
    N = len(gradient_list)
    shape = tuple(gradient_list[0].shape)
    D = int(np.prod(shape))

    # Stack flattened, L1-normalized gradients
    X = torch.stack([g.reshape(-1) for g in gradient_list], dim=0)  # [N, D]
    X = X / (X.abs().sum(dim=1, keepdim=True) + 1e-12)

    mean_vec = X.mean(dim=0, keepdim=True)  # [1, D]
    mean_vec = mean_vec / (mean_vec.abs().sum() + 1e-12)
    X_centered = X - mean_vec

    # PCA via SVD
    q = min(rank, N, D)
    U, S, V = torch.pca_lowrank(X_centered, q=q, center=False)

    basis = V[:, :q].T.contiguous()  # [q, D]
    explained_var = float(S[:q].square().sum() / (X_centered.square().sum() + 1e-12))

    return SharedSubspace(
        mean_direction=mean_vec.reshape(shape),
        basis=basis,
        shape=shape,
        singular_values=S[:q],
        explained_variance=explained_var,
    )


def project_to_subspace(
    grad: torch.Tensor,         # [1, C, H, W]
    subspace: SharedSubspace,
) -> torch.Tensor:
    """Project a gradient onto the shared radial subspace."""
    g_flat = grad.reshape(1, -1)
    coeff = g_flat @ subspace.basis.T.to(grad.device)  # [1, K]
    proj = coeff @ subspace.basis.to(grad.device)       # [1, D]
    return proj.reshape(subspace.shape)


def blend_with_subspace(
    grad: torch.Tensor,
    subspace: Optional[SharedSubspace],
    proj_weight: float = 0.15,
    mean_weight: float = 0.0,
) -> torch.Tensor:
    """Blend raw gradient with subspace projection for warm-start."""
    if subspace is None:
        return grad
    g_norm = grad / (grad.abs().sum() + 1e-12)
    g_proj = project_to_subspace(g_norm, subspace)
    g_proj_norm = g_proj / (g_proj.abs().sum() + 1e-12)
    g_mean = subspace.mean_direction.to(grad.device)
    g_mean_norm = g_mean / (g_mean.abs().sum() + 1e-12)
    blended = g_norm + proj_weight * g_proj_norm + mean_weight * g_mean_norm
    return blended / (blended.abs().sum() + 1e-12)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_wilcoxon(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "two-sided",
) -> Dict:
    """Wilcoxon signed-rank test for paired samples."""
    if len(a) != len(b) or len(a) == 0:
        return {"test": "wilcoxon", "pvalue": float("nan"), "statistic": float("nan")}
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative=alternative)
        return {"test": "wilcoxon", "statistic": float(stat), "pvalue": float(p)}
    except Exception:
        diffs = [float(x) - float(y) for x, y in zip(a, b)]
        if all(abs(d) < 1e-12 for d in diffs):
            return {"test": "wilcoxon", "statistic": 0.0, "pvalue": 1.0}
        return {"test": "wilcoxon", "error": "computation failed"}


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    diff = x.mean() - y.mean()
    pooled_std = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0 + 1e-12)
    return float(diff / pooled_std)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def compute_drift_angle_histogram(
    feat_source: torch.Tensor,
    feat_target: torch.Tensor,
    n_bins: int = 36,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histogram of drift angles for visualization."""
    fs = feat_source.mean(dim=1, keepdim=True)
    ft = feat_target.mean(dim=1, keepdim=True)
    Gs = torch.fft.fft2(fs, norm="ortho")
    Gt = torch.fft.fft2(ft, norm="ortho")
    Delta = Gt - Gs
    carrier = torch.exp(1j * torch.angle(Gs))
    local = Delta * torch.conj(carrier)
    angles = torch.atan2(torch.imag(local), torch.real(local))
    angles_np = angles.detach().cpu().numpy().flatten()
    hist, bins = np.histogram(angles_np, bins=n_bins, range=(-np.pi, np.pi))
    return hist.astype(np.float64), bins
