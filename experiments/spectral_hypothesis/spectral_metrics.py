"""
FFT-based ΔA / ΔΦ metrics, radial binning, masked statistics, paired tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def aggregate_channels(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """x: [B, C, H, W] -> [B, 1, H, W]"""
    if mode == "mean":
        return x.mean(dim=1, keepdim=True)
    if mode == "l2":
        return x.norm(dim=1, keepdim=True)
    raise ValueError(mode)


def fft2_amplitude_phase(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    f: [B, 1, H, W] real
    Returns G (complex), A = |G|, Phi = angle(G)
    """
    g = torch.fft.fft2(f, norm="ortho")
    a = torch.abs(g)
    phi = torch.angle(g)
    return g, a, phi


def delta_spectrum_masked(
    g_s: torch.Tensor,
    g_t: torch.Tensor,
    eps: float = 1e-8,
    mask_tau: float = 1e-3,
    use_log_amp: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ΔA = A_t - A_s (or log ratio), ΔΦ = angle(G_t / (G_s + eps)).
    mask: valid where |G_s| >= mask_tau * max(|G_s|) (relative threshold per map).
    """
    a_s = torch.abs(g_s)
    a_t = torch.abs(g_t)
    if use_log_amp:
        la_s = torch.log(a_s + eps)
        la_t = torch.log(a_t + eps)
        delta_a = la_t - la_s
    else:
        delta_a = a_t - a_s
    ratio = g_t / (g_s + eps)
    delta_phi = torch.angle(ratio)
    # relative magnitude mask
    ref = a_s.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    mask = (a_s / ref) >= mask_tau
    return delta_a, delta_phi, mask


def masked_var(x: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.float()
    if m.sum() < 1:
        return float("nan")
    xm = x[mask]
    return float(xm.var(unbiased=False).item())


def circular_variance(phi: torch.Tensor, mask: torch.Tensor) -> float:
    """1 - |E[e^{i phi}]| on masked samples."""
    m = mask
    if m.sum() < 1:
        return float("nan")
    z = torch.exp(1j * phi[m].double())
    r = z.mean().abs().item()
    return float(1.0 - r)


def radial_bins(h: int, w: int, n_bins: int, device) -> torch.Tensor:
    """Integer radius bin index per pixel in frequency grid [H,W]."""
    yy = torch.fft.fftfreq(h, d=1.0, device=device)
    xx = torch.fft.fftfreq(w, d=1.0, device=device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(gy * gy + gx * gx)
    mx = r.max().clamp_min(1e-8)
    bins = (r / mx * (n_bins - 1e-6)).long().clamp(0, n_bins - 1)
    return bins


def radial_triplet_neck(
    neck_s: torch.Tensor,
    neck_t: torch.Tensor,
    n_bins: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    neck_* [1,C,H,W] single image. Returns radial mean |G_s|, |G_t|, |ΔA| per bin.
    """
    fs = aggregate_channels(neck_s, "mean")
    ft = aggregate_channels(neck_t, "mean")
    if fs.shape[-2:] != ft.shape[-2:]:
        ft = F.interpolate(ft, size=fs.shape[-2:], mode="bilinear", align_corners=False)
    _, a_s, _ = fft2_amplitude_phase(fs)
    _, a_t, _ = fft2_amplitude_phase(ft)
    rs = radial_profile(a_s, n_bins)
    rt = radial_profile(a_t, n_bins)
    da = radial_profile((a_t - a_s).abs(), n_bins)
    return rs, rt, da


def radial_profile(amplitude: torch.Tensor, n_bins: int = 32) -> np.ndarray:
    """amplitude [1,1,H,W], returns mean |.| per radial bin."""
    _, _, h, w = amplitude.shape
    dev = amplitude.device
    bins = radial_bins(h, w, n_bins, dev)
    prof = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    a = amplitude[0, 0].detach().cpu().numpy().ravel()
    b = bins.detach().cpu().numpy().ravel()
    for i in range(n_bins):
        sel = b == i
        if sel.any():
            prof[i] = a[sel].mean()
            counts[i] = sel.sum()
    return prof


def paired_wilcoxon_greater(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """H0: median(x-y)=0; returns statistic, p-value (two-sided scipy). x,y same length."""
    try:
        from scipy.stats import wilcoxon

        d = x - y
        if np.allclose(d, 0):
            return 0.0, 1.0
        res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


@dataclass
class LayerStats:
    layer_name: str
    var_delta_a: List[float]
    circ_var_delta_phi: List[float]
    mean_abs_delta_a: List[float]
    mean_abs_delta_phi: List[float]


def summarize_layer(layer_stats: LayerStats) -> Dict[str, float]:
    va = np.array(layer_stats.var_delta_a, dtype=np.float64)
    vp = np.array(layer_stats.circ_var_delta_phi, dtype=np.float64)
    return {
        "var_delta_a_mean": float(np.nanmean(va)),
        "var_delta_a_std": float(np.nanstd(va)),
        "circ_var_phi_mean": float(np.nanmean(vp)),
        "circ_var_phi_std": float(np.nanstd(vp)),
        "ratio_var_over_circ": float(np.nanmean(va / np.maximum(vp, 1e-12))),
    }


def process_pair_features(
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    mask_tau: float,
    use_log_amp: bool,
) -> Dict[str, float]:
    """feat_* [B,C,H,W]"""
    fs = aggregate_channels(feat_s, "mean")
    ft = aggregate_channels(feat_t, "mean")
    g_s, _, _ = fft2_amplitude_phase(fs)
    g_t, _, _ = fft2_amplitude_phase(ft)
    d_a, d_phi, mask = delta_spectrum_masked(g_s, g_t, mask_tau=mask_tau, use_log_amp=use_log_amp)
    return {
        "var_delta_a": masked_var(d_a, mask),
        "circ_var_delta_phi": circular_variance(d_phi, mask),
        "mean_abs_delta_a": float(d_a[mask].abs().mean().item()) if mask.any() else float("nan"),
        "mean_abs_delta_phi": float(d_phi[mask].abs().mean().item()) if mask.any() else float("nan"),
    }
