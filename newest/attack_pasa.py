"""
PASD: Phase-Aware Stochastic Disruption.

Core idea V1:
  probe = IFFT( N_white * A_clean^alpha * exp(j * shuffle(Phi_clean)) )

V2 Improvements:
  1. Layer-wise Spectral Decay:
     - Shallow layers: high alpha (strict amplitude prior), low sigma (less noise)
     - Deep layers: low alpha, high sigma (more random high-freq injection)
     Based on spectral bias theory: shallow→low-freq structure, deep→high-freq semantics.

  2. Adversarial Phase Inversion:
     - Top-K high-energy frequencies: precise π inversion (destructive interference)
     - Rest: random shuffling
     Upgrades P2 from "Phase Disruption" to "Phase Inversion" — mathematically guarantees
     maximal gradient push on highest-energy structural components.

Compared to SE-CGR:
  SE-CGR: probe = IFFT( N * A_clean^rho * exp(j * Phi_clean) )
  PASD:   probe = IFFT( N * A_clean^alpha * exp(j * shuffle(Phi_clean)) )
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# ---------------------------------------------------------------------------
# Probe synthesis
# ---------------------------------------------------------------------------

def synthesize_pasd_probe(
    F_clean: torch.Tensor,         # [1, C, H, W]
    sigma: float = 1.0,
    alpha: float = 1.0,            # amplitude modulation exponent (A_clean^alpha)
    seed: Optional[int] = None,
    carrier_mode: str = "shuffled", # "shuffled" | "clean" | "random" | "inversion"
    amp_mode: str = "amplitude",    # "amplitude" | "white" | "none"
    # ── V2: Layer-wise spectral decay ──
    layer_idx: int = 0,             # 0=shallow, num_layers-1=deep
    num_layers: int = 6,
    use_layer_decay: bool = False,
    # ── V2: Adversarial phase inversion ──
    top_k_ratio: float = 0.2,       # Top-K ratio of high-amp freqs to invert
    use_phase_inversion: bool = False,
) -> torch.Tensor:
    """
    Synthesize PASD frequency-domain probe.

    Carrier-phase modes:
      - "shuffled":  exp(j * shuffle(Phi_clean))   — V1: random shuffle all phases
      - "clean":     exp(j * Phi_clean)             — ablation: anchor on clean phase
      - "random":    exp(j * Phi_random)            — ablation: no structure at all
      - "inversion": Top-K π inversion + rest shuffle — V2: adversarial phase inversion

    Amplitude modes:
      - "amplitude": N_white * A_clean^alpha  — frequency prior
      - "white":     sigma * N_white           — no frequency prior
      - "none":      sigma (deterministic)     — no Gaussian

    V2: Layer-wise spectral decay (use_layer_decay=True):
      sigma_l = sigma_base * (0.5 + 1.5 * depth_ratio)  # 0.5→2.0
      alpha_l = alpha_base * (1.5 - depth_ratio)         # 1.5→0.5
      where depth_ratio = layer_idx / (num_layers - 1)
    """
    if seed is not None:
        torch.manual_seed(seed)

    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A_clean = torch.abs(G_clean)
    Phase_clean = torch.angle(G_clean)
    device = F_clean.device

    # ── V2: Layer-wise spectral decay ──
    if use_layer_decay:
        depth_ratio = layer_idx / max(num_layers - 1, 1)  # 0 (shallow) → 1 (deep)
        alpha = alpha * (1.5 - depth_ratio)                # 1.5*alpha → 0.5*alpha
        sigma = sigma * (0.5 + 1.5 * depth_ratio)          # 0.5*sigma → 2.0*sigma

    # ── Carrier phase ──
    if carrier_mode == "inversion":
        # V2: Top-K high-amplitude frequencies → π inversion (destructive interference)
        #      Rest → random shuffling
        A_flat = A_clean.flatten()
        k = max(1, int(A_flat.numel() * top_k_ratio))
        _, topk_idx = torch.topk(A_flat, k)

        shape = Phase_clean.shape
        idx = torch.randperm(Phase_clean.numel(), device=device)
        Phase_shuffled = Phase_clean.reshape(-1)[idx].reshape(shape)

        Phase_use = Phase_shuffled.clone()
        Phase_flat = Phase_use.reshape(-1)
        Phase_flat[topk_idx] = Phase_clean.reshape(-1)[topk_idx] + np.pi
        Phase_use = Phase_flat.reshape(shape)
        carrier = torch.exp(1j * Phase_use)
    elif carrier_mode == "shuffled":
        shape = Phase_clean.shape
        idx = torch.randperm(Phase_clean.numel(), device=device)
        Phase_use = Phase_clean.reshape(-1)[idx].reshape(shape)
        carrier = torch.exp(1j * Phase_use)
    elif carrier_mode == "clean":
        carrier = torch.exp(1j * Phase_clean)
    elif carrier_mode == "random":
        rand_phase = torch.rand_like(Phase_clean) * 2 * np.pi
        carrier = torch.exp(1j * rand_phase)
    else:
        raise ValueError(f"Unknown carrier_mode: {carrier_mode}")

    # ── Amplitude ──
    if amp_mode == "amplitude":
        N = torch.randn_like(A_clean, device=device)
        Sigma = sigma * A_clean.pow(alpha) + 1e-4
        fft_probe = N * Sigma * carrier
    elif amp_mode == "white":
        N = torch.randn_like(A_clean, device=device)
        fft_probe = sigma * N * carrier
    elif amp_mode == "none":
        fft_probe = sigma * carrier
    else:
        raise ValueError(f"Unknown amp_mode: {amp_mode}")

    return torch.fft.ifftn(fft_probe, dim=(-2, -1), norm="ortho").real


# Backward compatibility alias
synthesize_pasa_probe = synthesize_pasd_probe


# ---------------------------------------------------------------------------
# PASD Loss
# ---------------------------------------------------------------------------

criterion_l1 = torch.nn.L1Loss()


def _return_mean_value(x: torch.Tensor) -> torch.Tensor:
    x_abs = torch.abs(x)
    means = torch.stack([x_abs[i].mean() for i in range(len(x_abs))])
    return means.view(-1, *([1] * (x.dim() - 1)))


def calculate_pasd_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    features_layer: List[torch.Tensor],
    targets_layer: List[torch.Tensor],
    sigma: float = 1.0,
    alpha: float = 1.0,
    probe_amplification: float = 2.0,
    lambda_probe: float = 1.0,
    probe_mode: str = "pasd",          # see MODES dict
    K: int = 2,
    num_layers: int = 6,
    # ── V2 parameters ──
    use_layer_decay: bool = False,
    use_phase_inversion: bool = False,
    top_k_ratio: float = 0.2,
) -> torch.Tensor:
    """
    PASD loss: L = L_feat + lambda_probe * L_probe

    probe_mode variants:
      - "pasd":           N*A^alpha * exp(j*shuffle(Phi))      [V1 default]
      - "pasd_v2":        V2 full: inversion + layer decay       [V2 default]
      - "clean_phase":    N*A^alpha * exp(j*Phi_clean)          [w/o phase disruption]
      - "white_amp":      sigma*N * exp(j*shuffle(Phi))         [w/o amplitude prior]
      - "secgr":          N*A^rho * exp(j*Phi_clean)            [SE-CGR baseline]
      - "random_phase":   N*A^alpha * exp(j*Phi_random)         [random phase]
      - "no_gaussian":    sigma * exp(j*shuffle(Phi))           [deterministic amplitude]
      - "none":           no probe, pure L1 feature push        [no-probe baseline]

    V2 improvements (active when probe_mode="pasd_v2"):
      1. Layer-wise Spectral Decay: α↑σ↓ shallow, α↓σ↑ deep
      2. Adversarial Phase Inversion: Top-K π inversion + rest shuffle
    """
    loss_feat = criterion_l1(features * 0.9999, targets)

    if probe_mode == "none":
        return loss_feat

    # Map probe_mode to carrier_mode + amp_mode + V2 flags
    mode_map = {
        "pasd":           ("shuffled",  "amplitude", False, False),
        "pasd_v2":        ("inversion", "amplitude", True,  True),
        "clean_phase":    ("clean",     "amplitude", False, False),
        "white_amp":      ("shuffled",  "white",     False, False),
        "secgr":          ("clean",     "amplitude", False, False),
        "random_phase":   ("random",    "amplitude", False, False),
        "no_gaussian":    ("shuffled",  "none",      False, False),
    }
    if probe_mode not in mode_map:
        raise ValueError(f"Unknown probe_mode: {probe_mode}")
    carrier_mode, amp_mode, v2_decay, v2_inversion = mode_map[probe_mode]

    # Override V2 flags from explicit parameters
    if use_layer_decay:
        v2_decay = True
    if use_phase_inversion:
        v2_inversion = True
        carrier_mode = "inversion"

    # Select layers (fixed last-n)
    pre_layer = int(len(features_layer) * 0.9)
    n_avail = min(num_layers, pre_layer)
    layer_ids = list(range(pre_layer - n_avail, pre_layer))

    # Average over K noise samples
    probe_losses = []
    for k in range(K):
        diff_list = []
        for j, idx in enumerate(layer_ids):
            F_adv_i = features_layer[idx].unsqueeze(0)
            F_clean_i = targets_layer[idx].unsqueeze(0).detach()

            probe = synthesize_pasd_probe(
                F_clean=F_clean_i,
                sigma=sigma,
                alpha=alpha,
                carrier_mode=carrier_mode,
                amp_mode=amp_mode,
                # V2: pass layer index and flags
                layer_idx=j,
                num_layers=num_layers,
                use_layer_decay=v2_decay,
                use_phase_inversion=v2_inversion,
                top_k_ratio=top_k_ratio,
            )

            diff = (F_adv_i * 0.999 - F_clean_i) - probe_amplification * probe
            diff_list.append(diff.squeeze(0))

        diff = torch.stack(diff_list, dim=0)
        selected_t = torch.stack([targets_layer[i] for i in layer_ids], dim=0)
        input_mean = _return_mean_value(selected_t)
        if len(input_mean.shape) > len(diff.shape):
            input_mean = input_mean.view(diff.shape[0], 1, 1, 1)
        diff = diff / (input_mean + 1e-6)

        probe_losses.append(torch.abs(torch.mean(diff, dim=0)).mean())

    loss_probe = torch.stack(probe_losses).mean()
    return loss_feat + lambda_probe * loss_probe


# Backward compatibility
calculate_pasa_loss = calculate_pasd_loss


# ---------------------------------------------------------------------------
# PASD Attack
# ---------------------------------------------------------------------------

def _get_edge_map(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge_min = edge.reshape(edge.size(0), -1).min(dim=1)[0].reshape(-1, 1, 1, 1)
    edge_max = edge.reshape(edge.size(0), -1).max(dim=1)[0].reshape(-1, 1, 1, 1)
    return (edge - edge_min) / (edge_max - edge_min + 1e-6)


def _return_eta(momentum: torch.Tensor, step_size: float) -> torch.Tensor:
    upd = momentum * (step_size / (torch.mean(torch.abs(momentum)) + 1e-12))
    pos_mask = (momentum > 0).float()
    eta = torch.clamp(upd, -step_size, -0.25 * step_size) * (1 - pos_mask) + \
          pos_mask * torch.clamp(upd, 0.25 * step_size, step_size)
    return eta


def run_pasd_attack(
    image_encoder,
    input_image: torch.Tensor,       # [1, 3, H, W]
    eps_pix: float,
    step_pix: float,
    steps: int,
    sigma: float = 1.0,
    alpha: float = 1.0,
    probe_amplification: float = 2.0,
    lambda_probe: float = 1.0,
    probe_mode: str = "pasd",
    K: int = 2,
    num_layers: int = 6,
    edge_guidance: bool = True,
    edge_gamma: float = 3.0,
    momentum_decay: float = 0.5,
    return_trace: bool = False,
    # ── V2 parameters ──
    use_layer_decay: bool = False,
    use_phase_inversion: bool = False,
    top_k_ratio: float = 0.2,
) -> Tuple[torch.Tensor, Dict]:
    """
    Run PASD attack on a single image.

    V2 improvements (automatically enabled when probe_mode="pasd_v2"):
      1. Layer-wise Spectral Decay: α and σ vary by layer depth
      2. Adversarial Phase Inversion: Top-K π inversion + rest shuffle

    Returns:
        x_adv: [1, 3, H, W] adversarial image tensor (on device)
        trace: dict with loss curve and metadata
    """
    device = input_image.device
    max_val, min_val = input_image.max(), input_image.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

    edge_map = _get_edge_map(input_image).detach() if edge_guidance else None

    x_adv = Variable(input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True)
    momentum = torch.zeros_like(input_image).detach()

    losses = []

    for step in range(steps):
        image_encoder.zero_grad()

        input_for_encoder = x_adv
        if step == 0:
            input_for_encoder = x_adv + torch.randn_like(x_adv) * 0.0001

        features, features_layer = image_encoder(input_for_encoder)

        loss = calculate_pasd_loss(
            features=features,
            targets=targets,
            features_layer=features_layer,
            targets_layer=targets_layer,
            sigma=sigma,
            alpha=alpha,
            probe_amplification=probe_amplification,
            lambda_probe=lambda_probe,
            probe_mode=probe_mode,
            K=K,
            num_layers=num_layers,
            use_layer_decay=use_layer_decay,
            use_phase_inversion=use_phase_inversion,
            top_k_ratio=top_k_ratio,
        )

        losses.append(float(loss.detach().item()))

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        if edge_guidance and edge_map is not None:
            grad = grad * (1.0 + edge_gamma * edge_map)

        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = momentum_decay * momentum.detach() + grad
        momentum = momentum.detach()

        eta = _return_eta(momentum, step_pix).detach()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)

        eta_total = x_adv.data - input_image.data
        eta_total = torch.clamp(eta_total, -eps_pix, eps_pix)
        x_adv = Variable(input_image.data + eta_total, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, min_val, max_val), requires_grad=True)

    trace = {
        "losses": losses,
        "final_loss": losses[-1] if losses else float("nan"),
        "max_perturbation": float(eta_total.abs().max().item()),
        "mean_perturbation": float(eta_total.abs().mean().item()),
    }

    if return_trace:
        return x_adv.detach(), trace
    return x_adv.detach()


# Backward compatibility
run_pasa_attack = run_pasd_attack


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

PASD_MODES = {
    "pasd":          {"probe_mode": "pasd",          "label": "PASD-full",
                      "desc": "N*A^alpha*exp(j*shuffle(Phi))",
                      "theory": "V1: Phase disruption + amplitude prior"},
    "pasd_v2":       {"probe_mode": "pasd_v2",       "label": "PASD V2",
                      "desc": "Inversion+LayerDecay: pi-invert top-K, alpha/sigma per layer",
                      "theory": "V2: Adversarial phase inversion + layer-wise spectral decay"},
    "clean_phase":   {"probe_mode": "clean_phase",   "label": "PASD w/o phase disr",
                      "desc": "N*A^alpha*exp(j*Phi_clean)",
                      "theory": "Ablation: no phase disruption"},
    "white_amp":     {"probe_mode": "white_amp",     "label": "PASD w/o amp prior",
                      "desc": "sigma*N*exp(j*shuffle(Phi))",
                      "theory": "Ablation: no amplitude prior"},
    "secgr":         {"probe_mode": "secgr",         "label": "SE-CGR",
                      "desc": "N*A^rho*exp(j*Phi_clean)",
                      "theory": "Old baseline: clean phase + amp modulation"},
    "random_phase":  {"probe_mode": "random_phase",  "label": "Random phase",
                      "desc": "N*A^alpha*exp(j*Phi_random)",
                      "theory": "Ablation: random phase, not shuffled"},
    "no_gaussian":   {"probe_mode": "no_gaussian",   "label": "PASD w/o Gaussian",
                      "desc": "sigma*exp(j*shuffle(Phi))",
                      "theory": "Ablation: deterministic amplitude"},
    "none":          {"probe_mode": "none",           "label": "Pure L1 (no probe)",
                      "desc": "L1 feature push only",
                      "theory": "Ablation: no frequency domain"},
}

# Backward compatibility alias
PASA_MODES = PASD_MODES


def run_pasd_attack_wrapper(image_encoder, x, eps_pix, step_pix, steps,
                            edge_map_fn=None, **kwargs):
    """Wrapper compatible with existing experiment script patterns."""
    probe_mode = kwargs.pop("probe_mode", "pasd")
    return run_pasd_attack(
        image_encoder=image_encoder,
        input_image=x,
        eps_pix=eps_pix,
        step_pix=step_pix,
        steps=steps,
        probe_mode=probe_mode,
        **kwargs,
    )
