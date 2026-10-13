"""
CODD-based adversarial attack on fine-tuned SAM models.

Preserves the proven Gaussian frequency-domain modulation from SE-CGR,
reinterpreted under CODD theory: phase is the source carrier, not an
invariant of the target model.

Attack flow:
  1. Run clean image through SOURCE SAM -> get reference features
  2. Iteratively perturb image, run through SOURCE SAM
  3. Loss = L_org(features, targets) + L_ads(features, targets)
     - L_org: L1 distance between perturbed and clean features
     - L_ads: distance between real drift and CODD-structured probe
  4. Gradient w.r.t. input, momentum update, project to epsilon-ball
  5. Output: adversarial image

Offline phase:
  - Collect normalized attack gradients on proxy training data
  - Fit Shared Radial Subspace (SRS) via PCA
  - Save subspace + mean direction for online warm-start

Online phase:
  - Load SRS bundle
  - Warm-start momentum with SRS mean direction
  - Blend online gradients with SRS projection
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from theory import (
    SharedSubspace,
    blend_with_subspace,
    synthesize_codd_probe,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AttackConfig:
    # Attack hyperparameters
    att_step_size: float = 2.0       # step size in pixel space (out of 255)
    att_bound: float = 10.0          # L_inf bound (out of 255)
    att_numstep: int = 10            # number of PGD steps (10 = proven effective)
    momentum_decay: float = 0.5

    # CODD / ADS parameters
    ads_rho: float = 1.0             # power for amplitude envelope
    ads_gamma: float = 1e-4          # small floor for stability
    ads_tangential_beta: float = 0.0 # tangential component weight (0 = radial only)
    probe_amplification: float = 2.0 # amplify probe to balance directional gradients
                                     # 1.0=identity (tangential gets stronger grad since radial>>tangential)
                                     # 2.0=balanced (both directions get comparable gradient magnitudes)

    # Edge guidance (PSAA)
    edge_guidance: bool = True
    edge_gamma: float = 3.0

    # Direction mode for ablation studies
    direction_mode: str = "radial"   # "radial" | "tangential" | "isotropic" | "misaligned"

    # Direct gradient injection weight for directional component
    directional_weight: float = 1.0  # weight for maximizing directional drift energy

    # Shared Radial Subspace
    use_srs: bool = False
    srs_path: str = ""
    srs_proj_weight: float = 0.15
    srs_mean_weight: float = 0.1

    # Common perturbation (UAP from SE-CGR)
    use_common_perturbation: bool = False
    common_perturbation_path: str = ""

    # Random seed
    seed: int = 42

    # Whether to use ImageNet normalization (for 2D datasets like ISIC/REFUGE)
    use_imagenet_norm: bool = False


# ---------------------------------------------------------------------------
# Core attack
# ---------------------------------------------------------------------------

criterion_l1 = torch.nn.L1Loss()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_edge_map(x: torch.Tensor) -> torch.Tensor:
    """Compute normalized anatomical edge map for PSAA guidance."""
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    edge_min = edge.reshape(edge.size(0), -1).min(dim=1)[0].reshape(-1, 1, 1, 1)
    edge_max = edge.reshape(edge.size(0), -1).max(dim=1)[0].reshape(-1, 1, 1, 1)
    return (edge - edge_min) / (edge_max - edge_min + 1e-6)


def _return_mean_value(x: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute value per sample, reshape for broadcasting."""
    x_abs = torch.abs(x)
    means = torch.stack([x_abs[i].mean() for i in range(len(x_abs))])
    return means.view(-1, *([1] * (x.dim() - 1)))


def _return_eta(momentum: torch.Tensor, step_size: float) -> torch.Tensor:
    """Convert momentum to update step with asymmetric clamping."""
    upd = momentum * (step_size / (torch.mean(torch.abs(momentum)) + 1e-12))
    dev = momentum.device
    pos_mask = (momentum > 0).float()
    eta = torch.clamp(upd, -step_size, -0.25 * step_size) * (1 - pos_mask) + \
          pos_mask * torch.clamp(upd, 0.25 * step_size, step_size)
    return eta


def calculate_codd_loss(
    features: torch.Tensor,          # [B, C_out, H', W'] — final encoder output
    targets: torch.Tensor,           # [B, C_out, H', W'] — clean reference
    features_layer: List[torch.Tensor],
    targets_layer: List[torch.Tensor],
    config: AttackConfig,
    use_random_noise: bool = False,  # True = original SE-CGR random noise, False = CODD deterministic
) -> torch.Tensor:
    """
    CODD-structured loss matching the proven SE-CGR mechanism.

    L = L_org + L_ads

    L_org  = L1(0.9999*features, targets)
    L_ads  = L1(mean_over_layers((F_adv - F_clean) - xi_struct) / input_mean)

    xi_struct (probe) can be either:
      - use_random_noise=True: Gaussian noise along source carrier (original SE-CGR)
      - use_random_noise=False: DETERMINISTIC radial/tangential/isotropic/misaligned
        component of the actual drift Delta_G, decomposed in the source carrier basis.

    The key insight (CODD): by replacing random noise with the deterministic
    directional component, different direction_mode choices produce genuinely
    different attack gradients, enabling fair comparison of radial vs tangential.
    """
    loss_org = criterion_l1(features * 0.9999, targets)

    # Multi-layer directional loss — EXACT same structure as original SE-CGR
    pre_layer = int(len(features_layer) * 0.9)
    num_layers = min(6, pre_layer)
    indices = random.sample(range(num_layers), max(1, num_layers - 2))
    selected_f = torch.stack([features_layer[i + pre_layer - num_layers] for i in indices])
    selected_t = torch.stack([targets_layer[i + pre_layer - num_layers] for i in indices])

    feature_difference_list = []
    for i in range(selected_f.shape[0]):
        F_clean = selected_t[i].unsqueeze(0)
        F_adv = selected_f[i].unsqueeze(0)

        # Generate directional probe
        probe = synthesize_codd_probe(
            F_clean=F_clean,
            F_adv=F_adv,
            direction=config.direction_mode,
            ads_rho=config.ads_rho,
            ads_gamma=config.ads_gamma,
            use_random_noise=use_random_noise,
        )

        # Push features toward F_clean + amp * probe (direction-guided drift)
        # Amplification > 1.0 ensures comparable gradient magnitudes across directions:
        #   radial:  |drift - amp*radial| ≈ |(1-amp)*radial|    ≈ |radial|
        #   tangential: |drift - amp*tangential| ≈ |radial|     ≈ |radial|
        # With amp=1.0, tangential gets stronger gradients since radial >> tangential.
        feature_difference = (F_adv * 0.999 - F_clean) - config.probe_amplification * probe
        feature_difference_list.append(feature_difference.squeeze(0))

    feature_difference = torch.stack(feature_difference_list)
    input_mean = _return_mean_value(selected_t)
    if len(input_mean.shape) > len(feature_difference.shape):
        input_mean = input_mean.view(feature_difference.shape[0], 1, 1, 1)
    feature_difference = feature_difference / (input_mean + 1e-6)

    l1_losses = (torch.abs(torch.mean(feature_difference, dim=0))).mean()

    return loss_org + l1_losses


def run_codd_attack(
    image_encoder,                    # source SAM image_encoder
    input_image: torch.Tensor,       # [1, 3, H, W] clean image
    config: AttackConfig,
    srs_bundle: Optional[SharedSubspace] = None,
    common_perturbation: Optional[torch.Tensor] = None,
    return_trace: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Run CODD-based adversarial attack on a single image.

    Returns:
        adv_image: [3, H, W] numpy array (adversarial example)
        trace: dict with loss curve and metadata
    """
    device = input_image.device
    max_val, min_val = input_image.max(), input_image.min()
    step_size = float((max_val - min_val) * config.att_step_size / 255.0)
    epsilon = float((max_val - min_val) * config.att_bound / 255.0)

    # Get clean reference features (NO gradient needed)
    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

    # Compute anatomical edge map once
    edge_map = _get_edge_map(input_image).detach() if config.edge_guidance else None

    # Initialize perturbation
    eta_init = torch.randn_like(input_image) * 1e-5
    perturb_img = Variable(input_image.data + eta_init, requires_grad=True)

    # Initialize momentum
    momentum = torch.zeros_like(input_image).detach()

    # Warm-start from common perturbation (UAP)
    if config.use_common_perturbation and common_perturbation is not None:
        cp_norm = torch.norm(common_perturbation, p=1)
        if cp_norm > 1e-6:
            momentum = (common_perturbation / cp_norm).detach()

    # Warm-start from SRS mean direction
    if srs_bundle is not None and config.srs_mean_weight > 0:
        mean_dir = srs_bundle.mean_direction.to(device)
        mean_dir_norm = mean_dir / (mean_dir.abs().sum() + 1e-12)
        momentum = momentum + config.srs_mean_weight * mean_dir_norm
        momentum = momentum.detach()

    losses = []

    for step in range(config.att_numstep):
        image_encoder.zero_grad()

        # Add tiny noise on first step to break symmetry when no warm-start
        perturb_input = perturb_img
        if step == 0 and not config.use_common_perturbation:
            perturb_input = perturb_img + torch.randn_like(perturb_img) * 0.0001

        features, features_layer = image_encoder(perturb_input)

        # 1. Base disruption: push features AWAY from clean (same as original SE-CGR)
        loss_org = criterion_l1(features * 0.9999, targets)

        # 2. Multi-layer directional energy maximization
        #    Compute ||probe_direction||^2 for selected layers and maximize it.
        #    This directly injects directional preference into the gradient.
        pre_layer = int(len(features_layer) * 0.9)
        num_layers = min(6, pre_layer)
        n_pick = max(1, num_layers - 2)
        indices = random.sample(range(num_layers), n_pick)

        dir_energy = 0.0
        for idx in indices:
            F_clean_i = targets_layer[idx + pre_layer - num_layers]
            F_adv_i = features_layer[idx + pre_layer - num_layers]

            probe = synthesize_codd_probe(
                F_clean=F_clean_i.unsqueeze(0),
                F_adv=F_adv_i.unsqueeze(0),
                direction=config.direction_mode,
                ads_rho=config.ads_rho,
                ads_gamma=config.ads_gamma,
                use_random_noise=False,
                detach_adv=False,  # Allow gradients through F_adv → probe → energy
            )
            dir_energy += (probe ** 2).mean()

        dir_energy = dir_energy / n_pick

        # Combined loss: INCREASE disruption AND MAXIMIZE directional drift
        # Negative sign on dir_energy because gradient ASCENT increases it
        loss = loss_org - config.directional_weight * dir_energy
        losses.append(float(loss.detach().item()))

        grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]

        # Edge-weighted gradient guidance (PSAA)
        if config.edge_guidance and edge_map is not None:
            grad = grad * (1.0 + config.edge_gamma * edge_map)

        # Normalize gradient
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        # Blend with shared subspace
        if srs_bundle is not None:
            grad = blend_with_subspace(
                grad, srs_bundle,
                proj_weight=config.srs_proj_weight,
                mean_weight=0.0,
            )

        # Momentum update
        momentum = config.momentum_decay * momentum.detach() + grad
        momentum = momentum.detach()

        eta = _return_eta(momentum, step_size).detach()
        perturb_img = Variable(perturb_img.data + eta, requires_grad=True)

        # Project to epsilon ball
        eta_total = perturb_img.data - input_image.data
        eta_total = torch.clamp(eta_total, -epsilon, epsilon)
        perturb_img = Variable(input_image.data + eta_total, requires_grad=True)

        # Clamp to valid pixel range
        perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

    adv_image = perturb_img[0].detach().cpu().numpy()

    trace = {
        "losses": losses,
        "final_loss": losses[-1] if losses else float("nan"),
        "max_perturbation": float(eta_total.abs().max().item()),
        "mean_perturbation": float(eta_total.abs().mean().item()),
        "config": {k: str(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                   for k, v in config.__dict__.items()},
    }

    if return_trace:
        return adv_image, trace
    return adv_image


# ---------------------------------------------------------------------------
# Offline SRS distillation
# ---------------------------------------------------------------------------

def distill_srs(
    image_encoder,
    dataloader,
    config: AttackConfig,
    num_samples: int = 128,
    rank: int = 8,
    device: torch.device = torch.device("cuda"),
    is_3d: bool = False,
) -> SharedSubspace:
    """
    Offline phase: collect attack gradients on proxy training data,
    fit Shared Radial Subspace via PCA.
    """
    from theory import fit_shared_subspace

    gradient_list = []
    seen = 0

    for batch in dataloader:
        if seen >= num_samples:
            break

        image_tensor = batch["image"]
        if is_3d and image_tensor.dim() == 4:
            # Take middle slice for efficiency
            mid = image_tensor.shape[1] // 2
            image_tensor = image_tensor[:, mid:mid+1, :, :]

        # Handle shape variations
        if image_tensor.dim() == 5:
            image_tensor = image_tensor.squeeze(0)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        input_image = image_tensor.float().to(device)

        with torch.no_grad():
            targets, targets_layer = image_encoder(input_image)

        max_val, min_val = input_image.max(), input_image.min()
        step_size = float((max_val - min_val) * config.att_step_size / 255.0)
        epsilon = float((max_val - min_val) * config.att_bound / 255.0)

        perturb_img = Variable(input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True)
        edge_map = _get_edge_map(input_image).detach() if config.edge_guidance else None
        local_grads = []

        for _ in range(min(3, config.att_numstep)):  # 3-step lightweight attacks
            image_encoder.zero_grad()
            features, features_layer = image_encoder(perturb_img)
            loss = calculate_codd_loss(features, targets, features_layer, targets_layer, config)
            grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]

            if config.edge_guidance and edge_map is not None:
                grad = grad * (1.0 + config.edge_gamma * edge_map)

            grad_norm = grad.detach() / (grad.detach().abs().sum() + 1e-12)
            local_grads.append(grad_norm.cpu())

            # Simple step for quick gradient collection
            perturb_img = Variable(
                torch.clamp(perturb_img.data + step_size * grad_norm.sign(),
                           min_val, max_val),
                requires_grad=True,
            )
            eta = torch.clamp(perturb_img.data - input_image.data, -epsilon, epsilon)
            perturb_img = Variable(input_image.data + eta, requires_grad=True)

        # Average gradients across steps for this image
        avg_grad = torch.stack(local_grads, dim=0).mean(dim=0)
        gradient_list.append(avg_grad)
        seen += 1

        if seen % 32 == 0:
            print(f"  [SRS] collected {seen}/{num_samples} gradients")

    print(f"  [SRS] fitting subspace from {len(gradient_list)} gradients, rank={rank}")
    subspace = fit_shared_subspace(gradient_list, rank=rank)
    print(f"  [SRS] explained variance: {subspace.explained_variance:.4f}, "
          f"singular values: {subspace.singular_values.tolist()}")
    return subspace
