#!/usr/bin/env python3
"""
Optimized Radial-Directed Probe Attack — library module.

Implements PAID-theory-derived optimizations:
  1. AE-RP: Amplitude-Enhanced Radial Probe (weight D_r by |F_clean|^α)
  2. Dual A-Φ Loss: Simultaneously disrupt radial direction AND amplitude spectrum
  3. OOP: Optimized Offline Prior (average D_r from proxy images)
  4. Spatial Amplitude Loss: Direct |F_adv|-|F_clean| attack at each spatial location
  5. Orthogonal Φ-Probe: Isolate phase-channel attack orthogonal to radial direction

PAID Theory:
  I(F) = I_Φ(Φ_high) + I_A(A) + ε(x)
  Φ → feature direction f/|f| = radial direction (cross-model invariant)
  A → feature magnitude (model-specific information)
  Optimal attack disrupts BOTH channels simultaneously.

Channel interpretation:
  - Radial probe D_r = F/|F| pushes along F → attacks A (magnitude changes, direction preserved)
  - Orthogonal Φ-probe pushes ⟂ F → attacks Φ (direction changes, magnitude preserved)
  - Random IFFT probe has both components (inefficient but broad-spectrum)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── Core Probe Construction ───────────────────────────────────────────────────

def make_radial_probe(F_clean):
    """D_r = flatten(F_clean) / |flatten(F_clean)| — uniform radial direction map."""
    shape = F_clean.shape
    f_flat = F_clean.reshape(-1)
    d_r = F.normalize(f_flat, dim=0)
    return d_r.reshape(shape)


def make_radial_probe_amplitude_weighted(F_clean, alpha=0.5, weight_mode="magnitude"):
    """
    AE-RP: D_r_weighted = D_r * w / mean(w).

    weight_mode:
      - "magnitude": w = |F_clean|^α (original AE-RP, focuses on high-activation regions)
      - "edge": w = |∇F_clean|^α (edge-weighted, focuses on boundary/information-rich regions)
      - "entropy": w = local_std(F_clean)^α (variance-weighted)

    PAID justification:
      |F_clean(c,h,w)| encodes local information density (A component).
      Weighting by amplitude focuses attack on information-rich regions.
      Edge-weighting targets boundaries where Φ changes are most impactful.
    """
    shape = F_clean.shape
    f_flat = F_clean.reshape(-1)
    d_r = F.normalize(f_flat, dim=0)

    if weight_mode == "magnitude" or len(shape) <= 2:
        weights = torch.abs(f_flat).pow(alpha)
    else:
        # For spatial weight modes: reshape to [N, C, H, W] where N collapses all non-spatial dims
        spatial = F_clean.reshape(-1, *shape[-2:])  # [batch*C*..., H, W]
        # Add dummy channel dim for spatial ops: [N, 1, H, W]
        spatial_4d = spatial.unsqueeze(1)

        if weight_mode == "edge":
            grad_x = torch.abs(spatial_4d[:, :, :, 1:] - spatial_4d[:, :, :, :-1])
            grad_y = torch.abs(spatial_4d[:, :, 1:, :] - spatial_4d[:, :, :-1, :])
            edge_x = F.pad(grad_x, (0, 1, 0, 0))
            edge_y = F.pad(grad_y, (0, 0, 0, 1))
            edge_map_val = (edge_x + edge_y).mean(dim=(0, 1))  # pool over N and channel
            weights = edge_map_val.reshape(-1).pow(alpha)
        elif weight_mode == "entropy":
            local_mean = F.avg_pool2d(spatial_4d, kernel_size=3, stride=1, padding=1)
            local_mean_sq = F.avg_pool2d(spatial_4d ** 2, kernel_size=3, stride=1, padding=1)
            local_var = (local_mean_sq - local_mean ** 2).clamp(min=0)
            local_std = local_var.mean(dim=(0, 1))  # pool over N and channel
            weights = local_std.reshape(-1).pow(alpha)
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

    weights = weights / (weights.mean() + 1e-8)
    # Broadcast weights from spatial [H*W] to full shape if needed
    if weights.numel() < d_r.numel():
        weights = weights.reshape(*shape[-2:])  # [H, W]
        # Repeat across all channels/batch dimensions
        for _ in range(len(shape) - 2):
            weights = weights.unsqueeze(0)
        weights = weights.expand(*shape).reshape(-1)
    d_r_weighted = d_r * weights
    d_r_weighted = F.normalize(d_r_weighted, dim=0)
    return d_r_weighted.reshape(shape)


def make_random_probe(shape, device):
    """IFFT(complex white noise) — spatially correlated random pattern."""
    noise = torch.randn(*shape, device=device) + 1j * torch.randn(*shape, device=device)
    return torch.fft.ifftn(noise, dim=(-2, -1), norm="ortho").real


def make_orthogonal_phi_probe(F_clean):
    """
    Construct a Φ-channel probe: random direction with radial component projected out.

    PAID justification:
      The radial direction D_r attacks A (magnitude). To attack Φ (direction),
      we need a probe orthogonal to D_r. Random probe has both Φ and A components.
      Projecting out the radial component isolates the pure Φ attack.

      D_Φ = D_random - proj(D_random, D_r) = D_random - <D_random, D_r> * D_r
    """
    shape = F_clean.shape
    f_flat = F_clean.reshape(-1)
    d_r = F.normalize(f_flat, dim=0)  # [C*H*W]

    # Random direction in feature space
    d_random = torch.randn_like(f_flat)
    # Project out radial component
    projection = torch.dot(d_random, d_r) * d_r
    d_phi = d_random - projection
    d_phi = F.normalize(d_phi, dim=0)
    return d_phi.reshape(shape)


# ── Optimized Probe Construction (with offline prior) ────────────────────────

def compute_radial_maps(targets_layer, pre_layer, num_layers, alpha=0.5,
                        offline_layer_priors=None, prior_weight=0.0, device=None,
                        weight_mode="magnitude", include_phi_probe=False,
                        phi_weight=0.0):
    """
    Compute radial direction maps for selected layers, with optional:
    - Amplitude weighting (AE-RP) via alpha + weight_mode
    - Offline prior blending (OOP) via offline_layer_priors + prior_weight
    - Φ-channel probe (orthogonal to radial) via include_phi_probe + phi_weight
    """
    radial_maps = []
    phi_maps = []
    for i in range(num_layers):
        idx = i + pre_layer - num_layers
        F_i = targets_layer[idx]

        if alpha > 0:
            rmap = make_radial_probe_amplitude_weighted(F_i, alpha, weight_mode)
        else:
            rmap = make_radial_probe(F_i)

        if offline_layer_priors is not None and prior_weight > 0:
            prior = offline_layer_priors[i]
            if device is not None:
                prior = prior.to(device)
            # Reshape prior to match rmap shape if needed (handles per-channel vs global)
            if prior.shape != rmap.shape:
                prior = prior.reshape(rmap.shape)
            rmap = prior_weight * prior + (1.0 - prior_weight) * rmap
            rmap = F.normalize(rmap.reshape(-1), dim=0).reshape(rmap.shape)

        radial_maps.append(rmap)

        if include_phi_probe and phi_weight > 0:
            phi_map = make_orthogonal_phi_probe(F_i)
            phi_maps.append(phi_map)
        else:
            phi_maps.append(None)

    return radial_maps, phi_maps


# ── Dual A-Φ Loss ────────────────────────────────────────────────────────────

def compute_amplitude_loss(targets_layer, features_layer, pre_layer, num_layers,
                           selected_indices, mode="spectral"):
    """
    L_A = -mean_i |A(F_adv_i) - A(F_clean_i)||

    mode:
      - "spectral": A = |FFT(F)| — frequency-domain amplitude
      - "spatial":  A = |F| — per-element magnitude
      - "both":     sum of spectral and spatial

    PAID justification:
      I_A(A) encodes amplitude information. Maximizing the difference in
      amplitude spectra/spatial magnitudes disrupts this channel.
    """
    losses = []
    for idx in selected_indices:
        layer_idx = idx + pre_layer - num_layers
        F_c = targets_layer[layer_idx]
        F_a = features_layer[layer_idx]

        loss_terms = []

        if mode in ("spectral", "both"):
            A_clean = torch.abs(torch.fft.fftn(F_c, dim=(-2, -1), norm="ortho"))
            A_adv = torch.abs(torch.fft.fftn(F_a, dim=(-2, -1), norm="ortho"))
            loss_terms.append(-torch.abs(A_adv - A_clean).mean())

        if mode in ("spatial", "both"):
            A_clean_spatial = torch.abs(F_c)
            A_adv_spatial = torch.abs(F_a)
            loss_terms.append(-torch.abs(A_adv_spatial - A_clean_spatial).mean())

        losses.append(torch.stack(loss_terms).sum())

    return torch.stack(losses).mean()


def compute_phase_loss(targets_layer, features_layer, pre_layer, num_layers,
                       selected_indices):
    """
    L_Φ = -mean_i |cos_sim(F_adv_i, F_clean_i)|

    Directly maximizes angular deviation in feature space.
    Complements the radial probe by explicitly attacking Φ channel.

    PAID justification:
      I_Φ(Φ) encodes phase/direction information. Minimizing cosine
      similarity between clean and adversarial features directly
      disrupts the Φ channel.
    """
    losses = []
    for idx in selected_indices:
        layer_idx = idx + pre_layer - num_layers
        F_c = targets_layer[layer_idx].reshape(-1)
        F_a = features_layer[layer_idx].reshape(-1)
        cos_sim = F.cosine_similarity(F_a, F_c, dim=0)
        losses.append(cos_sim)  # maximize cos_sim loss = minimize similarity
    return torch.stack(losses).mean()


# ── Unified Optimized Attack ─────────────────────────────────────────────────

def attack_radial_optimized(
    image_encoder,          # LoRA/MSA-tuned encoder
    x,                      # clean input image [1,3,H,W]
    eps_pix, step_pix, steps,
    # AE-RP params:
    amplitude_alpha=0.5,
    weight_mode="magnitude",  # "magnitude", "edge", "entropy"
    # Dual-Loss params:
    lambda_amp=0.0,
    amp_loss_mode="both",     # "spectral", "spatial", "both"
    lambda_phase=0.0,         # direct phase/cosine loss weight
    # Φ-probe params:
    include_phi_probe=False,
    phi_weight=0.0,
    # OOP params:
    offline_layer_priors=None,
    prior_weight=0.0,
    # Core params:
    radial_weight=1.0,
    use_random_noise=True,
    criterion_l1=None,
    edge_map_fn=None,
):
    """
    Optimized Radial-Directed Probe Attack.

    Components (all toggle-able for ablation):
      AE-RP:     amplitude_alpha > 0 → amplitude-weighted radial probe
      Dual-Amp:  lambda_amp > 0 → add amplitude perturbation loss
      Dual-Φ:    lambda_phase > 0 → add direct phase/cosine loss
      Φ-probe:   include_phi_probe + phi_weight > 0 → orthogonal phase probe
      OOP:       offline_layer_priors + prior_weight > 0 → blend with offline prior

    PAID justification for combined approach:
      Radial probe attacks A channel (feature magnitude).
      Φ-probe + phase loss attack Φ channel (feature direction).
      Amplitude loss attacks A channel in frequency and/or spatial domain.
      Together they maximally degrade I(F) = I_Φ(Φ) + I_A(A).
    """
    if criterion_l1 is None:
        criterion_l1 = torch.nn.L1Loss()

    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    radial_maps, phi_maps = compute_radial_maps(
        targets_layer, pre_layer, num_layers,
        alpha=amplitude_alpha,
        weight_mode=weight_mode,
        offline_layer_priors=offline_layer_priors,
        prior_weight=prior_weight,
        device=x.device,
        include_phi_probe=include_phi_probe,
        phi_weight=phi_weight,
    )

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5

    if edge_map_fn is not None:
        edge_map = edge_map_fn(x).detach()
    else:
        edge_map = torch.zeros_like(x)

    momentum = torch.zeros_like(x)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        loss = criterion_l1(features * 0.9999, targets)

        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            layer_idx = idx + pre_layer - num_layers
            F_c = targets_layer[layer_idx]
            F_a = features_layer[layer_idx]
            shape = F_c.shape

            probe_radial = radial_maps[idx]
            r_scale = 0.01 / (probe_radial.abs().mean() + 1e-8)
            probe_radial = probe_radial * r_scale

            if include_phi_probe and phi_weight > 0 and phi_maps[idx] is not None:
                probe_phi = phi_maps[idx]
                p_scale = 0.01 / (probe_phi.abs().mean() + 1e-8)
                probe_phi = probe_phi * p_scale
                probe = (1.0 - phi_weight) * probe_radial + phi_weight * probe_phi
            elif use_random_noise and radial_weight < 1.0:
                probe_random = make_random_probe(shape, F_c.device)
                n_scale = 0.01 / (probe_random.abs().mean() + 1e-8)
                probe_random = probe_random * n_scale
                probe = radial_weight * probe_radial + (1.0 - radial_weight) * probe_random
            else:
                probe = probe_radial

            fd_list.append((F_a * 0.999 - F_c - 2.0 * probe))

        fd_stack = torch.stack(fd_list)
        selected_t = torch.stack(
            [targets_layer[i + pre_layer - num_layers] for i in indices])
        im = torch.stack([selected_t[j].abs().mean() for j in range(len(selected_t))])
        im = im.view(-1, *([1] * (fd_stack.dim() - 1)))
        probe_loss = torch.abs(torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()
        loss = loss + probe_loss

        if lambda_amp > 0:
            amp_loss = compute_amplitude_loss(
                targets_layer, features_layer, pre_layer, num_layers, indices,
                mode=amp_loss_mode)
            loss = loss + lambda_amp * amp_loss

        if lambda_phase > 0:
            phase_loss = compute_phase_loss(
                targets_layer, features_layer, pre_layer, num_layers, indices)
            loss = loss - lambda_phase * phase_loss  # maximize (cos_sim loss), minimize similarity

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ── Annealed Probe Attack (Adaptive Schedule) ───────────────────────────────

def attack_annealed_probe(
    image_encoder, x, eps_pix, step_pix, steps,
    amplitude_alpha=0.5, lambda_amp_max=0.1,
    prior_weight=0.0, offline_layer_priors=None,
    edge_map_fn=None, criterion_l1=None,
):
    """
    Annealed Probe Attack: adaptively transitions from targeted (radial) to
    broad (radial+random) exploration over the optimization steps.

    Schedule:
      β(t) = t/T:          random probe weight (0 → 1)
      λ_amp(t) = λ_max * (1 - t/T): amplitude loss weight (λ_max → 0)

    PAID justification:
      Early steps: targeted A-channel attack (radial probe) for fast convergence.
      Late steps: broad A+Φ exploration (mixed probe) for thorough degradation.
      The amplitude loss is strong early (push A away from clean) and weakens
      as the radial component decreases.

    This combines the sample-efficiency of radial probe (demonstrated at low
    step counts) with the broad coverage of random probe (demonstrated at
    high step counts).
    """
    if criterion_l1 is None:
        criterion_l1 = torch.nn.L1Loss()

    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    # Compute radial maps once (with optional AE-RP and OOP)
    radial_maps, _ = compute_radial_maps(
        targets_layer, pre_layer, num_layers,
        alpha=amplitude_alpha,
        weight_mode="magnitude",
        offline_layer_priors=offline_layer_priors,
        prior_weight=prior_weight,
        device=x.device,
    )

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5

    if edge_map_fn is not None:
        edge_map = edge_map_fn(x).detach()
    else:
        edge_map = torch.zeros_like(x)

    momentum = torch.zeros_like(x)

    for t in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        loss = criterion_l1(features * 0.9999, targets)

        # Annealing schedules
        beta = t / max(steps - 1, 1)  # 0 → 1
        lam_amp = lambda_amp_max * (1.0 - t / max(steps - 1, 1))  # λ_max → 0

        import random
        indices = random.sample(range(num_layers), n_pick)
        fd_list = []
        for idx in indices:
            layer_idx = idx + pre_layer - num_layers
            F_c = targets_layer[layer_idx]
            F_a = features_layer[layer_idx]
            shape = F_c.shape

            # Radial component
            probe_radial = radial_maps[idx]
            r_scale = 0.01 / (probe_radial.abs().mean() + 1e-8)
            probe_radial = probe_radial * r_scale

            # Random component (gradually introduced)
            if beta > 0:
                probe_random = make_random_probe(shape, F_c.device)
                n_scale = 0.01 / (probe_random.abs().mean() + 1e-8)
                probe_random = probe_random * n_scale
                probe = (1.0 - beta) * probe_radial + beta * probe_random
            else:
                probe = probe_radial

            fd_list.append((F_a * 0.999 - F_c - 2.0 * probe))

        fd_stack = torch.stack(fd_list)
        selected_t = torch.stack(
            [targets_layer[i + pre_layer - num_layers] for i in indices])
        im = torch.stack([selected_t[j].abs().mean() for j in range(len(selected_t))])
        im = im.view(-1, *([1] * (fd_stack.dim() - 1)))
        probe_loss = torch.abs(torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()
        loss = loss + probe_loss

        if lam_amp > 0:
            amp_loss = compute_amplitude_loss(
                targets_layer, features_layer, pre_layer, num_layers, indices,
                mode="spatial")
            loss = loss + lam_amp * amp_loss

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ── Baseline Attacks (for comparison) ────────────────────────────────────────

def attack_bare_radial(image_encoder, x, eps_pix, step_pix, steps, **kwargs):
    """Bare radial probe: α=0, no amp loss, no prior, no random noise."""
    return attack_radial_optimized(
        image_encoder, x, eps_pix, step_pix, steps,
        amplitude_alpha=0.0, lambda_amp=0.0, lambda_phase=0.0,
        offline_layer_priors=None, prior_weight=0.0,
        radial_weight=1.0, use_random_noise=False,
        include_phi_probe=False, phi_weight=0.0,
        **kwargs,
    )


def attack_random_probe(image_encoder, x, eps_pix, step_pix, steps, **kwargs):
    """Pure random IFFT probe: λ_r=0."""
    return attack_radial_optimized(
        image_encoder, x, eps_pix, step_pix, steps,
        amplitude_alpha=0.0, lambda_amp=0.0, lambda_phase=0.0,
        offline_layer_priors=None, prior_weight=0.0,
        radial_weight=0.0, use_random_noise=True,
        include_phi_probe=False, phi_weight=0.0,
        **kwargs,
    )


# ── Encoder Helpers ──────────────────────────────────────────────────────────

def get_image_encoder(model, model_type):
    """Get the correct image encoder regardless of model type."""
    if model_type == "samed":
        return model.sam.image_encoder
    elif model_type == "msa":
        return model.image_encoder
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── Offline Prior Computation ────────────────────────────────────────────────

def compute_offline_radial_prior(image_encoder, proxy_images, num_layers=6,
                                 pre_layer_frac=0.9, per_channel=False):
    """
    Compute offline radial direction prior from proxy training images.

    PAID justification:
      Φ determines f/|f| (radial direction). Since Φ is cross-model invariant,
      different images of the same modality share common Φ structure.
      Averaging D_r across proxy images isolates this shared structure,
      providing a stronger prior than PCA on noisy gradient directions.

    per_channel: if True, compute separate prior per channel (more expressive).
    """
    device = next(image_encoder.parameters()).device

    all_layer_drs = [[] for _ in range(num_layers)]

    for x in proxy_images:
        x = x.to(device).unsqueeze(0) if x.dim() == 3 else x.to(device)
        with torch.no_grad():
            _, targets_layer = image_encoder(x)

        pre_layer = int(len(targets_layer) * pre_layer_frac)
        for i in range(num_layers):
            idx = i + pre_layer - num_layers
            F_i = targets_layer[idx]
            if per_channel:
                # Per-channel radial directions [C, H*W]
                C = F_i.shape[0]
                f_flat = F_i.reshape(C, -1)
                d_r = F.normalize(f_flat, dim=1)
            else:
                f_flat = F_i.reshape(-1)
                d_r = F.normalize(f_flat, dim=0)
            all_layer_drs[i].append(d_r.cpu())

    layer_priors = []
    for i in range(num_layers):
        stacked = torch.stack(all_layer_drs[i], dim=0)
        mean_dr = stacked.mean(dim=0)
        if per_channel:
            mean_dr = F.normalize(mean_dr, dim=1)
        else:
            mean_dr = F.normalize(mean_dr, dim=0)
        layer_priors.append(mean_dr)

    return layer_priors
