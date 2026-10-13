#!/usr/bin/env python3
"""
Proper SE-CGR ablation: Test whether amplitude modulation (A-weighting)
and phase carrier (Phase_clean) actually contribute to attack strength.

SE-CGR theory:
  - FFT of clean features → A (amplitude spectrum) + Φ (phase spectrum)
  - Amplitude modulation: N_white weighted by A → focuses perturbation on
    high-energy frequencies (A is approximately invariant across models)
  - Phase carrier: uses exp(j*Φ_clean) → aligns perturbation with spatial
    structure of clean features (Φ_clean structure is shared across models)
  - Loss = L1(f_adv, f_clean) + alignment(drift, A_modulated_phase_locked_probe)

Ablation:
  1. Full SE-CGR: A-weighting + Phase_clean carrier
  2. No amplitude mod: flat spectrum (Sigma = constant)
  3. No phase carrier: random phase
  4. Neither: flat spectrum + random phase (= unstructured probe)
  5. No probe at all: pure L1 loss (= PGD with edge guidance)
"""
import os, sys, time, json, random
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from exp_paid_optimal_attack import load_synapse_slices, evaluate
from attack_codd import _get_edge_map, _return_mean_value

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT_REL = "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"
EPSILON = 16.0 / 255.0
STEP = 1.6 / 255.0
STEPS = 20
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
criterion_l1 = torch.nn.L1Loss()

net = load_samed_model("vit_b", SAM_CKPT, os.path.join(CKPT_BASE, CKPT_REL),
                       IMAGE_SIZE, 8, device=device).to(device)
source_sam = load_sam_encoder("vit_b", SAM_CKPT, IMAGE_SIZE, device=device)
image_encoder = source_sam.image_encoder
images = load_synapse_slices(5)
print(f"Loaded {len(images)} images")


def synthesize_probe(F_clean, F_adv, use_amp_mod=True, use_phase_carrier=True):
    """
    Synthesize structured frequency-domain probe.

    use_amp_mod=True:  Sigma = lambda * A + gamma (weighted by amplitude spectrum)
    use_amp_mod=False: Sigma = constant (flat spectrum)

    use_phase_carrier=True:  carrier = exp(j * Phase_clean)
    use_phase_carrier=False: carrier = exp(j * random_phase)
    """
    G_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
    A = torch.abs(G_clean)
    Phase_clean = torch.angle(G_clean)

    # Amplitude modulation
    if use_amp_mod:
        current_diff = F_adv.detach() * 0.999 - F_clean.detach()
        diff_energy = torch.mean(torch.abs(current_diff))
        spectrum_energy = torch.mean(A)
        lambda_val = diff_energy / (spectrum_energy + 1e-6)
        Sigma_freq = lambda_val * A + 1e-4
    else:
        # Flat spectrum: constant sigma
        Sigma_freq = torch.ones_like(A) * 0.01

    # Phase carrier
    if use_phase_carrier:
        carrier = torch.exp(1j * Phase_clean)
    else:
        rand_phase = torch.rand_like(Phase_clean) * 2 * np.pi - np.pi
        carrier = torch.exp(1j * rand_phase)

    # Construct probe
    N_white = torch.randn_like(A, device=A.device)
    fft_c = (N_white * Sigma_freq) * carrier
    return torch.fft.ifftn(fft_c, dim=(-2, -1), norm="ortho").real


def attack_secgr_ablated(image_encoder, x, eps_pix, step_pix, steps,
                         use_amp_mod=True, use_phase_carrier=True,
                         use_probe=True):
    """
    SE-CGR attack with controllable components.
    use_probe=False → pure L1 loss + edge guidance (no probe at all).
    """
    max_val, min_val = x.max(), x.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(x)

    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = _get_edge_map(x).detach()
    momentum = torch.zeros_like(x)

    pre_layer = int(len(targets_layer) * 0.9)
    num_layers = min(6, pre_layer)
    n_pick = max(1, num_layers - 2)

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        features, features_layer = image_encoder(x_adv)

        loss_org = criterion_l1(features * 0.9999, targets)

        if use_probe:
            indices = random.sample(range(num_layers), n_pick)
            fd_list = []
            for idx in indices:
                F_clean_i = targets_layer[idx + pre_layer - num_layers]
                F_adv_i = features_layer[idx + pre_layer - num_layers]

                probe = synthesize_probe(
                    F_clean_i.unsqueeze(0), F_adv_i.unsqueeze(0),
                    use_amp_mod=use_amp_mod,
                    use_phase_carrier=use_phase_carrier,
                )
                # SE-CGR alignment loss
                fd_list.append((F_adv_i.unsqueeze(0) * 0.999
                               - F_clean_i.unsqueeze(0)) - 2.0 * probe)

            fd_stack = torch.stack([f.squeeze(0) for f in fd_list])
            selected_t = torch.stack(
                [targets_layer[i + pre_layer - num_layers] for i in indices])
            im = _return_mean_value(selected_t)
            if len(im.shape) > len(fd_stack.shape):
                im = im.view(fd_stack.shape[0], 1, 1, 1)
            l1_losses = torch.abs(
                torch.mean(fd_stack / (im + 1e-6), dim=0)).mean()
            loss = loss_org + l1_losses
        else:
            loss = loss_org

        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()

    return x_adv.detach()


# ── Experiment ────────────────────────────────────────────────────────────

configs = [
    ("Full SE-CGR (A-mod + Φ-carrier)", True,  True,  True),
    ("No amplitude mod (Φ only)",        False, True,  True),
    ("No phase carrier (A only)",        True,  False, True),
    ("Neither (unstructured probe)",     False, False, True),
    ("No probe (L1 + edge only)",        True,  True,  False),
]

print(f"\n{'='*70}")
print("SE-CGR COMPONENT ABLATION")
print(f"ε={EPSILON:.4f}, step={STEP:.4f}, steps={STEPS}")
print(f"{'='*70}")

all_results = []
for name, use_amp, use_phase, use_probe in configs:
    deltas = []
    t0 = time.time()
    for img_data, lbl in images:
        x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
        clean_d = evaluate(net, img_data, lbl, "Synapse", device)
        if clean_d < 0.01:
            continue
        rg = x.max().item() - x.min().item()
        x_adv = attack_secgr_ablated(image_encoder, x,
                                     EPSILON * rg, STEP * rg, STEPS,
                                     use_amp_mod=use_amp,
                                     use_phase_carrier=use_phase,
                                     use_probe=use_probe)
        adv_np = x_adv.squeeze(0).cpu().numpy()
        att_d = evaluate(net, adv_np, lbl, "Synapse", device)
        deltas.append(clean_d - att_d)
    dt = time.time() - t0
    md, sd = float(np.mean(deltas)), float(np.std(deltas))
    all_results.append({"name": name, "mean_delta": md, "std_delta": sd, "time": dt})
    print(f"  {name:<35} Δ={md:+.4f}±{sd:.3f}  {dt:.1f}s")

# Save
with open(os.path.join(OUTPUT_DIR, "secgr_component_ablation.json"), "w") as f:
    json.dump(all_results, f, indent=2)

del net, source_sam
torch.cuda.empty_cache()
print("Done")
