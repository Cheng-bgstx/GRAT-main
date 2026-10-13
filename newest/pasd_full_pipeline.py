#!/usr/bin/env python3
"""
PASD Complete Pipeline: Offline Domain Prior + Online A-Φ Decoupled Attack.

Offline (Phase 1):
  1. Load ALL training Synapse slices
  2. Compute domain-averaged amplitude spectrum A_domain per encoder layer
  3. Compute domain phase disruption sensitivity map per layer
  4. Save offline prior for online use

Online (Phase 2) — A-Φ Decoupled Attack:
  1. Load offline domain prior
  2. For each test image:
     a. Extract clean features + phase/amplitude
     b. A-Φ decoupled loss:
        - L_amp: push amplitude away from domain average
        - L_phase: disrupt phase coherence (targeted by domain prior)
        - L_feat: weak feature-space push
     c. Iterative PGD with momentum + edge guidance
  3. Save adversarial images in format expected by test_samed.py

Evaluation (Phase 3):
  - test_samed.py for SAMed (LoRA) models
  - Medical-SAM-Adapter-main/val.py for MSA (Adapter) models

Usage:
  # Offline training
  python pasd_full_pipeline.py --phase offline --vit vit_b --eps 8/255

  # Online attack generation
  python pasd_full_pipeline.py --phase online --vit vit_b --eps 8/255 --steps 20

  # Full pipeline
  python pasd_full_pipeline.py --phase all --vit vit_b
"""
import sys, os, gc, time, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder
from attack_pasa import _get_edge_map, _return_eta, criterion_l1

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}
OFFLINE_DIR = os.path.join(NEWEST, "offline_priors")
ADV_OUTPUT_DIR = os.path.join(ROOT, "adv_outputs")
os.makedirs(OFFLINE_DIR, exist_ok=True)
os.makedirs(ADV_OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_all_train_slices():
    """Load ALL training Synapse slices."""
    tdir = f"{ROOT}/data/Synapse_preprocess/train_npz"
    files = sorted([f for f in os.listdir(tdir) if f.endswith('.npz')])
    slices = []
    for f in files:
        data = np.load(os.path.join(tdir, f))
        img = data['image'].astype(np.float32)
        lbl = data['label'].astype(np.int64)
        if (lbl > 0).sum() < 500: continue
        slices.append((np.stack([img, img, img], axis=0), lbl))
    return slices


def load_test_cases():
    """Load test case names from test_vol.txt."""
    list_path = f"{ROOT}/data/Synapse_preprocess/lists/test_vol.txt"
    with open(list_path) as f:
        cases = [line.strip() for line in f if line.strip()]
    return cases


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Offline — Domain Prior Extraction
# ═══════════════════════════════════════════════════════════════════════════

def offline_extract_domain_prior(vit_name="vit_b", n_train=None, num_layers=6):
    """
    Extract domain-level priors from ALL training data:

    1. A_domain[l]: domain-average amplitude spectrum per encoder layer
       - Captures which frequencies are consistently important for CT images

    2. Phi_sensitivity[l]: phase disruption sensitivity per layer
       - Which frequency positions have high phase variability = good targets

    3. Layer_weights: relative importance of each layer for attack
    """
    device = torch.device("cuda:0")

    # Load base SAM encoder
    sam = load_sam_encoder(vit_name, SAM_CKPTS[vit_name], IMAGE_SIZE, num_classes=8, device=device)
    encoder = sam.image_encoder
    encoder.eval()

    # Load training data
    train_slices = load_all_train_slices()
    if n_train:
        train_slices = train_slices[:n_train]
    print(f"[Offline] Loaded {len(train_slices)} training slices")

    # Initialize accumulators
    A_domain = [None] * num_layers
    Phi_samples = [[] for _ in range(num_layers)]  # Store phase tensors for sensitivity
    count = 0

    t0 = time.time()
    for i, (img, lbl) in enumerate(train_slices):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)

        with torch.no_grad():
            _, targets_layer = encoder(x)

        pre_layer = int(len(targets_layer) * 0.9)
        n_avail = min(num_layers, pre_layer)
        layer_ids = list(range(pre_layer - n_avail, pre_layer))

        for j, idx in enumerate(layer_ids):
            F = targets_layer[idx].unsqueeze(0)
            Z = torch.fft.fftn(F, dim=(-2, -1), norm="ortho")
            A = torch.abs(Z).cpu()
            Phi = torch.angle(Z).cpu()

            # Accumulate amplitude
            if A_domain[j] is None:
                A_domain[j] = A
            else:
                A_domain[j] += A

            # Store phase samples (subset to manage memory)
            if i < 500:  # Use first 500 images for phase stats
                Phi_samples[j].append(Phi)

        count += 1
        del x; gc.collect(); torch.cuda.empty_cache()

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [Offline] {i+1}/{len(train_slices)} ({elapsed:.0f}s)")

    # Average amplitude
    for j in range(num_layers):
        A_domain[j] = A_domain[j] / count
    print(f"  [Offline] A_domain computed for {num_layers} layers")

    # Phase sensitivity: per-frequency circular variance
    # Low circular variance = phase is stable across images = better to disrupt
    # High circular variance = phase already varies = less important to disrupt
    Phi_sensitivity = [None] * num_layers
    for j in range(num_layers):
        if Phi_samples[j]:
            phases = torch.stack(Phi_samples[j], dim=0)  # [N, 1, C, H, W]
            mean_carrier = torch.exp(1j * phases).mean(dim=0)  # Complex mean
            stability = torch.abs(mean_carrier)  # [1, C, H, W], 1=stable, 0=random
            # Sensitivity = 1 - stability: high when phase is stable (good to disrupt)
            Phi_sensitivity[j] = (1.0 - stability).cpu()
        else:
            Phi_sensitivity[j] = torch.ones(1, *A_domain[j].shape[1:])  # Default uniform

    print(f"  [Offline] Phi_sensitivity computed for {num_layers} layers")

    # Save offline prior
    prior = {
        "A_domain": [A.cpu() for A in A_domain],
        "Phi_sensitivity": [P.cpu() for P in Phi_sensitivity],
        "vit_name": vit_name,
        "num_layers": num_layers,
        "n_train": count,
    }

    prior_path = os.path.join(OFFLINE_DIR, f"domain_prior_{vit_name}.pth")
    torch.save(prior, prior_path)
    print(f"  [Offline] Prior saved to {prior_path}")

    del sam, encoder
    torch.cuda.empty_cache()
    return prior


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Online — A-Φ Decoupled Attack
# ═══════════════════════════════════════════════════════════════════════════

def aphi_decoupled_attack(encoder, x, eps_pix, step_pix, steps,
                           domain_prior, num_layers=6,
                           lambda_amp=0.3, lambda_phase=0.5, lambda_feat=0.1,
                           sigma_amp=1.0, alpha_amp=1.0):
    """
    A-Φ Decoupled Attack with domain prior.

    Loss components:
      L_amp:   Push amplitude away from domain average in probe construction
      L_phase: Disrupt phase coherence weighted by domain sensitivity
      L_feat:  Weak feature-space push (standard PASD L1)

    The probe is:
      probe = IFFT( (sigma*N + lambda_amp * (A_clean - A_domain)) *
                     A_clean^alpha *
                     exp(j * (Phi_clean + lambda_phase * Phi_sensitivity * N_phase)) )

    Key innovations over baseline PASD:
      1. A_domain provides stable frequency reference (domain prior)
      2. A_clean - A_domain term explicitly pushes amplitude OFF the domain manifold
      3. Phi_sensitivity targets phase disruption to stable (important) frequencies
      4. Three-term loss with tunable weights
    """
    from torch.autograd import Variable

    max_val, min_val = x.max(), x.min()
    device = x.device

    with torch.no_grad():
        targets, targets_layer = encoder(x)

    edge_map = _get_edge_map(x).detach()
    x_adv = Variable(x.clone().detach() + torch.randn_like(x) * 1e-5, requires_grad=True)
    momentum = torch.zeros_like(x).detach()

    pre_layer = int(len(targets_layer) * 0.9)
    n_avail = min(num_layers, pre_layer)
    lid = list(range(pre_layer - n_avail, pre_layer))

    # Move domain prior to device
    A_dom = [a.to(device) for a in domain_prior["A_domain"]]
    Phi_sens = [p.to(device) for p in domain_prior["Phi_sensitivity"]]

    for step_idx in range(steps):
        encoder.zero_grad()
        inp = x_adv + torch.randn_like(x_adv) * 0.0001 if step_idx == 0 else x_adv
        features, features_layer = encoder(inp)

        # L_feat: push final features away from clean
        loss_feat = criterion_l1(features * 0.9999, targets)

        # L_probe: multi-layer A-Φ decoupled probe loss
        probe_losses = []
        for _ in range(2):  # K=2 noise samples
            diff_list = []
            for j, idx in enumerate(lid):
                F_adv = features_layer[idx].unsqueeze(0)
                F_clean = targets_layer[idx].unsqueeze(0).detach()

                # Frequency domain
                Z_clean = torch.fft.fftn(F_clean, dim=(-2, -1), norm="ortho")
                A_clean = torch.abs(Z_clean)
                Phi_clean = torch.angle(Z_clean)

                # ── A-Φ Decoupled Probe Construction ──

                # Phase: targeted disruption by domain sensitivity
                # High Phi_sens → more phase disruption at that frequency
                shape_ph = Phi_clean.shape
                idx_perm = torch.randperm(Phi_clean.numel(), device=device)
                Phi_shuffled = Phi_clean.reshape(-1)[idx_perm].reshape(shape_ph)

                # Interpolate between clean and shuffled based on sensitivity
                ps = Phi_sens[j]  # [1, C, H, W]
                # Clamp sensitivity to [0, 0.5] for stability
                ps = torch.clamp(ps, 0, 0.5)
                Phi_disrupted = Phi_clean + ps * np.pi * (Phi_shuffled - Phi_clean)
                carrier = torch.exp(1j * Phi_disrupted)

                # Amplitude: push away from domain average
                A_domain_cur = A_dom[j]
                A_diff = (A_clean - A_domain_cur)  # Positive where A_clean > A_domain
                A_push = torch.abs(A_diff)  # Push magnitude

                # Base amplitude = A_clean^alpha (standard PASD prior)
                Sigma = A_clean.pow(alpha_amp) + 1e-4

                # Noise: random + domain-guided amplitude push
                N_white = torch.randn_like(A_clean, device=device)
                # Blend: sigma * random_noise + lambda_amp * domain_push
                N_blend = sigma_amp * N_white + lambda_amp * A_push

                probe_fft = N_blend * Sigma * carrier
                probe = torch.fft.ifftn(probe_fft, dim=(-2, -1), norm="ortho").real

                diff = (F_adv * 0.999 - F_clean) - 2.0 * probe
                diff_list.append(diff.squeeze(0))

            diff = torch.stack(diff_list, dim=0)
            sel_t = torch.stack([targets_layer[i] for i in lid], dim=0)
            im = torch.stack([sel_t[i].abs().mean() for i in range(len(sel_t))])
            im = im.view(diff.shape[0], 1, 1, 1)
            probe_losses.append(torch.abs(torch.mean(diff / (im + 1e-6), dim=0)).mean())

        loss_probe = torch.stack(probe_losses).mean()
        loss = lambda_feat * loss_feat + loss_probe

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum.detach() + grad
        momentum = momentum.detach()

        eta = _return_eta(momentum, step_pix).detach()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta_total = x_adv.data - x.data
        eta_total = torch.clamp(eta_total, -eps_pix, eps_pix)
        x_adv = Variable(x.data + eta_total, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, min_val, max_val), requires_grad=True)

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Online Attack Generation (all test cases)
# ═══════════════════════════════════════════════════════════════════════════

def generate_adv_samples(vit_name="vit_b", eps_frac=8.0/255, steps=20, attack_mode="aphi_decoupled"):
    """Generate adversarial samples for ALL test cases."""
    device = torch.device("cuda:0")

    # Load base SAM encoder for attack
    sam = load_sam_encoder(vit_name, SAM_CKPTS[vit_name], IMAGE_SIZE, num_classes=8, device=device)
    encoder = sam.image_encoder
    encoder.eval()

    # Load domain prior
    prior_path = os.path.join(OFFLINE_DIR, f"domain_prior_{vit_name}.pth")
    if os.path.exists(prior_path):
        domain_prior = torch.load(prior_path, map_location="cpu")
        print(f"[Online] Domain prior loaded from {prior_path}")
    else:
        print("[Online] WARNING: No domain prior found, running offline first...")
        domain_prior = offline_extract_domain_prior(vit_name)

    # Load test cases
    test_cases = load_test_cases()
    print(f"[Online] Processing {len(test_cases)} test cases")

    step = 2.0 / 255.0  # Fixed step matching original SE-CGR
    output_dir = os.path.join(ADV_OUTPUT_DIR, f"pasd_{vit_name}_eps{int(eps_frac*255)}")
    os.makedirs(output_dir, exist_ok=True)

    import h5py
    test_dir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"

    total_slices = 0
    t0 = time.time()

    for case_name in test_cases:
        h5_path = os.path.join(test_dir, f"{case_name}.npy.h5")
        if not os.path.exists(h5_path):
            print(f"  SKIP {case_name}: not found")
            continue

        with h5py.File(h5_path, 'r') as f:
            img_vol = f['image'][:]   # [D, H, W]
            lbl_vol = f['label'][:]   # [D, H, W]

        n_slices = img_vol.shape[0]
        case_output_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_output_dir, exist_ok=True)

        print(f"  {case_name}: {n_slices} slices...", end=" ", flush=True)

        for s in range(n_slices):
            slc = img_vol[s].astype(np.float32)
            # Convert grayscale to 3-channel
            img_3c = np.stack([slc, slc, slc], axis=0)
            x = torch.from_numpy(img_3c).unsqueeze(0).float().to(device)

            rg = x.max().item() - x.min().item()
            eps_pix = eps_frac * rg
            step_pix = step * rg

            if attack_mode == "aphi_decoupled":
                x_adv = aphi_decoupled_attack(encoder, x, eps_pix, step_pix, steps,
                                              domain_prior, num_layers=6,
                                              lambda_amp=0.3, lambda_phase=0.5, lambda_feat=0.1)
            elif attack_mode == "pasd_baseline":
                from attack_pasa import run_pasd_attack
                x_adv = run_pasd_attack(encoder, x, eps_pix, step_pix, steps,
                                       sigma=1.0, alpha=1.0, probe_mode="pasd")
            else:
                raise ValueError(f"Unknown attack_mode: {attack_mode}")

            # Save as grayscale PNG (Synapse format expected by test_samed.py)
            adv_np = x_adv[0].detach().cpu().numpy()  # [3, 512, 512]
            # Convert to grayscale for saving (mean of 3 channels)
            adv_gray = np.mean(adv_np, axis=0)  # [512, 512]
            adv_gray = np.clip(adv_gray * 255, 0, 255).astype(np.uint8)

            from PIL import Image
            Image.fromarray(adv_gray).save(os.path.join(case_output_dir, f"{s}.png"))

            del x_adv, x
            torch.cuda.empty_cache()

        total_slices += n_slices
        print(f"done. ({time.time()-t0:.0f}s total)")

    print(f"\n[Online] Generated {total_slices} adversarial slices in {output_dir}")
    print(f"[Online] Total time: {time.time()-t0:.0f}s")

    # Save attack metadata
    meta = {
        "vit_name": vit_name,
        "eps_frac": eps_frac,
        "steps": steps,
        "attack_mode": attack_mode,
        "test_cases": test_cases,
        "total_slices": total_slices,
    }
    with open(os.path.join(output_dir, "attack_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    del sam, encoder
    torch.cuda.empty_cache()
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=str, default="all",
                    choices=["offline", "online", "all"])
    ap.add_argument("--vit", type=str, default="vit_b",
                    choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--eps", type=float, default=8.0,
                    help="epsilon in /255 (e.g., 8 = 8/255)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--attack_mode", type=str, default="aphi_decoupled",
                    choices=["aphi_decoupled", "pasd_baseline"])
    ap.add_argument("--n_train", type=int, default=None,
                    help="Max training slices (None=all)")
    args = ap.parse_args()

    eps_frac = args.eps / 255.0

    if args.phase in ("offline", "all"):
        print("=" * 60)
        print("PHASE 1: OFFLINE — Domain Prior Extraction")
        print("=" * 60)
        offline_extract_domain_prior(args.vit, n_train=args.n_train)

    if args.phase in ("online", "all"):
        print("\n" + "=" * 60)
        print("PHASE 2: ONLINE — Adversarial Sample Generation")
        print("=" * 60)
        output_dir = generate_adv_samples(args.vit, eps_frac, args.steps, args.attack_mode)

        print("\n" + "=" * 60)
        print("NEXT: Evaluate with:")
        print(f"  python test_samed.py --volume_path {output_dir} --vit_name {args.vit}")
        print("=" * 60)
