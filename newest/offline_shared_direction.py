"""
Offline Shared Vulnerability Direction Extraction for PASD.

Methods:
  MD   — Momentum Distillation: PCA on 20-step PASD momenta from training data
  FDUD — Frequency-Domain Universal Direction: averaged frequency-domain drift from training data
  NAIVE — Naive perturbation averaging: mean of training perturbations (baseline)
  SRS  — Shared Radial Subspace (old method for comparison)

Online: use offline prior to warm-start PASD momentum.
"""
import sys, os, gc, time, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from attack_pasa import run_pasd_attack, PASD_MODES, _get_edge_map, _return_eta
from torch.autograd import Variable

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
}

device = torch.device("cuda:1")
EPS_FRAC = 8.0 / 255
STEPS = 20


# ═══════════════════════════════════════════════════════════════════════════
# Data loading: training set
# ═══════════════════════════════════════════════════════════════════════════

def load_train_slices(n=16):
    """Load training Synapse slices from train_npz directory."""
    tdir = f"{ROOT}/data/Synapse_preprocess/train_npz"
    files = sorted([f for f in os.listdir(tdir) if f.endswith('.npz')])
    slices = []
    for f in files:
        if len(slices) >= n:
            break
        data = np.load(os.path.join(tdir, f))
        img = data['image'].astype(np.float32)  # [H, W] grayscale
        lbl = data['label'].astype(np.int64)
        if (lbl > 0).sum() < 500:
            continue
        # Convert to 3-channel
        img_3c = np.stack([img, img, img], axis=0)
        slices.append((img_3c, lbl))
    print(f"  Loaded {len(slices)} training slices")
    return slices


def load_test_slices(n=8):
    """Load test Synapse slices (from previous experiment)."""
    from exp_paid_optimal_attack import load_synapse_slices
    return load_synapse_slices(n)


# ═══════════════════════════════════════════════════════════════════════════
# PASD with momentum capture (for offline collection)
# ═══════════════════════════════════════════════════════════════════════════

def pasd_attack_with_capture(image_encoder, input_image, eps_pix, step_pix, steps,
                              sigma=1.0, alpha=1.0, probe_mode="pasd",
                              capture_momentum=False, capture_drift=False,
                              momentum_init=None):
    """
    Run PASD attack, optionally capturing momentum and/or frequency drift.
    """
    max_val, min_val = input_image.max(), input_image.min()

    with torch.no_grad():
        targets, targets_layer = image_encoder(input_image)

    edge_map = _get_edge_map(input_image).detach()
    x_adv = Variable(input_image.data + torch.randn_like(input_image) * 1e-5, requires_grad=True)

    # Momentum warm-start
    if momentum_init is not None:
        momentum = momentum_init.clone().detach().to(input_image.device)
    else:
        momentum = torch.zeros_like(input_image).detach()

    captured_momentum = None
    captured_drift = None

    for step in range(steps):
        image_encoder.zero_grad()

        inp = x_adv + torch.randn_like(x_adv) * 0.0001 if step == 0 else x_adv
        features, features_layer = image_encoder(inp)

        # PASD loss
        from attack_pasa import calculate_pasd_loss
        loss = calculate_pasd_loss(
            features=features, targets=targets,
            features_layer=features_layer, targets_layer=targets_layer,
            sigma=sigma, alpha=alpha, probe_mode=probe_mode,
            K=2, num_layers=6,
        )

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        momentum = 0.5 * momentum.detach() + grad
        momentum = momentum.detach()

        eta = _return_eta(momentum, step_pix).detach()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta_total = x_adv.data - input_image.data
        eta_total = torch.clamp(eta_total, -eps_pix, eps_pix)
        x_adv = Variable(input_image.data + eta_total, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, min_val, max_val), requires_grad=True)

    if capture_momentum:
        captured_momentum = momentum.detach().cpu().clone()
    if capture_drift:
        with torch.no_grad():
            feat_adv, _ = image_encoder(x_adv)
            feat_clean, _ = image_encoder(input_image)
        # Frequency domain drift per layer (L1-normalized)
        drift_fft_list = []
        for i, (fa, fc) in enumerate(zip(features_layer[-6:], targets_layer[-6:])):
            Za = torch.fft.fftn(fa.unsqueeze(0), dim=(-2,-1), norm="ortho")
            Zc = torch.fft.fftn(fc.unsqueeze(0), dim=(-2,-1), norm="ortho")
            dZ = Za - Zc
            dZ_norm = dZ / (torch.abs(dZ).sum() + 1e-8)
            drift_fft_list.append(dZ_norm.detach().cpu())
        captured_drift = drift_fft_list

    return x_adv.detach(), captured_momentum, captured_drift


# ═══════════════════════════════════════════════════════════════════════════
# Offline methods
# ═══════════════════════════════════════════════════════════════════════════

def offline_md(encoder, train_slices, eps_pix, step_pix, steps, rank=8):
    """Momentum Distillation: PCA on final momenta."""
    momenta = []
    for i, (img, lbl) in enumerate(train_slices):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)
        rg = x.max().item() - x.min().item()
        _, mom, _ = pasd_attack_with_capture(
            encoder, x, eps_pix * rg, step_pix * rg, steps,
            capture_momentum=True, probe_mode="pasd")
        momenta.append(mom.flatten())
        if (i+1) % 8 == 0:
            print(f"  [MD] Collected {i+1}/{len(train_slices)} momenta")

    X = torch.stack(momenta, dim=0)  # [N, D]
    # L1 normalize
    X = X / (X.abs().sum(dim=1, keepdim=True) + 1e-12)
    mean_vec = X.mean(dim=0)
    mean_vec = mean_vec / (mean_vec.abs().sum() + 1e-12)
    X_centered = X - mean_vec

    # PCA
    q = min(rank, len(X), X.shape[1])
    U, S, V = torch.pca_lowrank(X_centered, q=q, center=False)
    basis = V[:, :q].T.contiguous()  # [q, D]

    return {
        "mean_direction": mean_vec,
        "basis": basis,
        "explained_variance": float(S[:q].square().sum() / (X_centered.square().sum() + 1e-12)),
    }


def offline_fdud(encoder, train_slices, eps_pix, step_pix, steps, num_layers=6):
    """Frequency-Domain Universal Direction: average frequency drift."""
    drift_accum = None
    count = 0
    for i, (img, lbl) in enumerate(train_slices):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)
        rg = x.max().item() - x.min().item()
        _, _, drifts = pasd_attack_with_capture(
            encoder, x, eps_pix * rg, step_pix * rg, steps,
            capture_drift=True, probe_mode="pasd")
        if drifts is not None:
            if drift_accum is None:
                drift_accum = [d.clone() for d in drifts]
            else:
                for j in range(len(drift_accum)):
                    drift_accum[j] += drifts[j]
            count += 1
        if (i+1) % 8 == 0:
            print(f"  [FDUD] Collected {i+1}/{len(train_slices)} drifts")

    # Average
    for j in range(len(drift_accum)):
        drift_accum[j] = drift_accum[j] / count
    return drift_accum  # List of [1, C, H, W] normalized frequency drift per layer


def offline_naive(encoder, train_slices, eps_pix, step_pix, steps):
    """Naive: average pixel-space perturbation."""
    pert_accum = None
    count = 0
    for i, (img, lbl) in enumerate(train_slices):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)
        rg = x.max().item() - x.min().item()
        x_adv, _, _ = pasd_attack_with_capture(
            encoder, x, eps_pix * rg, step_pix * rg, steps,
            probe_mode="pasd")
        pert = x_adv.detach().cpu() - x.cpu()
        if pert_accum is None:
            pert_accum = pert.clone()
        else:
            pert_accum += pert
        count += 1
        if (i+1) % 8 == 0:
            print(f"  [NAIVE] Collected {i+1}/{len(train_slices)} perturbations")

    mean_pert = pert_accum / count
    mean_pert = mean_pert / (mean_pert.abs().sum() + 1e-12)
    return mean_pert


# ═══════════════════════════════════════════════════════════════════════════
# Online: warm-start PASD with offline prior
# ═══════════════════════════════════════════════════════════════════════════

def pasd_with_md_warmstart(encoder, x, eps_pix, step_pix, steps, md_prior, proj_weight=0.3):
    """PASD with MD warm-start: blend online momentum with PCA subspace projection."""
    shape = x.shape
    D = int(np.prod(shape))

    mean_dir = md_prior["mean_direction"].to(device)
    basis = md_prior["basis"].to(device)

    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = encoder(x)
    edge_map = _get_edge_map(x).detach()
    x_adv = Variable(x.data + torch.randn_like(x) * 1e-5, requires_grad=True)

    # Warm-start momentum with MD mean direction
    momentum = mean_dir.reshape(shape).clone().to(device)

    for step in range(steps):
        encoder.zero_grad()
        inp = x_adv + torch.randn_like(x_adv) * 0.0001 if step == 0 else x_adv
        features, features_layer = encoder(inp)

        from attack_pasa import calculate_pasd_loss
        loss = calculate_pasd_loss(
            features=features, targets=targets,
            features_layer=features_layer, targets_layer=targets_layer,
            sigma=1.0, alpha=1.0, probe_mode="pasd", K=2, num_layers=6,
        )

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)

        # Blend with MD subspace (persistent warm-start, unlike old SRS)
        g_flat = grad.flatten()
        g_proj = g_flat @ basis.T  # [K]
        g_recon = g_proj @ basis   # [D]
        grad_blended = g_flat + proj_weight * g_recon
        grad_blended = grad_blended / (grad_blended.abs().sum() + 1e-12)
        grad = grad_blended.reshape(shape)

        momentum = 0.5 * momentum.detach() + grad
        momentum = momentum.detach()

        eta = _return_eta(momentum, step_pix).detach()
        x_adv = Variable(x_adv.data + eta, requires_grad=True)
        eta_total = x_adv.data - x.data
        eta_total = torch.clamp(eta_total, -eps_pix, eps_pix)
        x_adv = Variable(x.data + eta_total, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, min_val, max_val), requires_grad=True)

    return x_adv.detach()


def pasd_with_naive_warmstart(encoder, x, eps_pix, step_pix, steps, mean_pert):
    """PASD with naive perturbation average as momentum init."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = encoder(x)
    edge_map = _get_edge_map(x).detach()
    x_adv = Variable(x.data + torch.randn_like(x) * 1e-5, requires_grad=True)

    # Warm-start momentum with mean perturbation
    momentum = mean_pert.clone().to(device)

    for step in range(steps):
        encoder.zero_grad()
        inp = x_adv + torch.randn_like(x_adv) * 0.0001 if step == 0 else x_adv
        features, features_layer = encoder(inp)

        from attack_pasa import calculate_pasd_loss
        loss = calculate_pasd_loss(
            features=features, targets=targets,
            features_layer=features_layer, targets_layer=targets_layer,
            sigma=1.0, alpha=1.0, probe_mode="pasd", K=2, num_layers=6,
        )

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
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def eval_synapse(net, img_t, gt):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out["masks"]
    pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred == c).astype(np.float32)
        gc = (gt == c).astype(np.float32)
        if gc.sum() == 0: continue
        inter = (pc * gc).sum()
        dices.append(float(2.0 * inter / (pc.sum() + gc.sum() + 1e-8)))
    return float(np.mean(dices)) if dices else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N_TRAIN = 16
    N_TEST = 8
    MD_RANK = 8
    MD_PROJ_WEIGHT = 0.3

    print("=" * 80)
    print("OFFLINE SHARED DIRECTION EXPERIMENT")
    print(f"N_train={N_TRAIN}, N_test={N_TEST}, rank={MD_RANK}")
    print("=" * 80)

    # ── Load training data ──
    print("\n[1] Loading training data...")
    train_slices = load_train_slices(N_TRAIN)

    # ── Load test data ──
    print("\n[2] Loading test data...")
    test_slices = load_test_slices(N_TEST)
    print(f"  Loaded {len(test_slices)} test slices")

    # ── Load models ──
    print("\n[3] Loading models (vit_b)...")
    victim = load_samed_model("vit_b", SAM_CKPTS["vit_b"],
        os.path.join(CKPT_BASE, "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"),
        IMAGE_SIZE, 8, device=device)
    victim.eval()
    sam = load_sam_encoder("vit_b", SAM_CKPTS["vit_b"], IMAGE_SIZE, num_classes=8, device=device)
    encoder = sam.image_encoder
    encoder.eval()
    print("  Models loaded OK")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Offline — collect shared directions
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: OFFLINE — Collect shared directions from training data")
    print("=" * 60)

    # 1a: Momentum Distillation
    print("\n[1a] Momentum Distillation (MD)...")
    step = 2.0 / 255.0
    eps = EPS_FRAC
    t0 = time.time()
    md_prior = offline_md(encoder, train_slices, eps, step, STEPS, rank=MD_RANK)
    print(f"  MD done in {time.time()-t0:.1f}s. Explained var: {md_prior['explained_variance']:.4f}")
    print(f"  MD mean direction norm: {md_prior['mean_direction'].abs().sum():.4f}")
    print(f"  MD basis shape: {md_prior['basis'].shape}")

    # 1b: Frequency-Domain Universal Direction
    print("\n[1b] Frequency-Domain Universal Direction (FDUD)...")
    t0 = time.time()
    fdud_prior = offline_fdud(encoder, train_slices, eps, step, STEPS)
    print(f"  FDUD done in {time.time()-t0:.1f}s. {len(fdud_prior)} layers captured")

    # 1c: Naive perturbation averaging
    print("\n[1c] Naive Perturbation Averaging...")
    t0 = time.time()
    naive_prior = offline_naive(encoder, train_slices, eps, step, STEPS)
    print(f"  Naive done in {time.time()-t0:.1f}s. Shape: {naive_prior.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Online — evaluate all methods on test set
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: ONLINE — Evaluate on test set")
    print("=" * 60)

    # Methods to test
    test_methods = {
        "PASD (baseline)": lambda x, ep, sp: run_pasd_attack(
            encoder, x, ep, sp, STEPS, sigma=1.0, alpha=1.0, probe_mode="pasd"),
        "PASD + MD": lambda x, ep, sp: pasd_with_md_warmstart(
            encoder, x, ep, sp, STEPS, md_prior, proj_weight=MD_PROJ_WEIGHT),
        "PASD + NAIVE": lambda x, ep, sp: pasd_with_naive_warmstart(
            encoder, x, ep, sp, STEPS, naive_prior),
        "Pure L1 (baseline)": lambda x, ep, sp: run_pasd_attack(
            encoder, x, ep, sp, STEPS, probe_mode="none"),
    }

    results = {}
    for method_name, attack_fn in test_methods.items():
        print(f"\n  Testing: {method_name}...")
        dices, clean_dices, times = [], [], []

        for img_data in test_slices:
            x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
            gt = img_data[1]

            cd = eval_synapse(victim, x, gt)
            if cd < 0.01:
                del x; continue

            rg = x.max().item() - x.min().item()
            eps_pix = EPS_FRAC * rg
            step_pix = step * rg

            try:
                t0 = time.time()
                x_adv = attack_fn(x, eps_pix, step_pix)
                ad = eval_synapse(victim, x_adv, gt)
                times.append(time.time() - t0)
                dices.append(ad)
                clean_dices.append(cd)
            except Exception as e:
                print(f"    Error: {e}")
                continue
            del x_adv, x
            gc.collect()
            torch.cuda.empty_cache()

        mc, ma = np.mean(clean_dices), np.mean(dices)
        md = mc - ma
        mt = np.mean(times)
        results[method_name] = {
            "clean": mc, "adv": ma, "delta": md, "time": mt, "n": len(dices)
        }
        print(f"    Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f} ({mt:.1f}s) [{len(dices)}]")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    baseline_delta = results.get("PASD (baseline)", {}).get("delta", 0)
    for name, r in results.items():
        gain = ""
        if name != "PASD (baseline)" and "delta" in r:
            g = r["delta"] - baseline_delta
            gain = f" (vs baseline: {g:+.4f})"
        print(f"  {name:<30} Δ={r.get('delta',0):+.4f}{gain}")

    print("\nDone.")
