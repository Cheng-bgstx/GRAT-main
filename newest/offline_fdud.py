"""
FDUD: Frequency-Domain Universal Direction.

Key insight from PASD theory: the main effect is in FREQUENCY DOMAIN (+48%).
Offline method must operate in frequency domain, not pixel space.

Offline: average the frequency-domain drift vectors from PASD attacks on training data.
Online: inject the averaged drift into the PASD probe (replacing/guiding random noise).
"""
import sys, os, gc, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT); sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from attack_pasa import (_get_edge_map, _return_eta,
                          synthesize_pasd_probe, criterion_l1)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {"vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"}
device = torch.device("cuda:1")


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_train_slices(n=64):
    tdir = f"{ROOT}/data/Synapse_preprocess/train_npz"
    files = sorted([f for f in os.listdir(tdir) if f.endswith('.npz')])
    slices = []
    for f in files:
        if len(slices) >= n: break
        data = np.load(os.path.join(tdir, f))
        img = data['image'].astype(np.float32)
        lbl = data['label'].astype(np.int64)
        if (lbl > 0).sum() < 500: continue
        slices.append((np.stack([img,img,img], axis=0), lbl))
    return slices

def load_test_slices(n=8):
    from exp_paid_optimal_attack import load_synapse_slices
    return load_synapse_slices(n)

def eval_synapse(net, img_t, gt):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out["masks"]
    pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred == c).astype(np.float32); gc = (gt == c).astype(np.float32)
        if gc.sum() == 0: continue
        inter = (pc * gc).sum()
        dices.append(float(2.0*inter/(pc.sum()+gc.sum()+1e-8)))
    return float(np.mean(dices)) if dices else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FDUD: Offline collection of frequency-domain drift
# ═══════════════════════════════════════════════════════════════════════════

def collect_fdud(encoder, train_slices, eps_frac, step_frac, steps, num_layers=6):
    """
    For each training image, run PASD attack and collect:
    - Per-layer frequency-domain drift direction (L1-normalized)
    - Averaged across images → universal frequency drift direction

    Returns: fdud_mean[L] = [1, C, H, W] complex tensor (averaged FFT drift)
    """
    accumulator = None
    count = 0

    for i, (img, lbl) in enumerate(train_slices):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)
        rg = x.max().item() - x.min().item()
        eps_pix = eps_frac * rg
        step_pix = step_frac * rg

        # Run PASD and capture per-layer frequency drift
        max_val, min_val = x.max(), x.min()
        with torch.no_grad():
            targets, targets_layer = encoder(x)

        edge_map = _get_edge_map(x).detach()
        x_adv = Variable(x.data + torch.randn_like(x) * 1e-5, requires_grad=True)
        momentum = torch.zeros_like(x).detach()

        pre_layer = int(len(targets_layer) * 0.9)
        n_avail = min(num_layers, pre_layer)
        layer_ids = list(range(pre_layer - n_avail, pre_layer))

        for step in range(steps):
            encoder.zero_grad()
            inp = x_adv + torch.randn_like(x_adv) * 0.0001 if step == 0 else x_adv
            features, features_layer = encoder(inp)

            loss_feat = criterion_l1(features * 0.9999, targets)
            probe_losses = []
            for _ in range(2):  # K=2
                diff_list = []
                for idx in layer_ids:
                    F_adv_i = features_layer[idx].unsqueeze(0)
                    F_clean_i = targets_layer[idx].unsqueeze(0).detach()
                    probe = synthesize_pasd_probe(F_clean_i, sigma=1.0, alpha=1.0)
                    diff = (F_adv_i * 0.999 - F_clean_i) - 2.0 * probe
                    diff_list.append(diff.squeeze(0))
                diff = torch.stack(diff_list, dim=0)
                sel_t = torch.stack([targets_layer[i] for i in layer_ids], dim=0)
                im = torch.stack([sel_t[i].abs().mean() for i in range(len(sel_t))])
                im = im.view(diff.shape[0], 1, 1, 1)
                probe_losses.append(torch.abs(torch.mean(diff / (im+1e-6), dim=0)).mean())
            loss = loss_feat + torch.stack(probe_losses).mean()

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

        # Collect per-layer frequency drift (L1-normalized, complex)
        with torch.no_grad():
            _, feat_adv_layers = encoder(x_adv.detach())
        drifts = []
        for idx in layer_ids:
            Zc = torch.fft.fftn(targets_layer[idx].unsqueeze(0), dim=(-2,-1), norm="ortho")
            Za = torch.fft.fftn(feat_adv_layers[idx].unsqueeze(0), dim=(-2,-1), norm="ortho")
            dZ = Za - Zc
            dZ = dZ / (torch.abs(dZ).sum() + 1e-8)
            drifts.append(dZ.cpu())

        if accumulator is None:
            accumulator = [d.clone() for d in drifts]
        else:
            for j in range(len(accumulator)):
                accumulator[j] += drifts[j]
        count += 1

        del x_adv, x
        gc.collect(); torch.cuda.empty_cache()

        if (i+1) % 16 == 0:
            print(f"  [FDUD] {i+1}/{len(train_slices)}")

    # Average
    for j in range(len(accumulator)):
        accumulator[j] = accumulator[j] / count
    return accumulator  # List of [1, C, H, W] complex tensors


# ═══════════════════════════════════════════════════════════════════════════
# Online: PASD with FDUD-injected probe
# ═══════════════════════════════════════════════════════════════════════════

def pasd_with_fdud(encoder, x, eps_pix, step_pix, steps, fdud_prior, gamma=0.5,
                    sigma=1.0, alpha=1.0, num_layers=6):
    """
    PASD attack with FDUD prior injected into the probe.

    probe = IFFT( (sigma*N_white + gamma*D_universal) * A_clean^α * exp(j*Φ_disrupted) )

    gamma controls blend: 0 = pure PASD, 1 = pure FDUD
    """
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = encoder(x)

    edge_map = _get_edge_map(x).detach()
    x_adv = Variable(x.data + torch.randn_like(x) * 1e-5, requires_grad=True)
    momentum = torch.zeros_like(x).detach()

    pre_layer = int(len(targets_layer) * 0.9)
    n_avail = min(num_layers, pre_layer)
    layer_ids = list(range(pre_layer - n_avail, pre_layer))

    for step in range(steps):
        encoder.zero_grad()
        inp = x_adv + torch.randn_like(x_adv) * 0.0001 if step == 0 else x_adv
        features, features_layer = encoder(inp)

        loss_feat = criterion_l1(features * 0.9999, targets)
        probe_losses = []
        for _ in range(2):
            diff_list = []
            for j, idx in enumerate(layer_ids):
                F_adv_i = features_layer[idx].unsqueeze(0)
                F_clean_i = targets_layer[idx].unsqueeze(0).detach()

                # Standard PASD probe
                Z_clean = torch.fft.fftn(F_clean_i, dim=(-2,-1), norm="ortho")
                A_clean = torch.abs(Z_clean)
                Phase_clean = torch.angle(Z_clean)
                # Phase disruption
                shape_ph = Phase_clean.shape
                idx_perm = torch.randperm(Phase_clean.numel(), device=Phase_clean.device)
                Phase_use = Phase_clean.reshape(-1)[idx_perm].reshape(shape_ph)
                carrier = torch.exp(1j * Phase_use)

                # Amplitude prior
                Sigma = A_clean.pow(alpha) + 1e-4
                N_white = torch.randn_like(A_clean, device=A_clean.device)

                # FDUD injection: blend universal frequency direction with white noise
                if fdud_prior is not None and j < len(fdud_prior):
                    D_univ = fdud_prior[j].to(device)
                    # Blend: random + universal direction
                    probe_fft = (sigma * N_white + gamma * D_univ) * Sigma * carrier
                else:
                    probe_fft = sigma * N_white * Sigma * carrier

                probe = torch.fft.ifftn(probe_fft, dim=(-2,-1), norm="ortho").real
                diff = (F_adv_i * 0.999 - F_clean_i) - 2.0 * probe
                diff_list.append(diff.squeeze(0))

            diff = torch.stack(diff_list, dim=0)
            sel_t = torch.stack([targets_layer[i] for i in layer_ids], dim=0)
            im = torch.stack([sel_t[i].abs().mean() for i in range(len(sel_t))])
            im = im.view(diff.shape[0], 1, 1, 1)
            probe_losses.append(torch.abs(torch.mean(diff / (im+1e-6), dim=0)).mean())

        loss = loss_feat + torch.stack(probe_losses).mean()
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
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from attack_pasa import run_pasd_attack

    N_TRAIN = 64
    N_TEST = 8
    step = 2.0 / 255.0

    train_slices = load_train_slices(N_TRAIN)
    test_slices = load_test_slices(N_TEST)
    print(f"Train: {len(train_slices)}, Test: {len(test_slices)}")

    victim = load_samed_model("vit_b", SAM_CKPTS["vit_b"],
        os.path.join(CKPT_BASE, "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"),
        IMAGE_SIZE, 8, device=device)
    victim.eval()
    sam = load_sam_encoder("vit_b", SAM_CKPTS["vit_b"], IMAGE_SIZE, num_classes=8, device=device)
    encoder = sam.image_encoder; encoder.eval()

    # Offline: FDUD collection at eps=4/255
    print("\n=== OFFLINE FDUD (eps=4/255) ===")
    eps_frac = 4.0 / 255
    fdud_prior = collect_fdud(encoder, train_slices, eps_frac, step, 20)
    print(f"  Collected {len(fdud_prior)} layers of frequency drift")

    # Online: sweep gamma values
    print("\n=== ONLINE: FDUD gamma sweep ===")
    for gamma in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0]:
        dices, cds = [], []
        for img_data in test_slices:
            x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
            gt = img_data[1]
            cd = eval_synapse(victim, x, gt)
            if cd < 0.01: del x; continue
            rg = x.max().item() - x.min().item()
            try:
                if gamma == 0.0:
                    x_adv = run_pasd_attack(encoder, x, eps_frac*rg, step*rg, 20,
                                           sigma=1.0, alpha=1.0, probe_mode='pasd')
                else:
                    x_adv = pasd_with_fdud(encoder, x, eps_frac*rg, step*rg, 20,
                                           fdud_prior if gamma > 0 else None, gamma=gamma)
                ad = eval_synapse(victim, x_adv, gt)
                dices.append(ad); cds.append(cd)
            except Exception as e:
                continue
            del x_adv, x; gc.collect(); torch.cuda.empty_cache()
        mc, ma = np.mean(cds), np.mean(dices)
        print(f"  gamma={gamma:<5} Clean={mc:.4f} Adv={ma:.4f} Δ={mc-ma:+.4f}")

    del victim, sam; torch.cuda.empty_cache()
    print("Done")
