"""
UAP-PASD: Universal Adversarial Perturbation trained with PASD loss.

Key difference from failed approaches:
  NAIVE: average perturbations AFTER attack → cancel out
  MD: PCA on momenta → noisy, high-dim instability
  k-NN: retrieve perturbations → not similar enough

UAP-PASD: iteratively accumulate PASD LOSS GRADIENT across training images,
  projected back to epsilon ball. This is the classic UAP algorithm (Moosavi-Dezfooli
  et al., CVPR 2017) but using PASD's frequency-domain loss instead of cross-entropy.

  The accumulated UAP encodes the shared direction that consistently increases
  PASD loss across training images → a genuine shared vulnerability direction.

Online variants:
  1. UAP-init: x_adv = x + UAP, then standard PASD from there
  2. UAP-momentum: momentum_0 = UAP (persistent bias)
  3. UAP-guided: blend UAP into probe at each step
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
from attack_pasa import (run_pasd_attack, synthesize_pasd_probe,
                          _get_edge_map, _return_eta, criterion_l1,
                          calculate_pasd_loss)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {"vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"}
device = torch.device("cuda:1")  # Free GPU


def load_train_slices(n=64):
    tdir = f"{ROOT}/data/Synapse_preprocess/train_npz"
    files = sorted([f for f in os.listdir(tdir) if f.endswith('.npz')])
    slices = []
    for f in files:
        if len(slices) >= n: break
        data = np.load(os.path.join(tdir, f))
        img = data['image'].astype(np.float32); lbl = data['label'].astype(np.int64)
        if (lbl > 0).sum() < 500: continue
        slices.append((np.stack([img,img,img], axis=0), lbl))
    return slices


def load_test_slices(n=8):
    from exp_paid_optimal_attack import load_synapse_slices
    return load_synapse_slices(n)


def eval_syn(net, x, gt):
    with torch.no_grad():
        out = net(x, multimask_output=False, image_size=IMAGE_SIZE)
        m = out["masks"]
    pred = torch.argmax(torch.softmax(m.float(),dim=1),dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1,9):
        pc=(pred==c).astype(np.float32); gc=(gt==c).astype(np.float32)
        if gc.sum()==0: continue
        inter=(pc*gc).sum()
        dices.append(float(2*inter/(pc.sum()+gc.sum()+1e-8)))
    return float(np.mean(dices)) if dices else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Offline: UAP trained with PASD loss
# ═══════════════════════════════════════════════════════════════════════════

def train_uap_pasd(encoder, train_slices, eps_frac, step_frac, uap_steps=5, epochs=5):
    """
    Train a Universal Adversarial Perturbation using PASD's loss function.

    Classic UAP algorithm (Moosavi-Dezfooli et al.):
      δ = 0
      for each x_i (for multiple epochs):
          if PASD(x_i + δ) succeeds (loss is high enough):
              continue  # already fooled
          else:
              compute gradient of PASD loss w.r.t. δ
              δ += step * sign(grad)
              δ = clip(δ, -eps, eps)  # project to L_inf ball
    """
    rg = 1.0  # pixel range for normalized images
    eps_pix = eps_frac * rg
    step_pix = step_frac * rg
    max_val, min_val = 1.0, 0.0

    # Initialize UAP as small random
    uap = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    uap += torch.randn_like(uap) * 0.001
    uap = torch.clamp(uap, -eps_pix, eps_pix)

    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}...")
        n_fooled = 0
        for i, (img, lbl) in enumerate(train_slices):
            x = torch.from_numpy(img).unsqueeze(0).float().to(device)

            # Check if UAP already works on this image
            x_adv_curr = torch.clamp(x + uap, min_val, max_val)
            with torch.no_grad():
                feat_adv, feat_adv_layers = encoder(x_adv_curr)
                feat_clean, feat_clean_layers = encoder(x)

            # Compute PASD loss with current UAP
            loss_curr = calculate_pasd_loss(
                features=feat_adv, targets=feat_clean,
                features_layer=feat_adv_layers, targets_layer=feat_clean_layers,
                sigma=1.0, alpha=1.0, probe_mode="pasd", K=1, num_layers=6,
            )

            # If loss is already high enough, skip
            if loss_curr.item() > 2.0:  # threshold for "already fooled"
                n_fooled += 1
                continue

            # Otherwise: compute gradient of PASD loss w.r.t. UAP
            uap_var = Variable(uap.clone().detach(), requires_grad=True)
            x_adv = torch.clamp(x + uap_var, min_val, max_val)
            feat_adv, feat_adv_layers = encoder(x_adv)
            feat_clean, feat_clean_layers = encoder(x)

            loss = calculate_pasd_loss(
                features=feat_adv, targets=feat_clean,
                features_layer=feat_adv_layers, targets_layer=feat_clean_layers,
                sigma=1.0, alpha=1.0, probe_mode="pasd", K=1, num_layers=6,
            )

            grad = torch.autograd.grad(loss, uap_var)[0]

            # Update UAP: move in gradient direction to INCREASE PASD loss
            uap = uap + step_pix * torch.sign(grad)
            uap = torch.clamp(uap, -eps_pix, eps_pix)

            del x, uap_var, x_adv, feat_adv, feat_clean
            gc.collect(); torch.cuda.empty_cache()

            if (i+1) % 32 == 0:
                print(f"    {i+1}/{len(train_slices)}, fooled: {n_fooled}, loss: {loss.item():.3f}")

        print(f"    Epoch {epoch+1}: fooled {n_fooled}/{len(train_slices)}, "
              f"|UAP|_inf={uap.abs().max().item():.4f}, "
              f"|UAP|_1={uap.abs().mean().item():.6f}")

    return uap.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Online: PASD with UAP warm-start (multiple strategies)
# ═══════════════════════════════════════════════════════════════════════════

def pasd_with_uap_warmstart(encoder, x, eps_pix, step_pix, steps, uap,
                             strategy="momentum_init"):
    """
    PASD attack with UAP warm-start.

    Strategies:
      - "momentum_init": momentum_0 = uap / |uap| (directional warm-start)
      - "x_init": x_adv = x + clip(uap, -eps, eps) (perturbation init)
    """
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = encoder(x)
    edge_map = _get_edge_map(x).detach()

    if strategy == "x_init":
        delta_init = torch.clamp(uap, -eps_pix, eps_pix)
        x_adv = Variable(torch.clamp(x + delta_init, min_val, max_val), requires_grad=True)
        momentum = torch.zeros_like(x).detach()
    elif strategy == "momentum_init":
        x_adv = Variable(x.clone().detach() + torch.randn_like(x)*1e-5, requires_grad=True)
        # Normalized UAP as momentum direction
        uap_dir = uap / (uap.abs().sum() + 1e-12)
        momentum = uap_dir.clone().detach()
    elif strategy == "momentum_bias":
        x_adv = Variable(x.clone().detach() + torch.randn_like(x)*1e-5, requires_grad=True)
        momentum = torch.zeros_like(x).detach()
        uap_bias = uap / (uap.abs().sum() + 1e-12)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    pre_layer = int(len(targets_layer) * 0.9)
    n_avail = min(6, pre_layer)
    layer_ids = list(range(pre_layer - n_avail, pre_layer))

    for step in range(steps):
        encoder.zero_grad()
        inp = x_adv + torch.randn_like(x_adv)*0.0001 if step == 0 else x_adv
        features, features_layer = encoder(inp)

        loss_feat = criterion_l1(features*0.9999, targets)
        probe_losses = []
        for _ in range(2):
            diff_list = []
            for idx in layer_ids:
                F_adv = features_layer[idx].unsqueeze(0)
                F_clean = targets_layer[idx].unsqueeze(0).detach()
                probe = synthesize_pasd_probe(F_clean, sigma=1.0, alpha=1.0)
                diff = (F_adv*0.999 - F_clean) - 2.0*probe
                diff_list.append(diff.squeeze(0))
            diff = torch.stack(diff_list, dim=0)
            sel_t = torch.stack([targets_layer[i] for i in layer_ids], dim=0)
            im = torch.stack([sel_t[i].abs().mean() for i in range(len(sel_t))])
            im = im.view(diff.shape[0],1,1,1)
            probe_losses.append(torch.abs(torch.mean(diff/(im+1e-6), dim=0)).mean())
        loss = loss_feat + torch.stack(probe_losses).mean()

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
        grad = grad*(1.0+3.0*edge_map)
        grad = grad/(grad.norm(p=1)+1e-12)

        momentum = 0.5*momentum.detach()+grad
        if strategy == "momentum_bias":
            momentum = momentum + 0.1 * uap_bias  # persistent UAP bias

        momentum = momentum.detach()
        eta = _return_eta(momentum, step_pix).detach()
        x_adv = Variable(x_adv.data+eta, requires_grad=True)
        eta_t = torch.clamp(x_adv.data-x.data, -eps_pix, eps_pix)
        x_adv = Variable(x.data+eta_t, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, min_val, max_val), requires_grad=True)

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N_TRAIN = 32
    step = 2.0 / 255.0

    train_slices = load_train_slices(N_TRAIN)
    test_slices = load_test_slices(8)
    print(f"Train: {len(train_slices)}, Test: {len(test_slices)}")

    victim = load_samed_model("vit_b", SAM_CKPTS["vit_b"],
        os.path.join(CKPT_BASE, "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"),
        IMAGE_SIZE, 8, device=device)
    victim.eval()
    sam = load_sam_encoder("vit_b", SAM_CKPTS["vit_b"], IMAGE_SIZE, num_classes=8, device=device)
    encoder = sam.image_encoder; encoder.eval()

    # ═══ Train UAP-PASD at multiple eps ═══
    UAP_PRIORS = {}
    for eps_frac in [4.0/255, 8.0/255]:
        print(f"\n=== Train UAP-PASD: eps={eps_frac*255:.0f}/255 ===")
        uap = train_uap_pasd(encoder, train_slices, eps_frac, step, uap_steps=5, epochs=3)
        UAP_PRIORS[eps_frac] = uap
        print(f"  UAP norm: inf={uap.abs().max():.4f}, L1={uap.abs().mean():.6f}")

    # ═══ Online: Test all strategies ═══
    print("\n=== Online Evaluation ===")
    for eps_frac in [4.0/255, 8.0/255]:
        uap = UAP_PRIORS[eps_frac]
        print(f"\n  eps={eps_frac*255:.0f}/255:")

        configs = [
            ("PASD baseline", None, None),
            ("UAP momentum_init", "momentum_init", None),
            ("UAP momentum_bias", "momentum_bias", None),
            ("UAP x_init", "x_init", None),
            ("Pure L1 baseline", None, "none"),
        ]

        for label, strategy, probe_override in configs:
            dices, cds = [], []
            for img_data in test_slices:
                x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                gt = img_data[1]
                cd = eval_syn(victim, x, gt)
                if cd < 0.01: del x; continue
                rg = x.max().item()-x.min().item()
                ep = eps_frac*rg; sp = step*rg

                try:
                    if probe_override == "none":
                        x_adv = run_pasd_attack(encoder, x, ep, sp, 20, probe_mode="none")
                    elif strategy is None:
                        x_adv = run_pasd_attack(encoder, x, ep, sp, 20,
                                               sigma=1.0, alpha=1.0, probe_mode="pasd")
                    else:
                        x_adv = pasd_with_uap_warmstart(encoder, x, ep, sp, 20,
                                                        uap, strategy=strategy)
                    ad = eval_syn(victim, x_adv, gt)
                    dices.append(ad); cds.append(cd)
                except Exception as e:
                    print(f"    Error: {e}"); continue
                del x_adv, x; gc.collect(); torch.cuda.empty_cache()

            mc, ma = np.mean(cds), np.mean(dices)
            md = mc-ma
            baseline = None
            if label == "PASD baseline":
                baseline = md
            gain_str = ""
            if baseline and label != "PASD baseline":
                gain = md - baseline
                gain_str = f" (gain: {gain:+.4f})"
            print(f"    {label:<25} Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f}{gain_str}")

    del victim, sam; torch.cuda.empty_cache()
    print("\nDone")
