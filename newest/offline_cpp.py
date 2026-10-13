"""
CPP: Conditional Perturbation Predictor.

Trains a lightweight network g_θ: encoder_features → perturbation_direction.
Trained offline to mimic PASD perturbations, then used online for initialization.

Key difference from all failed approaches:
  - NOT averaging perturbations (NAIVE/k-NN/UAP)
  - NOT extracting a single direction (MD/FDUD/GA)
  - Instead: LEARNS a conditional mapping from features to perturbation

This preserves per-image specificity while leveraging cross-image knowledge.
"""
import sys, os, gc, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT); sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from attack_pasa import (run_pasd_attack, synthesize_pasd_probe,
                          _get_edge_map, _return_eta, criterion_l1)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {"vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"}
device = torch.device("cuda:1")


# ═══════════════════════════════════════════════════════════════════════════
# Perturbation Predictor Network
# ═══════════════════════════════════════════════════════════════════════════

class PerturbationPredictor(nn.Module):
    """
    Lightweight predictor: encoder features → perturbation map.
    Input: [B, 256, 64, 64] (base SAM encoder output)
    Output: [B, 3, 512, 512] (perturbation)
    """
    def __init__(self, in_channels=256, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden//2, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden//2, 3, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(hidden//2)

    def forward(self, features):
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)  # [B, 3, 64, 64]
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        return x


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
        img = data['image'].astype(np.float32); lbl = data['label'].astype(np.int64)
        if (lbl > 0).sum() < 500: continue
        slices.append((np.stack([img,img,img], axis=0), lbl))
    return slices

def load_test_slices(n=8):
    from exp_paid_optimal_attack import load_synapse_slices
    return load_synapse_slices(n)

def eval_syn(net, x, gt):
    with torch.no_grad():
        out = net(x, multimask_output=False, image_size=IMAGE_SIZE); m = out["masks"]
    pred = torch.argmax(torch.softmax(m.float(),dim=1),dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1,9):
        pc=(pred==c).astype(np.float32); gc=(gt==c).astype(np.float32)
        if gc.sum()==0: continue
        inter=(pc*gc).sum(); dices.append(float(2*inter/(pc.sum()+gc.sum()+1e-8)))
    return float(np.mean(dices)) if dices else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Train CPP
# ═══════════════════════════════════════════════════════════════════════════

def train_cpp(encoder, train_slices, epochs=50, batch_size=8, lr=1e-3):
    """
    Train the Conditional Perturbation Predictor.
    For each training image: extract features, run PASD to get target perturbation,
    train predictor to map features → perturbation.
    """
    predictor = PerturbationPredictor().to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    step = 2.0 / 255.0
    eps = 4.0 / 255

    # Collect training data (features, perturbations)
    print("  Collecting training data...")
    features_list, perts_list = [], []
    for i, (img, lbl) in enumerate(train_slices):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)
        rg = x.max().item() - x.min().item()

        # Extract features
        with torch.no_grad():
            feats, _ = encoder(x)  # [1, 256, 64, 64]

        # Run PASD to get target perturbation
        x_adv = run_pasd_attack(encoder, x, eps*rg, step*rg, 20,
                               sigma=1.0, alpha=1.0, probe_mode="pasd")
        pert = (x_adv - x).detach()

        features_list.append(feats.cpu())
        perts_list.append(pert.cpu())

        del x_adv, x; gc.collect(); torch.cuda.empty_cache()
        if (i+1) % 32 == 0:
            print(f"    Collected {i+1}/{len(train_slices)}")

    # Create dataset
    all_feats = torch.cat(features_list, dim=0)  # [N, 256, 64, 64]
    all_perts = torch.cat(perts_list, dim=0)     # [N, 3, 512, 512]
    dataset = TensorDataset(all_feats, all_perts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  Dataset: {len(dataset)} samples")

    # Train
    print("  Training predictor...")
    for epoch in range(epochs):
        total_loss = 0.0
        for feats_batch, perts_batch in loader:
            feats_batch = feats_batch.to(device)
            perts_batch = perts_batch.to(device)

            pred = predictor(feats_batch)

            # Cosine similarity loss (direction matters more than magnitude)
            p_flat = pred.reshape(pred.size(0), -1)
            t_flat = perts_batch.reshape(perts_batch.size(0), -1)
            cosine_sim = F.cosine_similarity(p_flat, t_flat, dim=1)
            loss = (1.0 - cosine_sim).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch+1) % 10 == 0:
            avg_sim = 1.0 - total_loss / len(loader)
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f}, avg cosine_sim={avg_sim:.4f}")

    return predictor


# ═══════════════════════════════════════════════════════════════════════════
# Online: PASD with CPP initialization
# ═══════════════════════════════════════════════════════════════════════════

def pasd_with_cpp_init(encoder, x, eps_pix, step_pix, steps,
                        predictor, alpha=1.0, cpp_weight=0.5):
    """
    PASD with CPP-predicted perturbation as initialization.
    cpp_weight: 0=ignore predictor, 1=full predictor init
    """
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = encoder(x)

    # Get CPP prediction
    with torch.no_grad():
        feats, _ = encoder(x)
        pred_pert = predictor(feats)

    edge_map = _get_edge_map(x).detach()

    # Initialize with blend of random + CPP prediction
    delta_init = cpp_weight * torch.clamp(pred_pert, -eps_pix, eps_pix)
    x_adv = Variable(torch.clamp(x + delta_init, min_val, max_val), requires_grad=True)
    momentum = torch.zeros_like(x).detach()

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
                probe = synthesize_pasd_probe(F_clean, sigma=1.0, alpha=alpha)
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
        momentum = 0.5*momentum.detach()+grad; momentum = momentum.detach()
        eta = _return_eta(momentum, step_pix).detach()
        x_adv = Variable(x_adv.data+eta, requires_grad=True)
        eta_t = torch.clamp(x_adv.data-x.data, -eps_pix, eps_pix)
        x_adv = Variable(x.data+eta_t, requires_grad=True)
        x_adv = Variable(torch.clamp(x_adv, min_val, max_val), requires_grad=True)

    return x_adv.detach()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N_TRAIN = 64
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

    # Train CPP
    print("\n=== Training CPP ===")
    predictor = train_cpp(encoder, train_slices, epochs=50, batch_size=8)

    # Test
    print("\n=== Online: CPP Init vs Baseline ===")
    for eps_frac in [4.0/255, 8.0/255]:
        for cpp_w in [0.0, 0.3, 0.5, 0.7, 1.0]:
            dices, cds = [], []
            for img_data in test_slices:
                x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                gt = img_data[1]
                cd = eval_syn(victim, x, gt)
                if cd < 0.01: del x; continue
                rg = x.max().item()-x.min().item(); ep = eps_frac*rg; sp = step*rg
                try:
                    if cpp_w == 0.0:
                        x_adv = run_pasd_attack(encoder, x, ep, sp, 20,
                                               sigma=1.0, alpha=1.0, probe_mode="pasd")
                    else:
                        x_adv = pasd_with_cpp_init(encoder, x, ep, sp, 20,
                                                   predictor, cpp_weight=cpp_w)
                    ad = eval_syn(victim, x_adv, gt)
                    dices.append(ad); cds.append(cd)
                except Exception as e:
                    print(f"    Error: {e}"); continue
                del x_adv, x; gc.collect(); torch.cuda.empty_cache()
            mc, ma = np.mean(cds), np.mean(dices)
            label = f"CPP w={cpp_w}" if cpp_w > 0 else "PASD baseline"
            print(f"  eps={eps_frac*255:.0f}/255 {label:<25} Clean={mc:.4f} Adv={ma:.4f} Δ={mc-ma:+.4f}")

    del victim, sam; torch.cuda.empty_cache()
    print("Done")
