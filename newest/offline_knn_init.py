"""
K-NN Perturbation Initialization for PASD.

Key insight: DON'T average perturbations across images (MD/FDUD/UAP all fail at this).
Instead, for each test image, find k nearest training images in FEATURE SPACE,
and blend THEIR perturbations as initialization.

Theory: images with similar encoder features have similar vulnerability structure.
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
                          _get_edge_map, _return_eta, criterion_l1)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {"vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"}
device = torch.device("cuda:1")


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


def pasd_with_init(encoder, x, eps_pix, step_pix, steps, x_init=None,
                    sigma=1.0, alpha=1.0, num_layers=6):
    """PASD with an explicit initial perturbation."""
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        targets, targets_layer = encoder(x)

    edge_map = _get_edge_map(x).detach()

    if x_init is not None:
        # Start from k-NN perturbation (clamped to epsilon)
        delta_init = torch.clamp(x_init - x, -eps_pix, eps_pix)
        x_adv = Variable(x + delta_init, requires_grad=True)
    else:
        x_adv = Variable(x.clone().detach() + torch.randn_like(x) * 1e-5, requires_grad=True)

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
            for idx in layer_ids:
                F_adv = features_layer[idx].unsqueeze(0)
                F_clean = targets_layer[idx].unsqueeze(0).detach()
                probe = synthesize_pasd_probe(F_clean, sigma=sigma, alpha=alpha)
                diff = (F_adv * 0.999 - F_clean) - 2.0 * probe
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


if __name__ == "__main__":
    N_TRAIN = 64
    K = 3  # number of neighbors
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

    # ═══ Offline: build perturbation database ═══
    print("\n=== Offline: Building perturbation database ===")
    database_features = []  # each: feature vector [D]
    database_perts = []     # each: perturbation tensor [1,3,512,512]

    for i, (img, lbl) in enumerate(train_slices):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)
        rg = x.max().item() - x.min().item()

        # Extract features (global average pool of encoder output)
        with torch.no_grad():
            feats, _ = encoder(x)
        feat_vec = F.adaptive_avg_pool2d(feats, (1,1)).flatten().cpu()

        # Run PASD to get perturbation
        x_adv = run_pasd_attack(encoder, x, 4.0/255*rg, step*rg, 20,
                               sigma=1.0, alpha=1.0, probe_mode="pasd")
        pert = (x_adv - x).detach().cpu()

        database_features.append(feat_vec)
        database_perts.append(pert)

        del x_adv, x
        gc.collect(); torch.cuda.empty_cache()

        if (i+1) % 16 == 0:
            print(f"  Processed {i+1}/{len(train_slices)}")

    # Stack features for fast k-NN
    all_features = torch.stack(database_features)  # [N, D]
    print(f"  Database: {len(database_features)} entries, feature dim={all_features.shape[1]}")

    # ═══ Online: k-NN initialization + PASD ═══
    print("\n=== Online: k-NN Init vs Baseline ===")

    for eps_frac in [4.0/255, 8.0/255]:
        for method_label, use_knn in [("PASD (baseline)", False), (f"PASD + {K}-NN init", True)]:
            dices, cds = [], []

            for img_data in test_slices:
                x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                gt = img_data[1]
                cd = eval_syn(victim, x, gt)
                if cd < 0.01: del x; continue

                rg = x.max().item() - x.min().item()
                ep = eps_frac * rg; sp = step * rg

                try:
                    x_init = None
                    if use_knn:
                        # Get test image features
                        with torch.no_grad():
                            feats, _ = encoder(x)
                        test_vec = F.adaptive_avg_pool2d(feats, (1,1)).flatten()

                        # k-NN in feature space
                        dists = torch.norm(all_features - test_vec.cpu(), dim=1)
                        _, knn_idx = torch.topk(dists, K, largest=False)

                        # Blend k nearest perturbations
                        blended_pert = torch.zeros_like(x).cpu()
                        total_weight = 0.0
                        for idx in knn_idx:
                            weight = 1.0 / (dists[idx].item() + 1e-8)
                            blended_pert += weight * database_perts[idx]
                            total_weight += weight
                        blended_pert = blended_pert / total_weight

                        x_init = (x.cpu() + blended_pert).to(device)

                    x_adv = pasd_with_init(encoder, x, ep, sp, 20, x_init=x_init)
                    ad = eval_syn(victim, x_adv, gt)
                    dices.append(ad); cds.append(cd)
                except Exception as e:
                    print(f"    Error: {e}"); continue
                del x_adv, x; gc.collect(); torch.cuda.empty_cache()

            mc, ma = np.mean(cds), np.mean(dices)
            md = mc - ma
            print(f"  eps={eps_frac*255:.0f}/255 {method_label:<25} Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f}")

    del victim, sam; torch.cuda.empty_cache()
    print("Done")
