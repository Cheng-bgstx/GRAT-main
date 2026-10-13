#!/usr/bin/env python3
"""
MSA Pure L1 Validation — Resume: REFUGE and Synapse models ONLY.

Continuation of exp_msa_comprehensive.py after the REFUGE import crash.
Covers: MSA_REF_vit_{b,l,h}, MSA_Syn_vit_{b,l,h}
"""

import os, sys, time, json, gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_medsa_model
from attack_codd import _get_edge_map

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0")
IM_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

MSA_MODELS = {
    "MSA_REF_vit_b": {"vit": "vit_b", "dir": "REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24", "ds": "REFUGE", "mm": 9},
    "MSA_REF_vit_l": {"vit": "vit_l", "dir": "REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24", "ds": "REFUGE", "mm": 9},
    "MSA_REF_vit_h": {"vit": "vit_h", "dir": "REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24",  "ds": "REFUGE", "mm": 9},
    "MSA_Syn_vit_b": {"vit": "vit_b", "dir": "Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24",  "ds": "Synapse", "mm": 9},
    "MSA_Syn_vit_l": {"vit": "vit_l", "dir": "Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24",  "ds": "Synapse", "mm": 9},
    "MSA_Syn_vit_h": {"vit": "vit_h", "dir": "Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24",   "ds": "Synapse", "mm": 9},
}

def load_synapse_data(n=12):
    from exp_paid_optimal_attack import load_synapse_slices
    return load_synapse_slices(n)

def load_refuge_data(n=12):
    from exp_paid_optimal_attack import load_refuge_images
    return load_refuge_images(n)

# ── MSA Evaluation ──────────────────────────────────────────────────────────────

def evaluate_msa_synapse(model, img_t, gt_slice):
    import random as _random
    _random.seed(42)
    pt_list, lbl_list = [], []
    for c in range(1, 9):
        indices = np.argwhere(gt_slice == c)
        if len(indices) > 0:
            r, cc = indices[_random.randint(0, len(indices)-1)]
            pt_list.append(np.array([cc, r]))
            lbl_list.append(1)
        else:
            pt_list.append(np.array([0, 0]))
            lbl_list.append(0)
    coords = torch.tensor(np.stack(pt_list, axis=0), dtype=torch.float, device=device).unsqueeze(0)
    labels = torch.tensor(lbl_list, dtype=torch.int, device=device).unsqueeze(0)
    img = img_t.to(dtype=torch.float32, device=device)
    with torch.no_grad():
        imgs_pre = model.preprocess(img)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    pred_classes = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred_classes == c).astype(np.float32)
        tc = (gt_slice == c).astype(np.float32)
        denom = pc.sum() + tc.sum()
        dices.append(2.0 * (pc * tc).sum() / (denom + 1e-8) if denom > 0 else 1.0)
    return float(np.mean(dices))


def evaluate_msa_refuge(model, img_t, gt_label):
    pt_coords = torch.tensor([[[0.0, 0.0]]], device=device)
    pt_labels = torch.tensor([[1]], dtype=torch.int, device=device)
    img = img_t.to(dtype=torch.float32, device=device)
    with torch.no_grad():
        imgs_pre = model.preprocess(img)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(pt_coords, pt_labels), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    probs = torch.sigmoid(pred[0])
    cup_pred = (probs[0] > 0.5).float().cpu().numpy()
    disc_pred = (probs[1] > 0.5).float().cpu().numpy()
    cup_gt = (gt_label[0] > 0.5).astype(np.float32)
    disc_gt = (gt_label[1] > 0.5).astype(np.float32)
    d_cup = 2.0 * (cup_pred * cup_gt).sum() / (cup_pred.sum() + cup_gt.sum() + 1e-8)
    d_disc = 2.0 * (disc_pred * disc_gt).sum() / (disc_pred.sum() + disc_gt.sum() + 1e-8)
    return float((d_cup + d_disc) / 2.0)

# ── Attack ──────────────────────────────────────────────────────────────────────

def attack_pure_l1_msa(model, x, eps_pix, step_pix, steps, edge_map_fn=None):
    criterion = torch.nn.L1Loss()
    max_val, min_val = x.max(), x.min()
    with torch.no_grad():
        imgs_pre = model.preprocess(x)
        targets = model.image_encoder(imgs_pre)
    x_adv = x.clone().detach() + torch.randn_like(x) * 1e-5
    edge_map = edge_map_fn(x).detach() if edge_map_fn else torch.zeros_like(x)
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        imgs_pre_adv = model.preprocess(x_adv)
        features = model.image_encoder(imgs_pre_adv)
        loss = criterion(features, targets)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad * (1.0 + 3.0 * edge_map)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 0.5 * momentum + grad
        x_adv = x_adv.detach() + step_pix * momentum.sign()
        eta = torch.clamp(x_adv - x, -eps_pix, eps_pix)
        x_adv = torch.clamp(x + eta, min_val, max_val).detach()
    return x_adv.detach()


def attack_random_direction(x, eps_pix):
    noise = torch.randn_like(x)
    noise = noise / (noise.abs().max() + 1e-8) * eps_pix
    max_val, min_val = x.max(), x.min()
    return torch.clamp(x + noise, min_val, max_val)

# ── Main ────────────────────────────────────────────────────────────────────────

EPS_LIST = [4.0/255, 8.0/255, 16.0/255]
STEPS_LIST = [10, 20, 40]
N_IMAGES = 12

# Load prior results
prior_path = os.path.join(OUTPUT_DIR, "msa_comprehensive.json")
all_results = []
if os.path.exists(prior_path):
    with open(prior_path) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} prior results from msa_comprehensive.json")

print("=" * 90)
print("MSA PURE L1 VALIDATION — RESUME (REFUGE + Synapse)")
print("=" * 90)

for model_key in sorted(MSA_MODELS.keys()):
    cfg = MSA_MODELS[model_key]
    ds = cfg["ds"]
    print(f"\n{'='*80}")
    print(f"  {model_key}  |  {cfg['vit']}  |  {ds}")
    print(f"{'='*80}")

    try:
        msa_dir = os.path.join(CKPT_BASE, "MedicalSAMAdapter", cfg["dir"])
        msa_ckpt = os.path.join(msa_dir, "Model", "best_dice_checkpoint.pth")
        net = load_medsa_model(cfg["vit"], SAM_CKPTS[cfg["vit"]],
                               msa_ckpt, IMAGE_SIZE,
                               multimask_output=cfg["mm"], device=device)
        net.eval()
    except Exception as e:
        print(f"  SKIP: Load failed: {e}")
        continue

    if ds == "Synapse":
        images = load_synapse_data(n=N_IMAGES)
    elif ds == "REFUGE":
        images = load_refuge_data(n=N_IMAGES)
    print(f"  Loaded {len(images)} images")

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:
            for method_key in ["pure_l1", "random"]:
                dices, clean_dices, times = [], [], []

                for img_data in images:
                    if ds == "Synapse":
                        x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                        gt = img_data[1]
                    elif ds == "REFUGE":
                        x = img_data[0].unsqueeze(0).float().to(device)
                        gt = img_data[1]

                    rg = x.max().item() - x.min().item()
                    eps_pix = eps_frac * rg
                    step_pix = eps_pix / max(n_steps, 1)

                    if ds == "Synapse":
                        clean_dice = evaluate_msa_synapse(net, x, gt)
                    elif ds == "REFUGE":
                        clean_dice = evaluate_msa_refuge(net, x, gt)
                    if clean_dice < 0.01:
                        continue

                    try:
                        t0 = time.time()
                        if method_key == "random":
                            x_adv = attack_random_direction(x, eps_pix)
                        else:
                            x_adv = attack_pure_l1_msa(net, x, eps_pix, step_pix, n_steps, _get_edge_map)

                        if ds == "Synapse":
                            adv_dice = evaluate_msa_synapse(net, x_adv, gt)
                        elif ds == "REFUGE":
                            adv_dice = evaluate_msa_refuge(net, x_adv, gt)
                        times.append(time.time() - t0)
                        dices.append(adv_dice)
                        clean_dices.append(clean_dice)
                    except Exception as e:
                        print(f"    Error [{model_key}]: {e}")
                        continue

                if len(dices) == 0:
                    print(f"  eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} {method_key}: NO VALID RESULTS")
                    continue

                mc = float(np.mean(clean_dices))
                ma = float(np.mean(dices))
                md = mc - ma
                mt = float(np.mean(times))

                all_results.append({
                    "model": model_key, "dataset": ds, "vit": cfg["vit"],
                    "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                    "method": method_key,
                    "mean_clean": mc, "mean_adv": ma, "mean_delta": md,
                    "mean_time": mt, "n_valid": len(dices),
                })

                method_label = "Pure L1" if method_key == "pure_l1" else "Random"
                print(f"  eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} {method_label:<20} "
                      f"Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f} ({mt:.1f}s)")

    del net
    torch.cuda.empty_cache()
    gc.collect()

    # Save incrementally
    with open(prior_path, "w") as f:
        json.dump(all_results, f, indent=2)

# ── Summary ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("MSA RESUME SUMMARY")
print(f"{'='*90}")
for eps_str in sorted(set(r["eps"] for r in all_results)):
    for n_steps in sorted(set(r["steps"] for r in all_results)):
        print(f"\n  eps={eps_str} steps={n_steps:2d}")
        for model_key in sorted(set(r["model"] for r in all_results)):
            pure = [r for r in all_results if r["model"]==model_key and r["eps"]==eps_str
                    and r["steps"]==n_steps and r["method"]=="pure_l1"]
            rand = [r for r in all_results if r["model"]==model_key and r["eps"]==eps_str
                    and r["steps"]==n_steps and r["method"]=="random"]
            pd = pure[0]["mean_delta"] if pure else 0
            rd = rand[0]["mean_delta"] if rand else 0
            print(f"  {model_key:<22} PureL1 Δ={pd:+.4f}  Random Δ={rd:+.4f}  margin={pd-rd:+.4f}")

print(f"\nSaved to {prior_path}")
print("Done — MSA resume complete")
