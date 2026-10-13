#!/usr/bin/env python3
"""
Pure L1 Feature-Space Attack — MSA Model Extension.

Validates Pure L1 optimality on MedicalSAMAdapter models.
Compares: Pure L1 (feature-space) vs pixel-space PGD vs Random direction.

MSA encoder returns a single tensor (no intermediate layers), so
feature-space attack uses only the final image embeddings.
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

from evaluate import load_medsa_model, load_samed_model
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
    "MSA_ISIC_vit_b": {"vit": "vit_b", "dir": "isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24", "ds": "ISIC", "n_class": 1, "mm": 9},
    "MSA_ISIC_vit_l": {"vit": "vit_l", "dir": "isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24", "ds": "ISIC", "n_class": 1, "mm": 9},
    "MSA_ISIC_vit_h": {"vit": "vit_h", "dir": "isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24", "ds": "ISIC", "n_class": 1, "mm": 9},
}


def load_isic_data(n=12):
    tdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
    mdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
    jpgs = sorted([f for f in os.listdir(tdir) if f.endswith('.jpg')])[:n]
    imgs = []
    for jpg in jpgs:
        img = Image.open(os.path.join(tdir, jpg)).convert('RGB').resize((512, 512), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        base = jpg.replace('.jpg', '')
        mask = Image.open(os.path.join(mdir, base + '_Segmentation.png')).convert('L')
        mask = mask.resize((512, 512), Image.NEAREST)
        gt = (np.array(mask) > 128).astype(np.float32)
        if gt.sum() < 100: continue
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        img_t = (img_t / 255.0 - IM_MEAN_CPU) / IM_STD_CPU
        imgs.append((img_t, gt))
    return imgs


def evaluate_msa_isic(model, img_t, gt):
    """Evaluate MSA model on ISIC with deterministic center-of-mass prompt."""
    import random as _random
    _random.seed(42)
    msk = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0).to(device)
    _, _, h, w = msk.shape

    # Use center of mass of lesion as deterministic prompt
    indices = torch.nonzero(msk[0, 0])
    if indices.size(0) == 0:
        pt = torch.tensor([[0.0, 0.0]], device=device)
        label_pt = torch.tensor([[0]], dtype=torch.int, device=device)
    else:
        com = indices.float().mean(dim=0)  # center of mass
        pt = com.flip(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 2] (xy format)
        label_pt = torch.tensor([[1]], dtype=torch.int, device=device)

    img = img_t.to(dtype=torch.float32, device=device)
    prompt = (pt.float(), label_pt)

    with torch.no_grad():
        imgs_pre = model.preprocess(img)
        image_embeddings = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=prompt, boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)

    probs = torch.sigmoid(pred).detach().cpu().numpy()
    best = 0.0
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred_bin = (probs[0, 0] > thresh).astype(np.float32)
        inter = (pred_bin * gt).sum()
        dice = float(2.0 * inter / (pred_bin.sum() + gt.sum() + 1e-8))
        if dice > best: best = dice
    return best


def attack_pure_l1_msa(model, x, eps_pix, step_pix, steps, edge_map_fn=None):
    """Pure L1 feature-space PGD for MSA models (no intermediate features)."""
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
    """Random direction baseline."""
    noise = torch.randn_like(x)
    noise = noise / (noise.abs().max() + 1e-8) * eps_pix
    max_val, min_val = x.max(), x.min()
    return torch.clamp(x + noise, min_val, max_val)


# ── Main ─────────────────────────────────────────────────────────────────────

EPS_LIST = [4.0/255, 8.0/255]
STEPS_LIST = [10, 20, 40]
N_IMAGES = 12

METHODS = {
    "pure_l1": ("Pure L1", attack_pure_l1_msa),
    "random": ("Random Direction", None),  # special handler
}

all_results = []

print("=" * 90)
print("MSA PURE L1 VALIDATION")
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

    images = load_isic_data(n=N_IMAGES)
    print(f"  Loaded {len(images)} images")

    for eps_frac in EPS_LIST:
        for n_steps in STEPS_LIST:
            for method_key, (method_label, attack_fn) in METHODS.items():
                dices, clean_dices, times = [], [], []

                for img_t, gt in images:
                    x = img_t.to(device)
                    rg = x.max().item() - x.min().item()
                    eps_pix = eps_frac * rg
                    step_pix = eps_pix / max(n_steps, 1)

                    clean_dice = evaluate_msa_isic(net, x, gt)
                    if clean_dice < 0.01:
                        continue

                    try:
                        t0 = time.time()
                        if method_key == "random":
                            x_adv = attack_random_direction(x, eps_pix)
                        else:
                            x_adv = attack_fn(net, x, eps_pix, step_pix, n_steps, _get_edge_map)
                        adv_dice = evaluate_msa_isic(net, x_adv, gt)
                        times.append(time.time() - t0)
                        dices.append(adv_dice)
                        clean_dices.append(clean_dice)
                    except Exception as e:
                        print(f"    Error: {e}")
                        continue

                if len(dices) == 0:
                    continue

                mc = float(np.mean(clean_dices))
                ma = float(np.mean(dices))
                md = mc - ma
                mt = float(np.mean(times))

                all_results.append({
                    "model": model_key, "dataset": ds, "vit": cfg["vit"],
                    "eps": f"{eps_frac*255:.0f}/255", "steps": n_steps,
                    "method": method_key, "method_label": method_label,
                    "mean_clean": mc, "mean_adv": ma, "mean_delta": md,
                    "mean_time": mt, "n_valid": len(dices),
                })

                print(f"  eps={eps_frac*255:4.0f}/255 steps={n_steps:2d} {method_label:<20} "
                      f"Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f} ({mt:.1f}s)")

    del net
    torch.cuda.empty_cache()
    gc.collect()

# Save
out_path = os.path.join(OUTPUT_DIR, "msa_pure_l1_validation.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done — MSA Pure L1 validation complete")
