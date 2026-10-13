#!/usr/bin/env python3
"""Smoke test: verify all PAID optimization components on 2 representative models."""
import os, sys, time
import numpy as np
import torch

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_samed_model
from attack_codd import _get_edge_map
from attack_radial_optimized import (
    attack_radial_optimized, attack_bare_radial, attack_random_probe,
    get_image_encoder,
)

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
}
device = torch.device("cuda:0")

MODELS = [
    ("Synapse_SAMed_vit_b", "vit_b",
     "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8),
    ("ISIC_SAMed_vit_b", "vit_b",
     "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
]


def load_data(dataset_name, n=8):
    from PIL import Image
    if dataset_name == "Synapse":
        from exp_paid_optimal_attack import load_synapse_slices
        return load_synapse_slices(n)
    else:
        tdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
        mdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
        IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        jpgs = sorted([f for f in os.listdir(tdir) if f.endswith('.jpg')])[:n]
        imgs = []
        for jpg in jpgs:
            img = Image.open(os.path.join(tdir, jpg)).convert('RGB').resize((512, 512), Image.BILINEAR)
            img_np = np.array(img).astype(np.float32)
            base = jpg.replace('.jpg', '')
            mask = Image.open(os.path.join(mdir, base + '_Segmentation.png')).convert('L')
            mask = mask.resize((512, 512), Image.NEAREST)
            gt = (np.array(mask) > 128).astype(np.float32)
            if gt.sum() < 100:
                continue
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
            img_t = (img_t / 255.0 - IM_MEAN) / IM_STD
            imgs.append((img_t, gt))
        return imgs


def evaluate_model(net, img_t, gt, dataset_name):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out['masks']
        if dataset_name == "Synapse":
            pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
        else:
            probs = torch.softmax(m.float(), dim=1)
            pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    pred_fg = (pred == 1).astype(np.float32) if dataset_name == "ISIC" else pred
    if dataset_name == "Synapse":
        # For multi-class Synapse, compute per-class dice averaged
        n_classes = 8
        dices = []
        for c in range(1, n_classes):
            pred_c = (pred == c).astype(np.float32)
            gt_c = (gt == c).astype(np.float32)
            if gt_c.sum() == 0:
                continue
            inter = (pred_c * gt_c).sum()
            dices.append(float(2.0 * inter / (pred_c.sum() + gt_c.sum() + 1e-8)))
        return float(np.mean(dices)) if dices else 0.0

    intersection = (pred_fg * gt).sum()
    return float(2.0 * intersection / (pred_fg.sum() + gt.sum() + 1e-8))


print("=" * 70)
print("SMOKE TEST: All PAID Optimization Components")
print("=" * 70)

for model_name, vit, ckpt_rel, num_classes in MODELS:
    dataset_name = model_name.split("_")[0]
    print(f"\n{'─'*60}")
    print(f"  {model_name}")
    print(f"{'─'*60}")

    net = load_samed_model(vit, SAM_CKPTS[vit],
                           os.path.join(CKPT_BASE, ckpt_rel),
                           IMAGE_SIZE, num_classes, device=device)
    net.eval()
    image_encoder = net.sam.image_encoder
    image_encoder.eval()
    images = load_data(dataset_name, n=4)

    eps = 8.0 / 255.0
    step = 0.8 / 255.0

    for img_idx, (img_data, lbl) in enumerate(images):
        if dataset_name == "Synapse":
            x = torch.from_numpy(img_data).unsqueeze(0).float().to(device)
            gt = lbl
        else:
            x = img_data.to(device)
            gt = lbl
        rg = x.max().item() - x.min().item()

        clean_dice = evaluate_model(net, x, gt, dataset_name)
        if clean_dice < 0.01:
            continue

        results = {}

        # 0. Random Probe (baseline)
        t0 = time.time()
        x_adv = attack_random_probe(
            image_encoder, x, eps * rg, step * rg, 20,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Random Probe"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 1. Bare Radial (α=0, no amp loss, no prior)
        t0 = time.time()
        x_adv = attack_bare_radial(
            image_encoder, x, eps * rg, step * rg, 20,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Bare Radial"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 2. AE-RP magnitude (α=0.5)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.5, weight_mode="magnitude",
            lambda_amp=0.0, lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["AE-RP mag α=0.5"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 3. AE-RP edge-weighted (α=0.5)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.5, weight_mode="edge",
            lambda_amp=0.0, lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["AE-RP edge α=0.5"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 4. Dual spectral amp loss (λ_amp=0.05)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.0, lambda_amp=0.05, amp_loss_mode="spectral",
            lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Dual spec λ=0.05"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 5. Dual spatial amp loss (λ_amp=0.05)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.0, lambda_amp=0.05, amp_loss_mode="spatial",
            lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Dual spat λ=0.05"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 6. Dual both (spectral+spatial) amp loss
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.0, lambda_amp=0.05, amp_loss_mode="both",
            lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Dual both λ=0.05"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 7. Phase loss (λ_phase=0.01)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.0, lambda_amp=0.0, lambda_phase=0.01,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Phase λ=0.01"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 8. Φ-probe (orthogonal, φ=0.3)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.0, lambda_amp=0.0, lambda_phase=0.0,
            include_phi_probe=True, phi_weight=0.3,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Φ-probe φ=0.3"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 9. AE-RP edge + Both amp loss (best combo candidates)
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.5, weight_mode="edge", lambda_amp=0.05,
            amp_loss_mode="both", lambda_phase=0.0,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["EdgeAE + DualBoth"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        # 10. Full combo: Edge AE-RP + Dual both + Φ-probe
        t0 = time.time()
        x_adv = attack_radial_optimized(
            image_encoder, x, eps * rg, step * rg, 20,
            amplitude_alpha=0.5, weight_mode="edge", lambda_amp=0.05,
            amp_loss_mode="both", lambda_phase=0.005,
            include_phi_probe=True, phi_weight=0.2,
            prior_weight=0.0, radial_weight=1.0, use_random_noise=False,
            edge_map_fn=_get_edge_map,
        )
        adv_dice = evaluate_model(net, x_adv, gt, dataset_name)
        results["Full Combo"] = (adv_dice, clean_dice - adv_dice, time.time() - t0)

        print(f"  [{img_idx}] Clean={clean_dice:.4f}")
        best_delta = max(v[1] for v in results.values())
        for name, (dice, delta, dt) in sorted(results.items(), key=lambda x: -x[1][1]):
            best = "← BEST" if delta == best_delta else ""
            print(f"    {name:<25} Dice={dice:.4f}  Δ={delta:+.4f}  {dt:.1f}s  {best}")

        break  # Just one image per model for smoke test

    del net
    torch.cuda.empty_cache()

print("\nDone — smoke test passed")
