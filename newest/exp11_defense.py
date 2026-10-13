#!/usr/bin/env python3
"""
Experiment 11: Defense Robustness.
Tests SE-CGR attack against 5 input defenses on Synapse SAMed vit_b.

Defenses: JPEG compression (Q=75,50), Gaussian blur (sigma=1,2),
          Median filter (3x3,5x5), Bit-depth reduction (4bit), Resize+pad
"""
import os, sys, json, gc, random
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from io import BytesIO
from PIL import Image as PILImage
from scipy.ndimage import median_filter, gaussian_filter

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIT = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT = f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

N_CASES = 6
NUM_SLICES = 4

DEFENSES = {
    "none": lambda x: x,
    "jpeg_q75": lambda x: jpeg_compress(x, 75),
    "jpeg_q50": lambda x: jpeg_compress(x, 50),
    "gaussian_blur_s1": lambda x: gaussian_blur(x, 1.0),
    "gaussian_blur_s2": lambda x: gaussian_blur(x, 2.0),
    "median_3x3": lambda x: median_filter(np.atleast_2d(x.squeeze()), size=3),
    "median_5x5": lambda x: median_filter(np.atleast_2d(x.squeeze()), size=5),
    "bitdepth_4": lambda x: reduce_bitdepth(x, 4),
    "resize_pad": lambda x: resize_pad(x, 0.9),
}


def jpeg_compress(x, quality):
    """JPEG compression defense. Always assumes [H,W] grayscale input."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    # Ensure [H, W] grayscale
    if x.ndim > 2:
        x = x.squeeze()
    if x.ndim == 3:
        x = x.mean(axis=0)
    img = PILImage.fromarray(x.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    restored = np.array(PILImage.open(buf)).astype(np.float32)
    # JPEG may read back as [H,W] grayscale or [H,W,3] RGB
    if restored.ndim == 3 and restored.shape[-1] == 3:
        restored = restored.mean(axis=-1)
    return restored


def gaussian_blur(x, sigma):
    """Gaussian blur defense. Handles any input shape, returns [H,W]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.atleast_2d(x)
    if x.ndim > 2:
        x = x.squeeze()
    if x.ndim == 3:
        x = x.mean(axis=0)
    return gaussian_filter(x, sigma=sigma)


def reduce_bitdepth(x, bits):
    """Reduce color bit depth. Returns [H,W]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.atleast_2d(x)
    if x.ndim > 2:
        x = x.squeeze()
    if x.ndim == 3:
        x = x.mean(axis=0)
    levels = 2 ** bits
    return np.round(x / 255.0 * (levels - 1)) / (levels - 1) * 255.0


def resize_pad(x, scale):
    """Resize then pad back to original size. Returns [H,W]."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.atleast_2d(x)
    if x.ndim > 2:
        x = x.squeeze()
    if x.ndim == 3:
        x = x.mean(axis=0)
    x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    h, w = x_t.shape[-2:]
    new_h, new_w = int(h * scale), int(w * scale)
    resized = F.interpolate(x_t, size=(new_h, new_w), mode='bilinear')
    pad_h = (h - new_h) // 2
    pad_w = (w - new_w) // 2
    result = F.pad(resized, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
    if result.shape[-2:] != (h, w):
        result = F.interpolate(result, size=(h, w), mode='bilinear')
    return result.squeeze().numpy()


def evaluate_slice(net, image_np, label_np, case_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("root_utils", f"{ROOT}/utils.py")
    root_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_utils)
    test_single_volume = root_utils.test_single_volume
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]
    img = torch.from_numpy(image_np.copy()).unsqueeze(0).float()
    lbl = torch.from_numpy(label_np.copy()).unsqueeze(0).float()
    metric = test_single_volume(
        image=img, label=lbl, net=net, classes=8, multimask_output=True,
        patch_size=[IMAGE_SIZE, IMAGE_SIZE], test_save_path=None, case=case_name, z_spacing=1,
    )
    return float(np.mean(metric, axis=0)[0])


def main():
    print("Loading models...")
    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    target = load_samed_model(VIT, SAM_CKPT, CKPT, IMAGE_SIZE, 8, device=DEVICE)

    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:N_CASES]

    all_slices = []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:NUM_SLICES]
        for s in valid:
            all_slices.append((img[s].astype(np.float32), lbl[s], h5f, s))

    print(f"Processing {len(all_slices)} slices × {len(DEFENSES)} defenses...")

    results = {dname: [] for dname in DEFENSES}

    for slc, lbl_slice, h5f, si in all_slices:
        clean_d = evaluate_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
        slc_3ch = np.stack([slc]*3, axis=0)
        img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

        # Generate adversarial image once
        config = AttackConfig(
            att_step_size=2.0, att_bound=10.0, att_numstep=10,
            direction_mode="radial", probe_amplification=2.0,
            edge_guidance=True, seed=42,
        )
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        adv_np, _ = run_codd_attack(source_sam.image_encoder, img_t, config, return_trace=True)

        for dname, defense_fn in DEFENSES.items():
            # Apply defense to adversarial image — all defenses return [H,W] grayscale
            defended = defense_fn(adv_np[0].copy())
            if isinstance(defended, torch.Tensor):
                defended = defended.detach().cpu().numpy()
            defended = np.atleast_2d(defended)  # safety: ensure at least 2D

            adv_d = evaluate_slice(target, defended, lbl_slice, f"{h5f}_s{si}")
            delta = clean_d - adv_d
            results[dname].append(delta)

    summary = {}
    for dname in DEFENSES:
        deltas = results[dname]
        summary[dname] = {"mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)), "n": len(deltas)}
        print(f"  {dname}: ΔDice={summary[dname]['mean_delta']:.4f} ± {summary[dname]['std_delta']:.4f}")

    output = {"config": {"vit": VIT, "defenses": list(DEFENSES.keys())}, "results": summary}
    out_path = os.path.join(OUTPUT_DIR, "exp11_defense.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
