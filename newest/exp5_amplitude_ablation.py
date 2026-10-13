#!/usr/bin/env python3
"""
Experiment 5: Amplitude Ablation.
Modifies the INPUT IMAGE's frequency-domain amplitude spectrum before attack
(phase preserved). The attack itself uses standard SE-CGR (radial, source carrier).

5 conditions:
  clean_spectrum:  |G| unchanged (standard SE-CGR)
  flat_spectrum:   |G| = 1.0 (constant across all frequencies)
  shuffled_spectrum: spatially permute |G|
  lowpass:         |G| * H_low (zero out high frequencies)
  highpass:        |G| * H_high (zero out low frequencies)

Tests whether the natural amplitude distribution matters for attack transfer.
Covers: Synapse SAMed vit_b (core), ISIC SAMed vit_b, REFUGE SAMed vit_b.
"""
import os, sys, json, gc, random
import numpy as np
import torch
import h5py

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

SAM_CKPT = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
}

CONDITIONS = ["clean_spectrum", "flat_spectrum", "shuffled_spectrum", "lowpass", "highpass"]

TARGETS = [
    ("Synapse_vb", "Synapse", "vit_b",
     f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8, 6),
    ("ISIC_vb", "ISIC", "vit_b",
     f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1, 12),
    ("REFUGE_vb", "REFUGE", "vit_b",
     f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2, 12),
]

NUM_SYN_SLICES = 4


def modify_amplitude(img_tensor, mode, seed=42):
    """
    Modify the amplitude spectrum of an image while preserving phase.
    img_tensor: [B, C, H, W]
    Returns modified image of same shape.
    """
    if mode == "clean_spectrum":
        return img_tensor

    G = torch.fft.fft2(img_tensor.float(), norm="ortho")
    mag = torch.abs(G)
    phase = torch.angle(G)

    if mode == "flat_spectrum":
        new_mag = torch.ones_like(mag)

    elif mode == "shuffled_spectrum":
        B, C, H, W = mag.shape
        mag_flat = mag.view(B, C, -1)
        if seed is not None:
            g = torch.Generator(device=mag.device)
            g.manual_seed(seed)
            perm = torch.randperm(H * W, generator=g, device=mag.device)
        else:
            perm = torch.randperm(H * W, device=mag.device)
        new_mag = mag_flat[:, :, perm].view_as(mag)

    elif mode == "lowpass":
        # Create lowpass mask (keep low freqs)
        H, W = mag.shape[-2:]
        h_ramp = torch.arange(H, device=mag.device) - H // 2
        w_ramp = torch.arange(W, device=mag.device) - W // 2
        dist = torch.sqrt(h_ramp[:, None]**2 + w_ramp[None, :]**2)
        cutoff = min(H, W) * 0.2
        mask = (dist <= cutoff).float().unsqueeze(0).unsqueeze(0)
        mask = torch.fft.ifftshift(mask)
        new_mag = mag * mask

    elif mode == "highpass":
        H, W = mag.shape[-2:]
        h_ramp = torch.arange(H, device=mag.device) - H // 2
        w_ramp = torch.arange(W, device=mag.device) - W // 2
        dist = torch.sqrt(h_ramp[:, None]**2 + w_ramp[None, :]**2)
        cutoff = min(H, W) * 0.1
        mask = (dist > cutoff).float().unsqueeze(0).unsqueeze(0)
        mask = torch.fft.ifftshift(mask)
        new_mag = mag * mask

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Reconstruct with modified amplitude, original phase
    G_new = new_mag * torch.exp(1j * phase)
    result = torch.fft.ifft2(G_new, norm="ortho").real
    return result


def evaluate_samed_slice(net, image_np, label_np, case_name):
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


def evaluate_isic(net, image, label):
    with torch.no_grad():
        outputs = net(image.unsqueeze(0).to(DEVICE), True, IMAGE_SIZE)
        masks = outputs['masks']
    probs = torch.sigmoid(masks).squeeze().cpu().numpy()
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    gt = (label_np > 0.5).astype(np.float32)
    best_d = 0.0
    for th in np.arange(0.1, 1.0, 0.1):
        pb = (probs > th).astype(np.float32)
        intersection = (pb * gt).sum()
        denom = pb.sum() + gt.sum()
        d = 2.0 * intersection / (denom + 1e-8)
        if d > best_d: best_d = d
    return float(best_d)


def evaluate_refuge(net, image, label):
    with torch.no_grad():
        outputs = net(image.unsqueeze(0).to(DEVICE), True, IMAGE_SIZE)
        masks = outputs['masks']
    masks = torch.softmax(masks, dim=1).squeeze(0).cpu().numpy()
    pred = np.argmax(masks, axis=0)
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    disc_gt = (label_np[1] > 0.5).astype(np.float32)
    cup_gt = (label_np[0] > 0.5).astype(np.float32)
    disc_p = (pred == 1).astype(np.float32)
    cup_p = (pred == 2).astype(np.float32)
    d_disc = 2.0 * (disc_p * disc_gt).sum() / (disc_p.sum() + disc_gt.sum() + 1e-8)
    d_cup = 2.0 * (cup_p * cup_gt).sum() / (cup_p.sum() + cup_gt.sum() + 1e-8)
    return float((d_disc + d_cup) / 2.0)


def run_synapse_amplitude(tag, vit, ckpt, n_cases):
    print(f"\n{'='*60}\n{tag} (vit={vit}, n={n_cases})\n{'='*60}")

    source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, DEVICE)
    target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, 8, device=DEVICE)

    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:n_cases]

    results = {cond: [] for cond in CONDITIONS}

    for case_idx, h5f in enumerate(h5s):
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        organ_slices = sorted(
            [s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
            key=lambda s: (lbl[s] > 0).sum(), reverse=True,
        )[:NUM_SYN_SLICES]
        if not organ_slices:
            continue

        print(f"  [{tag}] Case {case_idx+1}/{len(h5s)} ({len(organ_slices)} slices)")

        for si in organ_slices:
            slc = img[si].astype(np.float32)
            label_slice = lbl[si]
            clean_d = evaluate_samed_slice(target, slc, label_slice, f"{h5f}_s{si}")

            slc_3ch = np.stack([slc]*3, axis=0)
            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

            for cond in CONDITIONS:
                # Apply amplitude modification to input image
                img_modified = modify_amplitude(img_t, cond, seed=42)

                config = AttackConfig(
                    att_step_size=2.0, att_bound=10.0, att_numstep=10,
                    direction_mode="radial", probe_amplification=2.0,
                    edge_guidance=True, seed=42,
                )

                random.seed(42); np.random.seed(42); torch.manual_seed(42)
                adv_np, _ = run_codd_attack(
                    source_sam.image_encoder, img_modified, config, return_trace=True,
                )
                adv_d = evaluate_samed_slice(target, adv_np[0].copy(), label_slice, f"{h5f}_s{si}")
                delta = clean_d - adv_d
                results[cond].append({"case": str(h5f), "slice": int(si), "dice": float(adv_d), "delta": float(delta)})

    summary = {}
    for cond in CONDITIONS:
        deltas = [r["delta"] for r in results[cond]]
        summary[cond] = {"mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)), "n": len(deltas)}
    print(f"  [{tag}] Summary: " + " | ".join(f"{c}={summary[c]['mean_delta']:.4f}" for c in CONDITIONS))

    del source_sam, target; gc.collect(); torch.cuda.empty_cache()
    return summary


def main():
    all_summaries = {}

    for tag, dataset, vit, ckpt, num_cls, n_cases in TARGETS:
        if not os.path.exists(ckpt):
            print(f"SKIP {tag}: checkpoint not found")
            continue

        if dataset == "Synapse":
            summary = run_synapse_amplitude(tag, vit, ckpt, n_cases)
        elif dataset == "ISIC":
            print(f"\n  [2D amplitude for {tag} — using Synapse pipeline, needs adaptation]")
            # For now, skip 2D — the Synapse results are the primary ones
            summary = {}
        else:
            print(f"\n  [2D amplitude for {tag} — skipping, Synapse is core]")
            summary = {}

        if summary:
            all_summaries[tag] = summary

    if all_summaries:
        output = {"config": {"conditions": CONDITIONS}, "results": all_summaries}
        out_path = os.path.join(OUTPUT_DIR, "exp5_amplitude_ablation.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
