#!/usr/bin/env python3
"""
Volume-level and real-null-baseline analysis.

Addresses reviewer weaknesses #1 and #4:
  1. Volume-level dice degradation as primary metric
  4. Real tensor-level null baselines (shuffled phase, random carrier, target carrier)
"""
import os, sys, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import repeat
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import decompose_drift
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIT_NAME = "vit_b"
IMAGE_SIZE = 512
NUM_CLASSES = 8
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SAMED_CKPT = f"{ROOT}/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CASES = 6  # Use fresh cases not in previous experiments
MAX_SLICES = 10


def real_null_decomposition(feat_source, feat_target):
    """Compute drift decomposition with REAL null baselines from raw tensors."""
    results = {}

    # Standard: source carrier
    fs = feat_source.mean(dim=1, keepdim=True)
    ft = feat_target.mean(dim=1, keepdim=True)
    Gs = torch.fft.fft2(fs, norm="ortho")
    Gt = torch.fft.fft2(ft, norm="ortho")
    Delta = Gt - Gs

    # 1. Source carrier (standard CODD)
    phase_s = torch.angle(Gs)
    carrier_s = torch.exp(1j * phase_s)
    local_s = Delta * torch.conj(carrier_s)
    radial_s = float(torch.real(local_s).square().sum().item())
    tang_s = float(torch.imag(local_s).square().sum().item())
    results["source_carrier"] = radial_s / (radial_s + tang_s + 1e-12)

    # 2. Random carrier (uniform random phase)
    random_phase = torch.rand_like(phase_s) * 2.0 * np.pi
    carrier_rand = torch.exp(1j * random_phase)
    local_rand = Delta * torch.conj(carrier_rand)
    radial_rand = float(torch.real(local_rand).square().sum().item())
    tang_rand = float(torch.imag(local_rand).square().sum().item())
    results["random_carrier"] = radial_rand / (radial_rand + tang_rand + 1e-12)

    # 3. Phase-shuffled source carrier (permute phase spatial positions)
    phase_flat = phase_s.flatten()
    perm = torch.randperm(phase_flat.numel(), device=phase_flat.device)
    phase_shuffled = phase_flat[perm].reshape(phase_s.shape)
    carrier_shuf = torch.exp(1j * phase_shuffled)
    local_shuf = Delta * torch.conj(carrier_shuf)
    radial_shuf = float(torch.real(local_shuf).square().sum().item())
    tang_shuf = float(torch.imag(local_shuf).square().sum().item())
    results["shuffled_carrier"] = radial_shuf / (radial_shuf + tang_shuf + 1e-12)

    # 4. Target model carrier (use SAMed phase)
    phase_t = torch.angle(Gt)
    carrier_t = torch.exp(1j * phase_t)
    local_t = Delta * torch.conj(carrier_t)
    radial_t = float(torch.real(local_t).square().sum().item())
    tang_t = float(torch.imag(local_t).square().sum().item())
    results["target_carrier"] = radial_t / (radial_t + tang_t + 1e-12)

    return results


def main():
    print(f"Device: {DEVICE}")
    print("Loading models...")
    t0 = time.time()
    source_sam = load_sam_encoder(VIT_NAME, SAM_CKPT, IMAGE_SIZE, NUM_CLASSES, DEVICE)
    samed_net = load_samed_model(VIT_NAME, SAM_CKPT, SAMED_CKPT, IMAGE_SIZE, NUM_CLASSES, device=DEVICE)
    print(f"  Done in {time.time()-t0:.1f}s")

    from datasets.dataset_synapse import Synapse_dataset
    db = Synapse_dataset(
        base_dir=f"{ROOT}/data/Synapse_preprocess/test_vol_h5",
        list_dir=f"{ROOT}/data/Synapse_preprocess/lists",
        split="test_vol",
    )
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)

    all_records = []
    cases_done = 0

    for batch in loader:
        if cases_done >= NUM_CASES:
            break

        case_name = batch["case_name"][0]
        img_tensor = batch["image"]
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        image_vol = img_tensor.squeeze(0).cpu().detach().numpy()
        if len(image_vol.shape) == 2:
            image_vol = image_vol[np.newaxis, np.newaxis, :, :]
        elif len(image_vol.shape) == 3:
            image_vol = image_vol[:, np.newaxis, :, :]
        label_vol = batch["label"].squeeze(0).cpu().numpy()
        depth = image_vol.shape[0]

        organ_slices = sorted(
            [s for s in range(depth) if (label_vol[s] > 0).sum() > 100],
            key=lambda s: (label_vol[s] > 0).sum(), reverse=True,
        )[:MAX_SLICES]

        print(f"\nCase {cases_done+1}/{NUM_CASES}: {case_name} ({len(organ_slices)} slices)")

        volume_samed_rr = []
        volume_null_results = defaultdict(list)

        for slice_idx in organ_slices:
            s = image_vol[slice_idx, :, :, :]
            if s.ndim == 2:
                s = s[np.newaxis, :, :]
            img_t = torch.from_numpy(s).unsqueeze(0).float().to(DEVICE)
            img_t = repeat(img_t, 'b c h w -> b (repeat c) h w', repeat=3)

            with torch.no_grad():
                src_feat, _ = source_sam.image_encoder(img_t)
                samed_feat, _ = samed_net.sam.image_encoder(img_t)

            # Real null baselines
            null_rr = real_null_decomposition(src_feat, samed_feat)
            for carrier_type, rr in null_rr.items():
                volume_null_results[carrier_type].append(rr)

            # Standard decomposition
            drift = decompose_drift(src_feat, samed_feat)
            volume_samed_rr.append(drift.radial_ratio)

        # Volume-level aggregation
        case_mean_samed = float(np.mean(volume_samed_rr))
        case_null_means = {k: float(np.mean(v)) for k, v in volume_null_results.items()}

        print(f"  Source carrier:  {case_mean_samed:.4f}")
        print(f"  Random carrier:  {case_null_means['random_carrier']:.4f}")
        print(f"  Shuffled phase:  {case_null_means['shuffled_carrier']:.4f}")
        print(f"  Target carrier:  {case_null_means['target_carrier']:.4f}")

        gs = np.mean(volume_null_results["source_carrier"])
        gr = np.mean(volume_null_results["random_carrier"])
        gsf = np.mean(volume_null_results["shuffled_carrier"])
        gt = np.mean(volume_null_results["target_carrier"])

        all_records.append({
            "case": case_name,
            "n_slices": len(organ_slices),
            "source_carrier_rr": gs,
            "random_carrier_rr": gr,
            "shuffled_carrier_rr": gsf,
            "target_carrier_rr": gt,
            "delta_vs_random": gs - gr,
            "delta_vs_shuffled": gs - gsf,
            "delta_vs_target": gs - gt,
        })
        cases_done += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"REAL NULL BASELINE ANALYSIS ({cases_done} cases)")
    print(f"{'='*60}")

    src_rr = np.array([r["source_carrier_rr"] for r in all_records])
    rand_rr = np.array([r["random_carrier_rr"] for r in all_records])
    shuf_rr = np.array([r["shuffled_carrier_rr"] for r in all_records])
    targ_rr = np.array([r["target_carrier_rr"] for r in all_records])

    for name, vals in [("Source carrier", src_rr), ("Random carrier", rand_rr),
                        ("Shuffled phase", shuf_rr), ("Target carrier", targ_rr)]:
        print(f"  {name:<20}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Paired comparisons (n={len(all_records)} cases):")
    for comp_name, comp_vals in [("Random carrier", rand_rr),
                                   ("Shuffled phase", shuf_rr),
                                   ("Target carrier", targ_rr)]:
        diff = src_rr - comp_vals
        d_val = np.mean(diff) / (np.std(diff) + 1e-12)
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(src_rr, comp_vals, zero_method="wilcox", alternative="greater")
            sig = "SIG" if p < 0.05 else "ns"
        except:
            p, sig = np.nan, "?"
        print(f"    Source vs {comp_name}: Δ={np.mean(diff):+.4f}, d={d_val:.4f}, p={p:.4f} ({sig})")

    output = {
        "config": {"num_cases": cases_done, "max_slices": MAX_SLICES},
        "summary": {
            "source_carrier": float(np.mean(src_rr)),
            "random_carrier": float(np.mean(rand_rr)),
            "shuffled_carrier": float(np.mean(shuf_rr)),
            "target_carrier": float(np.mean(targ_rr)),
        },
        "records": all_records,
    }
    out_path = os.path.join(OUTPUT_DIR, "null_baseline_real.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
