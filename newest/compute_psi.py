#!/usr/bin/env python3
"""
PAID (Phase-Amplitude Information Decomposition) Theory — Complete Implementation.

Computes PSI (Phase Sensitivity Index) for all 18 models and validates against
Exp14 phase-shuffled attack results.

Theory: A model's vulnerability to phase disruption is proportional to how much
task-relevant information is encoded in FFT phase vs amplitude.

PSI can be computed WITHOUT running any attack — pure model property.
"""
import os, sys, json, gc, random
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model, load_medsa_model
from utils import calculate_metric_percase

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 512

# ============================================================
# GPU CONFIG — use multiple GPUs
# ============================================================
GPU_MAP = {
    "Synapse_SAMed_vit_b": 0,
    "Synapse_SAMed_vit_l": 0,
    "Synapse_SAMed_vit_h": 0,
    "Synapse_MSA_vit_b": 1,
    "Synapse_MSA_vit_l": 1,
    "Synapse_MSA_vit_h": 1,
    "ISIC_SAMed_vit_b": 2,
    "ISIC_SAMed_vit_l": 2,
    "ISIC_SAMed_vit_h": 2,
    "ISIC_MSA_vit_b": 2,
    "ISIC_MSA_vit_l": 2,
    "ISIC_MSA_vit_h": 2,
    "REFUGE_SAMed_vit_b": 3,
    "REFUGE_SAMed_vit_l": 3,
    "REFUGE_SAMed_vit_h": 3,
    "REFUGE_MSA_vit_b": 3,
    "REFUGE_MSA_vit_l": 3,
    "REFUGE_MSA_vit_h": 3,
}

CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

# Exact checkpoint paths from exp14_full_matrix.py
SAMED_CFG = {
    "Synapse": {
        "vit_b": (f"{CKPT_BASE}/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8),
        "vit_l": (f"{CKPT_BASE}/SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 8),
        "vit_h": (f"{CKPT_BASE}/SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", 8),
    },
    "ISIC": {
        "vit_b": (f"{CKPT_BASE}/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1),
        "vit_l": (f"{CKPT_BASE}/SAMed/isic_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 1),
        "vit_h": (f"{CKPT_BASE}/SAMed/isic_512_lrexp3.0_pretrain_vit_h_epo300_bs4_lr0.005/epoch_159.pth", 1),
    },
    "REFUGE": {
        "vit_b": (f"{CKPT_BASE}/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2),
        "vit_l": (f"{CKPT_BASE}/SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_159.pth", 2),
        "vit_h": (f"{CKPT_BASE}/SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_159.pth", 2),
    },
}

MSA_CFG = {
    "Synapse": {
        "vit_b": (f"{CKPT_BASE}/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 9),
        "vit_l": (f"{CKPT_BASE}/MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 9),
        "vit_h": (f"{CKPT_BASE}/MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/checkpoint_best.pth", 9),
    },
    "ISIC": {
        "vit_b": (f"{CKPT_BASE}/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 1),
        "vit_l": (f"{CKPT_BASE}/MedicalSAMAdapter/isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24/Model/checkpoint_best.pth", 1),
        "vit_h": (f"{CKPT_BASE}/MedicalSAMAdapter/isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24/Model/checkpoint_best.pth", 1),
    },
    "REFUGE": {
        "vit_b": (f"{CKPT_BASE}/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_l": (f"{CKPT_BASE}/MedicalSAMAdapter/REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 2),
        "vit_h": (f"{CKPT_BASE}/MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/checkpoint_best.pth", 2),
    },
}

MODEL_CONFIGS = [
    (arch, ds, vit)
    for arch, cfg in [("SAMed", SAMED_CFG), ("MSA", MSA_CFG)]
    for ds in ["Synapse", "ISIC", "REFUGE"]
    for vit in ["vit_b", "vit_l", "vit_h"]
    if os.path.exists(cfg[ds][vit][0])
]

def get_ckpt_path(arch, ds, vit):
    """Get checkpoint path and num_classes for a given model config."""
    cfg = SAMED_CFG if arch == "SAMed" else MSA_CFG
    return cfg[ds][vit]


# ============================================================
# PSI COMPUTATION
# ============================================================
def compute_psi_from_features(features, n_perturbations=20, perturb_std=0.1):
    """
    Compute Phase Sensitivity Index from a feature tensor.

    features: [C, H, W] tensor (single image features, e.g., from ViT reshape)

    Returns:
        psi: Phase Sensitivity Index ∈ [0, 1]
        phase_sensitivity: ||Δf||² for phase perturbation
        ampl_sensitivity: ||Δf||² for amplitude perturbation
    """
    C, H, W = features.shape
    F_freq = torch.fft.fft2(features, norm='ortho')  # [C, H, W] complex
    A = torch.abs(F_freq)  # [C, H, W]
    Phi = torch.angle(F_freq)  # [C, H, W]

    phase_deltas = []
    ampl_deltas = []

    for _ in range(n_perturbations):
        # Phase perturbation (keep amplitude fixed)
        delta_phi = torch.randn_like(Phi) * perturb_std
        F_phase_perturbed = A * torch.exp(1j * (Phi + delta_phi))
        f_phase_perturbed = torch.fft.ifft2(F_phase_perturbed, norm='ortho').real
        phase_delta = torch.norm(f_phase_perturbed - features, p=2).item()
        phase_deltas.append(phase_delta)

        # Amplitude perturbation (keep phase fixed)
        delta_A = torch.randn_like(A) * perturb_std * A.std()
        F_ampl_perturbed = (A + delta_A) * torch.exp(1j * Phi)
        f_ampl_perturbed = torch.fft.ifft2(F_ampl_perturbed, norm='ortho').real
        ampl_delta = torch.norm(f_ampl_perturbed - features, p=2).item()
        ampl_deltas.append(ampl_delta)

    mean_phase = np.mean(phase_deltas)
    mean_ampl = np.mean(ampl_deltas)

    psi = mean_phase / (mean_phase + mean_ampl + 1e-12)

    return psi, mean_phase, mean_ampl


def compute_psi_multi_layer(encoder, img_tensor, n_layers=4, n_perturbations=10):
    """
    Compute PSI across multiple ViT layers.

    Returns per-layer PSI and mean PSI.
    """
    with torch.no_grad():
        # Get intermediate features using hooks
        features_dict = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                features_dict[name] = output.detach()
            return hook

        # Register hooks on specific layers
        blocks = encoder.blocks
        total_layers = len(blocks)
        layer_indices = np.linspace(total_layers // 4, total_layers - 1, n_layers, dtype=int)

        for idx in layer_indices:
            hook = blocks[idx].register_forward_hook(make_hook(f"layer_{idx}"))
            hooks.append(hook)

        # Forward pass
        _ = encoder(img_tensor)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute PSI per layer
        layer_psi = {}
        for layer_name, feat in features_dict.items():
            # feat shape: [1, N, C] — ViT token format
            # Reshape to spatial for FFT: take CLS token out, reshape to [C, H, W]
            if feat.dim() == 3:  # [B, N, C]
                tokens = feat[0, 1:, :]  # Remove CLS token
                N_tokens = tokens.shape[0]
                H = W = int(np.sqrt(N_tokens))
                if H * W == N_tokens:
                    tokens = tokens.permute(1, 0).reshape(-1, H, W)  # [C, H, W]
                    psi, ps, al = compute_psi_from_features(tokens, n_perturbations)
                    layer_psi[layer_name] = {"psi": psi, "phase_sens": ps, "ampl_sens": al}
                else:
                    # Non-square: compute in token space
                    tokens_2d = tokens.unsqueeze(0)  # [1, N, C]
                    tokens_2d = tokens_2d.permute(0, 2, 1)  # [1, C, N]
                    # Pad to square
                    H2 = int(np.ceil(np.sqrt(N_tokens)))
                    pad = H2 * H2 - N_tokens
                    if pad > 0:
                        tokens_2d = F.pad(tokens_2d, (0, pad))
                    tokens_2d = tokens_2d.reshape(-1, H2, H2)
                    psi, ps, al = compute_psi_from_features(tokens_2d, n_perturbations)
                    layer_psi[layer_name] = {"psi": psi, "phase_sens": ps, "ampl_sens": al}

        if layer_psi:
            mean_psi = np.mean([v["psi"] for v in layer_psi.values()])
        else:
            mean_psi = 0.5

        return mean_psi, layer_psi


# ============================================================
# DATA LOADING
# ============================================================
def get_synapse_test_images(n_images=8):
    """Load test images from Synapse dataset."""
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])
    images = []
    for h5f in h5s:
        if len(images) >= n_images:
            break
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img = f['image'][:]
            lbl = f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 500],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:2]
        for s in valid:
            if len(images) >= n_images:
                break
            slc = img[s].astype(np.float32)
            slc_3ch = np.stack([slc]*3, axis=0)
            images.append(torch.from_numpy(slc_3ch))
    return images


def get_isic_test_images(n_images=8):
    """Load test images from ISIC dataset."""
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for _, row in df.iterrows():
        if len(images) >= n_images:
            break
        img_path = os.path.join(data_dir, str(row['img']))
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img) * 255)
    return images


def get_refuge_test_images(n_images=8):
    """Load test images from REFUGE dataset."""
    from PIL import Image
    from torchvision import transforms
    test_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    if not os.path.isdir(test_dir):
        return []
    folders = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])[:n_images]
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for folder in folders:
        img_path = os.path.join(test_dir, folder, f"{folder}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img) * 255)
    return images


# ============================================================
# MODEL LOADING
# ============================================================
def load_model(arch, ds, vit, device):
    """Load a single model and return (model, source_sam)."""
    sam_ckpt = SAM_CKPTS[vit]
    ckpt_path, ncls_or_mmo = get_ckpt_path(arch, ds, vit)

    if arch == "SAMed":
        net = load_samed_model(vit, sam_ckpt, ckpt_path, IMAGE_SIZE, ncls_or_mmo, device=device)
    else:
        net = load_medsa_model(vit, sam_ckpt, ckpt_path, IMAGE_SIZE, ncls_or_mmo, device=device)

    # Load source SAM for feature extraction (use num_classes=8 for encoder)
    num_classes_src = 8 if ds == "Synapse" else (2 if ds == "ISIC" else 3)
    source_sam = load_sam_encoder(vit, sam_ckpt, IMAGE_SIZE, num_classes_src, device)

    return net, source_sam


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_psi_experiment():
    print("="*70)
    print("PAID Theory: PSI Computation across 18 Models")
    print("="*70)

    all_results = {}

    for arch, ds, vit in MODEL_CONFIGS:
        target_name = f"{ds}_{arch}_{vit}"
        gpu_id = GPU_MAP.get(target_name, 0)
        device = torch.device(f"cuda:{gpu_id}")

        print(f"\n{'='*60}")
        print(f"  {target_name} (GPU {gpu_id})")
        print(f"{'='*60}")

        try:
            # Check checkpoint exists
            ckpt_path, ncls_or_mmo = get_ckpt_path(arch, ds, vit)
            if not os.path.exists(ckpt_path):
                print(f"  SKIP: no checkpoint at {ckpt_path}")
                all_results[target_name] = {"error": "no checkpoint"}
                continue

            # Load model
            net, source_sam = load_model(arch, ds, vit, device)
            # Use TARGET model's encoder (with LoRA/MSA adapters) for PSI
            if arch == "SAMed":
                encoder = net.sam.image_encoder
            else:
                encoder = net.image_encoder

            # Get test images
            if ds == "Synapse":
                test_images = get_synapse_test_images(6)
            elif ds == "ISIC":
                test_images = get_isic_test_images(6)
            else:
                test_images = get_refuge_test_images(6)

            if not test_images:
                print(f"  SKIP: no test images")
                all_results[target_name] = {"error": "no test images"}
                del net, source_sam; gc.collect(); torch.cuda.empty_cache()
                continue

            print(f"  Loaded {len(test_images)} test images")

            # Compute PSI for each image
            all_psi = []
            all_layer_psi = defaultdict(list)

            for img_idx, img_tensor in enumerate(test_images):
                img_t = img_tensor.unsqueeze(0).float().to(device)
                if img_t.max() > 1.0:
                    img_t = img_t / 255.0  # Normalize to [0,1] for SAM

                mean_psi, layer_psi = compute_psi_multi_layer(encoder, img_t, n_layers=4, n_perturbations=10)
                all_psi.append(mean_psi)

                for layer_name, vals in layer_psi.items():
                    all_layer_psi[layer_name].append(vals["psi"])

                if (img_idx + 1) % 3 == 0:
                    print(f"    [{img_idx+1}/{len(test_images)}] mean_psi={np.mean(all_psi):.4f}")

            mean_psi = float(np.mean(all_psi))
            std_psi = float(np.std(all_psi))

            # Average across layers
            layer_avg = {k: float(np.mean(v)) for k, v in all_layer_psi.items()}

            all_results[target_name] = {
                "mean_psi": mean_psi,
                "std_psi": std_psi,
                "n_images": len(test_images),
                "per_layer_psi": layer_avg,
            }

            print(f"  => PSI = {mean_psi:.4f} ± {std_psi:.4f}")
            for ln, lv in sorted(layer_avg.items()):
                print(f"     {ln}: PSI={lv:.4f}")

            del net, source_sam
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[target_name] = {"error": str(e)}
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(OUTPUT_DIR, "paid_psi_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ============================================================
    # VALIDATION: Correlate PSI with Exp14 phase-shuffled ΔDice
    # ============================================================
    print("\n" + "="*70)
    print("VALIDATION: PSI vs Phase-Shuffled Attack Effectiveness")
    print("="*70)

    # Load Exp14 results
    exp14_path = os.path.join(OUTPUT_DIR, "exp14_full_matrix.json")
    if os.path.exists(exp14_path):
        with open(exp14_path) as f:
            exp14 = json.load(f)
    else:
        print("  ERROR: exp14_full_matrix.json not found")
        return

    # Build phase-shuffled ΔDice lookup
    ps_deltas = {}
    for entry in exp14.get("entries", []):
        ds = entry["dataset"]
        arch = entry.get("arch", "samed" if "SAMed" in entry.get("model", "") else "msa")
        vit = entry.get("vit", "vit_b")
        target_name = f"{ds}_{arch.upper() if arch == 'msa' else 'SAMed'}_{vit}"

        # Fix arch naming
        if arch == "msa":
            target_name = f"{ds}_MSA_{vit}"
        else:
            target_name = f"{ds}_SAMed_{vit}"

        # Get phase-shuffled delta
        for cond_name, cond_val in entry.get("conditions", {}).items():
            if "phase_shuffled" in cond_name.lower() or "phase-shuffled" in cond_name.lower():
                ps_deltas[target_name] = cond_val.get("mean_delta", cond_val.get("delta_dice", 0))

    # Also check the structured results
    for group_name, group_data in exp14.get("results", {}).items():
        if isinstance(group_data, dict):
            for model_name, model_data in group_data.items():
                # Parse model name to target_name
                if "phase_shuffled" in model_data:
                    ps_deltas[model_name] = model_data["phase_shuffled"]

    print(f"\n  Phase-shuffled ΔDice available for {len(ps_deltas)} models")

    # Build correlation pairs
    pairs = []
    for target_name, psi_data in all_results.items():
        if "error" in psi_data:
            continue
        if target_name in ps_deltas:
            pairs.append({
                "model": target_name,
                "psi": psi_data["mean_psi"],
                "delta_dice": ps_deltas[target_name],
            })

    if len(pairs) < 6:
        print(f"  Only {len(pairs)} paired models — need at least 6 for meaningful correlation")
        print("  Available PSI models:", list(all_results.keys()))
        print("  Available ΔDice models:", list(ps_deltas.keys()))
        return

    psis = [p["psi"] for p in pairs]
    deltas = [p["delta_dice"] for p in pairs]

    try:
        r_pearson, p_pearson = pearsonr(psis, deltas)
    except Exception:
        r_pearson, p_pearson = 0, 1.0
    try:
        r_spearman, p_spearman = spearmanr(psis, deltas)
    except Exception:
        r_spearman, p_spearman = 0, 1.0

    print(f"\n  Correlation Results:")
    print(f"  ┌─────────────────────┬─────────┬──────────┐")
    print(f"  │ Metric              │    r    │    p     │")
    print(f"  ├─────────────────────┼─────────┼──────────┤")
    print(f"  │ Pearson r           │ {r_pearson:7.4f} │ {p_pearson:8.4f} │")
    print(f"  │ Spearman ρ          │ {r_spearman:7.4f} │ {p_spearman:8.4f} │")
    print(f"  └─────────────────────┴─────────┴──────────┘")

    print(f"\n  Per-Model Breakdown:")
    print(f"  {'Model':<30s} {'PSI':>8s} {'ΔDice':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for p in sorted(pairs, key=lambda x: x["psi"]):
        print(f"  {p['model']:<30s} {p['psi']:8.4f} {p['delta_dice']:8.4f}")

    # Theory assessment
    print(f"\n  Theory Assessment:")
    if r_pearson > 0.7:
        print(f"  ✅ STRONG VALIDATION: PSI predicts phase-shuffled attack (r={r_pearson:.3f})")
    elif r_pearson > 0.5:
        print(f"  ⚠️ MODERATE: PSI has predictive power but incomplete (r={r_pearson:.3f})")
        print(f"  → Need to identify missing factor")
    else:
        print(f"  ❌ WEAK: PSI alone insufficient (r={r_pearson:.3f})")
        print(f"  → Theory needs refinement")

    # Save correlation analysis
    correlation_data = {
        "pairs": pairs,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "n_pairs": len(pairs),
        "assessment": "strong" if r_pearson > 0.7 else "moderate" if r_pearson > 0.5 else "weak",
    }
    with open(os.path.join(OUTPUT_DIR, "paid_correlation.json"), "w") as f:
        json.dump(correlation_data, f, indent=2)

    return correlation_data


if __name__ == "__main__":
    run_psi_experiment()
