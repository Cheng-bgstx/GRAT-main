#!/usr/bin/env python3
"""
CODD Theory Core Verification: C1 (Phase Direction Stable) + C2 (Amplitude Random)

Uses forward hooks on ViT blocks to extract per-layer features identically
from both base SAM encoder and fine-tuned victim encoders.
"""
import sys, os, gc, time, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from theory import decompose_drift

IMAGE_SIZE = 512
VIT = "vit_b"
N_SAMPLES = 200
DEVICE = torch.device("cuda:0")
OUT_DIR = os.path.join(NEWEST, "results")
os.makedirs(OUT_DIR, exist_ok=True)

SAM_CKPT = os.path.join(ROOT, "checkpoints", f"sam_{VIT}_01ec64.pth")
CKPT_BASE = os.path.join(ROOT, "output_runs_correct")

VICTIMS = [
    ("SAMed", "Synapse", os.path.join(CKPT_BASE, "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth")),
    ("SAMed", "ISIC",    os.path.join(CKPT_BASE, "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth")),
    ("SAMed", "REFUGE",  os.path.join(CKPT_BASE, "SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth")),
    ("MSA",   "Synapse", os.path.join(CKPT_BASE, "MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth")),
    ("MSA",   "ISIC",    os.path.join(CKPT_BASE, "MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth")),
    ("MSA",   "REFUGE",  os.path.join(CKPT_BASE, "MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth")),
]


# ═══════════════════════════════════════════════════════════
# Hook-based feature extraction (works for any ViT encoder)
# ═══════════════════════════════════════════════════════════

def register_hooks(blocks):
    """Register forward hooks on transformer blocks to capture outputs."""
    features = []
    handles = []
    def hook_fn(module, input, output):
        features.append(output.detach())  # output: [B, N, C] or [B, H, W, C]
    for blk in blocks:
        handles.append(blk.register_forward_hook(hook_fn))
    return features, handles


@torch.no_grad()
def extract_layer_features_via_hooks(encoder_fn, x, blocks):
    """Extract per-block features using hooks. encoder_fn does the full forward."""
    features, handles = register_hooks(blocks)
    encoder_fn(x)
    for h in handles:
        h.remove()
    # features[i]: output of block i, shape varies by model
    return features


# ═══════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════

def load_base_sam():
    """Load clean SAM model. Returns (sam_model, blocks)."""
    from segment_anything import sam_model_registry
    sam, _ = sam_model_registry[VIT](
        image_size=IMAGE_SIZE, num_classes=8,
        checkpoint=SAM_CKPT, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    sam = sam.to(DEVICE).eval()
    sam.pixel_mean = sam.pixel_mean.to(DEVICE)
    sam.pixel_std = sam.pixel_std.to(DEVICE)
    blocks = sam.image_encoder.blocks
    return sam, blocks


def load_samed_victim(lora_ckpt, num_classes=8):
    """Load SAMed model (SAM + LoRA). Returns (model, blocks, forward_fn)."""
    from evaluate import _safe_load_sam
    sam = _safe_load_sam(VIT, SAM_CKPT, IMAGE_SIZE, num_classes)
    ckpt_state = torch.load(lora_ckpt, map_location="cpu")
    if any(k.startswith("sa_a_") for k in ckpt_state.keys()):
        module_name = "sam_lora_image_encoder_mask_decoder"
    else:
        module_name = "sam_lora_image_encoder"
    pkg = __import__(module_name, fromlist=["LoRA_Sam"])
    lora_keys = sorted([k for k in ckpt_state.keys() if k.startswith("w_a_")])
    n_blocks = len(lora_keys) // 2 if lora_keys else len(sam.image_encoder.blocks)
    net = pkg.LoRA_Sam(sam, 4, lora_layer=list(range(n_blocks))).to(DEVICE)
    net.load_lora_parameters(lora_ckpt)
    net = net.to(DEVICE)
    net.sam.pixel_mean = net.sam.pixel_mean.to(DEVICE)
    net.sam.pixel_std = net.sam.pixel_std.to(DEVICE)
    net.eval()
    blocks = net.sam.image_encoder.blocks

    def forward_fn(x):
        net.sam.image_encoder(x)
    return net, blocks, forward_fn


def load_msa_victim(msa_ckpt, num_classes=9):
    """Load MSA model. Returns (model, blocks, forward_fn)."""
    msa_root = os.path.join(ROOT, "Medical-SAM-Adapter-main")
    sys.path.insert(0, msa_root)
    from models.sam import sam_model_registry as msa_registry
    from argparse import Namespace
    args = Namespace(
        dataset="Synapse", image_size=IMAGE_SIZE, multimask_output=num_classes,
        sam_ckpt=SAM_CKPT, encoder=VIT, mod="sam_adpt",
        mid_dim=None, thd=False, chunk=None,
    )
    model = msa_registry[VIT](args, checkpoint=SAM_CKPT).to(DEVICE)
    sys.path.remove(msa_root)
    ckpt = torch.load(msa_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model_dict = model.state_dict()
    filtered_sd = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape != model_dict[k].shape:
            continue
        filtered_sd[k] = v
    model.load_state_dict(filtered_sd, strict=False)
    model.pixel_mean = model.pixel_mean.to(DEVICE)
    model.pixel_std = model.pixel_std.to(DEVICE)
    model.eval()
    blocks = model.image_encoder.blocks

    def forward_fn(x):
        x_pre = model.preprocess(x)
        model.image_encoder(x_pre)
    return model, blocks, forward_fn


# ═══════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════

def load_synapse_slices(n=N_SAMPLES):
    npz_dir = os.path.join(ROOT, "data/Synapse_preprocess/train_npz")
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    slices = []
    for f in files:
        data = np.load(f)
        img, lbl = data["image"], data["label"]
        if lbl.sum() < 500:
            continue
        slc = img.astype(np.float32)
        slc_3ch = np.stack([slc] * 3, axis=0)
        slices.append(torch.from_numpy(slc_3ch))
        if len(slices) >= n:
            break
    print(f"  Synapse: {len(slices)} slices")
    return slices


def load_isic_images(n=N_SAMPLES):
    img_dir = os.path.join(ROOT, "data/ISIC/ISBI2016_ISIC_Part1_Training_Data")
    files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:n]
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    print(f"  ISIC: {len(imgs)} images")
    return imgs


def load_refuge_images(n=N_SAMPLES):
    """REFUGE: each numbered subdirectory contains the fundus image (*.jpg)."""
    train_dir = os.path.join(ROOT, "data/REFUGE-Multirater/Training-400")
    subdirs = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d))])
    imgs = []
    for sd in subdirs:
        sd_path = os.path.join(train_dir, sd)
        jpgs = sorted(glob.glob(os.path.join(sd_path, "*.jpg")))
        for f in jpgs:
            img = Image.open(f).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            arr = np.array(img).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
            if len(imgs) >= n:
                break
        if len(imgs) >= n:
            break
    print(f"  REFUGE: {len(imgs)} images")
    return imgs


# ═══════════════════════════════════════════════════════════
# Drift analysis per image
# ═══════════════════════════════════════════════════════════

def reshape_features(feat_list):
    """Convert list of transformer block outputs [B,N,C] to [1,C,H,W] for FFT.
    Each block output: [1, N_tokens, C] where N_tokens = 1 + H*W for ViT with cls token.
    Returns list of [1, C', H, W] by averaging channels and removing cls token.
    """
    out = []
    for f in feat_list:
        if f.dim() == 3:  # [B, N, C] format
            # Remove cls token, reshape to spatial
            spatial = f[:, 1:, :]  # [1, H*W, C]
            # Infer H=W=sqrt(N)
            h = w = int(spatial.shape[1] ** 0.5)
            if h * w == spatial.shape[1]:
                spatial = spatial.reshape(1, h, w, -1).permute(0, 3, 1, 2)  # [1, C, H, W]
            else:
                # Non-square, just use as-is
                spatial = spatial.permute(0, 2, 1).reshape(1, -1, 1, h * w)  # fallback
                continue
        elif f.dim() == 4:
            spatial = f  # Already [B, C, H, W]
        else:
            continue
        out.append(spatial)
    return out


def per_channel_drift_decompose(bf, vf):
    """Decompose drift per-channel (vectorized), average RR across channels.

    FFT is done on all channels simultaneously: [1, C, H, W] -> [1, C, H, W]
    Returns per-channel averaged RR_source, RR_random, CV_radial.
    """
    eps = 1e-8
    # Vectorized FFT over all channels
    Gb = torch.fft.fft2(bf, norm="ortho")  # [1, C, H, W]
    Gv = torch.fft.fft2(vf, norm="ortho")
    Delta = Gv - Gb  # [1, C, H, W]

    # Source carrier decomposition
    phase_s = torch.angle(Gb)
    carrier_s = torch.exp(1j * phase_s)
    local_s = Delta * torch.conj(carrier_s)
    alpha_s = torch.real(local_s)  # [1, C, H, W]
    beta_s = torch.imag(local_s)

    # Per-channel energy
    alpha2 = alpha_s.reshape(alpha_s.shape[1], -1).square().sum(dim=1)  # [C]
    beta2 = beta_s.reshape(beta_s.shape[1], -1).square().sum(dim=1)
    rr_src = (alpha2 / (alpha2 + beta2 + eps)).mean().item()

    # Random carrier baseline (vectorized)
    rand_phase = torch.rand_like(phase_s) * 2 * np.pi
    carrier_r = torch.exp(1j * rand_phase)
    local_r = Delta * torch.conj(carrier_r)
    alpha_r = torch.real(local_r)
    beta_r = torch.imag(local_r)
    alpha2_r = alpha_r.reshape(alpha_r.shape[1], -1).square().sum(dim=1)
    beta2_r = beta_r.reshape(beta_r.shape[1], -1).square().sum(dim=1)
    rr_rnd = (alpha2_r / (alpha2_r + beta2_r + eps)).mean().item()

    # CV of radial response
    mag = torch.abs(alpha_s)  # [1, C, H, W]
    mag_mean = mag.reshape(mag.shape[1], -1).mean(dim=1)  # [C]
    mag_std = mag.reshape(mag.shape[1], -1).std(dim=1)
    cv = (mag_std / (mag_mean + eps)).mean().item()

    return rr_src, rr_rnd, cv


@torch.no_grad()
def analyze_drift_per_image(base_blocks, base_forward_fn,
                            victim_blocks, victim_forward_fn,
                            img_tensor, n_layers_last=6):
    """Extract features via hooks, decompose drift in source carrier basis.

    Returns per-layer: rr_source, rr_random, cv_radial
    Uses PER-CHANNEL decomposition (FFT each channel independently).
    """
    x = img_tensor.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

    base_feats_raw = extract_layer_features_via_hooks(base_forward_fn, x, base_blocks)
    victim_feats_raw = extract_layer_features_via_hooks(victim_forward_fn, x, victim_blocks)

    base_feats = reshape_features(base_feats_raw)[-n_layers_last:]
    victim_feats = reshape_features(victim_feats_raw)[-n_layers_last:]

    rr_source, rr_random, cv_radial = [], [], []

    for bf, vf in zip(base_feats, victim_feats):
        if bf.shape != vf.shape:
            if bf.shape[-2:] != vf.shape[-2:]:
                vf = F.interpolate(vf, size=bf.shape[-2:], mode='bilinear', align_corners=False)
            if bf.shape[1] != vf.shape[1]:
                bf = bf.mean(dim=1, keepdim=True)
                vf = vf.mean(dim=1, keepdim=True)

        rr_s, rr_r, cv = per_channel_drift_decompose(bf, vf)
        rr_source.append(rr_s)
        rr_random.append(rr_r)
        cv_radial.append(cv)

    return rr_source, rr_random, cv_radial


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("CODD Theory Verification: C1 (Phase Direction) + C2 (Amplitude Random)")
    print(f"ViT: {VIT} | N={N_SAMPLES} | Device: {DEVICE}")
    print("=" * 60)

    # Load base SAM
    print("\n[1] Loading base SAM ...")
    base_sam, base_blocks = load_base_sam()
    base_forward_fn = lambda x: base_sam.image_encoder(x)
    n_blocks = len(base_blocks)
    print(f"  {n_blocks} transformer blocks")

    # Load data
    print("\n[2] Loading training data ...")
    datasets = {
        "Synapse": load_synapse_slices(N_SAMPLES),
        "ISIC": load_isic_images(N_SAMPLES),
        "REFUGE": load_refuge_images(N_SAMPLES),
    }

    all_results = {}

    for method, ds_name, ckpt_path in VICTIMS:
        key = f"{method}_{ds_name}"
        print(f"\n[3] {key} ...")
        t0 = time.time()

        if not os.path.exists(ckpt_path):
            print(f"  ⚠️  Missing: {ckpt_path}")
            continue

        # Load victim
        try:
            if method == "SAMed":
                num_cls = {"Synapse": 8, "ISIC": 1, "REFUGE": 2}[ds_name]
                _, victim_blocks, victim_forward_fn = load_samed_victim(ckpt_path, num_cls)
            else:
                num_cls = {"Synapse": 9, "ISIC": 1, "REFUGE": 2}[ds_name]
                _, victim_blocks, victim_forward_fn = load_msa_victim(ckpt_path, num_cls)
        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            continue

        images = datasets[ds_name]
        if len(images) == 0:
            print(f"  ⚠️  No images for {ds_name}")
            continue

        n_layers = 6  # last 6 layers
        rr_src_sum = np.zeros(n_layers)
        rr_rnd_sum = np.zeros(n_layers)
        cv_sum = np.zeros(n_layers)
        count = 0

        for i, img in enumerate(images):
            try:
                rr_s, rr_r, cv_r = analyze_drift_per_image(
                    base_blocks, base_forward_fn,
                    victim_blocks, victim_forward_fn,
                    img, n_layers_last=n_layers,
                )
                if len(rr_s) == n_layers:
                    rr_src_sum += np.array(rr_s)
                    rr_rnd_sum += np.array(rr_r)
                    cv_sum += np.array(cv_r)
                    count += 1
            except Exception as e:
                if i < 3:
                    print(f"  Warning img {i}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(images)}")

        if count == 0:
            print(f"  ❌ All images failed")
            continue

        rr_src_mean = rr_src_sum / count
        rr_rnd_mean = rr_rnd_sum / count
        cv_mean = cv_sum / count

        avg_rr_src = float(rr_src_mean.mean())
        avg_rr_rnd = float(rr_rnd_mean.mean())
        avg_cv = float(cv_mean.mean())

        all_results[key] = {
            "avg_rr_source": avg_rr_src,
            "avg_rr_random": avg_rr_rnd,
            "avg_cv_radial": avg_cv,
            "per_layer_rr_source": rr_src_mean.tolist(),
            "per_layer_rr_random": rr_rnd_mean.tolist(),
            "per_layer_cv": cv_mean.tolist(),
            "n_images": count,
        }

        print(f"  ✓ {key}: RR_src={avg_rr_src:.4f}, RR_rnd={avg_rr_rnd:.4f}, CV={avg_cv:.4f} (n={count}, {time.time()-t0:.0f}s)")
        for li in range(n_layers):
            layer_i = n_blocks - n_layers + li
            print(f"    Layer {layer_i}: RR_src={rr_src_mean[li]:.4f}, RR_rnd={rr_rnd_mean[li]:.4f}, CV={cv_mean[li]:.4f}")

        del victim_blocks, victim_forward_fn
        torch.cuda.empty_cache()

    # ─── Summary ───
    print("\n" + "=" * 72)
    print("CODD THEORY VERIFICATION RESULTS")
    print("=" * 72)
    print(f"{'Victim':<25} {'RR_source':>10} {'RR_random':>10} {'CV_radial':>10} {'C1(RR>0.7)':>12} {'C2(CV>0.5)':>12}")
    print("-" * 72)

    c1_pass = 0
    c2_pass = 0
    for key, res in all_results.items():
        c1 = "✓" if res["avg_rr_source"] > 0.7 else "✗"
        c2 = "✓" if res["avg_cv_radial"] > 0.5 else "✗"
        if c1 == "✓": c1_pass += 1
        if c2 == "✓": c2_pass += 1
        print(f"{key:<25} {res['avg_rr_source']:>10.4f} {res['avg_rr_random']:>10.4f} {res['avg_cv_radial']:>10.4f} {c1:>12} {c2:>12}")

    # Save
    out_path = os.path.join(OUT_DIR, "codd_theory_verification.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    n_total = len(all_results)
    print(f"\nC1 (Phase Direction Stable): {c1_pass}/{n_total} passed")
    print(f"C2 (Amplitude Random):      {c2_pass}/{n_total} passed")


if __name__ == "__main__":
    main()
