#!/usr/bin/env python3
"""
CODD Theory Verification: C1 + C2 across ALL 3 ViT sizes × 3 datasets × 2 methods.
Loops over vit_b, vit_l, vit_h automatically.
"""
import sys, os, gc, time, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT); sys.path.insert(0, NEWEST)

from theory import decompose_drift

IMAGE_SIZE = 512
N_SAMPLES = 200
DEVICE = torch.device("cuda:1")  # cuda:0 used for training
OUT_DIR = os.path.join(NEWEST, "results")
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_BASE = os.path.join(ROOT, "output_runs_correct")

SAM_CKPTS = {
    "vit_b": os.path.join(ROOT, "checkpoints/sam_vit_b_01ec64.pth"),
    "vit_l": os.path.join(ROOT, "checkpoints/sam_vit_l_0b3195.pth"),
    "vit_h": os.path.join(ROOT, "checkpoints/sam_vit_h_4b8939.pth"),
}

# All victim paths: (method, dataset, vit_name, ckpt_path)
def build_victim_list():
    victims = []
    # SAMed
    samed_paths = {
        ("Synapse", "vit_b"): "SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth",
        ("Synapse", "vit_l"): "SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth",  # ⚠️ broken
        ("Synapse", "vit_h"): "SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth",
        ("ISIC",    "vit_b"): "SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth",
        ("ISIC",    "vit_l"): "SAMed/isic_512_lrexp5.0_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth",
        ("ISIC",    "vit_h"): "SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_99.pth",
        ("REFUGE",  "vit_b"): "SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth",
        ("REFUGE",  "vit_l"): "SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_99.pth",
        ("REFUGE",  "vit_h"): "SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_99.pth",
    }
    for (ds, vit), rel in samed_paths.items():
        victims.append(("SAMed", ds, vit, os.path.join(CKPT_BASE, rel)))

    # MSA
    msa_paths = {
        ("Synapse", "vit_b"): "MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("Synapse", "vit_l"): "MedicalSAMAdapter/Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("Synapse", "vit_h"): "MedicalSAMAdapter/Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("ISIC",    "vit_b"): "MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("ISIC",    "vit_l"): "MedicalSAMAdapter/isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("ISIC",    "vit_h"): "MedicalSAMAdapter/isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("REFUGE",  "vit_b"): "MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("REFUGE",  "vit_l"): "MedicalSAMAdapter/REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24/Model/best_dice_checkpoint.pth",
        ("REFUGE",  "vit_h"): "MedicalSAMAdapter/REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24/Model/best_dice_checkpoint.pth",
    }
    for (ds, vit), rel in msa_paths.items():
        victims.append(("MSA", ds, vit, os.path.join(CKPT_BASE, rel)))
    return victims


# ═══ Hook-based feature extraction ═══
def register_hooks(blocks):
    features, handles = [], []
    def hook_fn(module, input, output):
        features.append(output.detach())
    for blk in blocks:
        handles.append(blk.register_forward_hook(hook_fn))
    return features, handles

@torch.no_grad()
def extract_layer_features_via_hooks(encoder_fn, x, blocks):
    features, handles = register_hooks(blocks)
    encoder_fn(x)
    for h in handles: h.remove()
    return features

# ═══ Model loading ═══
def load_base_sam(vit_name):
    from segment_anything import sam_model_registry
    sam, _ = sam_model_registry[vit_name](
        image_size=IMAGE_SIZE, num_classes=8,
        checkpoint=SAM_CKPTS[vit_name], pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    sam = sam.to(DEVICE).eval()
    sam.pixel_mean = sam.pixel_mean.to(DEVICE)
    sam.pixel_std = sam.pixel_std.to(DEVICE)
    return sam, sam.image_encoder.blocks

def load_samed_victim(vit_name, lora_ckpt, num_classes=8):
    from segment_anything import sam_model_registry
    sam, _ = sam_model_registry[vit_name](
        image_size=IMAGE_SIZE, num_classes=num_classes,
        checkpoint=SAM_CKPTS[vit_name], pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
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
    return net, net.sam.image_encoder.blocks, lambda x: net.sam.image_encoder(x)

def load_msa_victim(vit_name, msa_ckpt, num_classes=9):
    msa_root = os.path.join(ROOT, "Medical-SAM-Adapter-main")
    sys.path.insert(0, msa_root)
    from models.sam import sam_model_registry as msa_registry
    from argparse import Namespace
    args = Namespace(dataset="Synapse", image_size=IMAGE_SIZE, multimask_output=num_classes,
                     sam_ckpt=SAM_CKPTS[vit_name], encoder=vit_name, mod="sam_adpt",
                     mid_dim=None, thd=False, chunk=None)
    model = msa_registry[vit_name](args, checkpoint=SAM_CKPTS[vit_name]).to(DEVICE)
    sys.path.remove(msa_root)
    ckpt = torch.load(msa_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model_dict = model.state_dict()
    filtered_sd = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(filtered_sd, strict=False)
    model.pixel_mean = model.pixel_mean.to(DEVICE)
    model.pixel_std = model.pixel_std.to(DEVICE)
    model.eval()
    return model, model.image_encoder.blocks, lambda x: model.image_encoder(model.preprocess(x))


# ═══ Data ═══
def load_synapse_slices(n=200):
    files = sorted(glob.glob(os.path.join(ROOT, "data/Synapse_preprocess/train_npz/*.npz")))
    slices = []
    for f in files:
        data = np.load(f)
        img, lbl = data["image"], data["label"]
        if lbl.sum() < 500: continue
        slc = np.stack([img.astype(np.float32)] * 3, axis=0)
        slices.append(torch.from_numpy(slc))
        if len(slices) >= n: break
    return slices

def load_isic_images(n=200):
    files = sorted(glob.glob(os.path.join(ROOT, "data/ISIC/ISBI2016_ISIC_Part1_Training_Data/*.jpg")))[:n]
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1))
    return imgs

def load_refuge_images(n=200):
    train_dir = os.path.join(ROOT, "data/REFUGE-Multirater/Training-400")
    subdirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    imgs = []
    for sd in subdirs:
        for f in sorted(glob.glob(os.path.join(train_dir, sd, "*.jpg"))):
            img = Image.open(f).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            imgs.append(torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1))
            if len(imgs) >= n: break
        if len(imgs) >= n: break
    return imgs

# ═══ Analysis ═══
def reshape_features(feat_list):
    out = []
    for f in feat_list:
        if f.dim() == 3:
            spatial = f[:, 1:, :]
            h = w = int(spatial.shape[1]**0.5)
            if h*w == spatial.shape[1]:
                spatial = spatial.reshape(1, h, w, -1).permute(0, 3, 1, 2)
            else:
                continue
        elif f.dim() == 4:
            spatial = f
        else:
            continue
        out.append(spatial)
    return out

def per_channel_drift_decompose(bf, vf):
    eps = 1e-8
    Gb = torch.fft.fft2(bf, norm="ortho")
    Gv = torch.fft.fft2(vf, norm="ortho")
    Delta = Gv - Gb

    phase_s = torch.angle(Gb)
    local_s = Delta * torch.conj(torch.exp(1j * phase_s))
    alpha_s = torch.real(local_s); beta_s = torch.imag(local_s)
    a2 = alpha_s.reshape(alpha_s.shape[1], -1).square().sum(dim=1)
    b2 = beta_s.reshape(beta_s.shape[1], -1).square().sum(dim=1)
    rr_src = (a2/(a2+b2+eps)).mean().item()

    rand_phase = torch.rand_like(phase_s) * 2 * np.pi
    local_r = Delta * torch.conj(torch.exp(1j * rand_phase))
    ar = torch.real(local_r); br = torch.imag(local_r)
    a2r = ar.reshape(ar.shape[1], -1).square().sum(dim=1)
    b2r = br.reshape(br.shape[1], -1).square().sum(dim=1)
    rr_rnd = (a2r/(a2r+b2r+eps)).mean().item()

    mag = torch.abs(alpha_s)
    mag_mean = mag.reshape(mag.shape[1], -1).mean(dim=1)
    mag_std = mag.reshape(mag.shape[1], -1).std(dim=1)
    cv = (mag_std/(mag_mean+eps)).mean().item()
    return rr_src, rr_rnd, cv


@torch.no_grad()
def analyze_drift_per_image(base_blocks, base_fn, victim_blocks, victim_fn, img_tensor, n_layers=6):
    x = img_tensor.unsqueeze(0).to(DEVICE)
    bf_raw = extract_layer_features_via_hooks(base_fn, x, base_blocks)
    vf_raw = extract_layer_features_via_hooks(victim_fn, x, victim_blocks)
    bf = reshape_features(bf_raw)[-n_layers:]
    vf = reshape_features(vf_raw)[-n_layers:]

    rr_src, rr_rnd, cv = [], [], []
    for bi, vi in zip(bf, vf):
        if bi.shape != vi.shape:
            if bi.shape[-2:] != vi.shape[-2:]:
                vi = F.interpolate(vi, size=bi.shape[-2:], mode='bilinear', align_corners=False)
            if bi.shape[1] != vi.shape[1]:
                bi = bi.mean(dim=1, keepdim=True); vi = vi.mean(dim=1, keepdim=True)
        rs, rr, c = per_channel_drift_decompose(bi, vi)
        rr_src.append(rs); rr_rnd.append(rr); cv.append(c)
    return rr_src, rr_rnd, cv


# ═══ Main ═══
DATASET_LOADERS = {"Synapse": load_synapse_slices, "ISIC": load_isic_images, "REFUGE": load_refuge_images}
DATASET_CLASSES = {"SAMed": {"Synapse": 8, "ISIC": 1, "REFUGE": 2},
                   "MSA": {"Synapse": 9, "ISIC": 1, "REFUGE": 2}}

def process_vit(vit_name, victims, datasets):
    print(f"\n{'='*60}")
    print(f"CODD Verification: {vit_name}")
    print(f"{'='*60}")

    base_sam, base_blocks = load_base_sam(vit_name)
    base_fn = lambda x: base_sam.image_encoder(x)
    n_total_blocks = len(base_blocks)
    n_layers = min(6, n_total_blocks)
    print(f"  {n_total_blocks} blocks, analyzing last {n_layers}")

    results = {}
    subset = [(m, d, v, p) for (m, d, v, p) in victims if v == vit_name]
    print(f"  {len(subset)} victims to process")

    for method, ds_name, vit, ckpt_path in subset:
        key = f"{method}_{ds_name}_{vit}"
        print(f"\n  [{vit}] {method}/{ds_name} ...")
        t0 = time.time()

        if not os.path.exists(ckpt_path):
            print(f"    ⚠️  Missing checkpoint, skipping")
            continue

        try:
            if method == "SAMed":
                nc = DATASET_CLASSES[method][ds_name]
                _, vblocks, vfn = load_samed_victim(vit_name, ckpt_path, nc)
            else:
                nc = DATASET_CLASSES[method][ds_name]
                _, vblocks, vfn = load_msa_victim(vit_name, ckpt_path, nc)
        except Exception as e:
            print(f"    ❌ Load failed: {e}")
            continue

        images = datasets[ds_name]
        if len(images) == 0:
            print(f"    ⚠️  No images")
            continue

        rr_src_sum = np.zeros(n_layers)
        rr_rnd_sum = np.zeros(n_layers)
        cv_sum = np.zeros(n_layers)
        count = 0

        for i, img in enumerate(images):
            try:
                rs, rr, cv = analyze_drift_per_image(base_blocks, base_fn, vblocks, vfn, img, n_layers)
                rr_src_sum += np.array(rs); rr_rnd_sum += np.array(rr); cv_sum += np.array(cv)
                count += 1
            except Exception as e:
                if i < 2: print(f"    W img {i}: {e}")
            if (i+1) % 100 == 0:
                print(f"    ... {i+1}/{len(images)}")

        if count == 0:
            print(f"    ❌ All images failed")
            continue

        rr_src_mean = rr_src_sum / count
        rr_rnd_mean = rr_rnd_sum / count
        cv_mean = cv_sum / count
        avg_rr_src = float(rr_src_mean.mean())
        avg_rr_rnd = float(rr_rnd_mean.mean())
        avg_cv = float(cv_mean.mean())

        results[key] = {
            "vit": vit, "method": method, "dataset": ds_name,
            "avg_rr_source": avg_rr_src, "avg_rr_random": avg_rr_rnd,
            "avg_cv_radial": avg_cv, "n_images": count, "n_blocks": n_total_blocks,
            "per_layer_rr_source": rr_src_mean.tolist(),
            "per_layer_rr_random": rr_rnd_mean.tolist(),
            "per_layer_cv": cv_mean.tolist(),
        }
        print(f"    ✓ RR_src={avg_rr_src:.4f} RR_rnd={avg_rr_rnd:.4f} CV={avg_cv:.4f} (n={count}, {time.time()-t0:.0f}s)")

        del vblocks, vfn
        torch.cuda.empty_cache()

    return results


def main():
    all_results = {}
    victims = build_victim_list()
    vit_names = ["vit_b", "vit_l", "vit_h"]

    # Load data once
    print("Loading training data ...")
    datasets = {ds: loader(N_SAMPLES) for ds, loader in DATASET_LOADERS.items()}
    for ds, imgs in datasets.items():
        print(f"  {ds}: {len(imgs)} images")

    for vit in vit_names:
        res = process_vit(vit, victims, datasets)
        all_results.update(res)

    # ─── Aggregate summary ───
    print("\n" + "=" * 90)
    print("CODD THEORY VERIFICATION — ALL 3 ViT SIZES")
    print("=" * 90)
    header = f"{'Victim':<30} {'RR_src':>8} {'RR_rnd':>8} {'CV':>8} {'C1':>6} {'C2':>6}"
    print(header)
    print("-" * 90)

    all_keys = sorted(all_results.keys(), key=lambda k: (k.split('_')[-1], k.split('_')[0], k.split('_')[1]))
    for key in all_keys:
        r = all_results[key]
        c1 = "✓" if r["avg_rr_source"] > 0.7 else ("~" if r["avg_rr_source"] > 0.6 else "✗")
        c2 = "✓" if r["avg_cv_radial"] > 0.5 else "✗"
        label = f"{r['method']}_{r['dataset']}_{r['vit']}"
        print(f"{label:<30} {r['avg_rr_source']:>8.4f} {r['avg_rr_random']:>8.4f} {r['avg_cv_radial']:>8.4f} {c1:>6} {c2:>6}")

    # ─── By vit size summary ───
    print("\n" + "=" * 90)
    print("BY VIT SIZE (RR_source mean ± std)")
    print("-" * 90)
    for vit in vit_names:
        subset = {k: v for k, v in all_results.items() if v["vit"] == vit}
        if not subset: continue
        rrs = [v["avg_rr_source"] for v in subset.values()]
        cvs = [v["avg_cv_radial"] for v in subset.values()]
        print(f"  {vit} ({len(subset)} victims): RR_src = {np.mean(rrs):.4f} ± {np.std(rrs):.4f},  CV = {np.mean(cvs):.4f} ± {np.std(cvs):.4f}")

    # By method
    print("\nBY METHOD:")
    for method in ["SAMed", "MSA"]:
        subset = {k: v for k, v in all_results.items() if v["method"] == method}
        if not subset: continue
        rrs = [v["avg_rr_source"] for v in subset.values()]
        print(f"  {method} ({len(subset)} victims): RR_src = {np.mean(rrs):.4f} ± {np.std(rrs):.4f}")

    out_path = os.path.join(OUT_DIR, "codd_theory_all_vit.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
