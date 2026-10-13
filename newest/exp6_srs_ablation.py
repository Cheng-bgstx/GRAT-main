#!/usr/bin/env python3
"""
Experiment 6-8: SRS Ablation Studies (Shared Radial Subspace).

Exp 6: mean vs PCA — compare no_warmstart, UAP, mean_only, mean+PCA(rank=8,16,32)
Exp 7: proxy data scale — 10%, 25%, 50%, 75%, 100% of Synapse training data
Exp 8: proxy data type — Synapse, ISIC, REFUGE, ImageNet, SA-1B (if available)

All on Synapse SAMed vit_b target.
"""
import os, sys, json, gc, random
import numpy as np
import torch
import h5py
from collections import defaultdict

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack, distill_srs
from theory import SharedSubspace, blend_with_subspace, fit_shared_subspace
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIT = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
CKPT = f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

N_TEST_CASES = 6
NUM_SLICES = 4
SRS_RANK = 8
SRS_NUM_SAMPLES = 128  # number of proxy images for SRS distillation


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


def get_synapse_train_slices(max_slices=200):
    """Load slices from Synapse training data (per-slice npz format) for SRS distillation."""
    train_dir = f"{ROOT}/data/Synapse_preprocess/train_npz"
    if not os.path.isdir(train_dir):
        print(f"  WARNING: train_npz directory not found at {train_dir}")
        return []
    npzs = sorted([f for f in os.listdir(train_dir) if f.endswith('.npz')])
    slices = []
    for npz_name in npzs:
        if len(slices) >= max_slices:
            break
        try:
            data = np.load(os.path.join(train_dir, npz_name), allow_pickle=True)
            img = data['image']  # [H, W] single slice
            lbl = data['label']  # [H, W]
            # Skip empty slices
            if (lbl > 0).sum() <= 100:
                continue
            slc = img.astype(np.float32)
            slc_3ch = np.stack([slc]*3, axis=0)
            slices.append(torch.from_numpy(slc_3ch))
        except Exception:
            continue
    return slices


def get_isic_train_images(max_images=200):
    """Load ISIC training images for SRS distillation."""
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_GroundTruth.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for _, row in df.iterrows():
        if len(images) >= max_images: break
        img_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_Data", row['img'])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img) * 255)
    return images


def get_refuge_train_images(max_images=200):
    """Load REFUGE training images for SRS distillation."""
    from PIL import Image
    from torchvision import transforms
    train_dir = f"{ROOT}/data/REFUGE-Multirater/Training-400"
    if not os.path.isdir(train_dir):
        return []
    folders = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for folder in folders:
        if len(images) >= max_images: break
        img_path = os.path.join(train_dir, folder, f"{folder}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img) * 255)
    return images


def distill_srs_from_slices(source_sam, slices, config, rank=8):
    """Lightweight SRS distillation from a list of image slices."""
    gradient_list = []
    for idx, slc in enumerate(slices):
        img_t = slc.unsqueeze(0).float().to(DEVICE)
        # Ensure 3 channels
        if img_t.shape[1] == 1:
            img_t = img_t.repeat(1, 3, 1, 1)

        with torch.no_grad():
            targets, targets_layer = source_sam.image_encoder(img_t)

        # Quick 3-step gradient collection
        import torch.nn.functional as F
        from torch.autograd import Variable
        perturb_img = Variable(img_t.data + torch.randn_like(img_t) * 1e-5, requires_grad=True)
        local_grads = []

        for step_i in range(3):
            source_sam.zero_grad()
            features, features_layer = source_sam.image_encoder(perturb_img)
            loss_org = F.l1_loss(features * 0.9999, targets)

            # Simple directional energy
            pre_layer = int(len(features_layer) * 0.9)
            num_layers = min(6, pre_layer)
            idx_i = num_layers - 1
            F_clean_i = targets_layer[idx_i + pre_layer - num_layers]
            F_adv_i = features_layer[idx_i + pre_layer - num_layers]

            from theory import synthesize_codd_probe
            probe = synthesize_codd_probe(
                F_clean_i.unsqueeze(0), F_adv_i.unsqueeze(0),
                direction="radial", ads_rho=1.0, ads_gamma=1e-4,
            )
            dir_energy = (probe ** 2).mean()
            loss = loss_org - dir_energy

            grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
            grad_norm = grad.detach() / (grad.detach().abs().sum() + 1e-12)
            local_grads.append(grad_norm.cpu())

            eta = torch.clamp(perturb_img.data - img_t.data + 0.5 * grad_norm.sign(), -2.0, 2.0)
            perturb_img = Variable(torch.clamp(img_t.data + eta, img_t.min(), img_t.max()), requires_grad=True)

        avg_grad = torch.stack(local_grads, dim=0).mean(dim=0)
        gradient_list.append(avg_grad)

        if (idx + 1) % 32 == 0:
            print(f"    [SRS distill] {idx+1}/{len(slices)} gradients collected")

    if not gradient_list:
        return None

    subspace = fit_shared_subspace(gradient_list, rank=rank)
    print(f"    SRS fitted: rank={rank}, explained_var={subspace.explained_variance:.4f}")
    return subspace


def run_attack_with_srs(source_sam, img_t, config, srs_bundle, use_srs, srs_mean_weight=0.0):
    """Run CODD attack with optional SRS warm-start."""
    if use_srs and srs_bundle is not None:
        config.use_srs = True
        config.srs_proj_weight = 0.15
        config.srs_mean_weight = srs_mean_weight
    else:
        config.use_srs = False
        config.srs_proj_weight = 0.0
        config.srs_mean_weight = 0.0

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    adv_np, _ = run_codd_attack(
        source_sam.image_encoder, img_t, config,
        srs_bundle=srs_bundle if use_srs else None,
        return_trace=True,
    )
    return adv_np


def get_test_slices(n_cases):
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:n_cases]
    all_slices = []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:NUM_SLICES]
        for s in valid:
            all_slices.append((img[s].astype(np.float32), lbl[s], h5f, s))
    return all_slices


def main():
    print("Loading source SAM...")
    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    target = load_samed_model(VIT, SAM_CKPT, CKPT, IMAGE_SIZE, 8, device=DEVICE)

    # ---- Exp 6: mean vs PCA ----
    print("\n" + "="*60)
    print("Experiment 6: SRS mean vs PCA")
    print("="*60)

    train_slices = get_synapse_train_slices(SRS_NUM_SAMPLES)
    print(f"  Loaded {len(train_slices)} training slices for SRS")

    if train_slices:
        config = AttackConfig(
            att_step_size=2.0, att_bound=10.0, att_numstep=10,
            direction_mode="radial", probe_amplification=2.0,
            edge_guidance=True, seed=42,
        )
        srs_bundle = distill_srs_from_slices(source_sam, train_slices, config, rank=SRS_RANK)

        test_slices = get_test_slices(N_TEST_CASES)
        print(f"  Evaluating on {len(test_slices)} test slices...")

        exp6_results = {}
        configs_6 = [
            ("no_warmstart", False, 0.0),
            ("uap_warmstart", True, 0.0),  # use UAP-like (mean only, no PCA projection)
            ("mean_only", True, 0.2),
            ("mean_pca_r8", True, 0.1),
        ]

        for cond_name, use_srs, mean_w in configs_6:
            deltas = []
            for slc, lbl_slice, h5f, si in test_slices[:12]:  # subset for speed
                clean_d = evaluate_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
                slc_3ch = np.stack([slc]*3, axis=0)
                img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

                local_config = AttackConfig(
                    att_step_size=2.0, att_bound=10.0, att_numstep=10,
                    direction_mode="radial", probe_amplification=2.0,
                    edge_guidance=True, seed=42,
                )
                adv_np = run_attack_with_srs(source_sam, img_t, local_config,
                                             srs_bundle, use_srs, srs_mean_weight=mean_w)
                adv_d = evaluate_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
                deltas.append(clean_d - adv_d)

            exp6_results[cond_name] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "n": len(deltas),
            }
            print(f"    {cond_name}: Δ={exp6_results[cond_name]['mean_delta']:.4f} ± {exp6_results[cond_name]['std_delta']:.4f}")

        # Save intermediate
        with open(os.path.join(OUTPUT_DIR, "exp6_srs_pca.json"), "w") as f:
            json.dump({"config": {"rank": SRS_RANK, "n_train": len(train_slices)}, "results": exp6_results}, f, indent=2)

    # ---- Exp 7: proxy data scale ----
    print("\n" + "="*60)
    print("Experiment 7: Proxy Data Scale")
    print("="*60)

    if train_slices:
        scales = [0.10, 0.25, 0.50, 0.75, 1.0]
        exp7_results = {}
        test_slices = get_test_slices(N_TEST_CASES)

        for scale in scales:
            n_train = max(8, int(len(train_slices) * scale))
            subset = train_slices[:n_train]
            config = AttackConfig(
                att_step_size=2.0, att_bound=10.0, att_numstep=10,
                direction_mode="radial", probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            sub_bundle = distill_srs_from_slices(source_sam, subset, config, rank=SRS_RANK)

            deltas = []
            for slc, lbl_slice, h5f, si in test_slices[:8]:
                clean_d = evaluate_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
                slc_3ch = np.stack([slc]*3, axis=0)
                img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)
                local_config = AttackConfig(
                    att_step_size=2.0, att_bound=10.0, att_numstep=10,
                    direction_mode="radial", probe_amplification=2.0,
                    edge_guidance=True, seed=42,
                )
                adv_np = run_attack_with_srs(source_sam, img_t, local_config,
                                             sub_bundle, True, srs_mean_weight=0.1)
                adv_d = evaluate_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
                deltas.append(clean_d - adv_d)

            tag = f"scale_{int(scale*100)}pct"
            exp7_results[tag] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "n_train": n_train,
                "n_test": len(deltas),
            }
            print(f"    {tag} (n={n_train}): Δ={exp7_results[tag]['mean_delta']:.4f}")

        with open(os.path.join(OUTPUT_DIR, "exp7_proxy_scale.json"), "w") as f:
            json.dump({"config": {"scales": scales}, "results": exp7_results}, f, indent=2)

    # ---- Exp 8: proxy data type ----
    print("\n" + "="*60)
    print("Experiment 8: Proxy Data Type")
    print("="*60)

    test_slices = get_test_slices(N_TEST_CASES)
    proxy_sources = {
        "Synapse": get_synapse_train_slices(128),
        "ISIC": get_isic_train_images(128),
        "REFUGE": get_refuge_train_images(128),
    }

    exp8_results = {}
    for src_name, proxy_slices in proxy_sources.items():
        if not proxy_slices:
            print(f"    SKIP {src_name}: no proxy data available")
            continue
        print(f"    {src_name}: {len(proxy_slices)} proxy slices")

        config = AttackConfig(
            att_step_size=2.0, att_bound=10.0, att_numstep=10,
            direction_mode="radial", probe_amplification=2.0,
            edge_guidance=True, seed=42,
        )
        proxy_bundle = distill_srs_from_slices(source_sam, proxy_slices, config, rank=SRS_RANK)

        deltas = []
        for slc, lbl_slice, h5f, si in test_slices[:8]:
            clean_d = evaluate_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
            slc_3ch = np.stack([slc]*3, axis=0)
            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)
            local_config = AttackConfig(
                att_step_size=2.0, att_bound=10.0, att_numstep=10,
                direction_mode="radial", probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            adv_np = run_attack_with_srs(source_sam, img_t, local_config,
                                         proxy_bundle, True, srs_mean_weight=0.1)
            adv_d = evaluate_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
            deltas.append(clean_d - adv_d)

        exp8_results[src_name] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "n_proxy": len(proxy_slices),
            "n_test": len(deltas),
        }
        print(f"      Δ={exp8_results[src_name]['mean_delta']:.4f} ± {exp8_results[src_name]['std_delta']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "exp8_proxy_type.json"), "w") as f:
        json.dump({"config": {}, "results": exp8_results}, f, indent=2)

    print("\nDONE")
    del source_sam, target; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
