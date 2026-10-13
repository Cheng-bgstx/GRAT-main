#!/usr/bin/env python3
"""
Complete remaining experiments: Exp 8 (proxy type), Exp 11 (computation cost), Exp 13 (qualitative).

Exp 8: Add ISIC, SA-1B proxy data types to SRS proxy type ablation
Exp 11: Full computational cost benchmark (PGD, MI-FGSM, NI-FGSM, DI-FGSM, TI-FGSM, SI-FGSM, SE-CGR)
Exp 13: Qualitative visualization — 6 cases (3 datasets × 2 cases each)
"""
import os, sys, json, time, random, gc, tarfile, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack, _get_edge_map, _return_eta, synthesize_codd_probe
from theory import SharedSubspace, blend_with_subspace, fit_shared_subspace, decompose_drift
from evaluate import load_sam_encoder, load_samed_model

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda:3")
IMAGE_SIZE = 512
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "qualitative"), exist_ok=True)

VIT = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SYN_CKPT = f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

SRS_RANK = 8
N_PROXY = 128
N_TEST_EXP8 = 8
ATTACK_STEPS = 20
criterion_l1 = nn.L1Loss()


# ============================================================
# SAFE EVALUATION (device-aware, no .cuda() bug)
# ============================================================
def evaluate_samed_slice(net, image_np, label_np, case_name):
    from scipy.ndimage import zoom
    from utils import calculate_metric_percase
    device = next(net.parameters()).device
    if image_np.ndim == 2:
        image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2:
        label_np = label_np[np.newaxis, :, :]
    prediction = np.zeros_like(label_np)
    for ind in range(image_np.shape[0]):
        slc = image_np[ind, :, :]
        x, y = slc.shape
        if x != IMAGE_SIZE or y != IMAGE_SIZE:
            slc = zoom(slc, (IMAGE_SIZE / x, IMAGE_SIZE / y), order=3)
        nx, ny = slc.shape
        if nx != IMAGE_SIZE or ny != IMAGE_SIZE:
            slc = zoom(slc, (IMAGE_SIZE / nx, IMAGE_SIZE / ny), order=3)
        inputs = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().to(device)
        inputs = inputs.repeat(1, 3, 1, 1)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, True, IMAGE_SIZE)
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            oh, ow = out.shape
            if x != oh or y != ow:
                pred = zoom(out, (x / oh, y / ow), order=0)
            else:
                pred = out
            prediction[ind] = pred
    metric_list = []
    for i in range(1, 9):
        metric_list.append(calculate_metric_percase(prediction == i, label_np == i))
    return float(np.mean(metric_list, axis=0)[0])


# ============================================================
# PROXY DATA LOADERS
# ============================================================
def get_synapse_train_slices(max_slices=200):
    train_dir = f"{ROOT}/data/Synapse_preprocess/train_npz"
    if not os.path.isdir(train_dir):
        return []
    npzs = sorted([f for f in os.listdir(train_dir) if f.endswith('.npz')])
    slices = []
    for npz_name in npzs:
        if len(slices) >= max_slices:
            break
        try:
            data = np.load(os.path.join(train_dir, npz_name), allow_pickle=True)
            img, lbl = data['image'], data['label']
            if (lbl > 0).sum() <= 100:
                continue
            slc = img.astype(np.float32)
            slc_3ch = np.stack([slc]*3, axis=0)
            slices.append(torch.from_numpy(slc_3ch))
        except Exception:
            continue
    return slices


def get_isic_train_images(max_images=200):
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    csv_path = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_GroundTruth.csv")
    if not os.path.exists(csv_path):
        print(f"  ISIC CSV not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for _, row in df.iterrows():
        if len(images) >= max_images:
            break
        # row['img'] already includes the subdirectory path
        img_path = os.path.join(data_dir, str(row['img']))
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img) * 255)
    return images


def get_refuge_train_images(max_images=200):
    from torchvision import transforms
    train_dir = f"{ROOT}/data/REFUGE-Multirater/Training-400"
    if not os.path.isdir(train_dir):
        return []
    folders = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for folder in folders:
        if len(images) >= max_images:
            break
        img_path = os.path.join(train_dir, folder, f"{folder}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img) * 255)
    return images


def get_sa1b_images(max_images=200):
    """Load natural images from SA-1B tar files."""
    from torchvision import transforms
    sa1b_dir = f"{ROOT}/data/SA-1B"
    if not os.path.isdir(sa1b_dir):
        return []
    tars = sorted([f for f in os.listdir(sa1b_dir) if f.endswith('.tar')])
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    images = []
    for tarname in tars:
        if len(images) >= max_images:
            break
        tarpath = os.path.join(sa1b_dir, tarname)
        try:
            with tarfile.open(tarpath) as tar:
                for member in tar.getmembers():
                    if len(images) >= max_images:
                        break
                    if member.isfile() and member.name.endswith('.jpg'):
                        f = tar.extractfile(member)
                        if f:
                            img = Image.open(io.BytesIO(f.read())).convert('RGB')
                            images.append(transform(img) * 255)
        except Exception as e:
            print(f"  SA-1B error reading {tarname}: {e}")
            continue
    return images


# ============================================================
# SRS DISTILLATION
# ============================================================
def distill_srs_from_slices(source_sam, slices, rank=8):
    gradient_list = []
    for idx, slc in enumerate(slices):
        img_t = slc.unsqueeze(0).float().to(DEVICE)
        if img_t.shape[1] == 1:
            img_t = img_t.repeat(1, 3, 1, 1)
        with torch.no_grad():
            targets, targets_layer = source_sam.image_encoder(img_t)
        from torch.autograd import Variable
        perturb_img = Variable(img_t.data + torch.randn_like(img_t) * 1e-5, requires_grad=True)
        local_grads = []
        for step_i in range(3):
            source_sam.zero_grad()
            features, features_layer = source_sam.image_encoder(perturb_img)
            loss_org = F.l1_loss(features * 0.9999, targets)
            pre_layer = int(len(features_layer) * 0.9)
            num_layers = min(6, pre_layer)
            idx_i = num_layers - 1
            F_clean_i = targets_layer[idx_i + pre_layer - num_layers]
            F_adv_i = features_layer[idx_i + pre_layer - num_layers]
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
            print(f"    [SRS] {idx+1}/{len(slices)} gradients")
    if not gradient_list:
        return None
    subspace = fit_shared_subspace(gradient_list, rank=rank)
    return subspace


# ============================================================
# ATTACK WITH SRS
# ============================================================
def run_attack_with_srs(source_sam, img_t, config, srs_bundle, use_srs, srs_mean_weight=0.0):
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


def get_synapse_test_slices(n_cases):
    import h5py
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:n_cases]
    all_slices = []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:6]
        for s in valid:
            all_slices.append((img[s].astype(np.float32), lbl[s], h5f, s))
    return all_slices


# ============================================================
# EXP 8: PROXY DATA TYPE (extended)
# ============================================================
def run_exp8_proxy_type():
    print("="*60)
    print("Experiment 8: Proxy Data Type (Extended)")
    print("="*60)

    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    target = load_samed_model(VIT, SAM_CKPT, SYN_CKPT, IMAGE_SIZE, 8, device=DEVICE)

    test_slices = get_synapse_test_slices(8)
    print(f"  Loaded {len(test_slices)} test slices")

    proxy_sources = {
        "Synapse": get_synapse_train_slices(N_PROXY),
        "ISIC": get_isic_train_images(N_PROXY),
        "REFUGE": get_refuge_train_images(N_PROXY),
        "SA-1B": get_sa1b_images(N_PROXY),
    }

    exp8_results = {}
    for src_name, proxy_slices in proxy_sources.items():
        if not proxy_slices:
            print(f"    SKIP {src_name}: no proxy data")
            exp8_results[src_name] = {"error": "no proxy data available"}
            continue

        print(f"    {src_name}: {len(proxy_slices)} proxy images")

        config = AttackConfig(
            att_step_size=2.0, att_bound=10.0, att_numstep=10,
            direction_mode="radial", probe_amplification=2.0,
            edge_guidance=True, seed=42,
        )
        proxy_bundle = distill_srs_from_slices(source_sam, proxy_slices, rank=SRS_RANK)

        deltas = []
        for slc, lbl_slice, h5f, si in test_slices[:N_TEST_EXP8]:
            clean_d = evaluate_samed_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
            slc_3ch = np.stack([slc]*3, axis=0)
            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)
            local_config = AttackConfig(
                att_step_size=2.0, att_bound=10.0, att_numstep=10,
                direction_mode="radial", probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            adv_np = run_attack_with_srs(source_sam, img_t, local_config,
                                         proxy_bundle, True, srs_mean_weight=0.1)
            adv_d = evaluate_samed_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
            deltas.append(clean_d - adv_d)

        exp8_results[src_name] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "n_proxy": len(proxy_slices),
            "n_test": len(deltas),
        }
        print(f"      Δ={exp8_results[src_name]['mean_delta']:.4f} ± {exp8_results[src_name]['std_delta']:.4f}")

    # Also test no_warmstart baseline
    deltas_no_ws = []
    for slc, lbl_slice, h5f, si in test_slices[:N_TEST_EXP8]:
        clean_d = evaluate_samed_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
        slc_3ch = np.stack([slc]*3, axis=0)
        img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)
        local_config = AttackConfig(
            att_step_size=2.0, att_bound=10.0, att_numstep=10,
            direction_mode="radial", probe_amplification=2.0,
            edge_guidance=True, seed=42,
        )
        adv_np, _ = run_codd_attack(source_sam.image_encoder, img_t, local_config, return_trace=True)
        adv_d = evaluate_samed_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
        deltas_no_ws.append(clean_d - adv_d)

    exp8_results["no_warmstart"] = {
        "mean_delta": float(np.mean(deltas_no_ws)),
        "std_delta": float(np.std(deltas_no_ws)),
        "n_proxy": 0,
        "n_test": len(deltas_no_ws),
    }
    print(f"    no_warmstart: Δ={exp8_results['no_warmstart']['mean_delta']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "exp8_proxy_type_extended.json"), "w") as f:
        json.dump({"config": {"n_proxy": N_PROXY, "n_test": N_TEST_EXP8}, "results": exp8_results}, f, indent=2)

    print("Exp 8 DONE\n")
    del source_sam, target; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# EXP 11: COMPUTATIONAL COST (FULL)
# ============================================================
def _pgd_attack(image_encoder, input_image, epsilon, step_size, steps):
    """Standard PGD attack with L1 feature loss."""
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    perturb_img = perturb_img + torch.randn_like(perturb_img) * 1e-5
    for _ in range(steps):
        perturb_img = perturb_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(perturb_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, perturb_img)[0]
        perturb_img = perturb_img.detach() + step_size * grad.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _mi_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, decay=1.0):
    """MI-FGSM attack."""
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    momentum = torch.zeros_like(input_image)
    for _ in range(steps):
        perturb_img = perturb_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(perturb_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, perturb_img)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = decay * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _ni_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, decay=1.0):
    """NI-FGSM (Nesterov) attack."""
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    momentum = torch.zeros_like(input_image)
    for _ in range(steps):
        # Nesterov: look ahead
        nes_img = perturb_img.detach() + decay * step_size * momentum
        nes_img = nes_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(nes_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, nes_img)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = decay * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _di_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, prob=0.5):
    """DI-FGSM with input diversity (random resize)."""
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    momentum = torch.zeros_like(input_image)
    H, W = input_image.shape[2], input_image.shape[3]
    for _ in range(steps):
        grad_img = perturb_img.clone().detach().requires_grad_(True)
        # Input diversity: random resize
        if random.random() < prob:
            r = random.randint(int(0.9 * H), H)
            grad_img_resized = F.interpolate(grad_img, size=(r, r), mode='bilinear')
            grad_img_resized = F.interpolate(grad_img_resized, size=(H, W), mode='bilinear')
        else:
            grad_img_resized = grad_img
        features, _ = image_encoder(grad_img_resized)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, grad_img)[0]
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 1.0 * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _ti_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, kernel_size=5):
    """TI-FGSM with Gaussian kernel smoothing."""
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    momentum = torch.zeros_like(input_image)
    # Gaussian kernel
    ks = kernel_size
    k = torch.zeros(1, 1, ks, ks)
    for i in range(ks):
        for j in range(ks):
            k[0, 0, i, j] = (1.0 / (ks * ks))
    if input_image.shape[1] == 3:
        k = k.repeat(3, 1, 1, 1)
    k = k.to(input_image.device)
    for _ in range(steps):
        perturb_img = perturb_img.clone().detach().requires_grad_(True)
        features, _ = image_encoder(perturb_img)
        loss = criterion_l1(features, targets)
        grad = torch.autograd.grad(loss, perturb_img)[0]
        # Convolve gradient with Gaussian kernel
        grad = F.conv2d(grad, k, padding=ks//2, groups=3)
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 1.0 * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def _si_fgsm_attack(image_encoder, input_image, epsilon, step_size, steps, num_scales=3):
    """SI-FGSM with scale invariance — resize back to IMAGE_SIZE for SAM encoder."""
    max_val, min_val = input_image.max(), input_image.min()
    with torch.no_grad():
        targets, _ = image_encoder(input_image)
    perturb_img = input_image.clone().detach()
    momentum = torch.zeros_like(input_image)
    H, W = input_image.shape[2], input_image.shape[3]
    for _ in range(steps):
        total_grad = torch.zeros_like(input_image)
        for s in range(num_scales):
            scale = 1.0 / (2 ** s)
            # Scale down then back up to preserve SAM-compatible input size
            scaled = F.interpolate(perturb_img.detach(), scale_factor=scale, mode='bilinear')
            scaled = F.interpolate(scaled, size=(H, W), mode='bilinear')
            scaled = scaled.clone().detach().requires_grad_(True)
            features, _ = image_encoder(scaled)
            loss = criterion_l1(features, targets)
            grad = torch.autograd.grad(loss, scaled)[0]
            total_grad += grad
        grad = total_grad / num_scales
        grad = grad / (torch.norm(grad, p=1) + 1e-12)
        momentum = 1.0 * momentum + grad
        perturb_img = perturb_img.detach() + step_size * momentum.sign()
        eta = torch.clamp(perturb_img - input_image, -epsilon, epsilon)
        perturb_img = torch.clamp(input_image + eta, min_val, max_val).detach()
    return perturb_img.detach()


def run_exp11_computational_cost():
    print("="*60)
    print("Experiment 11: Computational Cost (Full)")
    print("="*60)

    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)

    # Create a fixed test image
    torch.manual_seed(42)
    test_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE) * 50 + 128
    test_img = torch.clamp(test_img, 0, 255)

    EPSILON = 10.0
    STEP_SIZE = 2.0
    STEPS = 10
    max_val, min_val = test_img.max(), test_img.min()
    eps_pix = float((max_val - min_val) * EPSILON / 255.0)
    step_pix = float((max_val - min_val) * STEP_SIZE / 255.0)

    WARMUP = 3
    BENCH_RUNS = 10

    methods = {
        "PGD": lambda: _pgd_attack(source_sam.image_encoder, test_img, eps_pix, step_pix, STEPS),
        "MI-FGSM": lambda: _mi_fgsm_attack(source_sam.image_encoder, test_img, eps_pix, step_pix, STEPS),
        "NI-FGSM": lambda: _ni_fgsm_attack(source_sam.image_encoder, test_img, eps_pix, step_pix, STEPS),
        "DI-FGSM": lambda: _di_fgsm_attack(source_sam.image_encoder, test_img, eps_pix, step_pix, STEPS),
        "TI-FGSM": lambda: _ti_fgsm_attack(source_sam.image_encoder, test_img, eps_pix, step_pix, STEPS),
        "SI-FGSM": lambda: _si_fgsm_attack(source_sam.image_encoder, test_img, eps_pix, step_pix, STEPS),
    }

    results = {}

    for method_name, attack_fn in methods.items():
        print(f"  Benchmarking {method_name}...")

        # Warmup
        for _ in range(WARMUP):
            _ = attack_fn()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(DEVICE)

        times = []
        for _ in range(BENCH_RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = attack_fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)

        results[method_name] = {
            "mean_time_s": float(np.mean(times)),
            "std_time_s": float(np.std(times)),
            "peak_mem_gb": float(peak_mem),
            "steps": STEPS,
        }
        print(f"    {results[method_name]['mean_time_s']:.4f}s ± {results[method_name]['std_time_s']:.4f}s, {results[method_name]['peak_mem_gb']:.2f}GB")
        gc.collect()
        torch.cuda.empty_cache()

    # SE-CGR variants
    print("  Benchmarking SE-CGR (no SRS)...")
    for _ in range(WARMUP):
        config = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=10, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, config)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    times = []
    for _ in range(BENCH_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        config = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=10, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, config)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
    results["SE-CGR"] = {
        "mean_time_s": float(np.mean(times)),
        "std_time_s": float(np.std(times)),
        "peak_mem_gb": float(peak_mem),
        "steps": 10,
    }
    print(f"    SE-CGR: {results['SE-CGR']['mean_time_s']:.4f}s ± {results['SE-CGR']['std_time_s']:.4f}s, {results['SE-CGR']['peak_mem_gb']:.2f}GB")

    # SE-CGR 20 steps
    print("  Benchmarking SE-CGR (20 steps)...")
    for _ in range(WARMUP):
        config = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=20, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    times = []
    for _ in range(BENCH_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        config = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=20, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, config)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
    results["SE-CGR (20 steps)"] = {
        "mean_time_s": float(np.mean(times)),
        "std_time_s": float(np.std(times)),
        "peak_mem_gb": float(peak_mem),
        "steps": 20,
    }
    print(f"    SE-CGR 20st: {results['SE-CGR (20 steps)']['mean_time_s']:.4f}s ± {results['SE-CGR (20 steps)']['std_time_s']:.4f}s, {results['SE-CGR (20 steps)']['peak_mem_gb']:.2f}GB")

    with open(os.path.join(OUTPUT_DIR, "exp11_cost_full.json"), "w") as f:
        json.dump({
            "config": {"vit": VIT, "image_size": IMAGE_SIZE, "epsilon": EPSILON,
                       "steps": STEPS, "warmup": WARMUP, "bench_runs": BENCH_RUNS},
            "results": results
        }, f, indent=2)

    print("Exp 11 DONE\n")
    del source_sam; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# EXP 13: QUALITATIVE VISUALIZATIONS
# ============================================================
def run_exp13_qualitative():
    print("="*60)
    print("Experiment 13: Qualitative Visualizations")
    print("="*60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from scipy.ndimage import zoom
    from utils import calculate_metric_percase

    # Load Synapse models
    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    target_syn = load_samed_model(VIT, SAM_CKPT, SYN_CKPT, IMAGE_SIZE, 8, device=DEVICE)

    # ---- Synapse cases ----
    print("  Processing Synapse cases...")
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:4]

    syn_cases = []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        # Find slices with most labels
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 500],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)
        if len(valid) >= 2:
            # Success case: highest label count
            syn_cases.append(("Synapse", h5f, img[valid[0]], lbl[valid[0]], "success"))
            # Edge case: lowest label count
            syn_cases.append(("Synapse", h5f, img[valid[-1]], lbl[valid[-1]], "edge"))
            break

    print(f"    Found {len(syn_cases)} Synapse cases")

    # Run attack and generate visualizations for each case
    all_vis_data = []
    for dataset, case_id, slc_img, slc_lbl, case_type in syn_cases[:2]:
        print(f"    Processing {dataset}/{case_id} ({case_type})...")

        # Prepare image
        slc_3ch = np.stack([slc_img.astype(np.float32)]*3, axis=0)
        img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

        # Run attack
        config = AttackConfig(
            att_step_size=2.0, att_bound=10.0, att_numstep=20,
            direction_mode="radial", probe_amplification=2.0,
            edge_guidance=True, seed=42,
        )
        adv_np, trace = run_codd_attack(source_sam.image_encoder, img_t, config, return_trace=True)

        # Get predictions
        target_syn.eval()
        with torch.no_grad():
            clean_out = target_syn(img_t, True, IMAGE_SIZE)
            clean_pred = torch.argmax(torch.softmax(clean_out['masks'], dim=1), dim=1).squeeze(0).cpu().numpy()

            adv_t = torch.from_numpy(adv_np).unsqueeze(0).float().to(DEVICE)
            adv_out = target_syn(adv_t, True, IMAGE_SIZE)
            adv_pred = torch.argmax(torch.softmax(adv_out['masks'], dim=1), dim=1).squeeze(0).cpu().numpy()

        # Perturbation
        pert = adv_np[0, 0] - slc_img.astype(np.float32)

        all_vis_data.append({
            "dataset": dataset,
            "case_id": str(case_id),
            "case_type": case_type,
            "clean_img": slc_img,
            "adv_img": adv_np[0, 0],
            "pert": pert,
            "clean_pred": clean_pred,
            "adv_pred": adv_pred,
            "gt": slc_lbl,
        })

    # Plot figure
    n_cases = len(all_vis_data)
    fig, axes = plt.subplots(n_cases, 6, figsize=(30, 5 * n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Clean Image", "Adv Image", "Perturbation (x5)", "Clean Pred", "Adv Pred", "Ground Truth"]

    for row_idx, data in enumerate(all_vis_data):
        images = [
            data["clean_img"],
            data["adv_img"],
            data["pert"] * 5,  # amplify for visibility
            data["clean_pred"],
            data["adv_pred"],
            data["gt"],
        ]
        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]
            if col_idx < 3:  # Image type
                im = ax.imshow(img, cmap='gray')
            else:  # Segmentation type
                im = ax.imshow(img, cmap='jet', vmin=0, vmax=8)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, fontweight='bold')
            ax.axis('off')
        axes[row_idx, 0].set_ylabel(f"{data['dataset']} ({data['case_type']})", fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "qualitative", "exp13_synapse.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved Synapse qualitative figure")

    # Save data
    vis_export = []
    for d in all_vis_data:
        vis_export.append({
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in d.items()
        })
    with open(os.path.join(OUTPUT_DIR, "qualitative", "exp13_data.json"), "w") as f:
        json.dump(vis_export, f, indent=2)

    print("Exp 13 DONE\n")
    del source_sam, target_syn; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "exp8" or mode == "all":
        run_exp8_proxy_type()
    if mode == "exp11" or mode == "all":
        run_exp11_computational_cost()
    if mode == "exp13" or mode == "all":
        run_exp13_qualitative()

    print("\n" + "="*60)
    print("ALL REMAINING EXPERIMENTS COMPLETE")
    print("="*60)
