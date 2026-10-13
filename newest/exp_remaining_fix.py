#!/usr/bin/env python3
"""Fix remaining experiment gaps: ISIC proxy for Exp 8, full Exp 11, Exp 13."""
import os, sys, json, time, random, gc, tarfile, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack, _get_edge_map, synthesize_codd_probe
from theory import fit_shared_subspace
from evaluate import load_sam_encoder, load_samed_model

DEVICE = torch.device("cuda:3")
IMAGE_SIZE = 512
OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "qualitative"), exist_ok=True)

VIT = "vit_b"
SAM_CKPT = f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth"
SYN_CKPT = f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth"

criterion_l1 = nn.L1Loss()


# ---- Safe evaluation ----
def evaluate_samed_slice(net, image_np, label_np, case_name):
    from scipy.ndimage import zoom
    from utils import calculate_metric_percase
    device = next(net.parameters()).device
    if image_np.ndim == 2: image_np = image_np[np.newaxis, :, :]
    if label_np.ndim == 2: label_np = label_np[np.newaxis, :, :]
    prediction = np.zeros_like(label_np)
    for ind in range(image_np.shape[0]):
        slc = image_np[ind, :, :]
        x, y = slc.shape
        if x != IMAGE_SIZE or y != IMAGE_SIZE:
            slc = zoom(slc, (IMAGE_SIZE / x, IMAGE_SIZE / y), order=3)
        inputs = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().to(device)
        inputs = inputs.repeat(1, 3, 1, 1)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, True, IMAGE_SIZE)
            out = torch.argmax(torch.softmax(outputs['masks'], dim=1), dim=1).squeeze(0).cpu().detach().numpy()
            if x != out.shape[0]:
                pred = zoom(out, (x / out.shape[0], y / out.shape[1]), order=0)
            else:
                pred = out
            prediction[ind] = pred
    metric_list = [calculate_metric_percase(prediction == i, label_np == i) for i in range(1, 9)]
    return float(np.mean(metric_list, axis=0)[0])


# ---- ISIC proxy loader (fixed path) ----
def get_isic_train_images(max_images=200):
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
        # CSV 'img' column has path relative to data_dir
        img_path = os.path.join(data_dir, str(row['img']))
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img) * 255)
    print(f"  Loaded {len(images)} ISIC images")
    return images


# ---- SRS distillation ----
def distill_srs_from_slices(source_sam, slices, rank=8):
    from torch.autograd import Variable
    gradient_list = []
    for idx, slc in enumerate(slices):
        img_t = slc.unsqueeze(0).float().to(DEVICE)
        if img_t.shape[1] == 1: img_t = img_t.repeat(1, 3, 1, 1)
        with torch.no_grad():
            targets, targets_layer = source_sam.image_encoder(img_t)
        perturb_img = Variable(img_t.data + torch.randn_like(img_t) * 1e-5, requires_grad=True)
        local_grads = []
        for step_i in range(3):
            source_sam.zero_grad()
            features, features_layer = source_sam.image_encoder(perturb_img)
            loss_org = F.l1_loss(features * 0.9999, targets)
            pre_layer = int(len(features_layer) * 0.9)
            num_layers = min(6, pre_layer)
            idx_i = num_layers - 1
            probe = synthesize_codd_probe(
                targets_layer[idx_i + pre_layer - num_layers].unsqueeze(0),
                features_layer[idx_i + pre_layer - num_layers].unsqueeze(0),
                direction="radial", ads_rho=1.0, ads_gamma=1e-4,
            )
            loss = loss_org - (probe ** 2).mean()
            grad = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]
            local_grads.append(grad.detach() / (grad.detach().abs().sum() + 1e-12))
            eta = torch.clamp(perturb_img.data - img_t.data + 0.5 * local_grads[-1].sign(), -2.0, 2.0)
            perturb_img = Variable(torch.clamp(img_t.data + eta, img_t.min(), img_t.max()), requires_grad=True)
        gradient_list.append(torch.stack(local_grads, dim=0).mean(dim=0))
        if (idx + 1) % 32 == 0:
            print(f"    [SRS] {idx+1}/{len(slices)}")
    return fit_shared_subspace(gradient_list, rank=rank) if gradient_list else None


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
# FIX 1: ISIC proxy for Exp 8
# ============================================================
def fix_exp8_isic():
    print("="*60)
    print("FIX: Exp 8 ISIC Proxy Data")
    print("="*60)
    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    target = load_samed_model(VIT, SAM_CKPT, SYN_CKPT, IMAGE_SIZE, 8, device=DEVICE)

    test_slices = get_synapse_test_slices(8)
    isic_images = get_isic_train_images(128)
    if not isic_images:
        print("  ISIC images still empty — aborting")
        return

    config = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=10,
                          direction_mode="radial", probe_amplification=2.0,
                          edge_guidance=True, seed=42)
    proxy_bundle = distill_srs_from_slices(source_sam, isic_images, rank=8)

    deltas = []
    for slc, lbl_slice, h5f, si in test_slices[:8]:
        clean_d = evaluate_samed_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
        slc_3ch = np.stack([slc]*3, axis=0)
        img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)
        local_config = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=10,
                                    direction_mode="radial", probe_amplification=2.0,
                                    edge_guidance=True, seed=42,
                                    use_srs=True, srs_proj_weight=0.15, srs_mean_weight=0.1)
        adv_np, _ = run_codd_attack(source_sam.image_encoder, img_t, local_config,
                                     srs_bundle=proxy_bundle, return_trace=True)
        adv_d = evaluate_samed_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
        deltas.append(clean_d - adv_d)

    isic_result = {"mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)),
                    "n_proxy": len(isic_images), "n_test": len(deltas)}
    print(f"  ISIC proxy: Δ={isic_result['mean_delta']:.4f} ± {isic_result['std_delta']:.4f}")

    # Merge with existing exp8 results
    exp8_path = os.path.join(OUTPUT_DIR, "exp8_proxy_type_extended.json")
    if os.path.exists(exp8_path):
        with open(exp8_path) as f:
            data = json.load(f)
    else:
        data = {"config": {}, "results": {}}
    data["results"]["ISIC"] = isic_result
    with open(exp8_path, "w") as f:
        json.dump(data, f, indent=2)
    print("  ISIC result merged into exp8_proxy_type_extended.json")

    del source_sam, target; gc.collect(); torch.cuda.empty_cache()
    print("Exp 8 ISIC fix DONE\n")


# ============================================================
# FIX 2: Full Exp 11 Computational Cost
# ============================================================
def _pgd_attack(ie, img, eps, ss, steps):
    mx, mn = img.max(), img.min()
    with torch.no_grad(): targets, _ = ie(img)
    p = img.clone().detach() + torch.randn_like(img)*1e-5
    for _ in range(steps):
        p = p.clone().detach().requires_grad_(True)
        f, _ = ie(p); loss = criterion_l1(f, targets)
        g = torch.autograd.grad(loss, p)[0]
        p = p.detach() + ss * g.sign()
        p = torch.clamp(img + torch.clamp(p - img, -eps, eps), mn, mx).detach()
    return p.detach()


def _mi_fgsm_attack(ie, img, eps, ss, steps):
    mx, mn = img.max(), img.min()
    with torch.no_grad(): targets, _ = ie(img)
    p = img.clone().detach(); m = torch.zeros_like(img)
    for _ in range(steps):
        p = p.clone().detach().requires_grad_(True)
        f, _ = ie(p); loss = criterion_l1(f, targets)
        g = torch.autograd.grad(loss, p)[0]
        g = g / (torch.norm(g, p=1) + 1e-12)
        m = 1.0 * m + g
        p = p.detach() + ss * m.sign()
        p = torch.clamp(img + torch.clamp(p - img, -eps, eps), mn, mx).detach()
    return p.detach()


def _ni_fgsm_attack(ie, img, eps, ss, steps):
    mx, mn = img.max(), img.min()
    with torch.no_grad(): targets, _ = ie(img)
    p = img.clone().detach(); m = torch.zeros_like(img)
    for _ in range(steps):
        nes = (p.detach() + 1.0 * ss * m).clone().detach().requires_grad_(True)
        f, _ = ie(nes); loss = criterion_l1(f, targets)
        g = torch.autograd.grad(loss, nes)[0]
        g = g / (torch.norm(g, p=1) + 1e-12)
        m = 1.0 * m + g
        p = p.detach() + ss * m.sign()
        p = torch.clamp(img + torch.clamp(p - img, -eps, eps), mn, mx).detach()
    return p.detach()


def _di_fgsm_attack(ie, img, eps, ss, steps):
    mx, mn = img.max(), img.min()
    with torch.no_grad(): targets, _ = ie(img)
    p = img.clone().detach(); m = torch.zeros_like(img)
    H, W = img.shape[2], img.shape[3]
    for _ in range(steps):
        di = p.clone().detach().requires_grad_(True)
        if random.random() < 0.5:
            r = random.randint(int(0.9*H), H)
            di2 = F.interpolate(di, size=(r,r), mode='bilinear')
            di2 = F.interpolate(di2, size=(H,W), mode='bilinear')
            f, _ = ie(di2)
            loss = criterion_l1(f, targets)
        else:
            f, _ = ie(di)
            loss = criterion_l1(f, targets)
        g = torch.autograd.grad(loss, di)[0]
        g = g / (torch.norm(g, p=1) + 1e-12)
        m = 1.0 * m + g
        p = p.detach() + ss * m.sign()
        p = torch.clamp(img + torch.clamp(p - img, -eps, eps), mn, mx).detach()
    return p.detach()


def _ti_fgsm_attack(ie, img, eps, ss, steps):
    mx, mn = img.max(), img.min()
    with torch.no_grad(): targets, _ = ie(img)
    p = img.clone().detach(); m = torch.zeros_like(img)
    ks = 5
    k = torch.ones(1, 1, ks, ks) / (ks*ks)
    if img.shape[1] == 3: k = k.repeat(3, 1, 1, 1)
    k = k.to(img.device)
    for _ in range(steps):
        p = p.clone().detach().requires_grad_(True)
        f, _ = ie(p); loss = criterion_l1(f, targets)
        g = torch.autograd.grad(loss, p)[0]
        g = F.conv2d(g, k, padding=ks//2, groups=3)
        g = g / (torch.norm(g, p=1) + 1e-12)
        m = 1.0 * m + g
        p = p.detach() + ss * m.sign()
        p = torch.clamp(img + torch.clamp(p - img, -eps, eps), mn, mx).detach()
    return p.detach()


def _si_fgsm_attack(ie, img, eps, ss, steps):
    """SI-FGSM — scale down then back up to maintain SAM input size."""
    mx, mn = img.max(), img.min()
    with torch.no_grad(): targets, _ = ie(img)
    p = img.clone().detach(); m = torch.zeros_like(img)
    H, W = img.shape[2], img.shape[3]
    for _ in range(steps):
        total_g = torch.zeros_like(img)
        for s in range(3):
            scale = 1.0 / (2**s)
            scaled = F.interpolate(p.detach(), scale_factor=scale, mode='bilinear')
            scaled = F.interpolate(scaled, size=(H,W), mode='bilinear')
            scaled = scaled.clone().detach().requires_grad_(True)
            f, _ = ie(scaled); loss = criterion_l1(f, targets)
            g = torch.autograd.grad(loss, scaled)[0]
            total_g += g
        g = total_g / 3.0
        g = g / (torch.norm(g, p=1) + 1e-12)
        m = 1.0 * m + g
        p = p.detach() + ss * m.sign()
        p = torch.clamp(img + torch.clamp(p - img, -eps, eps), mn, mx).detach()
    return p.detach()


def run_exp11_full():
    print("="*60)
    print("FIX: Exp 11 Computational Cost (Full)")
    print("="*60)

    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    torch.manual_seed(42)
    test_img = torch.clamp(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE) * 50 + 128, 0, 255)
    mx, mn = test_img.max(), test_img.min()
    eps_pix = float((mx - mn) * 10.0 / 255.0)
    step_pix = float((mx - mn) * 2.0 / 255.0)
    STEPS = 10
    WARMUP, BENCH = 3, 10

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
        for _ in range(WARMUP): _ = attack_fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        times = []
        for _ in range(BENCH):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = attack_fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        results[method_name] = {
            "mean_time_s": float(np.mean(times)), "std_time_s": float(np.std(times)),
            "peak_mem_gb": float(torch.cuda.max_memory_allocated(DEVICE)/(1024**3)), "steps": STEPS,
        }
        print(f"    {results[method_name]['mean_time_s']:.4f}s ± {results[method_name]['std_time_s']:.4f}s, {results[method_name]['peak_mem_gb']:.2f}GB")
        gc.collect(); torch.cuda.empty_cache()

    # SE-CGR (10 steps)
    print("  Benchmarking SE-CGR (10 steps)...")
    for _ in range(WARMUP):
        cfg = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=10, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, cfg)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(DEVICE)
    times = []
    for _ in range(BENCH):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cfg = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=10, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, cfg)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    results["SE-CGR"] = {"mean_time_s": float(np.mean(times)), "std_time_s": float(np.std(times)),
                          "peak_mem_gb": float(torch.cuda.max_memory_allocated(DEVICE)/(1024**3)), "steps": 10}
    print(f"    SE-CGR: {results['SE-CGR']['mean_time_s']:.4f}s, {results['SE-CGR']['peak_mem_gb']:.2f}GB")

    # SE-CGR (20 steps)
    print("  Benchmarking SE-CGR (20 steps)...")
    for _ in range(WARMUP):
        cfg = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=20, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, cfg)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(DEVICE)
    times = []
    for _ in range(BENCH):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cfg = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=20, direction_mode="radial", probe_amplification=2.0, edge_guidance=True)
        _ = run_codd_attack(source_sam.image_encoder, test_img, cfg)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    results["SE-CGR (20 steps)"] = {"mean_time_s": float(np.mean(times)), "std_time_s": float(np.std(times)),
                                     "peak_mem_gb": float(torch.cuda.max_memory_allocated(DEVICE)/(1024**3)), "steps": 20}
    print(f"    SE-CGR 20st: {results['SE-CGR (20 steps)']['mean_time_s']:.4f}s, {results['SE-CGR (20 steps)']['peak_mem_gb']:.2f}GB")

    with open(os.path.join(OUTPUT_DIR, "exp11_cost_full.json"), "w") as f:
        json.dump({"config": {"vit": VIT, "epsilon": 10.0, "steps": 10, "warmup": WARMUP, "bench_runs": BENCH},
                   "results": results}, f, indent=2)

    del source_sam; gc.collect(); torch.cuda.empty_cache()
    print("Exp 11 DONE\n")


# ============================================================
# FIX 3: Exp 13 Qualitative
# ============================================================
def run_exp13_full():
    print("="*60)
    print("FIX: Exp 13 Qualitative (Full) — PIL version")
    print("="*60)

    from PIL import Image as PILImage, ImageDraw, ImageFont
    import h5py

    source_sam = load_sam_encoder(VIT, SAM_CKPT, IMAGE_SIZE, 8, DEVICE)
    target_syn = load_samed_model(VIT, SAM_CKPT, SYN_CKPT, IMAGE_SIZE, 8, device=DEVICE)

    # Collect 2 Synapse cases
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:4]

    cases = []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 500],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)
        if len(valid) >= 2:
            cases.append(("Synapse", h5f, img[valid[0]], lbl[valid[0]], "success"))
            cases.append(("Synapse", h5f, img[valid[-1]], lbl[valid[-1]], "edge"))
            break

    print(f"  Processing {len(cases)} cases...")

    all_data = []
    for dataset, case_id, slc_img, slc_lbl, case_type in cases:
        print(f"    {dataset}/{case_id} ({case_type})...")
        slc_3ch = np.stack([slc_img.astype(np.float32)]*3, axis=0)
        img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

        cfg = AttackConfig(att_step_size=2.0, att_bound=10.0, att_numstep=20,
                           direction_mode="radial", probe_amplification=2.0,
                           edge_guidance=True, seed=42)
        adv_np, trace = run_codd_attack(source_sam.image_encoder, img_t, cfg, return_trace=True)

        target_syn.eval()
        with torch.no_grad():
            clean_out = target_syn(img_t, True, IMAGE_SIZE)
            clean_pred = torch.argmax(torch.softmax(clean_out['masks'], dim=1), dim=1).squeeze(0).cpu().numpy()
            adv_t = torch.from_numpy(adv_np).unsqueeze(0).float().to(DEVICE)
            adv_out = target_syn(adv_t, True, IMAGE_SIZE)
            adv_pred = torch.argmax(torch.softmax(adv_out['masks'], dim=1), dim=1).squeeze(0).cpu().numpy()

        pert = np.array(adv_np[0], dtype=np.float64) - np.array(slc_img, dtype=np.float64)

        all_data.append({
            "dataset": dataset, "case_id": str(case_id), "case_type": case_type,
            "clean_img": np.array(slc_img, dtype=np.float64),
            "adv_img": np.array(adv_np[0], dtype=np.float64),
            "pert": np.array(pert, dtype=np.float64),
            "clean_pred": np.array(clean_pred, dtype=np.int32),
            "adv_pred": np.array(adv_pred, dtype=np.int32),
            "gt": np.array(slc_lbl, dtype=np.int32),
            "max_pert": float(np.abs(pert).max()), "mean_pert": float(np.abs(pert).mean()),
        })

    # Build composite image using PIL
    def norm_to_uint8(arr):
        """Normalize array to 0-255 uint8."""
        arr = np.array(arr, dtype=np.float64)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr

    def seg_to_color(arr, num_classes=9):
        """Convert segmentation labels to color image using tab10 colormap."""
        # Simple color map for up to num_classes
        colors = [
            (0, 0, 0),        # 0 background
            (255, 0, 0),      # 1 red
            (0, 255, 0),      # 2 green
            (0, 0, 255),      # 3 blue
            (255, 255, 0),    # 4 yellow
            (255, 0, 255),    # 5 magenta
            (0, 255, 255),    # 6 cyan
            (128, 128, 0),    # 7 olive
            (128, 0, 128),    # 8 purple
        ]
        arr = np.array(arr, dtype=np.int32).clip(0, num_classes - 1)
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx, color in enumerate(colors[:num_classes]):
            rgb[arr == cls_idx] = color
        return rgb

    H = IMAGE_SIZE
    margin = 10
    header_h = 20
    label_w = 80
    tile_size = H
    n_cols = 6
    n_rows = len(all_data)

    col_labels = ["Clean", "Adv", "Pert x5", "Clean Pred", "Adv Pred", "GT"]

    total_w = label_w + n_cols * tile_size + (n_cols - 1) * margin
    total_h = header_h + n_rows * (tile_size + margin)

    canvas = PILImage.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Column headers
    for ci, label in enumerate(col_labels):
        x = label_w + ci * (tile_size + margin)
        draw.text((x + tile_size // 2 - 20, 2), label, fill=(0, 0, 0))

    for ri, data in enumerate(all_data):
        # Prepare images
        tiles = []
        # Clean image
        tiles.append(norm_to_uint8(data["clean_img"]))
        # Adv image
        tiles.append(norm_to_uint8(data["adv_img"]))
        # Perturbation (x5 amplified)
        pert_amp = data["pert"] * 5
        tiles.append(norm_to_uint8(pert_amp))
        # Clean prediction
        tiles.append(seg_to_color(data["clean_pred"]))
        # Adv prediction
        tiles.append(seg_to_color(data["adv_pred"]))
        # Ground truth
        tiles.append(seg_to_color(data["gt"]))

        # Row label
        y_center = header_h + ri * (tile_size + margin) + tile_size // 2
        draw.text((2, y_center - 8), f"{data['dataset']}\n{data['case_type']}", fill=(0, 0, 0))

        for ci, tile in enumerate(tiles):
            x = label_w + ci * (tile_size + margin)
            y = header_h + ri * (tile_size + margin)

            if tile.ndim == 2:
                pil_img = PILImage.fromarray(tile, mode='L').convert('RGB')
            elif tile.shape[2] == 3:
                pil_img = PILImage.fromarray(tile, mode='RGB')
            else:
                pil_img = PILImage.fromarray(tile)

            pil_img = pil_img.resize((tile_size, tile_size), PILImage.NEAREST)
            canvas.paste(pil_img, (x, y))

    out_path = os.path.join(OUTPUT_DIR, "qualitative", "exp13_synapse_full.png")
    canvas.save(out_path, quality=95)
    print(f"  Saved to {out_path} ({canvas.size})")

    # Also save individual case images for higher resolution
    for ri, data in enumerate(all_data):
        tiles = [
            norm_to_uint8(data["clean_img"]),
            norm_to_uint8(data["adv_img"]),
            norm_to_uint8(data["pert"] * 5),
            seg_to_color(data["clean_pred"]),
            seg_to_color(data["adv_pred"]),
            seg_to_color(data["gt"]),
        ]
        case_w = n_cols * tile_size + (n_cols - 1) * margin
        case_h = tile_size
        case_canvas = PILImage.new('RGB', (case_w, case_h), (255, 255, 255))
        for ci, tile in enumerate(tiles):
            x = ci * (tile_size + margin)
            if tile.ndim == 2:
                pil_img = PILImage.fromarray(tile, mode='L').convert('RGB')
            else:
                pil_img = PILImage.fromarray(tile, mode='RGB')
            pil_img = pil_img.resize((tile_size, tile_size), PILImage.NEAREST)
            case_canvas.paste(pil_img, (x, 0))
        case_path = os.path.join(OUTPUT_DIR, "qualitative", f"exp13_{data['dataset']}_{data['case_type']}.png")
        case_canvas.save(case_path, quality=95)

    # Export data
    vis_export = []
    for d in all_data:
        vis_export.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()})
    with open(os.path.join(OUTPUT_DIR, "qualitative", "exp13_data_full.json"), "w") as f:
        json.dump(vis_export, f, indent=2)

    del source_sam, target_syn; gc.collect(); torch.cuda.empty_cache()
    print("Exp 13 DONE\n")


# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("exp8", "all"): fix_exp8_isic()
    if mode in ("exp11", "all"): run_exp11_full()
    if mode in ("exp13", "all"): run_exp13_full()
    print("\nALL FIXES COMPLETE")
