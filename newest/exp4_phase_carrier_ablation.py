#!/usr/bin/env python3
"""
Experiment 4: Phase Carrier Ablation — full matrix.
Tests 5 conditions (SE-CGR, phase-shuffled, tangential, random-phase, no-probe)
across 8 targets with proper MSA evaluation for all datasets.

Conditions:
  SE-CGR:         radial direction + source phase carrier
  phase-shuffled: radial direction + spatially shuffled phase
  tangential:     tangential direction (orthogonal to source carrier)
  random-phase:   isotropic direction + random phase
  no-probe:       clean image (no attack)
"""
import os, sys, json, time, random, gc
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from attack_codd import AttackConfig, run_codd_attack
from evaluate import load_sam_encoder, load_samed_model, load_medsa_model

DEVICE = torch.device("cuda:0")
IMAGE_SIZE = 512

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAM_CKPT = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

CONDITIONS = ["se_cgr", "phase_shuffled", "tangential", "random_phase", "no_probe"]
DIRECTION_MAP = {
    "se_cgr": "radial",
    "phase_shuffled": "radial",
    "tangential": "tangential",
    "random_phase": "isotropic",
    "no_probe": "radial",
}

TARGETS = [
    # (tag, dataset, model_type, vit, ckpt, num_cls, mmo, n_cases)
    ("Synapse_SAMed_vb", "Synapse", "SAMed", "vit_b",
     f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8, 9, 12),
    ("Synapse_MSA_vb", "Synapse", "MSA", "vit_b",
     f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth", 8, 9, 8),
    ("ISIC_SAMed_vb", "ISIC", "SAMed", "vit_b",
     f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 1, 1, 12),
    ("ISIC_MSA_vb", "ISIC", "MSA", "vit_b",
     f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 1, 1, 8),
    ("REFUGE_SAMed_vb", "REFUGE", "SAMed", "vit_b",
     f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth", 2, 2, 12),
    ("REFUGE_MSA_vb", "REFUGE", "MSA", "vit_b",
     f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth", 2, 2, 8),
]

NUM_SYN_SLICES = 4


# ---------------------------------------------------------------------------
# SAMed evaluation
# ---------------------------------------------------------------------------

def evaluate_samed_slice(net, image_np, label_np, case_name):
    """Evaluate SAMed on one Synapse slice."""
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


def evaluate_samed_isic(net, image, label):
    """SAMed ISIC binary Dice with optimal threshold."""
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


def evaluate_samed_refuge(net, image, label):
    """SAMed REFUGE mean Dice (disc + cup)."""
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


# ---------------------------------------------------------------------------
# MSA evaluation
# ---------------------------------------------------------------------------

def evaluate_msa_isic(model, image_np, gt_mask):
    """
    MSA ISIC per-case evaluation.
    image_np: [H, W] grayscale numpy array (adv/clean image).
    gt_mask: [H, W] ground truth binary mask for THIS case.
    Returns best Dice across thresholds.
    """
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda

    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_uint8).convert('RGB')
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Lambda(lambda x: x * 255)])
    img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

    if torch.is_tensor(gt_mask):
        gt_mask = gt_mask.cpu().numpy()
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    indices = np.argwhere(gt_mask > 0)
    if len(indices) == 0:
        pt = torch.tensor([[[0.0, 0.0]]], device=DEVICE)
        pt_label = torch.tensor([[0]], dtype=torch.int, device=DEVICE)
    else:
        r, c = indices[random.randint(0, len(indices)-1)]
        pt = torch.tensor([[[float(c), float(r)]]], device=DEVICE)
        pt_label = torch.tensor([[1]], dtype=torch.int, device=DEVICE)

    with torch.no_grad():
        imgs_pre = model.preprocess(img_t)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(pt, pt_label), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    probs = torch.sigmoid(pred).squeeze().cpu().numpy()
    best_d = 0.0
    for th in np.arange(0.1, 1.0, 0.1):
        pb = (probs > th).astype(np.float32)
        d = 2.0 * (pb * gt_mask).sum() / (pb.sum() + gt_mask.sum() + 1e-8)
        if d > best_d: best_d = d
    return float(best_d)


def evaluate_msa_refuge(model, image_np, gt_label):
    """
    MSA REFUGE per-case evaluation.
    image_np: [H, W] grayscale numpy array.
    gt_label: [3, H, W] label array (cup/disc/bg channels).
    Returns avg Dice (disc + cup) / 2 for THIS case only.
    """
    from PIL import Image as PILImage
    from torchvision.transforms import Resize, Compose, ToTensor, Lambda

    img_uint8 = np.clip(image_np, 0, 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_uint8).convert('RGB')
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), ToTensor(), Lambda(lambda x: x * 255)])
    img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

    pt_coords = torch.tensor([[[0.0, 0.0]]], device=DEVICE)
    pt_labels = torch.tensor([[1]], dtype=torch.int, device=DEVICE)

    with torch.no_grad():
        imgs_pre = model.preprocess(img_t)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(pt_coords, pt_labels), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    probs = torch.sigmoid(pred[0])
    cup_pred = (probs[0] > 0.5).float().cpu().numpy()
    disc_pred = (probs[1] > 0.5).float().cpu().numpy()

    label_np = gt_label.cpu().numpy() if torch.is_tensor(gt_label) else gt_label
    disc_gt = (label_np[1] > 0.5).astype(np.float32)
    cup_gt = (label_np[0] > 0.5).astype(np.float32)

    d_disc = 2.0 * (disc_pred * disc_gt).sum() / (disc_pred.sum() + disc_gt.sum() + 1e-8)
    d_cup = 2.0 * (cup_pred * cup_gt).sum() / (cup_pred.sum() + cup_gt.sum() + 1e-8)
    return float((d_disc + d_cup) / 2.0)


def evaluate_msa_synapse_slice(model, image_np, label_slice):
    """
    MSA Synapse single slice evaluation with click prompts.
    image_np: [H, W] grayscale, label_slice: [H, W]
    """
    img_3ch = np.stack([image_np]*3, axis=0)
    img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(DEVICE)
    lbl_t = torch.from_numpy(label_slice.copy()).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # Generate click prompt for each class
    pt_list, lbl_list = [], []
    for c in range(1, 9):  # classes 1-8
        class_mask = (label_slice == c).astype(np.uint8)
        indices = np.argwhere(class_mask > 0)
        if len(indices) > 0:
            r, cc = indices[random.randint(0, len(indices)-1)]
            pt_list.append(np.array([cc, r]))
            lbl_list.append(1)
        else:
            pt_list.append(np.array([0, 0]))
            lbl_list.append(0)
    coords = torch.tensor(np.stack(pt_list, axis=0), dtype=torch.float, device=DEVICE).unsqueeze(0)
    labels = torch.tensor(lbl_list, dtype=torch.int, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        imgs_pre = model.preprocess(img_t)
        image_emb = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )
        pred = F.interpolate(pred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    pred_classes = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred_classes == c).astype(np.float32)
        tc = (label_slice == c).astype(np.float32)
        inter = (pc * tc).sum()
        denom = pc.sum() + tc.sum()
        dices.append(2.0 * inter / (denom + 1e-8) if denom > 0 else 1.0)
    return float(np.mean(dices))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def get_isic_samples(n_cases):
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    df = pd.read_csv(os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"))
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    samples = []
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, row['img'])
        gt_path = os.path.join(data_dir, row['seg'])
        img = Image.open(img_path).convert('RGB')
        img_raw = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))  # [H, W, 3] uint8
        gt = np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32) / 255.0
        samples.append((transform(img), (gt > 0.5).astype(np.float32), img_raw))
        if len(samples) >= n_cases: break
    return samples


def get_refuge_samples(n_cases):
    from PIL import Image
    from torchvision import transforms
    data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    samples = []
    for folder in folders:
        img_path = os.path.join(data_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(data_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(data_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img)
        img_raw = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))  # [H, W, 3] uint8
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        label = np.stack([
            (cup_raw < 128).astype(np.float32),       # cup
            (disc_raw <= 128).astype(np.float32),     # disc
            1.0 - (disc_raw <= 128).astype(np.float32) - (cup_raw < 128).astype(np.float32),  # bg
        ], axis=0)
        samples.append((img_t, label, img_raw))
        if len(samples) >= n_cases: break
    return samples


def normalized_gray_to_uint8(gray_normalized):
    """Convert ImageNet-normalized grayscale back to uint8.
    gray_normalized: [H, W] or [3, H, W]."""
    if gray_normalized.ndim == 3 and gray_normalized.shape[0] == 3:
        gray_normalized = gray_normalized[0]
    MEAN_AVG = (0.485 + 0.456 + 0.406) / 3.0  # 0.449
    STD_AVG = (0.229 + 0.224 + 0.225) / 3.0   # 0.226
    return np.clip((gray_normalized * STD_AVG + MEAN_AVG) * 255, 0, 255).astype(np.float32)


def run_2d_experiment(tag, vit, ckpt, model_type, dataset, n_cases):
    """Phase carrier ablation on ISIC/REFUGE."""
    print(f"\n{'='*60}\n{tag} (vit={vit}, n={n_cases})\n{'='*60}")

    source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, DEVICE)

    if model_type == "SAMed":
        num_cls = 1 if dataset == "ISIC" else 2
        target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, num_cls, device=DEVICE)
        eval_fn = evaluate_samed_isic if dataset == "ISIC" else evaluate_samed_refuge
    else:
        mmo = 1 if dataset == "ISIC" else 2
        target = load_medsa_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, mmo, DEVICE)
        eval_fn = evaluate_msa_isic if dataset == "ISIC" else evaluate_msa_refuge

    samples = get_isic_samples(n_cases) if dataset == "ISIC" else get_refuge_samples(n_cases)

    results = {cond: [] for cond in CONDITIONS}

    for case_idx, (img_t, label, img_raw) in enumerate(samples):
        print(f"  [{tag}] Case {case_idx+1}/{n_cases}")

        # Clean dice
        if model_type == "SAMed":
            clean_d = eval_fn(target, img_t, torch.from_numpy(label).float() if not torch.is_tensor(label) else label)
        else:
            # MSA: use raw uint8 image directly, grayscale
            img_uint8_np = img_raw.mean(axis=2).astype(np.float32)
            label_t = torch.from_numpy(label).float() if not torch.is_tensor(label) else label
            clean_d = eval_fn(target, img_uint8_np, label_t)

        # Prepare attack input (grayscale 3-channel, normalized)
        img_gray = img_t.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        img_tensor = img_gray.unsqueeze(0).float().to(DEVICE)

        for cond in CONDITIONS:
            if cond == "no_probe":
                results[cond].append({"case": case_idx, "dice": clean_d, "delta": 0.0})
                continue

            # Prepare modified image for phase-shuffled / random-phase conditions
            if cond == "phase_shuffled":
                G = torch.fft.fft2(img_tensor, norm="ortho")
                mag = torch.abs(G)
                phase = torch.angle(G)
                B, C, H, W = phase.shape
                phase_shuffled = phase.view(B, C, -1)[:, :, torch.randperm(H*W)].view_as(phase)
                G_shuffled = mag * torch.exp(1j * phase_shuffled)
                img_for_attack = torch.fft.ifft2(G_shuffled, norm="ortho").real
            elif cond == "random_phase":
                G = torch.fft.fft2(img_tensor, norm="ortho")
                mag = torch.abs(G)
                rand_phase = torch.rand_like(G) * 2 * np.pi
                G_rand = mag * torch.exp(1j * rand_phase)
                img_for_attack = torch.fft.ifft2(G_rand, norm="ortho").real
            else:
                img_for_attack = img_tensor

            config = AttackConfig(
                att_step_size=2.0, att_bound=10.0, att_numstep=10,
                direction_mode=DIRECTION_MAP[cond],
                probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )

            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            adv_np, _ = run_codd_attack(
                source_sam.image_encoder, img_for_attack, config, return_trace=True,
            )

            # Evaluate adversarial image
            if model_type == "SAMed":
                adv_gray = torch.from_numpy(adv_np.copy()).float()  # [3, H, W]
                adv_d = eval_fn(target, adv_gray, label)
            else:
                # MSA: convert normalized grayscale back to uint8
                adv_uint8 = normalized_gray_to_uint8(adv_np[0].copy())
                label_t = torch.from_numpy(label).float() if not torch.is_tensor(label) else label
                adv_d = eval_fn(target, adv_uint8, label_t)

            delta = clean_d - adv_d
            results[cond].append({"case": case_idx, "dice": float(adv_d), "delta": float(delta), "clean_dice": float(clean_d)})

    # Summary
    summary = {}
    for cond in CONDITIONS:
        deltas = [r["delta"] for r in results[cond]]
        summary[cond] = {
            "mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)),
            "mean_dice": float(np.mean([r["dice"] for r in results[cond]])),
            "n": len(deltas),
        }
    print(f"  [{tag}] Summary: " + " | ".join(f"{c}={summary[c]['mean_delta']:.4f}" for c in CONDITIONS))

    del source_sam, target; gc.collect(); torch.cuda.empty_cache()
    return summary, results


def run_synapse_experiment(tag, vit, ckpt, model_type, n_cases):
    """Phase carrier ablation on Synapse."""
    import h5py

    source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, DEVICE)

    if model_type == "SAMed":
        target = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, 8, device=DEVICE)
    else:
        target = load_medsa_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, 9, DEVICE)

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

        print(f"  [{tag}] Case {case_idx+1}/{n_cases}: {h5f} ({len(organ_slices)} slices)")

        for si in organ_slices:
            slc = img[si].astype(np.float32)
            label_slice = lbl[si]

            # Clean dice
            if model_type == "SAMed":
                clean_d = evaluate_samed_slice(target, slc, label_slice, f"{h5f}_s{si}")
            else:
                clean_d = evaluate_msa_synapse_slice(target, slc, label_slice)

            slc_3ch = np.stack([slc]*3, axis=0)
            img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)

            for cond in CONDITIONS:
                if cond == "no_probe":
                    results[cond].append({"case": str(h5f), "slice": int(si), "dice": clean_d, "delta": 0.0})
                    continue

                if cond == "phase_shuffled":
                    G = torch.fft.fft2(img_t, norm="ortho")
                    mag = torch.abs(G)
                    phase = torch.angle(G)
                    B, C, H, W = phase.shape
                    phase_shuffled = phase.view(B, C, -1)[:, :, torch.randperm(H*W)].view_as(phase)
                    G_shuffled = mag * torch.exp(1j * phase_shuffled)
                    img_for_attack = torch.fft.ifft2(G_shuffled, norm="ortho").real
                elif cond == "random_phase":
                    G = torch.fft.fft2(img_t, norm="ortho")
                    mag = torch.abs(G)
                    rand_phase = torch.rand_like(G) * 2 * np.pi
                    G_rand = mag * torch.exp(1j * rand_phase)
                    img_for_attack = torch.fft.ifft2(G_rand, norm="ortho").real
                else:
                    img_for_attack = img_t

                config = AttackConfig(
                    att_step_size=2.0, att_bound=10.0, att_numstep=10,
                    direction_mode=DIRECTION_MAP[cond],
                    probe_amplification=2.0,
                    edge_guidance=True, seed=42,
                )

                random.seed(42); np.random.seed(42); torch.manual_seed(42)
                adv_np, _ = run_codd_attack(
                    source_sam.image_encoder, img_for_attack, config, return_trace=True,
                )

                adv_gray = adv_np[0].copy()
                if model_type == "SAMed":
                    adv_d = evaluate_samed_slice(target, adv_gray, label_slice, f"{h5f}_s{si}")
                else:
                    adv_d = evaluate_msa_synapse_slice(target, adv_gray, label_slice)

                delta = clean_d - adv_d
                results[cond].append({"case": str(h5f), "slice": int(si), "dice": float(adv_d), "delta": float(delta), "clean_dice": float(clean_d)})

    summary = {}
    for cond in CONDITIONS:
        deltas = [r["delta"] for r in results[cond]]
        summary[cond] = {
            "mean_delta": float(np.mean(deltas)), "std_delta": float(np.std(deltas)),
            "mean_dice": float(np.mean([r["dice"] for r in results[cond]])),
            "n": len(deltas),
        }
    print(f"  [{tag}] Summary: " + " | ".join(f"{c}={summary[c]['mean_delta']:.4f}" for c in CONDITIONS))

    del source_sam, target; gc.collect(); torch.cuda.empty_cache()
    return summary, results


def main():
    all_summaries = {}
    # Load already-computed Synapse results
    partial_path = os.path.join(OUTPUT_DIR, "exp4_synapse_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_summaries = json.load(f)
        print(f"Loaded {len(all_summaries)} completed results from {partial_path}")

    for tag, dataset, model_type, vit, ckpt, num_cls, mmo, n_cases in TARGETS:
        if tag in all_summaries:
            print(f"SKIP {tag}: already completed")
            continue
        if not os.path.exists(ckpt):
            print(f"SKIP {tag}: checkpoint not found")
            continue

        try:
            if dataset == "Synapse":
                summary, _ = run_synapse_experiment(tag, vit, ckpt, model_type, n_cases)
            else:
                summary, _ = run_2d_experiment(tag, vit, ckpt, model_type, dataset, n_cases)

            all_summaries[tag] = summary

            # Save intermediate results after each target
            output = {"config": {"conditions": CONDITIONS}, "results": all_summaries}
            out_path = os.path.join(OUTPUT_DIR, "exp4_phase_carrier_ablation.json")
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            # Also update the partial file
            with open(partial_path, "w") as f:
                json.dump(all_summaries, f, indent=2)
            print(f"  [SAVED] {tag} → {out_path} ({len(all_summaries)}/{len(TARGETS)} targets)")

        except Exception as e:
            print(f"  [ERROR] {tag}: {e}")
            import traceback
            traceback.print_exc()
            # Save what we have so far
            output = {"config": {"conditions": CONDITIONS}, "results": all_summaries}
            out_path = os.path.join(OUTPUT_DIR, "exp4_phase_carrier_ablation.json")
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            with open(partial_path, "w") as f:
                json.dump(all_summaries, f, indent=2)
            print(f"  [SAVED partial after error] {out_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()
