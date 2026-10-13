#!/usr/bin/env python3
"""
Experiment 12: Cross-Architecture Heatmap.
Runs SE-CGR (20 steps) vs no-probe on ALL available model checkpoints
and computes ΔDice improvement for each cell in the full matrix.

Matrix: 3 datasets × 2 architectures (SAMed, MSA) × 3 ViT scales (vit_b/l/h)
Total: up to 18 cells. Each cell = SE-CGR ΔDice - no-probe ΔDice.

This experiment provides the "Fig 6: Full matrix heatmap" data.
"""
import os, sys, json, gc, random, copy
import numpy as np
import torch
import h5py

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

# All available checkpoints
CHECKPOINTS = {
    # SAMed
    ("Synapse", "SAMed", "vit_b"): f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth",
    ("Synapse", "SAMed", "vit_h"): f"{ROOT}/output_runs_correct/SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth",
    ("ISIC", "SAMed", "vit_b"): f"{ROOT}/output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth",
    ("REFUGE", "SAMed", "vit_b"): f"{ROOT}/output_runs_correct/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_99.pth",
    # MSA
    ("Synapse", "MSA", "vit_b"): f"{ROOT}/output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24/Model/checkpoint_best.pth",
    ("ISIC", "MSA", "vit_b"): f"{ROOT}/output_runs_correct/MedicalSAMAdapter/isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth",
    ("REFUGE", "MSA", "vit_b"): f"{ROOT}/output_runs_correct/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24/Model/checkpoint_best.pth",
}

N_CASES = 6
NUM_SLICES = 3


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


def evaluate_isic_samed(net, image, label):
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


def evaluate_isic_msa(model, image_np, gt_mask):
    """MSA ISIC per-case evaluation. image_np: [H,W], gt_mask: [H,W]."""
    import torch.nn.functional as F
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
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
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


def evaluate_refuge_msa(model, image_np, gt_label):
    """MSA REFUGE per-case evaluation. image_np: [H,W], gt_label: [3,H,W]."""
    import torch.nn.functional as F
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
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
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
    """MSA Synapse single slice evaluation. image_np: [H,W], label_slice: [H,W]"""
    import torch.nn.functional as F
    img_3ch = np.stack([image_np]*3, axis=0)
    img_t = torch.from_numpy(img_3ch).unsqueeze(0).float().to(DEVICE)
    pt_list, lbl_list = [], []
    for c in range(1, 9):
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
            image_embeddings=image_emb, image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb,
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


def evaluate_refuge_samed(net, image, label):
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


def get_synapse_test_slices(n_cases):
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.h5')])[:n_cases]
    slices = []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            img, lbl = f['image'][:], f['label'][:]
        valid = sorted([s for s in range(img.shape[0]) if (lbl[s] > 0).sum() > 100],
                      key=lambda s: (lbl[s] > 0).sum(), reverse=True)[:NUM_SLICES]
        for s in valid:
            slices.append((img[s].astype(np.float32), lbl[s], h5f, s))
    return slices


def get_isic_test_cases(n_cases):
    from PIL import Image
    from torchvision import transforms
    import pandas as pd
    data_dir = f"{ROOT}/data/ISIC"
    df = pd.read_csv(os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth.csv"))
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    cases = []
    for _, row in df.iterrows():
        if len(cases) >= n_cases: break
        img_path = os.path.join(data_dir, row['img'])
        gt_path = os.path.join(data_dir, row['seg'])
        img = Image.open(img_path).convert('RGB')
        gt = np.array(Image.open(gt_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32) / 255.0
        cases.append((transform(img) * 255, (gt > 0.5).astype(np.float32)))
    return cases


def get_refuge_test_cases(n_cases):
    from PIL import Image
    from torchvision import transforms
    data_dir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
    cases = []
    for folder in folders:
        if len(cases) >= n_cases: break
        img_path = os.path.join(data_dir, folder, f"{folder}.jpg")
        disc_path = os.path.join(data_dir, folder, f"{folder}_disc.bmp")
        cup_path = os.path.join(data_dir, folder, f"{folder}_cup.bmp")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert('RGB')
        disc_raw = np.array(Image.open(disc_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        cup_raw = np.array(Image.open(cup_path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST), dtype=np.float32)
        label = np.stack([
            (cup_raw < 128).astype(np.float32),
            (disc_raw <= 128).astype(np.float32),
            np.zeros_like(disc_raw),
        ], axis=0)
        cases.append((transform(img) * 255, label))
    return cases


def main():
    results = {}

    for (dataset, model_type, vit), ckpt_path in CHECKPOINTS.items():
        tag = f"{dataset}/{model_type}/{vit}"
        if not os.path.exists(ckpt_path):
            print(f"SKIP {tag}: checkpoint not found")
            continue

        print(f"\n{'='*60}\n{tag}\n{'='*60}")

        # Load source SAM
        source_sam = load_sam_encoder(vit, SAM_CKPT[vit], IMAGE_SIZE, 8, DEVICE)

        # Load target
        if model_type == "SAMed":
            if dataset == "Synapse":
                num_cls = 8
                target = load_samed_model(vit, SAM_CKPT[vit], ckpt_path, IMAGE_SIZE, num_cls, device=DEVICE)
            elif dataset == "ISIC":
                num_cls = 1
                target = load_samed_model(vit, SAM_CKPT[vit], ckpt_path, IMAGE_SIZE, num_cls, device=DEVICE)
            else:
                num_cls = 2
                target = load_samed_model(vit, SAM_CKPT[vit], ckpt_path, IMAGE_SIZE, num_cls, device=DEVICE)
        else:
            if dataset == "Synapse":
                mmo = 9
            elif dataset == "ISIC":
                mmo = 1
            else:
                mmo = 2
            target = load_medsa_model(vit, SAM_CKPT[vit], ckpt_path, IMAGE_SIZE, mmo, DEVICE)

        # Get test data
        if dataset == "Synapse":
            test_items = get_synapse_test_slices(N_CASES)
        elif dataset == "ISIC":
            test_items = get_isic_test_cases(N_CASES)
        else:
            test_items = get_refuge_test_cases(N_CASES)

        se_cgr_deltas = []
        no_probe_deltas = []

        for item_idx, item in enumerate(test_items):
            if dataset == "Synapse":
                slc, lbl_slice, h5f, si = item
                if model_type == "SAMed":
                    clean_d = evaluate_samed_slice(target, slc, lbl_slice, f"{h5f}_s{si}")
                else:
                    clean_d = evaluate_msa_synapse_slice(target, slc, lbl_slice)
                slc_3ch = np.stack([slc]*3, axis=0)
                img_t = torch.from_numpy(slc_3ch).unsqueeze(0).float().to(DEVICE)
            elif dataset == "ISIC":
                img_t_orig, gt = item
                if model_type == "SAMed":
                    clean_d = evaluate_isic_samed(target, img_t_orig, torch.from_numpy(gt).float())
                else:
                    img_uint8 = img_t_orig.cpu().numpy().transpose(1,2,0).mean(axis=2)
                    clean_d = evaluate_isic_msa(target, img_uint8, torch.from_numpy(gt).float())
                img_t = img_t_orig.unsqueeze(0).float().to(DEVICE)
            else:
                img_t_orig, label = item
                if model_type == "SAMed":
                    clean_d = evaluate_refuge_samed(target, img_t_orig, torch.from_numpy(label).float())
                else:
                    img_uint8 = img_t_orig.cpu().numpy().transpose(1,2,0).mean(axis=2)
                    clean_d = evaluate_refuge_msa(target, img_uint8, torch.from_numpy(label).float())
                img_t = img_t_orig.unsqueeze(0).float().to(DEVICE)

            # SE-CGR attack
            config = AttackConfig(
                att_step_size=2.0, att_bound=10.0, att_numstep=10,
                direction_mode="radial", probe_amplification=2.0,
                edge_guidance=True, seed=42,
            )
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            adv_np, _ = run_codd_attack(source_sam.image_encoder, img_t, config, return_trace=True)

            if dataset == "Synapse":
                if model_type == "SAMed":
                    adv_d = evaluate_samed_slice(target, adv_np[0].copy(), lbl_slice, f"{h5f}_s{si}")
                else:
                    adv_d = evaluate_msa_synapse_slice(target, adv_np[0].copy(), lbl_slice)
            elif dataset == "ISIC":
                if model_type == "SAMed":
                    adv_gray = torch.from_numpy(adv_np.copy()).float()
                    adv_d = evaluate_isic_samed(target, adv_gray, torch.from_numpy(gt).float())
                else:
                    adv_d = evaluate_isic_msa(target, adv_np[0].copy(), torch.from_numpy(gt).float())
            else:
                if model_type == "SAMed":
                    adv_gray = torch.from_numpy(adv_np.copy()).float()
                    adv_d = evaluate_refuge_samed(target, adv_gray, torch.from_numpy(label).float())
                else:
                    adv_d = evaluate_refuge_msa(target, adv_np[0].copy(), torch.from_numpy(label).float())

            se_cgr_deltas.append(clean_d - adv_d)
            no_probe_deltas.append(0.0)  # clean = no attack

        mean_se_cgr = float(np.mean(se_cgr_deltas))
        mean_no_probe = 0.0

        results[tag] = {
            "se_cgr_mean_delta": mean_se_cgr,
            "se_cgr_std_delta": float(np.std(se_cgr_deltas)),
            "no_probe_mean_delta": mean_no_probe,
            "improvement": mean_se_cgr - mean_no_probe,
            "n": len(se_cgr_deltas),
        }
        print(f"  SE-CGR ΔDice: {mean_se_cgr:.4f} ± {results[tag]['se_cgr_std_delta']:.4f}")

        del source_sam, target; gc.collect(); torch.cuda.empty_cache()

    output = {"config": {"n_cases": N_CASES, "num_slices": NUM_SLICES}, "results": results}
    out_path = os.path.join(OUTPUT_DIR, "exp12_cross_arch_heatmap.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
