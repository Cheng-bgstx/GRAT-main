#!/usr/bin/env python3
"""
统一"诚实"评测：用 test_samed.py 的 metric（medpy dc/hd95 + calculate_metric_percase）
评测 MSA 模型，使与 SAMed 口径一致。

MSA 推理沿用其标准协议：
  - Synapse: 每类一个点击点(8点，正样本) —— 与 eval_all_models.py / MSA 官方一致
  - ISIC/REFUGE: 单正点击(前景质心)
metric 用 SAMed 的 calculate_metric_percase（argmax / 0.5 阈值，空类按对角惩罚）。
"""
import sys, os, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import h5py
import random

ROOT = "/root/data2/fyc/fyc/GRAT-main"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "newest"))

from evaluate import load_medsa_model

DEVICE = torch.device("cuda:0")
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}
# (dataset, vit) -> MSA 训练目录名
MSA_DIRS = {
    ("Synapse", "vit_b"): "Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s24",
    ("Synapse", "vit_l"): "Synapse_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24",
    ("Synapse", "vit_h"): "Synapse_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24",
    ("ISIC", "vit_b"): "isic_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24",
    ("ISIC", "vit_l"): "isic_512_pretrain_vit_l_epo100_bs16_lr0.0001_s24",
    ("ISIC", "vit_h"): "isic_512_pretrain_vit_h_epo100_bs8_lr0.0001_s24",
    ("REFUGE", "vit_b"): "REFUGE_512_pretrain_vit_b_epo100_bs24_lr0.0001_s24",
    ("REFUGE", "vit_l"): "REFUGE_512_pretrain_vit_l_epo100_bs12_lr0.0001_s24",
    ("REFUGE", "vit_h"): "REFUGE_512_pretrain_vit_h_epo100_bs6_lr0.0001_s24",
}


def calculate_metric_percase(pred, gt):
    from medpy import metric
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0
    else:
        max_dist = np.sqrt(np.sum(np.array(pred.shape) ** 2))
        return 0.0, max_dist


def msa_forward(model, img_tensor, coords, labels, num_classes, image_size=512):
    """coords: [1,N,2], labels: [1,N]。返回 [1,C,H,W] logits。"""
    with torch.no_grad():
        imgs = model.preprocess(img_tensor)
        ie = model.image_encoder(imgs)
        c = torch.as_tensor(coords, dtype=torch.float, device=DEVICE)
        l = torch.as_tensor(labels, dtype=torch.int, device=DEVICE)
        se, de = model.prompt_encoder(points=(c, l), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=ie,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=(num_classes > 1),
        )
        pred = F.interpolate(pred, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return pred


def eval_synapse(model, num_classes=9, seed=42):
    rng = random.Random(seed)
    with open(f"{ROOT}/data/Synapse_preprocess/lists/test_vol.txt") as f:
        cases = [l.strip() for l in f if l.strip()]
    per_class = {c: [] for c in range(1, num_classes)}
    for case in cases:
        with h5py.File(f"{ROOT}/data/Synapse_preprocess/test_vol_h5/{case}.npy.h5", "r") as f:
            vol = f["image"][:].astype(np.float32)
            lbl = f["label"][:].astype(np.int64)
        pred_vol = np.zeros_like(lbl)
        for s in range(vol.shape[0]):
            slc = vol[s]
            x = torch.from_numpy(np.stack([slc] * 3)).unsqueeze(0).float().to(DEVICE)
            gt_s = lbl[s]
            pts, lbls = [], []
            for c in range(1, num_classes):
                inds = np.argwhere(gt_s == c)
                if len(inds) > 0:
                    r, cc = inds[rng.randint(0, len(inds) - 1)]
                    pts.append([cc, r]); lbls.append(1)
                else:
                    pts.append([0, 0]); lbls.append(0)
            pred = msa_forward(model, x, np.array(pts)[None], np.array(lbls)[None], num_classes)
            pred_vol[s] = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        for c in range(1, num_classes):
            d, h = calculate_metric_percase(pred_vol == c, lbl == c)
            per_class[c].append((d, h))
    dice = np.mean([np.mean([v[0] for v in per_class[c]]) for c in range(1, num_classes)])
    hd95 = np.mean([np.mean([v[1] for v in per_class[c]]) for c in range(1, num_classes)])
    return dice, hd95


def eval_isic(model):
    imgs = sorted(glob.glob(f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data/*.jpg"))
    dices, hds = [], []
    for ip in imgs:
        bp = ip.replace("_Data", "_GroundTruth").replace(".jpg", "_Segmentation.png")
        if not os.path.exists(bp):
            continue
        img = Image.open(ip).convert("RGB").resize((512, 512), Image.BILINEAR)
        im = torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        gt = (np.array(Image.open(bp).convert("L").resize((512, 512), Image.NEAREST)) > 128).astype(np.uint8)
        ys, xs = np.nonzero(gt)
        if len(xs) == 0:
            pt, lab = ([0, 0]), 0
        else:
            pt, lab = ([float(xs.mean()), float(ys.mean())]), 1
        pred = msa_forward(model, im, np.array([pt])[None], np.array([[lab]]), num_classes=1)
        pmask = (torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
        d, h = calculate_metric_percase(pmask, gt)
        dices.append(d); hds.append(h)
    return float(np.mean(dices)), float(np.mean(hds))


def eval_refuge(model):
    base = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    subfolders = sorted([f.path for f in os.scandir(base) if f.is_dir()])
    disc_d, disc_h, cup_d, cup_h = [], [], [], []
    for sub in subfolders:
        name = os.path.basename(sub)
        img = Image.open(os.path.join(sub, name + ".jpg")).convert("RGB").resize((512, 512), Image.BILINEAR)
        im = torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        disc = [np.array(Image.open(os.path.join(sub, f"{name}_seg_disc_{i}.png")).convert("L").resize((512,512), Image.NEAREST)) for i in range(1, 8)]
        cup = [np.array(Image.open(os.path.join(sub, f"{name}_seg_cup_{i}.png")).convert("L").resize((512,512), Image.NEAREST)) for i in range(1, 8)]
        # SAMed 口径：soft-mean > 0（任意 rater 标为前景）
        gt_disc = (np.mean(np.stack([(d > 128) for d in disc], 0).astype(np.float32), 0) > 0).astype(np.uint8)
        gt_cup = (np.mean(np.stack([(c > 128) for c in cup], 0).astype(np.float32), 0) > 0).astype(np.uint8)
        # MSA 官方 val.py 对 REFUGE 用 pt=[0,0], label=1（prompt='none' 的自动分割模式），
        # 不是前景点击。这里沿用其协议，避免前景点偏置模型。
        pt, lab = ([0.0, 0.0]), 1
        pred = msa_forward(model, im, np.array([pt])[None], np.array([[lab]]), num_classes=2)
        probs = torch.sigmoid(pred).squeeze(0).cpu().numpy()  # [2,H,W] 通道0=cup, 通道1=disc（与 MSA dataset 的 concat([cup,disc]) 一致）
        pc = (probs[0] > 0.5).astype(np.uint8)
        pd = (probs[1] > 0.5).astype(np.uint8)
        dd, dh = calculate_metric_percase(pd, gt_disc)
        cd, ch = calculate_metric_percase(pc, gt_cup)
        disc_d.append(dd); disc_h.append(dh); cup_d.append(cd); cup_h.append(ch)
    return (float(np.mean(disc_d)), float(np.mean(cup_d))), (float(np.mean(disc_h)), float(np.mean(cup_h)))


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    vit = sys.argv[2] if len(sys.argv) > 2 else "vit_b"
    results = {}
    for d in (["Synapse", "ISIC", "REFUGE"] if ds == "all" else [ds]):
        ckpt = f"{ROOT}/output_runs_correct/MedicalSAMAdapter/{MSA_DIRS[(d, vit)]}/Model/best_dice_checkpoint.pth"
        print(f"Loading {d} {vit} ...", flush=True)
        model = load_medsa_model(vit, SAM_CKPTS[vit], ckpt, image_size=512,
                                 multimask_output=(9 if d == "Synapse" else (2 if d == "REFUGE" else 1)),
                                 device=DEVICE)
        if d != "Synapse":
            model.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(DEVICE)
            model.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(DEVICE)
        if d == "Synapse":
            dice, hd95 = eval_synapse(model)
            results[d] = {"dice": round(dice, 4), "hd95": round(hd95, 4)}
            print(f"[MSA {d} {vit}] dice={dice:.4f} hd95={hd95:.4f}", flush=True)
        elif d == "ISIC":
            dice, hd95 = eval_isic(model)
            results[d] = {"dice": round(dice, 4), "hd95": round(hd95, 4)}
            print(f"[MSA {d} {vit}] dice={dice:.4f} hd95={hd95:.4f}", flush=True)
        else:
            (dd, cd), (dh, ch) = eval_refuge(model)
            avg_d, avg_h = (dd + cd) / 2.0, (dh + ch) / 2.0
            results[d] = {"disc_dice": round(dd, 4), "cup_dice": round(cd, 4), "avg_dice": round(avg_d, 4),
                          "disc_hd95": round(dh, 4), "cup_hd95": round(ch, 4), "avg_hd95": round(avg_h, 4)}
            print(f"[MSA {d} {vit}] disc={dd:.4f}/{dh:.4f} cup={cd:.4f}/{ch:.4f} avg={avg_d:.4f}/{avg_h:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))
