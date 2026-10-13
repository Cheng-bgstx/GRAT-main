#!/usr/bin/env python3
"""Comprehensive checkpoint audit: test ALL on their test sets, report Dice + HD95."""
import os, sys, json, gc, traceback
import numpy as np
import torch
import h5py
from PIL import Image

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
SAVE_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(SAVE_DIR, exist_ok=True)
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_samed_model
from scipy.ndimage import zoom as ndi_zoom
from scipy.ndimage import distance_transform_edt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512

# ImageNet normalization (used during SAMed training for RGB images)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def _normalize_rgb(img_t):
    """Apply ImageNet normalization to RGB tensor [1,3,H,W] in [0,255] range."""
    img_t = img_t / 255.0
    img_t = (img_t - _IMAGENET_MEAN) / _IMAGENET_STD
    return img_t

SAM_CKPT = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}
OUT = f"{ROOT}/output_runs_correct"

def _dice(p, g, eps=1e-8):
    return float(2.0 * (p * g).sum() / (p.sum() + g.sum() + eps))

def _hd95(p, g):
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float('nan')
    d1 = distance_transform_edt(1 - p)
    d2 = distance_transform_edt(1 - g)
    return float(max(np.percentile(d1[g > 0], 95), np.percentile(d2[p > 0], 95)))

def _forward(net, img_t, multimask_output=False):
    with torch.no_grad():
        out = net(img_t, multimask_output=multimask_output, image_size=IMAGE_SIZE)
        m = out['masks']
        pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    return pred

# ── ISIC ──────────────────────────────────────────────────────────────
def eval_isic(ckpt, vit, num_classes, note):
    print(f"[ISIC] {vit} {note}  ", flush=True, end="")
    net = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, num_classes, device=DEVICE)
    net.eval()
    tdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_Data"
    mdir = f"{ROOT}/data/ISIC/ISBI2016_ISIC_Part1_Test_GroundTruth"
    jpgs = sorted([f for f in os.listdir(tdir) if f.endswith('.jpg')])
    dices, hds = [], []
    for jpg in jpgs:
        img = Image.open(os.path.join(tdir, jpg)).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32)
        base = jpg.replace('.jpg', '')
        mask = Image.open(os.path.join(mdir, base + '_Segmentation.png')).convert('L')
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        gt = (np.array(mask) > 128).astype(np.float32)
        if gt.sum() < 100:
            continue
        img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        img_t = _normalize_rgb(img_t)
        pred = _forward(net, img_t)
        pred_fg = (pred == 1).astype(np.float32)
        dices.append(_dice(pred_fg, gt))
        hds.append(_hd95(pred_fg, gt))

    del net; gc.collect(); torch.cuda.empty_cache()
    vd = [x for x in dices if not np.isnan(x)]
    vh = [x for x in hds if not np.isnan(x)]
    r = {"dataset":"ISIC","vit":vit,"note":note,"n":len(dices),
         "mdice":np.mean(vd),"std_dice":np.std(vd),"med_dice":np.median(vd),
         "mhd95":np.mean(vh),"med_hd95":np.median(vh)}
    print(f"Dice={r['mdice']:.4f}+-{r['std_dice']:.4f}  HD95={r['mhd95']:.1f}  N={r['n']}", flush=True)
    return r

# ── Synapse ───────────────────────────────────────────────────────────
def eval_synapse(ckpt, vit, num_classes, note):
    print(f"[Synapse] {vit} {note}  ", flush=True, end="")
    net = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, num_classes, device=DEVICE)
    net.eval()
    tdir = f"{ROOT}/data/Synapse_preprocess/test_vol_h5"
    h5s = sorted([f for f in os.listdir(tdir) if f.endswith('.npy.h5')])
    case_dices, case_hds = [], []
    for h5f in h5s:
        with h5py.File(os.path.join(tdir, h5f), 'r') as f:
            image = f['image'][:]; label = f['label'][:]
        pred_vol = np.zeros_like(label, dtype=np.float32)
        for s in range(image.shape[0]):
            slc = image[s]
            slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
            slc_t = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().repeat(1,3,1,1).to(DEVICE)
            p = _forward(net, slc_t, multimask_output=True)
            if p.shape != slc.shape:
                p = ndi_zoom(p, (slc.shape[0]/p.shape[0], slc.shape[1]/p.shape[1]), order=0)
            pred_vol[s] = p
        cd = []
        for c in range(1, num_classes + 1):
            g = (label == c).astype(np.float32)
            if g.sum() == 0: continue
            cd.append(_dice((pred_vol == c).astype(np.float32), g))
        case_dices.append(np.mean(cd) if cd else 0.0)
        case_hds.append(_hd95((pred_vol > 0).astype(np.float32), (label > 0).astype(np.float32)))

    del net; gc.collect(); torch.cuda.empty_cache()
    vd = [x for x in case_dices if not np.isnan(x)]
    vh = [x for x in case_hds if not np.isnan(x)]
    r = {"dataset":"Synapse","vit":vit,"note":note,"n":len(case_dices),
         "mdice":np.mean(vd),"std_dice":np.std(vd),"med_dice":np.median(vd),
         "mhd95":np.mean(vh),"med_hd95":np.median(vh),
         "per_case_dice":case_dices,"per_case_hd95":case_hds}
    print(f"Dice={r['mdice']:.4f}+-{r['std_dice']:.4f}  HD95={r['mhd95']:.1f}  N={r['n']}", flush=True)
    return r

# ── REFUGE ────────────────────────────────────────────────────────────
def eval_refuge(ckpt, vit, num_classes, note):
    print(f"[REFUGE] {vit} {note}  ", flush=True, end="")
    net = load_samed_model(vit, SAM_CKPT[vit], ckpt, IMAGE_SIZE, num_classes, device=DEVICE)
    net.eval()
    tdir = f"{ROOT}/data/REFUGE-Multirater/Test-400"
    dirs = sorted([d for d in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, d))])
    cup_dices, disc_dices, hds = [], [], []
    for d in dirs:
        dp = os.path.join(tdir, d)
        img_p = os.path.join(dp, f"{d}.jpg")
        if not os.path.exists(img_p): continue
        cup_raters = sorted([f for f in os.listdir(dp) if 'seg_cup_' in f and f.endswith('.png')])
        disc_raters = sorted([f for f in os.listdir(dp) if 'seg_disc_' in f and f.endswith('.png')])
        if not cup_raters or not disc_raters: continue

        cup_sum = None
        for cf in cup_raters:
            c = Image.open(os.path.join(dp, cf)).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            c = (np.array(c) > 128).astype(np.float32)
            cup_sum = c if cup_sum is None else cup_sum + c
        cup_gt = (cup_sum / len(cup_raters) > 0.5).astype(np.float32)

        disc_sum = None
        for df_ in disc_raters:
            di = Image.open(os.path.join(dp, df_)).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            di = (np.array(di) > 128).astype(np.float32)
            disc_sum = di if disc_sum is None else disc_sum + di
        disc_gt = (disc_sum / len(disc_raters) > 0.5).astype(np.float32)

        img = Image.open(img_p).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img).astype(np.float32)).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        img_t = _normalize_rgb(img_t)
        pred = _forward(net, img_t)  # 0=bg, 1=disc, 2=cup

        pred_cup = (pred == 2).astype(np.float32)
        pred_disc = ((pred == 1) | (pred == 2)).astype(np.float32)
        cup_dices.append(_dice(pred_cup, cup_gt))
        disc_dices.append(_dice(pred_disc, disc_gt))
        hds.append(_hd95((pred > 0).astype(np.float32), (cup_gt + disc_gt > 0).astype(np.float32)))

    del net; gc.collect(); torch.cuda.empty_cache()
    avg_dices = [(c + d) / 2.0 for c, d in zip(cup_dices, disc_dices)]
    vh = [x for x in hds if not np.isnan(x)]
    r = {"dataset":"REFUGE","vit":vit,"note":note,"n":len(avg_dices),
         "mdice":np.mean(avg_dices),"std_dice":np.std(avg_dices),"med_dice":np.median(avg_dices),
         "mhd95":np.mean(vh),"med_hd95":np.median(vh),
         "cup_dice":np.mean(cup_dices),"disc_dice":np.mean(disc_dices)}
    print(f"Dice={r['mdice']:.4f}+-{r['std_dice']:.4f}  HD95={r['mhd95']:.1f}  N={r['n']}", flush=True)
    return r

# ── Main ──────────────────────────────────────────────────────────────
CHECKPOINTS = [
    ("ISIC", "vit_b", f"{OUT}/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 1, "correct"),
    ("ISIC", "vit_l", f"{OUT}/SAMed/isic_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_249.pth", 1, "correct_running"),
    ("ISIC", "vit_h", f"{OUT}/SAMed/isic_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_139.pth", 1, "correct_running"),
    ("ISIC", "vit_l", f"{OUT}/SAMed/isic_512_pretrain_vit_l_epo200_bs12_lr0.005/epoch_159.pth", 1, "WRONG_module"),
    ("ISIC", "vit_h", f"{OUT}/SAMed/isic_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_159.pth", 1, "WRONG_module"),
    ("REFUGE", "vit_b", f"{OUT}/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_199.pth", 2, "correct_overtrained"),
    ("REFUGE", "vit_l", f"{OUT}/SAMed/REFUGE_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_299.pth", 2, "correct_OVERFIT"),
    ("REFUGE", "vit_h", f"{OUT}/SAMed/REFUGE_512_pretrain_vit_h_epo300_bs8_lr0.005/epoch_19.pth", 2, "correct_early"),
    ("REFUGE", "vit_l", f"{OUT}/SAMed/REFUGE_512_pretrain_vit_l_epo200_bs12_lr0.005/epoch_199.pth", 2, "WRONG_module"),
    ("REFUGE", "vit_h", f"{OUT}/SAMed/REFUGE_512_pretrain_vit_h_epo200_bs8_lr0.005/epoch_199.pth", 2, "WRONG_module"),
    ("Synapse", "vit_b", f"{OUT}/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth", 8, "correct_no_log"),
    ("Synapse", "vit_h", f"{OUT}/SAMed/Synapse_512_pretrain_vit_h_epo200_bs6_lr0.005/epoch_299.pth", 8, "correct_dir_mismatch"),
    ("Synapse", "vit_l", f"{OUT}/SAMed/Synapse_512_pretrain_vit_l_epo200_bs12_lr0.005/epoch_159.pth", 8, "WRONG_module"),
]

print("=" * 90)
print("COMPREHENSIVE CHECKPOINT AUDIT")
print(f"Device: {DEVICE}  |  {len(CHECKPOINTS)} checkpoints to test")
print("=" * 90)

all_results = []
fn_map = {"ISIC": eval_isic, "Synapse": eval_synapse, "REFUGE": eval_refuge}

for dataset, vit, ckpt_path, nclass, note in CHECKPOINTS:
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: not found: {ckpt_path}")
        continue
    try:
        r = fn_map[dataset](ckpt_path, vit, nclass, note)
        all_results.append(r)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        traceback.print_exc()

print("\n\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)
print(f"{'Dataset':<10} {'Vit':<6} {'Note':<28} {'mDice':<10} {'stdDice':<10} {'medDice':<10} {'mHD95':<10} {'medHD95':<10}")
print("-" * 100)
for r in all_results:
    print(f"{r['dataset']:<10} {r['vit']:<6} {r['note']:<28} "
          f"{r['mdice']:<10.4f} {r['std_dice']:<10.4f} {r['med_dice']:<10.4f} "
          f"{r['mhd95']:<10.1f} {r['med_hd95']:<10.1f}")

print("\n--- ANOMALIES ---")
for r in all_results:
    if r['mdice'] < 0.5 or np.isnan(r['mdice']):
        print(f"  ⚠️  LOW DICE: {r['dataset']} {r['vit']} {r['note']}: mdice={r['mdice']:.4f}")
    if not np.isnan(r['mhd95']) and r['mhd95'] > 50:
        print(f"  ⚠️  HIGH HD95: {r['dataset']} {r['vit']} {r['note']}: mhd95={r['mhd95']:.1f}")

out_p = os.path.join(SAVE_DIR, "checkpoint_audit.json")
with open(out_p, "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved to {out_p}")
