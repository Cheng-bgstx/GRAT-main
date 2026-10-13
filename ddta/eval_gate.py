"""D2TA gate — correct, self-contained target eval for Synapse (3D).

Uses the two established, honest protocols:
  * SAMed : forward as in test_samed.py        (clean reference ~0.807)
  * Med-SA: 8-point-per-class prompt + metric  as in unified_honest_eval.py (~0.810)
  * metric: calculate_metric_percase (both-empty -> 1.0, one-empty -> 0).

For each attack group it evaluates the adversarial images on both victims and
writes Dice / delta-Dice to a JSON.  Clean reference is adv_dir == "__clean__".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
for _p in (ROOT, os.path.join(ROOT, "newest")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate import load_samed_model, load_medsa_model

SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}


def calc_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = (pred_bin > 0)
    g = (gt_bin > 0)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    return float(2.0 * (p & g).sum() / (p.sum() + g.sum()))


def samed_forward(net, slc: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(slc).unsqueeze(0).unsqueeze(0).float().to(device)  # [1,1,H,W]
    x = x.repeat(1, 3, 1, 1)
    with torch.no_grad():
        out = net(x, True, 512)  # multimask_output=True (8 classes)
    return torch.argmax(torch.softmax(out["masks"].float(), dim=1), dim=1).squeeze(0).cpu().numpy()


def medsa_forward(model, slc: np.ndarray, gt: np.ndarray, num_classes: int, device, rng) -> np.ndarray:
    x = torch.from_numpy(np.stack([slc] * 3)).unsqueeze(0).float().to(device)
    pts, lbls = [], []
    for c in range(1, num_classes + 1):
        inds = np.argwhere(gt == c)
        if len(inds) > 0:
            r, cc = inds[rng.randint(0, len(inds) - 1)]
            pts.append([cc, r])
            lbls.append(1)
        else:
            pts.append([0, 0])
            lbls.append(0)
    coords = torch.as_tensor(np.array(pts)[None], dtype=torch.float, device=device)
    labels = torch.as_tensor(np.array(lbls)[None], dtype=torch.int, device=device)
    with torch.no_grad():
        imgs = model.preprocess(x)
        ie = model.image_encoder(imgs)
        se, de = model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=ie,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=(num_classes > 1),
        )
        pred = F.interpolate(pred, size=(512, 512), mode="bilinear", align_corners=False)
    return torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()


def load_case_images(case: str, adv_dir: str):
    """Return (images [D,H,W], labels [D,H,W]). Clean if adv_dir is empty/'__clean__'."""
    with h5py.File(f"{ROOT}/data/Synapse_preprocess/test_vol_h5/{case}.npy.h5", "r") as f:
        vol = f["image"][:].astype(np.float32)
        lbl = f["label"][:].astype(np.int64)
    if adv_dir in ("", "__clean__"):
        return vol, lbl

    pngs = sorted(glob.glob(os.path.join(adv_dir, case, "*.png")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not pngs:
        return None, None
    import matplotlib.image as mpimg
    imgs = [mpimg.imread(p) for p in pngs]
    imgs = [a[:, :, 0] if a.ndim == 3 else a for a in imgs]
    idxs = [int(os.path.splitext(os.path.basename(p))[0]) for p in pngs]
    return np.stack(imgs, axis=0).astype(np.float32), lbl[idxs]


def eval_dir(samed, medsa, adv_dir, num_classes, device, max_cases):
    with open(f"{ROOT}/data/Synapse_preprocess/lists/test_vol.txt") as f:
        cases = [l.strip() for l in f if l.strip()]
    rng = random.Random(42)
    samed_d, medsa_d = [], []
    for i, case in enumerate(cases):
        if i >= max_cases:
            break
        imgs, lbls = load_case_images(case, adv_dir)  # [D,H,W], [D,H,W]
        if imgs is None:
            continue
        # accumulate FULL-VOLUME predictions, then metric per class per volume
        # (matches test_samed.py / unified_honest_eval.py — not per-slice)
        pred_s = np.zeros_like(lbls)
        pred_m = np.zeros_like(lbls)
        for s in range(imgs.shape[0]):
            pred_s[s] = samed_forward(samed, imgs[s], device)
            pred_m[s] = medsa_forward(medsa, imgs[s], lbls[s], num_classes, device, rng)
        for c in range(1, num_classes + 1):
            samed_d.append(calc_dice(pred_s == c, lbls == c))
            medsa_d.append(calc_dice(pred_m == c, lbls == c))
    return float(np.mean(samed_d)) if samed_d else float("nan"), \
           float(np.mean(medsa_d)) if medsa_d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit", type=str, default="vit_b")
    ap.add_argument("--samed_ckpt", type=str, default=f"{ROOT}/checkpoints/best_weights/SAMed_Synapse_vit_b.pth")
    ap.add_argument("--medsa_ckpt", type=str, default=f"{ROOT}/checkpoints/best_weights/MSA_Synapse_vit_b.pth")
    ap.add_argument("--groups", type=str, required=True, help='JSON {"name": "adv_dir", ...}')
    ap.add_argument("--out", type=str, default=f"{ROOT}/ddta/gate_results.json")
    ap.add_argument("--num_classes", type=int, default=8)
    ap.add_argument("--max_cases", type=int, default=9999)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    groups = json.loads(args.groups)

    print("Loading SAMed ...", flush=True)
    samed = load_samed_model(args.vit, SAM_CKPTS[args.vit], args.samed_ckpt,
                             512, args.num_classes, device=device)
    print("Loading Med-SA ...", flush=True)
    medsa = load_medsa_model(args.vit, SAM_CKPTS[args.vit], args.medsa_ckpt,
                             512, args.num_classes + 1, device=device)

    results = {"vit": args.vit, "groups": {}}

    print("Clean reference ...", flush=True)
    c_s, c_m = eval_dir(samed, medsa, "__clean__", args.num_classes, device, args.max_cases)
    results["clean"] = {"samed_dice": c_s, "medsa_dice": c_m}
    print(f"  clean SAMed={c_s:.4f}  Med-SA={c_m:.4f}", flush=True)

    for name, adv_dir in groups.items():
        print(f"Eval '{name}' -> {adv_dir}", flush=True)
        d_s, d_m = eval_dir(samed, medsa, adv_dir, args.num_classes, device, args.max_cases)
        results["groups"][name] = {
            "samed_dice": d_s, "medsa_dice": d_m,
            "samed_ddice": c_s - d_s, "medsa_ddice": c_m - d_m,
        }
        print(f"  SAMed={d_s:.4f} (Δ={c_s-d_s:+.4f})  Med-SA={d_m:.4f} (Δ={c_m-d_m:+.4f})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
