"""
Evaluation pipeline for adversarial attacks on fine-tuned SAM models.

Evaluates segmentation quality of victim models (SAMed, MedSA) on
clean vs. adversarial images. Supports Synapse (3D), ISIC (2D), REFUGE (2D).

Metrics: Dice score, HD95, IoU (per-case and aggregated).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

SYNAPSE_VOLUME_PATH = str(ROOT / "data/Synapse_preprocess/test_vol_h5")
SYNAPSE_LIST_DIR = str(ROOT / "data/Synapse_preprocess/lists")
ISIC_DATA_PATH = str(ROOT / "data/ISIC")
REFUGE_DATA_PATH = str(ROOT / "data/REFUGE-Multirater")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _safe_load_sam(vit_name, sam_ckpt, image_size, num_classes):
    """Load SAM model with forced CPU checkpoint loading to avoid multi-GPU issues."""
    from segment_anything import sam_model_registry
    import builtins
    _real_load = torch.load
    def _cpu_load(f, *a, **kw):
        kw.setdefault("map_location", "cpu")
        return _real_load(f, *a, **kw)
    torch.load = _cpu_load
    try:
        sam, _ = sam_model_registry[vit_name](
            image_size=image_size, num_classes=num_classes,
            checkpoint=sam_ckpt, pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
        )
    finally:
        torch.load = _real_load
    return sam


def load_sam_encoder(vit_name: str, sam_ckpt: str, image_size: int = 512,
                     num_classes: int = 8, device: torch.device = torch.device("cuda")):
    sam = _safe_load_sam(vit_name, sam_ckpt, image_size, num_classes)
    sam = sam.to(device).eval()
    # Explicitly move pixel_mean/pixel_std — they can linger on CPU or cuda:0
    sam.pixel_mean = sam.pixel_mean.to(device)
    sam.pixel_std = sam.pixel_std.to(device)
    return sam


def load_samed_model(vit_name: str, sam_ckpt: str, lora_ckpt: str,
                     image_size: int = 512, num_classes: int = 8,
                     rank: Optional[int] = None,
                     device: torch.device = torch.device("cuda")):
    from importlib import import_module
    sam = _safe_load_sam(vit_name, sam_ckpt, image_size, num_classes)
    ckpt_state = torch.load(lora_ckpt, map_location="cpu")
    # Auto-detect module: sam_lora_image_encoder_mask_decoder checkpoints
    # have sa_a_* keys (LoRA applied to mask decoder self-attention);
    # sam_lora_image_encoder checkpoints only have w_a_*/w_b_* keys.
    if any(k.startswith("sa_a_") for k in ckpt_state.keys()):
        module_name = "sam_lora_image_encoder_mask_decoder"
    else:
        module_name = "sam_lora_image_encoder"
    pkg = import_module(module_name)
    lora_keys = sorted([k for k in ckpt_state.keys() if k.startswith("w_a_")])
    if lora_keys:
        inferred_rank = int(ckpt_state[lora_keys[0]].shape[0])
        if rank is None:
            rank = inferred_rank
        elif rank != inferred_rank:
            raise ValueError(
                f"LoRA rank mismatch: requested {rank}, checkpoint uses {inferred_rank}"
            )
    elif rank is None:
        rank = 4
    n_blocks = len(lora_keys) // 2 if lora_keys else len(sam.image_encoder.blocks)
    net = pkg.LoRA_Sam(sam, rank, lora_layer=list(range(n_blocks))).to(device)
    net.load_lora_parameters(lora_ckpt)
    net = net.to(device)  # ensure LoRA weights moved to device after load_lora_parameters replaces params on CPU
    net.sam.pixel_mean = net.sam.pixel_mean.to(device)
    net.sam.pixel_std = net.sam.pixel_std.to(device)
    net.eval()
    return net


def load_medsa_model(vit_name: str, sam_ckpt: str, msa_ckpt: str,
                     image_size: int = 512, multimask_output: int = 9,
                     device: torch.device = torch.device("cuda"),
                     dataset: str = "Synapse"):
    msa_root = str(ROOT / "Medical-SAM-Adapter-main")
    saved_path = list(sys.path)
    if msa_root in sys.path:
        sys.path.remove(msa_root)
    sys.path.insert(0, msa_root)
    try:
        from models.sam import sam_model_registry as msa_registry
        from argparse import Namespace
        args = Namespace(
            dataset=dataset, image_size=image_size, multimask_output=multimask_output,
            sam_ckpt=sam_ckpt, encoder=vit_name, mod="sam_adpt",
            mid_dim=None, thd=False, chunk=None,
        )
        model = msa_registry[vit_name](args, checkpoint=sam_ckpt).to(device)
    finally:
        sys.path = saved_path
    ckpt = torch.load(msa_ckpt, map_location="cpu")
    # Remove mask_decoder keys with size mismatches (MedSA uses different mask token count)
    state_dict = ckpt["state_dict"]
    model_dict = model.state_dict()
    filtered_sd = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape != model_dict[k].shape:
            continue  # skip size-mismatched keys
        filtered_sd[k] = v
    model.load_state_dict(filtered_sd, strict=False)
    model.pixel_mean = model.pixel_mean.to(device)
    model.pixel_std = model.pixel_std.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def compute_dice_hd95(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0, 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0, 0.0
    intersection = float((pred_bin & gt_bin).sum())
    dice = 2.0 * intersection / float(pred_bin.sum() + gt_bin.sum())
    try:
        from medpy.metric import hd95
        hd = float(hd95(pred_bin, gt_bin)) if pred_bin.sum() > 0 and gt_bin.sum() > 0 else 0.0
    except Exception:
        hd = 0.0
    return dice, hd


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    union = float((pred_bin | gt_bin).sum())
    if union == 0:
        return 1.0
    return float((pred_bin & gt_bin).sum()) / union


# ---------------------------------------------------------------------------
# SAMed evaluation
# ---------------------------------------------------------------------------

def evaluate_samed(
    net,                      # LoRA_Sam model
    dataloader,
    adv_dir: str,
    num_classes: int = 8,
    image_size: int = 512,
    is_3d: bool = True,
    max_cases: int = 9999,
    max_eval_slices: int = 0,
    device: torch.device = torch.device("cuda"),
) -> List[Dict]:
    from importlib import util as _iu
    _spec = _iu.spec_from_file_location("_root_utils", str(ROOT / "utils.py"))
    _root_utils = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_root_utils)
    test_single_volume = _root_utils.test_single_volume

    rows = []
    for idx, batch in enumerate(dataloader):
        if len(rows) >= max_cases:
            break
        image_clean = batch["image"]
        label = batch["label"]
        case_name = batch["case_name"][0] if "case_name" in batch else batch.get("image_meta_dict", {}).get("filename_or_obj", [f"case_{idx}"])[0]

        if adv_dir in ("", "__clean__"):
            image_adv = image_clean
            if not torch.is_tensor(image_adv):
                image_adv = torch.as_tensor(image_adv)
            image_adv = image_adv.float()
            # Restrict to organ-containing slices for fair comparison
            if max_eval_slices > 0 and is_3d:
                depth = image_adv.shape[1]
                n_slices = min(depth, max_eval_slices)
                # Find valid slices where labels contain organs
                label_np = label.squeeze(0).cpu().numpy()  # [D, H, W]
                valid_slices = [s for s in range(depth) if (label_np[s] > 0).sum() > 100]
                if valid_slices:
                    valid_mid = len(valid_slices) // 2
                    start_idx = max(0, valid_mid - n_slices // 2)
                    end_idx = min(len(valid_slices), start_idx + n_slices)
                    indices = [valid_slices[i] for i in range(start_idx, end_idx)]
                else:
                    mid = depth // 2
                    start = max(0, mid - n_slices // 2)
                    indices = list(range(start, min(start + n_slices, depth)))
                image_adv = image_adv[:, indices, :, :]
                label = label[:, indices, :, :]
        else:
            image_adv, png_slices = _load_adv_images(adv_dir, str(case_name), is_3d=is_3d)
            if image_adv is None:
                continue
            image_adv = torch.from_numpy(image_adv).float()
            # Select matching label slices from PNG filenames
            if png_slices is not None and is_3d:
                label = label[:, png_slices, :, :]
            elif is_3d and image_adv.dim() >= 3:
                adv_depth = image_adv.shape[0]
                if adv_depth < label.shape[1]:
                    label = label[:, :adv_depth, :, :]

        if is_3d and image_adv.dim() >= 3 and image_adv.shape[1] != label.shape[1]:
            if adv_dir in ("", "__clean__") and image_adv.shape[1] < label.shape[1]:
                label = label[:, :image_adv.shape[1], :, :]

        metric = test_single_volume(
            image=image_adv, label=label,
            net=net, classes=num_classes, multimask_output=True,
            patch_size=[image_size, image_size], test_save_path=None,
            case=case_name, z_spacing=1,
        )
        rows.append({
            "case_name": str(case_name),
            "mean_dice": float(np.mean(metric, axis=0)[0]),
            "mean_hd95": float(np.mean(metric, axis=0)[1]),
        })
    return rows


# ---------------------------------------------------------------------------
# MedSA evaluation
# ---------------------------------------------------------------------------

def evaluate_medsa(
    model,                    # MedSA model
    dataloader,
    adv_dir: str,
    num_classes: int = 8,
    image_size: int = 512,
    is_3d: bool = True,
    max_cases: int = 9999,
    max_eval_slices: int = 0,
    device: torch.device = torch.device("cuda"),
) -> List[Dict]:
    rows = []

    if is_3d:
        for idx, batch in enumerate(dataloader):
            if idx >= max_cases:
                break
            case_name = _extract_case_name(batch)
            label = batch["label"]

            if adv_dir in ("", "__clean__"):
                image = batch["image"]
                if not torch.is_tensor(image):
                    image = torch.as_tensor(image)
                image = image.float()
                # Restrict to organ-containing slices
                if max_eval_slices > 0:
                    depth = image.shape[1]
                    n_slices = min(depth, max_eval_slices)
                    label_np = label.squeeze(0).cpu().numpy()
                    valid_slices = [s for s in range(depth) if (label_np[s] > 0).sum() > 100]
                    if valid_slices:
                        valid_mid = len(valid_slices) // 2
                        start_idx = max(0, valid_mid - n_slices // 2)
                        end_idx = min(len(valid_slices), start_idx + n_slices)
                        indices = [valid_slices[i] for i in range(start_idx, end_idx)]
                    else:
                        mid = depth // 2
                        start = max(0, mid - n_slices // 2)
                        indices = list(range(start, min(start + n_slices, depth)))
                    image = image[:, indices, :, :]
                    label = label[:, indices, :, :]
            else:
                image_np, png_slices = _load_adv_images(adv_dir, str(case_name), is_3d=True)
                if image_np is None:
                    continue
                image = torch.from_numpy(image_np).float()
                if png_slices is not None:
                    label = label[:, png_slices, :, :]

            # image: [D, H, W] -> [1, 1, H, W, D] for _evaluate_medsa_3d
            if image.dim() == 3:
                image = image.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
            elif image.dim() == 4:  # [1, D, H, W] from dataloader
                image = image.squeeze(0).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)

            # label: [1, D, H, W] from dataloader -> [1, 1, H, W, D]
            label = label.float()
            if label.dim() == 4:
                label = label.permute(0, 2, 3, 1).unsqueeze(1)

            metrics = _evaluate_medsa_3d(model, image, label, num_classes, image_size, device)
            rows.append({"case_name": str(case_name), **metrics})
    else:
        for idx, batch in enumerate(dataloader):
            if idx >= max_cases:
                break
            case_name = _extract_case_name(batch)
            label = batch["label"]
            if adv_dir in ("", "__clean__"):
                image = batch["image"]
            else:
                image_np, _ = _load_adv_images(adv_dir, str(case_name), is_3d=False)
                if image_np is None:
                    continue
                image = torch.from_numpy(image_np).float()
            metrics = _evaluate_medsa_2d(model, image, label, device)
            rows.append({"case_name": str(case_name), **metrics})
    return rows


def _evaluate_medsa_3d(model, image, label, num_classes, image_size, device):
    from medpy.metric import binary
    model.eval()
    img = image.to(dtype=torch.float32, device=device)
    msk = label.to(dtype=torch.float32, device=device)
    total_dice, total_hd95, total_iou = 0.0, 0.0, 0.0
    evl_chunk = 1
    steps = 0
    buoy = 0
    while (buoy + evl_chunk) <= img.size(-1):
        imgs = img[..., buoy:buoy + evl_chunk]
        masks = msk[..., buoy:buoy + evl_chunk]
        buoy += evl_chunk
        steps += 1
        if imgs.dim() == 5:
            b, c, h, w, d = imgs.size()
            imgs = imgs.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, h, w)
            masks = masks.permute(0, 4, 1, 2, 3).contiguous().view(-1, 1, h, w)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        # Generate click prompt
        pt, pt_labels = _medsa_click_prompt(masks, num_classes)
        pt = pt.permute(0, 2, 1)
        coords = torch.as_tensor(pt, dtype=torch.float, device=device)
        labels_pt = torch.as_tensor(pt_labels, dtype=torch.int, device=device)
        if labels_pt.dim() == 1:
            coords = coords.unsqueeze(1)
            labels_pt = labels_pt.unsqueeze(1)
        prompt = (coords, labels_pt)
        with torch.no_grad():
            imgs_pre = model.preprocess(imgs)
            image_embeddings = model.image_encoder(imgs_pre)
            sparse_emb, dense_emb = model.prompt_encoder(points=prompt, boxes=None, masks=None)
            pred, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=(num_classes > 1),
            )
            pred = F.interpolate(pred, size=(image_size, image_size), mode="bilinear", align_corners=False)
        # Per-class metrics
        pred_classes = torch.argmax(pred, dim=1).cpu().numpy()
        true_mask = masks.squeeze(1).cpu().numpy()
        for c in range(1, num_classes):
            pc = pred_classes == c
            tc = true_mask == c
            intersection = float((pc & tc).sum())
            union = float(pc.sum() + tc.sum())
            dice = 2.0 * intersection / union if union > 0 else 1.0
            total_dice += dice
            if pc.sum() > 0 and tc.sum() > 0:
                try:
                    total_hd95 += float(binary.hd95(pc, tc))
                except Exception:
                    pass
            total_iou += intersection / (float((pc | tc).sum()) + 1e-8)
    n = max(1, steps * (num_classes - 1))
    return {"mean_dice": total_dice / n, "mean_hd95": total_hd95 / max(1, steps), "mean_iou": total_iou / n}


def _evaluate_medsa_2d(model, image, label, device):
    import random as _random
    img = image.to(dtype=torch.float32, device=device)
    msk = label.to(dtype=torch.float32, device=device)
    # Click prompt
    if msk.dim() == 3:
        msk = msk.unsqueeze(0)
    b, _, h, w = msk.shape
    pt_list, lbl_list = [], []
    for j in range(b):
        m = msk[j, 0]
        indices = torch.nonzero(m)
        if indices.size(0) == 0:
            pt_list.append(torch.tensor([0, 0]))
            lbl_list.append(0)
        else:
            ri = indices[_random.randint(0, indices.size(0) - 1)].flip(0)
            pt_list.append(ri)
            lbl_list.append(1)
    pt = torch.stack(pt_list, dim=0).unsqueeze(1).to(device)
    labels_pt = torch.tensor(lbl_list, dtype=torch.int, device=device).unsqueeze(0)
    prompt = (pt.float(), labels_pt)
    with torch.no_grad():
        imgs_pre = model.preprocess(img)
        image_embeddings = model.image_encoder(imgs_pre)
        sparse_emb, dense_emb = model.prompt_encoder(points=prompt, boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
    probs = torch.sigmoid(pred).detach().cpu().numpy()
    mask_np = msk.detach().cpu().numpy()
    dices, hd95s, ious = [], [], []
    for th in [0.1, 0.3, 0.5, 0.7, 0.9]:
        pred_bin = (probs[:, 0] > th).astype(np.uint8)
        gt_bin = (mask_np[:, 0] > th).astype(np.uint8)
        for pb, gb in zip(pred_bin, gt_bin):
            d, h = compute_dice_hd95(pb, gb)
            dices.append(d)
            hd95s.append(h)
            ious.append(compute_iou(pb, gb))
    return {"mean_dice": float(np.mean(dices)), "mean_hd95": float(np.mean(hd95s)), "mean_iou": float(np.mean(ious))}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _medsa_click_prompt(msk, num_classes):
    import random as _random
    if msk.dim() == 3:
        msk = F.one_hot(msk.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    if msk.dim() == 4:
        msk = msk.unsqueeze(1).permute(0, 1, 3, 4, 2)
    b, _, h, w, d = msk.size()
    msk = msk[:, 0]
    pt_list, label_list = [], []
    for i in range(d):
        pt_s, lbl_s = [], []
        for j in range(b):
            m = msk[j, :, :, i]
            indices = torch.nonzero(m)
            if indices.size(0) == 0:
                pt_s.append(torch.tensor([0, 0], device=msk.device))
                lbl_s.append(0)
            else:
                pt_s.append(indices[_random.randrange(indices.size(0))].flip(0))
                lbl_s.append(1)
        pt_list.append(torch.stack(pt_s, dim=0))
        label_list.append(torch.tensor(lbl_s, device=msk.device))
    return torch.stack(pt_list, dim=-1), torch.stack(label_list, dim=-1).to(torch.int)


def _load_adv_images(folder_path: str, case_name: str, is_3d: bool = True):
    import glob as _glob
    if is_3d:
        case_dir = os.path.join(folder_path, case_name)
        if not os.path.isdir(case_dir):
            return None, None
        pngs = sorted(_glob.glob(os.path.join(case_dir, "*.png")),
                      key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        if not pngs:
            return None, None
        import matplotlib.image as mpimg
        slices = [mpimg.imread(p) for p in pngs]
        slices_2d = [s[:, :, 0] if s.ndim == 3 else s for s in slices]
        slice_indices = [int(os.path.splitext(os.path.basename(p))[0]) for p in pngs]
        return np.stack(slices_2d, axis=0), slice_indices
    else:
        file_base = os.path.splitext(os.path.basename(case_name))[0]
        png_path = os.path.join(folder_path, f"{file_base}.png")
        npy_path = os.path.join(folder_path, f"{file_base}.npy")
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            if arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
            return arr, None
        if os.path.exists(png_path):
            import matplotlib.image as mpimg
            img = mpimg.imread(png_path)
            if img.ndim == 3:
                img = np.transpose(img, (2, 0, 1))
            return img, None
        return None, None


def _extract_case_name(batch):
    if "case_name" in batch:
        cn = batch["case_name"]
        return cn[0] if isinstance(cn, (list, tuple)) else str(cn)
    if "image_meta_dict" in batch:
        meta = batch["image_meta_dict"]
        if isinstance(meta, dict) and "filename_or_obj" in meta:
            fn = meta["filename_or_obj"]
            return fn[0] if isinstance(fn, (list, tuple)) else str(fn)
    return "unknown"
