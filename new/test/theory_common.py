from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from medpy import metric
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
NEW_DIR = ROOT / "new"
MSA_DIR = ROOT / "Medical-SAM-Adapter-main"


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def append_sys_path(path: str | Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def device_from_arg(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def to_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_json(path: str | Path, payload: Dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=to_serializable)


def write_csv(path: str | Path, rows: Sequence[Dict]) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class SynapsePaths:
    volume_path: str = str(ROOT / "data/Synapse_preprocess/test_vol_h5")
    list_dir: str = str(ROOT / "data/Synapse_preprocess/lists")
    train_path: str = str(ROOT / "data/Synapse_preprocess/train_npz")


@dataclass
class ISICPaths:
    data_path: str = str(ROOT / "data/ISIC")


@dataclass
class REFUGEPaths:
    data_path: str = str(ROOT / "data/REFUGE-Multirater")


def build_norm_transform() -> transforms.Normalize:
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def build_2d_transforms(image_size: int):
    image_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    return image_tf, mask_tf


def get_2d_loader(dataset: str, image_size: int, num_workers: int = 0) -> DataLoader:
    append_sys_path(ROOT)
    image_tf, mask_tf = build_2d_transforms(image_size)
    args = SimpleNamespace(img_size=image_size)
    if dataset == "ISIC":
        from datasets.isic import ISIC2016

        db = ISIC2016(args=args, data_path=ISICPaths().data_path, mode="Test", transform=image_tf, transform_msk=mask_tf)
    elif dataset == "REFUGE":
        from datasets.refuge import REFUGE

        db = REFUGE(args=args, data_path=REFUGEPaths().data_path, mode="Test", transform=image_tf, transform_msk=mask_tf)
    else:
        raise ValueError(f"Unsupported 2D dataset: {dataset}")
    return DataLoader(db, batch_size=1, shuffle=False, num_workers=num_workers)


def extract_case_name(sampled_batch: Dict) -> str:
    if "case_name" in sampled_batch:
        case_name = sampled_batch["case_name"]
        if isinstance(case_name, (list, tuple)):
            return str(case_name[0])
        return str(case_name)
    if "image_meta_dict" in sampled_batch:
        meta = sampled_batch["image_meta_dict"]
        if isinstance(meta, dict) and "filename_or_obj" in meta:
            filename = meta["filename_or_obj"]
            if isinstance(filename, (list, tuple)):
                return str(filename[0])
            return str(filename)
    raise KeyError("Cannot infer case_name from sampled_batch")


def prepare_2d_image(sampled_batch: Dict, device: torch.device) -> Tuple[torch.Tensor, str]:
    image = sampled_batch["image"]
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    image = image.float().to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    case_name = extract_case_name(sampled_batch)
    return image, case_name


def save_2d_adv(case_dir: str | Path, case_name: str, image_tensor: np.ndarray) -> None:
    ensure_dir(case_dir)
    if image_tensor.ndim == 3 and image_tensor.shape[0] in (1, 3):
        chw = image_tensor
    elif image_tensor.ndim == 3 and image_tensor.shape[-1] in (1, 3):
        chw = np.transpose(image_tensor, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected 2D image shape: {image_tensor.shape}")
    np.save(Path(case_dir) / f"{case_name}.npy", np.transpose(chw, (1, 2, 0)))
    image_np = np.clip(np.transpose(chw, (1, 2, 0)), 0.0, 1.0)
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    from PIL import Image

    Image.fromarray((image_np * 255.0).astype(np.uint8)).save(Path(case_dir) / f"{case_name}.png")


def get_synapse_loader(volume_path: str, list_dir: str, num_workers: int = 0) -> DataLoader:
    append_sys_path(ROOT)
    from datasets.dataset_synapse import Synapse_dataset

    dataset = Synapse_dataset(base_dir=volume_path, list_dir=list_dir, split="test_vol")
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)


def load_source_sam(vit_name: str, ckpt: str, image_size: int, num_classes: int, device: torch.device):
    append_sys_path(ROOT)
    from segment_anything import sam_model_registry

    sam, _ = sam_model_registry[vit_name](
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    return sam.to(device).eval()


def load_samed_encoder(
    vit_name: str,
    sam_ckpt: str,
    lora_ckpt: str,
    image_size: int,
    num_classes: int,
    rank: int,
    device: torch.device,
):
    append_sys_path(ROOT)
    from importlib import import_module
    from segment_anything import sam_model_registry

    sam, _ = sam_model_registry[vit_name](
        image_size=image_size,
        num_classes=num_classes,
        checkpoint=sam_ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    pkg = import_module("sam_lora_image_encoder")
    ckpt_state = torch.load(lora_ckpt, map_location="cpu")
    lora_keys = sorted([k for k in ckpt_state.keys() if k.startswith("w_a_")])
    inferred_blocks = len(lora_keys) // 2 if lora_keys else len(sam.image_encoder.blocks)
    lora_layer = list(range(inferred_blocks))
    net = pkg.LoRA_Sam(sam, rank, lora_layer=lora_layer).to(device)
    net.load_lora_parameters(lora_ckpt)
    net.eval()
    return net, net.sam.image_encoder


def load_medsa_model(
    vit_name: str,
    sam_ckpt: str,
    msa_ckpt: str,
    image_size: int,
    multimask_output: int,
    device: torch.device,
):
    append_sys_path(MSA_DIR)
    from models.sam import sam_model_registry as msa_registry

    args = SimpleNamespace(
        dataset="Synapse",
        image_size=image_size,
        multimask_output=multimask_output,
        sam_ckpt=sam_ckpt,
        encoder=vit_name,
        mod="sam_adpt",
        mid_dim=None,
        thd=False,
        chunk=None,
    )
    model = msa_registry[vit_name](args, checkpoint=sam_ckpt).to(device)
    ckpt = torch.load(msa_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def prepare_synapse_slice(
    sampled_batch: Dict,
    slice_index: int,
    device: torch.device,
) -> Tuple[torch.Tensor, str]:
    img_tensor = sampled_batch["image"]
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(1)
    image_vol = img_tensor.squeeze(0).cpu().detach().numpy()
    if len(image_vol.shape) == 2:
        image_vol = image_vol[np.newaxis, np.newaxis, :, :]
    elif len(image_vol.shape) == 3:
        image_vol = image_vol[:, np.newaxis, :, :]
    input_image = image_vol[slice_index, :, :, :]
    input_image_torch = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
    if input_image_torch.shape[1] == 1:
        input_image_torch = repeat(input_image_torch, "b c h w -> b (repeat c) h w", repeat=3)
    case_name = sampled_batch["case_name"][0]
    return input_image_torch, case_name


def get_synapse_depth(sampled_batch: Dict) -> int:
    img = sampled_batch["image"]
    if img.dim() == 4:
        return int(img.shape[1])
    if img.dim() == 5:
        return int(img.shape[-1])
    if img.dim() == 3:
        return int(img.shape[0])
    raise ValueError(f"Unexpected Synapse image shape: {tuple(img.shape)}")


def select_synapse_slices(depth: int, max_slices: int) -> List[int]:
    if max_slices <= 0 or max_slices >= depth:
        return list(range(depth))
    if max_slices == 1:
        return [depth // 2]
    return np.linspace(0, depth - 1, num=max_slices, dtype=int).tolist()


def get_deep_layer_indices(num_layers: int, take_last_k: int) -> List[int]:
    start = max(0, num_layers - take_last_k)
    return list(range(start, num_layers))


def fft_radial_mask(
    h: int,
    w: int,
    device: torch.device,
    min_radius_ratio: float = 0.05,
    max_radius_ratio: float = 1.0,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=h, device=device),
        torch.linspace(-1.0, 1.0, steps=w, device=device),
        indexing="ij",
    )
    rr = torch.sqrt(xx.square() + yy.square())
    return (rr >= min_radius_ratio) & (rr <= max_radius_ratio)


def build_support_mask(
    source_freq: torch.Tensor,
    min_radius_ratio: float = 0.05,
    max_radius_ratio: float = 1.0,
    energy_quantile: float = 0.8,
) -> torch.Tensor:
    _, _, h, w = source_freq.shape
    radial_mask = fft_radial_mask(h, w, source_freq.device, min_radius_ratio, max_radius_ratio)
    amp = torch.abs(source_freq)
    amp_masked = amp[radial_mask.unsqueeze(0).unsqueeze(0).expand_as(amp)]
    if amp_masked.numel() == 0:
        return radial_mask.unsqueeze(0).unsqueeze(0).expand_as(amp)
    threshold = torch.quantile(amp_masked, energy_quantile)
    energy_mask = amp >= threshold
    return energy_mask & radial_mask.unsqueeze(0).unsqueeze(0)


def decompose_drift_energy(
    feat_s: torch.Tensor,
    feat_t: torch.Tensor,
    support_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    gs = torch.fft.fft2(feat_s, norm="ortho")
    gt = torch.fft.fft2(feat_t, norm="ortho")
    delta = gt - gs
    carrier = torch.exp(1j * torch.angle(gs))
    local = delta * torch.conj(carrier)
    alpha = torch.real(local)
    beta = torch.imag(local)
    if support_mask is None:
        support_mask = torch.ones_like(alpha, dtype=torch.bool)
    mask = support_mask.to(alpha.device)
    alpha_masked = alpha[mask]
    beta_masked = beta[mask]
    radial_energy = torch.sum(alpha_masked.square()).item()
    tangential_energy = torch.sum(beta_masked.square()).item()
    total = radial_energy + tangential_energy
    radial_ratio = radial_energy / (total + eps)
    tangential_ratio = tangential_energy / (total + eps)
    angle = torch.atan2(beta_masked, alpha_masked + eps)
    return {
        "radial_energy": float(radial_energy),
        "tangential_energy": float(tangential_energy),
        "radial_ratio": float(radial_ratio),
        "tangential_ratio": float(tangential_ratio),
        "angle_mean": float(torch.mean(angle).item() if angle.numel() else 0.0),
        "angle_std": float(torch.std(angle).item() if angle.numel() > 1 else 0.0),
        "num_coeffs": int(alpha_masked.numel()),
    }


def calculate_metric_percase(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = pred.copy()
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        return float(metric.binary.dc(pred, gt)), float(metric.binary.hd95(pred, gt))
    if pred.sum() > 0 and gt.sum() == 0:
        return 1.0, 0.0
    return 0.0, 0.0


def calculate_iou_percase(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.copy() > 0
    gt = gt.copy() > 0
    union = float((pred | gt).sum())
    if union == 0:
        return 1.0
    return float((pred & gt).sum()) / union


def thresholded_sigmoid_metrics(
    pred_logits: torch.Tensor,
    true_mask: torch.Tensor,
    dataset: str,
    thresholds: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> Dict[str, float]:
    probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    mask_np = true_mask.detach().cpu().numpy()
    if dataset == "ISIC":
        dices = []
        hd95s = []
        ious = []
        for th in thresholds:
            pred_bin = (probs[:, 0] > th).astype(np.uint8)
            gt_bin = (mask_np[:, 0] > th).astype(np.uint8)
            for pred_case, gt_case in zip(pred_bin, gt_bin):
                dice, hd95 = calculate_metric_percase(pred_case, gt_case)
                dices.append(dice)
                hd95s.append(hd95)
                ious.append(calculate_iou_percase(pred_case, gt_case))
        return {
            "mean_dice": float(np.mean(dices)),
            "mean_hd95": float(np.mean(hd95s)),
            "mean_iou": float(np.mean(ious)),
        }
    if dataset == "REFUGE":
        disc_dices = []
        cup_dices = []
        disc_hd95s = []
        cup_hd95s = []
        disc_ious = []
        cup_ious = []
        for th in thresholds:
            pred_bin = (probs > th).astype(np.uint8)
            gt_bin = (mask_np > th).astype(np.uint8)
            disc_pred = pred_bin[:, 1]
            cup_pred = pred_bin[:, 0]
            disc_gt = gt_bin[:, 1]
            cup_gt = gt_bin[:, 0]
            for dp, dg, cp, cg in zip(disc_pred, disc_gt, cup_pred, cup_gt):
                d_dice, d_hd95 = calculate_metric_percase(dp, dg)
                c_dice, c_hd95 = calculate_metric_percase(cp, cg)
                disc_dices.append(d_dice)
                cup_dices.append(c_dice)
                disc_hd95s.append(d_hd95)
                cup_hd95s.append(c_hd95)
                disc_ious.append(calculate_iou_percase(dp, dg))
                cup_ious.append(calculate_iou_percase(cp, cg))
        return {
            "mean_dice": float((np.mean(disc_dices) + np.mean(cup_dices)) / 2.0),
            "mean_hd95": float((np.mean(disc_hd95s) + np.mean(cup_hd95s)) / 2.0),
            "mean_iou": float((np.mean(disc_ious) + np.mean(cup_ious)) / 2.0),
            "disc_dice": float(np.mean(disc_dices)),
            "cup_dice": float(np.mean(cup_dices)),
            "disc_hd95": float(np.mean(disc_hd95s)),
            "cup_hd95": float(np.mean(cup_hd95s)),
        }
    raise ValueError(f"Unsupported dataset for thresholded metrics: {dataset}")


def repack_synapse_png_to_h5(src_dir: str | Path, dst_dir: str | Path, gt_volume_path: str | Path) -> None:
    ensure_dir(dst_dir)
    for case_dir in sorted(Path(src_dir).glob("case*")):
        pngs = sorted(case_dir.glob("*.png"), key=lambda p: int(p.stem))
        if not pngs:
            continue
        slices = []
        for png_path in pngs:
            from PIL import Image

            img = np.array(Image.open(png_path))
            if img.ndim == 3:
                img = img[..., 0]
            slices.append(img.astype(np.float32) / 255.0)
        image_np = np.stack(slices, axis=0)
        if not (0.0 <= float(image_np.min()) and float(image_np.max()) <= 1.0):
            raise ValueError(f"{case_dir.name}: adversarial volume violates [0,1] input scale")
        with h5py.File(Path(gt_volume_path) / f"{case_dir.name}.npy.h5", "r") as f:
            label_np = f["label"][:]
        with h5py.File(Path(dst_dir) / f"{case_dir.name}.npy.h5", "w") as f:
            f.create_dataset("image", data=image_np)
            f.create_dataset("label", data=label_np)


def medsa_generate_click_prompt(msk: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if msk.dim() == 3:
        msk = F.one_hot(msk.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    if msk.dim() == 4:
        msk = msk.unsqueeze(1).permute(0, 1, 3, 4, 2)
    b, _, h, w, d = msk.size()
    msk = msk[:, 0, :, :, :]
    pt_list = []
    label_list = []
    for i in range(d):
        pt_slice = []
        label_slice = []
        for j in range(b):
            msk_s = msk[j, :, :, i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                random_index = torch.tensor([0, 0], device=msk.device)
                lbl = 0
            else:
                random_index = indices[random.randrange(indices.size(0))].flip(0)
                lbl = 1
            pt_slice.append(random_index)
            label_slice.append(lbl)
        pt_list.append(torch.stack(pt_slice, dim=0))
        label_list.append(torch.tensor(label_slice, device=msk.device))
    pt = torch.stack(pt_list, dim=-1)
    labels = torch.stack(label_list, dim=-1).to(torch.int)
    return pt, labels


def medsa_eval_seg(pred: torch.Tensor, true_mask: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    pred_classes = torch.argmax(pred, dim=1).cpu().numpy()
    true_mask_np = true_mask.squeeze(1).cpu().numpy()
    dice_list: List[float] = []
    hd95_list: List[float] = []
    iou_list: List[float] = []
    for i in range(1, num_classes):
        pred_i = pred_classes == i
        true_i = true_mask_np == i
        intersection = float((pred_i & true_i).sum())
        union = float(pred_i.sum() + true_i.sum())
        if union == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = 2.0 * intersection / union
            area_union = union - intersection
            iou = 1.0 if area_union == 0 else intersection / area_union
        dice_list.append(dice)
        iou_list.append(iou)
        if pred_i.sum() > 0 and true_i.sum() > 0:
            hd95_list.append(float(metric.binary.hd95(pred_i, true_i)))
    if not dice_list:
        return 0.0, 0.0, 0.0
    return float(np.mean(dice_list)), float(np.mean(hd95_list) if hd95_list else 0.0), float(np.mean(iou_list))


def evaluate_medsa_synapse_case(
    model,
    image: torch.Tensor,
    label: torch.Tensor,
    device: torch.device,
    multimask_output: int = 9,
    image_size: int = 512,
    out_size: int = 512,
    evl_chunk: int = 1,
) -> Dict[str, float]:
    model.eval()
    imgsw = image.to(dtype=torch.float32, device=device)
    masksw = label.to(dtype=torch.float32, device=device)
    total_dice = 0.0
    total_hd95 = 0.0
    total_iou = 0.0
    real_val_step = 0
    buoy = 0
    while (buoy + evl_chunk) <= imgsw.size(-1):
        imgs = imgsw[..., buoy : buoy + evl_chunk]
        masks = masksw[..., buoy : buoy + evl_chunk]
        buoy += evl_chunk
        real_val_step += 1
        if imgs.dim() == 5:
            b, c, h, w, d = imgs.size()
            imgs = imgs.permute(0, 4, 1, 2, 3).contiguous().view(-1, c, h, w)
            masks = masks.permute(0, 4, 1, 2, 3).contiguous().view(-1, 1, h, w)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)
        pt, point_labels = medsa_generate_click_prompt(masks.squeeze(1), multimask_output)
        pt = pt.permute(0, 2, 1)
        showp = pt[..., [1, 0]]
        coords_torch = torch.as_tensor(pt, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
        if len(point_labels.shape) == 1:
            coords_torch = coords_torch.unsqueeze(1)
            labels_torch = labels_torch.unsqueeze(1)
            showp = showp.unsqueeze(1)
        prompt = (coords_torch, labels_torch)
        imgs = imgs.to(device=device)
        with torch.no_grad():
            imgs_pre = model.preprocess(imgs)
            image_embeddings = model.image_encoder(imgs_pre)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=prompt, boxes=None, masks=None)
            pred, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=(multimask_output > 1),
            )
            pred = F.interpolate(pred, size=(out_size, out_size), mode="bilinear", align_corners=False)
            dice, hd95, iou = medsa_eval_seg(pred, masks, multimask_output)
            total_dice += dice
            total_hd95 += hd95
            total_iou += iou
    if real_val_step == 0:
        real_val_step = 1
    return {
        "mean_dice": total_dice / real_val_step,
        "mean_hd95": total_hd95 / real_val_step,
        "mean_iou": total_iou / real_val_step,
    }


def maybe_wilcoxon(x: Sequence[float], y: Sequence[float]) -> Dict[str, float | str]:
    if len(x) != len(y) or len(x) == 0:
        return {"test": "wilcoxon", "pvalue": "nan", "statistic": "nan"}
    diffs = [float(a) - float(b) for a, b in zip(x, y)]
    if all(abs(d) < 1e-12 for d in diffs):
        return {
            "test": "wilcoxon",
            "statistic": 0.0,
            "pvalue": 1.0,
            "note": "all paired differences are zero",
        }
    try:
        from scipy.stats import wilcoxon

        statistic, pvalue = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return {"test": "wilcoxon", "statistic": float(statistic), "pvalue": float(pvalue)}
    except Exception as exc:
        return {"test": "wilcoxon", "error": str(exc)}


def summarize_paired_differences(
    left: Sequence[float],
    right: Sequence[float],
    left_name: str,
    right_name: str,
) -> Dict[str, object]:
    if len(left) != len(right) or len(left) == 0:
        return {
            "left": left_name,
            "right": right_name,
            "num_pairs": 0,
            "delta_mean": "nan",
            "delta_median": "nan",
            "delta_abs_mean": "nan",
            "num_positive": 0,
            "num_negative": 0,
            "num_zero": 0,
            "wilcoxon": maybe_wilcoxon(left, right),
        }
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    delta = left_arr - right_arr
    return {
        "left": left_name,
        "right": right_name,
        "num_pairs": int(delta.size),
        "left_mean": float(np.mean(left_arr)),
        "right_mean": float(np.mean(right_arr)),
        "delta_mean": float(np.mean(delta)),
        "delta_median": float(np.median(delta)),
        "delta_abs_mean": float(np.mean(np.abs(delta))),
        "num_positive": int(np.sum(delta > 1e-12)),
        "num_negative": int(np.sum(delta < -1e-12)),
        "num_zero": int(np.sum(np.abs(delta) <= 1e-12)),
        "wilcoxon": maybe_wilcoxon(left_arr.tolist(), right_arr.tolist()),
    }


def build_qc_flags(
    clean_dice: Sequence[float],
    adv_dice: Sequence[float],
    clean_hd95: Sequence[float],
    adv_hd95: Sequence[float],
) -> Dict[str, object]:
    clean_dice_arr = np.asarray(clean_dice, dtype=np.float64)
    adv_dice_arr = np.asarray(adv_dice, dtype=np.float64)
    clean_hd95_arr = np.asarray(clean_hd95, dtype=np.float64)
    adv_hd95_arr = np.asarray(adv_hd95, dtype=np.float64)
    if clean_dice_arr.size == 0 or adv_dice_arr.size == 0:
        return {
            "num_pairs": 0,
            "any_attack_improves_dice": False,
            "any_attack_reduces_hd95": False,
            "all_dice_equal": False,
            "all_hd95_equal": False,
            "dice_saturated_floor": False,
            "dice_saturated_ceiling": False,
        }
    return {
        "num_pairs": int(min(clean_dice_arr.size, adv_dice_arr.size)),
        "any_attack_improves_dice": bool(np.any(adv_dice_arr > clean_dice_arr + 1e-12)),
        "any_attack_reduces_hd95": bool(np.any(adv_hd95_arr < clean_hd95_arr - 1e-12)),
        "all_dice_equal": bool(np.all(np.abs(adv_dice_arr - clean_dice_arr) <= 1e-12)),
        "all_hd95_equal": bool(np.all(np.abs(adv_hd95_arr - clean_hd95_arr) <= 1e-12)),
        "dice_saturated_floor": bool(np.all(adv_dice_arr <= 1e-6)),
        "dice_saturated_ceiling": bool(np.all(adv_dice_arr >= 1.0 - 1e-6)),
    }
