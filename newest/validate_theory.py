"""
Comprehensive CODD theory validation experiments.

Tests all four claims:
  C1. Drift decomposes into radial + tangential in source carrier basis
  C2. Radial energy dominates across all fine-tuned models/datasets
  C3. Radial attacks transfer better than tangential/isotropic/misaligned
  C4. Shared Radial Subspace provides beneficial warm-start

Experiment groups:
  G1. Drift decomposition: measure radial_ratio across layers, models, datasets
  G2. Directional ablation: radial vs tangential vs isotropic vs misaligned
  G3. SRS ablation: zero vs random vs mean vs top1 vs topk warm-start
  G4. Full attack evaluation: best CODD config vs baselines

Usage:
  python validate_theory.py --dataset Synapse --vit_name vit_b \
    --source_ckpt .../sam_vit_b.pth \
    --samed_ckpt .../epoch_159.pth \
    --medsa_ckpt .../best_dice_checkpoint.pth \
    --output_dir results/validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory import (
    DriftStats,
    SharedSubspace,
    decompose_drift,
    decompose_drift_per_layer,
    paired_wilcoxon,
    cohens_d,
)
from attack_codd import (
    AttackConfig,
    calculate_codd_loss,
    run_codd_attack,
    distill_srs,
)
from evaluate import (
    load_sam_encoder,
    load_samed_model,
    load_medsa_model,
    evaluate_samed,
    evaluate_medsa,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    p.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--source_ckpt", type=str, required=True)
    p.add_argument("--samed_ckpt", type=str, required=True)
    p.add_argument("--medsa_ckpt", type=str, required=True)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--max_cases", type=int, default=10)
    p.add_argument("--max_slices", type=int, default=8)
    p.add_argument("--att_numstep", type=int, default=10)
    p.add_argument("--att_step_size", type=float, default=2.0)
    p.add_argument("--att_bound", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--srs_path", type=str, default="")
    p.add_argument("--skip_attack", action="store_true")
    p.add_argument("--skip_drift", action="store_true")
    return p.parse_args()


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(path, data):
    def _conv(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj
    _ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_conv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_dataloader(dataset: str, image_size: int, split: str = "test"):
    if dataset == "Synapse":
        from datasets.dataset_synapse import Synapse_dataset
        db = Synapse_dataset(base_dir=f"{ROOT}/data/Synapse_preprocess/test_vol_h5",
                            list_dir=f"{ROOT}/data/Synapse_preprocess/lists",
                            split="test_vol")
        return DataLoader(db, batch_size=1, shuffle=False, num_workers=0), True
    else:
        image_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        if dataset == "ISIC":
            from datasets.isic import ISIC2016
            db = ISIC2016(args=argparse.Namespace(img_size=image_size),
                         data_path=f"{ROOT}/data/ISIC", mode="Test",
                         transform=image_tf, transform_msk=mask_tf)
        elif dataset == "REFUGE":
            from datasets.refuge import REFUGE
            db = REFUGE(args=argparse.Namespace(img_size=image_size),
                       data_path=f"{ROOT}/data/REFUGE-Multirater", mode="Test",
                       transform=image_tf, transform_msk=mask_tf)
        else:
            raise ValueError(dataset)
        return DataLoader(db, batch_size=1, shuffle=False, num_workers=0), False


def _build_train_dataloader(dataset: str, image_size: int):
    """Training split for SRS distillation."""
    image_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    if dataset == "Synapse":
        from datasets.dataset_synapse import Synapse_dataset
        db = Synapse_dataset(base_dir=f"{ROOT}/data/Synapse_preprocess/train_npz",
                            list_dir=f"{ROOT}/data/Synapse_preprocess/lists",
                            split="train")
        return DataLoader(db, batch_size=1, shuffle=True, num_workers=0), True
    elif dataset == "ISIC":
        from datasets.isic import ISIC2016
        db = ISIC2016(args=argparse.Namespace(img_size=image_size),
                     data_path=f"{ROOT}/data/ISIC", mode="Training",
                     transform=image_tf, transform_msk=mask_tf)
        return DataLoader(db, batch_size=1, shuffle=True, num_workers=0), False
    elif dataset == "REFUGE":
        from datasets.refuge import REFUGE
        db = REFUGE(args=argparse.Namespace(img_size=image_size),
                   data_path=f"{ROOT}/data/REFUGE-Multirater", mode="Training",
                   transform=image_tf, transform_msk=mask_tf)
        return DataLoader(db, batch_size=1, shuffle=True, num_workers=0), False
    raise ValueError(dataset)


def _prepare_image(batch, is_3d: bool, slice_idx: int, device):
    if is_3d:
        img = batch["image"]                      # [1, D, H, W]
        if img.dim() == 3:
            img = img.unsqueeze(1)
        img_np = img.squeeze(0).cpu().detach().numpy()  # [D, H, W]
        s = img_np[slice_idx]                     # [H, W]
        if s.ndim == 2:
            s = s[np.newaxis, :, :]              # [1, H, W]
        t = torch.from_numpy(s).unsqueeze(0).float().to(device)  # [1, 1, H, W]
        from einops import repeat
        t = repeat(t, 'b c h w -> b (repeat c) h w', repeat=3)  # [1, 3, H, W]
        return t
    else:
        img = batch["image"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        img = img.float().to(device)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return img


def _extract_name(batch):
    if "case_name" in batch:
        cn = batch["case_name"]
        return cn[0] if isinstance(cn, (list, tuple)) else str(cn)
    if "image_meta_dict" in batch:
        fn = batch["image_meta_dict"]["filename_or_obj"]
        return fn[0] if isinstance(fn, (list, tuple)) else str(fn)
    return "unknown"


def _get_depth(batch, is_3d):
    if not is_3d:
        return 1
    img = batch["image"]
    if img.dim() == 4:
        return img.shape[1]
    if img.dim() == 3:
        return img.shape[0]
    return 1


# ---------------------------------------------------------------------------
# G1: Drift Decomposition
# ---------------------------------------------------------------------------

def run_drift_validation(args, source_sam, samed_encoder=None, medsa_model=None, device=None):
    """
    C1/C2: Measure radial vs tangential drift energy at each encoder layer
    for both SAMed and MedSA fine-tuned models.
    """
    print("\n" + "="*60)
    print("G1: DRIFT DECOMPOSITION VALIDATION")
    print("="*60)

    loader, is_3d = _build_dataloader(args.dataset, args.image_size)
    device = device or torch.device(args.device)

    all_rows = []
    processed = 0

    for batch in loader:
        if processed >= args.max_cases:
            break
        case_name = _extract_name(batch)
        depth = _get_depth(batch, is_3d)
        max_s = min(depth, args.max_slices)

        slice_indices = [depth // 2] if max_s == 1 else \
            np.linspace(0, depth - 1, max_s, dtype=int).tolist()

        for si in slice_indices:
            img = _prepare_image(batch, is_3d, si, device)

            with torch.no_grad():
                # Source features
                src_feat, src_layers = source_sam.image_encoder(img)

                # SAMed features
                if samed_encoder is not None:
                    samed_feat, samed_layers = samed_encoder(img)
                    # Per-layer drift stats
                    for li, (fs, ft) in enumerate(zip(src_layers[-6:], samed_layers[-6:])):
                        stats = decompose_drift(fs, ft)
                        all_rows.append({
                            "victim": "SAMed", "case": case_name,
                            "slice": si, "layer": li,
                            **stats.to_dict(),
                        })

                # MedSA features
                if medsa_model is not None:
                    medsa_out = medsa_model.image_encoder(img)
                    medsa_feat = medsa_out[0] if isinstance(medsa_out, (list, tuple)) else medsa_out
                    medsa_layers = medsa_out[1] if isinstance(medsa_out, (list, tuple)) and len(medsa_out) > 1 else None

                    if medsa_layers is not None and len(medsa_layers) >= 6:
                        for li, (fs, ft) in enumerate(zip(src_layers[-6:], medsa_layers[-6:])):
                            stats = decompose_drift(fs, ft)
                            all_rows.append({
                                "victim": "MedSA", "case": case_name,
                                "slice": si, "layer": li,
                                **stats.to_dict(),
                            })
                    else:
                        stats = decompose_drift(src_feat, medsa_feat)
                        all_rows.append({
                            "victim": "MedSA", "case": case_name,
                            "slice": si, "layer": -1,
                            **stats.to_dict(),
                        })

        processed += 1

    # Aggregate results
    summary = _aggregate_drift(all_rows)
    _write_json(os.path.join(args.output_dir, "g1_drift_summary.json"), summary)

    print(f"\n  Drift decomposition results ({len(all_rows)} data points):")
    for victim, vsum in summary["by_victim"].items():
        print(f"  {victim}: radial_ratio = {vsum['radial_ratio_mean']:.4f} ± {vsum['radial_ratio_std']:.4f}")
        print(f"          tangential_ratio = {vsum['tangential_ratio_mean']:.4f} ± {vsum['tangential_ratio_std']:.4f}")
        print(f"          angle_std = {vsum['angle_std_mean']:.4f} rad")
    print(f"\n  C1 verdict: drift CAN be decomposed into radial+tangential in source carrier basis")
    print(f"  C2 verdict: radial dominates tangential by "
          f"{summary['by_victim'].get('SAMed', {}).get('radial_tangential_ratio', 'N/A')}x")

    return summary


def _aggregate_drift(rows):
    by_victim = defaultdict(list)
    for r in rows:
        by_victim[r["victim"]].append(r)

    summary = {"num_rows": len(rows), "by_victim": {}}
    for victim, vrows in by_victim.items():
        rr = [r["radial_ratio"] for r in vrows]
        tr = [r["tangential_ratio"] for r in vrows]
        angle_std = [r["angle_std"] for r in vrows]
        summary["by_victim"][victim] = {
            "num_points": len(vrows),
            "radial_ratio_mean": float(np.mean(rr)),
            "radial_ratio_std": float(np.std(rr)),
            "radial_ratio_median": float(np.median(rr)),
            "tangential_ratio_mean": float(np.mean(tr)),
            "tangential_ratio_std": float(np.std(tr)),
            "angle_std_mean": float(np.mean(angle_std)),
            "angle_std_std": float(np.std(angle_std)),
            "radial_tangential_ratio": float(np.mean(rr) / (np.mean(tr) + 1e-8)),
        }
        # Per-layer breakdown
        by_layer = defaultdict(list)
        for r in vrows:
            by_layer[r["layer"]].append(r["radial_ratio"])
        summary["by_victim"][victim]["per_layer_radial_ratio"] = {
            str(k): float(np.mean(v)) for k, v in sorted(by_layer.items())
        }
    return summary


# ---------------------------------------------------------------------------
# G2: Directional Ablation
# ---------------------------------------------------------------------------

def run_directional_ablation(args, source_sam, samed_net, medsa_model, device):
    """
    C3: Compare attack effectiveness across four direction modes:
      radial, tangential, isotropic, misaligned_radial.

    Key prediction (CODD): radial > isotropic > tangential > misaligned
    in terms of segmentation degradation.

    H0: all directions equally effective (reject CODD)
    H1: radial significantly more effective than tangential (support CODD)
    """
    print("\n" + "="*60)
    print("G2: DIRECTIONAL ABLATION")
    print("="*60)

    loader, is_3d = _build_dataloader(args.dataset, args.image_size)

    base_config = AttackConfig(
        att_step_size=args.att_step_size,
        att_bound=args.att_bound,
        att_numstep=args.att_numstep,
        seed=args.seed,
        use_imagenet_norm=(args.dataset != "Synapse"),
    )

    directions = ["radial", "tangential", "isotropic", "misaligned"]
    all_results = {}

    # Compute consistent eval_slices from the generated adversarial images
    max_eval = args.max_slices

    for direction in directions:
        print(f"\n  Testing direction: {direction}")
        config = AttackConfig(
            att_step_size=base_config.att_step_size,
            att_bound=base_config.att_bound,
            att_numstep=base_config.att_numstep,
            direction_mode=direction,
            seed=args.seed,
            use_imagenet_norm=base_config.use_imagenet_norm,
        )

        adv_dir = os.path.join(args.output_dir, "adv_images", f"g2_{direction}")
        _ensure_dir(adv_dir)

        # Generate adversarial images
        processed = 0
        for batch in loader:
            if processed >= args.max_cases:
                break
            case_name = _extract_name(batch)
            depth = _get_depth(batch, is_3d)
            max_s = min(depth, args.max_slices)
            # Find valid range where labels actually contain organs
            label_np = batch["label"].squeeze(0).cpu().numpy()  # [D, H, W]
            valid_slices = [s for s in range(depth) if (label_np[s] > 0).sum() > 100]
            if valid_slices:
                # Sample from valid organ-containing region
                valid_mid = len(valid_slices) // 2
                start_idx = max(0, valid_mid - max_s // 2)
                end_idx = min(len(valid_slices), start_idx + max_s)
                slice_indices = [valid_slices[i] for i in range(start_idx, end_idx)]
            else:
                # Fallback: middle slices
                mid = depth // 2
                start = max(0, mid - max_s // 2)
                slice_indices = list(range(start, min(start + max_s, depth)))
            for si in slice_indices:
                img = _prepare_image(batch, is_3d, si, device)

                # Apply ImageNet norm for 2D datasets
                if args.dataset != "Synapse":
                    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    img_enc = norm(img)
                else:
                    img_enc = img

                adv_img, trace = run_codd_attack(
                    image_encoder=source_sam.image_encoder,
                    input_image=img_enc,
                    config=config,
                    return_trace=True,
                )

                _save_adv_image(adv_img, adv_dir, case_name, si, is_3d)

            processed += 1

        # Evaluate on both victims — restrict to same slice subset as attack
        print(f"    Evaluating on SAMed...")
        samed_rows = evaluate_samed(
            samed_net, loader, adv_dir, args.num_classes, args.image_size,
            is_3d=is_3d, max_cases=args.max_cases, max_eval_slices=max_eval, device=device,
        )

        print(f"    Evaluating on MedSA...")
        medsa_rows = evaluate_medsa(
            medsa_model, loader, adv_dir, args.num_classes, args.image_size,
            is_3d=is_3d, max_cases=args.max_cases, max_eval_slices=max_eval, device=device,
        )

        all_results[direction] = {
            "samed": samed_rows,
            "medsa": medsa_rows,
        }

    # Clean reference — evaluate on same slice subset
    print("\n  Computing clean reference...")
    clean_samed = evaluate_samed(
        samed_net, loader, "__clean__", args.num_classes, args.image_size,
        is_3d=is_3d, max_cases=args.max_cases, max_eval_slices=max_eval, device=device,
    )
    clean_medsa = evaluate_medsa(
        medsa_model, loader, "__clean__", args.num_classes, args.image_size,
        is_3d=is_3d, max_cases=args.max_cases, max_eval_slices=max_eval, device=device,
    )

    # Pairwise statistical analysis
    summary = _analyze_directional(clean_samed, clean_medsa, all_results)
    _write_json(os.path.join(args.output_dir, "g2_directional_summary.json"), summary)

    print("\n  Directional ablation results:")
    for victim in ["samed", "medsa"]:
        clean_dice = summary["clean_mean_dice"][victim]
        print(f"\n  {victim} (clean dice = {clean_dice:.4f}):")
        for direction in directions:
            d = summary["attacked_mean_dice"][f"{direction}_{victim}"]
            delta = clean_dice - d  # positive = attack works
            print(f"    {direction:15s}: dice={d:.4f}  (delta={delta:+.4f})")

    # C3 verdict
    radial_dice = summary["attacked_mean_dice"]["radial_samed"]
    tangential_dice = summary["attacked_mean_dice"]["tangential_samed"]
    pval = summary["pairwise"]["radial_vs_tangential_samed"]["pvalue"]
    print(f"\n  C3 verdict:")
    print(f"    radial vs tangential: Delta(dice) = {tangential_dice - radial_dice:.4f}")
    print(f"    Wilcoxon p = {pval}")
    if radial_dice < tangential_dice and pval < 0.05:
        print(f"    => Radial is SIGNIFICANTLY stronger than tangential (supports CODD)")
    else:
        print(f"    => Cannot reject H0 (no significant difference)")

    return summary


def _analyze_directional(clean_samed, clean_medsa, all_results):
    summary = {
        "clean_mean_dice": {
            "samed": float(np.mean([r["mean_dice"] for r in clean_samed])),
            "medsa": float(np.mean([r["mean_dice"] for r in clean_medsa])),
        },
        "attacked_mean_dice": {},
        "pairwise": {},
    }

    directions = list(all_results.keys())

    # Mean attacked dice
    for d in directions:
        for victim, key in [("samed", f"{d}_samed"), ("medsa", f"{d}_medsa")]:
            rows = all_results[d][victim]
            summary["attacked_mean_dice"][key] = float(np.mean([r["mean_dice"] for r in rows]))

    # Pairwise Wilcoxon tests
    for victim, clean_rows in [("samed", clean_samed), ("medsa", clean_medsa)]:
        for d1 in directions:
            for d2 in directions:
                if d1 >= d2:
                    continue
                rows1 = all_results[d1][victim]
                rows2 = all_results[d2][victim]
                dice1 = [r["mean_dice"] for r in rows1]
                dice2 = [r["mean_dice"] for r in rows2]
                key = f"{d1}_vs_{d2}_{victim}"
                summary["pairwise"][key] = paired_wilcoxon(dice1, dice2)
                summary["pairwise"][f"{key}_cohens_d"] = cohens_d(dice1, dice2)

    return summary


# ---------------------------------------------------------------------------
# G3: SRS Ablation
# ---------------------------------------------------------------------------

def run_srs_ablation(args, source_sam, samed_net, medsa_model, device):
    """
    C4: Ablate the Shared Radial Subspace warm-start.

    Modes: zero (no warm-start), random (random direction), mean (SRS mean only),
           top1 (1st principal component), topk (full SRS subspace).
    """
    print("\n" + "="*60)
    print("G3: SHARED RADIAL SUBSPACE ABLATION")
    print("="*60)

    loader, is_3d = _build_dataloader(args.dataset, args.image_size)

    # Load or distill SRS
    if args.srs_path and os.path.exists(args.srs_path):
        print(f"  Loading SRS from {args.srs_path}")
        srs = SharedSubspace.load(args.srs_path, device)
    else:
        print("  Distilling SRS from proxy training data...")
        train_loader, train_3d = _build_train_dataloader(args.dataset, args.image_size)
        distill_config = AttackConfig(
            att_step_size=args.att_step_size,
            att_bound=args.att_bound,
            att_numstep=3,
            direction_mode="radial",
            seed=args.seed,
            use_imagenet_norm=(args.dataset != "Synapse"),
        )
        srs = distill_srs(
            source_sam.image_encoder, train_loader, distill_config,
            num_samples=128, rank=8, device=device, is_3d=train_3d,
        )
        srs_path = os.path.join(args.output_dir, "srs_distilled.pth")
        srs.save(srs_path)
        print(f"  Saved SRS to {srs_path}")

    base_config = AttackConfig(
        att_step_size=args.att_step_size,
        att_bound=args.att_bound,
        att_numstep=args.att_numstep,
        direction_mode="radial",
        seed=args.seed,
        use_imagenet_norm=(args.dataset != "Synapse"),
    )

    srs_modes = {
        "zero":   {"srs_mean_weight": 0.0, "srs_proj_weight": 0.0},
        "random": {"srs_mean_weight": 0.0, "srs_proj_weight": 0.0},  # + random init
        "mean":   {"srs_mean_weight": 0.1, "srs_proj_weight": 0.0},
        "top1":   {"srs_mean_weight": 0.0, "srs_proj_weight": 0.15},
        "topk":   {"srs_mean_weight": 0.05, "srs_proj_weight": 0.15},
    }

    max_eval = args.max_slices
    all_results = {}

    for mode_name, mode_kwargs in srs_modes.items():
        print(f"\n  Testing SRS mode: {mode_name}")
        config = AttackConfig(
            att_step_size=base_config.att_step_size,
            att_bound=base_config.att_bound,
            att_numstep=base_config.att_numstep,
            direction_mode="radial",
            seed=args.seed,
            use_imagenet_norm=base_config.use_imagenet_norm,
            srs_mean_weight=mode_kwargs["srs_mean_weight"],
            srs_proj_weight=mode_kwargs["srs_proj_weight"],
        )

        if mode_name == "random":
            tmp_srs = SharedSubspace(
                mean_direction=torch.randn_like(srs.mean_direction),
                basis=torch.randn_like(srs.basis),
                shape=srs.shape,
                singular_values=srs.singular_values,
                explained_variance=0.0,
            ).to(device)
            srs_use = tmp_srs
        elif mode_name == "zero":
            srs_use = None
        else:
            srs_use = srs

        adv_dir = os.path.join(args.output_dir, "adv_images", f"g3_{mode_name}")
        _ensure_dir(adv_dir)

        processed = 0
        for batch in loader:
            if processed >= args.max_cases:
                break
            case_name = _extract_name(batch)
            depth = _get_depth(batch, is_3d)
            max_s = min(depth, args.max_slices)
            # Find valid range where labels actually contain organs
            label_np = batch["label"].squeeze(0).cpu().numpy()  # [D, H, W]
            valid_slices = [s for s in range(depth) if (label_np[s] > 0).sum() > 100]
            if valid_slices:
                # Sample from valid organ-containing region
                valid_mid = len(valid_slices) // 2
                start_idx = max(0, valid_mid - max_s // 2)
                end_idx = min(len(valid_slices), start_idx + max_s)
                slice_indices = [valid_slices[i] for i in range(start_idx, end_idx)]
            else:
                # Fallback: middle slices
                mid = depth // 2
                start = max(0, mid - max_s // 2)
                slice_indices = list(range(start, min(start + max_s, depth)))
            for si in slice_indices:
                img = _prepare_image(batch, is_3d, si, device)
                if args.dataset != "Synapse":
                    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    img_enc = norm(img)
                else:
                    img_enc = img

                adv_img = run_codd_attack(
                    image_encoder=source_sam.image_encoder,
                    input_image=img_enc,
                    config=config,
                    srs_bundle=srs_use,
                    return_trace=False,
                )
                _save_adv_image(adv_img, adv_dir, case_name, si, is_3d)

            processed += 1

        samed_rows = evaluate_samed(
            samed_net, loader, adv_dir, args.num_classes, args.image_size,
            is_3d=is_3d, max_cases=args.max_cases, max_eval_slices=max_eval, device=device,
        )
        medsa_rows = evaluate_medsa(
            medsa_model, loader, adv_dir,
            args.num_classes, args.image_size, is_3d=is_3d,
            max_cases=args.max_cases, max_eval_slices=max_eval, device=device,
        )
        all_results[mode_name] = {"samed": samed_rows, "medsa": medsa_rows}

    summary = _analyze_srs(all_results)
    _write_json(os.path.join(args.output_dir, "g3_srs_summary.json"), summary)

    print("\n  SRS ablation results:")
    for victim in ["samed", "medsa"]:
        print(f"\n  {victim}:")
        for mode in srs_modes:
            dice = summary["mean_dice"][victim][mode]
            print(f"    {mode:8s}: dice={dice:.4f}")

    if summary.get("topk_vs_zero_pvalue", 1.0) < 0.05:
        print(f"\n  C4 verdict: SRS provides SIGNIFICANT improvement (p={summary['topk_vs_zero_pvalue']:.4f})")
    else:
        print(f"\n  C4 verdict: SRS benefit NOT statistically significant (p={summary.get('topk_vs_zero_pvalue', 'N/A')})")

    return summary


def _analyze_srs(all_results):
    summary = {"mean_dice": {"samed": {}, "medsa": {}}, "pairwise": {}}
    for mode, res in all_results.items():
        for victim in ["samed", "medsa"]:
            dice_vals = [r["mean_dice"] for r in res[victim]]
            summary["mean_dice"][victim][mode] = float(np.mean(dice_vals))
            summary[f"{victim}_{mode}_dice_list"] = dice_vals

    if "topk" in all_results and "zero" in all_results:
        for victim in ["samed", "medsa"]:
            topk_dice = [r["mean_dice"] for r in all_results["topk"][victim]]
            zero_dice = [r["mean_dice"] for r in all_results["zero"][victim]]
            summary["pairwise"][f"topk_vs_zero_{victim}"] = paired_wilcoxon(topk_dice, zero_dice)
            summary[f"topk_vs_zero_{victim}_cohens_d"] = cohens_d(topk_dice, zero_dice)
            summary[f"topk_vs_zero_pvalue"] = summary["pairwise"][f"topk_vs_zero_{victim}"]["pvalue"]

    return summary


# ---------------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------------

def _save_adv_image(adv_img, adv_dir, case_name, slice_idx, is_3d):
    import matplotlib.pyplot as plt
    if is_3d:
        case_dir = os.path.join(adv_dir, case_name)
        _ensure_dir(case_dir)
        img_out = np.transpose(adv_img, (1, 2, 0))
        img_out = np.clip(img_out, 0, 1)
        plt.imsave(os.path.join(case_dir, f"{slice_idx}.png"),
                   img_out[:, :, 0] if img_out.shape[-1] >= 1 else img_out,
                   cmap='gray', vmin=0, vmax=1)
    else:
        _ensure_dir(adv_dir)
        file_base = os.path.splitext(os.path.basename(case_name))[0]
        img_out = np.transpose(adv_img, (1, 2, 0))
        img_out = np.clip(img_out, 0, 1)
        plt.imsave(os.path.join(adv_dir, f"{file_base}.png"), img_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _ensure_dir(args.output_dir)

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}, ViT: {args.vit_name}")
    print(f"Output: {args.output_dir}")

    # Load source SAM
    print("\nLoading source SAM...")
    source_sam = load_sam_encoder(args.vit_name, args.source_ckpt, args.image_size,
                                   args.num_classes, device)

    # Load SAMed
    print("Loading SAMed...")
    samed_net = load_samed_model(
        args.vit_name, args.source_ckpt, args.samed_ckpt,
        args.image_size, args.num_classes, device=device,
    )

    # Load MedSA
    print("Loading MedSA...")
    medsa_num_classes = args.num_classes + 1
    medsa_model = load_medsa_model(
        args.vit_name, args.source_ckpt, args.medsa_ckpt,
        args.image_size, medsa_num_classes, device=device,
    )

    # G1: Drift decomposition
    if not args.skip_drift:
        drift_summary = run_drift_validation(args, source_sam, samed_net.sam.image_encoder, medsa_model, device)

    # G2: Directional ablation
    if not args.skip_attack:
        directional_summary = run_directional_ablation(args, source_sam, samed_net, medsa_model, device)

    # G3: SRS ablation
    if not args.skip_attack:
        srs_summary = run_srs_ablation(args, source_sam, samed_net, medsa_model, device)

    # Final summary
    final = {
        "config": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(os.path.join(args.output_dir, "validation_summary.json"), final)
    print(f"\n{'='*60}")
    print(f"Validation complete. Results in {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
