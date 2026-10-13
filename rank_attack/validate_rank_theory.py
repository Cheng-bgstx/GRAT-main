"""Theory validation for rank-constrained adaptation sensitivity.

Validates the two experiments that should be run before a full attack matrix:

Gate 2 (when victim encoder is loadable with compatible merged linear weights):
    Compare first-order prediction sum_c <Q_c, DeltaW_c> against the observed
    source->victim gain drift G_victim - G_source.  This prediction is reported
    only on compatible linear channels (e.g. qkv LoRA merged into qkv.weight).

Gate 3:
    Compare correlation of target gain drift with
      (a) rank-constrained sensitivity S_rank, and
      (b) generic Frobenius sensitivity ||Q||_F.

The script never uses victim weights to *generate* adversarial images. Victim
access is analysis-only.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from rank_adapt_utils import (
    ROOT,
    RankProfileBundle,
    build_candidate_channels,
    core_disruption_gain,
    encoder_forward,
    estimate_channel_q,
    rank_metrics_multi,
    resolve_profile_channels,
)

from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    p.add_argument("--vit_name", default="vit_b")
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--ckpt", default="checkpoints/sam_vit_b_01ec64.pth")
    p.add_argument("--victim_ckpt", required=True)
    p.add_argument("--victim_factory", default=None,
                   help="optional module:function returning an encoder or model with .image_encoder; receives args")
    p.add_argument("--victim_label", default="victim")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--volume_path", default="data/Synapse_preprocess/test_vol_h5")
    p.add_argument("--list_dir", default="data/Synapse_preprocess/lists")
    p.add_argument("--profile", required=True)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--states", default="micro,random,attack")
    p.add_argument("--state_attack_steps", type=int, default=3)
    p.add_argument("--att_step_size", type=float, default=2.0)
    p.add_argument("--att_bound", type=float, default=10.0)
    p.add_argument("--svd_oversample", type=int, default=8)
    p.add_argument("--svd_niter", type=int, default=4)
    p.add_argument("--out_csv", default="rank_attack/theory_validation.csv")
    p.add_argument("--out_json", default="rank_attack/theory_validation_summary.json")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_dataset(args):
    tfs = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    tfs_msk = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    if args.dataset == "Synapse":
        db = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol")
    elif args.dataset == "REFUGE":
        db = REFUGE(args=args, data_path=args.volume_path, mode="Test", transform=tfs, transform_msk=tfs_msk)
    else:
        db = ISIC2016(args=args, data_path=args.volume_path, mode="Test", transform=tfs, transform_msk=tfs_msk)
    return DataLoader(db, batch_size=1, shuffle=False, num_workers=4)


def batch_to_single_image(batch: Dict, dataset: str) -> torch.Tensor:
    t = batch["image"].float()
    if dataset == "Synapse":
        # For a 3-D volume use a deterministic central nonempty-ish slice for screening.
        while t.dim() > 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() == 5:  # [B,D,C,H,W] or similar
            t = t[:, t.shape[1] // 2]
        elif t.dim() == 4 and t.shape[1] not in (1, 3):  # [B,D,H,W]
            t = t[:, t.shape[1] // 2:t.shape[1] // 2 + 1]
        elif t.dim() == 3:
            t = t.unsqueeze(1)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    if float(t.max()) > 1.1:
        t = t / 255.0
    return t


def gain_value(enc, x, xa, is_synapse, norm, layer_weight, core_num_layers) -> float:
    with torch.no_grad():
        fc, lc = encoder_forward(enc, x, is_synapse, norm)
        fa, la = encoder_forward(enc, xa, is_synapse, norm)
        return float(core_disruption_gain(fa, fc, la, lc, layer_weight, core_num_layers).item())


def source_attack_state(enc, x, is_synapse, norm, layer_weight, core_num_layers, steps, step_size, eps):
    lo, hi = float(x.min()), float(x.max())
    xa = (x + torch.randn_like(x) * min(1e-5, eps * 0.01)).clamp(lo, hi).detach()
    with torch.no_grad():
        fc, lc = encoder_forward(enc, x, is_synapse, norm)
        fc, lc = fc.detach(), [z.detach() for z in lc]
    for _ in range(max(1, steps)):
        z = xa.detach().requires_grad_(True)
        fa, la = encoder_forward(enc, z, is_synapse, norm)
        g = core_disruption_gain(fa, fc, la, lc, layer_weight, core_num_layers)
        grad = torch.autograd.grad(g, z)[0]
        xa = z.detach() + step_size * torch.sign(grad.detach())
        xa = (x + torch.clamp(xa - x, -eps, eps)).clamp(lo, hi).detach()
    return xa


def make_states(enc, x, is_synapse, norm, args, layer_weight, core_num_layers):
    value_range = max(float(x.max() - x.min()), 1e-6)
    eps = value_range * args.att_bound / 255.0
    step = value_range * args.att_step_size / 255.0
    wanted = {s.strip() for s in args.states.split(",") if s.strip()}
    out = []
    if "micro" in wanted:
        out.append(("micro", (x + torch.randn_like(x) * min(1e-5, eps * 0.01)).clamp(float(x.min()), float(x.max())).detach()))
    if "random" in wanted:
        d = torch.empty_like(x).uniform_(-0.25 * eps, 0.25 * eps)
        out.append(("random", (x + d).clamp(float(x.min()), float(x.max())).detach()))
    if "attack" in wanted:
        out.append(("attack", source_attack_state(enc, x, is_synapse, norm, layer_weight, core_num_layers,
                                                   args.state_attack_steps, step, eps)))
    return out


def load_source(args, device):
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size, num_classes=args.num_classes, checkpoint=ckpt,
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    return sam.image_encoder.to(device).eval()


def load_victim(args, device):
    vckpt = args.victim_ckpt if os.path.isabs(args.victim_ckpt) else os.path.join(ROOT, args.victim_ckpt)
    if args.victim_factory:
        mod_name, fn_name = args.victim_factory.split(":", 1)
        fn = getattr(importlib.import_module(mod_name), fn_name)
        obj = fn(args)
        enc = obj.image_encoder if hasattr(obj, "image_encoder") else obj
        return enc.to(device).eval()
    # Works for merged/compatible SAM checkpoints. For SAMed/Med-SA wrappers,
    # use --victim_factory to preserve the repository's authoritative loader.
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size, num_classes=args.num_classes, checkpoint=vckpt,
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    return sam.image_encoder.to(device).eval()


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def corr(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx, ry = _rankdata(x), _rankdata(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def main():
    args = parse_args(); set_seed(args.random_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    profile_path = args.profile if os.path.isabs(args.profile) else os.path.join(ROOT, args.profile)
    profile = RankProfileBundle.load(profile_path)
    source = load_source(args, device); victim = load_victim(args, device)
    for m in (source, victim):
        for p in m.parameters(): p.requires_grad_(False)

    selected_channels, selected_meta = resolve_profile_channels(source, profile)
    victim_all = build_candidate_channels(victim, profile.candidate_kinds)
    victim_by_key = {c.key: c for c in victim_all}

    layer_weight = float(profile.meta.get("layer_weight", 0.25))
    core_num_layers = int(profile.meta.get("core_num_layers", 4))
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    is_synapse = args.dataset == "Synapse"
    loader = load_dataset(args)

    rows = []
    seen = 0
    for batch in tqdm(loader, desc=f"theory [{args.dataset}/{args.victim_label}]"):
        if seen >= args.num_samples: break
        x = batch_to_single_image(batch, args.dataset).to(device)
        for state_name, xa in make_states(source, x, is_synapse, norm, args, layer_weight, core_num_layers):
            gsrc, qdict = estimate_channel_q(source, x, xa, selected_channels, is_synapse, norm,
                                             layer_weight, core_num_layers)
            gv = gain_value(victim, x, xa, is_synapse, norm, layer_weight, core_num_layers)
            rank_score = 0.0; generic_score = 0.0; kappa_score = 0.0
            pred_linear = 0.0; pred_terms = 0
            for key, meta in selected_meta.items():
                q = qdict[key]
                frob, sr_map, kp_map = rank_metrics_multi(q, [meta.rank], args.svd_oversample, args.svd_niter)
                sr, kp = sr_map[meta.rank], kp_map[meta.rank]
                rank_score += meta.weight * meta.rho * sr
                generic_score += meta.weight * meta.rho * frob
                kappa_score += meta.weight * kp

                # Gate-2 partial exact prediction for compatible merged linear channels.
                src_spec = next(c for c in selected_channels if c.key == key)
                vic_spec = victim_by_key.get(key)
                if src_spec.kind != "block" and vic_spec is not None \
                        and hasattr(src_spec.module, "weight") and hasattr(vic_spec.module, "weight"):
                    ws = src_spec.module.weight.detach()
                    wv = vic_spec.module.weight.detach()
                    if ws.shape == wv.shape == q.shape:
                        dw = (wv - ws).to(q.device, q.dtype)
                        pred_linear += float((q * dw).sum().item())
                        pred_terms += 1

            rows.append({
                "sample": seen, "state": state_name,
                "gain_source": gsrc, "gain_victim": gv,
                "gain_drift_signed": gv - gsrc,
                "gain_drift_abs": abs(gv - gsrc),
                "rank_score": rank_score,
                "generic_frob_score": generic_score,
                "weighted_kappa": kappa_score,
                "pred_linear": pred_linear if pred_terms > 0 else float("nan"),
                "pred_linear_terms": pred_terms,
            })
        seen += 1
        torch.cuda.empty_cache()

    if not rows: raise RuntimeError("no validation rows")
    target = [r["gain_drift_abs"] for r in rows]
    rank_vals = [r["rank_score"] for r in rows]
    generic_vals = [r["generic_frob_score"] for r in rows]
    signed = [r["gain_drift_signed"] for r in rows]
    pred_pairs = [(r["pred_linear"], r["gain_drift_signed"]) for r in rows if np.isfinite(r["pred_linear"])]

    summary = {
        "victim": args.victim_label,
        "n_rows": len(rows),
        "rank_vs_abs_target_drift": corr(rank_vals, target),
        "generic_frob_vs_abs_target_drift": corr(generic_vals, target),
        "gate3_rank_better_pearson": None,
        "gate3_rank_better_spearman": None,
    }
    rc = summary["rank_vs_abs_target_drift"]; gc = summary["generic_frob_vs_abs_target_drift"]
    if np.isfinite(rc["pearson"]) and np.isfinite(gc["pearson"]):
        summary["gate3_rank_better_pearson"] = bool(rc["pearson"] > gc["pearson"])
    if np.isfinite(rc["spearman"]) and np.isfinite(gc["spearman"]):
        summary["gate3_rank_better_spearman"] = bool(rc["spearman"] > gc["spearman"])
    if len(pred_pairs) >= 3:
        pp = [x[0] for x in pred_pairs]; yy = [x[1] for x in pred_pairs]
        summary["gate2_linear_first_order_vs_signed_drift"] = corr(pp, yy)
        summary["gate2_linear_rows"] = len(pred_pairs)
    else:
        summary["gate2_linear_first_order_vs_signed_drift"] = None
        summary["gate2_linear_rows"] = len(pred_pairs)
        summary["gate2_note"] = "No compatible merged linear victim weights. Use --victim_factory returning a merged encoder, or evaluate Gate 2 with the repository-specific LoRA merge loader."

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(ROOT, args.out_csv)
    out_json = args.out_json if os.path.isabs(args.out_json) else os.path.join(ROOT, args.out_json)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True); os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[theory] rows -> {out_csv}")
    print(f"[theory] summary -> {out_json}")


if __name__ == "__main__":
    main()
