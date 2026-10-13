"""Offline medical low-rank adaptation profiling for SAM.

This stage uses only the open-source SAM checkpoint and proxy medical images.
It does NOT access adapted victim weights and does NOT train any surrogate
LoRA/Adapter module.

Output: a RankProfileBundle containing selected adaptation channels, per-channel
rank/rho/weight, and a risk threshold used by the online robust attack.

The script preserves the user's existing dataset/checkpoint interface:
    --dataset {Synapse,ISIC,REFUGE}
    --vit_name vit_b|vit_l|vit_h
    --num_classes ...
    --ckpt ...
    --img_size ...
    --volume_path ...
    --list_dir ...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from rank_adapt_utils import (
    ROOT,
    RankProfileBundle,
    SelectedChannel,
    build_candidate_channels,
    core_disruption_gain,
    encoder_forward,
    estimate_channel_q,
    normalize_l1,
    rank_metrics_multi,
)

from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016


def parse_int_list(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return sorted(set(vals))


def parse_str_list(s: str) -> List[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("empty list")
    return vals


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    p.add_argument("--vit_name", type=str, default="vit_b")
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--ckpt", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--volume_path", type=str, default="data/Synapse_preprocess/train_npz")
    p.add_argument("--list_dir", type=str, default="data/Synapse_preprocess/lists")

    # profiling scope
    p.add_argument("--num_samples", type=int, default=32,
                   help="Gate-1 screening: start with 20-50; formal profile can use >=128")
    p.add_argument("--candidate_kinds", type=parse_str_list, default=parse_str_list("qkv,block"),
                   help="qkv,block by default; optional proj,mlp1,mlp2")
    p.add_argument("--ranks", type=parse_int_list, default=parse_int_list("2,4,8,16"))
    p.add_argument("--channel_chunk", type=int, default=6,
                   help="channels profiled per forward pair; lower saves GPU memory")
    p.add_argument("--top_channels", type=int, default=6)
    p.add_argument("--target_kappa", type=float, default=0.60,
                   help="choose the smallest rank whose mean low-rank energy fraction reaches this")
    p.add_argument("--min_kappa", type=float, default=0.20,
                   help="Gate-1 eligibility threshold at the largest tested rank")
    p.add_argument("--ensure_type_coverage", type=int, default=1,
                   help="reserve at least one selected channel per candidate kind when possible")

    # uncertainty envelope / risk threshold
    p.add_argument("--adapt_rho", type=float, default=0.05,
                   help="Frobenius budget of each local low-rank adaptation operator")
    p.add_argument("--risk_quantile", type=float, default=0.70)

    # attack states used to profile along the optimization trajectory
    p.add_argument("--num_states", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--state_attack_steps", type=int, default=3)
    p.add_argument("--att_step_size", type=float, default=2.0)
    p.add_argument("--att_bound", type=float, default=10.0)

    # deterministic core gain
    p.add_argument("--layer_weight", type=float, default=0.25)
    p.add_argument("--core_num_layers", type=int, default=4)

    # SVD controls
    p.add_argument("--svd_oversample", type=int, default=8)
    p.add_argument("--svd_niter", type=int, default=4)

    p.add_argument("--out_path", type=str, default="rank_attack/profile_Synapse_vit_b.pth")
    p.add_argument("--rows_csv", type=str, default=None,
                   help="defaults to <out_path_without_ext>_rows.csv")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(args):
    tfs = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    tfs_msk = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    if args.dataset == "Synapse":
        db = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="train")
    elif args.dataset == "REFUGE":
        db = REFUGE(args=args, data_path=args.volume_path, mode="Training", transform=tfs, transform_msk=tfs_msk)
    elif args.dataset == "ISIC":
        db = ISIC2016(args=args, data_path=args.volume_path, mode="Training", transform=tfs, transform_msk=tfs_msk)
    else:
        raise ValueError(args.dataset)
    return DataLoader(db, batch_size=1, shuffle=True, num_workers=4)


def batch_to_image(batch: Dict, dataset: str) -> torch.Tensor:
    img = batch["image"]
    if dataset == "Synapse":
        if torch.is_tensor(img):
            t = img.float()
        else:
            t = torch.from_numpy(np.asarray(img).astype(np.float32))
        # Common train_npz shapes: [1,H,W], [H,W], occasionally [1,1,H,W].
        while t.dim() > 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            if t.shape[0] in (1, 3):
                t = t.unsqueeze(0)
            else:
                # defensive fallback: choose first slice
                t = t[:1].unsqueeze(1)
        if t.dim() != 4:
            raise ValueError(f"unexpected Synapse image shape: {tuple(t.shape)}")
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        if float(t.max()) > 1.1:
            t = t / 255.0
        return t

    t = img.float() if torch.is_tensor(img) else torch.from_numpy(np.asarray(img).astype(np.float32))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    return t


def source_gain_and_grad(image_encoder, x_clean, x_adv, is_synapse, norm_transform, args):
    with torch.no_grad():
        fc, lc = encoder_forward(image_encoder, x_clean, is_synapse, norm_transform)
        fc = fc.detach()
        lc = [z.detach() for z in lc]
    xa = x_adv.detach().clone().requires_grad_(True)
    fa, la = encoder_forward(image_encoder, xa, is_synapse, norm_transform)
    gain = core_disruption_gain(fa, fc, la, lc, args.layer_weight, args.core_num_layers)
    grad = torch.autograd.grad(gain, xa, retain_graph=False, create_graph=False)[0]
    return gain.detach(), grad.detach()


def build_attack_states(image_encoder, x, is_synapse, norm_transform, args) -> List[Tuple[str, torch.Tensor]]:
    """Micro-jitter, small-random, and short source-attack states.

    Exact delta=0 is avoided because the L1 feature-disruption gradient is zero
    at identical clean/adv features.
    """
    max_val, min_val = float(x.max()), float(x.min())
    value_range = max(max_val - min_val, 1e-6)
    eps = value_range * float(args.att_bound) / 255.0
    step = value_range * float(args.att_step_size) / 255.0

    micro = torch.clamp(x + torch.randn_like(x) * min(1e-5, eps * 0.01), min_val, max_val)
    states = [("micro", micro.detach())]
    if args.num_states >= 2:
        rnd = torch.empty_like(x).uniform_(-0.25 * eps, 0.25 * eps)
        states.append(("random", torch.clamp(x + rnd, min_val, max_val).detach()))
    if args.num_states >= 3:
        xa = micro.clone().detach()
        for _ in range(max(1, int(args.state_attack_steps))):
            _, grad = source_gain_and_grad(image_encoder, x, xa, is_synapse, norm_transform, args)
            xa = xa + step * torch.sign(grad)
            delta = torch.clamp(xa - x, -eps, eps)
            xa = torch.clamp(x + delta, min_val, max_val).detach()
        states.append(("attack", xa))
    return states


def choose_rank(mean_kappa_by_rank: Dict[int, float], ranks: Sequence[int], target: float) -> int:
    for r in sorted(ranks):
        if mean_kappa_by_rank.get(r, 0.0) >= target:
            return int(r)
    return int(max(ranks))


def main():
    args = parse_args()
    set_seed(args.random_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    image_encoder = sam.image_encoder.to(device).eval()
    for p in image_encoder.parameters():
        p.requires_grad_(False)

    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    is_synapse = args.dataset == "Synapse"
    loader = load_dataset(args)

    channels = build_candidate_channels(image_encoder, args.candidate_kinds)
    if not channels:
        raise RuntimeError(f"no candidate channels found for kinds={args.candidate_kinds}")
    channel_by_key = {c.key: c for c in channels}
    print(f"[profile] {len(channels)} candidate channels: {args.candidate_kinds}")

    # rows[state_id][channel] -> scalar/rank metrics
    rows: List[Dict] = []
    summary_values = defaultdict(lambda: defaultdict(list))
    seen = 0
    state_counter = 0

    pbar = tqdm(loader, desc=f"rank-profile [{args.dataset}/{args.vit_name}]")
    for batch in pbar:
        if seen >= args.num_samples:
            break
        x = batch_to_image(batch, args.dataset).to(device)
        states = build_attack_states(image_encoder, x, is_synapse, norm_transform, args)

        for state_name, xa in states:
            state_counter += 1
            sid = f"s{seen:04d}_{state_name}"
            state_gain = None
            for start in range(0, len(channels), max(1, args.channel_chunk)):
                chunk = channels[start:start + max(1, args.channel_chunk)]
                gain, qdict = estimate_channel_q(
                    image_encoder, x, xa, chunk, is_synapse, norm_transform,
                    args.layer_weight, args.core_num_layers,
                )
                if state_gain is None:
                    state_gain = gain
                for spec in chunk:
                    q = qdict[spec.key]
                    frob, sr, kp = rank_metrics_multi(
                        q, args.ranks, args.svd_oversample, args.svd_niter,
                    )
                    row = {
                        "sample": seen,
                        "state": state_name,
                        "state_id": sid,
                        "gain": float(gain),
                        "channel": spec.key,
                        "kind": spec.kind,
                        "block": spec.block_idx,
                        "frob": frob,
                    }
                    summary_values[spec.key]["frob"].append(frob)
                    for r in args.ranks:
                        row[f"sr_r{r}"] = sr[r]
                        row[f"kappa_r{r}"] = kp[r]
                        summary_values[spec.key][f"sr_r{r}"].append(sr[r])
                        summary_values[spec.key][f"kappa_r{r}"].append(kp[r])
                    rows.append(row)
            pbar.set_postfix(samples=seen + 1, states=state_counter)
        seen += 1
        torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("no profile rows collected; check dataset paths")

    # Per-channel summary and rank selection.
    channel_stats = {}
    candidates = []
    max_rank = max(args.ranks)
    for key, vals in summary_values.items():
        spec = channel_by_key[key]
        mean_kappa = {r: float(np.mean(vals[f"kappa_r{r}"])) for r in args.ranks}
        mean_sr = {r: float(np.mean(vals[f"sr_r{r}"])) for r in args.ranks}
        rank = choose_rank(mean_kappa, args.ranks, args.target_kappa)
        mk = mean_kappa[rank]
        ms = mean_sr[rank]
        mf = float(np.mean(vals["frob"]))
        score = ms * math.sqrt(max(mk, 0.0))
        eligible = mean_kappa[max_rank] >= args.min_kappa
        item = {
            "key": key, "kind": spec.kind, "block_idx": spec.block_idx,
            "rank": rank, "mean_kappa": mk, "mean_sr": ms, "mean_frob": mf,
            "score": score, "eligible": bool(eligible),
            "mean_kappa_by_rank": mean_kappa, "mean_sr_by_rank": mean_sr,
        }
        channel_stats[key] = item
        candidates.append(item)

    eligible = [x for x in candidates if x["eligible"]]
    ranked_pool = sorted(eligible if eligible else candidates, key=lambda z: z["score"], reverse=True)
    selected_items: List[Dict] = []

    # Optional family coverage: one best qkv and one best block when available.
    if args.ensure_type_coverage:
        for kind in args.candidate_kinds:
            choices = [x for x in ranked_pool if x["kind"] == kind]
            if choices and choices[0] not in selected_items:
                selected_items.append(choices[0])
    for item in ranked_pool:
        if len(selected_items) >= args.top_channels:
            break
        if item not in selected_items:
            selected_items.append(item)
    selected_items = selected_items[:args.top_channels]
    if not selected_items:
        raise RuntimeError("no channels selected")

    scores = np.asarray([max(float(x["score"]), 1e-12) for x in selected_items], dtype=np.float64)
    weights = scores / scores.sum()
    selected: List[SelectedChannel] = []
    for item, w in zip(selected_items, weights.tolist()):
        selected.append(SelectedChannel(
            key=item["key"], kind=item["kind"], block_idx=int(item["block_idx"]),
            rank=int(item["rank"]), weight=float(w), rho=float(args.adapt_rho),
            mean_kappa=float(item["mean_kappa"]), mean_sr=float(item["mean_sr"]),
            mean_frob=float(item["mean_frob"]), score=float(item["score"]),
        ))

    # Build state-level risk values from already-computed rows.
    row_map = {(r["state_id"], r["channel"]): r for r in rows}
    state_ids = sorted({r["state_id"] for r in rows})
    gains = []
    for sid in state_ids:
        # same gain is repeated across channel rows
        for c in selected:
            rr = row_map.get((sid, c.key))
            if rr is not None:
                gains.append(abs(float(rr["gain"])))
                break
    positive_gains = [g for g in gains if g > 1e-12]
    gain_floor = float(np.quantile(positive_gains, 0.10)) if positive_gains else 1e-6
    gain_floor = max(gain_floor, 1e-8)

    risks = []
    for sid in state_ids:
        bound = 0.0
        gain = None
        complete = True
        for c in selected:
            rr = row_map.get((sid, c.key))
            if rr is None:
                complete = False
                break
            gain = abs(float(rr["gain"]))
            bound += c.weight * c.rho * float(rr[f"sr_r{c.rank}"])
        if complete and gain is not None:
            risks.append(bound / max(gain, gain_floor))
    if not risks:
        raise RuntimeError("could not compute risk threshold")
    tau_risk = float(np.quantile(risks, float(args.risk_quantile)))

    gate1_pass = bool(len(eligible) > 0 and any(x["mean_kappa"] >= args.min_kappa for x in selected_items))

    out = args.out_path if os.path.isabs(args.out_path) else os.path.join(ROOT, args.out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bundle = RankProfileBundle(
        candidate_kinds=list(args.candidate_kinds),
        selected=selected,
        tau_risk=tau_risk,
        gain_floor=gain_floor,
        ranks_tested=list(args.ranks),
        target_kappa=float(args.target_kappa),
        min_kappa=float(args.min_kappa),
        risk_quantile=float(args.risk_quantile),
        gate1_pass=gate1_pass,
        meta={
            "dataset": args.dataset,
            "vit_name": args.vit_name,
            "num_samples": seen,
            "num_states": args.num_states,
            "candidate_count": len(channels),
            "state_count": len(state_ids),
            "layer_weight": args.layer_weight,
            "core_num_layers": args.core_num_layers,
            "adapt_rho": args.adapt_rho,
            "channel_stats": channel_stats,
            "risk_mean": float(np.mean(risks)),
            "risk_median": float(np.median(risks)),
        },
    )
    bundle.save(out)

    rows_csv = args.rows_csv
    if rows_csv is None:
        rows_csv = os.path.splitext(out)[0] + "_rows.csv"
    elif not os.path.isabs(rows_csv):
        rows_csv = os.path.join(ROOT, rows_csv)
    os.makedirs(os.path.dirname(rows_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(rows_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)

    print("\n[profile] selected channels")
    for c in selected:
        print(f"  {c.key:12s} kind={c.kind:5s} rank={c.rank:2d} weight={c.weight:.3f} "
              f"kappa={c.mean_kappa:.3f} S_r={c.mean_sr:.4g} rho={c.rho:.3f}")
    print(f"[profile] Gate-1 pass={bundle.gate1_pass}; tau_risk={tau_risk:.4g}; gain_floor={gain_floor:.4g}")
    print(f"[profile] saved -> {out}")
    print(f"[profile] rows  -> {rows_csv}")


if __name__ == "__main__":
    main()
