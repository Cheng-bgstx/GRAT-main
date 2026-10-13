"""D2TA / AD-CODD -- online stage: instance-specific drift-decoupled attack.

Given the offline-mined adaptation-drift basis  B_drift, the online stage takes
a standard transferable optimizer (MI-FGSM over the image-encoder feature
disruption objective -- the strong backbone) and applies three operations to its
update direction:

  1. Offline drift-basis decoupling
         g_drift = B_drift B_drift^T g_base
         g_off   = g_base - alpha_t * g_drift

  2. Online instance residual decoupling  (views {v0, v_d, v_t})
         g_bar^t = mean_v g_v^t / ||g_v^t||_1
         r_v^t   = g_v^t / ||g_v^t||_1 - g_bar^t
         g_on^t  = g_off^t - beta_t * sum_v <g_off^t, r_v^t>/(||r_v^t||^2+eta) r_v^t

  3. Backbone consistency gate (residual-correction form)
         g_final = g_base + rho_t * (||g_base||_1 / (||g_on-g_base||_1+eta)) * (g_on - g_base)
         s.t. cos(g_final, g_base) >= tau

then a standard momentum + asymmetric-step update converts g_final to the
perturbation step.

The base update is produced by a standard transferable encoder-disruption
optimizer; the method's contribution is the adaptation-drift decoupling applied
on top of it.

Usage (vit_b -> SAMed / Med-SA transfer on Synapse):
    PY=/root/anaconda3/envs/adattack/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY ddta/online_ddta_attack.py \
        --dataset Synapse --vit_name vit_b --num_classes 8 \
        --ckpt checkpoints/sam_vit_b_01ec64.pth --img_size 512 \
        --volume_path data/Synapse_preprocess/test_vol_h5 \
        --list_dir data/Synapse_preprocess/lists \
        --drift_prior ddta/drift_prior_Synapse_vit_b.pth \
        --use_offline_decouple --use_online_decouple \
        --out_path testset/adv_ddta_Synapse_vit_b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from drift_utils import (
    DriftBundle,
    ROOT,
    base_loss,
    consistency_gate,
    core_disruption_loss,
    drift_correlation,
    encoder_forward,
    forward_view,
    make_random_basis,
    normalize_l1,
    residual_decouple_online,
    return_eta,
    soft_decouple_offline,
)

from segment_anything import sam_model_registry
from datasets.dataset_synapse import Synapse_dataset
from datasets.refuge import REFUGE
from datasets.isic import ISIC2016


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    p.add_argument("--vit_name", type=str, default="vit_b")
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--ckpt", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--volume_path", type=str, default="data/Synapse_preprocess/test_vol_h5")
    p.add_argument("--list_dir", type=str, default="data/Synapse_preprocess/lists")
    p.add_argument("--test_batch", type=str, default=None, help="run a single case only")
    p.add_argument("--max_slices", type=int, default=-1, help="cap slices per case for quick tests (-1 = all)")

    # drift decoupling
    p.add_argument("--drift_prior", type=str, default="ddta/drift_prior.pth")
    p.add_argument("--use_offline_decouple", action="store_true")
    p.add_argument("--use_online_decouple", action="store_true")
    p.add_argument("--random_basis", action="store_true",
                   help="use a random (non-learned) drift basis instead of the offline prior")
    p.add_argument("--basis_rank", type=int, default=16, help="rank of the random drift basis")
    p.add_argument("--alpha", type=float, default=0.3, help="offline soft-decoupling weight")
    p.add_argument("--beta", type=float, default=0.3, help="online residual-decoupling weight")
    p.add_argument("--rho", type=float, default=0.5, help="consistency-gate reweight")
    p.add_argument("--tau", type=float, default=0.85, help="cosine consistency threshold")
    p.add_argument("--online_domain_view", type=str, default="intensity")
    p.add_argument("--online_task_view", type=str, default="adapter", choices=["adapter", "lora"])
    p.add_argument("--task_rank", type=int, default=8)
    p.add_argument("--task_scale", type=float, default=0.1)

    # base optimizer (standard transferable encoder-disruption optimizer)
    p.add_argument("--use_ads", type=int, default=1, help="enable the stochastic robust term of the base optimizer")
    p.add_argument("--ads_rho", type=float, default=1.0)
    p.add_argument("--ads_gamma", type=float, default=1e-4)
    p.add_argument("--ads_tangential_beta", type=float, default=0.0)
    # core loss (must match the offline drift-prior training)
    p.add_argument("--core_lambda", type=float, default=1.0)
    p.add_argument("--core_num_layers", type=int, default=4)
    p.add_argument("--edge_guidance", type=int, default=1)
    p.add_argument("--edge_gamma", type=float, default=3.0)
    p.add_argument("--momentum_decay", type=float, default=0.5)
    p.add_argument("--att_step_size", type=float, default=2.0)
    p.add_argument("--att_bound", type=float, default=10.0)
    p.add_argument("--att_numstep", type=int, default=10)

    p.add_argument("--out_path", type=str, default="testset/adv_ddta")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_edge_map(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx + dy
    emin = edge.view(edge.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    emax = edge.view(edge.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    return (edge - emin) / (emax - emin + 1e-6)


def load_dataset(args):
    tfs = transforms.Compose([transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()])
    tfs_msk = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    if args.dataset == "Synapse":
        db = Synapse_dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol")
        is_3d = True
    elif args.dataset == "REFUGE":
        db = REFUGE(args=args, data_path=args.volume_path, mode="Test", transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    elif args.dataset == "ISIC":
        db = ISIC2016(args=args, data_path=args.volume_path, mode="Test", transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    else:
        raise ValueError(args.dataset)
    return DataLoader(db, batch_size=1, shuffle=False, num_workers=4), is_3d


def get_case_name(batch: Dict) -> str:
    if "case_name" in batch:
        cn = batch["case_name"]
        return cn[0] if isinstance(cn, (list, tuple)) else str(cn)
    if "image_meta_dict" in batch:
        return str(batch["image_meta_dict"]["filename_or_obj"][0])
    return "unknown"


def batch_to_slices(batch: Dict, dataset: str) -> np.ndarray:
    """Return an array of [D, 1, H, W] (Synapse) or [1, C, H, W] (2D)."""
    img_tensor = batch["image"]
    if dataset == "Synapse":
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(1)
        vol = img_tensor.squeeze(0).cpu().detach().numpy()
        if len(vol.shape) == 2:
            vol = vol[np.newaxis, np.newaxis, :, :]
        elif len(vol.shape) == 3:
            vol = vol[:, np.newaxis, :, :]
        return vol
    else:
        return img_tensor.cpu().detach().numpy()


def save_adv(image_np: np.ndarray, case_name: str, ind: int, is_3d: bool, out_path: str):
    # image_np: [C, H, W] float 0-1
    if is_3d:
        folder = os.path.join(out_path, case_name)
        os.makedirs(folder, exist_ok=True)
        plt.imsave(os.path.join(folder, f"{ind}.png"), image_np[0], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        os.makedirs(out_path, exist_ok=True)
        file_base = os.path.splitext(os.path.basename(case_name))[0]
        np.save(os.path.join(out_path, f"{file_base}.npy"), np.clip(np.transpose(image_np, (1, 2, 0)), 0.0, 1.0))
        plt.imsave(os.path.join(out_path, f"{file_base}.png"), np.clip(np.transpose(image_np, (1, 2, 0)), 0.0, 1.0))


def run_attack(
    image_encoder,
    x_clean: torch.Tensor,
    args,
    bundle: Optional[DriftBundle],
    norm_transform,
    is_synapse: bool,
) -> Tuple[np.ndarray, Dict]:
    """Attack a single image/slice. Returns (adv_image [C,H,W], trace)."""
    device = x_clean.device
    max_val, min_val = x_clean.max(), x_clean.min()
    step_size = float((max_val - min_val) * args.att_step_size / 255.0)
    epsilon = float((max_val - min_val) * args.att_bound / 255.0)

    # clean references (source view)
    with torch.no_grad():
        targets, targets_layer = image_encoder(
            x_clean[:, 0:1].repeat(1, 3, 1, 1) if is_synapse else norm_transform(x_clean)
        )
        # online-view clean references (for residual decoupling); the task-view
        # seed is pinned in the view dict so the clean and perturbed passes share
        # the SAME injected LoRA/adapter residual.
        if args.use_online_decouple:
            domain_view = {"name": args.online_domain_view, "type": "domain"}
            task_view = {"name": f"task_{args.online_task_view}", "type": "task",
                         "kind": args.online_task_view, "seed": args.random_seed + 3000}
            clean_vd, clean_vdl = forward_view(image_encoder, x_clean, domain_view, is_synapse,
                                               norm_transform, args.task_scale, args.task_rank)
            clean_vt, clean_vtl = forward_view(image_encoder, x_clean, task_view, is_synapse,
                                               norm_transform, args.task_scale, args.task_rank)
        else:
            domain_view = task_view = None
            clean_vd = clean_vdl = clean_vt = clean_vtl = None

    edge_map = get_edge_map(x_clean).detach() if args.edge_guidance == 1 else None

    perturb_img = Variable(x_clean.data + torch.randn_like(x_clean) * 1e-5, requires_grad=True)
    momentum = torch.zeros_like(x_clean).detach()
    trace = {"losses": [], "cosines": [], "rho_used": [], "drift_before": [], "drift_after": []}

    for t in range(args.att_numstep):
        image_encoder.zero_grad()

        # ---- base update (strong backbone) ----
        p_in = perturb_img[:, 0:1].repeat(1, 3, 1, 1) if is_synapse else perturb_img
        p_pre = p_in if is_synapse else norm_transform(p_in)
        if t == 0:
            p_pre = p_pre + torch.randn_like(p_pre) * 0.0001
        features, features_layer = image_encoder(p_pre)
        loss = base_loss(features, targets, features_layer, targets_layer,
                         enable_ads=bool(args.use_ads), ads_rho=args.ads_rho,
                         ads_gamma=args.ads_gamma, ads_tangential_beta=args.ads_tangential_beta,
                         rng=random)
        g_base = torch.autograd.grad(loss, perturb_img, retain_graph=False, create_graph=False)[0]

        if edge_map is not None:
            g_base = g_base * (1.0 + args.edge_gamma * edge_map)
        g_base = normalize_l1(g_base)

        # ---- 1. offline drift-basis decoupling ----
        if args.use_offline_decouple and bundle is not None:
            g_off = soft_decouple_offline(g_base, bundle, args.alpha)
        else:
            g_off = g_base

        # ---- 2. online instance residual decoupling ----
        if args.use_online_decouple:
            g_v0 = g_base  # already L1-normalized

            def _view_grad(view, clean_feats, clean_layers):
                feats, feats_layer = forward_view(image_encoder, perturb_img, view, is_synapse,
                                                  norm_transform, args.task_scale, args.task_rank)
                l = core_disruption_loss(feats, clean_feats.detach(), feats_layer, clean_layers,
                                         lam=args.core_lambda, num_layers=args.core_num_layers)
                g = torch.autograd.grad(l, perturb_img, retain_graph=False, create_graph=False)[0]
                if edge_map is not None:
                    g = g * (1.0 + args.edge_gamma * edge_map)
                return normalize_l1(g)

            g_vd = _view_grad(domain_view, clean_vd, clean_vdl)
            g_vt = _view_grad(task_view, clean_vt, clean_vtl)
            g_bar = (g_v0 + g_vd + g_vt) / 3.0
            residuals = [g_v0 - g_bar, g_vd - g_bar, g_vt - g_bar]
            g_on = residual_decouple_online(g_off, residuals, args.beta)
        else:
            g_on = g_off

        # ---- 3. backbone consistency gate ----
        g_final, rho_used = consistency_gate(g_base, g_on, args.rho, args.tau)
        g_final = normalize_l1(g_final)

        # ---- momentum + asymmetric step ----
        momentum = args.momentum_decay * momentum.detach() + g_final
        momentum = momentum.detach()
        eta = return_eta(momentum, step_size).detach()

        perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
        eta_total = torch.clamp(perturb_img.data - x_clean.data, -epsilon, epsilon)
        perturb_img = Variable(x_clean.data + eta_total, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, min_val, max_val), requires_grad=True)

        cos = float(F.cosine_similarity(
            g_final.view(1, -1), g_base.view(1, -1)).item())
        trace["losses"].append(float(loss.detach().item()))
        trace["cosines"].append(cos)
        trace["rho_used"].append(rho_used)
        if bundle is not None:
            trace["drift_before"].append(float(drift_correlation(g_base, bundle).item()))
            trace["drift_after"].append(float(drift_correlation(g_final, bundle).item()))

    adv = perturb_img[0].detach().cpu().numpy()
    return adv, trace


def main():
    args = parse_args()
    set_seed(args.random_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=os.path.join(ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    image_encoder = sam.image_encoder.to(device).eval()
    for p in image_encoder.parameters():
        p.requires_grad_(False)

    bundle = None
    # the online residual decoupling only uses the current instance's views and
    # does NOT depend on the offline drift prior; only --use_offline_decouple loads it.
    if args.use_offline_decouple:
        if args.random_basis:
            shape = (1, 3, args.img_size, args.img_size)
            basis = make_random_basis(args.basis_rank, shape, device, seed=args.random_seed)
            bundle = DriftBundle(basis=basis, shape=shape, w_share=0.0, w_drift=0.0,
                                 singular_values=torch.zeros(args.basis_rank), meta={"random": True})
            print(f"[online] using RANDOM drift basis rank={args.basis_rank} shape={shape}")
        else:
            prior_path = os.path.join(ROOT, args.drift_prior) if not os.path.isabs(args.drift_prior) else args.drift_prior
            if not os.path.exists(prior_path):
                raise FileNotFoundError(f"drift prior not found: {prior_path}")
            bundle = DriftBundle.load(prior_path, device=device)
            print(f"[online] loaded drift prior rank={bundle.basis.shape[0]} shape={bundle.shape}")

    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    is_synapse = args.dataset == "Synapse"
    loader, is_3d = load_dataset(args)

    os.makedirs(args.out_path, exist_ok=True)
    acc = {"cos": [], "rho": [], "loss": [], "drift_before": [], "drift_after": []}
    for batch in tqdm(loader, desc=f"online [{args.dataset}/{args.vit_name}]"):
        case_name = get_case_name(batch)
        if args.test_batch is not None and case_name != args.test_batch:
            continue
        slices = batch_to_slices(batch, args.dataset)
        n_slices = slices.shape[0] if args.max_slices < 0 else min(args.max_slices, slices.shape[0])
        for ind in range(n_slices):
            inp = slices[ind]
            x = torch.from_numpy(inp).unsqueeze(0).float().to(device)  # [1,1,H,W] or [1,C,H,W]
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            adv, trace = run_attack(image_encoder, x, args, bundle, norm_transform, is_synapse)
            save_adv(adv, case_name, ind, is_3d, args.out_path)
            acc["cos"].extend(trace["cosines"])
            acc["rho"].extend(trace["rho_used"])
            acc["loss"].extend(trace["losses"])
            acc["drift_before"].extend(trace["drift_before"])
            acc["drift_after"].extend(trace["drift_after"])
        torch.cuda.empty_cache()

    def _mean(v):
        return float(np.mean(v)) if v else None

    stats = {
        "dataset": args.dataset, "vit": args.vit_name,
        "offline_decouple": bool(args.use_offline_decouple),
        "online_decouple": bool(args.use_online_decouple),
        "random_basis": bool(args.random_basis),
        "alpha": args.alpha, "beta": args.beta, "rho": args.rho, "tau": args.tau,
        "mean_cos": _mean(acc["cos"]),
        "mean_rho": _mean(acc["rho"]),
        "mean_drift_before": _mean(acc["drift_before"]),
        "mean_drift_after": _mean(acc["drift_after"]),
    }
    if stats["mean_drift_before"] is not None and stats["mean_drift_after"] is not None:
        stats["drift_reduced"] = bool(stats["mean_drift_after"] < stats["mean_drift_before"])
    with open(os.path.join(args.out_path, "_trace.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[online] done -> {args.out_path}")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
