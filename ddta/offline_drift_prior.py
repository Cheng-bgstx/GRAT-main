"""D2TA / AD-CODD -- offline stage: Adaptation Drift Prior Mining.

The offline stage does **not** learn a universal perturbation.  It learns the
adaptation-drift subspace  B_drift  that domain / task adaptation is most likely
to imprint on the attack update direction.

On proxy medical data (only the open-source SAM, never the adapted victims) we
construct adaptation views:

  * source view v0
  * domain-drift views  (CT window / intensity, gamma, contrast, blur, sharpen,
                         feature-statistics shift)
  * task-drift views    (LoRA-like attention residual, adapter-like MLP residual)

For every view we compute the attack gradient of a *core encoder-disruption
loss*  g_v = grad_delta L_core(delta), where L_core is the SAME objective used
for the online drift-residual computation (L1 on the final feature + a
deterministic subset of the last layers).  To let the drift basis cover the
attack trajectory rather than only the clean neighbourhood, the gradients are
sampled at several perturbation states:

      delta = 0 (tiny sym break), small random, 2-3 step attack state.

Then, per state, decompose

      g_bar = mean_v  g_v / ||g_v||_1           (shared transferable direction)
      r_v   = g_v / ||g_v||_1  -  g_bar         (adaptation-specific residual)

and SVD the residuals  {r_v}  (across views, states and images) to obtain
B_drift.  We also record  w_share = E ||g_bar||  and  w_drift = E_v ||r_v||.

Output bundle:  {basis, shape, w_share, w_drift, singular_values, meta}.

Usage (vit_b on Synapse, 128 slices, drift rank 16):
    PY=/root/anaconda3/envs/adattack/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY ddta/offline_drift_prior.py \
        --dataset Synapse --vit_name vit_b --num_classes 8 \
        --ckpt checkpoints/sam_vit_b_01ec64.pth --img_size 512 \
        --volume_path data/Synapse_preprocess/train_npz \
        --list_dir data/Synapse_preprocess/lists \
        --num_samples 128 --rank 16 --out_path ddta/drift_prior_Synapse_vit_b.pth
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from einops import repeat
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from drift_utils import (
    ROOT,
    core_disruption_loss,
    encoder_forward,
    fit_drift_basis,
    forward_view,
    make_views,
    normalize_l1,
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
    p.add_argument("--volume_path", type=str, default="data/Synapse_preprocess/train_npz")
    p.add_argument("--list_dir", type=str, default="data/Synapse_preprocess/lists")
    p.add_argument("--num_samples", type=int, default=128, help="number of proxy slices/images (>=128 for paper)")
    p.add_argument("--rank", type=int, default=16, help="drift basis rank K")
    p.add_argument("--task_rank", type=int, default=8, help="rank of the injected task-view residual")
    p.add_argument("--task_scale", type=float, default=0.1, help="strength of the injected task-view residual")
    # core loss (shared with the online drift-residual computation)
    p.add_argument("--core_lambda", type=float, default=1.0, help="weight of the layer term in L_core")
    p.add_argument("--core_num_layers", type=int, default=4, help="number of last layers in L_core")
    # multi-state sampling
    p.add_argument("--num_states", type=int, default=3, help="perturbation states per view (0/small/attack)")
    p.add_argument("--sym_break", type=float, default=1e-3, help="magnitude of the delta=0 sym break")
    p.add_argument("--attack_state_steps", type=int, default=3, help="base-ascent steps for the attack state")
    p.add_argument("--att_step_size", type=float, default=2.0)
    p.add_argument("--att_bound", type=float, default=10.0)
    p.add_argument("--out_path", type=str, default="ddta/drift_prior.pth")
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
        is_3d = False  # train npz are individual 2D slices
    elif args.dataset == "REFUGE":
        db = REFUGE(args=args, data_path=args.volume_path, mode="Training", transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    elif args.dataset == "ISIC":
        db = ISIC2016(args=args, data_path=args.volume_path, mode="Training", transform=tfs, transform_msk=tfs_msk)
        is_3d = False
    else:
        raise ValueError(args.dataset)
    return DataLoader(db, batch_size=1, shuffle=True, num_workers=4), is_3d


def batch_to_image(batch: Dict, dataset: str) -> torch.Tensor:
    """Extract a single [1, 3, H, W] float 0-1 image tensor from a batch."""
    img = batch["image"]
    if dataset == "Synapse":
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis, :, :]
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr[:1, None, :, :]
        elif arr.ndim == 3:
            arr = arr[None, :, :, :]
        else:
            arr = arr[np.newaxis, :, :, :]
        t = torch.from_numpy(arr.astype(np.float32))
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        if t.max() > 1.1:
            t = t / 255.0
        return t
    t = img if torch.is_tensor(img) else torch.from_numpy(np.asarray(img).astype(np.float32))
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    return t


def compute_view_gradient(
    image_encoder,
    x: torch.Tensor,
    delta: torch.Tensor,
    view: Dict,
    is_synapse: bool,
    norm_transform,
    task_scale: float,
    task_rank: int,
    core_lambda: float,
    core_num_layers: int,
    task_seed: int = 0,
) -> torch.Tensor:
    """grad_delta L_core(delta) under a single adaptation view."""
    with torch.no_grad():
        clean_feats, clean_layers = forward_view(image_encoder, x, view, is_synapse, norm_transform,
                                                 task_scale, task_rank, task_seed)
    perturb = (x + delta).clone().detach().requires_grad_(True)
    feats, feats_layer = forward_view(image_encoder, perturb, view, is_synapse, norm_transform,
                                      task_scale, task_rank, task_seed)
    loss = core_disruption_loss(feats, clean_feats.detach(), feats_layer, clean_layers,
                                lam=core_lambda, num_layers=core_num_layers)
    grad = torch.autograd.grad(loss, perturb, retain_graph=False, create_graph=False)[0]
    return grad.detach()


def generate_attack_state_delta(
    image_encoder,
    x: torch.Tensor,
    is_synapse: bool,
    norm_transform,
    steps: int,
    step_size: float,
    epsilon: float,
    core_lambda: float,
    core_num_layers: int,
) -> torch.Tensor:
    """Run a few core-ascent steps to obtain a mid-trajectory perturbation state."""
    with torch.no_grad():
        clean_feats, clean_layers = encoder_forward(image_encoder, x, is_synapse, norm_transform)
    perturb = (x + torch.randn_like(x) * 1e-3).clone().detach().requires_grad_(True)
    for _ in range(steps):
        image_encoder.zero_grad()
        feats, fl = encoder_forward(image_encoder, perturb, is_synapse, norm_transform)
        loss = core_disruption_loss(feats, clean_feats.detach(), fl, clean_layers,
                                    lam=core_lambda, num_layers=core_num_layers)
        grad = torch.autograd.grad(loss, perturb, retain_graph=False, create_graph=False)[0]
        g = normalize_l1(grad)
        perturb = (perturb.data + step_size * g.sign()).clone().detach().requires_grad_(True)
        eta = torch.clamp(perturb.data - x.data, -epsilon, epsilon)
        perturb = (x.data + eta).clone().detach().requires_grad_(True)
        perturb = torch.clamp(perturb, x.min(), x.max()).clone().detach().requires_grad_(True)
    return (perturb - x).detach()


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

    norm_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    is_synapse = args.dataset == "Synapse"
    loader, _ = load_dataset(args)

    views = make_views()
    residuals: List[torch.Tensor] = []
    share_mags: List[float] = []
    drift_mags: List[float] = []
    shape = None
    seen = 0

    pbar = tqdm(loader, desc=f"offline [{args.dataset}/{args.vit_name}]")
    for batch in pbar:
        if seen >= args.num_samples:
            break
        x = batch_to_image(batch, args.dataset).to(device)
        shape = tuple(x.shape)

        max_val, min_val = x.max(), x.min()
        step_size = float((max_val - min_val) * args.att_step_size / 255.0)
        epsilon = float((max_val - min_val) * args.att_bound / 255.0)

        # perturbation states covering the attack trajectory
        deltas: List[torch.Tensor] = []
        if args.num_states >= 1:
            deltas.append(torch.randn_like(x) * args.sym_break)                       # delta ~ 0
        if args.num_states >= 2:
            deltas.append(torch.randn_like(x) * (0.5 * epsilon))                      # small random
        if args.num_states >= 3:
            deltas.append(generate_attack_state_delta(                             # attack state
                image_encoder, x, is_synapse, norm_transform,
                args.attack_state_steps, step_size, epsilon,
                args.core_lambda, args.core_num_layers,
            ))

        for delta in deltas:
            grads = []
            for view in views:
                g = compute_view_gradient(
                    image_encoder, x, delta, view, is_synapse, norm_transform,
                    args.task_scale, args.task_rank, args.core_lambda, args.core_num_layers,
                    task_seed=args.random_seed + 1000,
                )
                grads.append(normalize_l1(g))

            g_bar = torch.stack(grads, dim=0).mean(dim=0)          # shared direction
            share_mags.append(float(g_bar.abs().sum().item()))
            for g in grads:
                r_v = g - g_bar
                drift_mags.append(float(r_v.abs().sum().item()))
                residuals.append(r_v.cpu())

        seen += 1
        pbar.set_postfix(samples=seen, residuals=len(residuals))

    if not residuals:
        raise RuntimeError("No residuals collected; check the data path / list dir.")

    w_share = float(np.mean(share_mags)) if share_mags else 0.0
    w_drift = float(np.mean(drift_mags)) if drift_mags else 0.0
    print(f"[offline] {len(residuals)} residuals, rank={args.rank}, shape={shape}")
    print(f"[offline] w_share={w_share:.4f} w_drift={w_drift:.4f}")

    bundle = fit_drift_basis(
        residuals, rank=args.rank, shape=shape,
        w_share=w_share, w_drift=w_drift,
        meta={
            "dataset": args.dataset, "vit_name": args.vit_name,
            "num_samples": seen, "num_states": args.num_states,
            "task_scale": args.task_scale, "task_rank": args.task_rank,
            "core_lambda": args.core_lambda, "core_num_layers": args.core_num_layers,
        },
    )
    out = args.out_path if os.path.isabs(args.out_path) else os.path.join(ROOT, args.out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bundle.save(out)
    print(f"[offline] saved drift prior -> {out}")
    print(f"[offline] explained variance={bundle.meta.get('explained_variance', 0):.4f}, "
          f"singular values={bundle.singular_values.tolist()}")


if __name__ == "__main__":
    main()
