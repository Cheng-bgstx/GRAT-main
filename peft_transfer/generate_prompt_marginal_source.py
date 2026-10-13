#!/usr/bin/env python3
"""Single-image, source-only SAM mask disruption over a fixed prompt grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from peft_transfer.evaluate_synapse_slice import ROOT, SAM
from peft_transfer.generate import save_image
from segment_anything import sam_model_registry


def encode(model, image):
    result = model.image_encoder(image)
    return result[0] if isinstance(result, tuple) else result


def prompt_grid(device, mode):
    coordinates = [(x, y) for y in (128, 256, 384) for x in (128, 256, 384)]
    grid = torch.tensor(coordinates, dtype=torch.float32, device=device)
    if mode == "single":
        points = grid[:, None]
        labels = torch.ones((len(coordinates), 1), dtype=torch.int, device=device)
    elif mode == "joint":
        patterns = torch.stack((
            torch.ones(9, dtype=torch.int, device=device),
            torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int, device=device),
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int, device=device),
        ))
        points = torch.where(patterns[:, :, None].bool(), grid[None], torch.zeros_like(grid)[None])
        labels = patterns
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")
    return points, labels


def decoded_masks(model, embedding, sparse, dense):
    masks, quality = model.mask_decoder(
        image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
    )
    return masks, quality


def clean_targets(model, image, sparse, dense):
    with torch.no_grad():
        masks, quality = decoded_masks(model, encode(model, image), sparse, dense)
        areas = (masks > 0).float().mean((-2, -1))
        eligible = (areas >= 0.02) & (areas <= 0.8)
        ranked = quality.masked_fill(~eligible, -torch.inf)
        indices = torch.where(eligible.any(1), ranked.argmax(1), quality.argmax(1))
        chosen = masks[torch.arange(len(masks), device=image.device), indices]
        targets = (chosen > 0).float()
        valid = (targets.sum((-2, -1)) > 0) & (targets.sum((-2, -1)) < targets.shape[-2] * targets.shape[-1])
    return targets, indices, valid, areas


def balanced_mask_loss(logits, targets, valid):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    foreground = (bce * targets).sum((-2, -1)) / targets.sum((-2, -1)).clamp_min(1)
    background = (bce * (1 - targets)).sum((-2, -1)) / (1 - targets).sum((-2, -1)).clamp_min(1)
    return (0.5 * (foreground + background))[valid].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--slice", type=int, required=True)
    parser.add_argument("--vit", choices=SAM, default="vit_b")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--prompt-mode", choices=("single", "joint"), default="single")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = str(SAM[args.vit])
    model, _ = sam_model_registry[args.vit](
        image_size=512, num_classes=3, checkpoint=checkpoint,
        pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
    )
    model = model.to(device).eval()
    state = torch.load(checkpoint, map_location="cpu")
    if not torch.equal(model.mask_decoder.mask_tokens.weight.detach().cpu(),
                       state["mask_decoder.mask_tokens.weight"]):
        raise RuntimeError("Public pretrained mask tokens were not loaded")
    del state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with h5py.File(ROOT / f"data/Synapse_preprocess/test_vol_h5/{args.case}.npy.h5") as handle:
        clean_array = handle["image"][args.slice].astype(np.float32)
    clean = torch.from_numpy(clean_array)[None, None].repeat(1, 3, 1, 1).to(device)
    points, labels = prompt_grid(device, args.prompt_mode)
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=(points, labels), boxes=None, masks=None)
    targets, selected, valid, areas = clean_targets(model, clean, sparse, dense)
    if not bool(valid.any()):
        raise RuntimeError("No usable pretrained SAM masks on the clean image")
    epsilon = args.epsilon / 255.0
    step_size = args.step_size / 255.0
    adversarial = clean.clone()
    momentum = torch.zeros_like(clean)
    losses = []
    for _ in range(args.steps):
        variable = adversarial.detach().requires_grad_(True)
        masks, _ = decoded_masks(model, encode(model, variable), sparse, dense)
        logits = masks[torch.arange(len(masks), device=device), selected]
        loss = balanced_mask_loss(logits, targets, valid)
        gradient = torch.autograd.grad(loss, variable)[0]
        gradient = gradient.mean(1, keepdim=True).repeat(1, 3, 1, 1)
        momentum = momentum + gradient / gradient.abs().mean().clamp_min(1e-12)
        adversarial = (variable.detach() + step_size * momentum.sign())
        adversarial = torch.max(torch.min(adversarial, clean + epsilon), clean - epsilon).clamp(0, 1)
        losses.append(float(loss.detach()))
    output = ROOT / args.out
    save_image(adversarial, clean, epsilon, output, args.case, args.slice, True)
    (output / "attack_meta.json").write_text(json.dumps({
        "source_only": True, "target_checkpoint_used": False,
        "ground_truth_used": False, "trained_prior_used": False,
        "public_pretrained_mask_tokens_verified": True,
        "args": vars(args), "valid_prompts": int(valid.sum()),
        "selected_mask_tokens": selected.cpu().tolist(),
        "clean_mask_area": areas.cpu().tolist(), "source_losses": losses,
    }, indent=2))
    print(json.dumps({"valid_prompts": int(valid.sum()), "source_losses": losses}))


if __name__ == "__main__":
    main()
