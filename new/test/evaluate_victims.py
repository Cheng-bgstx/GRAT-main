from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path("/root/data2/fyc/fyc/GRAT-main")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_common import (
    ISICPaths,
    REFUGEPaths,
    SynapsePaths,
    build_2d_transforms,
    calculate_metric_percase,
    device_from_arg,
    ensure_dir,
    evaluate_medsa_synapse_case,
    extract_case_name,
    get_synapse_loader,
    load_medsa_model,
    load_samed_encoder,
    medsa_generate_click_prompt,
    setup_seed,
    thresholded_sigmoid_metrics,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--victim", type=str, required=True, choices=["samed", "medsa"])
    parser.add_argument("--dataset", type=str, default="Synapse", choices=["Synapse", "ISIC", "REFUGE"])
    parser.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, default="/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--victim_ckpt", type=str, required=True)
    parser.add_argument("--adv_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def dataset_num_classes(dataset: str, num_classes: int) -> int:
    if dataset == "ISIC":
        return 1
    if dataset == "REFUGE":
        return 2
    return num_classes


def build_samed_2d_dataset(args):
    from datasets.isic import ISIC2016
    from datasets.refuge import REFUGE

    image_tf, mask_tf = build_2d_transforms(args.image_size)
    dataset_args = argparse.Namespace(img_size=args.image_size)
    if args.dataset == "ISIC":
        return ISIC2016(args=dataset_args, data_path=ISICPaths().data_path, mode="Test", transform=image_tf, transform_msk=mask_tf)
    if args.dataset == "REFUGE":
        return REFUGE(args=dataset_args, data_path=REFUGEPaths().data_path, mode="Test", transform=image_tf, transform_msk=mask_tf)
    raise ValueError(args.dataset)


def evaluate_samed(args, device: torch.device):
    from datasets.dataset_synapse import Synapse_dataset
    from test_samed import clean_batch_to_adv_tensor, load_adv_images, test_single_volume

    net, _ = load_samed_encoder(
        vit_name=args.vit_name,
        sam_ckpt=args.sam_ckpt,
        lora_ckpt=args.victim_ckpt,
        image_size=args.image_size,
        num_classes=args.num_classes,
        rank=args.rank,
        device=device,
    )
    if args.dataset == "Synapse":
        synapse = SynapsePaths()
        dataset = Synapse_dataset(base_dir=synapse.volume_path, list_dir=synapse.list_dir, split="test_vol")
        use_clean_image = str(args.adv_dir) in {synapse.volume_path, str(Path(synapse.volume_path)), ""}
        is_3d = True
    else:
        dataset = build_samed_2d_dataset(args)
        use_clean_image = args.adv_dir in {"", "__clean__"}
        is_3d = False
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    folder_path = str(args.adv_dir)
    if not folder_path.endswith("/"):
        folder_path = folder_path + "/"
    rows = []
    for idx, sampled_batch in enumerate(loader):
        if len(rows) >= args.max_cases:
            break
        case_name = extract_case_name(sampled_batch)
        image_clean = sampled_batch["image"]
        label = sampled_batch["label"]
        if use_clean_image:
            image_adv = clean_batch_to_adv_tensor(image_clean, is_3d=is_3d)
        else:
            images_array = load_adv_images(folder_path, case_name, is_3d=is_3d)
            if images_array is None:
                continue
            image_adv = torch.tensor(images_array).float().unsqueeze(0)
        if is_3d and image_adv.shape[1] != label.shape[1]:
            label = label[:, : image_adv.shape[1], :, :]
            image_clean = image_clean[:, : image_adv.shape[1], :, :]
        metric_i = test_single_volume(
            image=image_adv,
            image_clean=image_clean,
            label=label,
            net=net,
            classes=args.num_classes,
            multimask_output=True,
            patch_size=[args.image_size, args.image_size],
            test_save_path=None,
            case=case_name,
            z_spacing=1,
            is_3d=is_3d,
        )
        rows.append(
            {
                "case_name": case_name,
                "mean_dice": float(np.mean(metric_i, axis=0)[0]),
                "mean_hd95": float(np.mean(metric_i, axis=0)[1]),
            }
        )
    write_csv(Path(args.output_dir) / "per_case_metrics.csv", rows)
    write_json(
        Path(args.output_dir) / "summary.json",
        {
            "victim": "samed",
            "num_cases": len(rows),
            "mean_dice": float(np.mean([r["mean_dice"] for r in rows])) if rows else 0.0,
            "mean_hd95": float(np.mean([r["mean_hd95"] for r in rows])) if rows else 0.0,
        },
    )


def evaluate_medsa_2d_case(model, image: torch.Tensor, label: torch.Tensor, device: torch.device, dataset: str) -> dict:
    imgs = image.to(dtype=torch.float32, device=device)
    masks = label.to(dtype=torch.float32, device=device)
    pt, generated_labels = medsa_generate_click_prompt(masks, 2 if dataset == "REFUGE" else 1)
    masks_prompt = masks
    pt = pt.permute(0, 2, 1)
    coords_torch = torch.as_tensor(pt, dtype=torch.float, device=device)
    labels_torch = torch.as_tensor(generated_labels, dtype=torch.int, device=device)
    if len(generated_labels.shape) == 1:
        coords_torch = coords_torch.unsqueeze(1)
        labels_torch = labels_torch.unsqueeze(1)
    prompt = (coords_torch, labels_torch)
    with torch.no_grad():
        imgs_pre = model.preprocess(image.to(dtype=torch.float32, device=device))
        image_embeddings = model.image_encoder(imgs_pre)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=prompt, boxes=None, masks=None)
        pred, _ = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=(dataset == "REFUGE"),
        )
        pred = torch.nn.functional.interpolate(pred, size=label.shape[-2:], mode="bilinear", align_corners=False)
    return thresholded_sigmoid_metrics(pred, masks_prompt, dataset)


def evaluate_medsa(args, device: torch.device):
    import h5py
    import os

    msa_root = ROOT / "Medical-SAM-Adapter-main"
    if str(msa_root) not in sys.path:
        sys.path.insert(0, str(msa_root))

    model = load_medsa_model(
        vit_name=args.vit_name,
        sam_ckpt=args.sam_ckpt,
        msa_ckpt=args.victim_ckpt,
        image_size=args.image_size,
        multimask_output=args.num_classes + 1,
        device=device,
    )
    rows = []
    if args.dataset == "Synapse":
        for idx, h5_path in enumerate(sorted(Path(args.adv_dir).glob("*.npy.h5"))):
            if idx >= args.max_cases:
                break
            with h5py.File(h5_path, "r") as f:
                image = torch.from_numpy(f["image"][:].astype(np.float32)).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
                label = torch.from_numpy(f["label"][:].astype(np.float32)).permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
            metrics = evaluate_medsa_synapse_case(
                model=model,
                image=image,
                label=label,
                device=device,
                multimask_output=args.num_classes + 1,
                image_size=args.image_size,
                out_size=args.image_size,
                evl_chunk=1,
            )
            rows.append({"case_name": h5_path.stem.replace(".npy", ""), **metrics})
    else:
        from datasets.isic import ISIC2016
        from datasets.refuge import REFUGE

        image_tf, mask_tf = build_2d_transforms(args.image_size)
        dataset_args = argparse.Namespace(img_size=args.image_size)
        if args.dataset == "ISIC":
            dataset = ISIC2016(args=dataset_args, data_path=ISICPaths().data_path, mode="Test", transform=image_tf, transform_msk=mask_tf)
        else:
            dataset = REFUGE(args=dataset_args, data_path=REFUGEPaths().data_path, mode="Test", transform=image_tf, transform_msk=mask_tf)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for idx, sampled_batch in enumerate(loader):
            if idx >= args.max_cases:
                break
            case_name = extract_case_name(sampled_batch)
            label = sampled_batch["label"]
            if args.adv_dir in {"", "__clean__"}:
                image = sampled_batch["image"]
            else:
                from test_samed import load_adv_images

                image_np = load_adv_images(str(args.adv_dir), case_name, is_3d=False)
                if image_np is None:
                    continue
                image = torch.from_numpy(image_np).float().unsqueeze(0)
            metrics = evaluate_medsa_2d_case(model=model, image=image, label=label, device=device, dataset=args.dataset)
            rows.append({"case_name": case_name, **metrics})
    write_csv(Path(args.output_dir) / "per_case_metrics.csv", rows)
    write_json(
        Path(args.output_dir) / "summary.json",
        {
            "victim": "medsa",
            "num_cases": len(rows),
            "mean_dice": float(np.mean([r["mean_dice"] for r in rows])) if rows else 0.0,
            "mean_hd95": float(np.mean([r["mean_hd95"] for r in rows])) if rows else 0.0,
        },
    )


def main():
    args = parse_args()
    args.num_classes = dataset_num_classes(args.dataset, args.num_classes)
    setup_seed(args.seed)
    ensure_dir(args.output_dir)
    device = device_from_arg(args.device)
    if args.victim == "samed":
        evaluate_samed(args, device)
    else:
        evaluate_medsa(args, device)


if __name__ == "__main__":
    main()
