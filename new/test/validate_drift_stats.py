from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from theory_common import (
    SynapsePaths,
    build_support_mask,
    decompose_drift_energy,
    device_from_arg,
    get_synapse_loader,
    get_synapse_depth,
    load_medsa_model,
    load_samed_encoder,
    load_source_sam,
    prepare_synapse_slice,
    select_synapse_slices,
    setup_seed,
    to_serializable,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Synapse")
    parser.add_argument("--vit_name", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--sam_ckpt", type=str, default="/root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--samed_lora_ckpt", type=str, required=True)
    parser.add_argument("--medsa_ckpt", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=2)
    parser.add_argument("--max_slices", type=int, default=4)
    parser.add_argument("--support_min_radius", type=float, default=0.05)
    parser.add_argument("--support_max_radius", type=float, default=1.0)
    parser.add_argument("--support_energy_quantile", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    device = device_from_arg(args.device)
    synapse = SynapsePaths()
    loader = get_synapse_loader(synapse.volume_path, synapse.list_dir)

    source_sam = load_source_sam(args.vit_name, args.sam_ckpt, args.image_size, args.num_classes, device)
    _, samed_encoder = load_samed_encoder(
        vit_name=args.vit_name,
        sam_ckpt=args.sam_ckpt,
        lora_ckpt=args.samed_lora_ckpt,
        image_size=args.image_size,
        num_classes=args.num_classes,
        rank=args.rank,
        device=device,
    )
    medsa_model = load_medsa_model(
        vit_name=args.vit_name,
        sam_ckpt=args.sam_ckpt,
        msa_ckpt=args.medsa_ckpt,
        image_size=args.image_size,
        multimask_output=args.num_classes + 1,
        device=device,
    )

    rows = []
    processed_cases = 0
    with torch.no_grad():
        for sampled_batch in loader:
            if processed_cases >= args.max_cases:
                break
            case_name = sampled_batch["case_name"][0]
            depth = get_synapse_depth(sampled_batch)
            selected_slices = select_synapse_slices(depth, args.max_slices)
            for slice_index in selected_slices:
                image_t, _ = prepare_synapse_slice(sampled_batch, slice_index, device)
                src_feat, _ = source_sam.image_encoder(image_t)
                samed_feat, _ = samed_encoder(image_t)
                medsa_out = medsa_model.image_encoder(image_t)
                feat_medsa = medsa_out[0] if isinstance(medsa_out, (list, tuple)) else medsa_out
                support_mask = build_support_mask(
                    torch.fft.fft2(src_feat, norm="ortho"),
                    min_radius_ratio=args.support_min_radius,
                    max_radius_ratio=args.support_max_radius,
                    energy_quantile=args.support_energy_quantile,
                )
                samed_stats = decompose_drift_energy(src_feat, samed_feat, support_mask=support_mask)
                rows.append(
                    {
                        "victim": "SAMed",
                        "case_name": case_name,
                        "slice_idx": slice_index,
                        "layer_idx": -1,
                        **samed_stats,
                    }
                )
                medsa_stats = decompose_drift_energy(src_feat, feat_medsa, support_mask=support_mask)
                rows.append(
                    {
                        "victim": "MedSA",
                        "case_name": case_name,
                        "slice_idx": slice_index,
                        "layer_idx": -1,
                        **medsa_stats,
                    }
                )
            processed_cases += 1

    summary = {}
    by_victim = defaultdict(list)
    for row in rows:
        by_victim[row["victim"]].append(row)
    for victim, victim_rows in by_victim.items():
        radial_ratios = [r["radial_ratio"] for r in victim_rows]
        tangential_ratios = [r["tangential_ratio"] for r in victim_rows]
        summary[victim] = {
            "num_rows": len(victim_rows),
            "radial_ratio_mean": float(np.mean(radial_ratios)),
            "radial_ratio_median": float(np.median(radial_ratios)),
            "radial_ratio_iqr": float(np.percentile(radial_ratios, 75) - np.percentile(radial_ratios, 25)),
            "tangential_ratio_mean": float(np.mean(tangential_ratios)),
            "angle_std_mean": float(np.mean([r["angle_std"] for r in victim_rows])),
        }

    write_csv(f"{args.output_dir}/drift_rows.csv", rows)
    write_json(
        f"{args.output_dir}/drift_summary.json",
        {
            "config": to_serializable(vars(args)),
            "summary": summary,
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
