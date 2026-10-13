#!/usr/bin/env python3
"""Compile CODD experiment results into summary tables."""
import os, sys, json, argparse
import numpy as np
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Collect all eval files
    eval_files = sorted([f for f in os.listdir(args.input_dir) if f.startswith("eval_") and f.endswith(".json")])
    if not eval_files:
        print("No eval files found!")
        return

    print(f"Found {len(eval_files)} eval files")

    all_data = {}
    for fname in eval_files:
        path = os.path.join(args.input_dir, fname)
        with open(path) as f:
            data = json.load(f)
        all_data[fname] = data

    # Build summary table: rows = victims, cols = attacks
    # Collect unique victims and attack modes
    victims = set()
    modes = set()
    for fname, data in all_data.items():
        for key, metrics in data.items():
            # key format: "Dataset_Method" e.g., "ISIC_SAMed"
            victims.add(key)
            # Extract mode from filename: eval_vit_b_radial.json → radial
            mode = fname.replace("eval_vit_b_", "").replace("eval_vit_l_", "").replace("eval_vit_h_", "").replace(".json", "")
            # Extract vit from filename
            vit = "vit_b" if "_vit_b_" in fname else ("vit_l" if "_vit_l_" in fname else "vit_h")
            modes.add(f"{vit}_{mode}")

    victims = sorted(victims)
    modes = sorted(modes)

    # Build table
    summary = {
        "victims": list(victims),
        "modes": list(modes),
        "table": {},
        "by_vit": defaultdict(list),
        "by_method": defaultdict(list),
        "by_dataset": defaultdict(list),
    }

    # Aggregate across all files
    combined = {}  # (vit, mode, victim) → metrics
    for fname, data in all_data.items():
        # Extract vit and mode
        if "_vit_b_" in fname:
            vit = "vit_b"
        elif "_vit_l_" in fname:
            vit = "vit_l"
        else:
            vit = "vit_h"
        mode = fname.replace(f"eval_{vit}_", "").replace(".json", "")

        for key, metrics in data.items():
            combined[(vit, mode, key)] = metrics

    # Build per-victim rows
    for victim in victims:
        row = {}
        for mode_key in modes:
            vit_part, mode_part = mode_key.split("_", 1)
            key = (vit_part, mode_part, victim)
            if key in combined:
                m = combined[key]
                row[mode_key] = {
                    "clean": m.get("mean_clean_dice", float("nan")),
                    "adv": m.get("mean_adv_dice", float("nan")),
                    "drop": m.get("mean_dice_drop", float("nan")),
                    "n": m.get("n_cases", 0),
                }

                # Track by categories
                dataset, method = victim.split("_", 1)
                summary["by_vit"][vit_part].append(m.get("mean_dice_drop", float("nan")))
                summary["by_method"][method].append(m.get("mean_dice_drop", float("nan")))
                summary["by_dataset"][dataset].append(m.get("mean_dice_drop", float("nan")))

        summary["table"][victim] = row

    # Compute aggregate statistics
    def agg_stats(drops_list):
        drops = [d for d in drops_list if not np.isnan(d)]
        if not drops:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        return {"mean": float(np.mean(drops)), "std": float(np.std(drops)), "n": len(drops)}

    summary["aggregates"] = {
        "overall": agg_stats([d for drops in summary["by_vit"].values() for d in drops]),
        "by_vit": {k: agg_stats(v) for k, v in summary["by_vit"].items()},
        "by_method": {k: agg_stats(v) for k, v in summary["by_method"].items()},
        "by_dataset": {k: agg_stats(v) for k, v in summary["by_dataset"].items()},
    }

    # Print summary
    print(f"\n{'='*80}")
    print("CODD EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"\nOverall: Mean Drop = {summary['aggregates']['overall']['mean']:.4f} ± "
          f"{summary['aggregates']['overall']['std']:.4f} "
          f"(N={summary['aggregates']['overall']['n']})")

    print(f"\nBy ViT:")
    for vit in ["vit_b", "vit_l", "vit_h"]:
        stats = summary["aggregates"]["by_vit"].get(vit)
        if stats:
            print(f"  {vit}: {stats['mean']:.4f} ± {stats['std']:.4f} (N={stats['n']})")

    print(f"\nBy Method:")
    for method in ["SAMed", "MSA"]:
        stats = summary["aggregates"]["by_method"].get(method)
        if stats:
            print(f"  {method}: {stats['mean']:.4f} ± {stats['std']:.4f} (N={stats['n']})")

    print(f"\nBy Dataset:")
    for ds in ["Synapse", "ISIC", "REFUGE"]:
        stats = summary["aggregates"]["by_dataset"].get(ds)
        if stats:
            print(f"  {ds}: {stats['mean']:.4f} ± {stats['std']:.4f} (N={stats['n']})")

    # Detailed table
    print(f"\n{'='*80}")
    print("PER-MODEL RESULTS")
    print(f"{'='*80}")
    print(f"{'Victim':<30} {'Clean':>8} {'Adv':>8} {'Drop':>8}")
    print("-" * 55)
    for victim in victims:
        row = summary["table"].get(victim, {})
        first_mode = list(row.keys())[0] if row else None
        if first_mode:
            m = row[first_mode]
            print(f"{victim:<30} {m['clean']:>8.4f} {m['adv']:>8.4f} {m['drop']:>8.4f}")

    # Save
    with open(args.output, 'w') as f:
        # Convert defaultdict to regular dict for JSON
        output_data = {
            "victims": summary["victims"],
            "modes": summary["modes"],
            "table": summary["table"],
            "aggregates": {
                "overall": summary["aggregates"]["overall"],
                "by_vit": dict(summary["aggregates"]["by_vit"]),
                "by_method": dict(summary["aggregates"]["by_method"]),
                "by_dataset": dict(summary["aggregates"]["by_dataset"]),
            },
        }
        json.dump(output_data, f, indent=2)
    print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()
