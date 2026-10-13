#!/usr/bin/env python3
"""Merge all 6 Phase E worker JSONs into final Tables 2-4."""
import json, os, sys

JSON_DIR = "/root/data2/fyc/fyc/GRAT-main/newest/ablation_results"
ATTACK_ORDER = ["PGD", "MI-FGSM", "NI-FGSM", "DI-FGSM", "TI-FGSM", "SI-FGSM", "SE-CGR", "SE-CGR+SRS"]

def load_all():
    workers = {}
    for f in sorted(os.listdir(JSON_DIR)):
        if f.startswith('phaseE_worker_gpu') and f.endswith('.json'):
            path = os.path.join(JSON_DIR, f)
            data = json.load(open(path))
            for key, summary in data.get('summaries', {}).items():
                workers[key] = summary
    return workers

def format_delta(mean, std=None):
    sign = "+" if mean > 0 else "" if mean == 0 else "-"
    s = f"{sign}{mean:.3f}"
    if std is not None:
        s += f"±{std:.3f}"
    return s

def build_table_2(workers):
    """Table 2: Synapse (CT 3D) — All 6 models × 8 attacks"""
    print("\n" + "=" * 90)
    print("TABLE 2: Phase E Main Results — Synapse (CT 3D)")
    print("=" * 90)

    models = [
        ("SAMed", "vit_b", "Synapse_SAMed_vit_b"),
        ("SAMed", "vit_l", "Synapse_SAMed_vit_l"),
        ("SAMed", "vit_h", "Synapse_SAMed_vit_h"),
        ("MSA", "vit_b", "Synapse_MSA_vit_b"),
        ("MSA", "vit_l", "Synapse_MSA_vit_l"),
        ("MSA", "vit_h", "Synapse_MSA_vit_h"),
    ]

    # Header
    header = f"{'Model':<18} {'Clean':>8}"
    for a in ATTACK_ORDER:
        header += f" {a:>17}"
    print(header)
    print("-" * len(header))

    table_data = {}
    for arch, vit, key in models:
        if key not in workers:
            continue
        w = workers[key]
        clean = w.get('clean_dice', 0.0)
        row = f"{arch} {vit:<12} {clean:8.4f}"
        model_data = {"clean": clean}
        for a in ATTACK_ORDER:
            ad = w.get(a, {})
            delta = ad.get('mean_delta', 0.0)
            row += f" {delta:+17.4f}"
            model_data[a] = delta
        print(row)
        table_data[key] = model_data
    return table_data

def build_table_3(workers):
    """Table 3: ISIC (Dermoscopy 2D) — All 6 models × 8 attacks"""
    print("\n" + "=" * 90)
    print("TABLE 3: Phase E Main Results — ISIC (Dermoscopy 2D)")
    print("=" * 90)

    models = [
        ("SAMed", "vit_b", "ISIC_SAMed_vit_b"),
        ("SAMed", "vit_l", "ISIC_SAMed_vit_l"),
        ("SAMed", "vit_h", "ISIC_SAMed_vit_h"),
        ("MSA", "vit_b", "ISIC_MSA_vit_b"),
        ("MSA", "vit_l", "ISIC_MSA_vit_l"),
        ("MSA", "vit_h", "ISIC_MSA_vit_h"),
    ]

    header = f"{'Model':<18} {'Clean':>8}"
    for a in ATTACK_ORDER:
        header += f" {a:>17}"
    print(header)
    print("-" * len(header))

    table_data = {}
    for arch, vit, key in models:
        if key not in workers:
            continue
        w = workers[key]
        clean = w.get('clean_dice', 0.0)
        row = f"{arch} {vit:<12} {clean:8.4f}"
        model_data = {"clean": clean}
        for a in ATTACK_ORDER:
            ad = w.get(a, {})
            delta = ad.get('mean_delta', 0.0)
            row += f" {delta:+17.4f}"
            model_data[a] = delta
        print(row)
        table_data[key] = model_data
    return table_data

def build_table_4(workers):
    """Table 4: REFUGE (Fundus 2D) — All 6 models × 8 attacks"""
    print("\n" + "=" * 90)
    print("TABLE 4: Phase E Main Results — REFUGE (Fundus 2D)")
    print("=" * 90)

    models = [
        ("SAMed", "vit_b", "REFUGE_SAMed_vit_b"),
        ("SAMed", "vit_l", "REFUGE_SAMed_vit_l"),
        ("SAMed", "vit_h", "REFUGE_SAMed_vit_h"),
        ("MSA", "vit_b", "REFUGE_MSA_vit_b"),
        ("MSA", "vit_l", "REFUGE_MSA_vit_l"),
        ("MSA", "vit_h", "REFUGE_MSA_vit_h"),
    ]

    header = f"{'Model':<18} {'Clean':>8}"
    for a in ATTACK_ORDER:
        header += f" {a:>17}"
    print(header)
    print("-" * len(header))

    table_data = {}
    for arch, vit, key in models:
        if key not in workers:
            continue
        w = workers[key]
        clean = w.get('clean_dice', 0.0)
        row = f"{arch} {vit:<12} {clean:8.4f}"
        model_data = {"clean": clean}
        for a in ATTACK_ORDER:
            ad = w.get(a, {})
            delta = ad.get('mean_delta', 0.0)
            row += f" {delta:+17.4f}"
            model_data[a] = delta
        print(row)
        table_data[key] = model_data
    return table_data

def analyze_findings(table2, table3, table4):
    """Extract key findings."""
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)

    # Synapse: all attacks should have near-zero or negative delta
    print("\n## Synapse (CT 3D)")
    for key, data in table2.items():
        deltas = [data[a] for a in ATTACK_ORDER]
        mean_d = sum(deltas) / len(deltas)
        max_d = max(deltas)
        min_d = min(deltas)
        print(f"  {key}: clean={data['clean']:.4f}, mean_Δ={mean_d:+.4f}, range=[{min_d:+.4f}, {max_d:+.4f}]")

    # ISIC: Mixed effects
    print("\n## ISIC (Dermoscopy 2D)")
    for key, data in table3.items():
        deltas = [data[a] for a in ATTACK_ORDER]
        mean_d = sum(deltas) / len(deltas)
        effective = sum(1 for d in deltas if d > 0.01)
        print(f"  {key}: clean={data['clean']:.4f}, mean_Δ={mean_d:+.4f}, attacks_degrading={effective}/{len(deltas)}")

    # REFUGE: Should show strong degradation on SAMed, invalid on MSA
    print("\n## REFUGE (Fundus 2D)")
    for key, data in table4.items():
        deltas = [data[a] for a in ATTACK_ORDER]
        mean_d = sum(deltas) / len(deltas)
        secgr_d = data.get('SE-CGR', 0)
        pgd_d = data.get('PGD', 0)
        print(f"  {key}: clean={data['clean']:.4f}, mean_Δ={mean_d:+.4f}, SE-CGR={secgr_d:+.4f}, PGD={pgd_d:+.4f}")

if __name__ == "__main__":
    workers = load_all()
    print(f"Loaded {len(workers)} model-dataset combinations from worker files")

    t2 = build_table_2(workers)
    t3 = build_table_3(workers)
    t4 = build_table_4(workers)
    analyze_findings(t2, t3, t4)

    # Save merged results
    merged = {
        "table2_synapse": t2,
        "table3_isic": t3,
        "table4_refuge": t4,
    }
    output_path = os.path.join(JSON_DIR, "phaseE_merged_tables.json")
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged results saved to {output_path}")
