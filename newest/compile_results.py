#!/usr/bin/env python3
"""
Comprehensive Final Results Compiler

Reads all experiment output JSONs and produces:
1. Master results table across all 18 models
2. Method comparison: Pure L1 vs all alternatives
3. Statistical summary
"""

import json, os
import numpy as np

OUTPUT_DIR = "/root/data2/fyc/fyc/GRAT-main/newest/ablation_results"

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return []

def main():
    print("=" * 90)
    print("COMPREHENSIVE FINAL RESULTS COMPILER")
    print("=" * 90)

    # ── Pure L1 SAMed Validation ──
    pure_l1_samed = load_json(os.path.join(OUTPUT_DIR, "pure_l1_validation.json"))

    # ── MSA Pure L1 ──
    msa_isic = load_json(os.path.join(OUTPUT_DIR, "msa_pure_l1_validation.json"))

    # ── Loss Decomposition ──
    loss_decomp = load_json(os.path.join(OUTPUT_DIR, "loss_decomposition.json"))

    # ── Pixel vs Feature ──
    pixel_vs_feat = load_json(os.path.join(OUTPUT_DIR, "pixel_vs_feature_space.json"))

    # ── Same-backbone transfer ──
    sb_transfer = load_json(os.path.join(OUTPUT_DIR, "same_backbone_transfer.json"))

    # ── Transfer Attack ──
    transfer = load_json(os.path.join(OUTPUT_DIR, "transfer_attack.json"))

    # ── MSA Comprehensive ──
    msa_comp = load_json(os.path.join(OUTPUT_DIR, "msa_comprehensive.json"))

    print(f"\nLoaded: pure_l1={len(pure_l1_samed)}, msa_isic={len(msa_isic)}, "
          f"loss_decomp={len(loss_decomp)}, pixel_vs_feat={len(pixel_vs_feat)}, "
          f"sb_transfer={len(sb_transfer)}, transfer={len(transfer)}, "
          f"msa_comp={len(msa_comp)}")

    # ── Summary 1: Pure L1 advantage over alternatives ──
    if pure_l1_samed:
        print(f"\n{'='*80}")
        print("PURE L1 vs BARE RADIAL vs RANDOM — All SAMed Models")
        print(f"{'='*80}")
        print(f"{'Model':<16} {'eps':>8} {'steps':>6} {'Pure L1 Δ':>12} {'BareRad Δ':>12} {'Random Δ':>12} {'L1-Rad':>10} {'L1-Rand':>10}")
        print("-" * 86)
        for r in sorted(pure_l1_samed, key=lambda x: (x.get("model",""), x.get("eps",""), x.get("steps",0))):
            model = r.get("model", "?")
            eps = r.get("eps", "?")
            steps = r.get("steps", 0)
            method = r.get("method", "?")
            md = r.get("mean_delta", 0)
            # Find matching rows
            pure = next((x for x in pure_l1_samed
                        if x.get("model")==model and x.get("eps")==eps and x.get("steps")==steps
                        and x.get("method")=="pure_l1"), None)
            rad = next((x for x in pure_l1_samed
                       if x.get("model")==model and x.get("eps")==eps and x.get("steps")==steps
                       and x.get("method")=="bare_radial"), None)
            rand = next((x for x in pure_l1_samed
                        if x.get("model")==model and x.get("eps")==eps and x.get("steps")==steps
                        and x.get("method")=="random"), None)
            if pure and (pure is r or method == "pure_l1"):
                pd = pure["mean_delta"]
                rd = rad["mean_delta"] if rad else 0
                nd = rand["mean_delta"] if rand else 0
                print(f"  {model:<16} {eps:>8} {steps:>6} {pd:>+12.4f} {rd:>+12.4f} {nd:>+12.4f} {pd-rd:>+10.4f} {pd-nd:>+10.4f}")

    # ── Summary 2: MSA models ──
    msa_data = msa_comp if msa_comp else msa_isic
    if msa_data:
        print(f"\n{'='*80}")
        print("MSA PURE L1 VALIDATION")
        print(f"{'='*80}")
        print(f"{'Model':<22} {'eps':>8} {'steps':>6} {'Pure L1 Δ':>12} {'Random Δ':>12} {'L1-Rand':>12}")
        print("-" * 72)
        for r in sorted(msa_data, key=lambda x: (x.get("model",""), x.get("eps",""), x.get("steps",0))):
            model = r.get("model", "?")
            eps = r.get("eps", "?")
            steps = r.get("steps", 0)
            method = r.get("method", "?")
            md = r.get("mean_delta", 0)
            if method == "pure_l1":
                rand = next((x for x in msa_data
                            if x.get("model")==model and x.get("eps")==eps and x.get("steps")==steps
                            and x.get("method") in ("random", "Random Direction")), None)
                nd = rand["mean_delta"] if rand else 0
                print(f"  {model:<22} {eps:>8} {steps:>6} {md:>+12.4f} {nd:>+12.4f} {md-nd:>+12.4f}")

    # ── Summary 3: Loss decomposition best config ──
    if loss_decomp:
        print(f"\n{'='*80}")
        print("LOSS DECOMPOSITION — Best Config per Model/Eps/Steps")
        print(f"{'='*80}")
        models = sorted(set(r.get("model","") for r in loss_decomp))
        for model in models:
            model_r = [r for r in loss_decomp if r.get("model")==model]
            print(f"\n  {model}:")
            for eps in sorted(set(r["eps"] for r in model_r)):
                for steps in sorted(set(r["steps"] for r in model_r)):
                    subset = [r for r in model_r if r["eps"]==eps and r["steps"]==steps]
                    best = max(subset, key=lambda r: r["mean_delta"])
                    l1 = next((r for r in subset if r.get("config")=="L1"), None)
                    l1d = l1["mean_delta"] if l1 else 0
                    imp = best["mean_delta"] - l1d
                    mark = " ***" if imp > 0.005 else ""
                    print(f"    eps={eps} steps={steps:2d}: best={best['config']:<22} Δ={best['mean_delta']:+.4f} (vs L1 {l1d:+.4f}, Δ={imp:+.4f}){mark}")

    # ── Summary 4: Feature-space vs Pixel-space ──
    if pixel_vs_feat:
        print(f"\n{'='*80}")
        print("FEATURE-SPACE vs PIXEL-SPACE PGD")
        print(f"{'='*80}")
        print(f"{'Model':<16} {'eps':>8} {'steps':>6} {'Pure L1 (feat)':>14} {'Pixel PGD (dice)':>16} {'Pixel PGD (MSE)':>16} {'Winner':<20}")
        print("-" * 82)
        models = sorted(set(r.get("model","") for r in pixel_vs_feat))
        for model in models:
            model_r = [r for r in pixel_vs_feat if r.get("model")==model]
            for eps in sorted(set(r["eps"] for r in model_r)):
                for steps in sorted(set(r["steps"] for r in model_r)):
                    pure = next((r for r in model_r if r["eps"]==eps and r["steps"]==steps and r["method"]=="pure_l1"), None)
                    dice = next((r for r in model_r if r["eps"]==eps and r["steps"]==steps and r["method"]=="pixel_pgd"), None)
                    mse = next((r for r in model_r if r["eps"]==eps and r["steps"]==steps and r["method"]=="pixel_mse"), None)
                    pd = pure["mean_delta"] if pure else 0
                    dd = dice["mean_delta"] if dice else 0
                    md = mse["mean_delta"] if mse else 0
                    winner = "Pure L1" if pd >= max(dd,md) else ("Pixel Dice" if dd >= md else "Pixel MSE")
                    print(f"  {model:<16} {eps:>8} {steps:>6} {pd:>+14.4f} {dd:>+16.4f} {md:>+16.4f} {winner:<20}")

    # ── Summary 5: Same-backbone transfer ──
    if sb_transfer:
        print(f"\n{'='*80}")
        print("SAME-BACKBONE TRANSFER (SAMed → MSA)")
        print(f"{'='*80}")
        print(f"{'Pair':<30} {'eps':>8} {'steps':>6} {'Src Δ':>12} {'Tgt Δ':>12} {'Transfer':>10} {'Verdict'}")
        print("-" * 78)
        for r in sorted(sb_transfer, key=lambda x: (x.get("pair",""), x.get("eps",""), x.get("steps",0))):
            verdict = "Φ SHARED" if r["transfer_ratio"] > 0.05 else "NOT SHARED"
            print(f"  {r['pair']:<30} {r['eps']:>8} {r['steps']:>6} "
                  f"{r['src_delta']:>+12.4f} {r['tgt_delta']:>+12.4f} "
                  f"{r['transfer_ratio']:>10.4f} {verdict}")

    # ── Summary 6: Cross-architecture transfer ──
    if transfer:
        print(f"\n{'='*80}")
        print("CROSS-ARCHITECTURE TRANSFER (vit_b → vit_l/h)")
        print(f"{'='*80}")
        print(f"{'eps':>8} {'steps':>6} {'method':>12} {'Src Δ':>10} {'vit_l Δ':>10} {'vit_l ratio':>12} {'vit_h Δ':>10} {'vit_h ratio':>12}")
        print("-" * 80)
        for r in sorted(transfer, key=lambda x: (x.get("eps",""), x.get("steps",0), x.get("method",""))):
            print(f"  {r['eps']:>8} {r['steps']:>6} {r['method']:>12} "
                  f"{r['src_delta']:>+10.4f} {r['vit_l_delta']:>+10.4f} {r['vit_l_transfer']:>+12.4f} "
                  f"{r['vit_h_delta']:>+10.4f} {r['vit_h_transfer']:>+12.4f}")

    print(f"\n{'='*90}")
    print("COMPILATION COMPLETE")
    print(f"{'='*90}")

if __name__ == "__main__":
    main()
