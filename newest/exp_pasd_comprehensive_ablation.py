#!/usr/bin/env python3
"""
PASD Complete Ablation Experiment.
Verifies the theory: "Phase carrier is a liability, frequency-domain amplitude is a prior."

Transfer setup: base SAM encoder (attack) → SAMed victim (eval)
Step size: 2.0/255 (matching original SE-CGR, NOT eps/steps)

Experiments:
  E1: Full method matrix — 7 PASD variants × 3 budgets × 2+ models
  E2: Budget sweep — eps ∈ {2,4,6,8,16}/255
  E3: Mechanism verification — radial ratio, phase coherence drop, amplitude correlation
"""
import sys, os, time, json, gc, traceback
import numpy as np
import torch
import torch.nn.functional as F

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from attack_pasa import run_pasd_attack, synthesize_pasd_probe, PASD_MODES

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

device = torch.device("cuda:1")  # Free GPU verified

# ── Model configs ────────────────────────────────────────────────────────────

SAMED_SYN = {
    "Syn_vit_b": {"vit":"vit_b","ckpt":"SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth","n_class":8},
    "Syn_vit_l": {"vit":"vit_l","ckpt":"SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_249.pth","n_class":8},
}

# ── Evaluation ───────────────────────────────────────────────────────────────

def eval_synapse(net, img_t, gt):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out["masks"]
    pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 9):
        pc = (pred == c).astype(np.float32)
        gc = (gt == c).astype(np.float32)
        if gc.sum() == 0: continue
        inter = (pc * gc).sum()
        dices.append(float(2.0 * inter / (pc.sum() + gc.sum() + 1e-8)))
    return float(np.mean(dices)) if dices else 0.0


# ── Mechanism diagnostics ────────────────────────────────────────────────────

def compute_drift_stats(att_enc, x, x_adv, layer_ids):
    """Compute radial ratio and phase coherence drop for drift diagnosis."""
    with torch.no_grad():
        _, targets_layer = att_enc(x)
        _, features_layer = att_enc(x_adv)

    rr_list, coh_drop_list, amp_corr_list = [], [], []
    for idx in layer_ids:
        F_clean = targets_layer[idx].unsqueeze(0)
        F_adv = features_layer[idx].unsqueeze(0)

        Z_clean = torch.fft.fftn(F_clean, dim=(-2,-1), norm="ortho")
        Z_adv = torch.fft.fftn(F_adv, dim=(-2,-1), norm="ortho")

        A_clean = torch.abs(Z_clean)
        A_adv = torch.abs(Z_adv)
        U_clean = Z_clean / (A_clean + 1e-8)

        # Radial ratio: drift decomposed in source carrier
        dZ = Z_adv - Z_clean
        q = dZ * torch.conj(U_clean)
        r_e = torch.real(q).square().sum().item()
        t_e = torch.imag(q).square().sum().item()
        rr_list.append(r_e / (r_e + t_e + 1e-8))

        # Phase coherence drop
        U_adv = Z_adv / (A_adv + 1e-8)
        coh = torch.real(U_adv * torch.conj(U_clean)).mean().item()
        coh_drop_list.append(1.0 - coh)

        # Amplitude correlation (log-space, per-channel mean)
        A_c = A_clean.mean(dim=1, keepdim=True).reshape(-1)
        A_a = A_adv.mean(dim=1, keepdim=True).reshape(-1)
        corr = float(torch.corrcoef(torch.stack([A_c, A_a]))[0, 1].item())
        amp_corr_list.append(max(0, corr) if not np.isnan(corr) else 0)

    return {
        "radial_ratio": float(np.mean(rr_list)),
        "phase_coherence_drop": float(np.mean(coh_drop_list)),
        "amplitude_correlation": float(np.mean(amp_corr_list)),
    }


# ── Main experiment ──────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(NEWEST, "ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pasd_comprehensive_ablation.json")

from exp_paid_optimal_attack import load_synapse_slices

N_IMAGES = 8
EPS_LIST = [2.0/255, 4.0/255, 6.0/255, 8.0/255, 16.0/255]
STEPS = 20
METHODS = ["pasd", "secgr", "clean_phase", "white_amp", "random_phase", "no_gaussian", "none"]
PASD_SIGMA = 1.0
PASD_ALPHA = 1.0
PASD_AMP = 2.0
PASD_K = 2
PASD_LAYERS = 6

# Load prior results for resume
all_results = []
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} prior results")

completed = set()
for r in all_results:
    completed.add((r["model"], r["eps"], r.get("steps", STEPS), r["method"]))

images = load_synapse_slices(N_IMAGES)
print(f"Loaded {len(images)} Synapse images")

for mk, cfg in sorted(SAMED_SYN.items()):
    print(f"\n{'='*80}")
    print(f"  {mk} | {cfg['vit']}")
    print(f"{'='*80}")

    # Load victim
    victim = load_samed_model(cfg["vit"], SAM_CKPTS[cfg["vit"]],
        os.path.join(CKPT_BASE, cfg["ckpt"]), IMAGE_SIZE, cfg["n_class"], device=device)
    victim.eval()
    print(f"  Victim loaded ({(sum(p.numel() for p in victim.parameters())/1e6):.0f}M params)")

    # Load attack encoder (base SAM)
    sam = load_sam_encoder(cfg["vit"], SAM_CKPTS[cfg["vit"]], IMAGE_SIZE, num_classes=8, device=device)
    att_enc = sam.image_encoder
    att_enc.eval()
    print(f"  Attack encoder loaded")

    # Pre-compute layer IDs for mechanism diagnostics
    pre_layer = int(len(sam.image_encoder.blocks) * 0.9)
    diag_layer_ids = list(range(pre_layer - PASD_LAYERS, pre_layer))

    for eps_frac in EPS_LIST:
        for mk_method in METHODS:
            combo_key = (mk, f"{eps_frac*255:.0f}/255", STEPS, mk_method)
            if combo_key in completed:
                continue

            pm = PASD_MODES[mk_method]["probe_mode"]
            dices, cds = [], []
            diag_rr, diag_coh, diag_acorr = [], [], []

            for img_data in images:
                x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                gt = img_data[1]
                cd = eval_synapse(victim, x, gt)
                if cd < 0.01:
                    del x; continue

                rg = x.max().item() - x.min().item()
                eps_pix = eps_frac * rg
                step_pix = 2.0 / 255.0 * rg  # CORRECT step size

                x_adv = None
                try:
                    x_adv = run_pasd_attack(att_enc, x, eps_pix, step_pix, STEPS,
                                           sigma=PASD_SIGMA, alpha=PASD_ALPHA,
                                           probe_amplification=PASD_AMP, K=PASD_K,
                                           probe_mode=pm, num_layers=PASD_LAYERS)
                    ad = eval_synapse(victim, x_adv, gt)
                    dices.append(ad); cds.append(cd)

                    # Mechanism diagnostics (first image only for speed)
                    if len(dices) == 1:
                        diag = compute_drift_stats(att_enc, x, x_adv, diag_layer_ids)
                        diag_rr.append(diag["radial_ratio"])
                        diag_coh.append(diag["phase_coherence_drop"])
                        diag_acorr.append(diag["amplitude_correlation"])
                except Exception as e:
                    traceback.print_exc()
                    print(f"  ERROR [{mk_method}]: {e}")

                if x_adv is not None:
                    del x_adv
                del x
                gc.collect()
                torch.cuda.empty_cache()

            if dices:
                mc, ma = np.mean(cds), np.mean(dices)
                md = mc - ma
                entry = {
                    "model": mk, "dataset": "Synapse", "vit": cfg["vit"],
                    "eps": f"{eps_frac*255:.0f}/255", "steps": STEPS,
                    "method": mk_method, "method_label": PASD_MODES[mk_method]["label"],
                    "mean_clean": float(mc), "mean_adv": float(ma),
                    "mean_delta": float(md), "n_valid": len(dices),
                    "radial_ratio": float(np.mean(diag_rr)) if diag_rr else None,
                    "phase_coherence_drop": float(np.mean(diag_coh)) if diag_coh else None,
                    "amplitude_correlation": float(np.mean(diag_acorr)) if diag_acorr else None,
                }
                all_results.append(entry)
                print(f"  eps={eps_frac*255:3.0f}/255 {PASD_MODES[mk_method]['label']:<28}"
                      f"Clean={mc:.4f} Adv={ma:.4f} Δ={md:+.4f}"
                      + (f" RR={diag_rr[0]:.3f} CohDrop={diag_coh[0]:.3f}" if diag_rr else ""))
            else:
                print(f"  eps={eps_frac*255:3.0f}/255 {PASD_MODES[mk_method]['label']:<28} NO RESULTS")

            # Incremental save
            with open(OUTPUT_PATH, "w") as f:
                json.dump(all_results, f, indent=2)

    del victim, sam, att_enc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  ── {mk} complete ──")

# ── Final summary ────────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("PASD COMPREHENSIVE ABLATION — FINAL SUMMARY")
print("=" * 100)

for eps_str in sorted(set(r["eps"] for r in all_results)):
    methods_seen = sorted(set(r["method"] for r in all_results if r["eps"] == eps_str))
    models_seen = sorted(set(r["model"] for r in all_results if r["eps"] == eps_str))

    print(f"\n── eps={eps_str}, steps=20 ──")
    # Header
    header = f"  {'Model':<18}"
    for m in methods_seen:
        header += f" {PASD_MODES[m]['label']:>14}"
    header += f" {'WINNER':>14}"
    print(header)
    print("  " + "-" * len(header))

    for mk in models_seen:
        row = f"  {mk:<18}"
        best_val, best_m = -999, ""
        for m in methods_seen:
            entries = [r for r in all_results if r["model"]==mk and r["eps"]==eps_str and r["method"]==m]
            md = entries[0]["mean_delta"] if entries else 0.0
            row += f" {md:>+14.4f}"
            if md > best_val: best_val, best_m = md, m
        row += f" {PASD_MODES[best_m]['label']:>14}" if best_m in PASD_MODES else f" {best_m:>14}"
        print(row)

    # Average
    row = f"  {'AVERAGE':<18}"
    for m in methods_seen:
        vals = [r["mean_delta"] for r in all_results if r["eps"]==eps_str and r["method"]==m]
        avg = np.mean(vals) if vals else 0.0
        row += f" {avg:>+14.4f}"
    print("  " + "-" * len(header))
    print(row)

# ── Mechanism summary ────────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("MECHANISM VERIFICATION (Syn_vit_b, eps=8/255)")
print("=" * 100)
print(f"  {'Method':<28} {'Radial Ratio':>12} {'Coh Drop':>12} {'Amp Corr':>12}")
print("  " + "-" * (28+36))
for m in ["pasd", "secgr", "clean_phase", "white_amp", "none"]:
    entries = [r for r in all_results if r["model"]=="Syn_vit_b" and r["eps"]=="8/255" and r["method"]==m]
    if entries and entries[0].get("radial_ratio"):
        r = entries[0]
        print(f"  {PASD_MODES[m]['label']:<28} {r['radial_ratio']:>12.4f} {r['phase_coherence_drop']:>12.4f} {r['amplitude_correlation']:>12.4f}")

# ── Theory chain validation ──────────────────────────────────────────────────

print("\n" + "=" * 100)
print("THEORY VALIDATION CHAIN (Syn_vit_b, eps=8/255)")
print("=" * 100)

entries = {r["method"]: r for r in all_results if r["model"]=="Syn_vit_b" and r["eps"]=="8/255"}
pasd = entries.get("pasd", {})
clean_phase = entries.get("clean_phase", {})
white_amp = entries.get("white_amp", {})
none = entries.get("none", {})
secgr = entries.get("secgr", {})

pasd_d = pasd.get("mean_delta", 0)
cp_d = clean_phase.get("mean_delta", 0)
wa_d = white_amp.get("mean_delta", 0)
no_d = none.get("mean_delta", 0)
sg_d = secgr.get("mean_delta", 0)

print(f"  P1 [Frequency Domain]:   PASD Δ={pasd_d:+.4f} vs Pure L1 Δ={no_d:+.4f} → gain={pasd_d-no_d:+.4f} ({100*(pasd_d-no_d)/(abs(no_d)+1e-8):.0f}%)")
print(f"  P2 [Phase Disruption]:   PASD Δ={pasd_d:+.4f} vs clean_phase Δ={cp_d:+.4f} → gain={pasd_d-cp_d:+.4f} ({100*(pasd_d-cp_d)/(abs(cp_d)+1e-8):.0f}%)")
print(f"  P3 [Amplitude Prior]:    PASD Δ={pasd_d:+.4f} vs white_amp Δ={wa_d:+.4f} → gain={pasd_d-wa_d:+.4f} ({100*(pasd_d-wa_d)/(abs(wa_d)+1e-8):.0f}%)")
print(f"  P4 [vs Old Baseline]:    PASD Δ={pasd_d:+.4f} vs SE-CGR Δ={sg_d:+.4f} → gain={pasd_d-sg_d:+.4f} ({100*(pasd_d-sg_d)/(abs(sg_d)+1e-8):.0f}%)")

# Winner count
win_count = {}
for r in all_results:
    key = (r["model"], r["eps"])
    if key not in win_count or r["mean_delta"] > win_count[key][1]:
        win_count[key] = (r["method"], r["mean_delta"])

print(f"\n  Winner tally across all models×budgets:")
tally = {}
for (mk, ep), (method, val) in win_count.items():
    tally[method] = tally.get(method, 0) + 1
for m, c in sorted(tally.items(), key=lambda x: -x[1]):
    print(f"    {PASD_MODES[m]['label']}: {c} wins")

with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")
print("Done — PASD comprehensive ablation complete.")
