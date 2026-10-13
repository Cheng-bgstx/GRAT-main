#!/usr/bin/env python3
"""PASD Synapse-only test with CORRECT step size (2.0/255). Memory-optimized."""
import sys, os, time, json, torch, numpy as np, gc

ROOT = "/root/data2/fyc/fyc/GRAT-main"
NEWEST = os.path.join(ROOT, "newest")
sys.path.insert(0, ROOT)
sys.path.insert(0, NEWEST)

from evaluate import load_sam_encoder, load_samed_model
from attack_pasa import run_pasd_attack, PASD_MODES

IMAGE_SIZE = 512
CKPT_BASE = f"{ROOT}/output_runs_correct"
SAM_CKPTS = {
    "vit_b": f"{ROOT}/checkpoints/sam_vit_b_01ec64.pth",
    "vit_l": f"{ROOT}/checkpoints/sam_vit_l_0b3195.pth",
    "vit_h": f"{ROOT}/checkpoints/sam_vit_h_4b8939.pth",
}

SAMED_SYN = {
    "Syn_vit_b": {"vit":"vit_b","ckpt":"SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth","n_class":8},
    "Syn_vit_l": {"vit":"vit_l","ckpt":"SAMed/Synapse_512_pretrain_vit_l_epo300_bs12_lr0.005/epoch_249.pth","n_class":8},
}

device = torch.device("cuda:1")

def eval_syn(net, img_t, gt):
    with torch.no_grad():
        out = net(img_t, multimask_output=False, image_size=IMAGE_SIZE)
        m = out["masks"]
    pred = torch.argmax(torch.softmax(m.float(), dim=1), dim=1).squeeze(0).cpu().numpy()
    dices = []
    for c in range(1, 8):
        pc = (pred == c).astype(np.float32)
        gc = (gt == c).astype(np.float32)
        if gc.sum() == 0: continue
        inter = (pc * gc).sum()
        dices.append(float(2.0 * inter / (pc.sum() + gc.sum() + 1e-8)))
    return float(np.mean(dices)) if dices else 0.0

from exp_paid_optimal_attack import load_synapse_slices
# Load only 4 volumes to reduce memory
images = load_synapse_slices(8)
print(f"Loaded {len(images)} Synapse images")

EPS_LIST = [4.0/255, 8.0/255, 16.0/255]
METHODS = ["pasd", "secgr", "clean_phase", "white_amp", "none"]

for mk, cfg in sorted(SAMED_SYN.items()):
    print(f"\n=== {mk} ===")

    # Load victim ONLY for evaluation
    victim = load_samed_model(cfg["vit"], SAM_CKPTS[cfg["vit"]],
        os.path.join(CKPT_BASE, cfg["ckpt"]), IMAGE_SIZE, cfg["n_class"], device=device)
    victim.eval()
    print(f"  Victim loaded ({(sum(p.numel() for p in victim.parameters())/1e6):.0f}M params)")

    # Load attack encoder
    sam = load_sam_encoder(cfg["vit"], SAM_CKPTS[cfg["vit"]], IMAGE_SIZE, num_classes=8, device=device)
    att_enc = sam.image_encoder
    att_enc.eval()
    print(f"  Attack encoder loaded")

    for eps_frac in EPS_LIST:
        for mk_method in METHODS:
            dices, cds = [], []
            pm = PASD_MODES[mk_method]["probe_mode"]

            for i, img_data in enumerate(images):
                x = torch.from_numpy(img_data[0]).unsqueeze(0).float().to(device)
                gt = img_data[1]

                cd = eval_syn(victim, x, gt)
                if cd < 0.01:
                    del x; continue

                rg = x.max().item() - x.min().item()
                eps_pix = eps_frac * rg
                step_pix = 2.0 / 255.0 * rg  # CORRECT: fixed step matching original SE-CGR

                x_adv = None
                try:
                    x_adv = run_pasd_attack(att_enc, x, eps_pix, step_pix, 20,
                                           sigma=1.0, alpha=1.0, probe_mode=pm)
                    ad = eval_syn(victim, x_adv, gt)
                    dices.append(ad); cds.append(cd)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  ERROR [{mk_method} img {i}]: {e}")

                # Aggressive cleanup
                if x_adv is not None: del x_adv
                del x
                gc.collect()
                torch.cuda.empty_cache()

            if dices:
                mc, ma = np.mean(cds), np.mean(dices)
                print(f"  eps={eps_frac*255:3.0f}/255 {PASD_MODES[mk_method]['label']:<28} Clean={mc:.4f} Adv={ma:.4f} Δ={mc-ma:+.4f} [{len(dices)}]")
            else:
                print(f"  eps={eps_frac*255:3.0f}/255 {PASD_MODES[mk_method]['label']:<28} NO RESULTS")

    del victim, sam, att_enc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Freed memory")

print("\nDone.")
