# D²TA / AD-CODD — Domain-Task Drift-Decoupled Transfer Attack

Black-box adversarial transfer attack against **medically adapted SAM** models
(SAMed / LoRA, Medical-SAM-Adapter / adapter), crafted on the **open-source SAM
foundation model** and transferred without any access to the victim weights.

## What the method claims (and does not claim)

We do **not** propose a new noise, a new frequency modulation, or a "universal
perturbation".  The base update is produced by a **standard transferable
encoder-disruption optimizer**; the contribution is a single mechanism on top of
it:

> **Adaptation-drift decoupling.**  The victim model is
> `f_m = f_{θ₀ + Δθ_d + Δθ_t}` where `Δθ_d` / `Δθ_t` are the unknown
> domain- and task-adaptation drifts.  A first-order expansion of the attack
> gain gives
>
>     G_m(δ) = G₀(δ) + ⟨∇_θ G₀(δ), Δθ_d + Δθ_t⟩ + O(‖Δθ‖²),
>
> so the transfer gap is dominated by the component of the update direction that
> is **aligned with the adaptation drift**.  We therefore (i) mine an offline
> adaptation-drift basis `B_drift` from proxy medical data, and (ii) softly
> decouple the drift-aligned component of the base update at the instance level,
> while a consistency gate keeps the base direction intact.

## Pipeline

```
┌───────────────────────────── offline ─────────────────────────────┐
│  source SAM + proxy medical data                                  │
│  construct adaptation views {v0, domain-drift, task-drift}        │
│  per view / per perturbation state:  g_v = ∇_δ L_core(δ)          │
│  decompose: g_bar = mean g_v/‖g_v‖₁,  r_v = g_v/‖g_v‖₁ - g_bar    │
│  SVD {r_v}  →  B_drift  (+ w_share, w_drift)                      │
└─────────────────────────────┬─────────────────────────────────────┘
                              │  B_drift
┌─────────────────────────────▼────────── online ───────────────────┐
│  per instance:                                                     │
│    g_base  = standard transferable update                          │
│    g_off   = g_base - α · B_drift B_driftᵀ g_base                 │
│    g_on    = g_off  - β · Σ_v ⟨g_off,r_v⟩/(‖r_v‖²+η) r_v          │
│    g_final = g_base + ρ·(‖g_base‖₁/‖g_on-g_base‖₁)(g_on-g_base)   │
│    s.t. cos(g_final, g_base) ≥ τ                                   │
│    momentum → asymmetric step → clip to ε-ball                     │
└────────────────────────────────────────────────────────────────────┘
```

`L_core = L1(0.9999 F, F*) + λ · mean_l L1(F_l, F_l*)` is the *same* objective
used in the offline view gradients and the online drift-residual computation,
so `B_drift` and the online update are aligned.  Offline samples the view
gradient at several perturbation states (`δ≈0`, small random, a 2-3 step attack
state) so the basis covers the attack trajectory rather than only the clean
neighbourhood.

## Files

| file | role |
|---|---|
| `drift_utils.py` | shared math: `DriftBundle`, SVD basis, decoupling operators, adaptation views (domain transforms + LoRA/adapter hooks), core/base objectives |
| `offline_drift_prior.py` | offline stage — mines `B_drift` |
| `online_ddta_attack.py` | online stage — runs the drift-decoupled attack |

## Quick start

```bash
PY=/root/anaconda3/envs/adattack/bin/python
cd /root/data2/fyc/fyc/GRAT-main

# 1. offline: mine the drift prior on vit_b / Synapse
CUDA_VISIBLE_DEVICES=0 $PY ddta/offline_drift_prior.py \
  --dataset Synapse --vit_name vit_b --num_classes 8 \
  --ckpt checkpoints/sam_vit_b_01ec64.pth --img_size 512 \
  --volume_path data/Synapse_preprocess/train_npz \
  --list_dir data/Synapse_preprocess/lists \
  --num_samples 128 --rank 16 --out_path ddta/drift_prior_Synapse_vit_b.pth

# 2. online: full attack (offline + online decoupling)
CUDA_VISIBLE_DEVICES=0 $PY ddta/online_ddta_attack.py \
  --dataset Synapse --vit_name vit_b --num_classes 8 \
  --ckpt checkpoints/sam_vit_b_01ec64.pth --img_size 512 \
  --volume_path data/Synapse_preprocess/test_vol_h5 \
  --list_dir data/Synapse_preprocess/lists \
  --drift_prior ddta/drift_prior_Synapse_vit_b.pth \
  --use_offline_decouple --use_online_decouple \
  --out_path testset/adv_ddta_Synapse_vit_b
```

## Scale guidance (for the paper experiments)

- **num_samples ≥ 128** (128–512).  The `16`-sample runs are only for debugging.
- **rank ablation** `K ∈ {4, 8, 16, 32, 64}`, reporting explained variance vs
  target attack performance.
- The offline stage runs once per backbone (`vit_b` / `vit_l` / `vit_h`); it only
  uses the source SAM, so it never touches the adapted victims.

## Ablations (verification plan)

| run | flags | needs drift prior? |
|---|---|---|
| base optimizer | (no decoupling flags) | no |
| + offline decoupling | `--use_offline_decouple` | yes |
| + online decoupling | `--use_online_decouple` | **no** |
| **full** | `--use_offline_decouple --use_online_decouple` | yes |

Note: the online residual decoupling is instance-local (it only uses the current
image's adaptation views), so `--use_online_decouple` alone must not require the
offline drift prior.

The bundle stores `w_share = E‖g_bar‖` and `w_drift = E_v‖r_v‖`; the online
hyper-parameters `α, β, ρ, τ` are exposed as CLI flags.

## The three backbones

Repeat offline + online for `--vit_name vit_b`, `vit_l`, `vit_h`
(base ckpts `sam_vit_b_01ec64.pth`, `sam_vit_l_0b3195.pth`,
`sam_vit_h_4b8939.pth`) and evaluate the adversarial images against the SAMed /
Medical-SAM-Adapter victims (see `checkpoints/best_weights/README.md` and
`test_samed.py` / `unified_honest_eval.py`).
