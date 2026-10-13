# Rank-Constrained PEFT Adaptation Sensitivity Attack

This package replaces the residual-SVD drift-basis main method with a
rank-constrained PEFT adaptation model while keeping the user's existing SAM
image/checkpoint/dataset interface.

## Empirical status

The mathematical low-rank inner problem and its implementation pass their
self-tests, but low-rank concentration alone did not predict Adapter attack
transfer on the fixed REFUGE ViT-B gate. A four-image source profile selected
rank-2 channels with mean energy fractions `0.877--0.944`. On eight held-out
test images, always enabling the worst low-rank branch obtained SAMed
Dice/HD95 `0.0295/354.03` and Medical-SAM-Adapter `0.7831/14.99`, under a
verified `10/255` budget. This is much weaker than the current PEFT-surrogate
attack on the Adapter victim. Treat this package as a falsified source-only
diagnostic, not as the repository's strongest attack.

## Files

- `rank_adapt_utils.py`
  - exact first-order adaptation-sensitivity matrix `Q`
  - top-r sensitivity `S_r = sqrt(sum_{i<=r} sigma_i(Q)^2)`
  - exact first-order worst-case rank-r Frobenius-bounded update
  - qkv (LoRA-like) and block-residual (Adapter-like) channels
  - offline profile bundle and online injection utilities
- `offline_rank_profile.py`
  - Gate 1: low-rank concentration analysis
  - builds medical PEFT susceptibility profile from source SAM + proxy medical images
- `online_rank_robust_attack.py`
  - source-only online attack
  - stop-gradient inner minimax: recomputes worst low-rank adaptation from current attack state
- `validate_rank_theory.py`
  - Gate 2 (when merged compatible victim weights are available): first-order prediction vs real source->victim gain drift
  - Gate 3: rank-constrained sensitivity vs generic Frobenius sensitivity correlation with target gain drift
- `selftest_rank_adapt.py`
  - CPU tests for the Q formula, shared clean/adv derivative, worst-case rank-r solution, and injection identity

## Theory implemented exactly

For each candidate adaptation channel `c`, introduce a low-rank linear update
`DeltaW_c` with

```text
rank(DeltaW_c) <= r_c,
||DeltaW_c||_F <= rho_c.
```

Let `G(delta)` be the deterministic encoder feature-disruption gain. At the
source model,

```text
G_adapt(delta)
  = G_0(delta) + <Q_c(delta), DeltaW_c>_F + O(||DeltaW_c||^2).
```

For one channel,

```text
min_{rank(DeltaW)<=r, ||DeltaW||F<=rho} <Q, DeltaW>F
  = -rho * sqrt(sum_{i=1}^r sigma_i(Q)^2).
```

The minimizing update is

```text
DeltaW* = -rho * Q_r / ||Q_r||F,
```

where `Q_r` is the best rank-r approximation of `Q`.

The implementation computes `Q` exactly from activations and downstream
gradients. For a channel implemented as

```text
y <- y + Linear(x, DeltaW),
```

```text
Q = G_y^T X.
```

Importantly, the same hypothetical adaptation is applied to both clean and
adversarial branches, so `Q` includes both branch derivatives.

## Adaptation channels

Default:

```text
qkv   : block.attn.qkv weight update, exact LoRA-like low-rank channel
block : y <- y + Linear(y, DeltaW), local low-rank adapter envelope
```

Optional profiling kinds are `proj`, `mlp1`, and `mlp2`.

The default `qkv,block` pair is intentionally matched to the current victim
families: SAMed/LoRA and Medical-SAM-Adapter/adapter, without assuming their
exact rank, placement, or learned parameter distribution.

## 0. Self-test

```bash
cd /path/to/repo/rank_attack
python selftest_rank_adapt.py
```

Expected:

```text
[ok] Q formula ...
[ok] shared clean/adv Q ...
[ok] worst rank-r ...
[ok] linear injection ...
all rank-adaptation self-tests passed
```

## 1. Gate 1 / offline profile

Do NOT start with 128 samples. First screen the hypothesis with 20-50 images.

```bash
PY=/root/anaconda3/envs/adattack/bin/python
cd /root/data2/fyc/fyc/GRAT-main

CUDA_VISIBLE_DEVICES=0 $PY rank_attack/offline_rank_profile.py \
  --dataset Synapse \
  --vit_name vit_b \
  --num_classes 8 \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --img_size 512 \
  --volume_path data/Synapse_preprocess/train_npz \
  --list_dir data/Synapse_preprocess/lists \
  --num_samples 32 \
  --candidate_kinds qkv,block \
  --ranks 2,4,8,16 \
  --channel_chunk 6 \
  --top_channels 6 \
  --adapt_rho 0.05 \
  --out_path rank_attack/profile_Synapse_vit_b.pth
```

Outputs:

```text
profile_Synapse_vit_b.pth
profile_Synapse_vit_b.json
profile_Synapse_vit_b_rows.csv
```

Inspect `kappa_r2/r4/r8/r16` in the CSV. If low-rank concentration is weak for
all channels, stop here rather than running the full attack matrix.

For a formal profile after Gate 1 passes:

```text
--num_samples 128
```

or higher.

## 2. Online source-only rank-robust attack

Dataset interface (preserved):

```bash
CUDA_VISIBLE_DEVICES=0 $PY rank_attack/online_rank_robust_attack.py \
  --dataset Synapse \
  --vit_name vit_b \
  --num_classes 8 \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --img_size 512 \
  --volume_path data/Synapse_preprocess/test_vol_h5 \
  --list_dir data/Synapse_preprocess/lists \
  --profile rank_attack/profile_Synapse_vit_b.pth \
  --att_numstep 10 \
  --att_step_size 2 \
  --att_bound 10 \
  --robust_lambda 0.5 \
  --out_path testset/adv_rank_robust_Synapse_vit_b
```

Direct single-image interface:

```bash
CUDA_VISIBLE_DEVICES=0 $PY rank_attack/online_rank_robust_attack.py \
  --input_image /path/to/image.png \
  --dataset ISIC \
  --vit_name vit_b \
  --num_classes 1 \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --img_size 512 \
  --profile rank_attack/profile_ISIC_vit_b.pth \
  --out_path testset/single_image_rank_robust
```

The online script never loads SAMed/Med-SA weights.

### Online objective

Each iteration first computes current `Q_c` on the selected profile channels.
It derives `DeltaW_c*` from truncated SVD. The relative first-order adaptation
risk is

```text
R(delta) = sum_c w_c rho_c S_r(Q_c) / max(|G_0(delta)|, gain_floor).
```

When `R(delta) <= tau_risk`, the attack uses the source objective only.
When `R(delta) > tau_risk`, it uses

```text
J = (1-lambda) * G_source + lambda * G_worst,
```

where `G_worst` evaluates clean and adversarial images under the same detached
worst-case low-rank operators.

This avoids differentiating through the SVD / parameter-gradient computation;
there are no second-order image gradients.

## 3. Gate 2 / Gate 3 analysis

### Compatible merged victim checkpoint

If an adapted checkpoint can be loaded through the same SAM registry with
merged qkv weights:

```bash
CUDA_VISIBLE_DEVICES=0 $PY rank_attack/validate_rank_theory.py \
  --dataset Synapse \
  --vit_name vit_b \
  --num_classes 8 \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --victim_ckpt /path/to/merged_samed_vit_b.pth \
  --victim_label SAMed \
  --profile rank_attack/profile_Synapse_vit_b.pth \
  --num_samples 32
```

### Repository-specific SAMed / Med-SA loader

Do not guess checkpoint structure. Use the repository's authoritative victim
loader by exposing a tiny factory function:

```python
# victim_factory.py

def build_samed_encoder(args):
    # call the repo's existing SAMed loader here
    model = ...
    model.load_state_dict(...)
    return model.image_encoder  # or return model if it has .image_encoder
```

Then:

```bash
--victim_factory victim_factory:build_samed_encoder \
--victim_ckpt checkpoints/best_weights/SAMed_....pth
```

The validator reports:

- correlation(`rank_score`, `|G_victim-G_source|`)
- correlation(`generic_frob_score`, `|G_victim-G_source|`)
- whether rank-sensitive correlation is stronger
- when compatible merged linear weights exist, correlation between
  `sum <Q, W_victim-W_source>` and signed real gain drift

This is analysis only; victim weights never enter adversarial example generation.

## Important implementation constraints

1. Use the **correct best target checkpoints** before theory evaluation.
2. Keep source/proxy train data separate from target test evaluation.
3. Do not claim the block-residual channel is the exact internal Med-SA adapter
   parameterization. It is a local low-rank adapter envelope.
4. For qkv, Gate-2 parameter prediction is exact only when the victim loader
   exposes the effective merged qkv weight in the same source coordinate system.
5. Do not run the 3-backbone x 3-dataset x 2-victim matrix until Gate 1-3 are
   empirically supported on the smallest Synapse/ViT-B setting.
