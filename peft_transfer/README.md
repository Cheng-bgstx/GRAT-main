# PEFT-robust transfer attack

## Current research position (2026-10-11)

The main theoretical direction is now
[`ACRS_THEORY_AND_GATE.md`](ACRS_THEORY_AND_GATE.md). ACRS studies the change in
the encoder's response to the same input perturbation after medical adaptation,
not source-decoder observability or clean-feature drift. Task-Observable Target
Disruption is retired as a main proposal. Fourier frequency is retained only as
a spatial diagnostic of the coupled response after explicit BHWC-to-BCHW
conversion; amplitude/phase locking and direct spectral modulation are not
selected attack modules.

The first frozen ACRS gate finds a weak, mostly positive LoRA/Adapter common
response component after matching preprocessing, but it does not pass the
predictive gate. True post-hoc ACRS metrics improve leave-one-case-out Dice MAE
by only `2.92%`, while the preregistered threshold is `10%`; source-only
structure improves it by `4.66%`, below the `5%` threshold. No ACRS online loss
is therefore claimed.

GR-only from UMI-GRAT is an external baseline to reproduce under the local
protocol. It is not a component of the proposed method. The strong Joint
Pareto results remain a separate proxy-assisted empirical branch and must not
be presented as source-only ACRS evidence.

The current research question allows training medical LoRA/Adapter surrogates
from public training data while keeping victim checkpoints and test masks out
of generation. See [`MEDICAL_ADAPTATION_FAMILY.md`](MEDICAL_ADAPTATION_FAMILY.md)
for the revised dataset protocol, conditional common-ascent theory, executable
ISIC mechanism gate, its negative target-gradient coverage result, and the
Synapse no-test-label holdout checks. The
base-SAM-only material below is retained as the previous threat-model record;
neither branch has established superiority across the full requested matrix.

`generate.py` optimizes a separate perturbation for every input image or CT
slice on a public foundation SAM model. It supports Synapse/BTCV, ISIC, REFUGE
and matching ViT-B/L/H encoders. The default `strict-base` threat model rejects
alternative checkpoints, trained UMI priors, persistent prior guidance, and
GRAT variants. It does not train an auxiliary model or use target weights.

Quick smoke test:

```bash
PY=/root/anaconda3/envs/adattack/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m peft_transfer.generate \
  --dataset Synapse --vit vit_b --case case0001 --max-slices 1 \
  --variant B --steps 2 --out testset/prsca_smoke
```

For a known CT slice, use `--slice-index 123`; its original slice number is
preserved in the output filename so the target evaluator loads the matching GT.

Strict base-SAM example:

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m peft_transfer.generate \
  --dataset Synapse --vit vit_b --variant K --case case0001 --slice-index 123 \
  --steps 10 --epsilon 10 --step-size 2 --out testset/acsg_case0001_slice123

CUDA_VISIBLE_DEVICES=0 $PY -m peft_transfer.generate \
  --dataset ISIC --vit vit_l --variant D --steps 10 --epsilon 10 --step-size 2 \
  --out testset/prsca/ISIC/vit_l
```

The PEFT-surrogate experiments below use trained LoRA and Adapter models.
They belong to the revised zero-victim-query threat model, rather than the
historical base-SAM-only setting. Checkpoint provenance must be audited:
at least one historical Synapse Adapter `best_dice_checkpoint.pth` was
selected using test-volume labels and is excluded from valid final claims.

Do not run the complete matrix until the sanity gate passes. The historical
fixed-slice runner includes UMI-GRAT comparisons and is not the current
strict-base method. For a strict-base attack, call `generate.py` without
`--threat-model legacy` or external priors.

```bash
$PY -m peft_transfer.generate --dataset Synapse --vit vit_b --variant B \
  --case case0001 --slice-index 123 --steps 10 --epsilon 10 --step-size 2 \
  --out testset/strict_base_b_case0001_slice123
```

For Synapse SAMed evaluation, point the existing honest evaluator at the output:

```bash
$PY test_samed.py --dataset Synapse --vit_name vit_b --num_classes 8 \
  --volume_path data/Synapse_preprocess/test_vol_h5 \
  --list_dir data/Synapse_preprocess/lists \
  --ckpt checkpoints/sam_vit_b_01ec64.pth \
  --lora_ckpt checkpoints/best_weights/SAMed_Synapse_vit_b.pth \
  --module sam_lora_image_encoder --consider_lora \
  --out_path testset/prsca/Synapse/vit_b --img_size 512
```

The original source-only variants have not passed the full-matrix gate.
In the strict-base single-slice smoke test, variant B reaches SAMed
`0.17053/390.73` and MedSA `0.87124/10.51` from clean `0.93883/0.68` and
`0.94000/1.20`, respectively. This does not meet the MedSA transfer goal.
The generation metadata and official report are in
`testset/strict_base_b_case0001_slice123/`.
CT contrast EOT (`CTEOT`) gave a small gain on slice123 but regressed on the
preselected adjacent slice122 for both targets, so it is not the selected
method. A first-block feature loss (`CTEARLY`) also regressed on slice123.
`analyze_layer_transfer.py` and `analyze_task_projection.py` are post-hoc
diagnostics that load target weights and labels only after candidate images
are frozen; they must not be used inside generation. Their results and the
path-integrated amplitude/phase analysis are in
`PHASE_AMPLITUDE_ROBUST_THEORY.md`.
Variant K implements residual-path gradient contraction; variants I/J/N are
theory probes rejected by the current gate. See `RESIDUAL_PATH_THEORY.md` and
`SANITY_RESULTS.md` for those results.

## PEFT-surrogate branch

`generate_2d_dual_peft.py` implements the currently strongest 2D branch. Its
threat model is zero-query transfer to held-out target weights. It uses a LoRA
trajectory proxy and independently trained Adapter proxies from the same
dataset; it is therefore **not** a strict base-SAM-only attack. Target
checkpoints and target gradients are not used during generation.

The selected REFUGE ViT-B configuration combines a LoRA midpoint with three
independent ten-epoch Adapter runs, applies a cross-seed coordinate confidence
penalty, and averages each Adapter gradient over a deterministic `+/-1/255`
input pair:

```bash
CUDA_VISIBLE_DEVICES=0 $PY peft_transfer/generate_2d_dual_peft.py \
  --dataset REFUGE --vit vit_b \
  --samed-proxy testset/refuge_trajectory_midpoint/epoch_79_119_midpoint.pth \
  --medsa-proxies \
    output_runs_correct/MedicalSAMAdapter_seed42proxy/REFUGE_512_pretrain_vit_b_epo10_bs24_lr0.0001_s42_backup/Model/best_dice_checkpoint.pth \
    output_runs_correct/MedicalSAMAdapter_seed43proxy/REFUGE_512_pretrain_vit_b_epo10_bs24_lr0.0001_s43/Model/best_dice_checkpoint.pth \
    output_runs_correct/MedicalSAMAdapter_seed44proxy/REFUGE_512_pretrain_vit_b_epo10_bs24_lr0.0001_s44/Model/best_dice_checkpoint.pth \
  --medsa-uncertainty-lambda 0.5 --gradient-smoothing-sigma 1 \
  --modes --samed-weights 0.45 --candidate-suffix _LCB05_SMOOTH1 \
  --all-cases --out testset/refuge_dual_peft_weighted --device cuda:0
```

On all 400 REFUGE test images this reaches SAMed Dice/HD95
`0.02165/371.14` and Medical-SAM-Adapter `0.36000/98.06`, with clean scores
`0.77834/10.84` and `0.80556/5.37`. Both NPY and quantized PNG artifacts pass
the exact `10/255` audit. The full report is
`testset/refuge_full_smooth_w045/official_full.json`. The exactly centered
boundary ablation also passes the fixed eight-image gate. The theory, novelty
boundary, and rejected phase/amplitude and spatial-target hypotheses are
documented in `PHASE_AMPLITUDE_ROBUST_THEORY.md`.

For Synapse, Medical-SAM-Adapter must receive `[0,1]` tensors because its
dataset-specific normalization is identity. `generate_synapse_dual_peft.py`
therefore passes `medsa_input_scale=1`; the generic 2D path retains 255 for
ISIC/REFUGE. Do not compare CT artifacts generated before this correction.

The strongest current Synapse mechanism is the prompt-free semantic prototype
attack in `generate_synapse_prototype_attack.py`. It uses the known image's
spatial labels and independent same-dataset LoRA/Adapter proxies, but never
uses the target checkpoint, prompt encoder, or mask decoder. Adapter features
are attacked at blocks 2/5/8/11 and the final neck; one mature seed and three
early independent seeds approximate depth and training-trajectory drift.

The frozen ViT-B configuration (`base_weight=0.70`) was evaluated on the
previously unused complete case0029. Under the repository's official
functions it reaches SAMed `0.08666/123.77` and MedSA `0.67810/55.49` at exact
`10/255`; see
`testset/synapse_multilayer_mature_b070_case0029_full/official_volume.json`.
The MedSA Dice is below the supplied best baseline `0.6951`, but its HD95 does
not exceed `85.40`. This is evidence for strong overlap transfer, not a claim
that every Synapse metric has passed. On case0001, the `base_weight=0.75`
configuration reaches SAMed `0.05793/150.69`, while MedSA remains
`0.75632/45.36` because 61/147 slices contain no foreground and the current
`val.py` assigns all absent classes Dice 1.

A same-evaluator audit was also run because the supplied table is not an
executable protocol. Legacy case0029 MI-FGSM and SE-CGR PNGs were projected
back from `11.8125/255` to the exact `10/255` budget, then evaluated together
with the frozen attack. MI-FGSM obtains SAMed `0.49647/43.72` and MedSA
`0.89992/9.17`; SE-CGR obtains `0.04267/118.23` and `0.89443/8.60`; the frozen
attack obtains `0.08666/123.77` and `0.67810/55.49`. Thus it clearly improves
PEFT transfer and SAMed HD95 under the repository protocol, while SE-CGR still
has lower SAMed Dice. See
`testset/baseline_requant_case0029/official_volume.json`.

Rejected branches are retained for reproducibility: diagonal-LDA class banks,
proxy-risk candidate selection, prompt-marginalized decoder coverage, pure and
mixed spatial collision, and a Fourier translation-phase carrier. None is a
selected part of the attack. In particular, the phase carrier changed
case0029 from `0.68224/53.72` to `0.68775/53.93` on MedSA, so it does not
support a positive amplitude/phase mechanism claim. A stronger preregistered
sweep confirms the rejection: carrier weights `0.25` and `0.50` produce MedSA
`0.68941/53.75` and `0.71769/49.64`, respectively, and SAMed
`0.12548/112.29` and `0.20541/83.75`. Increasing the exact linear-phase
component therefore weakens both overlap and boundary transfer. The complete
report is
`testset/synapse_phase_translation_b070_case0029_full/official_volume.json`.

A prompt-free UAD-inspired smooth deformation target is implemented as an
additional falsification branch. Unlike the rejected periodic translation,
it uses a non-periodic smooth displacement and attracts every Adapter depth
toward the deformed-image feature while repelling the clean feature. Official
slice evaluation rejects it: on case0001 slice123 it changes MedSA from the
selected attack's `0.25019/91.48` to `0.78798/17.20`. Post-hoc convex and
steepest fusion with SE-CGR also fails to preserve the PEFT attack. Neither
branch is part of the selected method.

For a quick two-victim diagnostic on one Synapse slice:

```bash
$PY -m peft_transfer.evaluate_synapse_slice --vit vit_b --case case0001 \
  --slice 123 --adv testset/prsca_smoke_case0001/case0001/123.png
```
