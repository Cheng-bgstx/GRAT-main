# Fixed Executable Experimental Protocol for CODD Validation

## Scope

This document defines the fixed, executable theory-validation protocol for the `new/test/` pipeline in the current repository. It is designed to be:

- directly runnable with the scripts already placed under `new/test/`
- statistically safer than the previous smoke-style protocol
- aligned with the CODD claim that transferable vulnerability is mainly carrier-aligned radial drift, while rotational drift is unstable

This protocol is the current repository-standard version. It focuses on the parts that can be executed reliably inside the `new/test/` pipeline now. External transfer baselines from legacy scripts are not part of this fixed suite unless they are explicitly re-wired into `new/test/` in a later iteration.

## Global Rules

### Datasets

- `Synapse`
- `ISIC`
- `REFUGE`

### Source Models

- `vit_b`
- `vit_l`
- `vit_h`

### Victim Models

- `SAMed` for LoRA-style PEFT
- `Medical-SAM-Adapter` for Adapter-style PEFT

### Statistical Unit

- The statistical unit is always the `case`.
- For `Synapse`, slices must be aggregated inside each case before any paired comparison.
- No significance claim may be based on treating slices as independent samples.

### Common Controls

All compared methods inside the same group must share:

- the same dataset split
- the same source backbone
- the same victim checkpoint
- the same attack budget
- the same attack step count
- the same seed
- the same support mask and preprocessing pipeline

### Quality-Control Gates

Any result block must be flagged for review if one of the following happens:

- attacked performance is better than clean performance on the main metric
- two compared attack modes produce numerically identical per-case outputs
- all paired deltas are exactly zero
- the metric is obviously saturated at a floor or ceiling and cannot separate methods

QC flags must be written into the summary JSON.

## Group 1: Core Hypothesis Validation

### Goal

Validate that carrier-aligned radial perturbation is more transferable than isotropic or rotational perturbation under the same attack budget.

### Attack Modes

- `radial`: carrier-aligned radial perturbation, the main CODD method
- `tangential`: pure carrier-orthogonal rotational perturbation
- `isotropic`: isotropic complex perturbation
- `misaligned_radial`: radial-amplitude perturbation injected along a wrong carrier direction

`Clean` is included only as a reference baseline. It is not treated as one of the attack methods in pairwise hypothesis testing.

### Required Outputs

- clean victim metrics
- attacked victim metrics for all four modes
- paired case-level comparisons:
  - radial vs tangential
  - radial vs isotropic
  - radial vs misaligned_radial
- main metrics:
  - attacked `mean_dice`
  - attacked `mean_hd95`
  - `dice_drop = clean_dice - attacked_dice`
  - `hd95_increase = attacked_hd95 - clean_hd95`
- robust paired statistics with:
  - mean delta
  - median delta
  - effect direction
  - Wilcoxon when valid
  - safe fallback when all paired deltas are zero
- QC flags

### Primary Success Criterion

`radial` must show stronger degradation than `tangential`, `isotropic`, and `misaligned_radial` on both victims under paired case-level analysis.

## Group 2: Shared Radial Subspace Initialization

### Goal

Verify that richer offline shared-radial initialization improves online optimization quality and final black-box transfer.

### Setups

- `zero`
- `random`
- `mean`
- `top1`
- `topk`

### Fixed Attack Mode

- always use `radial`

### Required Outputs

- final target metrics on `SAMed` and `Medical-SAM-Adapter`
- source optimization traces from `*_trace.json`
- per-setup summary entries:
  - final `mean_dice`
  - final `mean_hd95`
  - `mean_loss_by_step`
  - `loss_auc`
  - `loss_last`
- paired comparisons:
  - topk vs zero
  - topk vs random
  - topk vs mean
  - topk vs top1
- QC flags

### Interpretation Rule

This fixed executable suite uses source optimization curves plus final target transfer metrics as the mandatory evidence. If a full step-sweep study is needed, it should be run as an additional extension by repeating this group with different `att_numstep` values.

## Group 3: Cross-Architecture and Cross-Modality Robustness

### Goal

Test whether the full CODD attack remains effective across:

- LoRA-style victims and Adapter-style victims
- CT, dermoscopy, and fundus domains
- all three open-source SAM backbones

### Fixed Method

- run the full method only:
  - `radial`
  - `shared_mode=topk`

### Required Outputs

- clean metrics
- attacked metrics
- `dice_drop`
- `hd95_increase`
- QC flags

### Interpretation Rule

This repository-fixed executable suite validates cross-architecture and cross-modality robustness of the full method itself. Comparison against legacy transfer baselines is an external extension and is not part of the mandatory automated `new/test/` suite yet.

## Group 4: Geometry and Drift Visualization

### Goal

Provide direct geometric evidence for CODD in a carrier-aligned local coordinate system.

### Coordinate System

For each selected source frequency coefficient:

- `alpha`: projection of target drift onto the source carrier direction
- `beta`: projection onto the carrier-orthogonal direction

### Figures

- Figure A: drift distribution in `alpha-beta` coordinates for `SAMed` and `Medical-SAM-Adapter`
- Figure B: attack trajectories in `alpha-beta` coordinates for:
  - `radial`
  - `isotropic`
  - `tangential`

### Required Statistics

- `radial_ratio`
- `tangential_ratio`
- `angle_mean`
- `angle_std`
- `anisotropy_ratio = Var(alpha) / Var(beta)`

### Interpretation Rule

The figure is illustrative. The quantitative drift statistics are the primary evidence.

## Default Command Matrix

### Full Theory Suite

```bash
conda run -n adattack python /root/data2/fyc/fyc/GRAT-main/new/test/run_full_theory_validation.py \
  --device cuda \
  --max_cases 3 \
  --max_slices 6 \
  --att_numstep 5 \
  --output_dir /root/data2/fyc/fyc/GRAT-main/new/test_runs/theory_formal_suite
```

### Group 1 Only

```bash
conda run -n adattack python /root/data2/fyc/fyc/GRAT-main/new/test/run_directional_ablation.py \
  --dataset Synapse \
  --vit_name vit_b \
  --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth \
  --samed_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth \
  --medsa_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth \
  --max_cases 3 \
  --max_slices 6 \
  --att_numstep 5 \
  --device cuda \
  --output_dir /root/data2/fyc/fyc/GRAT-main/new/test_runs/group1_directional_vitb
```

### Group 2 Only

```bash
conda run -n adattack python /root/data2/fyc/fyc/GRAT-main/new/test/run_srs_ablation.py \
  --dataset Synapse \
  --vit_name vit_b \
  --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth \
  --samed_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/Synapse_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_159.pth \
  --medsa_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth \
  --shared_subspace_path /root/data2/fyc/fyc/GRAT-main/new/offline_codd_synapse_vit_b.pth \
  --max_cases 3 \
  --max_slices 6 \
  --att_numstep 5 \
  --device cuda \
  --output_dir /root/data2/fyc/fyc/GRAT-main/new/test_runs/group2_srs_vitb
```

### Group 3 Only

```bash
conda run -n adattack python /root/data2/fyc/fyc/GRAT-main/new/test/run_macro_robustness.py \
  --dataset ISIC \
  --vit_name vit_l \
  --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_l_0b3195.pth \
  --samed_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/isic_512_pretrain_vit_l_epo200_bs16_lr0.005/epoch_159.pth \
  --medsa_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/isic_256_pretrain_vit_l_epo100_bs1_lr0.0001_s24/Model/best_dice_checkpoint.pth \
  --shared_subspace_path /root/data2/fyc/fyc/GRAT-main/new/offline_codd_isic_vit_l.pth \
  --max_cases 3 \
  --max_slices 6 \
  --att_numstep 5 \
  --device cuda \
  --output_dir /root/data2/fyc/fyc/GRAT-main/new/test_runs/group3_macro_isic_vitl
```

### Group 4 Only

```bash
conda run -n adattack python /root/data2/fyc/fyc/GRAT-main/new/test/visualize_codd_geometry.py \
  --dataset REFUGE \
  --vit_name vit_b \
  --sam_ckpt /root/data2/fyc/fyc/GRAT-main/checkpoints/sam_vit_b_01ec64.pth \
  --samed_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/SAMed/REFUGE_512_pretrain_vit_b_epo200_bs8_lr0.005/epoch_159.pth \
  --medsa_ckpt /root/data2/fyc/fyc/GRAT-main/OUT_ROOT/MedicalSAMAdapter/REFUGE_512_pretrain_vit_b_epo100_bs2_lr0.0001_s24/Model/best_dice_checkpoint.pth \
  --att_numstep 5 \
  --device cuda \
  --output_dir /root/data2/fyc/fyc/GRAT-main/new/test_runs/group4_geometry_refuge
```

## Output Contract

Each group summary JSON must be self-contained enough to diagnose:

- which configuration was run
- how many cases were included
- what the clean and attacked metrics were
- whether pairwise superiority was supported
- whether QC flags indicate saturation or abnormal evaluation behavior

The summary JSON is the authoritative machine-readable artifact. Figures are secondary.
