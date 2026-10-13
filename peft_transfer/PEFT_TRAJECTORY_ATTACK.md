# PEFT trajectory continuation attack

## Threat model

The evaluated seed-24 checkpoint is hidden and receives no queries during
generation. An independently trained seed-42 model for the same dataset and
PEFT family is used as a surrogate trajectory sample. The attack is therefore
zero-query black-box, but it is not base-SAM-only.

The attack uses encoder feature disruption only. It does not use prompts,
labels, masks, or a prompt-conditioned decoder margin.

## Fixed ViT-B configuration

```text
epsilon               = 10/255
steps                 = 10
step size             = 2/255
proxy delta scale     = 0.375
prior guidance        = 1.0, constant
first-step multiplier = 2.5
```

Generate one Synapse slice:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 /root/anaconda3/envs/adattack/bin/python \
  peft_transfer/generate_leave_one_seed_peft.py \
  --vit vit_b --case case0001 --slice 123 \
  --proxy-checkpoint \
  output_runs_correct/MedicalSAMAdapter/Synapse_512_pretrain_vit_b_epo100_bs12_lr0.0001_s42/Model/best_dice_checkpoint.pth \
  --guidance 1 --schedules constant \
  --first-step-multiplier 2.5 --initial-delta-scale 0.375 \
  --out testset/peft_trajectory/case0001_slice123 --device cuda:0
```

Every generation record includes the proxy/source architectures, measured
budget, source/prior cosine, and `target_seed24_checkpoint_used=false`.

## Verified small gate

Four fixed maximum-organ slices from cases 0001--0004 were evaluated with the
repository's official SAMed and Medical-SAM-Adapter functions.

| method | SAMed Dice | SAMed HD95 | MedSA Dice | MedSA HD95 |
|---|---:|---:|---:|---:|
| local UMI-PASD | 0.4152 | 212.86 | 0.8303 | 4.77 |
| trajectory continuation | **0.4052** | **253.19** | **0.6938** | **54.08** |

The complete per-case output is
`testset/loseed_fixed0375/official_crosscase.json`. This is a small gate, not
a replacement for full-dataset mDSC/mHD.

## Architecture status

- ViT-B: passes the four-slice local baseline gate on all aggregate metrics.
- ViT-L: the same-architecture seed-42 endpoint is stronger than continuation
  on the tested slice, reaching SAMed/MedSA Dice 0.6870/0.6364.
- ViT-H: no independent same-architecture seed checkpoint is available. A
  ViT-L proxy improves MedSA but does not strictly dominate the ViT-H source
  attack on SAMed. ViT-H is not claimed solved.

## Fourier conclusion

Fourier amplitude and phase are useful diagnostic coordinates. Directly
cross-composing the perturbation amplitude and phase destroys the saturated
pixel-sign structure and weakens both victims. The selected generator does not
apply a frequency mask or an independent amplitude/phase graft.

The conditional Jacobian-drift bound, falsified alternatives, novelty limits,
and full evidence record are in `PHASE_AMPLITUDE_ROBUST_THEORY.md`.

## ISIC dual-PEFT gate

The current strongest ISIC ViT-B candidate uses:

```text
LoRA proxy       = parameter midpoint of epoch79 and epoch119
Adapter proxy    = independent seed42, 10 epochs
target weights   = never loaded during generation
loss             = final encoder feature L1 for both proxies
gradient         = 0.75 * normalize(g_LoRA) + 0.25 * normalize(g_Adapter)
optimizer        = MI-FGSM, momentum decay 1.0
epsilon/steps    = 10/255, 10 steps, step size 2/255
```

Generate the checkpoint midpoint:

```bash
/root/anaconda3/envs/adattack/bin/python \
  peft_transfer/interpolate_checkpoints.py \
  --left output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_79.pth \
  --right output_runs_correct/SAMed/isic_512_pretrain_vit_b_epo200_bs12_lr0.005/epoch_119.pth \
  --right-weight 0.5 \
  --output testset/isic_trajectory_midpoint/epoch_79_119_midpoint.pth
```

Generate adversarial images:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=5 /root/anaconda3/envs/adattack/bin/python \
  peft_transfer/generate_2d_dual_peft.py \
  --dataset ISIC --vit vit_b \
  --samed-proxy testset/isic_trajectory_midpoint/epoch_79_119_midpoint.pth \
  --medsa-proxy output_runs_correct/MedicalSAMAdapter_seed42proxy/isic_512_pretrain_vit_b_epo10_bs24_lr0.0001_s42/Model/best_dice_checkpoint.pth \
  --samed-weights 0.75 --indices 0,1,2,3,4,5,6,7 \
  --out testset/isic_dual_peft_weighted --device cuda:0
```

On the fixed eight-image gate, `W075` obtains SAMed Dice/HD95
`0.1864/194.4` and MedSA Dice/HD95 `0.6969/64.4`. The exact result and budget
audit are in `testset/isic_dual_peft_weighted/official_8.json`. These numbers
pass the cited baseline thresholds on this gate; they are not full-dataset
metrics.

The same frozen candidate was then evaluated on all 379 ISIC test images:

| victim | clean Dice/HD95 | attacked Dice/HD95 |
|---|---:|---:|
| SAMed | 0.8971 / 30.55 | **0.2100 / 231.64** |
| Medical-SAM-Adapter | 0.9149 / 8.00 | **0.6534 / 70.93** |

The complete official output is
`testset/isic_dual_peft_weighted/official_full.json`. Against the supplied
baseline table, the thresholds are SAMed 0.2398/185.01 and MedSA
0.7262/13.64, so the full result is stronger on all four metrics. This remains
a PEFT-surrogate result because attack generation uses a LoRA trajectory
midpoint and an independently trained Adapter proxy.
