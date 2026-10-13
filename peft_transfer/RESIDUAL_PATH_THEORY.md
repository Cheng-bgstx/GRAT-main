# Residual-path PEFT transfer: theory and gate results

## Scope

This note records the source-only hypotheses that survived or failed the
`case0001/slice123` gate. It does not use a prompt-conditioned decoder margin.
Victim gradients and checkpoints are used only after generation for diagnosis.

## PEFT residual-path bound

Write a frozen transformer block and its PEFT descendant as

```text
h_(l+1) = h_l + F_l(h_l)
h'_(l+1) = h'_l + F_l(h'_l) + A_l(h'_l),
```

where `A_l` is a LoRA-induced branch change or an Adapter residual. During
attack backpropagation, preserve the forward value but multiply the gradient
through `F_l` by `gamma`:

```text
F_sg(h) = stopgrad(F(h)) + gamma * (F(h) - stopgrad(F(h))).
```

The source Jacobian becomes `I + gamma J_F`. For a victim branch perturbation
`Delta_l`, the product-difference identity gives a first-order bound of the
form

```text
||J_victim(gamma) - J_source(gamma)||
 <= gamma * sum_l ||Delta_l|| * product_(j != l)(1 + gamma M_j)
    + O(||Delta||^2).
```

Thus smaller `gamma` contracts gradient drift introduced through PEFT branches.
The price is reduced source feature gain. `SkipGradientView` in `core.py`
implements this by changing only backward Jacobians.

The gamma gate supports the prediction. At `gamma=0.5`, the diagnostic gradient
cosine rose from H's `0.0435/0.0428` to `0.1127/0.0749` on SAMed/MedSA. The
endpoint scan exposed a real tradeoff: `gamma=0` is much stronger on the
Adapter victim, while `gamma=0.75` is stronger on LoRA.

## Why the rank-tail theorem was insufficient

For one Adapter output, the adversarial-clean correction has channel rank at
most the bottleneck width 192. Eckart--Young gives the exact local bound

```text
min_rank(E)<=192 ||D + E||_F^2 = sum_(i>192) sigma_i(D)^2.
```

The code implements this quantity and an orthogonal Haar-band version. It
increased intermediate rank-tail energy by roughly 20--40 percent, but only
changed MedSA Dice by about 0.006 in the diagnostic evaluator. A direction can
be impossible for one local Adapter to cancel and still be suppressed by later
blocks, the neck, or decoder. Variant J confirmed the gap (formal MedSA Dice
`0.9052`).

## Why fixed frequency decomposition was rejected

For orthogonal spatial projectors `P_b`, Parseval and rank monotonicity give

```text
||D + E||_F^2 >= sum_b sum_(i>r) sigma_i(P_b D)^2.
```

This valid representation bound does not imply segmentation damage. Variant N
maximized a worst-case spectral-ellipsoid separation bound to `0.9499`, yet
increased Dice on both victims. Direct low/high-band splicing restored MedSA
almost to clean (`0.916--0.945`). Effective perturbations depend on cross-band
phase coherence; a fixed low/mid/high assignment is contradicted by the gate.

## Formal one-slice gate

These values come from `test_samed.test_single_volume` and
`Medical-SAM-Adapter-main/function.validation_sam`. MedSA prompt RNG is reset
to seed 24 before every clean/adversarial pair.

| Perturbation | SAMed Dice / HD95 | MedSA Dice / HD95 |
|---|---:|---:|
| Clean | 0.93883 / 0.684 | 0.93977 / 1.204 |
| H stochastic artifact | 0.17598 / 401.185 | 0.88405 / 9.939 |
| K0 (`gamma=0`) | 0.92394 / 0.901 | 0.85258 / 0.500 |
| 10% K0 + 90% H sign fusion | 0.21963 / 393.012 | 0.87790 / 10.125 |

The fusion is a Pareto improvement over H on MedSA while retaining a strong
SAMed attack, but it is insufficient to claim superiority over published
dataset-level baselines.

## Reproducibility warning

H uses one random carrier probe per iteration. Two H artifacts generated from
nominally the same seed but different RNG state have different hashes and
formal SAMed Dice (`0.176` versus `0.727`). The old "accepted H" label is
withdrawn. Future selection must report probe seeds or use a multi-probe
expectation; a lucky single probe is not evidence of transfer theory.

## Gate decision

Do not run the full dataset x ViT x victim matrix. The supported finding is the
residual-path contraction mechanism and its LoRA/Adapter endpoint tradeoff.
Rank-tail energy, fixed frequency bands, spectral ellipsoid separation, and
per-step gradient averaging are rejected as primary objectives by this gate.
