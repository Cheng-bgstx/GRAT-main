# Synapse ViT-B sanity gate

Protocol: `case0001`, slice `123`, 10 steps, epsilon `10/255`, step size
`2/255`. Every saved PNG was checked at an actual pixel-space Linf of 10.
Victim gradients were used only after generation for diagnosis.

## Official evaluator results

The SAMed values below come directly from `test_samed.py`. The Med-SA values
come directly from `Medical-SAM-Adapter-main/val.py` and
`function.validation_sam`. A one-slice H5 was used so these are diagnostic
scores, not dataset mDSC and not comparable to the paper's volume results.

| Variant | SAMed Dice | SAMed HD95 | Med-SA Dice | Med-SA HD95 |
|---|---:|---:|---:|---:|
| Clean | 0.9388 | 0.6840 | 0.8531 | 0.6036 |
| A: MI-FGSM feature | 0.9228 | 4.3810 | 0.8500 | 0.6036 |
| B: SE-CGR/ADS | **0.7296** | 11.5137 | 0.9227 | 3.3855 |
| `UMI.pt` + local GRAT recreation | 0.7848 | **28.8154** | 0.9439 | 1.4472 |
| D: B + multilevel + PEFT | 0.7315 | 11.2910 | 0.9130 | 4.8185 |
| F: current full | 0.7530 | 10.3920 | **0.9096** | **5.3640** |

Lower Dice and higher HD95 indicate a stronger attack. B is the strongest of
these candidates on SAMed. On Med-SA, B, the local UMI/GRAT recreation, D, and
F increase Dice over clean, so they fail the attack sanity gate. This does not
constitute a reproduction of the official paper result; it establishes that
the current repository pipeline has not reproduced that baseline yet.

## Ordered module diagnosis

The fixed-prompt diagnostic evaluator produced:

| Variant | SAMed Dice | Med-SA Dice | cos(source, victim) | cos(final, victim) |
|---|---:|---:|---:|---:|
| C | 0.5691 | 0.8759 | 0.0702 / 0.0676 | 0.0702 / 0.0676 |
| D | 0.5704 | 0.8795 | 0.0745 / 0.0655 | 0.0779 / 0.0656 |
| E | 0.6016 | 0.8856 | 0.0771 / 0.0655 | 0.0779 / 0.0694 |
| F | 0.6048 | 0.8851 | 0.0770 / 0.0650 | 0.0842 / 0.0688 |

The slash-separated cosine values are SAMed / Med-SA. PEFT consensus slightly
raises SAMed gradient alignment but does not improve Dice. The spectral and
edge stages weaken SAMed transfer. They must remain disabled while the Med-SA
failure is investigated.

## Gate decision

Do not launch the dataset x ViT x victim matrix. The next experiment is a
small, organ-containing slice set evaluated through the two official scripts.
The gate to proceed is that the selected method must beat both B and UMI-GRAT
on the mean of SAMed and Med-SA without using victim gradients for generation.

## Historical source-only candidate: H / ACSG

The phase-trust-region attack in `CROSS_SPECTRAL_ANCHOR_THEORY.md` supersedes
the earlier PEFT-consensus and fixed-mid-frequency modules.  On the official
four-slice gate (case0001, slices 121--124), H beats B on both reported metrics:

| Victim | B Dice / HD95 | H Dice / HD95 |
|---|---:|---:|
| SAMed | 0.54718 / 204.506 | **0.43599 / 281.763** |
| Med-SA, paired prompt seed | 0.95240 / 2.040 | **0.94712 / 2.415** |

For slice 123 alone, H reaches 0.17598 / 401.185 on SAMed and 0.88405 / 9.939
on Med-SA.  Generation used only base SAM.  Med-SA prompt sampling was reset to
the same seed for clean, B, and H because `validation_sam` otherwise samples a
different click and makes one-slice comparisons non-reproducible.

The acceptance decision is withdrawn. H uses one stochastic carrier probe per
iteration, and nominally identical H runs with different inherited Torch RNG
state produced formal SAMed Dice 0.176 and 0.727. The strong single artifact is
therefore insufficient evidence for a stable method. See
`RESIDUAL_PATH_THEORY.md` for the residual-path bound, formal paired results,
rejected spectral hypotheses, and the current failed gate.
