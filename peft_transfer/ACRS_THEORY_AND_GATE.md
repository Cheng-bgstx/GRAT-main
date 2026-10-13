# Adaptation-Coupled Response Stability

Updated 2026-10-11. This document replaces Task-Observable Target Disruption
(TOTD) and amplitude/phase attack design as the main theoretical direction.

## Research object

Let `E_0(x) = E(x; theta_0)` be the public SAM encoder and
`E_m(x) = E(x; theta_0 + Delta theta)` a medically adapted encoder. For a
fixed input perturbation `delta`, define

```text
d_0 = E_0(x + delta) - E_0(x)
d_m = E_m(x + delta) - E_m(x)
C_DeltaTheta(x, delta) = d_m - d_0
```

The coupled error measures how adaptation changes the response to the same
input perturbation. Under the usual differentiability assumptions,

```text
C_DeltaTheta(x, delta)
  = integral_0^1 integral_0^1
      D_theta D_x E(x + t delta; theta_0 + s DeltaTheta)
      [delta, DeltaTheta] ds dt.
```

The reverse triangle inequality gives

```text
||d_m|| >= ||d_0|| - ||C_DeltaTheta||.
```

This is a feature-response bound, not a segmentation-loss or Dice guarantee.
Decoder behavior must be tested separately through frozen official outcomes.

## Corrected scope

- TOTD is not the main method because source-decoder sensitivity does not imply
  medical-decoder sensitivity.
- GR-only is an external experimental baseline. It is not part of the proposed
  method or combined with a frequency regularizer.
- Frequency is a diagnostic coordinate for `C_DeltaTheta`, not an attack prior.
- FFT is applied only after explicit BHWC-to-BCHW conversion, on spatial H/W.
- Amplitude locking, phase locking, fixed frequency masks and Gaussian spectral
  modulation are not restored.
- Joint Pareto remains a strong proxy-assisted empirical result, not the new
  source-only mechanism.

## Executable mechanism gate

[`analyze_acrs.py`](analyze_acrs.py) performs post-hoc analysis on frozen
candidates. LoRA targets are compared with public SAM under the source input
normalization. Adapter targets are compared with the public SAM loaded into the
Medical-SAM-Adapter architecture with all adapter residuals zeroed. This avoids
counting the Adapter implementation and preprocessing change as learned
adaptation response.

For each selected layer and target it records:

- `||C|| / ||d_0||`, response cosine and the reverse-triangle bound ratio;
- low/mid/high spatial-band mismatch `||P_b C|| / ||P_b d_0||`;
- cross-seed and cross-PEFT cosine structure of the normalized coupled error;
- source-only response strength, layer concentration and spectral concentration;
- official per-case Dice change from results frozen before this analysis.

The first small gate uses grouped leave-one-case-out ridge prediction. The
baseline controls source response strength and target identity. Source-only
observables and the true post-hoc ACRS quantities are tested separately for
incremental prediction. A source-only online objective is permitted only if:

1. true coupled error has stable cross-seed and cross-PEFT structure;
2. true ACRS quantities improve prediction beyond source response strength;
3. a source-computable observable predicts that structure and adds held-out
   Dice predictive value;
4. the result survives targets and cases not used to define the observable.

Failure at any stage is a negative mechanism result. It must not be repaired by
adding a frequency mask, random Adapter, Gaussian modulation or GR Loss.

## First frozen gate result: weak common response, predictive gate failed

The first gate used ISIC Test indices `220..223`, the already frozen `carrier`
and `joint` perturbations, transformer blocks `0/5/11` plus the neck, two LoRA
targets and two Adapter targets. Candidate generation was completed before this
analysis. The result is stored in
`testset/acrs_fresh4/acrs_gate.json`; the exact runner is
[`scripts/run_acrs_fresh4.sh`](../scripts/run_acrs_fresh4.sh), and the compact
record is `results/acrs_gate_20261011.json`.

An audit found that the first Adapter reference accidentally used the Synapse
normalization. The executable gate now requires the zero-Adapter reference and
the adapted target to use the same dataset preprocessing. The corrected coupled
error has the following structure:

| Comparison | Mean coupled-error cosine | Positive fraction |
| --- | ---: | ---: |
| Adapter seed/checkpoint pair | `0.49551` | `1.00` |
| LoRA seed/checkpoint pair | `0.31130` | `1.00` |
| LoRA versus Adapter | `0.21816` | `0.9766` |

The common direction is weak at block 0 (`0.06625`) and stronger at blocks 5
and 11 and the neck (`0.25240/0.28348/0.27052`). This supports a limited claim:
real LoRA and Adapter responses share a small late-depth component. It does not
show that this component controls segmentation failure. Pooled correlations
between Dice drop and low/mid/high relative mismatch are
`-0.212/-0.258/0.022`.

Grouped leave-one-case-out prediction gives:

| Predictor | MAE | Prediction correlation |
| --- | ---: | ---: |
| Target identity + source response strength | `0.15995` | `0.72195` |
| Add source layer/frequency observables | `0.15250` | `0.72302` |
| Add the true post-hoc ACRS mismatch | `0.15528` | `0.72989` |

The source observables improve MAE by `4.66%`, below the preregistered `5%`
small-gate threshold. Access to the true target coupled error improves MAE by
only `2.92%`, below the preregistered `10%` mechanism threshold. The first
mechanism gate therefore fails despite the positive cross-family cosine.

The correct conclusion is narrow but useful. Adaptation-coupled response has a
real and reproducible structure inside each PEFT family and a weak shared
late-depth component across LoRA and Adapter. However, neither total mismatch
nor frequency-resolved mismatch adds enough held-out Dice predictive value.
The strong `joint` attack result is therefore not explained by a universal
stable frequency band or by the measured shared component. Source response
magnitude currently explains the observed Dice drop better than the proposed
mismatch metrics.

No ACRS online attack objective is selected. A future mechanism study must use
a source-computable quantity that predicts the family-conditional response
without knowing which adaptation family is deployed, or introduce a justified
uncertainty set over families. Until such a quantity passes held-out targets,
ACRS remains a rigorous diagnostic framework rather than a claimed attack.
