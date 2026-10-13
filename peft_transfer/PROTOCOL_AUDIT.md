# Synapse transfer-attack protocol audit

## Input-scale contract

The Synapse training NPZ files and clean test H5 files are `float32` in
`[0,1]`. Medical-SAM-Adapter sets `pixel_mean=0` and `pixel_std=1` for
Synapse, so adversarial H5 files must use the same `[0,1]` scale.

The historical `tr.py` writer stored PNG byte values directly as float32.
Those adversarial H5 files therefore used `[0,255]`, while clean evaluation
used `[0,1]`. This scale mismatch reproduces the historical Medical-SAM-
Adapter values near 0.696, but it is not an epsilon-10/255 adversarial
comparison. `tr.py` and `new/test/theory_common.py` now divide PNG values by
255, and the official volume evaluator audits both range and L-infinity
distance.

## Case0008 exact-budget result

All attacks below are evaluated from PNGs on the same `[0,1]` input scale.
Legacy artifacts were projected into the exact integer epsilon-10 envelope.

| Model | Attack | mDSC down | mHD95 up |
|---|---:|---:|---:|
| SAMed | MI-FGSM | 0.336664 | 43.0829 |
| SAMed | legacy official UMI-GRAT | 0.049732 | 131.7117 |
| SAMed | current UMI-GRAT | 0.045140 | 135.3919 |
| SAMed | DX | **0.044918** | **139.6040** |
| Med-SA | MI-FGSM | 0.945397 | 2.8174 |
| Med-SA | legacy official UMI-GRAT | 0.932710 | 3.3901 |
| Med-SA | current UMI-GRAT | 0.933003 | **3.7724** |
| Med-SA | DX | **0.930969** | 3.5184 |

DX strictly improves both metrics over the requantized legacy official
UMI-GRAT on both victims. Against the newly regenerated UMI-GRAT trajectory,
DX improves Dice on both victims and both SAMed metrics, while Med-SA HD95 is
lower by 0.2540. This remaining exception must be reported rather than hidden.

Evidence:

- `testset/volume_case0008/official_results.json`
- `testset/volume_case0008/legacy_baseline_protocol_check.json`
- `testset/volume_case0008/amp_phase_official_results.json`

## Selected source-only method

DX uses the public UMI initialization, the exact GRAT late-feature objective,
and a weight-0.25 carrier spectral uncertainty term. For a BHWC block feature,
the carrier transform is over width and channel. Its probe uses randomized
phase and whitened amplitude, so it tests sensitivity to a family of carrier
reparameterizations instead of assuming a fixed spatial frequency band.

The contribution is the coupling of this carrier uncertainty with the strong
UMI-GRAT trajectory. Fourier phase/amplitude decomposition by itself is not a
novel claim, and the experiments do not support calling the carrier transform
a two-dimensional spatial spectrum.
