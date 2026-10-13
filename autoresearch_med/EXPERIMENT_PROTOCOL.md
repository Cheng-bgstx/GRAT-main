# Locked Experiment Protocol

## Threat model

- Source: public unmodified SAM ViT-B.
- Online input: one current image only.
- Budget: 10 iterations, step 2/255, L-infinity at most 10/255.
- Forbidden during generation: medical checkpoints, medical labels, target predictions, target gradients, target result files, and medical proxy training.

## Data partitions

- Contaminated historical development: ISIC test indices 200-223.
- Development: indices listed in `manifests/development.json`.
- Confirmation: indices listed in `manifests/confirmation.json`; never used in automatic research.
- Partition identity uses dataset row index plus resolved filename to avoid ordering ambiguity.

The new partitions are frozen prospectively on 2026-10-11. “Confirmation” means untouched by this new program, not proven untouched by every historical script; the repository-wide history limitation must be disclosed.

## Checkpoints

Development and confirmation checkpoint lineages are disjoint where available. Target checkpoints are loaded only by isolated diagnostics/evaluation. The public SAM checkpoint is the only checkpoint visible to generation.

## H1 diagnostic

- Reuse frozen perturbations first; do not tune attacks in the mechanism stage.
- Fixed path samples: `[0.0, 0.25, 0.5, 0.75, 1.0]`.
- Initial layers: blocks 0, 5, 11, and neck.
- Use the telescoping chord-to-polygonal-length estimator as the primary path consistency statistic.
- Report endpoint magnitude, path consistency, adjacent segment cosine, sign reversal, and telescoping residual.
- Directional-derivative quadrature is a numerical diagnostic only; it cannot be used for mechanism claims unless residual is below 5% per layer.
- Compare against endpoint magnitude and prior ACRS variables with grouped leave-one-case-out prediction.
- Use paired bootstrap confidence intervals by case and report each target family separately.

## GR-only baseline status

The repository contains a function labelled official GRAT gradient-noise feature loss and historical Synapse runs. It does not yet contain a frozen, audited local GR-only result on the current ISIC development partition for both target families. The paper's Table 2 value is not a local baseline. Baseline status remains `implementation_present_reproduction_pending`.

## Promotion

Tier 3 is not used during Phase 0. It remains executable as an explicitly recorded confirmation research stage. Any change to locked evaluation code invalidates downstream hashes and requires an explicit protocol amendment.

