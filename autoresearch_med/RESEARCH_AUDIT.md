# Phase 0 Research Audit

Date: 2026-10-11

## Current progress

- ACRS is implemented and corrected for the Medical-SAM-Adapter ISIC preprocessing/reference mismatch.
- Its corrected gate used ISIC indices 220-223, two perturbations, two SAMed checkpoints, and two adapter checkpoints.
- Cross-family response-error structure was weakly positive, but source observables improved grouped MAE by 4.66% and true post-hoc ACRS by 2.92%, below the predeclared 5% and 10% gates.
- No ACRS online loss was selected.
- Frozen Joint Pareto was strong on the four fresh cases, but it is proxy-assisted and excluded from source-only evidence.
- SAMed and Medical-SAM-Adapter remain the primary victim families.

## Threat-model audit

The historical `peft_transfer` directory mixes strict source-only, universal-prior, medical-proxy, and target-diagnostic experiments. New research is isolated here. Candidate imports and run manifests are checked before evaluation.

## GR-only audit

- `peft_transfer/core.py` contains `gradient_robust_disruption`, documented as the official GRAT gradient-noise feature loss.
- Historical exact GRAT/UMI-GRAT artifacts exist for a Synapse slice.
- The `GR` variant name in `ablation_config` is not GR-only; it is a quadrature/ADS schedule. Baseline code must therefore select an audited configuration explicitly and cannot rely on the misleading variant label.
- No frozen local ISIC GR-only comparison on both current primary victims was found during Phase 0.
- Status: implementation primitive present; current-protocol reproduction pending.

## Data contamination

ISIC indices 200-223 have participated in development or post-hoc diagnostics and are permanently marked contaminated. New development and confirmation ranges are frozen prospectively. Because the repository contains extensive historical result directories without a complete experiment database, untouched status before 2026-10-11 cannot be proven from filenames alone.

## Evaluation assets

- Public SAM ViT-B checkpoint is present.
- Multiple SAMed ISIC epochs and at least two Medical-SAM-Adapter ISIC lineages are present.
- Existing official evaluators and per-case result loaders are reusable only through isolated evaluation code.
- Budget and protocol audits are now local, deterministic Tier 0 gates.

## Unresolved items

1. Produce an audited local GR-only baseline on the frozen development set.
2. Resolve and hash all selected target checkpoint lineages before Tier 2.
3. Implement feature capture for fixed path points without exposing targets to generation.
4. Run H1 diagnostics on frozen candidates and test incremental prediction.
5. Keep confirmation disabled until a candidate and all hashes are frozen.

## Phase 0 validation

- Static compilation and JSON validation passed in the `adattack` environment.
- Six focused unit tests passed.
- The locked protocol audit passed all partition, import-boundary, budget, sample-identity, path-existence, and checkpoint-hash rules.
- The first append-only registration incorrectly recorded `HEAD` in a repository with no commit. The runner was corrected to record `unavailable_no_commit`; the original row is retained rather than overwritten.

