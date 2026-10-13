# Source-Only Medical SAM AutoResearch Rules

These instructions apply to every file under `autoresearch_med/`.

## Scientific boundary

- Candidate generation may read only the current image, the public unmodified SAM ViT-B checkpoint, and the source-only configuration.
- Candidate code must not import target loaders, medical checkpoints, medical labels, target metrics, or files under `evaluation/`.
- Target checkpoints are permitted only in isolated diagnostic or frozen evaluation processes.
- ISIC test indices 200 through 223 are development-contaminated. They may be used for historical analysis, never untouched confirmation.
- `configs/confirmation.yaml` and `manifests/confirmation.json` are frozen after Phase 0. Confirmation may run as an explicitly recorded research stage, but its outcomes must not be recycled into the same candidate's method selection.
- Frequency analysis must operate on explicit BCHW spatial H/W axes. Amplitude/phase locking, fixed frequency attack masks, and Gaussian spectral modulation are retired.
- GR-only is an external baseline. Do not incorporate its loss into a proposed method.
- ACRS is a diagnostic definition, not an accepted online objective.

## Change boundary

- During a candidate iteration, edit `candidates/active.py` and that run's append-only record only.
- Treat `evaluation/`, `configs/confirmation.yaml`, and `manifests/confirmation.json` as locked protocol code.
- Change locked files only for a documented protocol bug. Record the old and new hashes and invalidate affected results.
- Never overwrite prior experiment rows or run manifests.

## Promotion gates

1. Tier 0: mathematical checks, unit tests, gradient checks, budget and protocol audit.
2. Tier 1: source-only mechanism diagnostics on a small development subset.
3. Tier 2: paired evaluation on independent development images and development checkpoints.
4. Tier 3: explicitly recorded frozen confirmation after configuration freeze.

No online H1/H2/H3 attack may be implemented until its predeclared mechanism gate passes.

