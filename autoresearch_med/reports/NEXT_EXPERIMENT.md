# Next Experiment

## Phase 1, H1 diagnostic batch

1. Resolve an explicit GR-only configuration from `gradient_robust_disruption`; do not use the unrelated `GR` variant alias.
2. Generate or freeze source-only carrier and GR-only perturbations on development indices 224-227 only.
3. Measure the telescoping path statistic at 0, 0.25, 0.5, 0.75, and 1 for blocks 0, 5, 11, and neck.
4. Require exact budget audit and telescoping residual below `1e-6` before any target evaluation.
5. Add scale, straight-line synthetic, and segment-order negative controls.
6. In an isolated diagnostic process, measure paired Dice on development SAMed and Medical-SAM-Adapter checkpoints.
7. Test H1 incremental prediction beyond endpoint magnitude and ACRS with grouped held-out cases.

Do not implement an online H1 loss or access confirmation indices.

