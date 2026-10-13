# Medical SAM Source-Only Transfer AutoResearch

## Objective

Study why adversarial input directions generated on public base SAM remain effective, or fail, after unknown medical adaptation. Optimize scientific validity first, then paired transfer performance on both SAMed and Medical-SAM-Adapter.

The final generator is

\[
\delta = \mathcal A(x,E_0;\epsilon),\qquad \|\delta\|_\infty\le 10/255,
\]

where generation can access only the current image and public, unmodified SAM.

## Fixed protocol

- Initial backbone: SAM ViT-B.
- Attack: untargeted, per-image, 10 steps, step size 2/255, L-infinity budget 10/255.
- No medical proxy training, medical training labels, target weights, or target queries during generation.
- GR-only is the main external baseline and must be reproduced locally under the same protocol.
- SAMed and Medical-SAM-Adapter are separate primary targets; an average cannot hide failure on either family.
- Frequency is diagnostic only and uses true BCHW spatial H/W axes.
- Tier 3 confirmation is an explicit recorded research stage. It is executable, but confirmation feedback cannot be silently recycled into the same frozen candidate.

## Current research tree

1. H1, Path-Integrated Response Geometry: active diagnostic priority.
2. H2, Architecture-Constrained Response Preservation: fallback if H1 fails.
3. H3, Cross-Readout Response Consistency: high-risk fallback.
4. ACRS: retained post-hoc diagnostic; its 2026-10-11 online gate failed.
5. Joint Pareto: retained proxy-assisted performance reference; excluded from source-only evidence.

## Research loop

For each bounded iteration:

1. State one falsifiable hypothesis and one primary variable.
2. Audit overlap with related methods.
3. State assumptions, counterexample, source-computable statistic, and predicted target behavior.
4. Run Tier 0 checks.
5. Run the smallest development diagnostic that can falsify the claim.
6. Evaluate incremental predictive value beyond target identity, clean Dice, and source endpoint feature magnitude.
7. Record code/config/data/checkpoint hashes and raw per-case results.
8. Keep, revise, reject, or switch branches from evidence.
9. Implement an online candidate only after the mechanism gate passes.

## H1 predeclared gate

H1 first measures path response without changing any adversarial image. For layer `l`, path `z(s)=x+s delta`, and directional response `v_l(s)=J_x E_{0,l}(z(s)) delta`, record:

\[
P_l = \frac{\|\int_0^1 v_l(s)ds\|_2}{\int_0^1\|v_l(s)\|_2ds+\eta},
\qquad
M_l=\|E_{0,l}(x+\delta)-E_{0,l}(x)\|_2.
\]

Before any online H1 loss is written, source-only H1 statistics must achieve all of:

- at least 5% grouped held-out MAE improvement over the fixed reference predictor;
- positive held-out Spearman association with Dice drop after controlling endpoint magnitude;
- consistent sign on both SAMed and Medical-SAM-Adapter development checkpoints;
- survival of path-reversal/order-shuffle and scale negative controls;
- no use of confirmation data or target feedback in statistic construction.

Failure keeps H1 diagnostic-only and triggers a documented revision or H2.

## Required records

Every run gets an immutable `results/run_manifests/<run_id>.json` and one appended row in `results/experiments.jsonl`. Reports distinguish measured facts, inferences, and untested proposals.

