# Theory Register

## Problem decomposition

For target medical model `m`, path `z(s)=x+s delta`, encoder response

\[
a_m(s)=J_xE_m(z(s))\delta,
\]

and downstream readout gradient

\[
q_m(s)=\nabla_h L_m(D_m(h,p),y)|_{h=E_m(z(s))},
\]

the target loss change is

\[
\Delta L_m=\int_0^1\langle q_m(s),a_m(s)\rangle ds.
\]

This separates source vulnerability, adaptation response preservation, and downstream decision coupling. Large source feature movement alone does not imply target loss growth.

## ACRS diagnostic

\[
C_{\Delta\theta}(x,\delta)=
[E_m(x+\delta)-E_m(x)]-[E_0(x+\delta)-E_0(x)].
\]

The identity is exact and useful post hoc. The 2026-10-11 gate found weak cross-family structure but insufficient incremental Dice prediction, so ACRS is not an online objective.

## H1: Path-Integrated Response Geometry

For source layer `l`, define

\[
v_l(s)=J_xE_{0,l}(x+s\delta)\delta,
\]

\[
P_l=\frac{\|\int_0^1v_l(s)ds\|_2}
{\int_0^1\|v_l(s)\|_2ds+\eta},
\qquad
M_l=\|E_{0,l}(x+\delta)-E_{0,l}(x)\|_2.
\]

`P_l` is bounded by one in the exact integral limit via the triangle inequality. It measures cancellation along a fixed path, not adaptation stability or task alignment.

For finite computation, the primary estimator uses sampled features directly:

\[
\widehat P_l=
\frac{\|E_l(z_K)-E_l(z_0)\|_2}
{\sum_{k=0}^{K-1}\|E_l(z_{k+1})-E_l(z_k)\|_2+\eta}.
\]

Its numerator telescopes exactly and its denominator is the polygonal path length. This avoids treating Jacobian finite-difference quadrature error as response cancellation. Directional-derivative integration is retained only as a numerical convergence diagnostic.

### Testable prediction

At matched `M_l`, perturbations with larger path consistency should exhibit larger target Dice drop on unseen development checkpoints. The statistic must add grouped held-out predictive value beyond endpoint magnitude.

### Counterexamples

- A constant response direction can be orthogonal to the medical decoder gradient.
- A curved source path can still become task-aligned after adaptation.
- Coarse path quadrature can report artificial consistency.
- Rescaling responses can change endpoint magnitude without changing directional geometry.

### What H1 cannot currently prove

H1 does not give a source-only lower bound on Dice degradation, does not identify an invariant medical readout, and does not justify an online loss before the gate passes.

## H2 and H3 status

H2 will test architecture-constrained preservation only if H1 fails. Random low-rank probes and generic skip-gradient variants are not sufficient novelty. H3 will test multiple independent source readouts only after accounting for prior pseudo-task and prompt-attack failures. Neither is implemented.

## Frequency

Frequency projections may describe response or response-error organization only after explicit BHWC-to-BCHW conversion. No frequency-specific causal claim is accepted without incremental held-out evidence over token, channel, and spatial baselines.

