# Robust readout coverage

## Failure addressed

The previous objectives maximize a scalar distance such as

```text
||E(x_adv) - E(x)||_1.
```

This controls total feature energy but not its direction. A large displacement
can lie in a channel subspace ignored by an adapted decoder. The fixed-slice
results exhibit exactly this failure: source gain grows while Med-SA Dice does
not fall.

## Local model

Flatten the final SAM embedding into tokens and channels:

```text
DeltaZ_theta(delta) in R^(N x C).
```

For a locally linear channel readout `w` with `||w||_2 = 1`,

```text
||DeltaZ_theta w||_2^2 / N
  = w^T (DeltaZ_theta^T DeltaZ_theta / N) w
  >= lambda_min(DeltaZ_theta^T DeltaZ_theta / N).
```

Thus the least eigenvalue is a decoder-independent lower bound for every
nonzero channel readout. L1 or Frobenius feature disruption provides no such
worst-readout bound.

The implementation maximizes the differentiable lower approximation

```text
L_cov = -tau log sum_i exp(-lambda_i / tau) <= lambda_min,
```

after normalizing the covariance by clean feature energy.

## PEFT stability

Let the victim displacement be

```text
DeltaZ_v = DeltaZ_0 + R_v,
```

and assume the LoRA/Adapter-induced displacement drift obeys
`||R_v||_2 <= eta`. Weyl's singular-value perturbation bound gives

```text
sigma_min(DeltaZ_v) >= sigma_min(DeltaZ_0) - eta.
```

Therefore the source-computable certified margin is

```text
M(delta) = sigma_min(DeltaZ_0(delta)) - eta(delta).
```

Variant G implements the first term on top of the verified SE-CGR/ADS
backbone. A formal PEFT robustness claim additionally requires a validated
upper bound for `eta`; the current rank-attack scalar sensitivity is not that
bound and cannot be reused without changing its objective to feature-matrix
spectral drift.

## Scope of the guarantee

The bound guarantees encoder readout displacement under the stated local
linearity, nonzero-readout, and bounded-drift assumptions. It does not by
itself guarantee a particular Dice decrease: thresholding, prompts, spatial
mixing, and decoder nonlinearities remain outside the theorem. Beating an
empirical baseline remains an experimental claim.

## Falsification gate

Variant G is retained only if, with the same 10-step `10/255` protocol, it
improves over B on both official victim evaluators or on their prespecified
mean. Victim gradients may diagnose the result but may not tune the attack.
