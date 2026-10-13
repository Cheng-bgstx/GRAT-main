# Spectral drift theory for base-SAM transfer

**Status: estimator rejected.**  The random virtual-PEFT drift radius preserved
the rough band ranking on the fixed slice, but was much smaller than the real
SAMed/MedSA drift and its filtered attacks were weaker than variant B.  It is
kept as a falsified hypothesis.  The accepted method is described in
`CROSS_SPECTRAL_ANCHOR_THEORY.md`.

## Scope

The attacker observes only the base SAM encoder.  SAMed and Medical-SAM-Adapter
are used after generation for evaluation.  Their gradients may be inspected in
diagnostic experiments, but must never choose an attack update or a frequency
weight.

This rules out prompt-conditioned decoder objectives.  Prompts, mask tokens,
and downstream decoders change with the task and fine-tuning method, so a
decoder margin has no stable black-box object to optimize.

## The transfer quantity

Let `L_s(x)` be a source feature-disruption objective and `L_v(x)` the
corresponding objective on an unknown fine-tuned encoder.  Write their input
gradients as `g_s` and `g_v`.  For a small update `u`, the first-order victim
gain is

```
L_v(x + alpha u) - L_v(x) = alpha <g_v, u> + O(alpha^2).
```

Consequently, clean-feature amplitude or phase stability alone says nothing
about attack transfer.  The relevant Fourier object is the gradient
cross-spectrum.  Parseval's identity gives

```
<g_v, g_s>
  = sum_w Re(conj(G_v(w)) G_s(w))
  = sum_w |G_v(w)| |G_s(w)| cos(phi_s(w) - phi_v(w)).
```

Amplitude supplies usable gradient energy.  Phase agreement determines its
sign.  A high-energy band can hurt transfer when its cross-spectrum is
negative.  This is why a fixed low/mid/high-frequency preference has no
general transfer guarantee.

## PEFT drift uncertainty

For orthogonal Fourier bands `P_b`, model fine-tuning as a bandwise gradient
drift

```
P_b g_v = P_b g_s + e_b,       ||e_b||_2 <= eta_b.
```

The normalized source-band direction is

```
u_b = P_b g_s / ||P_b g_s||_2.
```

Cauchy-Schwarz yields the robust first-order lower bound

```
<g_v, u_b> >= ||P_b g_s||_2 - eta_b = m_b.
```

Thus a band is certifiably useful only when its source energy exceeds its PEFT
drift radius.  Under a unit L2 allocation across orthogonal bands, allocating
weight only to positive margins maximizes this separable lower bound.  Under
the actual L-infinity attack, the result motivates a gradient preconditioner;
it does not constitute a Dice guarantee.

## Estimating `eta_b` without a victim

The proposed source-only estimate is the maximum bandwise gradient deviation
over an uncertainty set of low-rank residual views placed at LoRA and adapter
locations:

```
eta_hat_b = max_q ||P_b (g_q - g_s)||_2.
```

This estimate is valid only if the uncertainty set covers real PEFT drift.
`spectral_transfer_audit.py` tests that prerequisite on the fixed diagnostic
slice by comparing the source-only robust-margin ranking with the hidden
SAMed and MedSA first-order band gains.  A negative or weak rank correlation
falsifies the estimator and blocks its use in an attack.

## Acceptance criteria before implementation

1. The FFT cross-spectrum must numerically reproduce the spatial gradient
   inner product.
2. The source-only margin ranking must have positive Spearman correlation with
   both victims and their worst-case band gain.
3. Selected positive-margin bands must improve the predicted worst-victim
   first-order gain over the unfiltered source direction.
4. A source-only attack using the accepted estimator must beat variant B in
   both official one-slice evaluators before any wider experiment.

These checks separate a mathematical lower bound from the empirical coverage
assumption needed to apply it to real LoRA and adapter checkpoints.
