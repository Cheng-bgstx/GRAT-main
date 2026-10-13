# PEFT transfer hypotheses and validation status

## Setting

For one known image `x`, the attack has access only to the matching foundation
SAM encoder `f_theta`.  The unknown victim is a parameter-efficient adaptation
`f_(theta+Delta)`, where `Delta` is produced by LoRA or an Adapter.  Victim
weights are excluded from generation and used only for final evaluation.

## PEFT uncertainty hypothesis

Both LoRA and bottleneck adapters induce low-rank residual changes in a
transformer block.  PRSCA uses a finite source-side uncertainty set:

```text
U = {source, fixed rank-4/8 attention residuals, fixed rank-4/8 MLP residuals}.
```

For view `u`, the normalized multi-level disruption is

```text
D_u(x') = d(f_u(x'), f_u(x))
          + lambda * mean_l d(f_u^l(x'), f_u^l(x)),
```

where `d` combines relative L1 distance, cosine distance, and channel moment
shift.  Relative normalization prevents a high-energy layer from dominating.

The robust objective is the lower tail of `{D_u}`.  Its first-order update is
approximated by coordinate-wise gradient consensus:

```text
g_bar = mean_u normalize(g_u)
a     = |mean_u sign(g_u)|
g     = g_bar * (c + (1-c) * a).
```

The intended mechanism is that coordinates whose directions change under
plausible PEFT residuals receive less weight. This is a hypothesis, not an
established property of the real SAMed or Med-SA gradients.

## Frequency hypothesis

The FFT is applied to the image gradient, not used to claim that feature phase
is invariant.  A smooth mid-band projection is blended with the unfiltered
gradient. The proposed explanation was that a retained broadband component
would preserve source strength while the band component suppressed unstable
pixel-scale directions.

## What the sanity experiment established

On Synapse `case0001/slice123`, virtual PEFT consensus slightly increased the
measured final-gradient alignment with SAMed but did not lower Dice in the
strict A--F chain. It also failed to improve Med-SA. The mid-frequency stage
reduced attack strength on SAMed, and edge guidance did not recover it.

Consequently, neither PEFT consensus nor the frequency hypothesis is supported
as a method contribution yet. They remain disabled outside explicit ablations.
The strongest current source-only candidate on the official SAMed evaluator is
the original B configuration (SE-CGR/ADS). See `SANITY_RESULTS.md`.

## Required validation

First report clean Dice, adversarial Dice, HD95, feature gain, and gradient
alignment on a fixed small slice set. Compare MI-FGSM, SE-CGR/ADS, UMI-GRAT,
and every incremental A--F configuration under identical epsilon, steps,
slice set, prompt protocol, and random seed. Victim gradients are diagnostic
only and cannot select attack updates or hyperparameters.

The full `dataset x ViT x victim` matrix is allowed only after one source-only
configuration beats both B and UMI-GRAT on the mean of SAMed and Med-SA.
