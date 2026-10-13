# Anchored Cross-Spectral Gradient (ACSG)

## Motivation

The source final-feature loss is the most direct shared objective across base
SAM and PEFT descendants.  The ADS/SE-CGR carrier is much stronger on SAMed,
but the sanity chain showed that it can reverse the attack on
Medical-SAM-Adapter.  ACSG treats the final-feature gradient as a PEFT anchor
and admits ADS gradient components only inside a frequency-wise trust region.
It uses no prompt, decoder, downstream weight, label, or victim gradient.

## Closed-form phase trust region

Let `a` be the normalized final-feature gradient and `b` the normalized ADS
gradient.  At frequency `w`, denote their Fourier coefficients by `A_w` and
`B_w`, with cross-spectrum

```
c_w = Re(conj(A_w) B_w).
```

For auxiliary weight `lambda` and anchor floor `tau`, ACSG uses

```
s_w = 1                                      if c_w >= 0
s_w = min(1, (1-tau)|A_w|^2 / (-lambda c_w)) otherwise
U_w = A_w + lambda s_w B_w.
```

Therefore, coefficient by coefficient,

```
Re(conj(A_w) U_w) >= tau |A_w|^2.
```

Summing over frequencies and applying Parseval gives the exact pre-momentum
guarantee

```
<a, u> >= tau ||a||_2^2.
```

For a PEFT victim gradient `g_v = a + e`, Cauchy-Schwarz further yields

```
<g_v, u> >= tau ||a||_2^2 - ||e||_2 ||u||_2.
```

This states the required condition precisely: ACSG preserves attack gain when
the shared encoder-feature anchor dominates PEFT gradient drift.  Amplitude
sets the available anchor budget; relative phase determines how much ADS can
be admitted.  There is no fixed claim that low, middle, or high frequency is
universally transferable.

The guarantee applies to the fused direction before momentum, clipping, and
the L-infinity projection.  Empirical segmentation metrics remain necessary.

## Implementation

Variant `H` uses `tau=0.5`, `lambda=1`, ten steps, epsilon `10/255`, and step
size `2/255`.  The implementation is `cross_spectral_anchor` in `core.py`.
The normal generator supports Synapse, ISIC, REFUGE and ViT-B/L/H:

```bash
PY=/root/anaconda3/envs/adattack/bin/python
CUDA_VISIBLE_DEVICES=0 $PY -m peft_transfer.generate \
  --dataset Synapse --vit vit_b --variant H --case case0001 \
  --slice-index 123 --steps 10 --epsilon 10 --step-size 2 \
  --out testset/acsg_case0001_slice123
```

## Prespecified small gate

All values below use the repository SAMed and Medical-SAM-Adapter evaluation
logic.  Lower Dice and higher HD95 indicate a stronger attack.

| Scope | Victim | Clean | B: ADS | H: ACSG |
|---|---|---:|---:|---:|
| slice 123 | SAMed | 0.93883 / 0.684 | 0.72956 / 11.514 | **0.17598 / 401.185** |
| slice 123, paired prompt seed | MedSA | 0.93977 / 1.204 | 0.91224 / 4.749 | **0.88405 / 9.939** |
| slices 121-124 | SAMed | 0.96765 / 0.625 | 0.54718 / 204.506 | **0.43599 / 281.763** |
| slices 121-124, paired prompt seed | MedSA | 0.96133 / 0.866 | 0.95240 / 2.040 | **0.94712 / 2.415** |

Medical-SAM-Adapter's `validation_sam` samples click prompts randomly.  A fair
small-slice comparison must reset the same prompt seed before clean, B, and H.
The prior unpaired one-slice numbers in `SANITY_RESULTS.md` are retained only as
historical diagnostics.

