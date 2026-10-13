# PEFT phase-amplitude robust transfer

## Strict-base transfer diagnosis (2026-10-04)

Generation uses one image and the frozen public SAM of matching ViT size.
There is no trained prior, proxy fine-tuning, target query, or target gradient.
The following target encoder measurements were made **after** the candidates
were frozen and are diagnostics, not information available to the attack.

For a spatial feature displacement `d_s` in base SAM and `d_v` in a fine-tuned
encoder, Parseval gives the exact cross-model identity (full 2D FFT):

```text
<d_s,d_v> = (1/HW) sum_(channel,frequency)
            A_s A_v cos(phi_v - phi_s).
```

This identity does not imply attack success. If a target segmentation loss has
feature gradient `q_v`, its first-order change is `<q_v,d_v>`, not
`<d_s,d_v>`. A sufficient local transfer condition would require a positive
task projection and a bounded remainder, e.g.

```text
<q_v,d_v> > (L/2) ||d_v||_2^2
```

for a target loss with `L`-Lipschitz feature gradient. Neither `q_v` nor `L`
can be certified from a frozen source SAM alone. LoRA rank and Adapter width
also do not give a bound on the target decoder gradient. Thus architecture
alone cannot guarantee that a source-only attack beats a victim-specific
baseline. This is a testable mechanism, not a transfer theorem without
additional measured assumptions.

The post-hoc layer diagnostic on Synapse case0001/slice123 is in
`testset/strict_base_cteot_case0001_slice123/layer_transfer.json`. For the
strict-base B perturbation, source/target spatial displacement cosine is
`0.996` at block 0 and `0.626` at the neck for SAMed; for MedSA it is `0.941`
and `0.270`, with neck displacement norm only `0.332` of the source norm.
MedSA neck amplitude cosine remains about `0.767` while strong-bin phase
cosine is only about `0.298`. This supports a specific **phase mismatch of
feature displacements** across the Adapter, not the claim that input-image
Fourier phase is universally stable or that amplitude matching causes target
loss to rise. These values are from one CT slice, not a cross-dataset estimate.

Two source-only hypotheses were tested at the same 10-step, `10/255` budget
through the official SAMed and MedSA slice evaluators:

| Candidate | SAMed Dice / HD95 | MedSA Dice / HD95 |
| --- | ---: | ---: |
| Clean | 0.93883 / 0.68 | 0.94000 / 1.20 |
| B | 0.17053 / 390.73 | 0.87124 / 10.51 |
| B + CT contrast EOT (`CTEOT`) | 0.16829 / 490.30 | 0.85158 / 12.81 |
| B + first-block displacement (`CTEARLY`) | 0.22316 / 387.52 | 0.87726 / 9.68 |

CTEOT is a small single-slice gain, far from a convincing Adapter attack.
CTEARLY is a negative control: preserving early displacement transfer did not
improve target segmentation disruption. The next theoretical question is
which source-computable, anatomy-relevant feature directions also have a
nonzero projection onto the unknown target task gradient. Frequency or early
layer stability alone cannot answer it. No superiority over baselines or
generalization across datasets/ViT sizes is established by these results.

### Why a clean-point phase/Jacobian criterion is insufficient

The target-task projection diagnostic is in
`testset/strict_base_cteot_case0001_slice123/task_projection.json`. It uses the
target checkpoints and ground truth **only after generation**. The measured
loss is a differentiable present-class soft Dice surrogate, with the same
fixed MedSA prompt seed as the official slice evaluation. Let `z_0` and `z_1`
be clean and attacked *target* neck embeddings and `d=z_1-z_0`. For the target
decoder loss `L`, the fundamental theorem of calculus gives

```text
L(z_1)-L(z_0) = integral_0^1 <grad L(z_0+t d), d> dt
              = (1/HW) integral_0^1 sum_(c,w)
                A_grad(t,c,w) A_d(c,w)
                cos(phi_d(c,w)-phi_grad(t,c,w)) dt.
```

The second equality is Parseval applied to the spatial Fourier axes. It
couples PEFT-dependent feature displacement, anatomy-dependent decoder
gradient, amplitude, and phase **along the whole path**. A clean-point
gradient replaces this integral by its `t=0` value and is valid only when
curvature is sufficiently small. On this slice it has the wrong sign for every
tested candidate:

| Victim / candidate | Actual soft-Dice loss change | Clean-point linear term | 8-point path estimate |
| --- | ---: | ---: | ---: |
| SAMed / B | +0.7274 | -0.0852 | +0.7348 |
| SAMed / CTEOT | +0.7778 | -0.1306 | +0.7814 |
| SAMed / CTEARLY | +0.6800 | -0.0631 | +0.6821 |
| MedSA / B | +0.1065 | -0.0275 | +0.1060 |
| MedSA / CTEOT | +0.1389 | -0.0128 | +0.1386 |
| MedSA / CTEARLY | +0.0982 | -0.0287 | +0.0970 |

The local negative projection alongside a positive endpoint change directly
refutes any claim that clean-point phase alignment or input-Jacobian cosine
predicts this 10/255 attack. It also explains why increasing an early feature
distance did not help: the decoder loss depends on the nonlinear continuation
from clean to attacked embedding. The path identity is exact, but its target
gradient is inaccessible in the strict black-box attack. A proposed generator
must therefore supply a source-computable anatomical proxy and demonstrate,
on held-out victims and cases, that its path-integrated proxy response tracks
the target response. This condition has not yet been verified; the identity
alone is not a new attack or a performance guarantee.

For B, the post-hoc task-gradient/feature-displacement amplitude cosine stays
in the approximate range `0.42--0.52` along the eight-point path. The
cross-spectral phase coherence changes from `-0.015` to `+0.109` by the fourth
midpoint in SAMed, and from `-0.005` to `+0.020` in MedSA. Thus the sign reversal
of the task projection in this diagnostic is associated primarily with a
change in weighted phase alignment along the path, while amplitude overlap is
present throughout. This observation concerns **target neck task gradients**;
it does not establish that changing the input-image phase directly will cause
the same response. The source-only challenge is to predict that path behavior
without access to the fine-tuned decoder.

The later frozen-candidate check on case0008/slice98 isolates this issue in
the Adapter. A/B/CTEOT have similar mean path amplitude overlap with the
target task gradient (0.381/0.389/0.387), but mean path phase coherence is
-0.0092/+0.0196/+0.0189 and the measured soft-Dice loss change is
-0.0629/+0.0388/+0.0281. A also *improves* the official MedSA Dice from
0.8433 to 0.8777. These values come from
`testset/medical_invariant_gate/case0008_slice98_task_projection.json` and
`case0008_slice98_official.json`. The path phase is computed with the target
task gradient after generation; using it to choose a perturbation would break
the strict black-box threat model. The required new object is a
source-computable estimate of its **signed** response, validated
prospectively on held-out LoRA and Adapter targets. An amplitude or phase
statistic computed only from input images or encoder displacement does not
supply that sign.

A direct source-only input-polar check reinforces the distinction. For each
frozen candidate we measured the Fourier amplitude change
`||A_adv-A_clean||/||A_clean||` and clean-amplitude-weighted phase change
`||A_clean (exp(i phi_adv)-exp(i phi_clean))||/||A_clean||` on the input image.
Each score selected the best MedSA perturbation on only two of six fixed
Synapse slices. On case0008/slice98, both selected A even though it improved
MedSA Dice, while B was the best attack. The report is
`testset/medical_invariant_gate/synapse_vitb_input_polar_6slices.json` and
the scorer is `peft_transfer/score_input_polar_candidates.py`. These input
scores must not be conflated with the target task-gradient phase measured
above. The experiment rejects direct maximization of input amplitude or
phase change as a justified replacement for signed task transfer.

The preselected neighboring slice122 is a counterexample to selecting CT
contrast EOT from slice123 alone. Under the same official evaluator, B obtains
SAMed/MedSA Dice `0.13515/0.93433`, while CTEOT obtains
`0.28388/0.95518`; CTEOT is weaker on both. Clean Dice is
`0.96742/0.96297`. The 8-point path integral remains numerically close to
the target soft-Dice loss change on slice122, but this validates the diagnostic
identity rather than the attack. For MedSA/B, actual/path-estimated change is
`+0.04547/+0.04526`, much smaller than slice123's `+0.10649/+0.10596`.
The quantitative files are
`testset/strict_base_cteot_case0001_slice122/official_slice.json` and
`testset/strict_base_cteot_case0001_slice122/task_projection.json`. CTEOT is
therefore rejected as a selected generator at the present two-slice gate.

Some older CODD/PASD notes used `fftn(feature, dim=(-2,-1))` on BHWC block
outputs. Those axes are width and channel, so their reported "spatial phase"
effects do not establish an HxW spatial-frequency mechanism. The new post-hoc
diagnostics explicitly use BCHW target neck embeddings and `fft2` on HxW.

## Strict-base threat model correction (2026-10-04)

The current task permits only a public, unmodified SAM encoder and the one
known input image during perturbation optimization. It excludes trained
LoRA/Adapter proxies, their checkpoints, learned UMI initializations, and
GRAT as a component of the proposed method. Target models are loaded only by
the separate official evaluator. `generate.py` now defaults to `strict-base`
and rejects those components. All dual-PEFT proxy results below are historical
diagnostics and cannot substantiate performance in this task.

An exact-budget, no-prior variant-B smoke test on case0001/slice123 gives
SAMed `0.17053/390.73` and MedSA `0.87124/10.51` under the official
evaluator, from clean `0.93883/0.68` and `0.94000/1.20`. The large difference
between victims is evidence that a foundation-encoder feature disturbance
alone does not currently control the Adapter's downstream decision boundary.
No phase or amplitude transfer claim is justified from this single image.

## Current conclusion (2026-09-30)

Fourier amplitude/phase decomposition is an exact and useful diagnostic, but
it is established prior art and is not the novelty of the current method.
Fixed frequency bands, exact polar-gradient confidence filtering, learned
phase corrections, and an amplitude-preserving translation-phase carrier all
failed controlled transfer gates. The last carrier follows the Fourier shift
theorem exactly at the image level, yet on complete case0029 it changed MedSA
from `0.68224/53.72` to `0.68775/53.93`; this falsifies the claim that a clean
phase argument alone yields the required transfer.

A stronger full-case sweep closes the remaining carrier-strength question.
Starting from the frozen `base_weight=0.70` attack, translation-phase weights
`0.25` and `0.50` give MedSA Dice/HD95 `0.68941/53.75` and `0.71769/49.64`,
while SAMed changes to `0.12548/112.29` and `0.20541/83.75`. The unmodified
attack is `0.67810/55.49` on MedSA and `0.08666/123.77` on SAMed. The ordered
regression with carrier strength rejects the hypothesis that insufficient
carrier weight caused the earlier negative result. The raw report is
`testset/synapse_phase_translation_b070_case0029_full/official_volume.json`.

The evidence instead supports a narrower PEFT hypothesis. Fine-tuning changes
the task Jacobian at multiple residual insertion depths and along a stochastic
training trajectory. An attack direction should therefore increase semantic
class risk across (i) several Adapter insertion depths and (ii) independently
trained early and mature PEFT checkpoints, while retaining a foundation-SAM
component. If normalized component gradients are `g_b` and `g_p`, then a
positive mixture has positive first-order gain for both whenever
`cos(g_b,g_p)>-1`; this is a local statement and is not a ten-step guarantee.
Antithetic input pairs reduce the leading odd local-curvature error in the
proxy gradient estimate. These facts motivate the implemented estimator but
do not constitute a generic transfer certificate.

The selected Synapse ViT-B result uses target-free generation with an
independent LoRA proxy, one mature and three early Adapter proxies, blocks
2/5/8/11 plus the final neck, public UMI initialization, and base weight 0.70.
On the previously unused full case0029 it obtains SAMed `0.08666/123.77` and
MedSA `0.67810/55.49` under the repository evaluators and exact `10/255`.
MedSA Dice beats the supplied `0.6951` baseline; MedSA HD95 remains below the
supplied `85.40` baseline. No full Synapse superiority claim is justified yet.

Several apparently plausible extensions were directly rejected. A
channel-rescaling-invariant diagonal LDA bank did not transfer. Proxy decoder
risk did not predict target risk. Prompt-marginalized absent-class coverage
overfit independent Adapter checkpoints. Encoder collision with translated
images increased SAMed distance but was repaired by the target Adapter. These
negative controls delimit the proposed contribution: depth and training-time
PEFT coverage explains the observed Dice transfer; decoder-level and generic
frequency arguments currently do not explain the remaining HD gap.

### Smooth deformation manifold test

UAD (Lu et al., CVPR 2024) reports that prompt-agnostic targeted encoder
features can transfer better when the target is a smoothly deformed image:
https://arxiv.org/abs/2404.02585. This motivates a stricter test than the
earlier periodic `torch.roll` collision. Let `D_u x` be a border-padded smooth
warp with spatially varying displacement `u`. For each PEFT depth `l`, the
implemented ascent loss is

```text
L_deform = mean_l [cos(f_l(x_adv), f_l(D_u x))
                   - cos(f_l(x_adv), f_l(x))].
```

It is prompt-free and uses the same deformation across independent PEFT
trajectory proxies. A spatially varying warp couples Fourier amplitude and
phase through the transport term `D_u x - x = grad(x) dot u + O(||u||^2)`;
unlike a global translation, it is not a diagonal linear-phase multiplier.
This gives a defensible amplitude/phase coupling hypothesis, but the official
gate rejects the tested realization. On case0001 slice123, the selected attack
has SAMed `0.13001/482.68` and MedSA `0.25019/91.48`, whereas the combined
semantic/deformation objective has `0.33416/345.94` and `0.78798/17.20`.
Slice65 also does not improve MedSA. Consequently, smooth deformation is kept
as a reproducible negative control and not claimed as the mechanism.

Post-hoc mixing does not repair this gap. The strongest tested SE-CGR/current
steepest blend on slice123 (`source_weight=0.25`) reaches SAMed
`0.14867/482.19` and MedSA `0.33843/76.19`, losing to the selected attack on
both Dice and MedSA HD95. Saturated sign attacks therefore need a shared
trajectory objective; output-space convexity does not imply common target
gain.

### Same-protocol baseline audit

The repository's complete case0029 legacy MI-FGSM and SE-CGR artifacts were
requantized to exact `10/255` and evaluated in one invocation alongside the
frozen method. MI-FGSM gives SAMed `0.49647/43.72`, MedSA `0.89992/9.17`;
SE-CGR gives SAMed `0.04267/118.23`, MedSA `0.89443/8.60`; the selected method
gives SAMed `0.08666/123.77`, MedSA `0.67810/55.49`. This proves a large PEFT
transfer gain under the executable protocol, but it also records the remaining
SAMed Dice deficit relative to SE-CGR. The authoritative report is
`testset/baseline_requant_case0029/official_volume.json`.

## Scope

This note replaces the unsupported assumption that one fixed frequency band is
intrinsically transferable. The attack remains source-only: victim weights,
prompts, labels, and gradients are unavailable during generation.

## Selected result and protocol correction (2026-09-29)

The final controlled gate selects variant `DX`: public UMI initialization,
the exact GRAT objective, and a weight-0.25 shuffled-phase/white-amplitude
carrier probe over the late six encoder blocks. On complete case0008 it scores
SAMed 0.044918/139.6040 and Medical-SAM-Adapter 0.930969/3.5184
(Dice down/HD95 up). The requantized legacy official UMI-GRAT scores
0.049732/131.7117 and 0.932710/3.3901, respectively, so DX is strictly better
on both metrics for both victims under the corrected common-scale protocol.

The historical Medical-SAM-Adapter baseline near 0.696 was traced to a scale
mismatch: clean H5 images are float32 `[0,1]`, while `tr.py` wrote adversarial
PNG byte values as float32 `[0,255]`. Synapse training data are also `[0,1]`,
and the model uses mean zero and standard deviation one. That number therefore
cannot be used as an epsilon-10/255 baseline. The writers and evaluators now
enforce a common `[0,1]` contract and report the measured budget.

The selected carrier transform is not a spatial H-by-W FFT. For BHWC features
it acts on W-by-C and should be interpreted as carrier/channel spectral
uncertainty. Its useful effect is empirical robustness to phase/amplitude
reparameterization along the UMI-GRAT trajectory. Generic Fourier
decomposition is established prior art; novelty can only be claimed for this
PEFT transfer coupling and its controlled evidence.

Exact PNA/PatchOut, rank-192 tail maximization, and final-feature DC amplitude
shift were subsequently implemented and rejected. PNA/PatchOut helped isolated
Med-SA sparse slices but weakened SAMed and the five-slice mean. Rank-tail and
DC objectives optimized mathematically persistent changes that were not
semantically damaging. These results reinforce the distinction between
surviving PEFT and disrupting the downstream task.

## Structural-view uncertainty set

Let `g_k = grad_x L_k(x)` be deterministic gradients obtained from the frozen
base SAM through `K` backward structural views. A view preserves the forward
encoder value and changes only an allowed backward path. Write each Fourier
coefficient as

```text
G_(k,w) = A_(k,w) exp(i phi_(k,w)).
```

The views define

```text
C = conv{g_1, ..., g_K} + {e : ||e||_2 <= eta}.
```

The convex hull represents PEFT gradient drift explained by source structural
views. The residual ball represents unexplained drift. Victim information is
used only after generation to test whether this model is valid.

## Proposition 1: view-hull robust gain

For any update direction `u`, linearity over a convex hull and the dual norm
identity give

```text
min_(g in C) <g,u> = min_k <g_k,u> - eta ||u||_2.
```

For a measured victim gradient `g_v`, with

```text
d_v = dist(g_v, conv{g_1, ..., g_K}),
```

the diagnostic bound is

```text
<g_v,u> >= min_k <g_k,u> - d_v ||u||_2.
```

This separates a weak robust direction from a structural-view family that
does not cover the PEFT descendant.

## Proposition 2: coefficient-wise phase-amplitude bound

For `U_w = B_w exp(i theta_w)`, suppose a PEFT coefficient satisfies

```text
A_v,w >= A_min,w,
|wrap(phi_v,w - phi_bar,w)| <= kappa_w.
```

Then, when the total angular deviation is at most `pi`,

```text
min Re(conj(G_v,w) U_w)
 >= A_min,w B_w cos(|wrap(theta_w-phi_bar,w)| + kappa_w).
```

This follows from the real cross-spectrum identity
`A_v,w B_w cos(phi_v,w-theta_w)`. It yields three useful consequences:

1. `kappa_w >= pi/2` gives no positive phase-only guarantee.
2. A reliable coefficient is maximized by `theta_w = phi_bar,w`.
3. Amplitude must follow robust gain; fixed low/mid/high masks have no special
   theoretical status.

Variant O computes a weighted circular center from deterministic residual-path
gradients. A soft-min emphasizes the least-served path. Each center coefficient
is attenuated by its weakest phase projection over all paths. The resulting
direction is fused with ADS only through the existing anchored cross-spectral
constraint.

## Falsifiable gate

The small gate must report clean and attacked Dice/HD95 on both victims,
per-path minimum cosine, phase concentration, rejected-energy fraction, and
artifact hashes from repeated generation. The hypothesis fails if variant O
does not improve the two-victim mean over ACSG and strong baselines, or if
repeated runs are not bitwise identical.

## Novelty boundary

Fourier decomposition, residual-gradient scaling, and robust optimization over
a convex hull are existing tools. The proposed contribution is their
PEFT-specific coupling: deterministic residual-path views approximate
LoRA/Adapter gradient drift, while phase intervals convert those views into a
coefficient-wise worst-case transfer objective. A novelty claim remains
conditional on real victim-hull coverage and superior formal evaluation.

## Formal gate update (2026-09-29)

The implementation and experiments now invalidate several assumptions in the
earlier proposal. These negative results are part of the method-selection gate
and must not be presented as positive theory.

### Correct Fourier axes

SAM block outputs are BHWC. The legacy PASD/SE-CGR code applies
`fftn(dim=(-2,-1))`, so its transform is over W-by-C rather than spatial
H-by-W. A correct channelwise spatial transform was implemented with BHWC to
BCHW conversion and `rfft2` over H-by-W. Spatially uniform random phase has
equal expected power over spatial frequencies, but this correction alone did
not improve transfer: on case0001/slice123, late-layer spatial PASD obtained
SAMed/MedSA Dice 0.2145/0.8033 versus 0.1587/0.8055 for legacy W-by-C PASD.
Therefore the strength of the legacy objective cannot be attributed to a
correct spatial amplitude-phase decomposition.

### What the current evidence rejects

Under the official `test_single_volume` and `validation_sam` functions, with
10 steps, step size 2/255, and exact L-infinity 10/255 quantization:

1. Persistent spatial or spectral UMI guidance weakens the attack.
2. Average token angular disruption does not improve Medical-SAM-Adapter.
3. Early-layer gradient alignment is high, but early-layer attacks are weak.
4. Static early/late and late/final loss sums do not yield robust transfer.
5. Translation-orbit gradient averaging removes useful high-frequency attack
   components and weakens both victims.
6. A per-frequency source-gain lower bound does not imply victim gain. The
   bound was numerically satisfied while MedSA transfer became worse.
7. Random rank-192 ghost adapters at the exact Space/MLP insertion sites have
   low sign consensus (about 0.177) and do not approximate the trained victim
   Jacobian well enough to improve transfer.

The last two observations rule out a source-only argument based solely on
preserving source Fourier gain or sampling random PEFT parameters. A useful
guarantee must control the change in the input Jacobian induced by trained
PEFT, not only the feature displacement or parameter rank.

### Evaluation protocol correction

The historical `testset/adv_Synapse_SECGR_vit_b` artifacts are not a valid
10/255 comparison: their measured distance from the quantized clean image is
12 gray levels. Newly generated candidates are explicitly quantized inside
the feasible interval and measure exactly 10. Baseline and proposed methods
must be regenerated by the same writer before dataset claims are made.

A single-slice result is also insufficient. A final2-to-late schedule appeared
to win on slice123 (SAMed/MedSA 0.1125/0.7300) but lost on four preregistered
organ-slice quantiles. On those four slices, the strict deterministic dynamic
baseline scored 0.2780/0.8479 while the schedule scored 0.3013/0.8730.

### Remaining defensible direction

The next hypothesis is a data-derived Jacobian uncertainty set. Instead of
random adapter weights, estimate a low-dimensional family of input-Jacobian
rotations from the frozen source across medical proxy images and
architecture-faithful bottleneck placements. The gate is prospective:

1. The learned views must increase source-to-victim gradient coverage on held
   out analysis slices without using victim gradients during fitting.
2. A source-only robust direction must beat the regenerated strict baseline
   on both SAMed and Medical-SAM-Adapter over the same fixed small slice set.
3. Only after this gate passes should ViT-L/ViT-H or full datasets be run.

## Subsequent falsification record (2026-09-29)

### Controlled image-phase experiments

Keeping the UMI amplitude spectrum fixed while replacing its phase established
that phase is causal but not sufficient. Random phase improved Medical-SAM-
Adapter on one slice while severely weakening SAMed. Radial or flat amplitude
with the original phase preserved much more SAMed strength, showing that the
original UMI phase carries most of its useful spatial organization. A learned
low-resolution phase correction, optimized source-only over canonical and
adapter-compatible SAM paths, produced a one-slice gain but failed the four-
slice preregistered gate. The result supports a phase role but rejects a stable
single phase correction as the transfer mechanism.

### Exact low-rank and shared-layer guarantees are task-incomplete

The Eckart--Young rank-tail lower bound is exact only at the first adapter
insertion when the comparison base follows the same architecture and all
adapter residuals are zero. It was numerically verified, but maximizing that
tail made both victim attacks weaker. A later experiment disrupted spatial
adjacency in the frozen patch embedding, which is exactly shared by LoRA and
adapter descendants. It also weakened SAMed monotonically and did not improve
Medical-SAM-Adapter. These experiments separate representation persistence
from task relevance: a perturbation can survive PEFT without affecting the
subspace used by the medical decoder.

### L-infinity geometry correction

Euclidean gradient inner-product preservation did not preserve a ten-step
attack because the implemented update uses coordinate signs and asymmetric
momentum magnitudes. Coordinate-wise sign-cone fusion also failed because
magnitude changes alter subsequent iterates. A stricter sign-flip rule limited
the flipped source-gradient mass to `rho`, giving the exact one-step bound

```text
<g_s, sign(d)> / ||g_s||_1 >= 1 - 2 rho.
```

Even `rho=0.001` (a 0.998 lower bound) weakened SAMed after ten steps. Thus a
one-step guarantee cannot be promoted to a trajectory guarantee without a
stability bound on gradient-sign changes along the entire projected path.

### Baseline identity correction

The local sanity runner previously mapped the label `UMI-GRAT` to ADS variant
`B`. The original repository's GRAT loss is final-feature L1 plus Gaussian
late-layer disruption and corresponds to variant `R`. The mapping is now
corrected. On case0001/slice123 under the official evaluators, regenerated
variant `R` scored SAMed/MedSA Dice 0.4135/0.9312, while the stronger local
UMI-PASD baseline scored 0.1587/0.8055. Results must name these methods
separately.

### Cross-case gate

Maximum-organ slices from cases 0001--0004 gave UMI-PASD SAMed Dice values
0.1587, 0.5019, 0.6693, and 0.3309, and Medical-SAM-Adapter Dice values
0.8055, 0.9542, 0.8623, and 0.6992. The corresponding means are about
0.4152/0.8303. Nearly constant source feature gains across these cases did not
predict target success. Any next theory must explain victim feature transfer
and downstream task relevance separately; source gain alone is not a valid
surrogate certificate.

## Low-phase depth continuation update (2026-09-29)

### Exact polar tangent decomposition

For an input coefficient `X_w = A_w exp(i phi_w)` and a local relative
amplitude/phase change, write

```text
dX_w = X_w (da_w + i dphi_w).
```

If `G_w` is the Fourier transform of an input gradient, Parseval's identity
gives the exact first-order response

```text
<g, dx> = sum_w Re(conj(G_w) X_w) da_w
         - sum_w Im(conj(G_w) X_w) dphi_w.
```

The two coefficient sensitivities are therefore
`S_A,w = Re(conj(G_w) X_w)` and
`S_phi,w = -Im(conj(G_w) X_w)`. The diagnostic in
`analyze_encoder_gradient_alignment.py` measures cross-model cosine in these
coordinates. This is more informative than comparing FFT energy because it
tests how the model responds to amplitude and phase motion.

At the clean point on case0001/slice123, low-band (`r < 0.2`) source-to-MedSA
phase alignment was 0.944/0.906/0.747/0.628/0.644 from early2 through final2.
The corresponding high-band values were
0.914/0.832/0.719/0.500/0.418. Low frequencies are therefore less affected by
PEFT drift for this diagnostic. Amplitude alignment showed a similar trend,
so this result does not establish that phase is intrinsically superior.

### Low-frequency phase chord

The selected input auxiliary is

```text
L_LP(x', x) =
  sum_(w in B) q_w [1 - cos(phi'_w - phi_w)] / sum_(w in B) q_w,

q_w = sqrt(A'_w A_w),      B = {w : normalized_radius(w) < 0.2}.
```

The phase chord avoids the branch cut of a direct angle difference and the
amplitude weight suppresses unstable phase at near-zero coefficients. For
small phase changes, `1-cos(dphi) = dphi^2/2 + O(dphi^4)`. Coefficientwise
phase difference is unchanged by positive amplitude rescaling, although the
normalized weighted average can change when the rescaling varies within the
band. The objective is architecture independent; its transfer still depends
on the encoder Jacobian and is not guaranteed by Fourier algebra alone.

The phase chord has zero gradient at an exactly matching spectrum. The attack
therefore retains the deterministic `1e-5` initial and `1e-4` first-step
jitters. Removing that detail would silently disable the auxiliary at the
initial point.

### Depth continuation and its limit

For a chosen layer family `l`, write a victim gradient as
`g_v,l = g_s,l + e_l`. For any update `u`, Cauchy--Schwarz gives the local
bound

```text
<g_v,l, u> >= <g_s,l, u> - ||e_l||_2 ||u||_2.
```

Early layers have a smaller measured residual but insufficient task
sensitivity; late layers have greater task sensitivity and greater PEFT
drift. The selected ten-step continuation uses final2 PASD for steps 1--5 and
late6 PASD for steps 6--10. It is a homotopy from concentrated semantic
disruption to a wider late-layer basin. The inequality motivates the tradeoff
but does not certify the nonlinear projected trajectory.

A seemingly stronger per-step soft minimum over final2 and late6 was tested
and rejected. On case0001/slice123 it raised SAMed Dice from 0.0883 for LPDC
to 0.7445. Loss values at different depths are not calibrated task risks, so
their numerical minimum is not a valid uncertainty-set objective.

### Controlled amplitude-phase evidence

Adding a relative radial amplitude loss to the same continuation reduced the
four-case SAMed mean Dice from 0.3945 to 0.3561, but increased MedSA Dice from
0.8160 to 0.8465. The amplitude term is therefore victim selective. In
contrast, the low-phase auxiliary at weight 0.05 reduced the four-case means
for both victims:

| method | SAMed Dice | SAMed HD95 | MedSA Dice | MedSA HD95 |
|---|---:|---:|---:|---:|
| UMI-PASD | 0.4152 | 212.86 | 0.8303 | 4.77 |
| depth continuation | 0.3951 | 232.76 | 0.8167 | 6.97 |
| full-band input phase | 0.3945 | 254.64 | 0.8160 | 7.28 |
| low-band input phase (LPDC) | **0.3629** | **277.30** | **0.8160** | **7.21** |

These are fixed maximum-organ slices from cases 0001--0004 under the official
evaluation functions, not dataset mDSC. The exact artifact and per-case record
is `testset/crosscase_input_phase_low/official_results.json`. LPDC regresses on
SAMed case0003, so the small gate supports aggregate improvement but not stable
per-case dominance.

### Novelty boundary after literature audit

Low-frequency perturbations, Fourier amplitude/phase decomposition, phase as
structural information, feature attacks, and frequency-based transfer attacks
all have prior art. Relevant primary sources include:

- Guo et al., *Low Frequency Adversarial Perturbation*, UAI 2020.
- Xu et al., *A Fourier-Based Framework for Domain Generalization*, CVPR 2021.
- Luo et al., *Frequency-Driven Imperceptible Adversarial Attack on Semantic
  Similarity*, CVPR 2022.
- Weng et al., *Exploring Frequencies via Feature Mixing and Meta-Learning for
  Improving Adversarial Transferability*, 2024.
- Xia et al., *Transferable Adversarial Attacks on SAM and Its Downstream
  Models*, NeurIPS 2024.

The defensible contribution is the PEFT-specific coupling and its evidence:
(1) measure layerwise gradient drift between a base SAM and LoRA/Adapter
descendants, (2) use depth continuation to balance task sensitivity against
that drift, and (3) regularize the trajectory in the low-frequency input phase
tangent while leaving amplitude unconstrained. Absolute novelty cannot be
claimed until a complete related-work review confirms that this exact coupling
has not appeared, and superiority cannot be claimed until matched full-dataset
evaluation succeeds.

## Leave-one-seed PEFT continuation (2026-09-29)

### Why a trained PEFT proxy is necessary in the current evidence

A white-box diagnostic that maximizes only the trained MedSA encoder's final
feature L1 distance reaches Dice 0.1902 on case0001/slice123. The task-CE
diagnostic reaches only 0.5390. The failure is therefore not caused by the
absence of a prompt-conditioned decoder loss. It is caused by input-Jacobian
drift in the trained encoder. Gradients of this encoder objective are nearly
orthogonal to the tested base-SAM objectives.

Random exact-shape adapters, public adapter initialization, spatial principal
subspaces, semantic partitions, cancellation-gap objectives, and independent
Fourier amplitude/phase replacement all failed to recover that direction.
High parameter cosine between public initialization and a trained adapter was
also insufficient because composition through all transformer blocks changes
the input Jacobian nonlinearly.

The first useful estimator is an independently trained same-task PEFT model.
The official seed-24 MedSA checkpoint is held out, while a seed-42 checkpoint
provides an encoder-only perturbation. Generation records explicitly contain
`target_seed24_checkpoint_used=false`.

### Conditional transfer bound

Let `g_z = grad_x L_z(x)` be the encoder-disruption gradient for a PEFT model
trained with seed `z`, and let `delta_p` be a proxy perturbation with
`||delta_p||_inf <= epsilon`. Assume the target loss is `M`-smooth in the
epsilon neighborhood and the held-out target and proxy satisfy

```text
||g_target - g_proxy||_1 <= eta_seed.
```

The dual-norm inequality and the descent lemma give

```text
L_target(x + rho delta_p) - L_target(x)
 >= rho <g_proxy, delta_p>
    - rho epsilon eta_seed
    - (M/2) rho^2 ||delta_p||_2^2.
```

This states exactly what the proxy supplies: useful target gain when
cross-seed Jacobian drift is smaller than the proxy margin. Low PEFT parameter
rank alone does not bound `eta_seed`, so rank-tail or random-adapter arguments
cannot replace this assumption.

The implemented continuation uses a scaled proxy endpoint for the first
projected step and then updates with

```text
u_k = normalize(g_CH,k + beta_k normalize(delta_p)).
```

`rho` allocates perturbation budget to the empirical PEFT trajectory, while
`beta_k` keeps subsequent source optimization inside its basin. This is a
regularized continuation, rather than a weighted sum of source and proxy
losses. Direct endpoint blending and PCGrad both failed, which is consistent
with the highly nonlinear ten-step projected trajectory.

### Fixed small-gate result

After selecting `rho=0.375`, constant `beta=1`, and first-step multiplier 2.5
on case0001/slice123, the configuration was frozen and generated separately on
the preregistered slices case0002/115, case0003/161, and case0004/118. Under
the official evaluators, the four-case means are:

| method | SAMed Dice | SAMed HD95 | MedSA Dice | MedSA HD95 |
|---|---:|---:|---:|---:|
| local UMI-PASD | 0.4152 | 212.86 | 0.8303 | 4.77 |
| leave-one-seed continuation | **0.4052** | **253.19** | **0.6938** | **54.08** |

Lower Dice and higher HD95 indicate a stronger attack. The continuation wins
all four aggregate metrics. It still regresses on SAMed case0003, whose clean
slice Dice is only 0.6777. This is a
successful small transfer gate, not a full-dataset superiority claim. The
complete record is `testset/loseed_fixed0375/official_crosscase.json`.

### Perturbation phase/amplitude conclusion

The perturbation itself, rather than the image, was decomposed as
`FFT(delta)=A_delta exp(i phi_delta)`. Cross-composing the CH and PEFT-proxy
amplitudes and phases raised both victim Dice values to roughly 0.8--0.9.
Thus amplitude and phase are jointly coupled by the saturated L-infinity sign
geometry. The decomposition remains useful for diagnosis, but independent
phase or amplitude grafting is rejected as a generation mechanism.

### Novelty and threat-model boundary

Neighboring-checkpoint transfer, meta-surrogate training, transferable
initialization, feature disruption, and Fourier sensitivity are prior art.
The exact PEFT-specific combination of a leave-one-seed encoder trajectory and
budget-split continuation appears distinct in the literature search performed
for this project, but this is not proof of novelty.

More importantly, this gate is not a strict base-SAM-only attack: it uses an
independently trained same-task PEFT surrogate while keeping the evaluated
checkpoint hidden. It is a valid zero-query transfer attack under a surrogate
PEFT threat model. It must not be reported as evidence that an arbitrary
unknown fine-tuning Jacobian can be inferred from the base weights alone. An
unconditional base-only guarantee is impossible without a restriction on the
fine-tuning map, because an allowed downstream residual can rotate the input
Jacobian away from every fixed base-model attack direction.

### Architecture gate

The same numerical continuation parameters do not transfer unchanged across
ViT sizes. On case0001/slice123:

| ViT | candidate | SAMed Dice/HD95 | MedSA Dice/HD95 |
|---|---|---:|---:|
| B | continuation | 0.1667 / 399.56 | 0.6087 / 68.81 |
| L | same-seed proxy endpoint | 0.6870 / 188.62 | 0.6364 / 32.96 |
| H | ViT-H CH | 0.6403 / 189.94 | 0.9491 / 3.27 |
| H | 8% cross-ViT continuation blend | 0.6606 / 190.55 | 0.8212 / 4.68 |

For ViT-L, continuation weakened the same-architecture proxy endpoint. For
ViT-H, where no independent same-architecture seed checkpoint exists, a
ViT-L task proxy improves the Adapter victim only at a small cost to SAMed.
It does not strictly dominate the ViT-H source attack. Therefore the ViT-B
result cannot be generalized to ViT-H, and a same-architecture independent
proxy or a new base-only drift estimator is still required.

The generator records the final source/prior cosine and can emit an adaptive
candidate without victim access. This selector is an engineering diagnostic,
not a theorem: the ViT-H experiment shows that a scalar cosine does not fully
predict the two-victim Pareto ordering.

## Reliability-weighted dual-PEFT trajectory attack (2026-09-30)

### Why the earlier objectives were insufficient

On the fixed ISIC gate, a white-box final-encoder L1 attack against the held-out
SAMed checkpoint reached Dice 0.2404. Centered token-Gram disruption reached
0.4094, while a feature Fourier phase plus log-amplitude loss reached 0.6160.
The corresponding MedSA Dice values were 0.7475, 0.9155, and 0.9188. This
rejects Gram and phase/amplitude disruption as generation objectives in this
setting. Phase and amplitude remain diagnostic coordinates only.

The optimizer also matters: UMI initialization plus standard MI-FGSM reached a
white-box SAMed Dice of 0.1112, whereas the previous asymmetric update did not.
This local improvement did not transfer from a single checkpoint, so it cannot
replace a model-uncertainty construction.

### Parameter-trajectory midpoint

Let `theta(t)` be a twice differentiable LoRA training trajectory and let the
held-out checkpoint lie at `t`. A midpoint proxy formed without loading the
held-out weights is

```text
theta_bar = 0.5 * (theta(t-h) + theta(t+h)).
```

Taylor expansion cancels the first-order trajectory term:

```text
||theta_bar - theta(t)|| <= (h^2 / 2) sup_s ||theta''(s)||.
```

If the input-loss Jacobian is `K_theta`-Lipschitz in model parameters, then

```text
||g(theta_bar) - g(theta(t))||_1
 <= K_theta (h^2 / 2) sup_s ||theta''(s)||.
```

This is a conditional second-order proxy bound. It does not follow from LoRA
rank alone. For the local ISIC run, epoch79 and epoch119 are averaged while the
evaluated epoch99 checkpoint is never loaded during attack generation.

### Cross-mechanism coverage and reliability weighting

The LoRA midpoint does not cover Adapter Jacobians. A separate ISIC Adapter was
therefore trained for ten epochs with seed42. It reaches validation Dice 0.9012
and HD95 2.91; the evaluated Adapter uses seed24 and is not loaded during
generation. Let `q_L` and `q_A` denote mean-absolute-normalized encoder L1
gradients from the LoRA midpoint and independent Adapter proxy. The selected
per-step gradient is

```text
h_w = w q_L + (1-w) q_A,
d_w = sign(h_w),
```

followed by standard MI-FGSM momentum. The fixed reliability weight is
`w=0.75`. Normalization prevents preprocessing and feature-scale differences
from implicitly choosing the weight.

For either target family `j`, assume its gradient is `g_j = g_proxy,j + e_j`.
The first-order target gain obeys

```text
<g_j, epsilon d_w>
 >= epsilon <g_proxy,j, d_w> - epsilon ||e_j||_1.
```

The midpoint construction reduces the LoRA error term under trajectory
smoothness, while the independent Adapter seed supplies the missing mechanism
support. The value `w=0.75` is empirically calibrated and is not implied by the
bound. Equal weighting and the exact two-gradient max-min direction were both
weaker on the eight-case aggregate.

### Fixed eight-image ISIC gate

The same `w=0.75` candidate was evaluated on ISIC test indices 0--7 with the
official SAMed and Medical-SAM-Adapter functions. The perturbation uses ten
steps, step size `2/255`, and `epsilon=10/255`. Both NPY and quantized PNG
audits report zero pixels over budget.

| method | SAMed Dice | SAMed HD95 | MedSA Dice | MedSA HD95 |
|---|---:|---:|---:|---:|
| strongest cited baseline threshold | 0.2398 | 185.01 | 0.7262 | 13.64 |
| dual-PEFT trajectory, `w=0.75` | **0.1864** | **194.4** | **0.6969** | **64.4** |

Lower Dice and higher HD95 are stronger. The exact record is
`testset/isic_dual_peft_weighted/official_8.json`. This passes the cited
baseline thresholds on all four aggregate metrics, but it is an eight-image
gate rather than a full-dataset result.

### Complete ISIC test set

The frozen `w=0.75` candidate was subsequently generated for all 379 ISIC
test images and evaluated with the same official functions and prompt seed.
No parameter, loss, or weight was changed after the eight-image gate.

| method | SAMed Dice | SAMed HD95 | MedSA Dice | MedSA HD95 |
|---|---:|---:|---:|---:|
| clean | 0.8971 | 30.55 | 0.9149 | 8.00 |
| strongest cited baseline threshold | 0.2398 | 185.01 | 0.7262 | 13.64 |
| dual-PEFT trajectory, `w=0.75` | **0.2100** | **231.64** | **0.6534** | **70.93** |

The NPY and quantized PNG audits both report a maximum L-infinity distance of
`10.0000067/255` and zero pixels above the `10.0001/255` numerical tolerance.
The complete record is
`testset/isic_dual_peft_weighted/official_full.json`. This establishes
full-test-set superiority for ISIC/ViT-B under the zero-query PEFT-surrogate
setting. It does not establish the strict base-SAM-only setting or results for
other datasets and backbones.

Across the 379 images and ten attack steps, the mean cosine between the
mean-absolute-normalized LoRA and Adapter proxy gradients is only `0.0655`;
`27.9%` of the 3790 stepwise cosines are negative (range `-0.3642` to
`0.6923`). Thus the two PEFT mechanisms do not supply interchangeable input
Jacobians. This supports separate mechanism coverage and explains why an
equal mean or a conservative max-min direction can cancel useful signal. It
does not analytically determine `w=0.75`; that value remains fixed empirical
calibration from the small gate.

### Threat-model and novelty boundary

The selected method is prompt independent and uses no labels, masks, decoder
margin, victim queries, or evaluated target weights during generation. It is
not a strict base-SAM-only attack: it uses same-task PEFT proxy training, and
the LoRA midpoint uses neighboring checkpoints from the same training run as
the held-out target. Claims must use the term zero-query PEFT-surrogate
transfer. A stricter experiment needs an independently trained LoRA seed for
ISIC.

Checkpoint interpolation, model soups, gradient ensembles, MI-FGSM, and
surrogate-model attacks are prior art. Li et al., *Enhancing Adversarial
Transferability with Checkpoints of a Single Model's Training* (CVPR 2025),
also explicitly attacks through multiple checkpoints from one training
trajectory, so checkpoint diversity or trajectory use cannot be claimed as a
new contribution. The potentially distinct contribution is the PEFT-specific
combination: a second-order midpoint error bound for LoRA, separate
LoRA/Adapter mechanism coverage, and reliability weighting after per-proxy
gradient normalization. This is not proof of absolute novelty and requires a
complete related-work audit and direct comparison before publication.

## Hierarchical PEFT uncertainty (2026-09-30)

### Why one flat ensemble is misspecified

The REFUGE Adapter target was saved at epoch 86, whereas the first independent
proxies were trained for roughly ten epochs. Their three-seed ensemble sharply
reduced target Dice, but its target HD95 plateaued near 41. This exposes two
different nuisance variables:

```text
seed s: initialization, minibatch order, and optimizer noise;
time t: location along the PEFT training trajectory.
```

Treating all `(s,t)` checkpoints as exchangeable loses this structure. The
implemented estimator orders the proxy checkpoints seed-major and models a
normalized input gradient coordinate as

```text
q_(s,t,i) = mu_i + a_(s,i) + b_(s,t,i),
```

where `a` is the between-seed effect and `b` is the within-seed trajectory
effect. For `S` seeds and `T` checkpoints per seed, define

```text
qbar_s        = (1/T) sum_t q_(s,t),
qbar          = (1/S) sum_s qbar_s,
se_seed       = sd_s(qbar_s) / sqrt(S),
se_trajectory = sqrt(mean_s var_t(q_(s,t))) / sqrt(T).
```

The retained coordinate and direction are

```text
c_i = (|qbar_i| - lambda_s se_seed,i
                  - lambda_t se_trajectory,i)_+,
h_i = sign(qbar_i) c_i.
```

`generate_2d_dual_peft.py` implements this formula through
`--medsa-proxy-group-size`, `--medsa-uncertainty-lambda`, and
`--medsa-trajectory-lambda`. With group size one it exactly reduces to the
previous cross-seed lower-confidence estimator.

### Conditional sign and gain guarantee

Assume `a_(s,i)` and `b_(s,t,i)` are centered sub-Gaussian variables with
coordinate scales `sigma_seed,i` and `sigma_trajectory,i`, and the held-out
target is another draw from the same constrained PEFT process. Standard
sub-Gaussian concentration and a union bound imply that, for suitable
confidence multipliers `beta_s,beta_t`, simultaneously over the retained
coordinates,

```text
|mu_i - qbar_i|
 <= beta_s sigma_seed,i/sqrt(S)
  + beta_t sigma_trajectory,i/sqrt(S T).
```

Therefore `c_i>0` certifies the sign of `mu_i`, and

```text
<mu, h> = sum_i |mu_i| c_i >= sum_i c_i^2.
```

For an `L_inf` sign step of size `alpha`, the corresponding first-order
expected PEFT loss gain on retained coordinates is at least
`alpha sum_i |mu_i|`; an `M`-smooth loss subtracts the usual second-order
remainder `(M/2)||delta||_2^2`. In code, empirical standard deviations replace
the unknown population scales, so the implementation is an empirical
confidence rule rather than a finite-sample certified attack.

### Phase/amplitude decomposition used as an uncertainty coordinate

Earlier experiments edited the perturbation spectrum or constrained a fixed
frequency band. Both operations weakened the attack. The new polar option does
neither. Let `X=FFT(x)` and `G=FFT(g)`. For an infinitesimal relative amplitude
change `da` and phase change `dphi`, Parseval gives the two real responses

```text
r_A   = Re(conj(G) X),
r_phi = -Im(conj(G) X).
```

They are orthogonal tangent coordinates of the same pixel gradient. Away from
zero-energy image frequencies, the map is exactly invertible:

```text
C       = r_A - i r_phi,
G       = conj(C) X / |X|^2,
g       = IFFT(G).
```

The implementation applies the hierarchical confidence rule separately to
`r_A` and `r_phi`, then uses this inverse. With both confidence penalties zero,
the result equals the ordinary mean pixel gradient up to FFT roundoff; direct
tests verify the round trip and ensemble identity. Therefore any measured
change comes from rejecting PEFT-unstable amplitude or phase responses, rather
than a frequency mask or a different attack loss. This is a falsifiable use of
the previous decomposition: it is retained only if held-out target alignment
and official attack metrics improve.

### Novelty boundary relative to SETA and checkpoint attacks

SETA, *Exploiting Fine-Tuning Structures to Improve Adversarial
Transferability on Downstream SAM*, already samples ghost LoRA/Adapter
parameters and attacks a random longitudinal ensemble. Li et al., CVPR 2025,
already use checkpoints from one training trajectory. SVRE already studies
cross-model gradient variance. Consequently, ghost PEFT models, checkpoint
diversity, or variance reduction alone are prior art.

The narrower potentially distinct claim is a hierarchical random-effects
model specialized to unknown PEFT descendants, with separate empirical
confidence penalties for seed uncertainty and training-time drift. This claim
still requires a direct SETA comparison and full held-out experiments; the
derivation alone is not evidence of novelty or superiority.

### Two-axis antithetic transfer estimator

The held-out PEFT loss varies along two nuisance axes: the unknown fine-tuning
state `theta` and the local input neighborhood `xi`. The attack now estimates

```text
g(x) = E_theta E_xi [grad_x L_theta(x + xi)]
```

with independent PEFT seeds for `theta` and a paired Rademacher neighborhood
`{+xi,-xi}` for each seed. A Taylor expansion of one proxy gradient gives

```text
grad L(x +/- xi)
  = grad L(x) +/- H(x)xi + 1/2 T(x)[xi,xi] + O(||xi||^3).
```

The pair average cancels the `H(x)xi` term exactly. Its deviation from the
center gradient is therefore second order in the radius, while an unpaired
local sample contains a first-order curvature term. Under bounded third
derivative, the paired local error is `O(sigma^2)` and its squared error is
`O(sigma^4)`. With the default projected (`clip`) implementation, this exact
cancellation applies to pixels in `[sigma,1-sigma]`; clipped boundary pixels
add a first-order remainder proportional to their active gradient mass. The
optional `centered` boundary rule shrinks the per-pixel radius to
`min(sigma,x,1-x)` and restores exact centering everywhere. This local
estimator is then passed through the cross-seed
lower-confidence rule above. The two operations address different errors:
antithetic pairing suppresses optimization-path curvature noise; the
confidence rule suppresses coordinates whose signs change across PEFT runs.

This does not make input smoothing or antithetic sampling individually new.
Wu et al.'s variance-reduced attack already averages gradients in a local
input neighborhood, and later variance-tuning and Gradient Relevance Attack
methods also use nearby samples to correct transferable updates. Input
transform ensembles such as SIA are another established route to local
diversity. Accordingly, neither `E_xi`, random input augmentation, nor local
flatness is claimed as a contribution here.

The candidate contribution is their structured use as a two-axis estimator
for foundation-to-PEFT transfer, together with the coordinate sign guarantee:
on every retained coordinate for which the confidence interval contains the
held-out mean gradient, the update has nonnegative first-order held-out gain.
The implementation exposes the neighborhood radius through
`--gradient-smoothing-sigma` in 0--255 units and uses the same deterministic
pair for all PEFT seeds at an attack step, preventing input noise from being
confounded with between-seed variation. `--smoothing-boundary centered`
selects the exactly centered variant, which is reported separately from the
preregistered projected run.

The centered boundary ablation passes the same eight-image gate: SAMed
`0.02424/371.61` and MedSA `0.31520/102.72`. Compared with projected clipping,
it lowers both Dice values (`0.02470 -> 0.02424`, `0.33969 -> 0.31520`) while
retaining baseline-beating HD95. Thus the observed gain does not require the
one-sided boundary residual. The full REFUGE run keeps the preregistered
projected variant; the centered result is reported as a boundary ablation
rather than selected after observing the gate.

#### Finite-sample sign proposition

For coordinate `i`, let `g_(s,+),i` and `g_(s,-),i` be gradients from PEFT
seed `s` at the paired inputs, and define

```text
z_(s,i) = (g_(s,+),i + g_(s,-),i) / 2,
zbar_i  = (1/S) sum_s z_(s,i).
```

Assume the centered pair is used, the coordinate third derivative is bounded
by `M3_i` in the radius-`sigma` neighborhood, and the seed effect in `z` is
sub-Gaussian with scale `tau_i`. If `mu_i` is the population PEFT gradient at
the center, Taylor expansion plus sub-Gaussian concentration gives, with
probability at least `1-delta` simultaneously over `d` coordinates,

```text
|zbar_i - mu_i|
 <= (M3_i / 2) sigma^2
  + tau_i sqrt(2 log(2d/delta) / S)
 =: B_i.
```

Every coordinate with `|zbar_i| > B_i` therefore has the population sign, and
the soft-thresholded direction

```text
h_i = sign(zbar_i) (|zbar_i| - B_i)_+
```

satisfies `<mu,h> >= ||h||_2^2`. For a sufficiently small ascent step, the
standard smoothness remainder gives positive expected held-out loss gain. The
implemented cross-seed standard error estimates the stochastic term; it does
not estimate `M3_i`. Using `sigma=1/255` controls this omitted term but does
not turn the empirical attack into a certificate. This is why the official
held-out gate remains necessary.

Relevant primary sources:

- Wu et al., *Understanding and Enhancing the Transferability of Adversarial
  Examples*, https://arxiv.org/abs/1802.09707
- Zhu et al., *Boosting Adversarial Transferability via Gradient Relevance
  Attack*, https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Boosting_Adversarial_Transferability_via_Gradient_Relevance_Attack_ICCV_2023_paper.html
- Wang et al., *Structure Invariant Transformation for better Adversarial
  Transferability*, https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Structure_Invariant_Transformation_for_better_Adversarial_Transferability_ICCV_2023_paper.html

### Current REFUGE evidence

On the fixed eight-image REFUGE gate, the three-seed, ten-epoch Adapter
ensemble with a seed-confidence penalty and the LoRA midpoint obtains SAMed
`0.0262/370.52` and MedSA `0.4781/41.40` Dice/HD95. It beats the supplied Dice
baselines and the SAMed HD95 baseline, but not the MedSA HD95 threshold
`81.93`. Replacing L1 disruption by a spatially rolled feature target reaches
at best MedSA `0.5496/38.02`, so spatial transport is rejected. Mature
seed-by-time trajectories at epoch 50 also regress to MedSA `0.6796/18.90`,
despite improving pointwise target-gradient cosine. This shows that alignment
at the clean input is insufficient to predict a ten-step attack path.

Adding a one-pixel-scale (`sigma=1/255`) antithetic neighborhood to the same
three-seed estimator changes the eight-image official gate to SAMed
`0.0260/367.34` and MedSA `0.3293/101.74`. Relative to the unsmoothed version,
MedSA Dice drops by `0.1488` and HD95 rises by `60.34`, while the perturbation
still passes the exact `10/255` NPY and PNG audits. It clears the supplied
MedSA baseline thresholds (`Dice <= 0.7283`, `HD95 >= 81.93`) and the SAMed
Dice threshold (`<= 0.0273`); SAMed HD95 is `2.59` below its supplied threshold
`369.93`. The preregistered weight adjustment from `0.40` to `0.45` obtains
SAMed `0.0247/372.32` and MedSA `0.3397/104.64`. It therefore clears all four
supplied baseline thresholds on the fixed gate. The SAMed HD95 increase from
weight `0.40` to `0.45` is positive on all eight cases (`+1.58` to `+8.19`),
so the aggregate crossing is not caused by a single outlier. A separate encoder-collision
objective, which pulls the adversarial representation toward the representation
of a translated clean image, improves SAMed to `0.0167/377.62` but regresses
MedSA to `0.6448/18.82`; it is rejected. The result supports local path
smoothing rather than an engineered spatial target.

On all 400 REFUGE test images, the preregistered projected variant obtains
SAMed `0.02165/371.14` and MedSA `0.36000/98.06`, from clean scores
`0.77834/10.84` and `0.80556/5.37`. These values clear all four supplied
baseline thresholds: SAMed `0.0273/369.93` and MedSA `0.7283/81.93`. The
budget audit covers 400 NPY and 400 quantized PNG artifacts; maximum measured
distance is `10.0000067/255` and zero pixels exceed the numerical tolerance
`10.0001/255`. The complete report is
`testset/refuge_full_smooth_w045/official_full.json`.

The successful smoothing result also does not support a phase-destruction
explanation. Relative to the unsmoothed `W045` attack, smoothing moves radial
perturbation energy from high frequencies (`0.4129 -> 0.3710`) toward low and
middle frequencies (`0.2660 -> 0.2949` and `0.3212 -> 0.3340`). However, its
amplitude-weighted phase chord decreases in the middle (`0.4198 -> 0.3785`)
and high (`0.7696 -> 0.7569`) bands. High-band relative amplitude change also
decreases (`4.0363 -> 3.7840`). The transfer gain is therefore better
explained by cancellation of local odd-order gradient variation than by
forcing a particular phase or amplitude response. The raw summary is stored
in `testset/refuge_dual_peft_weighted/spectral_smoothing_ablation_8.json`.

The post-hoc target-gradient analysis explains why the polar candidate loses.
For the three-seed mean, target cosine is highest in the low spatial band
(`0.153` versus `0.088` in the high band), but the target gain of the update
sign is `0.034` in the low band and `0.112` in the high band. Low-frequency
alignment is therefore more coherent but carries too little of the usable
attack derivative. In polar coordinates, low-frequency log-amplitude cosine
reaches `0.217`, yet its target sign gain is only `0.0014`. Selecting the
best-aligned band would discard most of the adversarial gain. This directly
rejects frequency coherence alone as a selection criterion; both alignment
and retained gradient mass are required.

Amplitude/phase separation itself is established in Fourier domain adaptation,
frequency adversarial training, and frequency attacks. It is not a novelty
claim here. Representative primary sources include Fourier Domain Adaptation
(https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html)
and Tan et al.'s adversarial amplitude swap analysis
(https://arxiv.org/abs/2305.08439). The contribution of the present experiments
is negative but useful: exact invertible phase/amplitude confidence filtering
does not improve held-out PEFT transfer, despite apparently stronger
low-frequency cosine alignment.

The source-only rank-constrained minimax hypothesis was also retested on this
gate. Its offline condition is strong: the selected rank-2 channel energy
fractions are `0.877--0.944`. With the learned risk gate, however, the robust
branch activates in only `1.25%` of attack iterations and obtains SAMed
`0.0338/357.53`, MedSA `0.7997/5.86`. Forcing the worst low-rank branch at every
iteration improves MedSA only to `0.7831/14.99` and weakens SAMed HD95 to
`354.03`. All saved inputs then pass the `10/255` audit. Thus low-rank
concentration is real but insufficient: it describes feature sensitivity to a
local residual, while it does not identify the task-trained Adapter input
Jacobian. This branch is rejected as the main generator.

### Synapse strict-base boundary

The antithetic estimator was also applied without trained PEFT proxies, using
only base SAM and fixed zero-mean virtual LoRA/Adapter residual views. On
case0001 slices 121--124, the best virtual-view candidate reaches only SAMed
`0.90394/24.05` and MedSA `0.92997/1.84`, from clean `0.96376/0.34` and
`0.96483/0.73`. The base-only candidate is similarly weak. This establishes a
threat-model boundary: local input smoothing cannot identify the CT
fine-tuning Jacobian when the PEFT model axis is represented only by random
untrained residuals. Independent Synapse PEFT proxies are required for the
next gate; the existing `best_weights` files are symlinks to the old training
outputs and are excluded from generation.

### Synapse scale correction and semantic prototype risk (2026-09-30)

The first Synapse PEFT-proxy experiments were invalidated by an input-scale
error. Medical-SAM-Adapter sets `pixel_mean=0` and `pixel_std=1` for Synapse,
and its official validation passes H5 values in `[0,1]` directly to
`preprocess`. The shared 2D proxy path instead multiplied these values by 255,
which is correct for the ISIC/REFUGE loaders but wrong for Synapse. The attack
API now keeps 255 as the 2D default and requires the Synapse caller to pass an
explicit scale of 1. Results generated before this correction cannot support
any CT mechanism claim.

Correcting the scale changes the held-out MedSA result on case0001/slice123
from about `0.93` Dice to `0.4024` for the three-seed encoder estimator. On
four preregistered maximum-organ slices from cases 0001--0004, the same
candidate obtains MedSA `0.48145/34.00` and SAMed `0.61483/172.46`, from clean
scores `0.92590/2.27` and `0.85707/49.09`. The report is
`testset/synapse_correct_scale_crosscase/official_4case.json`.

Global feature distance still spends gradient on directions that need not be
used by a segmentation readout. The prompt-free semantic prototype objective
instead downsamples the known spatial label to the encoder grid and constructs
one normalized clean prototype per represented class and proxy:

```text
mu_(s,c) = normalize(mean_(j:y_j=c) normalize(z_(s,j)(x))),
L_proto,s = mean_c mean_(j:y_j=c)
  CE({<normalize(z_(s,j)(x')), mu_(s,k)> / T}_k, c).
```

Classes are averaged before models, so background size cannot dominate the
attack. The same seed-confidence and centered antithetic estimator is applied
to gradients of `L_proto`. Neither a prompt encoder nor a mask decoder is
evaluated during generation.

A conditional readout statement explains what this objective controls. Let
`q_c(z)` be the target class score and `p_c(z)=<normalize(z),mu_c>/T` the proxy
prototype score. If `|q_c(z)-p_c(z)| <= eta` for every class at a token, then
any prototype inversion

```text
p_k(z_adv) - p_y(z_adv) > 2 eta
```

implies `q_k(z_adv) > q_y(z_adv)`. Thus a prototype-margin inversion transfers
to every downstream readout inside the uniform score envelope. The attack does
not estimate `eta`; independent targets provide the falsification test. This
is a task-relevance condition, not a general guarantee for arbitrary PEFT
weights.

On case0001/slice123, the three-seed Adapter prototype endpoint obtains SAMed
`0.65787/187.76` and MedSA `0.38298/102.71`. Combining 50% independent LoRA
and 50% Adapter prototype gradients obtains `0.62711/106.66` and
`0.35486/106.47`. The MedSA Dice/HD and SAMed HD clear the supplied thresholds;
SAMed Dice does not yet clear its supplied `0.074` threshold, so this is a
passed MedSA mechanism gate rather than a completed Synapse claim.
