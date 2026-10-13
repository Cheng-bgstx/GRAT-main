# Medical invariant transfer gate (Synapse, ViT-B)

## Protocol

The source-only basis was fitted on training cases 0005/0006/0007 and tested
for acquisition stability on 0009/0010. Target checkpoints were used only in
post-hoc alignment and official evaluation. Candidate perturbations used the
public ViT-B SAM, ten steps, an L-infinity limit of 10/255 and step size 2/255.
The fixed evaluation slices were 0001/122, 0001/123, 0002/104, 0003/98 and
0004/104. These slices are a diagnostic gate, not a dataset mDSC estimate.
The Synapse ViT-B MedSA checkpoint was audited against the instantiated
evaluator model: all 488 keys and tensor shapes match, with no missing or
unexpected entries. Its construction follows the same SAM registry entry as
`Medical-SAM-Adapter-main/val.py` via `get_network`.
The requested MedSA `val.py` metric assigns Dice 1 to classes absent in both
prediction and ground truth, and averages HD95 only over classes present in
both. `checkpoints/best_weights/README.md` explicitly distinguishes this
from its unified metric and reports inflated Synapse clean Dice under direct
`val.py`. We use `val.py` consistently within each MedSA candidate comparison;
its single-slice values are not interchangeable with SAMed's metric or a
published dataset table without a matched protocol audit.

## What the proposed bound actually requires

For each acquisition environment `e`, let `s_e` and `t_e` be the projected
source and target adversarial feature *displacements*. The statement

`t_e = s_e A + r_e`

implies `||t_e|| >= sigma_min(A) ||s_e|| - ||r_e||`. This is a conditional
linear-alignment inequality. A small residual on clean images does not bound
`r_e` on the adversarial path. If `||r_e|| <= b` uniformly on that path, and
the population mean and standard deviation of `||s_e||` are `mu` and `sigma`,
Cantelli gives, with probability at least `1-alpha` over a specified
environment distribution,

`||t_e|| >= sigma_min(A) [mu - sqrt((1-alpha)/alpha) sigma] - b`.

The seven deterministic views used here provide empirical moments, not a
population confidence guarantee. In particular, choosing `kappa=1` is a
ranking heuristic rather than a 95% lower confidence bound.

Even a positive target-feature lower bound would not imply higher segmentation
loss. For the target decoder loss along the target feature path `z(t)`, the
relevant quantity is exactly

`L(z(1)) - L(z(0)) = integral_0^1 <grad L(z(t)), z(1)-z(0)> dt`.

An anatomy-stable channel can be orthogonal to the target decoder's task
gradient, or point in a direction that improves its prediction. A transfer
claim therefore needs a separate, empirically supported source-computable
surrogate for this signed path response. Neither acquisition stability nor
clean-feature alignment supplies one.

For the repository's Synapse evaluation there is also prompt-policy shift.
SAMed evaluates without points, while MedSA's `validation_sam` calls
`generate_click_prompt` on the ground-truth mask to form one *joint* set of
nine class points. The seed-24 clean/adversarial comparison on
case0002/slice104 reproduced identical point coordinates and labels
(`1,0,0,1,0,1,1,1,1`), so prompt randomness does not explain the measured
attack difference. The label-dependent prompt distribution is nevertheless
unknown to an image-only attacker. If `h(p)` is a bounded prompt-specific
Dice-loss change with `|h(p)|<=1`, then for source prompt law `Q` and target
law `P`, `|E_P h-E_Q h|<=2 TV(P,Q)`. The single-point grid law used by the
public-decoder control and MedSA's joint nine-point law have disjoint prompt
types, making this bound vacuous. Matching prompt support would still leave
the PEFT task-gradient alignment condition below to verify.

An explicit counterexample shows why the extra assumption is necessary.
Suppose clean features lie on `z=(u,0)` with foreground label `y=1` for
`u>0`. Two downstream readouts with logits `u+kv` and `u-kv` agree on every
clean sample and share exactly the same (identity) encoder representation.
For an attack displacement `(0,d)` with `d>0`, their foreground
cross-entropies move in opposite directions. Thus perfect clean alignment,
perfect feature survival, and equally accurate clean segmentation still do
not determine the sign of attack transfer. This construction does not claim
that trained SAM decoders realize these exact logits; it disproves a guarantee
derived from representation survival alone.

One possible sufficient condition makes the required task alignment explicit.
Write the target feature displacement as `d_t=A d_s+r`, where `A` is
invertible on the selected subspace, and let `q_s(t)` be a source-computable
task-proxy gradient along a matched feature path. If the true target gradient
satisfies `q_t(t)=A^{-T}q_s(t)+e(t)`, then

`<q_t(t),d_t> = <q_s(t),d_s> + <A^{-T}q_s(t),r>
                 + <e(t),A d_s> + <e(t),r>`.

Consequently, with `sigma_min(A)>=m`, `||A||<=M_A`, `||r||<=R`,
`||q_s(t)||<=M_q`, and `||e(t)||<=E` along the path,

`Delta L_t >= integral_0^1 <q_s(t),d_s> dt
              - M_q R/m - E M_A ||d_s|| - E R`.

The inverse-transpose relation is the covariant gradient rule for a genuine
change of feature coordinates; it is **not** implied by a clean-feature
regression fit. The measured MedSA mapping has small `m`, and the tested
source-region proxy lacks the required task correspondence. Thus none of the
error terms is presently certified. This bound specifies a falsifiable
research condition rather than supporting the current attack.

## Observations

- Held-out acquisition nuisance/signal at source block 0 was 0.038 for the
  fitted subspace versus 0.287 for a random subspace; at the neck it was
  0.019 versus 0.055. This supports only acquisition stability under the
  chosen transformations.
- Clean-image SAM-to-MedSA neck alignment residual was 0.553 in the fitted
  subspace versus 0.768 random and 0.735 complement. For SAMed it was 0.458
  versus 0.663 and 0.681. MedSA's fitted neck mapping had minimum singular
  value 0.0167, and every measured attack had a zero nontrivial triangle
  lower bound.
- On case0002/slice104, MedSA clean Dice was 0.9066. The simple A attack
  reached 0.7637, whereas B/CTEOT/LPDC reached 0.8879/0.8801/0.9146.
  Neck invariant LCB ranked CTEOT highest and A lowest. Thus the proposed
  ranking selected a much weaker MedSA attack on this fixed case.
- Across 17 candidates from the five slices, neck LCB Spearman correlation
  with Dice drop was -0.108 for SAMed and +0.120 for MedSA. Candidates on
  the same slice are correlated, so these pooled numbers are descriptive;
  the per-slice rankings in the prediction report are the primary check.
- The best attack varied by slice and victim. On case0004/slice104 CTEOT
  reduced MedSA Dice from 0.8035 to 0.6393, while on case0003/slice98 all
  four candidates changed MedSA Dice by less than 0.006. Case0003 and
  case0004 also had low SAMed clean Dice, which limits their use as
  standalone attack-strength gates.
- Post-hoc decoder-path diagnostics make the missing factor concrete. On
  case0002/slice104, MedSA A/B/CTEOT neck displacement norms were
  16.22/21.23/20.99, but present-class soft-Dice loss changes were
  +0.231/+0.034/+0.046. On case0004/slice104, their changes were
  +0.086/+0.040/+0.208. Eight-point path integrals tracked these loss
  changes much more closely than clean-point linear terms. The target
  gradient and its Fourier phase in this diagnostic were never available
  during generation, so this is a mechanism check, not a usable attack score.
- An image-only SLIC-region source-SAM proxy also fails candidate selection.
  On case0002/slice104 its A/B/CTEOT loss increases were
  1.349/1.351/1.528, ranking the weakest MedSA candidate first and the
  strongest last. It ranked CTEOT first correctly on case0004/slice104, but
  the case0002 counterexample precludes using it as a general surrogate.
- Raw local-pixel order reversal was tested as a stricter acquisition
  invariant: a strictly increasing pointwise intensity transform preserves
  each *fixed pair's* order and whether the attack reversed it. Pair
  eligibility, based on the source image's contrast at 10/255, is fixed
  before any transform. The resulting source-only score selected the strongest
  MedSA candidate on only one of five fixed slices. It selected LPDC on
  case0002/slice104, where LPDC slightly improved MedSA Dice instead of
  causing a drop. Thus local rank disruption is also insufficient as a task
  proxy. This says nothing about arbitrary nonmonotone acquisition changes.
- A fixed 50:50 pixel-space blend of A and B failed a two-case pilot. On
  case0002/slice104 its SAMed/MedSA Dice was 0.8003/0.8953 versus
  0.3631/0.8879 for B and 0.7928/0.7637 for A. On
  case0004/slice104 it was 0.6248/0.7821, worse than both endpoints on
  SAMed and worse than A on MedSA. Endpoint input-perturbation cosines were
  0.373 and 0.346. Convex blending also reduced mean absolute perturbation,
  so this pilot rejects that fusion, not every possible joint objective.
- A ground-truth class-prototype score on frozen base-SAM neck tokens was
  checked **after** candidate generation. On the original four slices with
  at least two represented classes, it selected the strongest MedSA candidate
  in three, but missed case0002/slice104: it ranked CTEOT above A despite A's
  much larger MedSA Dice drop. This score requires an image's segmentation
  mask and is not an authorized label-free attack objective.
- For a new fixed case0008/slice98, the prototype prediction was saved before
  target evaluation: A/B/CTEOT source loss increases were
  0.904/1.066/1.490, predicting CTEOT. CTEOT was strongest on SAMed
  (Dice 0.279 versus 0.484/0.371 for A/B), but B was strongest on MedSA
  (Dice 0.828 versus 0.878/0.833). MedSA clean Dice was 0.843, so even the
  best candidate had a small drop. This is a prospective counterexample to
  using the label-prototype score as a universal PEFT transfer predictor.
- On that same prospective case, the MedSA neck strong-bin source-to-target
  displacement phase cosines for A/B/CTEOT were 0.296/0.276/0.224. A had
  the highest phase agreement, yet its target soft-Dice loss *decreased* by
  0.063; B and CTEOT increased it by 0.039 and 0.028. Eight-point path
  integration closely matched all three endpoint loss changes. Source-target
  displacement phase agreement therefore does not certify the signed
  downstream task response. The exact first-order low-rank inner problem in
  `rank_attack` likewise describes feature sensitivity, not this task sign;
  its existing REFUGE gate already found weak Adapter transfer.
- A label-free region alternative clustered the clean base-SAM neck tokens
  into eight groups on each image, then scored class-balanced prototype loss
  on frozen attacks. It selected the strongest MedSA candidate on only two
  of six fixed slices. In particular, it gave case0002/slice104 scores
  1.915/2.231/2.487 for A/B/CTEOT, again ranking CTEOT above the much
  stronger MedSA attack A. The clusters are source-feature partitions, not
  verified anatomical classes; this test does not justify promoting their
  loss to an online attack objective.
- A source-only prompt-marginal mask-disruption control used the **original
  pretrained four-mask public SAM decoder** (`num_classes=3`), verified its
  mask-token weights against the public checkpoint, and attacked clean masks
  from nine fixed grid points without labels or a trained prior. On
  case0002/slice104 its source loss rose from 0.166 to 7.403 in ten steps,
  but official MedSA Dice moved from 0.9066 clean to 0.9071 attacked;
  SAMed Dice was 0.6196, weaker than B/CTEOT. The PNG has exact 10/255
  maximum deviation and no over-budget pixels. This rejects the tested
  prompt-marginal source readout as a transfer objective and supports the
  user's concern about decoder generalization. It is not a test of every
  prompt distribution or mask loss.
- A stricter prompt-type control used three fixed *joint nine-point* prompt
  sets (all positive and two alternating positive/negative patterns), with
  no target labels or target weights. Its source loss rose 0.118 to 3.009 on
  case0002/slice104, but official MedSA Dice was 0.9070 versus 0.9066 clean
  and SAMed Dice was 0.8054. The PNG passed the exact 10/255 audit. Matching
  the joint prompt type is therefore insufficient in this pilot; the learned
  Adapter encoder/readout mismatch remains. Prompt-based modules are not
  promoted to the proposed strict-base method.
- A frozen-candidate token-geometry check measured centered, normalized
  source-SAM token Gram displacement at blocks 0/5/11 and the neck, using
  only the public base encoder and each input image. Each individual layer
  selected the strongest MedSA candidate on only two of six fixed slices.
  Block 0 correctly selected A on case0002/slice104 but incorrectly selected
  A on case0008/slice98; there A *improved* MedSA Dice from 0.8433 to
  0.8777, while B reduced it to 0.8277. Block 5 selected B on case0008 but
  selected LPDC on case0002, where A was much stronger. This is a post-hoc
  ranking diagnostic, not a newly generated attack. Its output is
  `synapse_vitb_source_geometry_6slices.json` and the scorer is
  `peft_transfer/score_source_token_geometry.py`.
- A post-hoc ground-truth boundary check measured the reduction in absolute
  local organ-inside/outside intensity contrast and the perturbation magnitude
  near each organ boundary. Both scores selected the best MedSA candidate on
  only two of six fixed slices. The contrast score ranked B above A on
  case0002/slice104 despite MedSA Dice 0.8879 for B versus 0.7637 for A.
  Thus even a label-informed image-boundary statistic did not identify the
  signed downstream task response in this gate. Ground truth was never used
  for perturbation generation. Report:
  `synapse_vitb_boundary_contrast_6slices.json`; scorer:
  `peft_transfer/score_boundary_contrast.py`.
- A direct diagnostic compared source-only skip-Jacobian tangent views with
  MedSA's supervised cross-entropy input gradient at three already-frozen
  attack endpoints. The source/target gradient cosines were +0.0026 on
  case0002/B, -0.0048 on case0008/A, and -0.0052 on case0008/B. Even the
  74-direction attention/MLP plus Riesz-quadrature affine reconstruction had
  target-gradient cosines +0.0029/-0.0070/-0.0032. These are *local*
  diagnostics with target weights, labels and GT-derived prompts, not a
  source-only attack or a path-integrated prediction. The source-only
  structural view family does not by itself identify the target task
  direction on these examples. A direct orthogonal projection gives the
  maximum attainable cosine in that fixed source-view span: 0.0208,
  0.0177, and 0.0184 respectively after adding Riesz spatial quadratures.
  This rules out only this fixed random-probe feature-loss view family at
  these endpoints, not every public-SAM objective. The post-hoc script is
  `peft_transfer/analyze_task_tangent_coverage.py` and the three reports are
  `case0002_slice104_B_task_tangent.json` and
  `case0008_slice98_{A,B}_task_tangent.json`.

## Theory boundary after the spatial-geometry check

The target is attackable under the same pixel budget: an existing
**diagnostic-only white-box** feature attack on the trained MedSA encoder
reaches Dice 0.1902 from clean 0.9400 on case0001/slice123, while a
target-task cross-entropy attack reaches 0.5148. The former result is in
`testset/medsa_encoder_ceiling/official.json`, the latter in
`testset/diagnostics/medsa_whitebox_slice123_official.json`. Neither is
eligible as a strict black-box attack or a fair baseline; together they
locate much of the transfer gap in the mismatch between public and adapted
encoder update directions, not in an immutable MedSA output.

The medical-invariant projector and token Gram are both functions of the
source representation. For any such source-only statistic `S(x, delta)`, two
fine-tuned decoders can agree on all clean examples yet assign opposite
signed task responses to the same source feature displacement, as in the
readout counterexample above. Consequently, maximizing `S` cannot by
itself guarantee a Dice decrease on every unknown PEFT target. A transfer
theorem needs an independently justified task-response condition, such as
a lower bound on the path-integrated alignment
`integral <grad L_target(z(t)), dz_target/dt> dt`. Clean-feature alignment
and acquisition-view stability do not identify its sign.

The practical gate remains source-only, per-image prediction of signed
downstream loss on held-out cases and both PEFT families. Choosing a layer
  or a score after inspecting target Dice would violate the strict black-box
  protocol.

## Frozen-perturbation reflection diagnostic

To test whether weak transfer is merely a wrong perturbation sign, three
existing candidates were reflected around their clean images, quantized, and
clipped to an exact 10/255 pixel budget. The construction uses no target
information; target metrics below are post-hoc only. The official slice
evaluator was run with the same MedSA prompt seed as the original candidates.

| Slice / original | MedSA clean | Original | Reflected | SAMed original | Reflected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0002/104 B | 0.9066 | 0.8879 | 0.8977 | 0.3631 | 0.9000 |
| 0004/104 A | 0.8035 | 0.7330 | 0.7981 | 0.5367 | 0.5649 |
| 0008/98 A | 0.8433 | 0.8777 | 0.8472 | 0.4839 | 0.7939 |

Every reflection was weaker on MedSA than its original direction; the
case0008 reflection was near clean rather than a strong opposite-sign
attack. The test rules out a simple antithetic sign choice on these examples,
not all possible bidirectional objectives. Artifacts and official reports are
under `testset/medical_invariant_gate/reflection_case*/`; construction is
`peft_transfer/reflect_attack.py`.

## Decision

The acquisition-invariant representation hypothesis passes the limited
source stability and clean alignment diagnostics, but fails as a sufficient
attack-transfer theory. The current evidence does not justify an online
invariant-LCB attack, a performance guarantee, or superiority over the cited
dataset-level baselines. A future source-only objective needs a validated
link to *signed downstream task response* across held-out cases and both
fine-tuning families before it is promoted to the full matrix.

Artifacts: `testset/medical_invariant_gate/synapse_vitb_fit.json`,
`synapse_vitb_alignment.json`, `synapse_vitb_prediction_5slices.json`, and
the three `case0002`-`case0004` `*_official.json` reports in that directory.
The post-hoc path reports are `case0002_slice104_task_projection.json` and
`case0004_slice104_task_projection.json`; the source-region checks are the
corresponding `*_source_anatomy_proxy.json` files.
The raw local-order report is `synapse_vitb_local_order_raw_5slices.json`;
the two blend directories contain their own `official.json` reports.
The prospective prototype report is
`case0008_slice98_blind_prototype_prediction.json`, saved before
`case0008_slice98_official.json` was generated.
The case0008 target-only mechanism files are
`case0008_slice98_layer_transfer.json` and
`case0008_slice98_task_projection.json`.
The source-token cluster check is
`synapse_vitb_source_clusters_6slices.json`.
The public-decoder control and its official result are under
`prompt_marginal_case0002_slice104/`.
The joint-prompt control is under `prompt_joint_case0002_slice104/`.
