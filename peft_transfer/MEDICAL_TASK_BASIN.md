# Medical task-preserving adaptation basin: proxy-assisted historical branch

This file records the strongest proxy-assisted dual-PEFT branch. It is not the
current source-only theoretical proposal. The main mechanism study is now
[`ACRS_THEORY_AND_GATE.md`](ACRS_THEORY_AND_GATE.md). Joint Pareto remains a
valid empirical result under a public medical-proxy threat model, but its LoRA
and Adapter gradients cannot be used as evidence that a universal source-only
adaptation response has been identified.

Updated 2026-10-04. This branch replaces pseudo-task common-ascent as the proposed mechanism. It retains the established dual-PEFT encoder-feature carrier and uses **public ISIC Training labels only offline** to construct PEFT model variants. Attack generation uses the input image, base SAM and trained proxies, without test masks, target weights or target queries.

## What the local bound actually says

For one trained surrogate with PEFT coordinates `phi`, let `F` be the empirical Fisher diagonal computed from per-image supervised training gradients. Set `Q = F + lambda I` with `lambda > 0`; the damping is essential. The local uncertainty set is `U_r = {Delta: Delta^T Q Delta <= r^2}`. For an attack objective with parameter gradient `g`, the exact first-order inner minimum is

```text
inf_{Delta in U_r} [ell(phi,delta) + g^T Delta]
    = ell(phi,delta) - r sqrt(g^T Q^{-1} g).
```

Using `F` without damping and replacing its inverse with a pseudoinverse is invalid whenever `g` has a component in `null(F)`: the first-order inner minimum is then unbounded below. The empirical Fisher is positive semidefinite and need not equal the Hessian of segmentation risk. A low Fisher quadratic form does **not** prove that a finite parameter perturbation remains task preserving.

For a finite `Delta`, suppose additionally that the attack loss has a local remainder satisfying `|ell(phi+Delta,delta)-ell(phi,delta)-g^T Delta| <= M Delta^T Q Delta / 2`. Then its lower bound over `U_r` is `ell(phi,delta) - r||g||_{Q^{-1}} - Mr^2/2`. We do not know `M` for the present SAM models. The code therefore treats the quadratic expression as a local ranking surrogate and verifies clean task risk/Dice on a calibration partition; it does not call it a certified black-box guarantee.

LoRA and Adapter have different parameter charts. Separate local sets `U_L` and `U_A` can be evaluated, and the smaller observed robust score can guide attack generation. Transfer to an unseen target still requires **coverage**: its input-output behavior along the adversarial path must be approximated by at least one calibrated family. The failed common-ascent coverage result demonstrates why this cannot be assumed from the construction alone. An empirical minimum over sampled models is also not a lower bound over the full ellipsoid.

## Data isolation

The new training options `--proxy_holdout_fraction 0.1 --proxy_split_seed 2026` (SAMed) and `-proxy_holdout_fraction 0.1 -proxy_split_seed 2026` (Med-SA) deterministically split ISIC **Training** by the same image-name SHA-256 ordering. Proxy training sees 90%; Fisher uses a fixed subset of that 90%; candidate task preservation is checked on the other 10%. With this option, Med-SA's validation loader also points to the Training holdout, never ISIC Test. Fixed epoch checkpoints are used; validation-selected `best_dice_checkpoint.pth` is excluded from the prospective family. The existing historical dual-PEFT carrier may still use older Training-all proxies as a comparison anchor, but its calibration results are in-sample and must be labeled accordingly.

`build_isic_task_basin.py` estimates a per-image diagonal empirical Fisher from supervised BCE/CE negative-log-likelihood gradients over the PEFT tensors only. Dice is **not** included in the Fisher score; it is used in the separate task-preservation gate. The tool samples inverse-precision and isotropic perturbations at **equal global PEFT parameter L2 norm** and retains only paired candidates with the same seed and norm that both pass the clean BCE/CE plus soft-Dice loss and hard Dice gate. The manifest records each image identifier, checkpoint, norm, task score, accepted delta and whether test data were accessed. A variant that fails either gate is not available to the online attack. This is a concrete feasibility test of the proposed task-preserving geometry, not evidence that Fisher variants transfer better than isotropic variants.

## Decision gate

First establish that two independent LoRA and two independent Adapter fixed-epoch proxies achieve adequate clean task performance on the reserved Training calibration set. Compare the frozen strong feature carrier with equal-norm isotropic variants and Fisher variants on the same prospectively fixed ISIC ViT-B images, prompt seeds, pixel budget and official evaluators. Report both SAMed and Med-SA Dice/HD95 per image and means. The proposed basin contribution passes only if Med-SA gains materially without reversing the carrier's SAMed effect; a successful calibration gate or a large source-side robust score alone is insufficient. No full-matrix or superiority claim follows from a four-image smoke test.

The first target gate is frozen before examining its results: ISIC Test indices `200..219`, ViT-B, public SAM plus the historical LoRA-midpoint/Adapter seed-42 carrier, carrier weight `.75`, ten `2/255` steps and `L_inf <= 10/255`. The basin correction uses two new independent LoRA and two new independent Adapter seed trajectories, two accepted perturbations per trajectory, soft-min temperature `.1`, correction weight `.3`, carrier cosine floor `.8`, and clean-prediction output-drift weight `.1`. The same settings apply to equal-norm isotropic and NLL-Fisher variants. The gate compares `carrier`, `carrier + isotropic`, and `carrier + Fisher`; a later full method must be frozen before evaluating targets. The requested effect is at least a `.05` absolute Med-SA Dice reduction relative to carrier without increasing SAMed Dice on this paired set. These numbers are a decision rule, not an achieved result.

The carrier generator previously invoked ISIC/REFUGE dataset `__getitem__`, which opened test masks even though the attack did not use them. It now opens only the image paths. All 20 regenerated carrier NPY arrays are **bitwise equal** to the earlier historical `W075` arrays (`max difference = 0`), so this protocol fix does not change the carrier attack.

## Frozen 20-image gate: result

The fixed-epoch proxy checkpoints are LoRA seeds 43/44 at `epoch_29.pth` and Adapter seeds 43/44 at `epoch_10.pth`. Their reserved Training-calibration clean Dice under the basin builder's task protocol are 0.9147/0.9183 and 0.8636/0.8873, respectively. Each seed has a paired, equal-parameter-L2-norm isotropic/Fisher candidate that passes the **positive and negative** risk/Dice gate. Passing this gate only establishes local task feasibility on the calibration images.

The official ISIC ViT-B evaluation used Test indices 200--219, the same prompt seed, 10 steps, and `L_inf <= 10/255` for every candidate. The NPY and PNG audit reports zero pixels over budget for all candidates. Lower Dice means a stronger attack.

| Attack | SAMed Dice | SAMed HD95 | Med-SA Dice | Med-SA HD95 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 0.9043 | 28.93 | 0.9010 | 7.59 |
| Historical dual-PEFT carrier | **0.1867** | 236.23 | 0.6892 | 55.31 |
| Carrier + paired isotropic variants | 0.2252 | 266.59 | 0.3982 | 113.71 |
| Carrier + paired Fisher variants | 0.2247 | 225.92 | 0.3925 | 113.71 |
| Carrier + four unperturbed proxy anchors | 0.2283 | 248.69 | **0.3898** | 121.25 |

The anchor control was added **after** the frozen three-way gate had been evaluated, as a diagnostic; it is not part of the preregistered comparison. It uses the same four independent trained proxies, correction weight and output term, but no parameter perturbations. Its results are in [`isic_basin_gate20_anchor_official.json`](../testset/isic_basin_gate20_anchor_official.json); the original three-way results and paired per-case metrics are in [`isic_basin_gate20_official.json`](../testset/isic_basin_gate20_official.json).

The prespecified success rule **failed**: Fisher increases SAMed Dice by 0.0380 relative to carrier, whereas the rule required no increase. Fisher lowers Med-SA Dice by 0.2966 relative to carrier, but isotropic and anchor controls reproduce essentially all of that gain. In the paired per-case evaluation, Fisher minus isotropic Med-SA Dice is -0.0032 (standard deviation 0.0490; Fisher wins on 11/20 images). Fisher minus anchor is about -0.0020. This is not evidence that Fisher geometry improves transfer; the supported finding is that independent *trained* medical proxies add useful Med-SA attack information, with a SAMed tradeoff.

The proposed analytic ellipsoid minimum was **not** implemented as an online certified objective. The online code uses a soft minimum over finite, calibration-gated model samples and a carrier-aligned gradient correction. Its local linear bound cannot establish unknown-target coverage or the finite-perturbation remainder. The present gate does not support claiming a task-basin transfer theorem or a superior dual-target attack. Future work would need a surrogate-held-out selection rule and an untouched target evaluation before making those claims.

## Pareto-constrained follow-up (2026-10-05)

The fixed-weight basin correction was followed by a constrained first-order
direction solver. At every attack step it maximizes the aggregate Adapter
proxy ascent margin while retaining a fraction `beta` of the positive
carrier, SAMed and individual LoRA margins measured at the carrier-sign
reference direction. The solver first requests a zero lower bound for a
nonpositive reference margin, matching the original 20-image run. If no
candidate satisfies that stronger system, it relaxes only those incompatible
bounds to the carrier reference margins and returns the reference direction.
This preserves the original behavior whenever it is feasible while making the
fallback feasible by construction. The guarantee is only local and
proxy-relative: it does not certify the complete ten-step path or an unseen
victim. All 200 recorded steps in the `beta=0.8` run satisfy the original
stronger bounds, so the fallback correction does not invalidate those saved
results.

The original evaluation reported SAMed Dice/HD95 `0.23114/223.97` for
`beta=0.8` and `0.18670/236.23` for the carrier. That comparison is not valid
black-box transfer evidence. The active `SAMed_ISIC_vit_b.pth` symlink pointed
to seed-1234, learning-rate `0.005`, epoch 99 from the same training run used
to construct the epoch-79/119 midpoint attack proxy. The midpoint and victim
LoRA vectors have cosine `0.92562`; the victim is exactly that run's epoch-99
checkpoint. These numbers must be labeled same-trajectory diagnostics and
excluded from transfer claims. The symlink has been corrected to the
documented learning-rate `0.0001`, rank-16 epoch-99 checkpoint, and the
evaluator now infers LoRA rank from the checkpoint.

On the corrected rank-16 SAMed target, clean/carrier/Pareto Dice is
`0.91201/0.81815/0.71313`, with HD95 `24.42/75.48/116.57`; Pareto beats the
carrier on 16/20 images. This is a real transfer effect but remains too weak
for the intended strong dual-target result. A beta sweep does not repair it:
`0.8`, `0.9`, `0.925`, `0.94`, and `0.95` give SAMed Dice `0.71313`,
`0.72130`, `0.81124`, `0.81815`, and `0.81815`, respectively. On the default
Med-SA target the same candidates give `0.22889`, `0.27870`, `0.70651`,
`0.68915`, and `0.68915`. Thus `beta=0.8` is already the best tested tradeoff;
the SAMed gap is a proxy-family problem, not a beta-tuning problem.

The next decision gate is to freeze the Pareto configuration using surrogate
development criteria, then evaluate it on an untouched victim checkpoint and
only afterward run the full ISIC test set. The current 20-image set has already
been used for method comparison and must not serve as the final untouched
target evaluation.

A joint fixed-epoch seed-45 victim check was run without regenerating either
candidate. On the independent SAMed checkpoint, the carrier and Pareto attacks
reach Dice/HD95 `0.82986/69.30` and `0.80318/72.01`; Pareto lowers Dice on
11/20 images, but both attacks are much weaker than on the default SAMed
checkpoint. On the independent Med-SA checkpoint, paired mean Dice/HD95 moves
from `0.68439/60.60` to `0.28362/137.05`, with lower Dice on all 20 images.
This strengthens the Adapter-transfer evidence while showing that the strong
same-trajectory SAMed result was not valid transfer evidence. The seed-45 evaluation is still
on the same 20 images used during method comparison and is therefore an
independent-victim diagnostic, not an untouched final test.

## Joint LoRA-family objective gate (2026-10-10)

The rank-16 proxy diagnostic showed that the old midpoint gradient is nearly
orthogonal to the corrected SAMed victim (`-0.00494` cosine on four diagnostic
images), while independent rank-16 LoRA proxies retain positive alignment.
The Pareto solver was therefore extended from one aggregate objective to a
strict max-min objective over the LoRA and Adapter proxy families. The carrier,
SAMed-proxy and individual-LoRA lower-bound constraints are unchanged. This is
still a first-order, proxy-relative construction; the change improves empirical
family coverage but does not turn the constraints into a victim certificate.

The joint objective was selected on indices `200..203` and then frozen before
generating indices `204..219`. On the 16-image frozen confirmation set, the
corrected rank-16 SAMed Dice/HD95 changes from `0.84141/59.94` for the carrier
and `0.74476/111.27` for the original `beta=0.8` Pareto attack to
`0.55815/161.50` for the joint objective. Joint beats the original Pareto
candidate on 13/16 images and the carrier on 16/16. Med-SA Dice/HD95 is
`0.72108/41.25`, `0.24766/137.82`, and `0.31130/132.64`, respectively. The
joint objective gives back `0.06363` Dice relative to the Adapter-heavy Pareto
candidate, but remains `0.40978` below the carrier and beats the carrier on all
16 images.

Across the complete 20-image diagnostic set, clean/carrier/original-Pareto/joint
SAMed Dice is `0.91201/0.81815/0.71313/0.53547`, with HD95
`24.42/75.48/116.57/160.50`. The corresponding paired Med-SA Dice is
`0.90100/0.68750/0.23653/0.28864`, with HD95
`7.61/55.89/139.83/133.51`. Joint improves SAMed over the original Pareto
candidate on 17/20 images and preserves a large Med-SA degradation on 20/20
images relative to the carrier. It is therefore the current best supported
dual-target candidate, replacing the claim that independent SAMed transfer is
still inadequate.

This is a successful small gate, not a final benchmark. The same 20 target
images had already been examined in earlier method comparisons, and the first
four were explicitly used for joint-objective development. The next valid step
is an untouched small target gate, preferably also including a second
independent rank-16 SAMed victim, before any full 379-image or cross-dataset
evaluation.

## Frozen fresh-image gate (2026-10-11)

The joint configuration above was kept fixed and applied to four previously
unused ISIC Test indices, `220..223`. A matched carrier was regenerated with the
same midpoint LoRA proxy, seed-42 Adapter proxy, ten `2/255` steps, carrier
weight `.75`, and `L_inf <= 10/255`. Generation did not load target checkpoints
or test masks. Both NPY and PNG audits report zero pixels over budget.

On the independent seed-45 targets, clean/carrier/joint SAMed Dice is
`0.94007/0.82554/0.55694`, while paired Med-SA Dice is
`0.91840/0.63355/0.26060`. On the corrected rank-16 SAMed plus default Med-SA
targets, the corresponding values are `0.93670/0.75073/0.52360` and
`0.91965/0.66653/0.23608`. Joint beats the matched carrier on all four images
for SAMed and all four paired prompts for Med-SA in both target configurations.
The HD95 changes are consistent with the Dice degradation rather than a metric
tradeoff.

This gate removes the earlier concern that the result was restricted to images
`200..219`, but it is still not an untouched-victim experiment: both target
checkpoint configurations had been evaluated in prior work. The next decisive
step is to train or reserve a new fixed-epoch SAMed and Adapter target pair,
freeze their checkpoint selection without attack feedback, and evaluate this
unchanged candidate on a small predeclared image set. The machine-readable
record is `peft_transfer/results/fresh4_gate_20261011.json`.
