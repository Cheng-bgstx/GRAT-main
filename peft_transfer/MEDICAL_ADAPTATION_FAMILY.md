# Medical Adaptation Family: threat model and first mechanism gate

Updated 2026-10-04. This is a new **zero-victim-query, trained-surrogate** setting. It replaces the previous base-SAM-only research question, but does not retroactively change the threat model of earlier results.

## Dataset and checkpoint protocol

The public base SAM and labeled **proxy training data** may be used to train LoRA and Adapter trajectories. The attacked test image remains unlabeled during generation; its mask is loaded only by a separate evaluator. The victim checkpoint and victim gradients are never loaded by a generator. Proxy checkpoints must have a documented training seed, dataset, architecture, rank/bottleneck, insertion placement and epoch. Target checkpoints must be absent from the proxy family. Checkpoint filename inequality alone does not prove independent training lineage.

| Setting | Proxy data | Target evaluation | What it can establish |
| --- | --- | --- | --- |
| I: same dataset, unseen checkpoint | Synapse training slices, ISIC Training, or REFUGE Training; labels permitted for proxy training | Corresponding held-out test cases/images and separate SAMed/Med-SA checkpoint | Generalization to a checkpoint/configuration. The training *distribution* is known, and proxy/target may have seen the same public training images. |
| II: leave-one-PEFT-family-out | Train only LoRA or only Adapter proxies on a fixed public training partition | Evaluate the excluded family, then reverse direction | Whether adaptation information is genuinely shared across LoRA and Adapter. |
| III: cross-domain medical adaptation | A disjoint medical dataset with the **same segmentation semantics** and a documented acquisition/site split | A held-out target-domain dataset and checkpoint | Domain transfer as well as model transfer. It cannot be inferred by mixing unrelated abdominal CT, skin-lesion and retina label spaces. |

Current data inventory: Synapse has `data/Synapse_preprocess/train_npz` and `test_vol_h5`; ISIC has separate `ISBI2016_ISIC_Part1_Training_*` and `Test_*` files; REFUGE has `Training-400`, `Validation-400`, and `Test-400`. These permit Settings I and II. This workspace has no second abdominal-CT organ-segmentation dataset, second skin-lesion dataset or documented matched-label retinal *dataset* with which to claim Setting III across the three benchmarks. REFUGE site/partition shift is a possible narrower domain-shift study, but it must be audited against the source checkpoint's training cohort first.

For model selection, reserve a proxy-development partition within public training data or a separately designated development cohort. The final test set must not be used to choose proxy composition, attack weights, loss, number of steps or prompt policy. The present four-image gate below is a **mechanism diagnostic**; because its target results have now been inspected, those four images cannot serve as a fresh confirmation set for later tuned variants.

## Common-ascent theorem: precise scope

Let `L_k` be a *specified* differentiable proxy task loss and `g_k = grad_x L_k(x)`. Let `u` satisfy `||u||_2 <= 1`, and `m = min_k <g_k,u>`. If the unknown target gradient for the **same loss definition and prompt policy** obeys

```text
g_target = sum_k alpha_k g_k + e,
alpha_k >= 0, sum_k alpha_k = 1, ||e||_2 <= epsilon_g,
```

then `<g_target,u> >= m - epsilon_g`. This is a conditional, *local first-order* statement. It says nothing when `m <= epsilon_g`; it is not a theorem that a particular medical family covers the victim. For a finite perturbation path `x(t)`, a target-loss increase requires positive integrated directional response along that path, and therefore coverage/margin conditions along the path rather than only at its clean endpoint. Pixel clipping, sign updates and momentum can alter the direction; the implemented solver checks actual feasible per-proxy margins instead of silently transferring the L2 theorem to an `L_inf` sign step.

If test-time ground truth is unavailable, a clean-prediction pseudo-task loss `L_k^pseudo` can be computed without a test mask. The theorem above then concerns the *pseudo-task* gradients. Claiming increased target supervised loss requires another bound on `grad L_target^supervised - grad L_target^pseudo`, including label and prompt-policy mismatch. A target evaluator that uses a ground-truth-derived click prompt is a different task loss from a no-prompt surrogate. ISIC is used for the first gate because its repository Med-SA evaluator uses a fixed center click, allowing a matched prompt type without reading the test mask.

The minimum-norm convex combination of gradients is standard multiobjective geometry, not a new algorithmic contribution. Trained checkpoint ensembles and medical PEFT proxy attacks are also established ideas. A possible contribution would require evidence that a carefully defined family/distribution gives a **smaller target coverage error**, especially for an excluded PEFT family and held-out medical domain. A conditional inequality alone does not establish novelty or effectiveness.

## First task-proxy cone gate: ISIC ViT-B

Generator: [`generate_isic_task_cone.py`](generate_isic_task_cone.py); solver: [`common_ascent.py`](common_ascent.py). It loads two SAMed LoRA checkpoints at epochs 79/119 from one training run (`seed=1234`) and one independently trained Med-SA Adapter checkpoint (`seed=42`, 10 epochs). The target is the separate `checkpoints/best_weights` SAMed and Med-SA pair. Input images are ISIC Test indices `0,1,2,3`; only the image file is opened by the generator. Each proxy's clean prediction fixes its own pseudo label, and the source loss is class-balanced CE/BCE on that label. Med-SA uses the official fixed positive center point. The comparison uses ten projected `2/255` steps, exact `10/255` budget and official paired-seed evaluation.

| Candidate, four-image mean | SAMed Dice / HD95 | Med-SA Dice / HD95 |
| --- | ---: | ---: |
| Clean | `.9205 / 14.44` | `.9418 / 2.49` |
| Two LoRA proxies only | `.1581 / 173.75` | `.9413 / 2.33` |
| One Adapter proxy only | `.9137 / 15.76` | `.8074 / 16.95` |
| Fixed task-gradient weight `.375/.375/.25` | `.1947 / 146.80` | `.8053 / 19.01` |
| Minimum-norm common-ascent cone | `.1070 / 168.25` | `.8048 / 16.98` |

The cone improves SAMed Dice over the fixed task-gradient weight and weakly improves Med-SA Dice, but the latter change is only `.0005` on four images and HD95 is worse. It is **not** a strong Med-SA result. A prior encoder-feature three-model max-min gate on the same four images reached Med-SA Dice `.5658`; the historical 379-image weighted PEFT-feature attack reached `.6534`. These are different objectives, not a controlled full-set comparison, but they show the pseudo-task cone is not ready to replace the stronger feature attack. Raw records: [`official_4_family.json`](../testset/isic_task_cone_gate/official_4_family.json), [`generation.json`](../testset/isic_task_cone_gate/generation.json), and historical [`isic_trajectory_envelope_threeway/official.json`](../testset/isic_trajectory_envelope_threeway/official.json).

The earlier eight-image *feature-loss* comparison had a similar objective tradeoff: fixed LoRA/Adapter weight `.75/.25` gave SAMed/Med-SA Dice `.1864/.6969`; two-model max-min gave `.2823/.6225`. Neither dominated both victims. This is additional reason not to equate a large proxy minimum margin with overall attack superiority. [`official_8.json`](../testset/isic_dual_peft_weighted/official_8.json).

All four candidate PNG and NPY groups pass the evaluator's pixel-space audit: maximum `10.0000067/255` from float rounding and **zero** pixels above the `10.0001/255` tolerance. An initial run used PIL's different default resizing and exceeded the *evaluator's* budget; it was replaced before these results and must not be cited.

### Direct test of the coverage assumption

The diagnostic-only [`diagnose_isic_task_cone.py`](diagnose_isic_task_cone.py) loads target weights and test masks **after the attacks were generated**. On all four clean test images, the common direction has positive inner product with every proxy pseudo-task gradient. Its cosine with the *true supervised target-task gradient* is nevertheless negative:

| ISIC test image | SAMed target cosine | Med-SA target cosine | Normalized target-to-proxy-hull residual, SAMed / Med-SA |
| --- | ---: | ---: | ---: |
| `ISIC_0000003` | `-.0768` | `-.0316` | `1.248 / 1.210` |
| `ISIC_0000012` | `-.0514` | `-.0569` | `1.255 / 1.224` |
| `ISIC_0000013` | `-.1145` | `-.0281` | `1.313 / 1.197` |
| `ISIC_0000014` | `-.0902` | `-.0345` | `1.276 / 1.195` |

The target unit gradient is far outside this three-proxy unit-gradient convex hull. Thus the proposed coverage assumption is **empirically false for this pseudo-task family at these clean points**. The negative clean-point derivative does not forbid a later nonlinear attack effect; indeed some four-image Dice values drop. It does invalidate a clean-point supervised-loss guarantee from the measured proxy cone. Raw post-hoc record: [`task_coverage_4.json`](../testset/isic_task_cone_gate/task_coverage_4.json).

## No-test-label semantic-prototype check: Synapse ViT-B

The existing multi-layer prototype generator now defaults to `--label-source proxy`. It derives spatial pseudo labels from an independently trained LoRA model's **clean prediction**; test H5 labels are read only when the explicit `--label-source ground_truth --allow-ground-truth-diagnostic` pair is supplied. No target checkpoint is loaded during generation. This experiment used one trained LoRA proxy, four trained Adapter proxies, the public SAM encoder, no UMI initialization, ten steps and `10/255`. Its online objective combines pseudo-label-conditioned encoder prototypes with source SAM feature disruption; it is **not** the task-gradient common-ascent method above. The four Adapter checkpoints include three distinct seeds and one longer seed-42 trajectory. Checkpoint SHA-256 values differ from both target checkpoints, though distinct hashes alone do not certify absence of training-data overlap.

The configuration was selected on case0001/slice123, then frozen for case0032/slice72 and case0035/slice47. The latter two indices were fixed as volume midpoints without consulting masks or target outputs. The same official evaluator and prompt seed were used for each clean/attack pair. All three saved PNGs have maximum exact pixel deviation `10/255`, with zero pixels above `10.0001/255`.

**Proxy provenance correction:** the seed-42 Adapter `best_dice_checkpoint.pth` used in these three rows was selected by validation on Synapse `test_vol_h5` (`val_freq=5` in its training log; `train.py` takes `nice_test_loader` from that directory and saves the best validation Dice). Therefore these rows **do not qualify as cleanly test-isolated evidence** under the new threat model, even though no test mask was loaded inside the attack generator. The separate seed-43/44/45 `epoch_10.pth` files were saved at a fixed epoch before their final test evaluation (`val_freq=999`, `save_freq=10`) and are the eligible Adapter proxies for subsequent prospective checks. The seed-42 selected checkpoint must be excluded from further method development and final claims.

| Case / slice | SAMed clean -> attack Dice | Med-SA clean -> attack Dice | Interpretation |
| --- | ---: | ---: | --- |
| 0001 / 123, selection slice | `.9388 -> .1449` | `.9400 -> .3407` | Promising diagnostic, already used for configuration selection. |
| 0032 / 72, fixed holdout | `.8642 -> .2521` | `.9884 -> .9896` | Strong LoRA attack, **no Adapter damage**. |
| 0035 / 47, fixed holdout | `.8750 -> .7500` | `1.0000 -> 1.0000` | Proxy predicts only background; prototype CE gradients are exactly zero and Med-SA is unchanged. |

Raw records: [`0001 generation`](../testset/synapse_proxy_label_gate/generation.json), [`0001 official`](../testset/synapse_proxy_label_gate/official_123.json), [`0032 generation`](../testset/synapse_proxy_label_holdout_0032/generation.json), [`0032 official`](../testset/synapse_proxy_label_holdout_0032/official.json), [`0035 generation`](../testset/synapse_proxy_label_holdout_0035/generation.json), and [`0035 official`](../testset/synapse_proxy_label_holdout_0035/official.json). Single-slice per-class means, especially the foreground-sparse 0035 slice, are not dataset mDSC estimates. The 0032 result also cautions against claiming that proxy semantic prototypes plus this source feature term reliably damage unseen Adapter models.

### Test-isolated checkpoint check

The first correction uses only the LoRA fixed-epoch checkpoint and the three Adapter `epoch_10.pth` checkpoints. Their logs show training on `train_npz`, `val_freq=999`, and fixed-epoch checkpoint saving before the final test evaluation. With unchanged attack settings and no reference bank, the prechosen midpoint case0036/slice92 has a background-only proxy pseudo label. Official SAMed Dice is `.9916 -> .4001`; Med-SA is `.9950 -> .9933`. Thus the clean checkpoint protocol alone does not solve Adapter transfer. [`generation`](../testset/synapse_clean_proxy_holdout_0036/generation.json), [`official`](../testset/synapse_clean_proxy_holdout_0036/official.json).

To test the one-class degeneration without test masks, a separate prospective case0038/slice50 attack added one public **training** slice (`train_npz/case0006_slice096.npz`) to the class-prototype bank. The reference was chosen from the first 300 sorted training slices by maximum number of training-label classes. It is not a target or test slice. The proxy bank then contained classes `0,2,3,4,5,6,7,8`, and its optimization loss was nonzero. Official SAMed Dice was `.8623 -> .5000`; Med-SA `.9602 -> .9273`. The saved PNG is exactly within `10/255`, with zero pixels over the tolerance. This modest Adapter drop is insufficient for the requested strong result, and the two cases differ so it is **not** an ablation of the reference bank. [`generation`](../testset/synapse_clean_proxy_ref_holdout_0038/generation.json), [`official`](../testset/synapse_clean_proxy_ref_holdout_0038/official.json).

## Decision

Do not run the full dataset/backbone matrix or claim that common-ascent or the pseudo-label prototype method has surpassed the baselines. The first required improvement is a proxy **task definition and family construction** that prospectively reduces held-out target gradient/path-response error. More proxy checkpoints alone do not prove this: LoRA-only -> Adapter and Adapter-only -> LoRA tests are weak, the present pseudo-task gradients miss the supervised target direction, and the clean-checkpoint Synapse holdouts show little Med-SA damage. Any next family should be evaluated with an untouched development split, a genuinely independent LoRA seed, fixed-epoch checkpoints selected without test labels, and a prospective slice/case selection rule. The strong historical dual-PEFT feature attack remains the empirical reference for this revised threat model, subject to checkpoint-provenance audit before final claims.
