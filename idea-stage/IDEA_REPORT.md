# Research Idea Report

**Direction**: Transfer-based Black-box Adversarial Attacks on SAM and Medical Downstream Models via Frequency-Domain Amplitude-Phase Manipulation — Integrated Offline/Online Design

**Generated**: 2026-06-21

**Ideas evaluated**: 11 generated → 7 survived filtering → 4 recommended → 0 piloted (Codex unavailable)

---

## Landscape Summary

The field of adversarial attacks on vision foundation models has seen rapid growth in 2024–2025, with SAM (Segment Anything Model) emerging as a critical target given its widespread use in medical imaging and other safety-critical domains. The dominant paradigm for attacking black-box fine-tuned SAM variants is **transfer-based attack**: craft perturbations on the open-source white-box SAM and hope they fool downstream models.

Our prior work **GRAT/UMI-GRAT** (NeurIPS 2024) introduced two key innovations: (1) Universal Meta-Initialization (UMI) extracts intrinsic vulnerability from the open-source SAM as a perturbation prior, and (2) Gradient Robust Loss (GRAT) addresses gradient deviation between surrogate and target models. However, GRAT operates entirely in the spatial/feature domain and does not exploit the frequency structure of ViT features.

A parallel line of work has emerged around **frequency-domain understanding** of adversarial examples. DAT (NeurIPS 2024) established that phase spectra carry semantic information and are the primary target of adversarial attacks, while amplitude spectra carry style/texture. SSA/ Spectrum Simulation Attack (ECCV 2022) introduced the concept of spectrum saliency — different models attend to different frequency components, and simulating diverse frequency responses boosts transferability. Centralized Perturbation (AAAI 2024) showed that DNNs share sensitivity to certain frequency coefficients, making frequency-centralized perturbations more transferable. FreqPure (2025) revealed that adversarial damage increases monotonically with frequency.

Our current SE-CGR method bridges these two lines: it operates on intermediate ViT features in the frequency domain, constructing a perturbation whose amplitude is modulated by the clean feature's spectrum (`Σ_freq = λ·A + γ`) while locking the phase to the clean features (`exp(1j·φ_clean)`). This is complemented by an offline meta-learning phase that extracts a "common perturbation" from a proxy dataset.

**Three fundamental gaps remain unsolved**:
1. **Phase is locked, not exploited**: We preserve the clean phase, but DAT shows phase is exactly where adversarial information lives. The question is how to perturb phase in a *structured* way that transfers.
2. **No theoretical framework for frequency-domain transfer**: λ is heuristic, Σ_freq is linear-in-A, and there's no principled reason why this should work — or when it fails.
3. **Offline-online disconnect**: The offline common perturbation extraction uses a single surrogate and doesn't capture shared frequency characteristics across models, making it a weak initialization rather than a true "vulnerability direction."

---

## Recommended Ideas (ranked)

### Idea 1: Spectral Causal Decomposition of Adversarial Transferability (SCDAT)
**Type**: Diagnostic + Theoretical

- **Core hypothesis**: Amplitude and phase perturbations transfer through *different causal mechanisms* — amplitude perturbations transfer via shared low-level feature statistics (e.g., contrast, texture energy) that are model-agnostic, while phase perturbations transfer only when they happen to align with model-specific structural priors. The total transferability decomposes as: `Transfer(ΔA, Δφ) = α·T_amp(ΔA) + β·T_phase(Δφ) + γ·T_interact(ΔA, Δφ)`, where the interaction term is hypothesized to be negative (phase perturbation reduces amplitude-driven transfer).

- **Method sketch**:
  1. **Causal mediation analysis in frequency domain**: Design controlled interventions where we independently vary ΔA and Δφ at specific frequency bands and measure the causal effect on transfer success rate to different target models.
  2. **Frequency-band-wise decomposition**: Split the spectrum into K bands (low/mid/high frequency). For each band, separate the amplitude-only, phase-only, and joint perturbation effects.
  3. **Key diagnostic experiment**: Compare `Transfer(same_ΔA, Δφ=0)` vs `Transfer(same_ΔA, Δφ=perturbed)` across models — if the drop is large for dissimilar models, it means phase perturbation is model-specific and hurts transfer.
  4. **Derive a predictive model** of transferability given only surrogate model statistics — this becomes the theoretical foundation for all subsequent attack methods.

- **Minimum viable experiment**: Take 3 SAM variants (base SAM, SAMed-LoRA, Medical-SAM-Adapter) on one dataset (Synapse). For N=50 images, run 100 controlled frequency interventions each. Measure transfer success rate. **Estimate: 4 GPU-hours** (mostly inference, no training).

- **Expected outcome**:
  - *Positive*: Clear decomposition showing amplitude transfers universally, phase is model-specific. The interaction term is significantly non-zero → publish as "Understanding Adversarial Transfer via Spectral Causality."
  - *Negative*: No clear decomposition, all components transfer equally → suggests the frequency-domain framing is not the right abstraction → still publishable as a negative diagnostic.

- **Novelty**: 9/10 — No existing work does causal mediation analysis in the frequency domain for adversarial transfer. Closest: DAT observes correlation (phase carries semantics), but doesn't establish causation. The causal decomposition framework is entirely new.

- **Feasibility**: 4 GPU-hours pilot, 20 GPU-hours full study. Requires no new training, only inference with controlled perturbations. Implementation complexity: LOW (modify existing FFT perturbation code to separate amp/phase interventions).

- **Risk**: LOW — the diagnostic question matters regardless of answer. The methodology is rigorous even if the decomposition turns out to be trivial.

- **Contribution type**: Diagnostic finding + theoretical framework

- **Reviewer's likely objection**: "This is a measurement study, not a new method." **Counter**: The causal framework enables *principled* design of subsequent methods — without it, all frequency-domain attacks are heuristic. Plus, the finding that phase perturbation hurts transfer (if confirmed) overturns the naive "perturb everything" approach.

- **Why we should do this first**: It answers the most fundamental question blocking progress — WHY do frequency perturbations transfer? — and every subsequent idea in this report benefits from its findings. It's also the lowest-risk, fastest-to-execute idea.

---

### Idea 2: Optimal Spectral Transport for Amplitude-Phase Budget Allocation (OST-AP)
**Type**: New Method + Theoretical

- **Core hypothesis**: The optimal per-frequency, per-component (amplitude vs. phase) perturbation budget can be derived by framing adversarial transfer as an **Optimal Transport problem**: transport the clean feature's spectral distribution to a maximally adversarial distribution, with *separate transport costs* for amplitude and phase dimensions that encode their different roles in transferability.

- **Method sketch**:
  1. **Spectral distribution representation**: For each intermediate ViT layer, represent the feature as a 2D spectral distribution: `P_clean(ω) = (A(ω), φ(ω))` for each frequency ω.
  2. **Cost matrix design**: Define `c_amp(ω₁, ω₂)` = cost of perturbing amplitude at ω₁ to match ω₂, and `c_phase(ω₁, ω₂)` = a *higher* cost for phase perturbation (encoding that phase carries structure). The ratio `c_phase / c_amp` is a hyperparameter that can be learned per frequency band.
  3. **Entropic OT (Sinkhorn) for budget allocation**: Given total perturbation budget ε, solve: `min_{π} Σ_{ω,ω'} π(ω,ω') · [c_amp(ΔA) + c_phase(Δφ)] + ε·H(π)` where π is the transport plan and H is entropy. The solution gives the *optimal allocation of ε across frequencies and across amp vs. phase*.
  4. **Online adaptation**: The transport plan is solved per-sample with differentiable Sinkhorn, making this a plug-in replacement for the current heuristic λ.

- **Minimum viable experiment**: Replace the λ computation in SE-CGR with the OT-derived budget allocation. Compare transfer dice on Synapse SAMed vs. baseline CGR. **Estimate: 6 GPU-hours** (OT computation adds ~10% overhead to attack iterations).

- **Expected outcome**:
  - *Positive*: OT allocation consistently outperforms heuristic λ by >5% in transfer dice, especially on hard cases (ISIC, REFUGE).
  - *Negative*: OT allocation performs similarly to heuristic → suggests the heuristic λ, despite lacking theory, already captures the right scaling.

- **Novelty**: 8/10 — OT has been used for adversarial robustness (distributional robustness) but NOT for per-frequency perturbation budget allocation in attacks. The "separate transport costs for amplitude vs. phase" is a novel conceptual contribution.

- **Feasibility**: 6 GPU-hours pilot, ~30 GPU-hours full study (multi-model, multi-dataset). OT solver (Sinkhorn) is differentiable and fast (O(N² log N) with N frequency bins). Implementation complexity: MEDIUM (need to implement differentiable Sinkhorn in PyTorch).

- **Risk**: MEDIUM — the theoretical framework is sound but the improvement over heuristic λ may be marginal if the heuristic already approximates the OT solution.

- **Contribution type**: New method grounded in optimal transport theory

- **Reviewer's likely objection**: "OT adds computational overhead without clear benefit if the cost ratio c_phase/c_amp is just another hyperparameter to tune." **Counter**: The cost ratio is *learned* from data (via gradient descent on a validation set of surrogate models), not hand-tuned. The learning process itself reveals which frequencies/components matter for transfer.

- **Why we should do this**: If Idea 1 tells us WHAT to perturb, Idea 2 tells us HOW MUCH to perturb at each frequency. The two together form a complete theoretical+algorithmic contribution.

---

### Idea 3: Phase-Coherent Spectral Augmentation for Offline Shared Vulnerability Extraction (PCSA)
**Type**: New Method

- **Core hypothesis**: The offline common perturbation fails because it overfits to the surrogate model's specific frequency response — it captures what makes *this* surrogate vulnerable, not what makes *all* SAM-derived models vulnerable. Applying **phase-coherent spectral augmentations** during meta-learning — transformations that modify amplitude while preserving phase structure — forces the learned perturbation to be invariant to amplitude variations across models, extracting the truly shared phase-based vulnerability.

- **Method sketch**:
  1. **Phase-coherent augmentations**: During each offline meta-learning iteration, before computing the attack loss, apply random spectral transformations to the features:
     - **Amplitude scaling**: `A'(ω) = A(ω) · (1 + ε_a)`, ε_a ~ N(0, σ²) — simulates different models having different amplitude responses
     - **Amplitude band-stop filtering**: Zero out random frequency bands — simulates model-specific frequency blindness
     - **Phase jitter**: `φ'(ω) = φ(ω) + ε_φ`, ε_φ ~ N(0, σ_φ²)` with σ_φ small — simulates minor model-specific phase shifts
  2. **Consistency regularization**: The common perturbation's effect (measured as the feature difference it induces) should be *invariant* to these augmentations: `L_consist = KL(p(y|F + P, aug₁) || p(y|F + P, aug₂))`
  3. **Multi-surrogate ensemble**: Use 2-3 different SAM checkpoints (different seeds, LoRA ranks) as surrogate ensemble during offline phase.
  4. **Final loss**: `L_total = L_attack + λ_consist · L_consist + λ_diverse · L_diverse` where L_diverse encourages the perturbation to work across all surrogates.

- **Minimum viable experiment**: Offline phase with and without phase-coherent augmentation. Compare the quality of the extracted common perturbation by measuring its transfer improvement on held-out models. **Estimate: 4 GPU-hours** offine + 2 GPU-hours online eval.

- **Expected outcome**:
  - *Positive*: Common perturbation extracted with PCSA provides >10% better initialization for online attacks compared to baseline offline method.
  - *Negative*: Augmentation doesn't help → suggests the offline phase's problem is not overfitting but something more fundamental (e.g., the very concept of a "shared" perturbation may be flawed for frequency-domain approaches).

- **Novelty**: 7/10 — Phase-coherent augmentation for adversarial robustness has been explored (DAT uses amplitude mix-up), but using it for *offline vulnerability extraction* in *transfer attacks* is novel. The specific combination of spectral augmentations + consistency regularization for this purpose is new.

- **Feasibility**: 6 GPU-hours pilot, ~24 GPU-hours full study. Implementation complexity: LOW-MEDIUM (augment existing meta-learning loop with spectral transforms).

- **Risk**: MEDIUM — the augmentation space is large and tuning the augmentation hyperparameters (σ_a, σ_φ, band-stop widths) may require significant experimentation. The multi-surrogate ensemble adds compute overhead.

- **Contribution type**: New method for offline vulnerability extraction

- **Reviewer's likely objection**: "Phase-coherent augmentation is just data augmentation in frequency domain — what's the novelty?" **Counter**: The novelty is the *purpose* — using spectral augmentation to enforce *invariance* during vulnerability extraction, which is conceptually different from using it to improve robustness (DAT) or simulate diverse surrogates (SSA).

- **Why we should do this**: This directly addresses the weakest part of our current pipeline (offline phase). Combined with Ideas 1 and 2, it forms a complete offline→online pipeline with theoretical grounding.

---

### Idea 4: Meta-Learned Spectral Prior via Frequency VAE for Online-Offline Bridge (F-VAE Bridge)
**Type**: New Method

- **Core hypothesis**: The offline phase should learn a **distribution over spectral perturbation patterns** (a frequency prior) rather than a single point estimate. This distribution captures the shared vulnerability structure across models. During online attack, the prior is sampled and refined in a few steps — combining the efficiency of offline preparation with the adaptability of online optimization.

- **Method sketch**:
  1. **Offline**: Train a Frequency VAE on the spectral gradients from multiple surrogate models:
     - Encoder: `q(z | ∇_ω L_surrogate(F))` — maps spectral gradient to latent z  
     - Decoder: `p(ΔF_spectral | z)` — generates spectral perturbation given z
     - Latent z captures the "shared vulnerability manifold"
  2. **Online**: For each test sample:
     - Sample K latent codes from the prior: `z_k ~ p(z)`
     - Decode to K candidate spectral perturbations: `ΔF_k = Decoder(z_k)`
     - Select the one that maximizes attack loss on the surrogate: `k* = argmax_k L_attack(F + ΔF_k)`
     - Refine with ≤3 steps of gradient-based optimization
  3. **Key insight**: The VAE latent space naturally separates shared (low-variance across surrogates) from model-specific (high-variance) perturbation patterns — the KL divergence in the VAE bottleneck automatically penalizes model-specific patterns.

- **Minimum viable experiment**: Train F-VAE on 2 surrogate models (base SAM + one LoRA variant), test on 3 held-out variants. Compare to baseline offline+online method. **Estimate: 3 GPU-hours** for VAE training + 2 GPU-hours for online eval = 5 GPU-hours total.

- **Expected outcome**:
  - *Positive*: F-VAE prior + few-step refinement achieves comparable or better transfer than full iterative online attack with 10+ steps, while being 3-5x faster.
  - *Negative*: F-VAE prior doesn't help — latent space doesn't capture meaningful structure → suggests the "shared vulnerability manifold" hypothesis is wrong.

- **Novelty**: 8/10 — Using generative models (VAE) in frequency domain for adversarial perturbation prior is unexplored. The FNO-based approach (Idea 11) is a related but distinct direction. The "learn distribution, sample-and-refine" pipeline is novel for adversarial attacks.

- **Feasibility**: 5 GPU-hours pilot, ~20 GPU-hours full. VAE training on spectral gradients is computationally light (small latent dim, moderate feature dim). Implementation complexity: MEDIUM (need to design VAE architecture for complex-valued spectral data).

- **Risk**: MEDIUM — VAE training on spectral gradients may be unstable. The "shared vulnerability manifold" may not exist as hypothesized.

- **Contribution type**: New method with generative modeling + adversarial attacks

- **Reviewer's likely objection**: "This adds a VAE training stage — is the computational overhead justified by the improvement?" **Counter**: The VAE is trained once offline and amortized across all online attacks. For applications requiring many attacks (e.g., adversarial training, robustness evaluation), the amortization provides net savings.

- **Why we should do this**: This idea bridges offline and online in the most principled way — learning a distribution rather than a point, then adapting. It's complementary to Ideas 2 and 3.

---

### Idea 5: Gradient Spectrum Alignment (GSA) — Direct Offline-Online Transfer
**Type**: New Method

- **Core hypothesis**: The offline phase should not extract a perturbation, but instead extract the **spectral gradient direction** — the direction in frequency space along which the loss increases most steeply *across all surrogate models*. Online, this direction is used as a prior for per-sample gradient computation, aligning individual sample gradients with the shared direction.

- **Method sketch**:
  1. **Offline**: For each training sample in the proxy dataset, compute the spectral gradient `g_i(ω) = ∇_{F(ω)} L_attack` on each surrogate model `i`. Find the direction `d_shared(ω)` that maximizes the average cosine similarity with all `g_i(ω)`: `d_shared = argmax_{||d||=1} Σ_i cos(d, g_i)`. This is solved via eigendecomposition of the gradient covariance matrix (effectively PCA on spectral gradients).
  2. **Online**: For each test sample, compute its spectral gradient `g_test(ω)` on the white-box surrogate. Project it onto `d_shared`: `g_aligned = α·g_test + (1-α)·(g_test · d_shared)·d_shared`. Use `g_aligned` for the perturbation update.
  3. **The α parameter** controls the trade-off between sample-specific and shared-direction following.

- **Minimum viable experiment**: Compute d_shared on Synapse proxy dataset with 2 surrogates. Test online attack with GSA vs baseline on 3 Synapse test volumes. **Estimate: 3 GPU-hours**.

- **Expected outcome**:
  - *Positive*: GSA-aligned gradients produce >5% better transfer than baseline (non-aligned) gradients, especially on hard cases.
  - *Negative*: Alignment doesn't help or hurts → suggests the shared direction extracted from limited surrogates doesn't generalize.

- **Novelty**: 7/10 — Gradient alignment is a known concept (e.g., in meta-learning, MAML), but applying it to spectral gradients for adversarial transfer is novel. The PCA-on-spectral-gradients technique is new.

- **Feasibility**: 3 GPU-hours pilot, ~15 GPU-hours full. Implementation complexity: LOW (PCA + gradient projection is simple).

- **Risk**: LOW-MEDIUM — the method is simple and well-motivated. The main risk is that the shared direction may not exist or may be too weak to help.

- **Contribution type**: New method

---

### Idea 6: Phase Perturbation via Lie Group Actions on the Complex Circle (PhaseLie)
**Type**: New Method + Theoretical

- **Core hypothesis**: Phase should NOT be locked — instead, it should be perturbed via **structured Lie group actions** (rotations on the complex circle) that preserve the topological structure of the phase spectrum while introducing adversarial signal. Specifically, the phase perturbation at frequency ω should be an element of SO(2) applied to the complex feature, parameterized by a smooth vector field over frequency.

- **Method sketch**:
  1. **Parameterize phase perturbation**: Instead of `exp(1j · φ_clean)`, use `exp(1j · (φ_clean + δφ(ω)))` where δφ(ω) is a smooth function over frequency — parameterized as a small neural network or a low-order 2D polynomial.
  2. **Smoothness constraint**: Add `L_smooth = Σ_ω ||∇_ω δφ(ω)||²` to penalize non-smooth phase perturbations. Smooth phase perturbations preserve image structure (no high-frequency artifacts) while introducing adversarial signal in mid-frequency bands.
  3. **Adaptive phase budget**: The total phase perturbation budget `||δφ||² ≤ ε_φ` is allocated across frequencies using the same OT framework from Idea 2.
  4. **Joint optimization**: Simultaneously optimize δA(ω) (amplitude perturbation, as before) and δφ(ω) (phase perturbation, new) with the OT-derived per-component budgets.

- **Minimum viable experiment**: Add smooth phase perturbation (2D polynomial parameterization) to CGR. Compare transfer dice with and without phase perturbation on Synapse. **Estimate: 4 GPU-hours**.

- **Expected outcome**:
  - *Positive*: Adding smooth phase perturbation improves transfer by >5%, especially on models where baseline fails (ISIC, REFUGE).
  - *Negative*: Phase perturbation hurts transfer → confirms Idea 1's hypothesis that phase is model-specific. Still publishable as a controlled experiment validating the causal decomposition.

- **Novelty**: 9/10 — Lie group parameterization of phase perturbation for adversarial attacks is completely novel. All existing work either locks phase or perturbs it randomly (which destroys structure).

- **Feasibility**: 4 GPU-hours pilot, ~20 GPU-hours full. Implementation complexity: MEDIUM (need to implement smooth phase field parameterization).

- **Risk**: MEDIUM-HIGH — the theory is elegant but whether smooth phase perturbation actually helps transfer is unknown. This is a genuine exploration.

- **Contribution type**: New method with Lie group theory

- **Reviewer's likely objection**: "Why Lie groups? This seems like over-mathematization of a simple idea." **Counter**: The Lie group structure (SO(2) on each frequency) naturally enforces that phase perturbations are structure-preserving — you can't accidentally introduce phase discontinuities that create spatial artifacts. The smoothness of the vector field is the key property, and Lie groups give the right language to describe it.

---

### Idea 7: Frequency-Domain Contrastive Proxy Dataset Construction (FreqCon)
**Type**: New Method + Diagnostic

- **Core hypothesis**: The proxy dataset used for offline vulnerability extraction should be **constructed** to maximize the diversity of frequency responses across surrogate models, not randomly sampled. A contrastive criterion — select samples where surrogate models DISAGREE most in their spectral responses — will produce a proxy dataset that captures the "boundary" of shared vulnerability.

- **Method sketch**:
  1. **Spectral response profiling**: For each candidate sample in a large unlabeled pool, compute the spectral gradient on each surrogate model: `g_s(ω)` for surrogate s.
  2. **Contrastive selection score**: Score each sample by `D(sample) = Σ_{s≠t} ||g_s - g_t||² / (||g_s||·||g_t||)` — high score means surrogates disagree on which frequencies matter.
  3. **Diverse subset selection**: Select K samples maximizing both individual D(sample) and pairwise diversity of selected samples (maximizing minimum distance in gradient space).
  4. **Offline extraction** on this curated proxy set, using any of the methods from Ideas 3-5.

- **Minimum viable experiment**: Profile 500 unlabeled images from Synapse. Select top 50 by contrastive score. Extract common perturbation. Compare transfer to random 50-sample baseline. **Estimate: 2 GPU-hours** profiling + 2 GPU-hours extraction = 4 GPU-hours.

- **Expected outcome**:
  - *Positive*: Contrastively selected proxy dataset produces >8% better offline perturbation than random selection.
  - *Negative*: No improvement → suggests the surrogate disagreement signal is too noisy or not predictive of transfer benefit.

- **Novelty**: 7/10 — Active/contrastive dataset selection is known, but applying it to proxy dataset construction for adversarial transfer is novel. The spectral gradient disagreement criterion is new.

- **Feasibility**: 4 GPU-hours pilot, ~15 GPU-hours full. Implementation complexity: LOW (just change which samples go into the offline phase).

- **Risk**: LOW — even if the contrastive criterion doesn't help, the experiment is cheap and the negative result is informative.

- **Contribution type**: Diagnostic + method

---

## Eliminated Ideas (for reference)

| # | Idea | Reason eliminated |
|---|------|-------------------|
| 8 | **Fourier Neural Operator for Adversarial Perturbation** — Train an FNO to directly map clean features → adversarial perturbation in frequency space | Requires training a new model architecture from scratch; high implementation complexity; unclear benefit over simpler VAE (Idea 4). Estimated >1 week GPU. |
| 9 | **Adversarial Amplitude Swap (AAS) for Attack** — Apply AAS but in reverse: swap adversarial amplitude onto clean phase to create strong attacks | Already explored in AAS paper for defense; reversing it for attack is an obvious extension (low novelty). |
| 10 | **Multi-Task Meta-Learning for Cross-Dataset Transfer** — Meta-learn across Synapse/ISIC/REFUGE simultaneously to extract dataset-invariant vulnerability | Requires training on all 3 datasets simultaneously → >8 GPU-hours for pilot, exceeding budget. Also, the datasets have different image characteristics (CT vs. dermoscopy vs. fundus) making joint meta-learning challenging without domain adaptation. |
| 11 | **Ensemble Surrogate Attack with Spectral Diversity** — Attack using an ensemble of surrogate models with diverse spectral responses | A natural extension of SSA. Novelty is limited — ensemble attacks are well-studied. The incremental contribution doesn't justify a standalone paper. |

---

## Suggested Execution Order

Based on risk, dependency, and potential impact, I recommend the following execution order:

### Phase 1: Foundation (Weeks 1-3) — Idea 1 + Idea 5
Start with the two lowest-risk ideas that establish the theoretical foundation:
1. **Idea 1 (SCDAT)**: The causal decomposition study. This should be done FIRST because every other idea benefits from knowing whether amplitude and phase perturbations transfer through different mechanisms. It's also the fastest to execute.
2. **Idea 5 (GSA)**: Gradient Spectrum Alignment. Low implementation complexity, cheap to run. If it works, it immediately improves our attack. If it doesn't, the negative result is still valuable as a diagnostic.

### Phase 2: Core Method (Weeks 4-8) — Idea 2 + Idea 6
Build the algorithmic contribution on top of the theoretical foundation:
3. **Idea 2 (OST-AP)**: Optimal Spectral Transport. This is the main methodological contribution. If Idea 1 confirms that amplitude and phase have different transfer mechanisms, the OT framework provides the principled way to allocate budget.
4. **Idea 6 (PhaseLie)**: Phase perturbation via Lie groups. Execute in parallel with Idea 2. The results of Idea 1 will inform whether this is worth pursuing — if phase is confirmed to be model-specific, PhaseLie may be less promising.

### Phase 3: Integration & Polish (Weeks 9-12) — Idea 3 + Idea 4
Integrate offline and online into a complete pipeline:
5. **Idea 3 (PCSA)**: Phase-coherent spectral augmentation for offline extraction.
6. **Idea 4 (F-VAE Bridge)**: Meta-learned spectral prior. Can be executed in parallel with Idea 3.

### Phase 4: Bonus (if time permits) — Idea 7
7. **Idea 7 (FreqCon)**: Contrastive proxy dataset construction. This is a nice-to-have that could further boost offline extraction quality but is not essential for the core contribution.

---

## Proposed Paper Architecture

The recommended ideas 1, 2, 3, and 6 could form a single strong paper:

**Title (tentative)**: *"Beyond Heuristic Frequency Attacks: A Principled Framework for Transferable Adversarial Perturbations via Spectral Causal Decomposition and Optimal Transport"*

**Structure**:
- **Section 3**: Spectral Causal Decomposition (Idea 1) — establishes WHY the problem is structured as it is
- **Section 4**: Optimal Spectral Transport (Idea 2) — provides the principled budget allocation
- **Section 5**: Phase-Coherent Augmentation (Idea 3) — addresses the offline phase
- **Section 6**: PhaseLie (Idea 6) — addresses the online phase, completing the pipeline
- **Experiments**: Synapse (CT), ISIC (dermoscopy), REFUGE (fundus) — 3 datasets × 3 model variants = 9 transfer scenarios

Alternatively, Ideas 1+2 could be a focused NeurIPS paper, and Ideas 3+4+6 could be a follow-up ICCV/ICLR paper with the integrated offline-online pipeline.

---

## Reviewer Anticipated Criticisms & Pre-buttals

| Criticism | Pre-buttal |
|-----------|-----------|
| "Why frequency domain? What's wrong with spatial domain attacks?" | Our causal decomposition (Idea 1) answers this: it shows that frequency domain enables DECOUPLED control over amplitude (shared, transferable) and phase (model-specific) perturbation channels, which is impossible in spatial domain. |
| "The OT / Lie group / VAE framework seems over-engineered." | Each component addresses a specific, identified failure mode of the current heuristic approach. We include thorough ablation studies showing each component's contribution. |
| "Results are only on medical imaging — does this generalize?" | Medical imaging is a harder test case (fine-grained structures, class imbalance, limited training data). If it works here, it likely works on natural images. We include one natural image dataset as validation. |
| "The offline phase requires access to multiple surrogate checkpoints — isn't this a stronger threat model?" | We show results with 1, 2, and 3 surrogates. The method works with 1 surrogate (same as GRAT) and improves with more — the user chooses the threat model strength. |

---

## Next Steps

- [ ] **Execute Idea 1 pilot** (SCDAT causal decomposition) — 4 GPU-hours, Synapse only
- [ ] **Execute Idea 5 pilot** (GSA gradient alignment) — 3 GPU-hours, Synapse only  
- [ ] Based on Idea 1 results, decide whether to prioritize phase perturbation (Idea 6) or amplitude-only methods (Idea 2)
- [ ] Write up Idea 1 as a standalone short paper if results are clear (could be a fast ICLR workshop paper)
- [ ] Once theoretical foundation is established, implement full pipeline (Ideas 2+3) for a top-tier submission
- [ ] Run `/novelty-check` on the final paper concept before committing to full experiments
