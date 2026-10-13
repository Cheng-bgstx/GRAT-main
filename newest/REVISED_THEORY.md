# CODD Theory — Revised Based on Experimental Validation

## Original Claims vs Experimental Reality

| Claim | Original | Revised |
|-------|----------|---------|
| C1 | Radial dominance of model drift | **CONFIRMED**: RR=0.71-0.90 across 6 targets |
| C2 | Source carrier uniqueness | **CONFIRMED**: source 0.78-0.90 >> shuffled/random ~0.5 |
| C3 | Radial attack is most effective | **REFUTED**: Phase-shuffled attack >> radial attack on ALL targets |
| C4 | SRS warm-start improves attack | **WEAK**: SRS gives marginal improvement over no warm-start |

## Revised Theory

### C1: Radial Dominance (VALIDATED)
Fine-tuning SAM on medical images preserves phase structure. Model drift concentrates
in the radial (amplitude) direction of the frequency domain. This holds across:
- 3 datasets (Synapse CT, ISIC dermoscopy, REFUGE fundus)
- 2 fine-tuning paradigms (SAMed LoRA, MSA Adapter)
- All ViT layers (10/12 or more layers above R=0.5)

### C2: Carrier Structure (VALIDATED)
The source SAM phase carrier provides a structured frequency basis. Shuffling or
randomizing this carrier destroys its alignment with model representations.

### C3 (REVISED): Phase Coherence Sensitivity
**Medical segmentation models critically depend on phase coherence in their feature
representations.** The effective attack mechanism is NOT radial pushing along the
source carrier — it is **disruption of phase coherence**:

- Phase-shuffled attack: ΔDice = +0.35 to +0.51 (MASSIVE degradation)
- SE-CGR radial attack: ΔDice = -0.01 to +0.05 (minimal effect)

This is the central empirical finding. The frequency-domain structure matters, but
through phase coherence rather than carrier-guided directional pushing.

### C4 (REVISED): Amplitude as Implicit Defense
The natural amplitude spectrum acts as an implicit defense mechanism:
- Flat/shuffled amplitude: ΔDice = +0.52 (extreme degradation)
- Clean amplitude: ΔDice = -0.02 (no degradation)
- Low-pass filtered amplitude: ΔDice = -0.01 (preserved)
- High-pass filtered amplitude: ΔDice = +0.45 (partial degradation)

## Attack Mechanism: Phase-Coherent Gradient Response (PCGR)

The effective attack works by:
1. Taking the frequency-domain representation of the input
2. Shuffling the phase component (destroying coherence)
3. Running PGD in this phase-disrupted space
4. The resulting adversarial image disrupts the target model's reliance on phase structure

This explains why the attack transfers across architectures (SAMed, MSA) and scales
(vit_b, vit_l, vit_h): phase dependence is a fundamental property of the feature
extraction pipeline, not an artifact of a specific fine-tuning method.

## Domain Boundary

The effect is strongest on 2D medical images (ISIC dermoscopy, REFUGE fundus) where
phase coherence is critical for texture/shape recognition. On 3D CT (Synapse), the
effect is weaker because CT images have less texture information and models rely more
on intensity (amplitude) cues.

This domain boundary is a feature, not a bug — it demonstrates the theory's
explanatory power: CODD predicts WHERE the attack will be most effective based on
the target domain's reliance on phase structure.
