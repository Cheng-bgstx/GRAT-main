# Novelty Matrix

This is a conservative audit register, not a claim of novelty. Bibliographic details must be verified against primary papers before publication.

| Work/line | Existing mechanism | Overlap risk | Required distinction |
| --- | --- | --- | --- |
| UMI-GRAT / GR-only | Gradient-noise feature robustness and optional universal perturbation | High if GR loss is reused | GR-only remains external baseline; candidate loss must be independent |
| DTA | Transfer-oriented disruption across model structure/features | High for generic transferable feature objectives | Show a distinct pathwise mechanism and incremental evidence |
| FSGR | Feature/gradient robustness for transfer | High for gradient smoothing or robustness claims | H1 must not reduce to averaging noisy gradients |
| CORTA | Parameter sensitivity and representation drift | High for PEFT sensitivity analysis | Source path statistic must predict real adaptation outcomes beyond ACRS |
| SETA / PEFT simulation | Simulated target adaptation structure | High for random LoRA/Adapter probes | No proxy target or simulated medical adapter in online generation |
| TPA and task-agnostic attacks | Task-independent foundation-model transfer | Medium/high | Demonstrate medical adaptation-specific explanatory evidence |
| SSA and spectral transfer attacks | Spectrum transforms to simulate model differences | High for frequency selection | Frequency stays diagnostic; no spectral attack mask without new evidence |
| SAM region-level spectrum attack | Region/spectrum transformations for SAM | High for spectral or regional perturbations | H1 operates on path response geometry, not frequency perturbation |
| Important-frequency optimization | Optimizes selected frequency coefficients | High for fixed bands | No fixed low/mid/high band attack |
| Phase-based medical DG/robustness | Phase constraints for medical generalization | Different threat model but conceptual overlap | Do not infer black-box adaptation transfer from DG results |
| Integrated gradients / path methods | Path integration of gradients or attributions | Fundamental overlap | Novelty cannot be path integration alone; needs a new, validated transfer relation |
| PGN and gradient averaging | Samples path/neighborhood gradients for transfer | High | H1 diagnostic must be distinguished from an optimizer that merely averages gradients |

## Current verdict

H1 is a research hypothesis, not yet an original method. A publishable contribution requires a verified relation between source-computable path geometry, preservation under unseen medical adaptation, and downstream segmentation degradation.

