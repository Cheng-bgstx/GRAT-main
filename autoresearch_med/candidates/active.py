"""Current candidate state.

Phase 0 intentionally exposes no new online attack. H1 is diagnostic-only until
its predeclared mechanism gate passes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateState:
    name: str = "h1_diagnostic_only"
    hypothesis: str = "Path-Integrated Response Geometry"
    online_attack_enabled: bool = False
    reason: str = "mechanism_gate_not_run"


STATE = CandidateState()


def generate(*_args, **_kwargs):
    raise RuntimeError(
        "No H1 online attack is authorized. Run and pass the locked mechanism gate first."
    )

