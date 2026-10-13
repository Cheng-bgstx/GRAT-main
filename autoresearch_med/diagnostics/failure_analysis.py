"""Controlled failure taxonomy for append-only experiment records."""
from __future__ import annotations


FAILURE_CLASSES = {
    "code_error",
    "protocol_error",
    "randomness_instability",
    "optimization_failure",
    "mechanism_falsified",
    "target_shift",
    "generalization_failure",
}


def validate_failure_class(value: str) -> str:
    if value not in FAILURE_CLASSES:
        raise ValueError(f"unknown failure class: {value}")
    return value

