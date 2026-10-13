"""L-infinity and image-range audit for generated examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def audit_arrays(clean: np.ndarray, adversarial: np.ndarray, epsilon_255: float) -> dict:
    clean = np.asarray(clean, dtype=np.float32)
    adversarial = np.asarray(adversarial, dtype=np.float32)
    if clean.shape != adversarial.shape:
        raise ValueError(f"shape mismatch: {clean.shape} != {adversarial.shape}")
    difference = np.abs(adversarial - clean)
    linf = float(difference.max(initial=0.0))
    return {
        "shape": list(clean.shape),
        "linf_255": linf,
        "epsilon_255": float(epsilon_255),
        "budget_pass": linf <= float(epsilon_255) + 1e-6,
        "range_pass": bool(adversarial.min() >= 0 and adversarial.max() <= 255),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--adversarial", required=True)
    parser.add_argument("--epsilon", type=float, default=10.0)
    args = parser.parse_args()
    clean = np.asarray(Image.open(args.clean).convert("RGB"))
    adversarial = np.asarray(Image.open(args.adversarial).convert("RGB"))
    result = audit_arrays(clean, adversarial, args.epsilon)
    print(json.dumps(result, indent=2))
    if not result["budget_pass"] or not result["range_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

