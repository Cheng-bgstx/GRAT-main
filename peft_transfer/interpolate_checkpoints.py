#!/usr/bin/env python3
"""Interpolate matching tensor entries from two PEFT checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def interpolate_state_dict(left, right, right_weight):
    if left.keys() != right.keys():
        raise KeyError("state-dict entries do not match")
    result = {}
    for key in left:
        left_value, right_value = left[key], right[key]
        if not torch.is_tensor(left_value) or not torch.is_tensor(right_value):
            raise TypeError(f"state-dict entry {key} is not a tensor")
        if left_value.shape != right_value.shape:
            raise ValueError(f"shape mismatch for {key}: {left_value.shape} vs {right_value.shape}")
        if left_value.is_floating_point():
            result[key] = (1.0 - right_weight) * left_value + right_weight * right_value
        else:
            if not torch.equal(left_value, right_value):
                raise ValueError(f"non-floating entry {key} differs")
            result[key] = left_value.clone()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--right-weight", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.right_weight <= 1.0:
        raise ValueError("right weight must lie in [0, 1]")
    left = torch.load(args.left, map_location="cpu")
    right = torch.load(args.right, map_location="cpu")
    if isinstance(left, dict) and "state_dict" in left:
        if not isinstance(right, dict) or "state_dict" not in right:
            raise KeyError("both checkpoints must contain state_dict")
        result = dict(left)
        result["state_dict"] = interpolate_state_dict(
            left["state_dict"], right["state_dict"], args.right_weight
        )
        result["optimizer"] = {}
        result["interpolation"] = {
            "left": args.left,
            "right": args.right,
            "right_weight": args.right_weight,
        }
    else:
        result = interpolate_state_dict(left, right, args.right_weight)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    metadata = {
        "left": args.left,
        "right": args.right,
        "right_weight": args.right_weight,
        "entries": len(result),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
