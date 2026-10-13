"""Static source-only and partition audit."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "autoresearch_med"
FORBIDDEN_IMPORT_FRAGMENTS = {
    "evaluate",
    "medsa_surrogate",
    "Medical-SAM-Adapter-main",
    "sam_lora_image_encoder",
}


def load_json_yaml(path: Path) -> dict:
    return json.loads(path.read_text())


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit() -> dict:
    source = load_json_yaml(WORKSPACE / "configs/source_only.yaml")
    contaminated = set(load_json_yaml(WORKSPACE / "manifests/contaminated.json")["indices"])
    development_manifest = load_json_yaml(WORKSPACE / "manifests/development.json")
    confirmation_manifest = load_json_yaml(WORKSPACE / "manifests/confirmation.json")
    development = set(development_manifest["indices"])
    confirmation = set(confirmation_manifest["indices"])
    candidate = WORKSPACE / "candidates/active.py"
    imports = imported_modules(candidate)
    forbidden_imports = sorted(
        module for module in imports
        if any(fragment in module for fragment in FORBIDDEN_IMPORT_FRAGMENTS)
    )
    checkpoint_manifest = load_json_yaml(WORKSPACE / "manifests/checkpoints.json")
    missing_paths = []
    hash_mismatches = []
    for section in ("source", "development_targets"):
        for name, value in checkpoint_manifest[section].items():
            if not (ROOT / value).is_file():
                missing_paths.append({"name": name, "path": value})
            elif checkpoint_manifest["sha256"].get(name) != sha256(ROOT / value):
                hash_mismatches.append({"name": name, "path": value})
    data_root = ROOT / "data/ISIC"
    missing_samples = []
    for manifest in (development_manifest, confirmation_manifest):
        if set(row["index"] for row in manifest["samples"]) != set(manifest["indices"]):
            missing_samples.append({"manifest": manifest["purpose"], "reason": "index mismatch"})
        for row in manifest["samples"]:
            for key in ("image", "mask"):
                if not (data_root / row[key]).is_file():
                    missing_samples.append({"index": row["index"], "path": row[key]})
    rules = {
        "fixed_budget": source["steps"] == 10
        and source["step_size_255"] == 2.0
        and source["epsilon_255"] == 10.0,
        "vit_b": source["backbone"] == "vit_b",
        "partitions_disjoint": not (
            contaminated & development
            or contaminated & confirmation
            or development & confirmation
        ),
        "candidate_import_boundary": not forbidden_imports,
        "confirmation_blocked_without_targets": not bool(
            checkpoint_manifest["confirmation_targets"]
        ),
        "declared_paths_exist": not missing_paths,
        "checkpoint_hashes_match": not hash_mismatches,
        "sample_identities_resolve": not missing_samples,
    }
    return {
        "passed": all(rules.values()),
        "rules": rules,
        "forbidden_imports": forbidden_imports,
        "missing_paths": missing_paths,
        "hash_mismatches": hash_mismatches,
        "missing_samples": missing_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit()
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

