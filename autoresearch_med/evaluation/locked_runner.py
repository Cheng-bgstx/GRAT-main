"""Append-only run registration for the locked evaluation boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "autoresearch_med"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, str]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "unavailable_no_commit"
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout
    return {"commit": commit or "unavailable", "dirty_sha256": hashlib.sha256(status.encode()).hexdigest()}


def register_run(
    run_id: str,
    tier: int,
    config: Path,
    candidate: Path,
    status: str = "registered_not_executed",
    research_stage: str = "development",
) -> Path:
    output = WORKSPACE / "results/run_manifests" / f"{run_id}.json"
    if output.exists():
        raise FileExistsError(f"run manifest already exists: {output}")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "research_stage": research_stage,
        "config": str(config.relative_to(ROOT)),
        "config_sha256": sha256(config),
        "candidate": str(candidate.relative_to(ROOT)),
        "candidate_sha256": sha256(candidate),
        "git": git_state(),
        "status": status,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    experiments = WORKSPACE / "results/experiments.jsonl"
    with experiments.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tier", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--config", default="autoresearch_med/configs/development.yaml")
    parser.add_argument(
        "--status",
        choices=("registered_not_executed", "completed", "failed", "invalidated"),
        default="registered_not_executed",
    )
    parser.add_argument(
        "--research-stage",
        choices=("smoke", "development", "confirmation"),
        default="development",
    )
    args = parser.parse_args()
    path = register_run(
        args.run_id,
        args.tier,
        ROOT / args.config,
        WORKSPACE / "candidates/active.py",
        args.status,
        args.research_stage,
    )
    print(path)


if __name__ == "__main__":
    main()

