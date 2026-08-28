"""Validate formal latent-training artifacts before downstream evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .physical_tokens import validate_token_identity_audit


REQUIRED_ZERO_AUDIT_FIELDS = (
    "evaluation_ids_seen_during_training",
    "evaluation_input_row_max_change",
    "evaluation_output_row_max_change",
)


def _assert_finite(value: Any, path: str = "results") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def audit_training_results(
    results_path: Path,
    *,
    expected_seed: int,
    expected_steps: int,
    expected_world_size: int,
    stage: str,
) -> dict[str, Any]:
    raw = results_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Training results must contain a JSON object")
    _assert_finite(payload)

    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Training results are missing config")
    observed = {
        "seed": int(config.get("seed", -1)),
        "completed_steps": int(payload.get("completed_steps", -1)),
        "world_size": int(payload.get("world_size", -1)),
    }
    expected = {
        "seed": expected_seed,
        "completed_steps": expected_steps,
        "world_size": expected_world_size,
    }
    if observed != expected:
        raise ValueError(f"Training configuration mismatch: {observed} != {expected}")

    physical_audit = payload.get("physical_token_audit")
    if not isinstance(physical_audit, dict):
        raise ValueError("Training results are missing physical_token_audit")
    missing = [key for key in REQUIRED_ZERO_AUDIT_FIELDS if key not in physical_audit]
    if missing:
        raise ValueError(f"Physical-token audit is incomplete: {missing}")
    failed = {
        key: physical_audit[key]
        for key in REQUIRED_ZERO_AUDIT_FIELDS
        if float(physical_audit[key]) != 0.0
    }
    if failed:
        raise ValueError(f"Physical-token isolation failed: {failed}")
    identity_audit = validate_token_identity_audit(physical_audit)
    if not identity_audit["passed"]:
        raise ValueError(
            f"Physical-token identity audit failed: {identity_audit}"
        )

    return {
        "kind": "formal_latent_training_audit",
        "stage": stage,
        "results_path": str(results_path.resolve()),
        "results_sha256": hashlib.sha256(raw).hexdigest(),
        **observed,
        "all_numeric_results_finite": True,
        "physical_token_isolation": True,
        "physical_token_identity": identity_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--stage", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_training_results(
        args.results,
        expected_seed=args.expected_seed,
        expected_steps=args.expected_steps,
        expected_world_size=args.expected_world_size,
        stage=args.stage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
