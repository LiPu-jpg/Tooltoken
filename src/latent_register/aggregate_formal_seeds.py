"""Aggregate one paired comparison cell across three independent seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from .compare_controlled_predictions import load_prediction_rows


FORMAL_SEEDS = (17, 29, 43)
T_CRITICAL_95_DF2 = 4.302652729696142
COMPARISON_CODE_FIELDS = (
    "compare_controlled_predictions_code_sha256",
    "benchmark_metrics_code_sha256",
)
REFERENCE_FIELDS = (
    "family",
    "source_condition",
    "example_id",
    "query",
    "reference_tools",
    "reference_arguments",
    "schema",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_fingerprint(path: Path) -> str:
    rows = load_prediction_rows(path)
    references = [
        {
            **{field: row.get(field) for field in REFERENCE_FIELDS},
            "source_condition": row.get("source_condition", row.get("condition")),
        }
        for row in rows
    ]
    references.sort(
        key=lambda row: (
            str(row.get("family")),
            str(row.get("source_condition", row.get("condition"))),
            str(row.get("example_id")),
        )
    )
    encoded = json.dumps(
        references, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value}")
    return result


def _seed_summary(values: Mapping[int, float]) -> dict[str, Any]:
    ordered = [_finite_float(values[seed], f"seed {seed}") for seed in FORMAL_SEEDS]
    mean = statistics.mean(ordered)
    sample_std = statistics.stdev(ordered)
    half_width = T_CRITICAL_95_DF2 * sample_std / math.sqrt(len(ordered))
    return {
        "values_by_seed": {str(seed): values[seed] for seed in FORMAL_SEEDS},
        "mean": mean,
        "sample_std": sample_std,
        "seed_t_95_ci": [mean - half_width, mean + half_width],
        "seed_count": len(ordered),
    }


def aggregate_formal_seed_comparisons(
    comparisons: Mapping[int, Path],
) -> dict[str, Any]:
    if set(comparisons) != set(FORMAL_SEEDS):
        raise ValueError(f"Comparisons must contain exactly seeds {FORMAL_SEEDS}")

    payloads: dict[int, dict[str, Any]] = {}
    fingerprints: dict[int, str] = {}
    code_hashes_by_seed: dict[int, dict[str, str]] = {}
    inputs: dict[str, Any] = {}
    for seed in FORMAL_SEEDS:
        path = comparisons[seed]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Comparison for seed {seed} is not a JSON object")
        sources = payload.get("sources")
        if not isinstance(sources, dict):
            raise ValueError(f"Comparison for seed {seed} is missing sources")
        baseline_path = Path(str(sources["baseline_path"]))
        candidate_path = Path(str(sources["candidate_path"]))
        baseline_fingerprint = reference_fingerprint(baseline_path)
        candidate_fingerprint = reference_fingerprint(candidate_path)
        if baseline_fingerprint != candidate_fingerprint:
            raise ValueError(f"Within-seed reference mismatch for seed {seed}")
        fingerprints[seed] = baseline_fingerprint
        code_hashes = {
            field: str(sources.get(field, "")) for field in COMPARISON_CODE_FIELDS
        }
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in code_hashes.values()
        ):
            raise ValueError(f"Comparison for seed {seed} has invalid code hashes")
        code_hashes_by_seed[seed] = code_hashes
        payloads[seed] = payload
        inputs[str(seed)] = {
            "comparison_path": str(path.resolve()),
            "comparison_sha256": _sha256(path),
            "baseline_predictions_path": str(baseline_path.resolve()),
            "baseline_predictions_sha256": _sha256(baseline_path),
            "candidate_predictions_path": str(candidate_path.resolve()),
            "candidate_predictions_sha256": _sha256(candidate_path),
            "reference_fingerprint": baseline_fingerprint,
        }

    if len(set(fingerprints.values())) != 1:
        raise ValueError(f"Cross-seed reference fingerprints differ: {fingerprints}")
    if len({tuple(sorted(value.items())) for value in code_hashes_by_seed.values()}) != 1:
        raise ValueError(f"Cross-seed comparison code hashes differ: {code_hashes_by_seed}")

    first = payloads[FORMAL_SEEDS[0]]
    invariant_fields = (
        "pair_count",
        "require_registry_match",
        "registry_pairs",
        "registry_mismatches",
        "selected_families",
    )
    for seed in FORMAL_SEEDS[1:]:
        for field in invariant_fields:
            if payloads[seed].get(field) != first.get(field):
                raise ValueError(f"Cross-seed comparison mismatch for {field}")

    first_families = first.get("families")
    if not isinstance(first_families, dict):
        raise ValueError("Comparison is missing families")
    family_names = set(first_families)
    if any(set(payloads[seed].get("families", {})) != family_names for seed in FORMAL_SEEDS):
        raise ValueError("Cross-seed family sets differ")

    families: dict[str, Any] = {}
    for family in sorted(family_names):
        metrics_by_seed = {
            seed: payloads[seed]["families"][family].get("paired_metrics", {})
            for seed in FORMAL_SEEDS
        }
        metric_names = set(metrics_by_seed[FORMAL_SEEDS[0]])
        if any(set(metrics_by_seed[seed]) != metric_names for seed in FORMAL_SEEDS):
            raise ValueError(f"Cross-seed metric sets differ for {family}")
        metrics: dict[str, Any] = {}
        for metric in sorted(metric_names):
            rows = {seed: metrics_by_seed[seed][metric] for seed in FORMAL_SEEDS}
            counts = {int(rows[seed]["count"]) for seed in FORMAL_SEEDS}
            if len(counts) != 1:
                raise ValueError(f"Cross-seed metric counts differ for {family}.{metric}")
            metrics[metric] = {
                "count_per_seed": counts.pop(),
                "baseline": _seed_summary(
                    {seed: _finite_float(rows[seed]["baseline"], "baseline") for seed in FORMAL_SEEDS}
                ),
                "candidate": _seed_summary(
                    {seed: _finite_float(rows[seed]["candidate"], "candidate") for seed in FORMAL_SEEDS}
                ),
                "delta": _seed_summary(
                    {seed: _finite_float(rows[seed]["delta"], "delta") for seed in FORMAL_SEEDS}
                ),
                "paired_bootstrap_95_ci_by_seed": {
                    str(seed): rows[seed]["paired_bootstrap_95_ci"]
                    for seed in FORMAL_SEEDS
                },
            }
        families[family] = {"metrics": metrics}

    return {
        "kind": "formal_three_seed_paired_comparison",
        "seeds": list(FORMAL_SEEDS),
        "reference_fingerprint": next(iter(fingerprints.values())),
        "code_hashes": code_hashes_by_seed[FORMAL_SEEDS[0]],
        **{field: first.get(field) for field in invariant_fields},
        "families": families,
        "inputs": inputs,
    }


def _parse_comparison(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Comparison must be SEED=/path/to/comparison.json")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed: {seed_text}") from exc
    return seed, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="append", type=_parse_comparison, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons = dict(args.comparison)
    if len(comparisons) != len(args.comparison):
        raise ValueError("Duplicate comparison seed")
    result = aggregate_formal_seed_comparisons(comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
