from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import benchmark_metrics as benchmark_metrics_module
from .benchmark_metrics import (
    aggregate_records,
    paired_bootstrap_interval,
    score_record,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _source_condition(row: Mapping[str, Any]) -> str:
    return str(row.get("source_condition", row.get("condition")))


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("family")),
        _source_condition(row),
        str(row.get("example_id")),
    )


def _index_rows(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate {label} prediction key: {key}")
        indexed[key] = row
    return indexed


def _validate_reference_pair(
    key: tuple[str, str, str],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    for field in (
        "query",
        "reference_tools",
        "reference_arguments",
        "schema",
    ):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Reference mismatch for {key}: {field}")
    information_conditions = {
        str(baseline.get("information_condition", "")),
        str(candidate.get("information_condition", "")),
    }
    if "common_document" in information_conditions:
        if not information_conditions.issubset(
            {"common_document", "full_document_oracle"}
        ):
            raise ValueError(f"Information-condition mismatch for {key}")
        for field in (
            "common_document_agent_model_path",
            "common_document_agent_audit_sha256",
        ):
            left = baseline.get(field)
            right = candidate.get(field)
            if not left or left != right:
                raise ValueError(f"Common-document Agent mismatch for {key}: {field}")


def compare_controlled_predictions(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    require_registry_match: bool,
    bootstrap_samples: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    baseline = _index_rows(baseline_rows, "baseline")
    candidate = _index_rows(candidate_rows, "candidate")
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            f"Prediction keys differ; missing candidate={missing[:5]}, "
            f"extra candidate={extra[:5]}"
        )

    pairs_by_family: defaultdict[
        str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]
    ] = defaultdict(list)
    registry_pairs = 0
    registry_mismatches = 0
    for key in sorted(baseline):
        left = baseline[key]
        right = candidate[key]
        _validate_reference_pair(key, left, right)
        left_hash = left.get("registry_identity_sha256")
        right_hash = right.get("registry_identity_sha256")
        if left_hash is not None or right_hash is not None:
            if left_hash is None or right_hash is None:
                if require_registry_match:
                    raise ValueError(f"Registry hash is missing for {key}")
            else:
                registry_pairs += 1
                if left_hash != right_hash:
                    registry_mismatches += 1
                    if require_registry_match:
                        raise ValueError(f"Registry identity mismatch for {key}")
        elif require_registry_match:
            raise ValueError(f"Registry hash is missing for {key}")
        pairs_by_family[key[0]].append((left, right))

    families: dict[str, Any] = {}
    for family, pairs in sorted(pairs_by_family.items()):
        baseline_family = [left for left, _ in pairs]
        candidate_family = [right for _, right in pairs]
        scored = [(score_record(left), score_record(right)) for left, right in pairs]
        metric_names = sorted(
            set.intersection(
                *(set(left) & set(right) for left, right in scored)
            )
        )
        paired_metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(metric_names):
            values = [
                (float(left[metric]), float(right[metric]))
                for left, right in scored
                if math.isfinite(float(left[metric]))
                and math.isfinite(float(right[metric]))
            ]
            if not values:
                continue
            differences = [right - left for left, right in values]
            baseline_mean = sum(left for left, _ in values) / len(values)
            candidate_mean = sum(right for _, right in values) / len(values)
            lower, upper = paired_bootstrap_interval(
                differences,
                samples=bootstrap_samples,
                seed=seed + metric_index,
            )
            paired_metrics[metric] = {
                "count": len(values),
                "baseline": baseline_mean,
                "candidate": candidate_mean,
                "delta": sum(differences) / len(differences),
                "candidate_over_baseline": (
                    candidate_mean / baseline_mean if baseline_mean > 0.0 else None
                ),
                "paired_bootstrap_95_ci": [lower, upper],
                "bootstrap_samples": bootstrap_samples,
            }
        families[family] = {
            "count": len(pairs),
            "baseline": aggregate_records(baseline_family),
            "candidate": aggregate_records(candidate_family),
            "paired_metrics": paired_metrics,
        }
    return {
        "version": 1,
        "pair_count": len(baseline),
        "require_registry_match": require_registry_match,
        "registry_pairs": registry_pairs,
        "registry_mismatches": registry_mismatches,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison of standardized controlled-benchmark predictions"
    )
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-registry-match", action="store_true")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    families = set(args.family)
    baseline_rows = load_prediction_rows(baseline_path)
    candidate_rows = load_prediction_rows(candidate_path)
    if families:
        baseline_rows = [row for row in baseline_rows if row.get("family") in families]
        candidate_rows = [row for row in candidate_rows if row.get("family") in families]
    result = compare_controlled_predictions(
        baseline_rows,
        candidate_rows,
        require_registry_match=args.require_registry_match,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["sources"] = {
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": _sha256_file(baseline_path),
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": _sha256_file(candidate_path),
        "compare_controlled_predictions_code_sha256": _sha256_file(Path(__file__)),
        "benchmark_metrics_code_sha256": _sha256_file(
            Path(str(benchmark_metrics_module.__file__))
        ),
    }
    result["selected_families"] = sorted(families) if families else ["all"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
