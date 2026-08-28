from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def exact_mcnemar_pvalue(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(baseline_only, candidate_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_boolean(
    baseline_values: list[bool], candidate_values: list[bool]
) -> dict[str, int | float]:
    if len(baseline_values) != len(candidate_values):
        raise ValueError("Paired outcomes must have equal lengths")
    both = sum(left and right for left, right in zip(baseline_values, candidate_values))
    baseline_only = sum(
        left and not right for left, right in zip(baseline_values, candidate_values)
    )
    candidate_only = sum(
        not left and right for left, right in zip(baseline_values, candidate_values)
    )
    neither = len(baseline_values) - both - baseline_only - candidate_only
    count = len(baseline_values)
    return {
        "examples": count,
        "baseline_rate": (both + baseline_only) / max(1, count),
        "candidate_rate": (both + candidate_only) / max(1, count),
        "delta": (candidate_only - baseline_only) / max(1, count),
        "both_correct": both,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "neither_correct": neither,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(
            baseline_only, candidate_only
        ),
    }


def row_outcomes(row: dict[str, Any]) -> dict[str, bool]:
    outcomes = {
        "json_valid": row.get("parsed") is not None,
        "arguments_exact": row.get("parsed") == row.get("target"),
    }
    if "selected_tool" in row or "selected_tool_id" in row:
        selected_key = row.get("selected_tool_id", row.get("selected_tool"))
        target_key = row.get("tool_id", row.get("tool_name"))
        selection_correct = selected_key == target_key
        outcomes["selection_correct"] = selection_correct
        outcomes["end_to_end_exact"] = selection_correct and outcomes["arguments_exact"]
    return outcomes


def compare_predictions(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row.get("condition")), str(row.get("example_id")))
            if key in indexed:
                raise ValueError(f"Duplicate prediction key: {key}")
            indexed[key] = row
        return indexed

    baseline = index(baseline_rows)
    candidate = index(candidate_rows)
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            f"Prediction keys differ; missing candidate rows={missing[:5]}, "
            f"extra candidate rows={extra[:5]}"
        )

    by_condition: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for key in sorted(baseline):
        left = baseline[key]
        right = candidate[key]
        for field in ("tool_name", "tool_id", "target"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Prediction target mismatch for {key}: {field}")
        by_condition.setdefault(key[0], []).append((left, right))

    result: dict[str, Any] = {}
    for condition, pairs in by_condition.items():
        left_outcomes = [row_outcomes(left) for left, _ in pairs]
        right_outcomes = [row_outcomes(right) for _, right in pairs]
        outcome_names = set(left_outcomes[0])
        if any(set(row) != outcome_names for row in left_outcomes + right_outcomes):
            raise ValueError(f"Inconsistent prediction fields for condition {condition}")
        result[condition] = {
            name: paired_boolean(
                [row[name] for row in left_outcomes],
                [row[name] for row in right_outcomes],
            )
            for name in sorted(outcome_names)
        }
    return result
