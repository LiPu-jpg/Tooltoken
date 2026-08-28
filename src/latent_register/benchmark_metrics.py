from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def parse_json_value(value: Any) -> tuple[bool, Any]:
    """Parse JSON strings while accepting already structured predictions."""
    if not isinstance(value, str):
        return True, value
    try:
        return True, json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return False, None


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _join_path(parent: str, component: str) -> str:
    return component if not parent else f"{parent}.{component}"


def flatten_json(value: Any, path: str = "") -> dict[str, Any]:
    """Map every scalar leaf to a stable object/array path."""
    if isinstance(value, Mapping):
        if not value:
            return {path or "$": {}}
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            flattened.update(flatten_json(value[key], _join_path(path, str(key))))
        return flattened
    if isinstance(value, list):
        if not value:
            return {path or "$": []}
        flattened = {}
        for index, item in enumerate(value):
            child = f"{path}[{index}]" if path else f"[{index}]"
            flattened.update(flatten_json(item, child))
        return flattened
    return {path or "$": value}


def _prf(predicted: set[str], reference: set[str]) -> tuple[float, float, float]:
    if not predicted and not reference:
        return 1.0, 1.0, 1.0
    overlap = len(predicted & reference)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_arguments(
    predicted: Any,
    reference: Any,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    predicted_valid, predicted_value = parse_json_value(predicted)
    reference_valid, reference_value = parse_json_value(reference)
    if not reference_valid:
        raise ValueError("Reference arguments must be valid JSON")

    if not predicted_valid:
        return {
            "json_valid": 0.0,
            "argument_exact": 0.0,
            "key_exact": 0.0,
            "key_precision": 0.0,
            "key_recall": 0.0,
            "key_f1": 0.0,
            "value_exact": 0.0,
            "schema_valid": 0.0 if schema is not None else math.nan,
        }

    predicted_flat = flatten_json(predicted_value)
    reference_flat = flatten_json(reference_value)
    key_precision, key_recall, key_f1 = _prf(
        set(predicted_flat), set(reference_flat)
    )
    if reference_flat:
        matching_values = sum(
            path in predicted_flat
            and canonical_json(predicted_flat[path]) == canonical_json(reference_value)
            for path, reference_value in reference_flat.items()
        )
        value_exact = matching_values / len(reference_flat)
    else:
        value_exact = 1.0 if not predicted_flat else 0.0

    schema_valid = math.nan
    if schema is not None:
        Draft202012Validator.check_schema(schema)
        schema_valid = float(not any(Draft202012Validator(schema).iter_errors(predicted_value)))

    return {
        "json_valid": 1.0,
        "argument_exact": float(canonical_json(predicted_value) == canonical_json(reference_value)),
        "key_exact": float(set(predicted_flat) == set(reference_flat)),
        "key_precision": key_precision,
        "key_recall": key_recall,
        "key_f1": key_f1,
        "value_exact": value_exact,
        "schema_valid": schema_valid,
    }


def _discounted_gain(relevances: Iterable[int]) -> float:
    return sum(
        relevance / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def ndcg_at_k(
    reference_tools: Sequence[str],
    predicted_tools: Sequence[str],
    k: int,
    *,
    paper_compatible: bool = False,
) -> float:
    """Binary NDCG with an optional reproduction of ToolGen's denominator bug."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(reference_tools)
    if not relevant:
        return 1.0 if not predicted_tools else 0.0
    ranked = list(dict.fromkeys(predicted_tools))
    actual = _discounted_gain(int(tool in relevant) for tool in ranked[:k])
    if paper_compatible:
        # The upstream evaluator marks only relevant tools present in its beam,
        # so missed labels disappear from the ideal ranking denominator.
        ideal_relevant = len(relevant & set(ranked))
    else:
        ideal_relevant = len(relevant)
    ideal = _discounted_gain([1] * min(ideal_relevant, k))
    return actual / ideal if ideal else 0.0


def score_retrieval(
    predicted_tools: Sequence[str],
    reference_tools: Sequence[str],
) -> dict[str, float]:
    relevant = set(reference_tools)
    ranked = list(dict.fromkeys(predicted_tools))
    first_rank = next(
        (rank for rank, tool in enumerate(ranked, start=1) if tool in relevant),
        None,
    )
    return {
        "hit_at_1": float(bool(ranked) and ranked[0] in relevant),
        "recall_at_5": (
            len(relevant & set(ranked[:5])) / len(relevant) if relevant else float(not ranked)
        ),
        "mrr": 1.0 / first_rank if first_rank is not None else 0.0,
        **{
            f"ndcg_corrected_at_{k}": ndcg_at_k(reference_tools, ranked, k)
            for k in (1, 3, 5)
        },
        **{
            f"ndcg_paper_compatible_at_{k}": ndcg_at_k(
                reference_tools, ranked, k, paper_compatible=True
            )
            for k in (1, 3, 5)
        },
    }


def score_record(record: Mapping[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    if "reference_tools" in record or "predicted_tools" in record:
        scores.update(
            score_retrieval(
                list(record.get("predicted_tools", [])),
                list(record.get("reference_tools", [])),
            )
        )
        scores["tool_call_set_exact"] = float(
            set(record.get("predicted_tools", []))
            == set(record.get("reference_tools", []))
        )
        scores["ordered_tool_calls_exact"] = float(
            list(record.get("predicted_tools", []))
            == list(record.get("reference_tools", []))
        )
    if "reference_arguments" in record:
        scores.update(
            score_arguments(
                record.get("predicted_arguments"),
                record["reference_arguments"],
                record.get("schema"),
            )
        )
    if "reference_tools" in record and "reference_arguments" in record:
        selection_exact = scores["tool_call_set_exact"]
        scores["end_to_end_argument_exact"] = (
            selection_exact * scores["argument_exact"]
        )
        scores["end_to_end_key_exact"] = selection_exact * scores["key_exact"]
        scores["end_to_end_key_recall"] = selection_exact * scores["key_recall"]
    return scores


def _mean_without_nan(values: Iterable[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else math.nan


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [score_record(record) for record in records]
    keys = sorted({key for sample in scored for key in sample})
    return {
        "count": len(records),
        "metrics": {
            key: _mean_without_nan(sample[key] for sample in scored if key in sample)
            for key in keys
        },
    }


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 17,
) -> tuple[float, float]:
    if not differences:
        raise ValueError("differences must not be empty")
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score standardized ToolGen/late-binding JSONL predictions"
    )
    parser.add_argument("predictions")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = aggregate_records(load_jsonl(args.predictions))
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
