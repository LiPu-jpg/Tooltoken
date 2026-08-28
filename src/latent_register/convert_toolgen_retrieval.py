from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latent_register.benchmark_metrics import aggregate_records


def recover_unique_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover query records from ToolGen's three repeated metric blocks."""
    if not logs:
        return []
    if len(logs) % 3 != 0:
        raise ValueError(
            "Official ToolGen logs are neither empty nor three equal metric blocks"
        )
    block_size = len(logs) // 3
    first = logs[:block_size]
    if logs[block_size : 2 * block_size] != first or logs[2 * block_size :] != first:
        raise ValueError("Official ToolGen metric log blocks are not identical")
    return first


def load_valid_actions(path: str | Path) -> set[str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        actions = value.values()
    elif isinstance(value, list):
        actions = value
    else:
        raise ValueError("Valid-action source must be a JSON object or array")
    result = {str(action) for action in actions if str(action)}
    if not result:
        raise ValueError("Valid-action source contains no actions")
    return result


def convert_result(
    value: dict[str, Any], *, valid_actions: set[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logs = value.get("logs", [])
    if not isinstance(logs, list) or not all(isinstance(item, dict) for item in logs):
        raise ValueError("ToolGen result must contain an object-valued logs list")
    unique_logs = recover_unique_logs(logs)
    records = []
    raw_prediction_count = 0
    invalid_prediction_count = 0
    for index, item in enumerate(unique_logs):
        labels = item.get("label", [])
        predictions = item.get("pred", [])
        if not isinstance(labels, list) or not isinstance(predictions, list):
            raise ValueError(f"Log {index} has non-list labels or predictions")
        raw_predictions = [str(prediction) for prediction in predictions if str(prediction)]
        invalid_predictions = (
            [prediction for prediction in raw_predictions if prediction not in valid_actions]
            if valid_actions is not None
            else []
        )
        filtered_predictions = (
            [prediction for prediction in raw_predictions if prediction in valid_actions]
            if valid_actions is not None
            else raw_predictions
        )
        raw_prediction_count += len(raw_predictions)
        invalid_prediction_count += len(invalid_predictions)
        records.append(
            {
                "id": str(index),
                "reference_tools": [str(label) for label in labels if str(label)],
                "predicted_tools": filtered_predictions,
                "invalid_predictions": invalid_predictions,
            }
        )
    summary = aggregate_records(records)
    summary["source_model"] = value.get("model")
    summary["official_ndcg"] = value.get("ndcg", {})
    summary["official_log_rows"] = len(logs)
    summary["unique_query_rows"] = len(records)
    summary["raw_prediction_count"] = raw_prediction_count
    summary["invalid_prediction_count"] = invalid_prediction_count
    summary["invalid_prediction_rate"] = (
        invalid_prediction_count / raw_prediction_count
        if raw_prediction_count
        else 0.0
    )
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official ToolGen retrieval output to benchmark JSONL"
    )
    parser.add_argument("official_result")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--valid-actions",
        help="JSON object values or array defining actions accepted by ToolGen",
    )
    args = parser.parse_args()

    source = Path(args.official_result)
    value = json.loads(source.read_text(encoding="utf-8"))
    valid_actions = load_valid_actions(args.valid_actions) if args.valid_actions else None
    records, summary = convert_result(value, valid_actions=valid_actions)
    with Path(args.predictions).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
