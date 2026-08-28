from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from latent_register.validate_toolgen_tokenizer import sha256_file, write_json_atomic


PAPER_TABLE1_MULTI_DOMAIN = {
    "G1": {1: 0.8767, 3: 0.8884, 5: 0.9154},
    "G2": {1: 0.8346, 3: 0.8624, 5: 0.8884},
    "G3": {1: 0.7900, 3: 0.7980, 5: 0.8479},
}


def parse_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3 or fields[2] not in {"True", "False"}:
                raise ValueError(f"Malformed retrieval task at line {line_number}")
            stage, split, limit_text = fields
            limit_to_stage_space = limit_text == "True"
            key = (stage, split, limit_to_stage_space)
            if key in seen:
                raise ValueError(f"Duplicate retrieval task: {key}")
            seen.add(key)
            tasks.append(
                {
                    "stage": stage,
                    "split": split,
                    "limit_to_stage_space": limit_to_stage_space,
                    "domain": "in_domain" if limit_to_stage_space else "multi_domain",
                }
            )
    return tasks


def _weighted_metrics(
    runs: list[dict[str, Any]], *, weighted: bool
) -> dict[str, float]:
    values: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    for run in runs:
        weight = int(run["count"]) if weighted else 1
        for key, value in run["metrics"].items():
            numeric = float(value)
            if math.isfinite(numeric):
                values[key].append((numeric, weight))
    return {
        key: sum(value * weight for value, weight in entries)
        / sum(weight for _, weight in entries)
        for key, entries in sorted(values.items())
    }


def validate_official_ndcg(summary: dict[str, Any], *, tolerance: float = 1e-12) -> None:
    official = summary.get("official_ndcg", {})
    metrics = summary.get("metrics", {})
    for k in (1, 3, 5):
        official_key = f"ndcg@{k}"
        corrected_key = f"ndcg_paper_compatible_at_{k}"
        if official_key not in official or corrected_key not in metrics:
            raise ValueError(f"Missing NDCG pair for k={k}")
        if not math.isclose(
            float(official[official_key]),
            float(metrics[corrected_key]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"Official and recomputed paper-compatible NDCG differ at k={k}"
            )


def compare_paper_table1_multi_domain(runs: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        str(run["stage"]): run
        for run in runs
        if run["domain"] == "multi_domain" and run["split"] == "instruction"
    }
    missing = sorted(set(PAPER_TABLE1_MULTI_DOMAIN) - set(indexed))
    if missing:
        return {
            "complete": False,
            "missing_stages": missing,
            "scope": "Table 1 Multi-Domain instruction retrieval",
        }
    cells: dict[str, dict[str, dict[str, float]]] = {}
    absolute_differences: list[float] = []
    for stage, paper_values in PAPER_TABLE1_MULTI_DOMAIN.items():
        cells[stage] = {}
        run = indexed[stage]
        for k, paper_value in paper_values.items():
            reproduced = float(run["metrics"][f"ndcg_paper_compatible_at_{k}"])
            difference = reproduced - paper_value
            absolute_differences.append(abs(difference))
            cells[stage][f"ndcg_at_{k}"] = {
                "paper": paper_value,
                "released_checkpoint": reproduced,
                "difference": difference,
                "difference_points": difference * 100.0,
            }
    macro: dict[str, dict[str, float]] = {}
    for k in (1, 3, 5):
        paper_value = sum(
            values[k] for values in PAPER_TABLE1_MULTI_DOMAIN.values()
        ) / len(PAPER_TABLE1_MULTI_DOMAIN)
        reproduced = sum(
            float(indexed[stage]["metrics"][f"ndcg_paper_compatible_at_{k}"])
            for stage in PAPER_TABLE1_MULTI_DOMAIN
        ) / len(PAPER_TABLE1_MULTI_DOMAIN)
        difference = reproduced - paper_value
        macro[f"ndcg_at_{k}"] = {
            "paper": paper_value,
            "released_checkpoint": reproduced,
            "difference": difference,
            "difference_points": difference * 100.0,
        }
    return {
        "complete": True,
        "scope": "Table 1 Multi-Domain instruction retrieval",
        "paper_model_training_scope": "combined I1+I2+I3",
        "cells": cells,
        "macro": macro,
        "max_absolute_cell_difference_points": max(absolute_differences) * 100.0,
    }


def aggregate_runs(
    *,
    root: str | Path,
    task_file: str | Path,
    tokenizer_audit: str | Path,
    expected_task_count: int = 12,
) -> dict[str, Any]:
    root = Path(root)
    task_file = Path(task_file)
    tokenizer_audit = Path(tokenizer_audit)
    tasks = parse_tasks(task_file)
    if len(tasks) != expected_task_count:
        raise ValueError(
            f"Expected {expected_task_count} retrieval tasks, found {len(tasks)}"
        )

    runs = []
    for task in tasks:
        directory = root / task["domain"] / f"{task['stage']}_{task['split']}"
        if not (directory / "COMPLETE").is_file():
            raise FileNotFoundError(f"Incomplete retrieval run: {directory}")
        metrics_path = directory / "metrics.json"
        predictions_path = directory / "predictions.jsonl"
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        validate_official_ndcg(summary)
        if int(summary.get("count", -1)) <= 0:
            raise ValueError(f"Retrieval run has no examples: {directory}")
        raw_prediction_count = int(summary.get("raw_prediction_count", 0))
        invalid_prediction_count = int(summary.get("invalid_prediction_count", 0))
        if not 0 <= invalid_prediction_count <= raw_prediction_count:
            raise ValueError(f"Invalid prediction counts in: {directory}")
        runs.append(
            {
                **task,
                "count": int(summary["count"]),
                "metrics": summary["metrics"],
                "official_ndcg": summary["official_ndcg"],
                "raw_prediction_count": raw_prediction_count,
                "invalid_prediction_count": invalid_prediction_count,
                "invalid_prediction_rate": (
                    invalid_prediction_count / raw_prediction_count
                    if raw_prediction_count
                    else 0.0
                ),
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(predictions_path),
            }
        )

    raw_prediction_count = sum(run["raw_prediction_count"] for run in runs)
    invalid_prediction_count = sum(run["invalid_prediction_count"] for run in runs)
    return {
        "version": 1,
        "task_count": len(runs),
        "example_count_sum_across_conditions": sum(run["count"] for run in runs),
        "released_evaluator_recomputation_passed": True,
        "paper_table1_multi_domain": compare_paper_table1_multi_domain(runs),
        "raw_prediction_count": raw_prediction_count,
        "invalid_prediction_count": invalid_prediction_count,
        "invalid_prediction_rate": (
            invalid_prediction_count / raw_prediction_count
            if raw_prediction_count
            else 0.0
        ),
        "macro_metrics": _weighted_metrics(runs, weighted=False),
        "micro_metrics": _weighted_metrics(runs, weighted=True),
        "runs": runs,
        "tokenizer_audit": json.loads(tokenizer_audit.read_text(encoding="utf-8")),
        "sources": {
            "task_file_sha256": sha256_file(task_file),
            "tokenizer_audit_sha256": sha256_file(tokenizer_audit),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate official ToolGen retrieval runs"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--tokenizer-audit", required=True)
    parser.add_argument("--expected-task-count", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate_runs(
        root=args.root,
        task_file=args.task_file,
        tokenizer_audit=args.tokenizer_audit,
        expected_task_count=args.expected_task_count,
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
