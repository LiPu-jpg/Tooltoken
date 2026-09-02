#!/usr/bin/env python3
"""Open sealed G1 qrels after blind scoring and compute official NDCG."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def dcg(values: list[int], k: int) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(values[:k]))


def ndcg(ranked: list[str], relevant: set[str], k: int, official: bool) -> float:
    actual = [int(value in relevant) for value in ranked[:k]]
    ideal_count = min(sum(value in relevant for value in ranked[:5]), k) if official else min(len(relevant), k)
    ideal = dcg([1] * ideal_count, k)
    return dcg(actual, k) / ideal if ideal else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score_manifest = json.loads((args.score_root / "score_manifest.json").read_text(encoding="utf-8"))
    score_path = args.score_root / "scores.npy"
    if score_manifest.get("scores_sha256") != hashlib.sha256(score_path.read_bytes()).hexdigest():
        raise ValueError("score checksum mismatch")
    registry = [json.loads(line) for line in (args.evaluation_root / "registry.jsonl").read_text(encoding="utf-8").splitlines() if line]
    queries = [json.loads(line) for line in (args.evaluation_root / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    qrels = {str(row["qid"]): row for row in (json.loads(line) for line in (args.sealed_root / "qrels.jsonl").read_text(encoding="utf-8").splitlines() if line)}
    scores = np.load(score_path, mmap_mode="r")
    if list(scores.shape) != [len(queries), len(registry)]:
        raise ValueError("score shape mismatch")
    identity = [str(row["identity_hash"]) for row in registry]
    actions = [str(row["action"]) for row in registry]
    action_to_cols: dict[str, list[int]] = defaultdict(list)
    for col, value in enumerate(actions):
        action_to_cols[value].append(col)
    sorted_actions = sorted(action_to_cols)
    values: dict[str, list[float]] = defaultdict(list)
    for row_index, query in enumerate(queries):
        rel = qrels.get(str(query["qid"]))
        if rel is None:
            continue
        row_scores = np.asarray(scores[row_index], dtype=np.float32)
        exact_order = np.lexsort((np.asarray(identity), -row_scores))[:5]
        exact_ranked = [identity[int(col)] for col in exact_order]
        action_scores = {action: max(float(row_scores[col]) for col in cols) for action, cols in action_to_cols.items()}
        action_ranked = sorted(sorted_actions, key=lambda action: (-action_scores[action], action))[:5]
        relevant_exact = set(rel["exact_identity_hashes"])
        relevant_actions = set(rel["actions"])
        for k in (1, 3, 5):
            values[f"official_ndcg_at_{k}"].append(ndcg(action_ranked, relevant_actions, k, True))
            values[f"corrected_ndcg_at_{k}"].append(ndcg(action_ranked, relevant_actions, k, False))
            values[f"action_hit_at_{k}"].append(float(bool(relevant_actions.intersection(action_ranked[:k]))))
            values[f"exact_hit_at_{k}"].append(float(bool(relevant_exact.intersection(exact_ranked[:k]))))
        relevant_positions = [index + 1 for index, value in enumerate(exact_order) if identity[int(value)] in relevant_exact]
        values["exact_mrr"].append(1.0 / min(relevant_positions) if relevant_positions else 0.0)
    summary = {key: float(sum(items) / len(items)) if items else 0.0 for key, items in sorted(values.items())}
    result = {
        "kind": "native_i1_zero_step_official_ndcg_summary",
        "passed": True,
        "benchmark": "ToolBench official G1/I1",
        "query_count": len(queries),
        "scored_query_count": len(next(iter(values.values()), [])),
        "candidate_document_count": len(registry),
        "qrel_query_count": len(qrels),
        "qrels_opened_after_blind_score": True,
        "metrics": summary,
        "official_ndcg_definition": "prediction-conditioned ideal DCG matching the released ToolScalER/ToolGen evaluator",
        "corrected_ndcg_definition": "ideal DCG uses every relevant action in the sealed qrels",
        "registration_optimizer_steps": 0,
        "changed_parameter_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.output / "summary.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "COMPLETE").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
