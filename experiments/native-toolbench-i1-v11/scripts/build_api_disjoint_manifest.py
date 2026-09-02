#!/usr/bin/env python3
"""Build a 1K-action training manifest disjoint from official G1/I1 test APIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latebound_sequence_sft.data import (  # noqa: E402
    action_key,
    iter_retrieval_rows,
    load_atomic_map,
    load_corpus,
    sha256_file,
)


def read_exclusion(path: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "toolbench_g1_test_exclusion_v1":
        raise ValueError("Unexpected I1 exclusion manifest kind")
    queries = {str(value) for value in payload.get("test_queries", [])}
    actions = {str(value) for value in payload.get("test_actions", [])}
    if not queries or not actions:
        raise ValueError("I1 exclusion manifest is empty")
    return queries, actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--atomic-map", type=Path, required=True)
    parser.add_argument("--i1-exclusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.api_count < 1:
        raise ValueError("api-count must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    test_queries, i1_actions = read_exclusion(args.i1_exclusion)
    corpus, corpus_hash = load_corpus(args.corpus)
    atomic_reverse, atomic_hash = load_atomic_map(args.atomic_map)

    # Only singleton action groups are eligible. This makes the 1K count an
    # exact-document count and avoids ambiguous action->document expansion.
    by_action: dict[str, list[Any]] = defaultdict(list)
    for document in corpus:
        by_action[document.action].append(document)
    eligible_actions = sorted(
        action
        for action, docs in by_action.items()
        if action not in i1_actions and len(docs) == 1 and action in atomic_reverse.values()
    )
    if len(eligible_actions) < args.api_count:
        raise ValueError(
            f"Only {len(eligible_actions)} singleton train actions remain; need {args.api_count}"
        )
    eligible_actions.sort(
        key=lambda value: hashlib.sha256(f"{args.seed}:{value}".encode("utf-8")).hexdigest()
    )
    selected_actions = sorted(eligible_actions[: args.api_count])
    selected_docs = [by_action[action][0] for action in selected_actions]
    selected_docids = {document.docid for document in selected_docs}

    examples: list[dict[str, Any]] = []
    excluded_query_rows = 0
    source_rows = 0
    for row in iter_retrieval_rows(args.retrieval):
        source_rows += 1
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError("Every retrieval row must contain two messages")
        query, answer = conversations
        query_text = str(query.get("content", "")).strip()
        if query_text in test_queries:
            excluded_query_rows += 1
            continue
        token = str(answer.get("content", "")).strip()
        action = atomic_reverse.get(token)
        if action is None:
            continue
        docs = by_action.get(action, [])
        if action not in selected_actions or len(docs) != 1:
            continue
        document = docs[0]
        if document.docid not in selected_docids:
            raise AssertionError("Selected action/document mismatch")
        examples.append(
            {
                "query": query_text,
                "action": action,
                "document_id": document.docid,
                "positive_document_count": 1,
            }
        )
    if not examples:
        raise ValueError("No training examples remain for selected APIs")

    documents_payload = [
        {"docid": document.docid, "action": document.action, "document": document.document}
        for document in sorted(selected_docs, key=lambda item: (int(item.docid), item.docid))
    ]
    train_identities = {
        hashlib.sha256(
            f"{document.docid}\n{json.dumps(document.document, ensure_ascii=False, sort_keys=True)}".encode(
                "utf-8"
            )
        ).hexdigest()
        for document in selected_docs
    }
    payload = {
        "kind": "native_latebound_standard_sequence_sft_manifest",
        "version": 3,
        "split_kind": "api_identity_disjoint_from_official_g1_test",
        "seed": args.seed,
        "requested_train_api_count": args.api_count,
        "train_api_count": len(selected_actions),
        "train_exact_document_count": len(documents_payload),
        "exact_identity_count": len(documents_payload),
        "train_action_keys": selected_actions,
        "train_document_identity_sha256": hashlib.sha256(
            json.dumps(sorted(train_identities), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "official_g1_test_query_count": len(test_queries),
        "official_g1_test_action_count": len(i1_actions),
        "train_i1_action_intersection_count": len(set(selected_actions) & i1_actions),
        "train_source_rows_read": source_rows,
        "excluded_i1_query_rows": excluded_query_rows,
        "train_example_count": len(examples),
        "retrieval_sha256": sha256_file(args.retrieval),
        "corpus_sha256": corpus_hash,
        "atomic_map_sha256": atomic_hash,
        "official_g1_test_query_sha256": hashlib.sha256(
            json.dumps(sorted(test_queries), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "i1_exclusion_sha256": sha256_file(args.i1_exclusion),
        "documents": documents_payload,
        "examples": examples,
        "document_selection_policy": "1000 singleton exact documents, action-disjoint from G1 test qrels",
        "zero_step_evaluation": True,
        "qrels_read_by_training_builder": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"manifest-ready train_apis={len(selected_actions)} train_examples={len(examples)} "
        f"i1_actions={len(i1_actions)} train_i1_intersection=0 excluded_queries={len(test_queries)}"
    )


if __name__ == "__main__":
    main()
