#!/usr/bin/env python3
"""Build the native meta-registration input for the frozen ToolBench-10K run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def endpoint(row: dict) -> tuple[str, str]:
    metadata = json.loads(row["api_params"])
    return str(metadata["exact_endpoint_identity"]), str(metadata["function_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-train", type=Path, required=True)
    parser.add_argument("--unseen-train", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout-tools-per-split", type=int, default=128)
    args = parser.parse_args()

    seen_rows = read_jsonl(args.seen_train)
    unseen_rows = read_jsonl(args.unseen_train)
    registry = {row["doc_id"]: row for row in read_jsonl(args.protocol / "documents.jsonl")}
    eval_queries = read_jsonl(args.protocol / "queries.jsonl")
    if len(seen_rows) != 10_000 or len(unseen_rows) != 10_000:
        raise ValueError("expected the two frozen 10,000-row training samples")
    if len(registry) != 10_000 or len(eval_queries) != 1_100:
        raise ValueError("frozen ToolBench-10K protocol cardinality changed")
    formal_query_texts = {str(row["query"]) for row in eval_queries}

    seen_by_endpoint: dict[str, dict] = {}
    query_targets: dict[str, set[str]] = defaultdict(set)
    excluded_query_rows = 0
    for row in seen_rows:
        exact, function_name = endpoint(row)
        seen_by_endpoint.setdefault(exact, row | {"function_name": function_name})
        query = str(row["query_for_retrieval"])
        if query in formal_query_texts:
            excluded_query_rows += 1
        else:
            query_targets[query].add(exact)
    if len(seen_by_endpoint) != 3_721:
        raise ValueError(f"seen tool count changed: {len(seen_by_endpoint)}")

    eval_positive = {
        doc_id for query in eval_queries for doc_id in query["positive_doc_ids"]
    }
    covered_positive = eval_positive & set(seen_by_endpoint)
    if len(covered_positive) != 1_608:
        raise ValueError(f"seen positive coverage changed: {len(covered_positive)}")

    heldout_candidates: dict[str, dict] = {}
    for row in unseen_rows:
        exact, function_name = endpoint(row)
        if exact not in seen_by_endpoint and exact not in registry:
            heldout_candidates.setdefault(exact, row | {"function_name": function_name})
    needed = 2 * args.heldout_tools_per_split
    heldout_ids = sorted(heldout_candidates, key=lambda value: (int(value.split(":")[-1]), value))[:needed]
    if len(heldout_ids) != needed:
        raise ValueError("insufficient disjoint held-out tools for internal validation")
    val_ids = set(heldout_ids[: args.heldout_tools_per_split])
    test_ids = set(heldout_ids[args.heldout_tools_per_split :])

    identity = {
        exact: digest(f"toolbench-exact-endpoint:{exact}")
        for exact in set(seen_by_endpoint) | val_ids | test_ids
    }
    tools: list[dict] = []
    for split, ids, source_rows in (
        ("train", set(seen_by_endpoint), seen_by_endpoint),
        ("validation", val_ids, heldout_candidates),
        ("test", test_ids, heldout_candidates),
    ):
        for exact in sorted(ids, key=lambda value: (int(value.split(":")[-1]), value)):
            row = source_rows[exact]
            # The overlapping training tools use exactly the frozen registry
            # document text that will be compiled at formal evaluation time.
            document = str(registry[exact]["text"]) if exact in registry else str(row["api_description"])
            tools.append(
                {
                    "identity_hash": identity[exact],
                    "group_hash": digest(f"toolbench-group:{exact}"),
                    "split": split,
                    "document": document,
                    "tool_name": str(row["function_name"]),
                    "endpoint_name": exact,
                    "source": "toolbench10k-frozen-v2",
                    "parameters": json.loads(row["api_params"]),
                }
            )

    retrieval: list[dict] = []
    for query, targets in sorted(query_targets.items()):
        ordered_targets = sorted(targets, key=lambda value: (int(value.split(":")[-1]), value))
        retrieval.append(
            {
                "query_hash": digest(f"train:{query}"),
                "query": query,
                "split": "train",
                "tool_identity_hashes": [identity[value] for value in ordered_targets],
                "source": "toolbench10k-seen-train",
            }
        )

    first_row: dict[str, dict] = {}
    for row in unseen_rows:
        exact, _ = endpoint(row)
        if exact in val_ids or exact in test_ids:
            first_row.setdefault(exact, row)
    for split, ids in (("validation", val_ids), ("test", test_ids)):
        for exact in sorted(ids, key=lambda value: (int(value.split(":")[-1]), value)):
            query = str(first_row[exact]["query_for_retrieval"])
            retrieval.append(
                {
                    "query_hash": digest(f"{split}:{exact}:{query}"),
                    "query": query,
                    "split": split,
                    "tool_identity_hashes": [identity[exact]],
                    "source": f"toolbench10k-{split}-internal",
                }
            )

    manifest = {
        "kind": "native_meta_registration_toolbench10k_seen_v1",
        "train_source_rows": len(seen_rows),
        "excluded_formal_query_rows": excluded_query_rows,
        "train_episodes": len(query_targets),
        "train_tools": len(seen_by_endpoint),
        "validation_tools": len(val_ids),
        "test_tools": len(test_ids),
        "formal_registry_tools": len(registry),
        "formal_queries": len(eval_queries),
        "formal_positive_tools_covered": len(covered_positive),
        "train_eval_query_overlap": len(set(query_targets) & formal_query_texts),
        "split_policy": "all 3721 seen tools train; 128+128 tools absent from train and formal registry for internal validation/test",
        "document_policy": "formal registry text for overlapping tools",
    }
    if manifest["train_eval_query_overlap"]:
        raise ValueError("formal evaluation queries leaked into meta-registration training")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tools.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in tools),
        encoding="utf-8",
    )
    (args.output_dir / "retrieval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in retrieval),
        encoding="utf-8",
    )
    split_manifest = {
        "token_pools": {
            "train": {"start_inclusive": 0, "end_exclusive": 256},
            "validation": {"start_inclusive": 256, "end_exclusive": 512},
            "test": {"start_inclusive": 512, "end_exclusive": 768},
        }
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
