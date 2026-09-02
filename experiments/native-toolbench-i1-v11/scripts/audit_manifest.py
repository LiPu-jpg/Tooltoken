#!/usr/bin/env python3
"""Audit the sequence-SFT manifest without reading any qrels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latebound_sequence_sft.data import evaluation_queries, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--atomic-map", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("kind") != "native_latebound_standard_sequence_sft_manifest":
        raise ValueError("Unexpected manifest kind")
    if manifest.get("retrieval_sha256") != sha256_file(args.retrieval):
        raise ValueError("Retrieval hash does not match manifest")
    if manifest.get("corpus_sha256") != sha256_file(args.corpus):
        raise ValueError("Corpus hash does not match manifest")
    if manifest.get("atomic_map_sha256") != sha256_file(args.atomic_map):
        raise ValueError("Atomic-map hash does not match manifest")
    docs = {str(row["docid"]): row for row in manifest["documents"]}
    examples = manifest["examples"]
    if len(docs) != manifest["exact_identity_count"]:
        raise ValueError("Exact document identity count is inconsistent")
    if any(str(row["document_id"]) not in docs for row in examples):
        raise ValueError("Example references an unknown exact document")
    evaluation = evaluation_queries(
        sorted(args.query_root.glob("G[123]/test_G*_*.query.txt"))
    )
    overlap = sorted({str(row["query"]) for row in examples}.intersection(evaluation))
    if overlap:
        raise ValueError(f"Training/evaluation query overlap: {len(overlap)}")
    action_weights: defaultdict[str, float] = defaultdict(float)
    action_docs: defaultdict[str, set[str]] = defaultdict(set)
    for row in examples:
        action = str(row["action"])
        count = int(row["positive_document_count"])
        if count < 1 or abs(float(row.get("loss_weight", 1.0 / count)) - 1.0 / count) > 1e-9:
            raise ValueError("Invalid inverse-count loss weight")
        action_weights[action] += 1.0 / count
        action_docs[action].add(str(row["document_id"]))
    if any(abs(value - round(value)) > 1e-9 for value in action_weights.values()):
        raise ValueError("Expanded action weights do not sum to whole examples")
    result = {
        "kind": "native_latebound_standard_sequence_sft_manifest_audit",
        "passed": True,
        "manifest_sha256": sha256_file(args.manifest),
        "retrieval_sha256": sha256_file(args.retrieval),
        "corpus_sha256": sha256_file(args.corpus),
        "atomic_map_sha256": sha256_file(args.atomic_map),
        "candidate_count": len(docs),
        "corpus_action_count": len({str(row["action"]) for row in docs.values()}),
        "example_count": len(examples),
        "unique_train_queries": len({str(row["query"]) for row in examples}),
        "unique_train_documents": len({str(row["document_id"]) for row in examples}),
        "evaluation_query_count": len(evaluation),
        "train_eval_query_overlap": len(overlap),
        "duplicate_action_count": sum(len(value) > 1 for value in action_docs.values()),
        "qrels_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    complete = args.output.with_name("COMPLETE")
    complete.write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"audit": str(args.output), "sha256": digest, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
