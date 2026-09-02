#!/usr/bin/env python3
"""Materialize the immutable STQ seen split into the v19 Native schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_ARTIFACT = "7af73147532dde8693d05a0b6e06ac1862fe13364e2d6804cac20e3b7d1ce16b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.artifact_root
    candidates = read_jsonl(root / "train_model" / "candidates.jsonl")
    rows = read_jsonl(root / "train_model" / "train.jsonl")
    unseen = read_jsonl(root / "unseen_model" / "candidates.jsonl")
    queries = read_jsonl(root / "unseen_model" / "queries.jsonl")
    if (len(candidates), len(rows), len(unseen), len(queries)) != (999, 10483, 837, 1066):
        raise ValueError("fixed STQ counts changed")
    seen = {row["identity"] for row in candidates}
    unseen_ids = {row["identity"] for row in unseen}
    if seen & unseen_ids:
        raise ValueError("seen/unseen exact identity overlap is non-zero")
    by_identity = {row["identity"]: str(index) for index, row in enumerate(candidates)}
    documents = [
        {
            "docid": str(index),
            "action": row["identity"],
            "document": row["document_text"],
        }
        for index, row in enumerate(candidates)
    ]
    examples = [
        {
            "query": row["query"],
            "document_id": by_identity[row["target_identity"]],
            "positive_document_count": 1,
        }
        for row in rows
    ]
    payload = {
        "kind": "native_latebound_standard_sequence_sft_manifest",
        "version": 3,
        "source_artifact_sha256": EXPECTED_ARTIFACT,
        "source_artifact_root": str(root),
        "source_files_sha256": {
            "train_candidates": sha256(root / "train_model" / "candidates.jsonl"),
            "train_rows": sha256(root / "train_model" / "train.jsonl"),
            "unseen_candidates": sha256(root / "unseen_model" / "candidates.jsonl"),
            "unseen_queries": sha256(root / "unseen_model" / "queries.jsonl"),
        },
        "candidate_count": len(documents),
        "exact_identity_count": len(documents),
        "example_count": len(examples),
        "unseen_candidate_count": len(unseen),
        "unseen_query_count": len(queries),
        "seen_unseen_exact_identity_overlap": 0,
        "document_selection_policy": "one_exact_document_per_seen_identity",
        "examples": examples,
        "documents": documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(args.output)
    args.output.with_name("manifest.sha256").write_text(digest + "  " + args.output.name + "\n", encoding="ascii")
    args.output.with_name("COMPLETE").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"passed": True, "manifest_sha256": digest, "train_rows": len(rows), "train_tools": len(documents)}, sort_keys=True))


if __name__ == "__main__":
    main()
