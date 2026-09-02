#!/usr/bin/env python3
"""Materialize the official G1/I1 test identities used for API-disjoint training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latebound_sequence_sft.data import action_key, sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    g1 = args.official_root / "data/retrieval/G1"
    queries = {
        line.split("\t", 1)[-1].strip()
        for line in (g1 / "test.query.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    docs: dict[str, dict[str, Any]] = {}
    with (g1 / "corpus.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            docid = str(row["docid"])
            document = json.loads(row["document_content"])
            canonical = json.dumps(document, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            previous = docs.setdefault(
                docid,
                {
                    "docid": docid,
                    "action": action_key(document.get("tool_name", ""), document.get("api_name", "")),
                    "document_sha256": digest,
                },
            )
            if previous["document_sha256"] != digest:
                raise ValueError(f"G1 docid {docid} maps to multiple documents")
    test_docids: set[str] = set()
    with (g1 / "qrels.test.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            _, _, docid, rel = line.rstrip("\n").split("\t")
            if rel != "1":
                continue
            if docid not in docs:
                raise ValueError(f"G1 qrel references missing corpus docid {docid}")
            test_docids.add(docid)
    actions = {docs[docid]["action"] for docid in test_docids}
    payload = {
        "kind": "toolbench_g1_test_exclusion_v1",
        "official_root_relative": "data/retrieval/G1",
        "test_query_count": len(queries),
        "test_docid_count": len(test_docids),
        "test_action_count": len(actions),
        "test_queries": sorted(queries),
        "test_docids": sorted(test_docids, key=lambda value: (int(value), value)),
        "test_actions": sorted(actions),
        "query_file_sha256": sha256_file(g1 / "test.query.txt"),
        "qrels_file_sha256": sha256_file(g1 / "qrels.test.tsv"),
        "corpus_file_sha256": sha256_file(g1 / "corpus.tsv"),
        "identity_policy": "action-disjoint exclusion; no train action may match an I1 qrel action",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"i1-exclusion-ready queries={len(queries)} docs={len(test_docids)} actions={len(actions)}")


if __name__ == "__main__":
    main()
