#!/usr/bin/env python3
"""Prepare a public G1/I1 query+registry root and sealed qrels."""

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
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--sealed-output", type=Path, required=True)
    args = parser.parse_args()
    if args.public_output.exists() or args.sealed_output.exists():
        raise FileExistsError("evaluation output already exists")
    g1 = args.official_root / "data/retrieval/G1"
    public = args.public_output
    sealed = args.sealed_output
    (public / "queries").mkdir(parents=True)
    sealed.mkdir(parents=True)

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
                    "document": document,
                    "action": action_key(document.get("tool_name", ""), document.get("api_name", "")),
                    "document_sha256": digest,
                },
            )
            if previous["document_sha256"] != digest:
                raise ValueError(f"G1 docid {docid} maps to multiple documents")
    if len(docs) != 10439:
        raise ValueError(f"Expected 10,439 unique G1 documents, found {len(docs)}")

    registry_path = public / "registry.jsonl"
    with registry_path.open("w", encoding="utf-8") as handle:
        for docid in sorted(docs, key=lambda value: (int(value), value)):
            row = docs[docid]
            identity = hashlib.sha256(
                f"{docid}\n{json.dumps(row['document'], ensure_ascii=False, sort_keys=True)}".encode(
                    "utf-8"
                )
            ).hexdigest()
            handle.write(
                json.dumps(
                    {
                        "docid": docid,
                        "identity_hash": identity,
                        "action": row["action"],
                        "document": row["document"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    query_rows: list[dict[str, str]] = []
    for line in (g1 / "test.query.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            qid, query = line.split("\t", 1)
            query_rows.append({"qid": str(qid), "query": query})
    query_path = public / "queries.jsonl"
    with query_path.open("w", encoding="utf-8") as handle:
        for row in query_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    qrels: dict[str, list[dict[str, Any]]] = {}
    for line in (g1 / "qrels.test.tsv").read_text(encoding="utf-8").splitlines():
        qid, _, docid, rel = line.split("\t")
        if rel != "1":
            continue
        row = docs.get(str(docid))
        if row is None:
            raise ValueError(f"qrel references missing G1 docid {docid}")
        identity = hashlib.sha256(
            f"{docid}\n{json.dumps(row['document'], ensure_ascii=False, sort_keys=True)}".encode(
                "utf-8"
            )
        ).hexdigest()
        qrels.setdefault(str(qid), []).append(
            {"docid": str(docid), "identity_hash": identity, "action": row["action"]}
        )
    sealed_path = sealed / "qrels.jsonl"
    with sealed_path.open("w", encoding="utf-8") as handle:
        for qid in sorted(qrels, key=lambda value: int(value)):
            rows = qrels[qid]
            handle.write(
                json.dumps(
                    {
                        "qid": qid,
                        "exact_identity_hashes": sorted({row["identity_hash"] for row in rows}),
                        "actions": sorted({row["action"] for row in rows}),
                        "qrel_rows": rows,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    sealed_manifest = {
        "qrels_sha256": sha256_file(sealed_path),
        "source_qrels_sha256": sha256_file(g1 / "qrels.test.tsv"),
    }
    (sealed / "sealed_manifest.json").write_text(
        json.dumps(sealed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "kind": "native_latebound_official_g1_zero_step_evaluation",
        "passed": True,
        "candidate_documents": len(docs),
        "query_rows": len(query_rows),
        "qrel_queries": len(qrels),
        "qrel_rows": sum(len(rows) for rows in qrels.values()),
        "unique_qrel_documents": len({row["docid"] for rows in qrels.values() for row in rows}),
        "unique_qrel_actions": len({row["action"] for rows in qrels.values() for row in rows}),
        "registry_sha256": sha256_file(registry_path),
        "queries_sha256": sha256_file(query_path),
        "sealed_manifest_sha256": sha256_file(sealed / "sealed_manifest.json"),
        "qrels_read_by_score": False,
        "registration_optimizer_steps": 0,
    }
    (public / "evaluation_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256((public / "evaluation_manifest.json").read_bytes()).hexdigest()
    (public / "COMPLETE").write_text(digest + "\n", encoding="ascii")
    (sealed / "COMPLETE").write_text(
        hashlib.sha256((sealed / "sealed_manifest.json").read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    print(
        f"i1-evaluation-ready candidates={len(docs)} queries={len(query_rows)} "
        f"qrel_queries={len(qrels)} qrel_docs={metadata['unique_qrel_documents']}"
    )


if __name__ == "__main__":
    main()
