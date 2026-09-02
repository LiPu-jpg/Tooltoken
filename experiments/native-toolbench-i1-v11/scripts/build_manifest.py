#!/usr/bin/env python3
"""Build the hash-bound, qrels-blind sequence-SFT manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latebound_sequence_sft.data import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--atomic-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    query_files = sorted(args.query_root.glob("G[123]/test_G*_*.query.txt"))
    if not query_files:
        raise SystemExit(f"No official query files under {args.query_root}")
    payload = build_manifest(
        args.retrieval,
        args.corpus,
        query_files,
        args.output,
        atomic_map_path=args.atomic_map,
    )
    print(
        "manifest-ready "
        f"examples={payload['example_count']} "
        f"candidates={payload['candidate_count']} "
        f"excluded_queries={payload['evaluation_query_count']} "
        f"excluded_rows={payload['excluded_query_rows']}"
    )


if __name__ == "__main__":
    main()
