"""Adapt an explicitly declared TRAINING export; never discover test files.

Supply exact aliases and schemas in --tools. Unknown APIs, malformed calls and
unsupported formats stop conversion with a source record id; nothing is skipped.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import ijson

from .toolbench_checkpoint import sha256
from .toolbench_data import compact, load_training_tools, read_jsonl, trajectory_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--input-format", choices=["json-array", "jsonl"], required=True)
    parser.add_argument("--source-format", choices=["toolbench", "toolgen"], required=True)
    parser.add_argument("--replace-source-system-prompt", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tools = load_training_tools(args.tools)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts: Counter = Counter()
    identities: set[str] = set()
    if args.input_format == "json-array":
        handle = args.training_source.open("rb")
        records = ijson.items(handle, "item", use_float=True)
    else:
        handle = None
        records = read_jsonl(args.training_source)
    index = -1
    row = {}
    try:
        with (args.output_dir / "train_trajectories.jsonl").open("x", encoding="utf-8") as output:
            for index, raw in enumerate(records):
                if not isinstance(raw, dict):
                    raise ValueError("Expected a trajectory object")
                row = {**raw, "id": str(raw.get("id", index)), "split": raw.get("split", "train"),
                       "replace_source_system_prompt": args.replace_source_system_prompt}
                if row["id"] in identities:
                    raise ValueError("Duplicate source id")
                identities.add(row["id"])
                steps = trajectory_steps(row, tools, source_format=args.source_format)
                counts.update(step.api_identity for step in steps)
                output.write(compact(row) + "\n")
        if not identities:
            raise ValueError("No training trajectories")
        # Copy only the explicitly supplied training registry, byte-for-byte.
        (args.output_dir / "train_tools.jsonl").write_bytes(args.tools.read_bytes())
        report = {"source_sha256": sha256(args.training_source), "tools_sha256": sha256(args.tools),
                  "source_format": args.source_format, "trajectories": len(identities),
                  "steps": sum(counts.values()), "steps_by_api": dict(counts),
                  "skipped_records": 0, "status": "prepared_training_only"}
        (args.output_dir / "PREPARED.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception as exc:
        report = {"status": "incomplete_do_not_train", "record_index": index,
                  "record_id": row.get("id"), "error": str(exc)}
        (args.output_dir / "FAILED.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    main()
