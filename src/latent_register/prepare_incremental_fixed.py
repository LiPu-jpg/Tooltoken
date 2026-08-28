"""Prepare document-only data for incremental fixed-token registration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .episodic_data import load_prepared_tools
from .prepare_controlled_benchmark import TOOL_TOKEN_PATTERN
from .validate_toolgen_tokenizer import write_json_atomic


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row["identity_hash"])
            if identity in mapping:
                raise ValueError(f"Duplicate identity at {path}:{line_number}")
            mapping[identity] = {
                "split": str(row["split"]),
                "fixed_token": str(row["fixed_token"]),
            }
    return mapping


def _conversation(document: str, token: str) -> dict[str, Any]:
    return {
        "conversations": [
            {"role": "user", "content": document, "loss": False},
            {"role": "assistant", "content": token, "loss": True},
        ]
    }


def _write_json_array(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def prepare_incremental_fixed(
    *,
    prepared_dir: str | Path,
    controlled_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Incremental split must be validation or test")
    prepared = Path(prepared_dir)
    controlled = Path(controlled_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tools_path = prepared / "tools.jsonl"
    mapping_path = controlled / "fixed_tokens.jsonl"
    base_tokens_path = controlled / "virtual_tokens_train.txt"
    incremental_source_path = controlled / f"virtual_tokens_{split}.txt"
    benchmark_manifest_path = controlled / "benchmark_manifest.json"
    for path in (
        tools_path,
        mapping_path,
        base_tokens_path,
        incremental_source_path,
        benchmark_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    tools = load_prepared_tools(tools_path)
    mapping = _read_mapping(mapping_path)
    identities = sorted(
        identity for identity, tool in tools.items() if tool.split == split
    )
    if not identities:
        raise ValueError(f"No tools found for incremental split {split}")
    missing = [identity for identity in identities if identity not in mapping]
    if missing:
        raise ValueError(f"Fixed-token mapping omits incremental tools: {missing[:3]}")
    wrong_split = [
        identity for identity in identities if mapping[identity]["split"] != split
    ]
    if wrong_split:
        raise ValueError(f"Fixed-token mapping split mismatch: {wrong_split[:3]}")

    incremental_tokens = [mapping[identity]["fixed_token"] for identity in identities]
    if len(set(incremental_tokens)) != len(incremental_tokens):
        raise ValueError("Incremental fixed-token strings are not unique")
    if any(TOOL_TOKEN_PATTERN.fullmatch(token) is None for token in incremental_tokens):
        raise ValueError("Incremental token does not use atomic ToolGen syntax")
    source_tokens = incremental_source_path.read_text(encoding="utf-8").splitlines()
    if incremental_tokens != source_tokens:
        raise ValueError("Incremental token order differs from the controlled manifest")

    base_tokens = base_tokens_path.read_text(encoding="utf-8").splitlines()
    if set(base_tokens).intersection(incremental_tokens):
        raise ValueError("Base and incremental fixed-token strings overlap")
    incremental_path = output / "virtual_tokens_incremental.txt"
    combined_path = output / "virtual_tokens_combined.txt"
    incremental_path.write_text(
        "".join(f"{token}\n" for token in incremental_tokens), encoding="utf-8"
    )
    combined_path.write_text(
        "".join(f"{token}\n" for token in (*base_tokens, *incremental_tokens)),
        encoding="utf-8",
    )

    training_path = output / "fixed_incremental_memorization.json"
    rows = [
        _conversation(tools[identity].document, mapping[identity]["fixed_token"])
        for identity in identities
    ]
    _write_json_array(training_path, rows)

    manifest = {
        "kind": "qwen_toolgen_incremental_document_registration",
        "version": 1,
        "split": split,
        "counts": {
            "incremental_tools": len(identities),
            "memorization_records": len(rows),
            "base_tokens": len(base_tokens),
            "combined_tokens": len(base_tokens) + len(incremental_tokens),
        },
        "training_information": ["tool_document", "fixed_token_label"],
        "audits": {
            "evaluation_query_records_used": 0,
            "evaluation_argument_records_used": 0,
            "evaluation_trajectory_records_used": 0,
            "incremental_tokens_unique": True,
            "base_incremental_token_overlap": 0,
        },
        "sources": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for path in (
                tools_path,
                mapping_path,
                base_tokens_path,
                incremental_source_path,
                benchmark_manifest_path,
            )
        },
        "outputs": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for path in (incremental_path, combined_path, training_path)
        },
    }
    write_json_atomic(output / "incremental_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare document-only incremental ToolGen registration data"
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--controlled-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    result = prepare_incremental_fixed(
        prepared_dir=args.prepared_dir,
        controlled_dir=args.controlled_dir,
        output_dir=args.output_dir,
        split=args.split,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
