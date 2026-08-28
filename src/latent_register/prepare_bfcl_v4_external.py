from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from jsonschema import Draft202012Validator

from .data import canonical_target_arguments, canonical_tool_document


CONFIRMATORY_CATEGORIES = ("simple_python",)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
            count += 1
    return count


def _extract_user_text(value: Any) -> str:
    if isinstance(value, str):
        return value.removeprefix("user:").strip()
    if isinstance(value, Mapping):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            return str(value["content"]).strip()
        return "\n".join(
            part for item in value.values() if (part := _extract_user_text(item))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "\n".join(part for item in value if (part := _extract_user_text(item)))
    return ""


def _ground_truth_calls(value: Any) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    values = value if isinstance(value, list) else [value]
    for item in values:
        if not isinstance(item, Mapping):
            continue
        for name, arguments in item.items():
            if isinstance(arguments, Mapping):
                calls.append((str(name), dict(arguments)))
    return calls


def _normalize_schema(value: Any) -> Any:
    """Convert BFCL's Python/Java type labels to validation-only JSON Schema."""
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    normalized = {str(key): _normalize_schema(item) for key, item in value.items()}
    raw_type = normalized.get("type")
    mapping = {
        "dict": "object",
        "HashMap": "object",
        "array": "array",
        "Array": "array",
        "ArrayList": "array",
        "tuple": "array",
        "float": "number",
        "double": "number",
        "long": "integer",
        "String": "string",
        "char": "string",
        "Boolean": "boolean",
        "file": "string",
        "directory": "string",
        "business": "string",
    }
    if raw_type == "any":
        normalized.pop("type", None)
    elif isinstance(raw_type, str) and raw_type in mapping:
        normalized["type"] = mapping[raw_type]
    if normalized.get("type") == "array" and "items" not in normalized:
        normalized["items"] = {}
    return normalized


def _tool_identity(function: Mapping[str, Any]) -> tuple[str, str, str]:
    canonical = {
        "name": str(function.get("name", "")),
        "description": str(function.get("description", "")),
        "parameters": function.get("parameters", {}),
    }
    document = canonical_tool_document(canonical)
    identity = _sha256_bytes(_canonical_json(canonical).encode("utf-8"))
    group = _sha256_bytes(str(canonical["name"]).encode("utf-8"))
    return identity, group, document


def _category_paths(data_root: Path, category: str) -> tuple[Path, Path]:
    stem = f"BFCL_v4_{category}.json"
    return data_root / stem, data_root / "possible_answer" / stem


def prepare_bfcl_v4_external(
    *,
    data_root: str | Path,
    split_manifest_template: str | Path,
    output_dir: str | Path,
    categories: Sequence[str] = CONFIRMATORY_CATEGORIES,
    upstream_commit: str,
) -> dict[str, Any]:
    root = Path(data_root)
    template_path = Path(split_manifest_template)
    output = Path(output_dir)
    prepared = output / "prepared"
    benchmark = output / "benchmark"
    prepared.mkdir(parents=True, exist_ok=True)
    benchmark.mkdir(parents=True, exist_ok=True)
    if not categories:
        raise ValueError("At least one BFCL category is required")

    template = json.loads(template_path.read_text(encoding="utf-8"))
    token_pools = template.get("token_pools")
    if not isinstance(token_pools, Mapping) or "test" not in token_pools:
        raise ValueError("Split manifest template is missing the test token pool")

    tools: dict[str, dict[str, Any]] = {}
    evaluation_rows: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    skip_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    query_hashes: set[str] = set()

    for category in categories:
        data_path, answer_path = _category_paths(root, category)
        for path in (data_path, answer_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing BFCL v4 source: {path}")
            source_files[str(path.relative_to(root))] = {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        answers = {str(row["id"]): row.get("ground_truth", []) for row in _read_jsonl(answer_path)}
        for row in _read_jsonl(data_path):
            example_id = str(row["id"])
            query = _extract_user_text(row.get("question"))
            if not query:
                skip_counts["empty_query"] += 1
                continue
            functions_by_name: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
            for function in row.get("function", []):
                if not isinstance(function, Mapping) or not function.get("name"):
                    continue
                identity, group, document = _tool_identity(function)
                schema = _normalize_schema(function.get("parameters", {}))
                Draft202012Validator.check_schema(schema)
                tools.setdefault(
                    identity,
                    {
                        "identity_hash": identity,
                        "group_hash": group,
                        "split": "test",
                        "document": document,
                        "tool_name": str(function["name"]),
                        "endpoint_name": str(function["name"]),
                        "source": f"bfcl_v4:{category}",
                        "parameters": schema,
                        "token": None,
                    },
                )
                functions_by_name.setdefault(str(function["name"]), []).append(
                    (identity, function)
                )

            calls = _ground_truth_calls(answers.get(example_id, []))
            if len(calls) != 1:
                skip_counts["not_single_reference_call"] += 1
                continue
            target_name, raw_arguments = calls[0]
            candidates = functions_by_name.get(target_name, [])
            if len(candidates) != 1:
                skip_counts["target_document_not_unique"] += 1
                continue
            target_identity, target_function = candidates[0]
            query_hash = _sha256_bytes(f"bfcl_v4:{category}:{example_id}:{query}".encode("utf-8"))
            if query_hash in query_hashes:
                raise ValueError(f"Duplicate BFCL query identity: {example_id}")
            query_hashes.add(query_hash)
            common = {
                "condition": "unseen_tool_unseen_token",
                "example_id": f"bfcl_v4:{category}:{example_id}",
                "query": query,
                "reference_tools": [target_identity],
                "reference_fixed_tokens": [],
                "source_split": "test",
                "external_benchmark": "bfcl_v4",
                "bfcl_category": category,
                "bfcl_source_id": example_id,
                "bfcl_candidate_count": len(functions_by_name),
            }
            evaluation_rows.append({"family": "retrieval", **common})
            evaluation_rows.append(
                {
                    "family": "arguments",
                    **common,
                    "reference_arguments": canonical_target_arguments(raw_arguments),
                    "reference_argument_candidates": raw_arguments,
                    "schema": _normalize_schema(target_function.get("parameters", {})),
                }
            )
            category_counts[category] += 1

    ordered_tools = [tools[identity] for identity in sorted(tools)]
    evaluation_rows.sort(
        key=lambda item: (str(item["family"]), str(item["example_id"]))
    )
    _write_jsonl(prepared / "tools.jsonl", ordered_tools)
    _write_jsonl(benchmark / "benchmark_eval.jsonl", evaluation_rows)

    split_manifest = {
        "version": 1,
        "kind": "bfcl_v4_external_evaluation_only",
        "token_pools": token_pools,
        "counts": {"tools_by_split": {"test": len(ordered_tools)}},
        "audits": {
            "evaluation_only": True,
            "training_rows": 0,
            "benchmark_queries_used_for_registration": 0,
            "benchmark_answers_used_for_registration": 0,
        },
        "token_pool_template": {
            "path": str(template_path.resolve()),
            "sha256": _sha256_file(template_path),
        },
    }
    (prepared / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "version": 1,
        "kind": "bfcl_v4_registry_stress",
        "upstream_repository": "https://github.com/ShishirPatil/gorilla",
        "upstream_commit": upstream_commit,
        "categories": list(categories),
        "confirmatory_categories": [
            value for value in categories if value in CONFIRMATORY_CATEGORIES
        ],
        "diagnostic_categories": [
            value for value in categories if value not in CONFIRMATORY_CATEGORIES
        ],
        "counts": {
            "distinct_tools": len(ordered_tools),
            "single_call_examples": sum(category_counts.values()),
            "evaluation_rows": len(evaluation_rows),
            "examples_by_category": dict(sorted(category_counts.items())),
            "skipped": dict(sorted(skip_counts.items())),
        },
        "sources": dict(sorted(source_files.items())),
        "outputs": {
            "tools": {
                "path": str((prepared / "tools.jsonl").resolve()),
                "sha256": _sha256_file(prepared / "tools.jsonl"),
            },
            "evaluation_rows": {
                "path": str((benchmark / "benchmark_eval.jsonl").resolve()),
                "sha256": _sha256_file(benchmark / "benchmark_eval.jsonl"),
            },
            "split_manifest": {
                "path": str((prepared / "split_manifest.json").resolve()),
                "sha256": _sha256_file(prepared / "split_manifest.json"),
            },
        },
        "audits": {
            "evaluation_only": True,
            "all_tools_test_split": all(row["split"] == "test" for row in ordered_tools),
            "one_reference_tool_per_row": all(
                len(row["reference_tools"]) == 1 for row in evaluation_rows
            ),
            "registration_uses_only_tool_documents": True,
            "official_bfcl_score": False,
        },
    }
    (benchmark / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an evaluation-only BFCL v4 late-registration manifest"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-manifest-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()
    manifest = prepare_bfcl_v4_external(
        data_root=args.data_root,
        split_manifest_template=args.split_manifest_template,
        output_dir=args.output_dir,
        categories=args.category or CONFIRMATORY_CATEGORIES,
        upstream_commit=args.upstream_commit,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
