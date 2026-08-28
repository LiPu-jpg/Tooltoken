from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ToolExample:
    example_id: str
    query: str
    tool_name: str
    tool_document: str
    target_arguments: Any = None
    schema_arguments: Any = None


def tool_registry_key(example: ToolExample, identity: str) -> str:
    """Return the registry identity without changing the human-readable name."""
    if identity == "name":
        return example.tool_name
    if identity == "document":
        digest = hashlib.sha256(example.tool_document.encode("utf-8")).hexdigest()
        return f"{example.tool_name}::{digest}"
    raise ValueError(f"Unknown tool identity: {identity}")


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def _extract_user_text(value: Any) -> str:
    if isinstance(value, str):
        return value.removeprefix("user:").strip()
    if isinstance(value, dict):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            return value["content"].strip()
        parts = [_extract_user_text(item) for item in value.values()]
        return "\n".join(part for part in parts if part)
    if isinstance(value, list):
        parts = [_extract_user_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def canonical_tool_document(function: dict[str, Any]) -> str:
    name = function.get("name", "")
    description = function.get("description", "")
    parameters = json.dumps(
        function.get("parameters", {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"Name: {name}\nDescription: {description}\nParameters: {parameters}"


def canonical_target_arguments(arguments: Any) -> Any:
    """Choose one executable argument object from BFCL possible-answer values."""
    if not isinstance(arguments, dict):
        return arguments
    canonical: dict[str, Any] = {}
    for name, candidates in arguments.items():
        if not isinstance(candidates, list):
            canonical[name] = candidates
            continue
        # BFCL uses an empty string as the alternative for omitting an optional
        # argument. Prefer omission over materializing a tool default.
        if "" in candidates or not candidates:
            continue
        canonical[name] = candidates[0]
    return canonical


def schema_target_arguments(function: dict[str, Any]) -> dict[str, None]:
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    if not isinstance(properties, dict):
        return {}
    return {str(name): None for name in properties}


def load_function_call_jsonl(
    data_path: str | Path,
    answer_path: str | Path,
) -> list[ToolExample]:
    """Load ACEBench/BFCL-style JSONL and retain single-tool targets."""
    answers: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(answer_path):
        ground_truth = row.get("ground_truth", row.get("answer", {}))
        if isinstance(ground_truth, dict):
            answers[str(row["id"])] = ground_truth
        elif isinstance(ground_truth, list):
            merged: dict[str, Any] = {}
            for item in ground_truth:
                if isinstance(item, dict):
                    merged.update(item)
            answers[str(row["id"])] = merged

    examples: list[ToolExample] = []
    for row in _read_jsonl(data_path):
        example_id = str(row["id"])
        answer = answers.get(example_id, {})
        target_names = list(answer)
        if len(target_names) != 1:
            continue
        functions = {item.get("name"): item for item in row.get("function", [])}
        target_name = target_names[0]
        target = functions.get(target_name)
        query = _extract_user_text(row.get("question"))
        if not target or not query:
            continue
        examples.append(
            ToolExample(
                example_id=example_id,
                query=query,
                tool_name=target_name,
                tool_document=canonical_tool_document(target),
                target_arguments=canonical_target_arguments(answer[target_name]),
                schema_arguments=schema_target_arguments(target),
            )
        )
    return examples


def stable_tool_split(
    examples: list[ToolExample],
    eval_ratio: float,
    seed: int,
) -> tuple[list[ToolExample], list[ToolExample]]:
    """Split by tool identity so no registered test tool appears in training."""
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")

    tool_names = sorted({item.tool_name for item in examples})
    if len(tool_names) < 2:
        raise ValueError("At least two distinct tools are required")

    ranked = sorted(
        tool_names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest(),
    )
    eval_count = min(len(ranked) - 1, max(1, round(len(ranked) * eval_ratio)))
    eval_tools = set(ranked[:eval_count])
    train = [item for item in examples if item.tool_name not in eval_tools]
    evaluation = [item for item in examples if item.tool_name in eval_tools]
    return train, evaluation
