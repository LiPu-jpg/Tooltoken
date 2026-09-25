"""Strict, causal ToolBench agent training data; never opens evaluation files.

This adapter accepts serial ToolBench function-call messages and the ToolGen
action/document/arguments conversation format. It does not infer calls from prose.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft7Validator, ValidationError

FINISH = "__toolbench_finish__"
FINISH_SCHEMA = {
    "type": "object",
    "properties": {
        "return_type": {"type": "string", "enum": ["give_answer", "give_up_and_restart"]},
        "final_answer": {"type": "string"},
    },
    "required": ["return_type"],
    "additionalProperties": False,
    "allOf": [{"if": {"properties": {"return_type": {"const": "give_answer"}}},
               "then": {"required": ["final_answer"]}}],
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        out: dict = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out

    def invalid(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def json_object(value: Any) -> dict:
    if isinstance(value, str):
        value = strict_json(value)
    if not isinstance(value, dict):
        raise ValueError("ToolBench arguments must be a JSON object, not an STQ array")
    compact(value)  # Reject NaN and unserializable values, without coercing types.
    return value


def _local_refs_only(value: Any) -> None:
    if isinstance(value, dict):
        if "$ref" in value and not str(value["$ref"]).startswith("#"):
            raise ValueError("Only local schema references are supported; no network schema loading")
        for item in value.values():
            _local_refs_only(item)
    elif isinstance(value, list):
        for item in value:
            _local_refs_only(item)


@dataclass(frozen=True)
class ToolSpec:
    api_identity: str
    document: str
    parameters: dict
    aliases: tuple[str, ...] = ()
    executable_contract: dict | None = None

    def __post_init__(self) -> None:
        if not self.api_identity or not self.document.strip():
            raise ValueError("Exact API identity and nonempty document are required")
        if self.parameters.get("type") != "object":
            raise ValueError("Provide the actual object JSON Schema; metadata is not a schema")
        _local_refs_only(self.parameters)
        Draft7Validator.check_schema(self.parameters)
        if self.executable_contract is not None:
            c = self.executable_contract
            expected = hashlib.sha256(compact(self.parameters).encode()).hexdigest()
            if (c.get("kind") != "fixed_top_level_signature_v1"
                    or c.get("api_identity") != self.api_identity
                    or c.get("schema_sha256") != expected
                    or not isinstance(c.get("evidence"), dict)
                    or not c["evidence"].get("source")
                    or len(str(c["evidence"].get("sha256", ""))) != 64):
                raise ValueError("Executable contract requires identity/schema-bound source evidence")
            if self.parameters.get("additionalProperties") not in (None, False) or self.parameters.get("patternProperties"):
                raise ValueError("Fixed contract cannot override explicitly open top-level schema")

    @property
    def registration_document(self) -> str:
        # Always include the complete schema, even when the description lacks it.
        result = self.document + "\nParameters JSON Schema:\n" + compact(self.parameters)
        if self.executable_contract is not None:
            result += "\nExecutable input contract: fixed top-level keys " + compact(sorted(self.parameters.get("properties", {})))
        return result

    @property
    def document_hash(self) -> str:
        return hashlib.sha256(self.registration_document.encode()).hexdigest()

    def validate_arguments(self, arguments: Any) -> dict:
        value = json_object(arguments)
        if self.executable_contract is not None:
            allowed = self.parameters.get("properties", {})
            extra = sorted(set(value) - set(allowed))
            if extra:
                raise ValueError("Arguments violate verified executable contract: " + compact({
                    "undeclared_keys": extra, "allowed_keys": sorted(allowed),
                    "required": self.parameters.get("required", []),
                    "types": {k: v.get("type", "unspecified") for k, v in allowed.items()}}))
        try:
            Draft7Validator(self.parameters).validate(value)
        except ValidationError as exc:
            raise ValueError(f"Arguments violate schema for {self.api_identity}: {exc.message}") from exc
        return value


def finish_tool() -> ToolSpec:
    return ToolSpec(FINISH, "Finish the task. Give the final answer, or give up and restart.",
                    FINISH_SCHEMA, ("Finish", "<<Finish>>"))


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = strict_json(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                yield row


def load_training_tools(path: str | Path) -> dict[str, ToolSpec]:
    return load_tools(path, split="train")


def load_tools(path: str | Path, *, split: str) -> dict[str, ToolSpec]:
    if split not in {"train", "dev", "validation"}:
        raise ValueError("Only explicitly selected train/development inputs are supported")
    result = {FINISH: finish_tool()}
    for row in read_jsonl(path):
        if row.get("split") != split:
            raise ValueError(f"Use a {split}-only registry; refusing cross-split documents")
        tool = ToolSpec(str(row["api_identity"]), row["document"], row["parameters"],
                        tuple(row.get("aliases", [])), row.get("executable_contract"))
        if tool.api_identity in result:
            raise ValueError(f"Duplicate/reserved identity: {tool.api_identity}")
        result[tool.api_identity] = tool
    alias_map(result)
    if len(result) < 2:
        raise ValueError("Empty API training registry")
    return result


def alias_map(tools: dict[str, ToolSpec]) -> dict[str, str]:
    result: dict[str, str] = {}
    for identity, tool in tools.items():
        if identity != tool.api_identity:
            raise ValueError("Registry key must equal exact API identity")
        for alias in (identity, *tool.aliases):
            if not isinstance(alias, str) or not alias:
                raise ValueError("Aliases must be nonempty strings")
            if alias in result and result[alias] != identity:
                raise ValueError(f"Ambiguous tool alias: {alias}")
            result[alias] = identity
    return result


@dataclass(frozen=True)
class AgentStep:
    source_id: str
    step_index: int
    history: tuple[dict, ...]
    thought: str
    api_identity: str
    arguments: dict


def call_history(thought: str, identity: str, arguments: dict) -> list[dict]:
    # This representation is shared by teacher forcing and live execution.
    return [{"type": "call", "thought": thought, "api_identity": identity,
             "arguments": arguments}]


def observation(identity: str, content: Any) -> dict:
    compact(content)
    return {"type": "observation", "api_identity": identity, "content": content}


def trajectory_steps(row: dict, tools: dict[str, ToolSpec], *, source_format: str, split: str = "train") -> list[AgentStep]:
    if split not in {"train", "dev", "validation"} or row.get("split") != split:
        raise ValueError("Trajectory does not match the explicitly selected train/development split")
    if source_format not in {"toolbench", "toolgen"}:
        raise ValueError("Choose the source format explicitly")
    aliases = alias_map(tools)
    raw = row.get("messages", row.get("conversations"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("A nonempty serial trajectory is required")
    messages = []
    for item in raw:
        role = item.get("role", item.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        messages.append({**item, "role": role, "content": item.get("content", item.get("value"))})
    history: list[dict] = []
    steps: list[AgentStep] = []
    pending: str | None = None
    pending_call_id: str | None = None
    thought = ""
    finished = False
    i = 0
    while i < len(messages):
        message = messages[i]
        role, content = message["role"], message["content"]
        if finished:
            raise ValueError("Unexpected messages after Finish")
        if role == "system":
            if history or i != 0:
                raise ValueError("System message must be the first message")
            # The source agent protocol is explicitly replaced by this adapter's
            # protocol. It is never copied with an enumerated gold tool list.
            if not row.get("replace_source_system_prompt", False) and content:
                raise ValueError("Explicit replace_source_system_prompt=true required")
            i += 1
            continue
        if role == "user" and not history:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Missing user query")
            history.append({"type": "user", "content": content})
            i += 1
            continue
        if role in {"tool", "function"}:
            if pending is None:
                raise ValueError("Observation without a preceding call")
            if message.get("name") and aliases.get(message["name"]) != pending:
                raise ValueError("Observation name does not match the preceding call")
            if message.get("tool_call_id") and message["tool_call_id"] != pending_call_id:
                raise ValueError("Observation tool_call_id mismatch")
            history.append(observation(pending, content))
            pending = pending_call_id = None
            i += 1
            continue
        if pending is not None:
            raise ValueError("A serial call must receive its observation before the next decision")
        if not history:
            raise ValueError("Trajectory must start with a user query")
        if source_format == "toolgen":
            if role == "user" and content in {"Generate the action.", "Please generate the action."}:
                i += 1
                continue
            if role != "assistant" or not isinstance(content, str):
                raise ValueError("Unrecognized ToolGen message; refusing to silently drop it")
            if content.strip() not in aliases:
                if content.strip().startswith("<<"):
                    raise ValueError(f"Unknown action token: {content}")
                thought = (thought + "\n" + content).strip()
                i += 1
                continue
            name = content.strip()
            if i + 2 >= len(messages):
                raise ValueError("Incomplete ToolGen action/document/arguments group")
            document, argument_message = messages[i + 1:i + 3]
            if (document["role"] not in {"user", "assistant"}
                    or not isinstance(document["content"], str)
                    or not document["content"].startswith("Please give the input. Here is the documentation:")
                    or argument_message["role"] != "assistant"):
                raise ValueError("Unsupported ToolGen documentation wrapper")
            arguments = argument_message["content"]
            call_id = None
            i += 3
        else:
            if role != "assistant":
                raise ValueError("Only initial user, serial assistant call and tool observations are supported")
            calls = message.get("tool_calls")
            if calls is not None:
                if len(calls) != 1:
                    raise ValueError("Parallel calls require a separate protocol; do not flatten them")
                function = calls[0]["function"]
                call_id = calls[0].get("id")
            else:
                function = message.get("function_call")
                call_id = None
            if not isinstance(function, dict):
                raise ValueError("Expected explicit function_call/tool_calls including Finish")
            name, arguments = function["name"], function["arguments"]
            thought = content or ""
            if not isinstance(thought, str):
                raise ValueError("Thought must be text")
            i += 1
        if name not in aliases:
            raise ValueError(f"Unknown or held-out API: {name}")
        identity = aliases[name]
        parsed = tools[identity].validate_arguments(arguments)
        steps.append(AgentStep(str(row["id"]), len(steps), tuple(history), thought, identity, parsed))
        history.extend(call_history(thought, identity, parsed))
        thought = ""
        finished = identity == FINISH
        if not finished:
            pending, pending_call_id = identity, call_id
    if not finished or not steps:
        raise ValueError("Training trajectory must end with an explicit, labeled Finish")
    return steps


def load_training_steps(path: str | Path, tools: dict[str, ToolSpec], *, source_format: str) -> list[AgentStep]:
    return load_steps(path, tools, source_format=source_format, split="train")


def load_steps(path: str | Path, tools: dict[str, ToolSpec], *, source_format: str, split: str) -> list[AgentStep]:
    result: list[AgentStep] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        identity = str(row["id"])
        if identity in seen:
            raise ValueError(f"Duplicate trajectory id: {identity}")
        seen.add(identity)
        result.extend(trajectory_steps(row, tools, source_format=source_format, split=split))
    if not result:
        raise ValueError("No training steps")
    return result
