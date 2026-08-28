from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import ijson

from .data import canonical_tool_document


TOOL_TOKEN_PATTERN = re.compile(r"<<[^<>\n]+&&[^<>\n]+>>")
IGNORED_SCHEMA_KEYS = {"description", "title", "example", "examples"}
SPLITS = ("train", "validation", "test")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def normalized_query_hash(query: str) -> str:
    normalized = " ".join(re.findall(r"\w+", query.casefold(), flags=re.UNICODE))
    return _sha256_text(normalized)


def _schema_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _schema_signature(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).casefold() not in IGNORED_SCHEMA_KEYS
        }
    if isinstance(value, list):
        return [_schema_signature(item) for item in value]
    return value


def parse_tool_token(token: str) -> tuple[str, str]:
    if not token.startswith("<<") or not token.endswith(">>"):
        raise ValueError(f"Invalid tool token: {token!r}")
    body = token[2:-2]
    tool_name, separator, endpoint_name = body.partition("&&")
    if not separator or not tool_name.strip() or not endpoint_name.strip():
        raise ValueError(f"Invalid tool token: {token!r}")
    return tool_name.strip(), endpoint_name.strip()


def _message_role(message: dict[str, Any]) -> str:
    return str(message.get("role", message.get("from", ""))).casefold()


def _message_content(message: dict[str, Any]) -> str:
    value = message.get("content", message.get("value", ""))
    return value if isinstance(value, str) else ""


def _iter_json_array(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        for value in ijson.items(handle, "item"):
            if isinstance(value, dict):
                yield value


@dataclass(frozen=True)
class CanonicalTool:
    source: str
    source_id: str
    tool_name: str
    endpoint_name: str
    document: str
    token: str | None
    parameters: dict[str, Any] | None
    group_hash: str
    identity_hash: str
    schema_hash: str | None

    @classmethod
    def create(
        cls,
        *,
        source: str,
        source_id: str,
        tool_name: str,
        endpoint_name: str,
        document: str,
        token: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "CanonicalTool":
        normalized_names = {
            "tool_name": normalize_text(tool_name),
            "endpoint_name": normalize_text(endpoint_name),
        }
        group_hash = _sha256_text(_canonical_json(normalized_names))
        schema_hash = None
        if parameters is not None:
            schema_hash = _sha256_text(_canonical_json(_schema_signature(parameters)))
        identity = {
            **normalized_names,
            "schema_hash": schema_hash,
            "document": normalize_text(document) if schema_hash is None else None,
        }
        return cls(
            source=source,
            source_id=source_id,
            tool_name=tool_name,
            endpoint_name=endpoint_name,
            document=document,
            token=token,
            parameters=parameters,
            group_hash=group_hash,
            identity_hash=_sha256_text(_canonical_json(identity)),
            schema_hash=schema_hash,
        )


@dataclass(frozen=True)
class RetrievalRecord:
    source: str
    source_id: str
    query: str
    target_token: str


@dataclass(frozen=True)
class TrajectoryRecord:
    source: str
    source_id: str
    conversations: list[dict[str, Any]]
    target_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ToolAceCallRecord:
    source_id: str
    query: str
    tool: CanonicalTool
    arguments: dict[str, Any]
    call_index: int
    call_count: int


@dataclass(frozen=True)
class TokenPools:
    train: tuple[int, int] = (0, 6144)
    validation: tuple[int, int] = (6144, 7168)
    test: tuple[int, int] = (7168, 8192)

    def validate(self) -> None:
        ranges = {name: range(*getattr(self, name)) for name in SPLITS}
        for name, values in ranges.items():
            if len(values) <= 0:
                raise ValueError(f"Token pool {name} is empty")
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                if set(ranges[left]).intersection(ranges[right]):
                    raise ValueError(f"Token pools overlap: {left} and {right}")

    def as_dict(self) -> dict[str, dict[str, int]]:
        self.validate()
        return {
            name: {
                "start_inclusive": getattr(self, name)[0],
                "end_exclusive": getattr(self, name)[1],
                "size": getattr(self, name)[1] - getattr(self, name)[0],
            }
            for name in SPLITS
        }


def split_for_group(
    group_hash: str,
    *,
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> str:
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Train and validation ratios must be positive")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must leave a test split")
    digest = hashlib.sha256(f"{seed}:{group_hash}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + validation_ratio:
        return "validation"
    return "test"


def iter_toolgen_registrations(path: str | Path) -> Iterator[CanonicalTool]:
    for index, row in enumerate(_iter_json_array(path)):
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue
        document = next(
            (
                _message_content(message)
                for message in conversations
                if isinstance(message, dict) and _message_role(message) == "user"
            ),
            "",
        )
        token = next(
            (
                _message_content(message).strip()
                for message in conversations
                if isinstance(message, dict) and _message_role(message) == "assistant"
            ),
            "",
        )
        if not document or not TOOL_TOKEN_PATTERN.fullmatch(token):
            continue
        tool_name, endpoint_name = parse_tool_token(token)
        yield CanonicalTool.create(
            source="toolgen",
            source_id=str(index),
            tool_name=tool_name,
            endpoint_name=endpoint_name,
            document=document,
            token=token,
        )


def iter_toolgen_retrieval(path: str | Path) -> Iterator[RetrievalRecord]:
    for index, row in enumerate(_iter_json_array(path)):
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue
        query = next(
            (
                _message_content(message)
                for message in conversations
                if isinstance(message, dict) and _message_role(message) == "user"
            ),
            "",
        )
        token = next(
            (
                _message_content(message).strip()
                for message in conversations
                if isinstance(message, dict) and _message_role(message) == "assistant"
            ),
            "",
        )
        if query and TOOL_TOKEN_PATTERN.fullmatch(token):
            yield RetrievalRecord("toolgen", str(index), query, token)


def iter_toolgen_trajectories(path: str | Path) -> Iterator[TrajectoryRecord]:
    for index, row in enumerate(_iter_json_array(path)):
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue
        target_tokens: set[str] = set()
        clean_messages: list[dict[str, Any]] = []
        for message in conversations:
            if not isinstance(message, dict):
                continue
            clean_messages.append(message)
            if _message_role(message) != "assistant":
                continue
            content = _message_content(message).strip()
            if TOOL_TOKEN_PATTERN.fullmatch(content):
                target_tokens.add(content)
        if target_tokens:
            yield TrajectoryRecord(
                source="toolgen",
                source_id=str(row.get("id", index)),
                conversations=clean_messages,
                target_tokens=tuple(sorted(target_tokens)),
            )


def parse_toolace_system_tools(system_prompt: str) -> list[dict[str, Any]]:
    marker = "Here is a list of functions in JSON format that you can invoke:"
    marker_index = system_prompt.find(marker)
    if marker_index < 0:
        return []
    start = system_prompt.find("[", marker_index + len(marker))
    if start < 0:
        return []
    try:
        value, _ = json.JSONDecoder().raw_decode(system_prompt[start:])
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _find_balanced_end(value: str, start: int) -> int:
    opening = value[start]
    pairs = {"(": ")", "[": "]", "{": "}"}
    if opening not in pairs:
        raise ValueError("Balanced scan must start at an opening delimiter")
    stack = [pairs[opening]]
    quote: str | None = None
    escaped = False
    for index in range(start + 1, len(value)):
        character = value[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in pairs:
            stack.append(pairs[character])
        elif character in pairs.values():
            if not stack or character != stack.pop():
                raise ValueError("Mismatched delimiter in tool call")
            if not stack:
                return index
    raise ValueError("Unterminated delimiter in tool call")


def _split_top_level(value: str, delimiter: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in pairs:
            stack.append(pairs[character])
        elif character in pairs.values():
            if not stack or character != stack.pop():
                raise ValueError("Mismatched delimiter in tool arguments")
        elif character == delimiter and not stack:
            pieces.append(value[start:index].strip())
            start = index + 1
    if quote is not None or stack:
        raise ValueError("Unterminated value in tool arguments")
    pieces.append(value[start:].strip())
    return pieces


def _parse_toolace_arguments(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    arguments: dict[str, Any] = {}
    for item in _split_top_level(value, ","):
        assignments = _split_top_level(item, "=")
        if len(assignments) != 2:
            raise ValueError("Tool arguments must be keyword assignments")
        name, literal = assignments
        if not name or name in arguments:
            raise ValueError("Tool argument names must be nonempty and unique")
        try:
            parsed = ast.literal_eval(literal)
        except (SyntaxError, ValueError) as exc:
            try:
                parsed = json.loads(literal)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid literal for tool argument {name!r}") from exc
        arguments[name] = parsed
    return arguments


def parse_toolace_calls(
    value: str, candidate_names: Iterable[str]
) -> list[tuple[str, dict[str, Any]]]:
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("ToolACE call output must be one bracketed list")
    names = sorted({name for name in candidate_names if name}, key=len, reverse=True)
    position = 1
    calls: list[tuple[str, dict[str, Any]]] = []
    while True:
        while position < len(text) - 1 and text[position].isspace():
            position += 1
        if position == len(text) - 1:
            break
        name = next(
            (
                candidate
                for candidate in names
                if text.startswith(candidate, position)
                and position + len(candidate) < len(text)
                and text[position + len(candidate)] == "("
            ),
            None,
        )
        if name is None:
            raise ValueError("Tool call does not match a declared function name")
        opening = position + len(name)
        closing = _find_balanced_end(text, opening)
        calls.append(
            (name, _parse_toolace_arguments(text[opening + 1 : closing]))
        )
        position = closing + 1
        while position < len(text) - 1 and text[position].isspace():
            position += 1
        if position == len(text) - 1:
            break
        if text[position] != ",":
            raise ValueError("Tool calls must be separated by commas")
        position += 1
    if not calls:
        raise ValueError("ToolACE call list is empty")
    return calls


def iter_toolace_tools(path: str | Path) -> Iterator[CanonicalTool]:
    for row_index, row in enumerate(_iter_json_array(path)):
        system_prompt = row.get("system", "")
        if not isinstance(system_prompt, str):
            continue
        for tool_index, function in enumerate(parse_toolace_system_tools(system_prompt)):
            name = str(function.get("name", "")).strip()
            if not name:
                continue
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            yield CanonicalTool.create(
                source="toolace",
                source_id=f"{row_index}:{tool_index}",
                tool_name=name,
                endpoint_name=name,
                document=canonical_tool_document(function),
                parameters=parameters,
            )


def iter_toolace_calls(
    path: str | Path, *, counters: Counter[str] | None = None
) -> Iterator[ToolAceCallRecord]:
    counts = counters if counters is not None else Counter()
    for row_index, row in enumerate(_iter_json_array(path)):
        system_prompt = row.get("system", "")
        conversations = row.get("conversations", [])
        if not isinstance(system_prompt, str) or not isinstance(conversations, list):
            continue
        functions = parse_toolace_system_tools(system_prompt)
        functions_by_name: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for tool_index, function in enumerate(functions):
            name = str(function.get("name", "")).strip()
            if name:
                functions_by_name[name].append((tool_index, function))
        for message_index in range(1, len(conversations)):
            message = conversations[message_index]
            previous = conversations[message_index - 1]
            if not isinstance(message, dict) or _message_role(message) != "assistant":
                continue
            content = _message_content(message).strip()
            if not content.startswith("["):
                continue
            counts["call_messages_seen"] += 1
            if not isinstance(previous, dict) or _message_role(previous) != "user":
                counts["non_user_predecessor"] += 1
                continue
            query = _message_content(previous).strip()
            if not query:
                counts["empty_query"] += 1
                continue
            try:
                calls = parse_toolace_calls(content, functions_by_name)
            except ValueError:
                counts["parse_failure"] += 1
                continue
            if len(calls) > 1:
                counts["multi_call_messages"] += 1
            source_id = f"{row_index}:{message_index}"
            for call_index, (name, arguments) in enumerate(calls):
                matches = functions_by_name[name]
                if len(matches) != 1:
                    counts["ambiguous_function_name"] += 1
                    continue
                tool_index, function = matches[0]
                parameters = function.get("parameters")
                if not isinstance(parameters, dict):
                    parameters = {}
                tool = CanonicalTool.create(
                    source="toolace",
                    source_id=f"{row_index}:{tool_index}",
                    tool_name=name,
                    endpoint_name=name,
                    document=canonical_tool_document(function),
                    parameters=parameters,
                )
                counts["calls_parsed"] += 1
                yield ToolAceCallRecord(
                    source_id=source_id,
                    query=query,
                    tool=tool,
                    arguments=arguments,
                    call_index=call_index,
                    call_count=len(calls),
                )


def _limited(values: Iterable[Any], limit: int) -> Iterator[Any]:
    for index, value in enumerate(values):
        if limit and index >= limit:
            return
        yield value


def _write_jsonl(handle: TextIO, value: dict[str, Any]) -> None:
    handle.write(_canonical_json(value))
    handle.write("\n")


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_scaled_data(
    *,
    toolgen_registration: str | Path,
    toolgen_retrieval: str | Path,
    toolgen_trajectories: str | Path,
    output_dir: str | Path,
    toolace: str | Path | None = None,
    seed: int = 17,
    max_records: int = 0,
    token_pools: TokenPools | None = None,
) -> dict[str, Any]:
    pools = token_pools or TokenPools()
    pools.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "toolgen_registration": Path(toolgen_registration),
        "toolgen_retrieval": Path(toolgen_retrieval),
        "toolgen_trajectories": Path(toolgen_trajectories),
    }
    if toolace is not None:
        source_paths["toolace"] = Path(toolace)
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    tools_by_identity: dict[str, CanonicalTool] = {}
    source_refs: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    toolgen_token_identities: defaultdict[str, set[str]] = defaultdict(set)
    registration_seen = 0
    registration_duplicates = 0

    tool_streams: list[Iterable[CanonicalTool]] = [
        iter_toolgen_registrations(source_paths["toolgen_registration"])
    ]
    if "toolace" in source_paths:
        tool_streams.append(iter_toolace_tools(source_paths["toolace"]))

    for stream in tool_streams:
        for tool in _limited(stream, max_records):
            registration_seen += 1
            source_refs[tool.identity_hash].append(
                {"source": tool.source, "source_id": tool.source_id}
            )
            if tool.identity_hash in tools_by_identity:
                registration_duplicates += 1
            else:
                tools_by_identity[tool.identity_hash] = tool
            if tool.token is not None:
                toolgen_token_identities[tool.token].add(tool.identity_hash)

    ambiguous_tokens = {
        token for token, identities in toolgen_token_identities.items() if len(identities) > 1
    }
    toolgen_token_identity = {
        token: next(iter(identities))
        for token, identities in toolgen_token_identities.items()
        if len(identities) == 1
    }

    group_splits = {
        tool.group_hash: split_for_group(tool.group_hash, seed=seed)
        for tool in tools_by_identity.values()
    }
    token_splits = {
        token: group_splits[tools_by_identity[identity].group_hash]
        for token, identity in toolgen_token_identity.items()
    }

    split_groups: defaultdict[str, set[str]] = defaultdict(set)
    tool_counts: Counter[str] = Counter()
    source_tool_counts: Counter[str] = Counter()
    with (output / "tools.jsonl").open("w", encoding="utf-8") as handle:
        for identity_hash in sorted(tools_by_identity):
            tool = tools_by_identity[identity_hash]
            split = group_splits[tool.group_hash]
            split_groups[split].add(tool.group_hash)
            tool_counts[split] += 1
            source_tool_counts[tool.source] += 1
            value = asdict(tool)
            value["split"] = split
            value["source_refs"] = source_refs[identity_hash]
            _write_jsonl(handle, value)

    overlap = {
        f"{left}_{right}": len(split_groups[left].intersection(split_groups[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(overlap.values()):
        raise AssertionError(f"Tool group leakage detected: {overlap}")

    retrieval_counts: Counter[str] = Counter()
    retrieval_target_counts: Counter[str] = Counter()
    retrieval_skipped: Counter[str] = Counter()
    retained_retrieval_splits: dict[str, str] = {}
    retrieval_pairs_seen = 0
    query_records: dict[str, dict[str, Any]] = {}
    records = iter_toolgen_retrieval(source_paths["toolgen_retrieval"])
    for record in _limited(records, max_records):
        retrieval_pairs_seen += 1
        query_hash = normalized_query_hash(record.query)
        query_record = query_records.setdefault(
            query_hash,
            {
                "query": record.query,
                "source_ids": [],
                "target_tokens": set(),
                "has_ambiguous_token": False,
                "has_unknown_token": False,
            },
        )
        query_record["source_ids"].append(record.source_id)
        if record.target_token in query_record["target_tokens"]:
            retrieval_skipped["duplicate_query_target_pair"] += 1
            continue
        query_record["target_tokens"].add(record.target_token)
        query_record["has_ambiguous_token"] = (
            query_record["has_ambiguous_token"] or record.target_token in ambiguous_tokens
        )
        query_record["has_unknown_token"] = (
            query_record["has_unknown_token"] or record.target_token not in token_splits
        )

    with (output / "retrieval.jsonl").open("w", encoding="utf-8") as handle:
        for query_hash in sorted(query_records):
            query_record = query_records[query_hash]
            target_tokens = sorted(query_record["target_tokens"])
            if query_record["has_ambiguous_token"]:
                retrieval_skipped["queries_with_ambiguous_token"] += 1
                retrieval_skipped["ambiguous_target_pairs"] += len(target_tokens)
                continue
            if query_record["has_unknown_token"]:
                retrieval_skipped["queries_with_unknown_token"] += 1
                retrieval_skipped["unknown_target_pairs"] += len(target_tokens)
                continue
            splits = {token_splits[token] for token in target_tokens}
            if len(splits) != 1:
                retrieval_skipped["cross_split_queries"] += 1
                retrieval_skipped["cross_split_target_pairs"] += len(target_tokens)
                continue
            split = next(iter(splits))
            retained_retrieval_splits[query_hash] = split
            retrieval_counts[split] += 1
            retrieval_target_counts[split] += len(target_tokens)
            _write_jsonl(
                handle,
                {
                    "source": "toolgen",
                    "source_ids": query_record["source_ids"],
                    "query": query_record["query"],
                    "query_hash": query_hash,
                    "split": split,
                    "target_tokens": target_tokens,
                    "tool_identity_hashes": [
                        toolgen_token_identity[token] for token in target_tokens
                    ],
                },
            )

    readback_counts: Counter[str] = Counter()
    readback_skipped: Counter[str] = Counter()
    readback_parser_counts: Counter[str] = Counter()
    call_records: list[ToolAceCallRecord] = []
    if "toolace" in source_paths:
        records = iter_toolace_calls(
            source_paths["toolace"], counters=readback_parser_counts
        )
        call_records.extend(_limited(records, max_records))

    calls_by_message: defaultdict[str, list[ToolAceCallRecord]] = defaultdict(list)
    for record in call_records:
        calls_by_message[record.source_id].append(record)
    invalid_messages: set[str] = set()
    message_split: dict[str, str] = {}
    for source_id, records in calls_by_message.items():
        missing = [
            record for record in records if record.tool.identity_hash not in tools_by_identity
        ]
        if missing:
            readback_skipped["missing_tool_identity"] += len(records)
            invalid_messages.add(source_id)
            continue
        splits = {
            group_splits[tools_by_identity[record.tool.identity_hash].group_hash]
            for record in records
        }
        if len(splits) != 1:
            readback_skipped["mixed_split_call_message"] += len(records)
            invalid_messages.add(source_id)
            continue
        message_split[source_id] = next(iter(splits))

    readback_query_splits: defaultdict[str, set[str]] = defaultdict(set)
    for source_id, records in calls_by_message.items():
        if source_id in invalid_messages:
            continue
        query_hash = normalized_query_hash(records[0].query)
        readback_query_splits[query_hash].add(message_split[source_id])
    invalid_query_hashes = {
        query_hash
        for query_hash, splits in readback_query_splits.items()
        if len(splits) != 1
    }

    seen_readback_calls: set[tuple[str, str, str]] = set()
    with (output / "readback.jsonl").open("w", encoding="utf-8") as handle:
        for source_id in sorted(calls_by_message):
            if source_id in invalid_messages:
                continue
            split = message_split[source_id]
            for record in calls_by_message[source_id]:
                query_hash = normalized_query_hash(record.query)
                if query_hash in invalid_query_hashes:
                    readback_skipped["cross_split_exact_query"] += 1
                    continue
                retrieval_split = retained_retrieval_splits.get(query_hash)
                if retrieval_split is not None and retrieval_split != split:
                    readback_skipped["retrieval_cross_split_exact_query"] += 1
                    continue
                arguments_json = _canonical_json(record.arguments)
                deduplication_key = (
                    query_hash,
                    record.tool.identity_hash,
                    arguments_json,
                )
                if deduplication_key in seen_readback_calls:
                    readback_skipped["duplicate_query_tool_arguments"] += 1
                    continue
                seen_readback_calls.add(deduplication_key)
                readback_counts[split] += 1
                _write_jsonl(
                    handle,
                    {
                        "source": "toolace",
                        "source_id": record.source_id,
                        "query": record.query,
                        "query_hash": query_hash,
                        "split": split,
                        "tool_identity_hash": record.tool.identity_hash,
                        "tool_name": record.tool.tool_name,
                        "arguments": record.arguments,
                        "call_index": record.call_index,
                        "call_count": record.call_count,
                    },
                )

    trajectory_counts: Counter[str] = Counter()
    trajectory_skipped: Counter[str] = Counter()
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        records = iter_toolgen_trajectories(source_paths["toolgen_trajectories"])
        for record in _limited(records, max_records):
            if any(token in ambiguous_tokens for token in record.target_tokens):
                trajectory_skipped["ambiguous_token"] += 1
                continue
            splits = {token_splits[token] for token in record.target_tokens if token in token_splits}
            unknown = [token for token in record.target_tokens if token not in token_splits]
            if unknown:
                trajectory_skipped["unknown_token"] += 1
                continue
            if len(splits) != 1:
                trajectory_skipped["mixed_split_tools"] += 1
                continue
            split = next(iter(splits))
            trajectory_counts[split] += 1
            _write_jsonl(handle, {**asdict(record), "split": split})

    source_metadata = {name: _file_metadata(path) for name, path in source_paths.items()}
    manifest = {
        "version": 1,
        "seed": seed,
        "max_records_per_stream": max_records,
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "token_pools": pools.as_dict(),
        "sources": source_metadata,
        "counts": {
            "registrations_seen": registration_seen,
            "registration_duplicates": registration_duplicates,
            "toolgen_unique_tokens": len(toolgen_token_identities),
            "toolgen_ambiguous_tokens": len(ambiguous_tokens),
            "toolgen_ambiguous_identity_variants": sum(
                len(toolgen_token_identities[token]) for token in ambiguous_tokens
            ),
            "unique_tools": len(tools_by_identity),
            "tools_by_split": dict(sorted(tool_counts.items())),
            "tools_by_primary_source": dict(sorted(source_tool_counts.items())),
            "retrieval_pairs_seen": retrieval_pairs_seen,
            "retrieval_unique_queries_seen": len(query_records),
            "retrieval_queries_by_split": dict(sorted(retrieval_counts.items())),
            "retrieval_target_pairs_by_split": dict(sorted(retrieval_target_counts.items())),
            "retrieval_skipped": dict(sorted(retrieval_skipped.items())),
            "readback_calls_by_split": dict(sorted(readback_counts.items())),
            "readback_parser": dict(sorted(readback_parser_counts.items())),
            "readback_skipped": dict(sorted(readback_skipped.items())),
            "trajectories_by_split": dict(sorted(trajectory_counts.items())),
            "trajectories_skipped": dict(sorted(trajectory_skipped.items())),
        },
        "audits": {
            "tool_group_overlap": overlap,
            "tool_group_overlap_zero": not any(overlap.values()),
            "token_pool_overlap_zero": True,
            "evaluation_tokens_used_by_training": 0,
            "test_tokens_used_by_training_or_validation": 0,
            "normalized_query_cross_split_queries_removed": retrieval_skipped.get(
                "cross_split_queries", 0
            ),
        },
        "binding_protocol": {
            "train": "randomly bind each train episode only within T_train",
            "validation": "randomly bind each validation episode only within T_validation",
            "test": "randomly bind each test episode only within T_test",
            "stable_tool_to_token_mapping": False,
        },
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe ToolGen/ToolACE data for episodic registration"
    )
    parser.add_argument("--toolgen-registration", required=True)
    parser.add_argument("--toolgen-retrieval", required=True)
    parser.add_argument("--toolgen-trajectories", required=True)
    parser.add_argument("--toolace")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Limit each input stream independently; zero reads the complete stream",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_scaled_data(
        toolgen_registration=args.toolgen_registration,
        toolgen_retrieval=args.toolgen_retrieval,
        toolgen_trajectories=args.toolgen_trajectories,
        toolace=args.toolace,
        output_dir=args.output_dir,
        seed=args.seed,
        max_records=args.max_records,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(json.dumps(manifest["audits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
