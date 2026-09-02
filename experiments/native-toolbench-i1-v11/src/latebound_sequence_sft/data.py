from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import ijson
except ImportError:  # pragma: no cover - cluster dependency, stdlib fallback below
    ijson = None


ACTION_RE = re.compile(r"^<<(?P<tool>.*?)&&(?P<api>.*?)>>$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize(value: str) -> str:
    """Pinned ToolBench name normalization used by the official evaluator."""
    value = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", str(value))
    value = re.sub(r"(_)\1+", "_", value).lower().strip("_")
    if value and value[0].isdigit():
        value = "get_" + value
    return value


def normalize_name(value: str) -> str:
    """Compatibility helper for callers that need a human-readable key."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.split()).casefold()


def action_key(tool: str, api: str) -> str:
    tool_key = standardize(tool)
    api_key = standardize(api)
    if api_key in {"from", "class", "return", "false", "true", "id", "and"}:
        api_key = "is_" + api_key
    # Official ToolBench keeps rows whose API normalizes to an empty string
    # (for example ``/``) as ``_for_<tool>``.  Preserve them verbatim so the
    # exact registry remains one-to-one.
    return (f"{api_key}_for_{tool_key}")[-64:]


def parse_action(value: str) -> str:
    match = ACTION_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid atomic ToolBench action: {value!r}")
    return action_key(match.group("tool"), match.group("api"))


def load_atomic_map(path: Path) -> tuple[dict[str, str], str]:
    """Load the official token-to-action catalog without rewriting labels."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ToolBench atomic map must be a JSON object")
    reverse: dict[str, str] = {}
    for action, token in payload.items():
        action_value, token_value = str(action), str(token)
        previous = reverse.setdefault(token_value, action_value)
        if previous != action_value:
            raise ValueError(f"Atomic token maps to multiple actions: {token_value!r}")
    return reverse, sha256_file(path)


@dataclass(frozen=True)
class ToolDocument:
    """One exact corpus document; same action names never merge identities."""

    docid: str
    action: str
    document: dict[str, Any]

    @property
    def text(self) -> str:
        return json.dumps(self.document, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class SequenceExample:
    query: str
    action: str
    document_id: str
    positive_document_count: int

    @property
    def loss_weight(self) -> float:
        """Give each action equal mass when it has duplicate exact documents."""
        return 1.0 / float(self.positive_document_count)


def load_corpus(path: Path) -> tuple[list[ToolDocument], str]:
    """Load the official TSV while retaining exact docid identity and schema."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    documents: list[ToolDocument] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["", "docid", "document_content"]:
            raise ValueError(f"Unexpected ToolBench header: {reader.fieldnames}")
        for row in reader:
            docid = str(row["docid"])
            if not docid or docid in seen:
                raise ValueError(f"Duplicate or empty corpus docid: {docid!r}")
            seen.add(docid)
            document = json.loads(row["document_content"])
            if not isinstance(document, dict):
                raise ValueError(f"Document {docid} is not a JSON object")
            action = action_key(
                document.get("tool_name", ""), document.get("api_name", "")
            )
            documents.append(ToolDocument(docid, action, document))
    if not documents:
        raise ValueError("The ToolBench corpus is empty")
    return documents, sha256_file(path)


def iter_retrieval_rows(path: Path) -> Iterator[dict[str, Any]]:
    if ijson is not None:
        with path.open("rb") as handle:
            for row in ijson.items(handle, "item"):
                if not isinstance(row, dict):
                    raise ValueError("Every retrieval row must be an object")
                yield row
        return
    # Keep a bounded-memory fallback for environments without ijson.
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    expecting_value = True
    eof = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            if not eof:
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            cursor = 0
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            buffer = buffer[cursor:]
            if not started:
                if not buffer:
                    if eof:
                        raise ValueError("ToolGen retrieval source is empty")
                    continue
                if buffer[0] != "[":
                    raise ValueError("ToolGen retrieval source must be a JSON list")
                started = True
                buffer = buffer[1:]
                expecting_value = True
                continue
            while buffer and buffer[0].isspace():
                buffer = buffer[1:]
            if buffer.startswith("]"):
                return
            if not buffer:
                if eof:
                    raise ValueError("ToolGen retrieval array is truncated")
                continue
            if not expecting_value:
                if buffer[0] != ",":
                    raise ValueError("Malformed ToolGen retrieval JSON array")
                buffer = buffer[1:]
                expecting_value = True
                continue
            try:
                row, used = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("ToolGen retrieval array is truncated")
                more = handle.read(1024 * 1024)
                if more:
                    buffer += more
                    continue
                eof = True
                continue
            buffer = buffer[used:]
            expecting_value = False
            if not isinstance(row, dict):
                raise ValueError("Every retrieval row must be an object")
            yield row


def evaluation_queries(query_files: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in query_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                excluded.add(line.split("\t", 1)[-1].strip())
    return excluded


def build_examples(
    retrieval_path: Path,
    corpus: list[ToolDocument],
    *,
    excluded_queries: set[str],
    atomic_token_to_action: dict[str, str] | None = None,
) -> tuple[list[SequenceExample], int]:
    """Convert official rows to exact-document sequence examples.

    ToolGen labels an action name, while the corpus can contain several exact
    documents with that action.  Each exact document therefore gets its own
    sequence target; ``loss_weight`` keeps an action with duplicates from
    receiving more total training mass than an action with one document.
    """
    by_action: dict[str, list[ToolDocument]] = {}
    by_id: dict[str, ToolDocument] = {}
    for document in corpus:
        by_action.setdefault(document.action, []).append(document)
        by_id[document.docid] = document
    examples: list[SequenceExample] = []
    excluded_count = 0
    for row in iter_retrieval_rows(retrieval_path):
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError("Every retrieval row must contain two messages")
        query, answer = conversations
        query_text = str(query.get("content", "")).strip()
        if query.get("role") != "user" or answer.get("role") != "assistant":
            raise ValueError("ToolGen retrieval roles changed")
        if query_text in excluded_queries:
            excluded_count += 1
            continue
        token = str(answer.get("content", "")).strip()
        action = (
            atomic_token_to_action[token]
            if atomic_token_to_action is not None and token in atomic_token_to_action
            else parse_action(token)
        )
        candidates = by_action.get(action)
        if not candidates:
            raise ValueError(f"Retrieval action is absent from corpus: {action!r}")
        ordered = sorted(candidates, key=lambda item: item.docid)
        for document in ordered:
            if document.docid not in by_id:
                raise AssertionError("Internal corpus index lost a document")
            examples.append(
                SequenceExample(
                    query_text,
                    action,
                    document.docid,
                    positive_document_count=len(ordered),
                )
            )
    return examples, excluded_count


def render_sequence(query: str, physical_token: str) -> tuple[str, str]:
    """Return ToolGen's two-message Llama-3 style prompt and target suffix."""
    prompt = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{query}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt, f"{physical_token}<|eot_id|>"


def build_manifest(
    retrieval_path: Path,
    corpus_path: Path,
    query_files: Iterable[Path],
    output_path: Path,
    atomic_map_path: Path | None = None,
) -> dict[str, Any]:
    corpus, corpus_hash = load_corpus(corpus_path)
    atomic_map = None
    atomic_map_hash = None
    if atomic_map_path is not None:
        atomic_map, atomic_map_hash = load_atomic_map(atomic_map_path)
    query_files = list(query_files)
    excluded = evaluation_queries(query_files)
    examples, excluded_count = build_examples(
        retrieval_path,
        corpus,
        excluded_queries=excluded,
        atomic_token_to_action=atomic_map,
    )
    if not examples:
        raise ValueError("No leakage-safe sequence examples remain")
    payload = {
        "kind": "native_latebound_standard_sequence_sft_manifest",
        "version": 2,
        "retrieval_sha256": sha256_file(retrieval_path),
        "corpus_sha256": corpus_hash,
        "atomic_map_sha256": atomic_map_hash,
        "evaluation_query_count": len(excluded),
        "excluded_query_rows": excluded_count,
        "candidate_count": len(corpus),
        "example_count": len(examples),
        "exact_identity_count": len(corpus),
        "corpus_action_count": len({item.action for item in corpus}),
        "duplicate_action_count": sum(
            count > 1
            for count in Counter(item.action for item in corpus).values()
        ),
        "atomic_map_action_count": len(atomic_map) if atomic_map is not None else None,
        "document_selection_policy": "all_exact_docs_per_action_with_inverse_count_weight",
        "examples": [asdict(example) for example in examples],
        "documents": [
            {"docid": item.docid, "action": item.action, "document": item.document}
            for item in corpus
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
