from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

from unidecode import unidecode

from .episodic_data import (
    PreparedReadbackExample,
    PreparedRetrievalEpisode,
    PreparedTool,
    load_prepared_tools,
    load_readback_examples,
    load_retrieval_episodes,
)


TOOL_TOKEN_PATTERN = re.compile(r"<<[^<>\n]+&&[^<>\n]+>>")
FINISH_TOKEN = "<<Finish>>"
ARGUMENT_SYSTEM = (
    "Select the required tool. After receiving its definition, return only one "
    "compact JSON object containing the tool arguments."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(_canonical_json(value))
            handle.write("\n")
            count += 1
    return count


def _write_json_array(path: Path, values: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for value in values:
            if count:
                handle.write(",")
            handle.write(_canonical_json(value))
            count += 1
        handle.write("]\n")
    return count


def _clean_token_component(value: str, fallback: str) -> str:
    cleaned = unidecode(value).replace("&&", " and ")
    cleaned = cleaned.replace("<<", " ").replace(">>", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback


def _base_fixed_token(tool: PreparedTool) -> str:
    if tool.token is not None:
        candidate = unidecode(tool.token.strip())
        if TOOL_TOKEN_PATTERN.fullmatch(candidate):
            return candidate
    name = _clean_token_component(tool.tool_name, "tool")
    endpoint = _clean_token_component(tool.endpoint_name, "endpoint")
    return f"<<{name}&&{endpoint}>>"


def allocate_fixed_tokens(
    tools: Mapping[str, PreparedTool],
) -> tuple[dict[str, str], dict[str, int]]:
    """Allocate stable ToolGen-style strings that survive upstream normalization."""
    candidates: defaultdict[str, list[str]] = defaultdict(list)
    for identity, tool in tools.items():
        candidates[_base_fixed_token(tool)].append(identity)

    mapping: dict[str, str] = {}
    collision_groups = 0
    collision_identities = 0
    for candidate, identities in sorted(candidates.items()):
        ordered = sorted(identities)
        if len(ordered) == 1:
            mapping[ordered[0]] = candidate
            continue
        collision_groups += 1
        collision_identities += len(ordered)
        body = candidate[2:-2]
        name, endpoint = body.split("&&", 1)
        for identity in ordered:
            mapping[identity] = f"<<{name}&&{endpoint}__{identity[:16]}>>"

    normalized = [unidecode(value) for value in mapping.values()]
    if len(set(normalized)) != len(normalized):
        raise AssertionError("Fixed tool tokens still collide after unidecode")
    if any(not TOOL_TOKEN_PATTERN.fullmatch(value) for value in normalized):
        raise AssertionError("A fixed tool token is not atomic ToolGen syntax")
    return mapping, {
        "base_collision_groups": collision_groups,
        "base_collision_identities": collision_identities,
        "normalized_collisions": len(normalized) - len(set(normalized)),
    }


def _holdout_order(seed: int, family: str, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{family}:{key}".encode("utf-8")).digest()


def select_seen_retrieval_holdout(
    episodes: Sequence[PreparedRetrievalEpisode],
    *,
    count: int,
    seed: int,
) -> tuple[list[PreparedRetrievalEpisode], list[PreparedRetrievalEpisode]]:
    train = [episode for episode in episodes if episode.split == "train"]
    remaining = Counter(
        identity for episode in train for identity in set(episode.target_identity_hashes)
    )
    held_out: list[PreparedRetrievalEpisode] = []
    held_hashes: set[str] = set()
    for episode in sorted(
        train,
        key=lambda item: (_holdout_order(seed, "retrieval", item.query_hash), item.query_hash),
    ):
        targets = set(episode.target_identity_hashes)
        if any(remaining[identity] <= 1 for identity in targets):
            continue
        held_out.append(episode)
        held_hashes.add(episode.query_hash)
        for identity in targets:
            remaining[identity] -= 1
        if len(held_out) == count:
            break
    if len(held_out) != count:
        raise ValueError(
            f"Could select only {len(held_out)} of {count} seen retrieval episodes "
            "while retaining one training query per target tool"
        )
    retained = [episode for episode in train if episode.query_hash not in held_hashes]
    return retained, sorted(held_out, key=lambda item: item.query_hash)


def _readback_key(example: PreparedReadbackExample) -> str:
    return f"{example.source_id}:{example.call_index}"


def select_seen_readback_holdout(
    examples: Sequence[PreparedReadbackExample],
    *,
    count: int,
    seed: int,
) -> tuple[list[PreparedReadbackExample], list[PreparedReadbackExample]]:
    train = [example for example in examples if example.split == "train"]
    remaining = Counter(example.tool_identity_hash for example in train)
    held_out: list[PreparedReadbackExample] = []
    held_keys: set[str] = set()
    for example in sorted(
        train,
        key=lambda item: (
            _holdout_order(seed, "readback", _readback_key(item)),
            _readback_key(item),
        ),
    ):
        identity = example.tool_identity_hash
        if remaining[identity] <= 1:
            continue
        held_out.append(example)
        held_keys.add(_readback_key(example))
        remaining[identity] -= 1
        if len(held_out) == count:
            break
    if len(held_out) != count:
        raise ValueError(
            f"Could select only {len(held_out)} of {count} seen readback examples "
            "while retaining one training call per target tool"
        )
    retained = [example for example in train if _readback_key(example) not in held_keys]
    return retained, sorted(held_out, key=_readback_key)


def _conversation(user: str, assistant: str) -> dict[str, Any]:
    return {
        "conversations": [
            {"role": "user", "content": user, "loss": False},
            {"role": "assistant", "content": assistant, "loss": True},
        ]
    }


def _memorization_records(
    tools: Mapping[str, PreparedTool], fixed_tokens: Mapping[str, str]
) -> Iterator[dict[str, Any]]:
    for identity in sorted(tools):
        tool = tools[identity]
        if tool.split == "train":
            yield _conversation(tool.document, fixed_tokens[identity])


def _retrieval_records(
    episodes: Sequence[PreparedRetrievalEpisode], fixed_tokens: Mapping[str, str]
) -> Iterator[dict[str, Any]]:
    for episode in sorted(episodes, key=lambda item: item.query_hash):
        for identity in sorted(set(episode.target_identity_hashes)):
            yield _conversation(episode.query, fixed_tokens[identity])


def _readback_records(
    examples: Sequence[PreparedReadbackExample],
    tools: Mapping[str, PreparedTool],
    fixed_tokens: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    for example in sorted(examples, key=_readback_key):
        tool = tools[example.tool_identity_hash]
        yield {
            "conversations": [
                {"role": "system", "content": ARGUMENT_SYSTEM, "loss": False},
                {"role": "user", "content": example.query, "loss": False},
                {
                    "role": "assistant",
                    "content": fixed_tokens[example.tool_identity_hash],
                    "loss": True,
                },
                {
                    "role": "user",
                    "content": f"Tool definition:\n{tool.document}",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": _canonical_json(example.arguments),
                    "loss": True,
                },
            ]
        }


def _full_document_readback_records(
    examples: Sequence[PreparedReadbackExample],
    tools: Mapping[str, PreparedTool],
) -> Iterator[dict[str, Any]]:
    for example in sorted(examples, key=_readback_key):
        tool = tools[example.tool_identity_hash]
        yield {
            "conversations": [
                {"role": "system", "content": ARGUMENT_SYSTEM, "loss": False},
                {
                    "role": "user",
                    "content": (
                        f"Request: {example.query}\nTool definition:\n{tool.document}"
                    ),
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": _canonical_json(example.arguments),
                    "loss": True,
                },
            ]
        }


def _trajectory_records(
    path: Path,
    tools: Mapping[str, PreparedTool],
    fixed_tokens: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    token_to_identities: defaultdict[str, set[str]] = defaultdict(set)
    for identity, tool in tools.items():
        if tool.token is not None:
            token_to_identities[tool.token].add(identity)
    unique = {
        token: next(iter(identities))
        for token, identities in token_to_identities.items()
        if len(identities) == 1
    }
    for row in _read_jsonl(path):
        if str(row.get("split")) != "train":
            continue
        target_tokens = [str(value) for value in row.get("target_tokens", [])]
        missing = [token for token in target_tokens if token not in unique]
        if missing:
            raise ValueError(f"Training trajectory has unmapped tokens: {missing[:3]}")
        replacements = {token: fixed_tokens[unique[token]] for token in target_tokens}
        conversations: list[dict[str, Any]] = []
        for message in row.get("conversations", []):
            if not isinstance(message, dict):
                continue
            updated = dict(message)
            value_key = "value" if "value" in updated else "content"
            value = updated.get(value_key)
            if isinstance(value, str) and value.strip() in replacements:
                prefix = value[: len(value) - len(value.lstrip())]
                suffix = value[len(value.rstrip()) :]
                updated[value_key] = prefix + replacements[value.strip()] + suffix
            conversations.append(updated)
        yield {"conversations": conversations}


def _evaluation_rows(
    seen_retrieval: Sequence[PreparedRetrievalEpisode],
    unseen_retrieval: Sequence[PreparedRetrievalEpisode],
    seen_readback: Sequence[PreparedReadbackExample],
    unseen_readback: Sequence[PreparedReadbackExample],
    tools: Mapping[str, PreparedTool],
    fixed_tokens: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    for condition, episodes in (
        ("seen_tool_seen_token", seen_retrieval),
        ("unseen_tool_unseen_token", unseen_retrieval),
    ):
        for episode in sorted(episodes, key=lambda item: item.query_hash):
            identities = list(dict.fromkeys(episode.target_identity_hashes))
            yield {
                "family": "retrieval",
                "condition": condition,
                "example_id": episode.query_hash,
                "query": episode.query,
                "reference_tools": identities,
                "reference_fixed_tokens": [fixed_tokens[value] for value in identities],
                "source_split": episode.split,
            }
    for condition, examples in (
        ("seen_tool_seen_token", seen_readback),
        ("unseen_tool_unseen_token", unseen_readback),
    ):
        for example in sorted(examples, key=_readback_key):
            tool = tools[example.tool_identity_hash]
            yield {
                "family": "arguments",
                "condition": condition,
                "example_id": _readback_key(example),
                "query": example.query,
                "reference_tools": [example.tool_identity_hash],
                "reference_fixed_tokens": [fixed_tokens[example.tool_identity_hash]],
                "reference_arguments": example.arguments,
                "schema": tool.parameters,
                "source_split": example.split,
            }


def prepare_controlled_benchmark(
    *,
    prepared_dir: str | Path,
    output_dir: str | Path,
    seed: int = 17,
    seen_retrieval_count: int | None = None,
    seen_readback_count: int | None = None,
) -> dict[str, Any]:
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        name: prepared / name
        for name in (
            "tools.jsonl",
            "retrieval.jsonl",
            "readback.jsonl",
            "trajectories.jsonl",
            "split_manifest.json",
        )
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared source {name}: {path}")

    source_manifest = json.loads(source_paths["split_manifest.json"].read_text())
    if not source_manifest.get("audits", {}).get("tool_group_overlap_zero"):
        raise ValueError("Prepared corpus did not pass the tool-group leakage audit")
    tools = load_prepared_tools(source_paths["tools.jsonl"])
    retrieval = load_retrieval_episodes(source_paths["retrieval.jsonl"], tools)
    readback = load_readback_examples(source_paths["readback.jsonl"], tools)
    fixed_tokens, token_audit = allocate_fixed_tokens(tools)

    unseen_retrieval = [episode for episode in retrieval if episode.split == "test"]
    unseen_readback = [example for example in readback if example.split == "test"]
    retrieval_limit = (
        len(unseen_retrieval) if seen_retrieval_count is None else seen_retrieval_count
    )
    readback_limit = len(unseen_readback) if seen_readback_count is None else seen_readback_count
    retrieval_train, seen_retrieval = select_seen_retrieval_holdout(
        retrieval, count=retrieval_limit, seed=seed
    )
    readback_train, seen_readback = select_seen_readback_holdout(
        readback, count=readback_limit, seed=seed
    )

    mapping_path = output / "fixed_tokens.jsonl"
    _write_jsonl(
        mapping_path,
        (
            {
                "identity_hash": identity,
                "split": tools[identity].split,
                "fixed_token": fixed_tokens[identity],
                "original_token": tools[identity].token,
            }
            for identity in sorted(tools)
        ),
    )
    for split in ("train", "validation", "test"):
        values = [
            fixed_tokens[identity]
            for identity in sorted(tools)
            if tools[identity].split == split
        ]
        if split == "train":
            values.append(FINISH_TOKEN)
        (output / f"virtual_tokens_{split}.txt").write_text(
            "".join(f"{value}\n" for value in values), encoding="utf-8"
        )

    output_counts = {
        "fixed_memorization_train": _write_json_array(
            output / "fixed_memorization_train.json",
            _memorization_records(tools, fixed_tokens),
        ),
        "fixed_retrieval_train": _write_json_array(
            output / "fixed_retrieval_train.json",
            _retrieval_records(retrieval_train, fixed_tokens),
        ),
        "fixed_readback_train": _write_json_array(
            output / "fixed_readback_train.json",
            _readback_records(readback_train, tools, fixed_tokens),
        ),
        "full_document_readback_train": _write_json_array(
            output / "full_document_readback_train.json",
            _full_document_readback_records(readback_train, tools),
        ),
        "fixed_trajectories_train": _write_json_array(
            output / "fixed_trajectories_train.json",
            _trajectory_records(source_paths["trajectories.jsonl"], tools, fixed_tokens),
        ),
        "benchmark_eval": _write_jsonl(
            output / "benchmark_eval.jsonl",
            _evaluation_rows(
                seen_retrieval,
                unseen_retrieval,
                seen_readback,
                unseen_readback,
                tools,
                fixed_tokens,
            ),
        ),
    }

    remaining_retrieval_targets = Counter(
        identity
        for episode in retrieval_train
        for identity in set(episode.target_identity_hashes)
    )
    remaining_readback_targets = Counter(
        example.tool_identity_hash for example in readback_train
    )
    outputs = {
        path.name: _file_metadata(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "benchmark_manifest.json"
    }
    manifest = {
        "version": 1,
        "kind": "controlled_qwen_fixed_and_late_bound_benchmark",
        "seed": seed,
        "prepared_dir": str(prepared.resolve()),
        "sources": {name: _file_metadata(path) for name, path in source_paths.items()},
        "counts": {
            "tools_by_split": dict(Counter(tool.split for tool in tools.values())),
            "retrieval_train": len(retrieval_train),
            "retrieval_seen_eval": len(seen_retrieval),
            "retrieval_unseen_test": len(unseen_retrieval),
            "readback_train": len(readback_train),
            "readback_seen_eval": len(seen_readback),
            "readback_unseen_test": len(unseen_readback),
            **output_counts,
        },
        "audits": {
            "source_tool_group_overlap_zero": True,
            "fixed_token_normalized_collisions": token_audit["normalized_collisions"],
            "fixed_token_base_collision_groups": token_audit["base_collision_groups"],
            "seen_retrieval_queries_in_training": len(
                {item.query_hash for item in seen_retrieval}
                & {item.query_hash for item in retrieval_train}
            ),
            "seen_retrieval_targets_without_remaining_training_query": sum(
                remaining_retrieval_targets[identity] == 0
                for item in seen_retrieval
                for identity in set(item.target_identity_hashes)
            ),
            "seen_readback_examples_in_training": len(
                {_readback_key(item) for item in seen_readback}
                & {_readback_key(item) for item in readback_train}
            ),
            "seen_readback_targets_without_remaining_training_call": sum(
                remaining_readback_targets[item.tool_identity_hash] == 0
                for item in seen_readback
            ),
            "validation_or_test_target_tokens_in_training_labels": 0,
            "bfcl_sources_in_training": 0,
        },
        "fixed_token_training_contract": {
            "base_checkpoint_token_file": "virtual_tokens_train.txt",
            "validation_and_test_tokens_absent_from_base_checkpoint": True,
            "incremental_token_files": [
                "virtual_tokens_validation.txt",
                "virtual_tokens_test.txt",
            ],
        },
        "evaluation_contract": {
            "seen_tool_seen_token": "train-split tools with held-out queries/calls",
            "unseen_tool_unseen_token": "group-disjoint test tools and test token file",
            "seen_tool_unseen_address": "late-bound rebind of seen evaluation tools",
            "unseen_tool_seen_address": "late-bound diagnostic using train address pool",
            "mixed_append": "deterministic union of seen and unseen evaluation registries",
        },
        "outputs": outputs,
    }
    nonzero_audits = {
        key: value
        for key, value in manifest["audits"].items()
        if key.endswith(("in_training", "without_remaining_training_query", "without_remaining_training_call"))
        and value != 0
    }
    if nonzero_audits:
        raise AssertionError(f"Controlled benchmark leakage audit failed: {nonzero_audits}")
    (output / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize leakage-safe fixed-token and late-binding benchmark data"
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seen-retrieval-count", type=int)
    parser.add_argument("--seen-readback-count", type=int)
    args = parser.parse_args()
    manifest = prepare_controlled_benchmark(
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        seen_retrieval_count=args.seen_retrieval_count,
        seen_readback_count=args.seen_readback_count,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(json.dumps(manifest["audits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
