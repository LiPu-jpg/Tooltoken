from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import ijson

from latent_register.prepare_scale_data import (
    TOOL_TOKEN_PATTERN,
    iter_toolgen_registrations,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("rb") as handle:
        for value in ijson.items(handle, "item"):
            if isinstance(value, dict):
                yield value


def message_role(message: Mapping[str, Any]) -> str:
    return str(message.get("role", message.get("from", ""))).strip().lower()


def message_content(message: Mapping[str, Any]) -> str:
    return str(message.get("content", message.get("value", "")))


def target_tokens(record: Mapping[str, Any]) -> set[str]:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, list):
        return set()
    tokens = set()
    for message in conversations:
        if not isinstance(message, dict) or message_role(message) != "assistant":
            continue
        content = message_content(message).strip()
        if TOOL_TOKEN_PATTERN.fullmatch(content):
            tokens.add(content)
    return tokens


def select_unique_tokens(
    registration_path: str | Path,
    *,
    tool_count: int,
    seed: int,
) -> list[dict[str, str]]:
    token_identities: defaultdict[str, set[str]] = defaultdict(set)
    token_documents: dict[str, str] = {}
    for tool in iter_toolgen_registrations(registration_path):
        if tool.token is None:
            continue
        token_identities[tool.token].add(tool.identity_hash)
        token_documents.setdefault(
            tool.token, hashlib.sha256(tool.document.encode("utf-8")).hexdigest()
        )
    candidates = [
        {
            "token": token,
            "identity_hash": next(iter(identities)),
            "document_hash": token_documents[token],
        }
        for token, identities in token_identities.items()
        if len(identities) == 1
    ]
    candidates.sort(
        key=lambda value: (
            hashlib.sha256(f"{seed}:{value['token']}".encode("utf-8")).hexdigest(),
            value["token"],
        )
    )
    if tool_count <= 0:
        raise ValueError("tool_count must be positive")
    if len(candidates) < tool_count:
        raise ValueError(
            f"Requested {tool_count} tools, but only {len(candidates)} unique tokens exist"
        )
    return candidates[:tool_count]


def write_filtered_array(
    source: str | Path,
    destination: str | Path,
    selected_tokens: set[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    first = True
    with Path(destination).open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        for record in iter_json_array(source):
            counts["seen"] += 1
            targets = target_tokens(record)
            if not targets:
                counts["excluded_no_tool_target"] += 1
                continue
            included = targets & selected_tokens
            excluded = targets - selected_tokens
            if included and excluded:
                counts["excluded_mixed_selected_and_unselected"] += 1
                continue
            if excluded:
                counts["excluded_outside_selection"] += 1
                continue
            if not first:
                handle.write(",\n")
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            first = False
            counts["retained"] += 1
            counts["retained_target_bindings"] += len(targets)
        handle.write("\n]\n")
    return dict(sorted(counts.items()))


def prepare_toolgen_pilot(
    *,
    registration: str | Path,
    retrieval: str | Path,
    trajectories: str | Path,
    output_dir: str | Path,
    tool_count: int = 5_000,
    seed: int = 17,
    upstream_commit: str = "6839374a255810efe69deea4056eec5c55e25802",
) -> dict[str, Any]:
    sources = {
        "memorization": Path(registration),
        "retrieval": Path(retrieval),
        "trajectories": Path(trajectories),
    }
    for name, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} source: {path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selection = select_unique_tokens(
        sources["memorization"], tool_count=tool_count, seed=seed
    )
    selected_tokens = {value["token"] for value in selection}
    if len(selected_tokens) != tool_count:
        raise AssertionError("Selected token mapping is not one-to-one")

    selected_path = output / "selected_tools.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for value in selection:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    virtual_tokens_path = output / "virtual_tokens.txt"
    virtual_tokens_path.write_text(
        "".join(f"{value['token']}\n" for value in selection), encoding="utf-8"
    )

    destinations = {
        "memorization": output / "toolgen_atomic_memorization.json",
        "retrieval": output / "toolgen_atomic_retrieval_G123.json",
        "trajectories": output / "toolgen_atomic_G123_dfs.json",
    }
    counts = {
        name: write_filtered_array(sources[name], destinations[name], selected_tokens)
        for name in sources
    }
    if counts["memorization"].get("retained", 0) < tool_count:
        raise AssertionError("Pilot lost one or more selected memorization targets")

    manifest = {
        "version": 1,
        "kind": "toolgen_official_pilot",
        "seed": seed,
        "selected_tool_count": tool_count,
        "upstream_commit": upstream_commit,
        "selection": "sha256(seed + ':' + atomic_token), unique identity tokens only",
        "mixed_record_policy": "exclude_complete_record",
        "sources": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
        "outputs": {
            name: {
                "path": destination.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
            for name, destination in destinations.items()
        },
        "selection_files": {
            "selected_tools": {
                "path": selected_path.name,
                "sha256": sha256_file(selected_path),
            },
            "virtual_tokens": {
                "path": virtual_tokens_path.name,
                "sha256": sha256_file(virtual_tokens_path),
            },
        },
        "counts": counts,
        "audits": {
            "selected_token_count_exact": len(selected_tokens) == tool_count,
            "selected_tokens_unique": len(selection) == len(selected_tokens),
            "mixed_records_partially_relabelled": 0,
        },
    }
    manifest_path = output / "pilot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic ToolGen pilot")
    parser.add_argument("--registration", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tool-count", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = prepare_toolgen_pilot(
        registration=args.registration,
        retrieval=args.retrieval,
        trajectories=args.trajectories,
        output_dir=args.output_dir,
        tool_count=args.tool_count,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
