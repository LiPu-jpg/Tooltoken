from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class PreparedTool:
    identity_hash: str
    group_hash: str
    split: str
    document: str
    tool_name: str
    endpoint_name: str
    source: str
    parameters: dict[str, Any] | None = None
    token: str | None = None


@dataclass(frozen=True)
class PreparedRetrievalEpisode:
    query_hash: str
    query: str
    split: str
    target_identity_hashes: tuple[str, ...]


@dataclass(frozen=True)
class PreparedReadbackExample:
    source_id: str
    query_hash: str
    query: str
    split: str
    tool_identity_hash: str
    arguments: dict[str, Any]
    call_index: int
    call_count: int


@dataclass(frozen=True)
class BoundRegistryEpisode:
    query_hash: str
    query: str
    split: str
    tools: tuple[PreparedTool, ...]
    slot_indices: tuple[int, ...]
    positive_positions: tuple[int, ...]

    @property
    def positive_slot_indices(self) -> tuple[int, ...]:
        return tuple(self.slot_indices[index] for index in self.positive_positions)


def _read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def load_prepared_tools(path: str | Path) -> dict[str, PreparedTool]:
    tools: dict[str, PreparedTool] = {}
    for row in _read_jsonl(path):
        tool = PreparedTool(
            identity_hash=str(row["identity_hash"]),
            group_hash=str(row["group_hash"]),
            split=str(row["split"]),
            document=str(row["document"]),
            tool_name=str(row["tool_name"]),
            endpoint_name=str(row["endpoint_name"]),
            source=str(row["source"]),
            parameters=row.get("parameters") if isinstance(row.get("parameters"), dict) else None,
            token=str(row["token"]) if row.get("token") is not None else None,
        )
        if tool.identity_hash in tools:
            raise ValueError(f"Duplicate prepared tool identity: {tool.identity_hash}")
        tools[tool.identity_hash] = tool
    return tools


def load_retrieval_episodes(
    path: str | Path,
    tools: dict[str, PreparedTool],
    *,
    split: str | None = None,
) -> list[PreparedRetrievalEpisode]:
    episodes: list[PreparedRetrievalEpisode] = []
    for row in _read_jsonl(path):
        row_split = str(row["split"])
        if split is not None and row_split != split:
            continue
        target_hashes = tuple(str(value) for value in row["tool_identity_hashes"])
        if not target_hashes:
            raise ValueError(f"Retrieval query has no targets: {row['query_hash']}")
        missing = [identity for identity in target_hashes if identity not in tools]
        if missing:
            raise ValueError(f"Retrieval query references missing tools: {missing[:3]}")
        wrong_split = [
            identity for identity in target_hashes if tools[identity].split != row_split
        ]
        if wrong_split:
            raise ValueError(
                f"Retrieval query {row['query_hash']} crosses prepared tool splits"
            )
        episodes.append(
            PreparedRetrievalEpisode(
                query_hash=str(row["query_hash"]),
                query=str(row["query"]),
                split=row_split,
                target_identity_hashes=target_hashes,
            )
        )
    return episodes


def load_readback_examples(
    path: str | Path,
    tools: dict[str, PreparedTool],
    *,
    split: str | None = None,
    single_call_only: bool = False,
) -> list[PreparedReadbackExample]:
    examples: list[PreparedReadbackExample] = []
    for row in _read_jsonl(path):
        row_split = str(row["split"])
        if split is not None and row_split != split:
            continue
        call_count = int(row["call_count"])
        if single_call_only and call_count != 1:
            continue
        identity = str(row["tool_identity_hash"])
        if identity not in tools:
            raise ValueError(f"Readback example references missing tool: {identity}")
        if tools[identity].split != row_split:
            raise ValueError(
                f"Readback example {row['source_id']} crosses prepared tool splits"
            )
        arguments = row.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError(f"Readback arguments must be an object: {row['source_id']}")
        examples.append(
            PreparedReadbackExample(
                source_id=str(row["source_id"]),
                query_hash=str(row["query_hash"]),
                query=str(row["query"]),
                split=row_split,
                tool_identity_hash=identity,
                arguments=arguments,
                call_index=int(row["call_index"]),
                call_count=call_count,
            )
        )
    return examples


def load_token_pools(manifest_path: str | Path) -> dict[str, range]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    pools: dict[str, range] = {}
    for split, values in manifest["token_pools"].items():
        pools[str(split)] = range(
            int(values["start_inclusive"]), int(values["end_exclusive"])
        )
    names = sorted(pools)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if set(pools[left]).intersection(pools[right]):
                raise ValueError(f"Prepared token pools overlap: {left} and {right}")
    return pools


class EpisodicRegistrySampler:
    def __init__(
        self,
        tools: dict[str, PreparedTool],
        token_pools: dict[str, range],
        *,
        seed: int,
    ) -> None:
        self.tools = tools
        self.token_pools = token_pools
        self.seed = seed
        self.identities_by_split: dict[str, tuple[str, ...]] = {}
        for split in token_pools:
            self.identities_by_split[split] = tuple(
                sorted(identity for identity, tool in tools.items() if tool.split == split)
            )

    def bind(
        self,
        episode: PreparedRetrievalEpisode,
        *,
        registry_size: int,
        epoch: int,
    ) -> BoundRegistryEpisode:
        if registry_size < 1:
            raise ValueError("Registry size must be positive")
        if episode.split not in self.token_pools:
            raise ValueError(f"No token pool for split {episode.split}")
        targets = tuple(dict.fromkeys(episode.target_identity_hashes))
        if any(self.tools[identity].split != episode.split for identity in targets):
            raise ValueError("Episode targets cross tool splits")
        final_size = max(registry_size, len(targets))
        pool = self.identities_by_split[episode.split]
        needed = final_size - len(targets)
        if needed > len(pool) - len(targets):
            raise ValueError(
                f"Split {episode.split} has only {len(pool)} tools for registry size {final_size}"
            )
        token_pool = self.token_pools[episode.split]
        if final_size > len(token_pool):
            raise ValueError(
                f"Token pool {episode.split} has only {len(token_pool)} slots for registry size {final_size}"
            )

        digest = hashlib.sha256(
            f"{self.seed}:{epoch}:{episode.query_hash}".encode("utf-8")
        ).digest()
        generator = random.Random(int.from_bytes(digest[:8], "big"))
        selected = set(targets)
        distractors: list[str] = []
        while len(distractors) < needed:
            identity = pool[generator.randrange(len(pool))]
            if identity in selected:
                continue
            selected.add(identity)
            distractors.append(identity)
        identities = list(targets) + distractors
        generator.shuffle(identities)
        slot_indices = generator.sample(token_pool, final_size)
        target_set = set(targets)
        positive_positions = tuple(
            index for index, identity in enumerate(identities) if identity in target_set
        )
        bound = BoundRegistryEpisode(
            query_hash=episode.query_hash,
            query=episode.query,
            split=episode.split,
            tools=tuple(self.tools[identity] for identity in identities),
            slot_indices=tuple(slot_indices),
            positive_positions=positive_positions,
        )
        self.audit(bound)
        return bound

    def audit(self, episode: BoundRegistryEpisode) -> None:
        if len(episode.tools) != len(episode.slot_indices):
            raise ValueError("Tool and slot counts do not match")
        if len(set(episode.slot_indices)) != len(episode.slot_indices):
            raise ValueError("An episodic registry reused a physical slot")
        allowed = self.token_pools[episode.split]
        if any(slot not in allowed for slot in episode.slot_indices):
            raise ValueError("An episodic registry used a slot from another split")
        if any(tool.split != episode.split for tool in episode.tools):
            raise ValueError("An episodic registry used a tool from another split")
        if not episode.positive_positions:
            raise ValueError("An episodic registry has no positive tool")
        if any(index < 0 or index >= len(episode.tools) for index in episode.positive_positions):
            raise ValueError("Positive registry position is out of range")
