from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .episodic_data import (
    EpisodicRegistrySampler,
    load_prepared_tools,
    load_retrieval_episodes,
    load_token_pools,
)


def audit_prepared_episodes(
    prepared_dir: str | Path,
    *,
    registry_sizes: tuple[int, ...] = (8, 32, 128),
    sample_per_split: int = 1000,
    seed: int = 17,
) -> dict[str, Any]:
    root = Path(prepared_dir)
    tools = load_prepared_tools(root / "tools.jsonl")
    episodes = load_retrieval_episodes(root / "retrieval.jsonl", tools)
    pools = load_token_pools(root / "split_manifest.json")
    sampler = EpisodicRegistrySampler(tools, pools, seed=seed)

    episodes_by_split: defaultdict[str, list[Any]] = defaultdict(list)
    for episode in episodes:
        episodes_by_split[episode.split].append(episode)

    results: dict[str, Any] = {}
    total_violations = 0
    for split in sorted(pools):
        split_episodes = sorted(
            episodes_by_split[split], key=lambda episode: episode.query_hash
        )[:sample_per_split]
        size_results: dict[str, Any] = {}
        for registry_size in registry_sizes:
            observed_slots: set[int] = set()
            positive_counts: Counter[int] = Counter()
            rebound = 0
            for episode in split_episodes:
                first = sampler.bind(episode, registry_size=registry_size, epoch=0)
                second = sampler.bind(episode, registry_size=registry_size, epoch=1)
                observed_slots.update(first.slot_indices)
                positive_counts[len(first.positive_positions)] += 1
                rebound += first.slot_indices != second.slot_indices
                allowed = pools[split]
                total_violations += sum(slot not in allowed for slot in first.slot_indices)
                total_violations += sum(tool.split != split for tool in first.tools)
            size_results[str(registry_size)] = {
                "episodes": len(split_episodes),
                "positive_count_histogram": {
                    str(key): value for key, value in sorted(positive_counts.items())
                },
                "unique_physical_slots_observed": len(observed_slots),
                "epoch_rebinding_rate": rebound / max(1, len(split_episodes)),
            }
        results[split] = {
            "available_episodes": len(episodes_by_split[split]),
            "available_tools": len(sampler.identities_by_split[split]),
            "token_pool_size": len(pools[split]),
            "samples": size_results,
        }

    return {
        "prepared_dir": str(root.resolve()),
        "seed": seed,
        "sample_per_split": sample_per_split,
        "registry_sizes": list(registry_sizes),
        "splits": results,
        "audits": {
            "cross_split_tool_or_token_violations": total_violations,
            "cross_split_tool_or_token_violations_zero": total_violations == 0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit episodic registry binding")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--registry-size", type=int, action="append")
    parser.add_argument("--sample-per-split", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit_prepared_episodes(
        args.prepared_dir,
        registry_sizes=tuple(args.registry_size or (8, 32, 128)),
        sample_per_split=args.sample_per_split,
        seed=args.seed,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
