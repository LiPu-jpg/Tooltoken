from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .controlled_registry import (
    ControlledRegistry,
    build_controlled_registries,
    extend_controlled_registry,
)
from .episodic_data import load_prepared_tools, load_token_pools


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def build_append_registries(
    rows: Sequence[Mapping[str, Any]],
    candidate_identities: Sequence[str],
    address_pool: range,
    *,
    initial_size: int,
    extended_size: int,
    seed: int,
) -> tuple[ControlledRegistry, ControlledRegistry, dict[str, bool | int]]:
    if not rows:
        raise ValueError("Append evaluation requires at least one row")
    targets_by_row = [
        [str(identity) for identity in row["reference_tools"]] for row in rows
    ]
    keys = [
        f"{row['family']}:{row['condition']}:{row['example_id']}" for row in rows
    ]
    initial_registries = build_controlled_registries(
        candidate_identities,
        targets_by_row,
        registry_size=initial_size,
        seed=seed,
        keys=keys,
        address_pool=address_pool,
        scope="shared",
    )
    initial = initial_registries[0]
    targets = sorted({identity for values in targets_by_row for identity in values})
    extended = extend_controlled_registry(
        initial,
        candidate_identities,
        targets,
        registry_size=extended_size,
        seed=seed,
        key="sequential_append",
        address_pool=address_pool,
    )
    prefix_size = len(initial.identities)
    audits: dict[str, bool | int] = {
        "old_identity_prefix_unchanged": (
            extended.identities[:prefix_size] == initial.identities
        ),
        "old_address_prefix_unchanged": (
            extended.address_slots[:prefix_size] == initial.address_slots
        ),
        "initial_identity_count": len(set(initial.identities)),
        "initial_address_count": len(set(initial.address_slots)),
        "extended_identity_count": len(set(extended.identities)),
        "extended_address_count": len(set(extended.address_slots)),
        "target_count": len(targets),
    }
    if not audits["old_identity_prefix_unchanged"]:
        raise AssertionError("Sequential append rebound an old identity")
    if not audits["old_address_prefix_unchanged"]:
        raise AssertionError("Sequential append rebound an old address")
    return initial, extended, audits


def _registry_record(registry: ControlledRegistry) -> dict[str, Any]:
    return {
        "identities": list(registry.identities),
        "address_slots": list(registry.address_slots),
        "identity_sha256": registry.identity_sha256,
        "registry_size": len(registry.identities),
    }


def prepare_registry_append(
    *,
    benchmark_eval_path: str | Path,
    tools_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    condition: str,
    candidate_splits: Sequence[str],
    families: Sequence[str],
    max_examples_per_family: int,
    initial_size: int,
    extended_size: int,
    seed: int,
) -> dict[str, Any]:
    eval_path = Path(benchmark_eval_path)
    tool_path = Path(tools_path)
    token_manifest_path = Path(split_manifest_path)
    tools = load_prepared_tools(tool_path)
    pools = load_token_pools(token_manifest_path)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    allowed_families = set(families)
    for row in sorted(
        _read_jsonl(eval_path),
        key=lambda item: (str(item["family"]), str(item["example_id"])),
    ):
        family = str(row["family"])
        if str(row["condition"]) != condition or family not in allowed_families:
            continue
        if max_examples_per_family and counts[family] >= max_examples_per_family:
            continue
        selected.append(row)
        counts[family] += 1
    candidate_split_set = set(candidate_splits)
    candidate_identities = sorted(
        identity for identity, tool in tools.items() if tool.split in candidate_split_set
    )
    address_split = "test" if condition == "unseen_tool_unseen_token" else "validation"
    initial, extended, audits = build_append_registries(
        selected,
        candidate_identities,
        pools[address_split],
        initial_size=initial_size,
        extended_size=extended_size,
        seed=seed,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "append_eval.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    initial_path = output / "registry_initial.json"
    extended_path = output / "registry_extended.json"
    initial_path.write_text(
        json.dumps(_registry_record(initial), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    extended_path.write_text(
        json.dumps(_registry_record(extended), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": 1,
        "kind": "sequential_registry_append_benchmark",
        "condition": condition,
        "families": sorted(allowed_families),
        "candidate_splits": sorted(candidate_split_set),
        "seed": seed,
        "counts": {"rows": len(selected), "rows_by_family": dict(counts)},
        "audits": audits,
        "sources": {
            "benchmark_eval_sha256": _sha256_file(eval_path),
            "tools_sha256": _sha256_file(tool_path),
            "split_manifest_sha256": _sha256_file(token_manifest_path),
        },
        "outputs": {
            "append_eval_sha256": _sha256_file(selected_path),
            "registry_initial_sha256": _sha256_file(initial_path),
            "registry_extended_sha256": _sha256_file(extended_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a registry-append benchmark with immutable old bindings"
    )
    parser.add_argument("--benchmark-eval", required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", default="unseen_tool_unseen_token")
    parser.add_argument("--candidate-split", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--max-examples-per-family", type=int, default=32)
    parser.add_argument("--initial-size", type=int, default=100)
    parser.add_argument("--extended-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = prepare_registry_append(
        benchmark_eval_path=args.benchmark_eval,
        tools_path=args.tools,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        condition=args.condition,
        candidate_splits=args.candidate_split or ["test"],
        families=args.family or ["retrieval", "arguments"],
        max_examples_per_family=args.max_examples_per_family,
        initial_size=args.initial_size,
        extended_size=args.extended_size,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
