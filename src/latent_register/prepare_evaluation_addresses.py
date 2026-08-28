from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .episodic_data import load_token_pools


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _range_record(values: range) -> dict[str, int]:
    return {
        "start_inclusive": values.start,
        "end_exclusive": values.stop,
        "count": len(values),
    }


def prepare_evaluation_address_manifest(
    source_manifest_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_addresses_per_split: int,
) -> dict[str, Any]:
    """Expand held-out physical addresses without changing training addresses."""
    if evaluation_addresses_per_split < 1:
        raise ValueError("Evaluation address count must be positive")
    source_path = Path(source_manifest_path)
    source_pools = load_token_pools(source_path)
    train = source_pools["train"]
    if train.start != 0:
        raise ValueError("The training address pool must start at logical slot zero")

    validation = range(train.stop, train.stop + evaluation_addresses_per_split)
    test = range(validation.stop, validation.stop + evaluation_addresses_per_split)
    pools = {"train": train, "validation": validation, "test": test}
    occupied = [slot for values in pools.values() for slot in values]
    dense_and_disjoint = len(occupied) == len(set(occupied)) and set(occupied) == set(
        range(test.stop)
    )
    if not dense_and_disjoint:
        raise AssertionError("Expanded address pools are not dense and disjoint")

    result: dict[str, Any] = {
        "version": 1,
        "kind": "evaluation_only_expanded_physical_address_manifest",
        "source_manifest": {
            "path": str(source_path.resolve()),
            "sha256": _sha256_file(source_path),
        },
        "token_pools": {
            split: _range_record(values) for split, values in pools.items()
        },
        "audits": {
            "training_pool_unchanged": train == source_pools["train"],
            "evaluation_pools_disjoint_from_training": not (
                set(train) & (set(validation) | set(test))
            ),
            "all_pools_dense_and_disjoint": dense_and_disjoint,
            "evaluation_only_no_training_labels": True,
        },
        "maximum_registry_size_without_address_reuse": evaluation_addresses_per_split,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an evaluation-only held-out physical-address manifest"
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-addresses-per-split", type=int, required=True)
    args = parser.parse_args()
    result = prepare_evaluation_address_manifest(
        args.source_manifest,
        args.output,
        evaluation_addresses_per_split=args.evaluation_addresses_per_split,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
