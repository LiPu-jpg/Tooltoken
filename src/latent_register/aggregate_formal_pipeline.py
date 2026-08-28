"""Aggregate every comparison cell from three formal seed pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .aggregate_formal_seeds import (
    FORMAL_SEEDS,
    aggregate_formal_seed_comparisons,
)
from .validate_toolgen_tokenizer import sha256_file, write_json_atomic


MATCHED_REGISTRY_SIZES = (10, 100, 1000)
SCALE_REGISTRY_SIZES = (10000, 47000)
ADDRESS_STATUSES = ("seen", "unseen")
CONTROLS = (
    "blank",
    "random",
    "wrong_memory",
    "permuted",
    "shared_vector",
    "query_only",
    "nearest_trained",
)
REQUIRED_COMPARISON_JOBS = (
    "compare_fixed_matched",
    "compare_oracle_matched",
    "compare_incremental_matched",
    "compare_fixed_scale",
    "compare_oracle_scale",
    "controls_compare",
    "query_only_compare",
    "nearest_trained_compare",
    "append_compare",
)
EXPECTED_CELL_COUNT = 27


def load_submission_manifest(path: Path, expected_seed: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Submission manifest is not a JSON object: {path}")
    if payload.get("kind") != "formal_controlled_pipeline_submission":
        raise ValueError(f"Unexpected submission manifest kind: {path}")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"Submission manifest seed mismatch for {path}: "
            f"{payload.get('seed')} != {expected_seed}"
        )
    if not payload.get("submitted"):
        raise ValueError(f"Formal aggregation rejects dry-run manifest: {path}")
    if payload.get("error"):
        raise ValueError(f"Submission manifest records an error: {path}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError(f"Submission manifest is missing jobs: {path}")
    missing = sorted(set(REQUIRED_COMPARISON_JOBS) - set(jobs))
    if missing:
        raise ValueError(f"Submission manifest is missing comparison jobs: {missing}")
    for name in REQUIRED_COMPARISON_JOBS:
        job_id = str(jobs[name].get("job_id", ""))
        if not job_id.isdigit():
            raise ValueError(f"Invalid submitted job ID for {name}: {job_id!r}")
    return payload


def _job_id(manifest: Mapping[str, Any], name: str) -> str:
    return str(manifest["jobs"][name]["job_id"])


def comparison_cells(manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve all comparison artifacts produced by one seed DAG."""

    output_root = Path(str(manifest["user_root"])) / "outputs"
    cells: dict[str, Path] = {}

    fixed_matched = output_root / (
        f"compare-controlled-matched-{_job_id(manifest, 'compare_fixed_matched')}"
    )
    oracle_matched = output_root / (
        f"compare-latebound-oracle-{_job_id(manifest, 'compare_oracle_matched')}"
    )
    for address in ADDRESS_STATUSES:
        for size in MATCHED_REGISTRY_SIZES:
            suffix = Path(f"seen-tool_{address}-address") / f"registry-{size}"
            cells[f"fixed_vs_latebound/{address}-address/registry-{size}"] = (
                fixed_matched / suffix / "comparison.json"
            )
            oracle_suffix = (
                Path(f"unseen-tool_{address}-address") / f"registry-{size}"
            )
            cells[f"oracle_vs_latebound/{address}-address/registry-{size}"] = (
                oracle_matched / oracle_suffix / "comparison.json"
            )

    incremental_matched = output_root / (
        "compare-incremental-latebound-matched-"
        f"{_job_id(manifest, 'compare_incremental_matched')}"
    )
    for size in MATCHED_REGISTRY_SIZES:
        suffix = Path("unseen-tool_unseen-address") / f"registry-{size}"
        cells[f"incremental_vs_latebound/unseen-address/registry-{size}"] = (
            incremental_matched / suffix / "comparison.json"
        )

    fixed_scale = output_root / (
        f"compare-controlled-scale-{_job_id(manifest, 'compare_fixed_scale')}"
    )
    oracle_scale = output_root / (
        f"compare-latebound-oracle-scale-{_job_id(manifest, 'compare_oracle_scale')}"
    )
    for size in SCALE_REGISTRY_SIZES:
        fixed_suffix = Path("seen-tool_unseen-address") / f"registry-{size}"
        cells[f"fixed_vs_latebound/unseen-address/registry-{size}"] = (
            fixed_scale / fixed_suffix / "comparison.json"
        )
        oracle_suffix = Path("unseen-tool_unseen-address") / f"registry-{size}"
        cells[f"oracle_vs_latebound/unseen-address/registry-{size}"] = (
            oracle_scale / oracle_suffix / "comparison.json"
        )

    controls = output_root / (
        f"compare-latebound-controls-{_job_id(manifest, 'controls_compare')}"
    )
    for control in CONTROLS:
        cells[f"registered_vs_control/{control}"] = (
            controls / f"control-{control}" / "comparison.json"
        )

    append = output_root / (
        f"compare-latebound-append-{_job_id(manifest, 'append_compare')}"
    )
    cells["append/registry-100-to-1000"] = append / "comparison.json"

    if len(cells) != EXPECTED_CELL_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_CELL_COUNT} comparison cells, resolved {len(cells)}"
        )
    return cells


def aggregate_formal_seed_pipelines(
    manifests: Mapping[int, Path],
) -> dict[str, Any]:
    """Validate three pipeline manifests and aggregate all corresponding cells."""

    if set(manifests) != set(FORMAL_SEEDS):
        raise ValueError(f"Manifests must contain exactly seeds {FORMAL_SEEDS}")

    payloads: dict[int, dict[str, Any]] = {}
    cells_by_seed: dict[int, dict[str, Path]] = {}
    inputs: dict[str, Any] = {}
    for seed in FORMAL_SEEDS:
        path = manifests[seed]
        payload = load_submission_manifest(path, seed)
        payloads[seed] = payload
        cells_by_seed[seed] = comparison_cells(payload)
        inputs[str(seed)] = {
            "submission_manifest_path": str(path.resolve()),
            "submission_manifest_sha256": sha256_file(path),
        }

    reference_names = set(cells_by_seed[FORMAL_SEEDS[0]])
    if any(set(cells_by_seed[seed]) != reference_names for seed in FORMAL_SEEDS):
        raise ValueError("Formal seed pipelines resolve different comparison cells")

    cells: dict[str, Any] = {}
    for name in sorted(reference_names):
        comparisons = {seed: cells_by_seed[seed][name] for seed in FORMAL_SEEDS}
        for seed, path in comparisons.items():
            if not (path.parent / "COMPLETE").is_file():
                raise FileNotFoundError(
                    f"Incomplete comparison cell {name} for seed {seed}: {path.parent}"
                )
            if not path.is_file():
                raise FileNotFoundError(path)
        cells[name] = aggregate_formal_seed_comparisons(comparisons)

    return {
        "kind": "formal_three_seed_pipeline_aggregate",
        "version": 1,
        "seeds": list(FORMAL_SEEDS),
        "cell_count": len(cells),
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "all_cells_complete": len(cells) == EXPECTED_CELL_COUNT,
        "cells": cells,
        "inputs": inputs,
    }


def write_pipeline_aggregate(output_dir: Path, result: dict[str, Any]) -> None:
    cells = result.get("cells", {})
    if not isinstance(cells, dict) or len(cells) != EXPECTED_CELL_COUNT:
        raise ValueError("Refusing to write an incomplete formal pipeline aggregate")
    for name, payload in cells.items():
        write_json_atomic(output_dir / "cells" / f"{name}.json", payload)
    aggregate_path = output_dir / "aggregate.json"
    write_json_atomic(aggregate_path, result)
    digest = sha256_file(aggregate_path)
    complete_path = output_dir / "COMPLETE"
    temporary = complete_path.with_suffix(".tmp")
    temporary.write_text(f"{digest}  aggregate.json\n", encoding="utf-8")
    temporary.replace(complete_path)


def _parse_manifest(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Manifest must be SEED=/path/to/manifest.json")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed: {seed_text}") from exc
    return seed, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate all 27 comparison cells across three formal seeds"
    )
    parser.add_argument("--manifest", action="append", type=_parse_manifest, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifests = dict(args.manifest)
    if len(manifests) != len(args.manifest):
        raise ValueError("Duplicate manifest seed")
    result = aggregate_formal_seed_pipelines(manifests)
    write_pipeline_aggregate(args.output_dir, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
