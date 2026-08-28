import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from latent_register.aggregate_formal_pipeline import (
    EXPECTED_CELL_COUNT,
    aggregate_formal_seed_pipelines,
    comparison_cells,
    write_pipeline_aggregate,
)


JOB_NAMES = (
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


class AggregateFormalPipelineTests(unittest.TestCase):
    def write_manifest(self, root: Path, seed: int) -> Path:
        user_root = root / f"user-{seed}"
        jobs = {
            name: {"job_id": str(seed * 100 + index)}
            for index, name in enumerate(JOB_NAMES)
        }
        payload = {
            "kind": "formal_controlled_pipeline_submission",
            "seed": seed,
            "submitted": True,
            "user_root": str(user_root),
            "jobs": jobs,
        }
        path = root / f"seed-{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_resolves_all_expected_comparison_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self.write_manifest(root, 17)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cells = comparison_cells(manifest)

        self.assertEqual(len(cells), EXPECTED_CELL_COUNT)
        self.assertIn("fixed_vs_latebound/seen-address/registry-10", cells)
        self.assertIn("oracle_vs_latebound/unseen-address/registry-47000", cells)
        self.assertIn(
            "incremental_vs_latebound/unseen-address/registry-1000", cells
        )
        self.assertIn("registered_vs_control/nearest_trained", cells)
        self.assertIn("append/registry-100-to-1000", cells)

    def test_aggregates_every_cell_and_writes_completion_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {
                seed: self.write_manifest(root, seed) for seed in (17, 29, 43)
            }
            for seed, manifest_path in manifests.items():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for path in comparison_cells(manifest).values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}\n", encoding="utf-8")
                    (path.parent / "COMPLETE").touch()

            def fake_aggregate(paths):
                return {
                    "kind": "test-cell",
                    "seeds": sorted(paths),
                    "paths": {str(seed): str(path) for seed, path in paths.items()},
                }

            with mock.patch(
                "latent_register.aggregate_formal_pipeline."
                "aggregate_formal_seed_comparisons",
                side_effect=fake_aggregate,
            ) as aggregate:
                result = aggregate_formal_seed_pipelines(manifests)
            output = root / "aggregate"
            write_pipeline_aggregate(output, result)

            self.assertEqual(aggregate.call_count, EXPECTED_CELL_COUNT)
            self.assertEqual(result["cell_count"], EXPECTED_CELL_COUNT)
            self.assertTrue(result["all_cells_complete"])
            self.assertTrue((output / "aggregate.json").is_file())
            self.assertTrue((output / "COMPLETE").is_file())
            self.assertEqual(
                len(list((output / "cells").rglob("*.json"))),
                EXPECTED_CELL_COUNT,
            )

    def test_rejects_dry_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {
                seed: self.write_manifest(root, seed) for seed in (17, 29, 43)
            }
            payload = json.loads(manifests[29].read_text(encoding="utf-8"))
            payload["submitted"] = False
            manifests[29].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dry-run"):
                aggregate_formal_seed_pipelines(manifests)


if __name__ == "__main__":
    unittest.main()
