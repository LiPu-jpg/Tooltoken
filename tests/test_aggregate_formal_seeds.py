import json
import tempfile
import unittest
from pathlib import Path

from latent_register.aggregate_formal_seeds import aggregate_formal_seed_comparisons


class AggregateFormalSeedsTests(unittest.TestCase):
    def write_seed(
        self,
        root: Path,
        seed: int,
        *,
        changed_query: bool = False,
        changed_code: bool = False,
    ) -> Path:
        seed_root = root / str(seed)
        seed_root.mkdir()
        reference = {
            "family": "retrieval",
            "example_id": "example-1",
            "query": "changed" if changed_query else "find weather",
            "reference_tools": ["weather"],
            "reference_arguments": None,
            "schema": None,
            "selected_tool": "weather",
        }
        baseline_row = {
            **reference,
            "condition": "unseen_tool_unseen_token",
        }
        candidate_row = {
            **reference,
            "source_condition": "unseen_tool_unseen_token",
            "condition": "unseen_tool_unseen_address",
        }
        baseline = seed_root / "baseline.jsonl"
        candidate = seed_root / "candidate.jsonl"
        baseline.write_text(json.dumps(baseline_row) + "\n", encoding="utf-8")
        candidate.write_text(json.dumps(candidate_row) + "\n", encoding="utf-8")
        delta = {17: 0.1, 29: 0.2, 43: 0.3}[seed]
        comparison = {
            "pair_count": 1,
            "require_registry_match": True,
            "registry_pairs": 1,
            "registry_mismatches": 0,
            "selected_families": ["all"],
            "families": {
                "retrieval": {
                    "paired_metrics": {
                        "selection_hit": {
                            "count": 1,
                            "baseline": 0.4,
                            "candidate": 0.4 + delta,
                            "delta": delta,
                            "paired_bootstrap_95_ci": [delta, delta],
                        }
                    }
                }
            },
            "sources": {
                "baseline_path": str(baseline),
                "candidate_path": str(candidate),
                "compare_controlled_predictions_code_sha256": (
                    "c" * 64 if changed_code else "a" * 64
                ),
                "benchmark_metrics_code_sha256": "b" * 64,
            },
        }
        path = seed_root / "comparison.json"
        path.write_text(json.dumps(comparison), encoding="utf-8")
        return path

    def test_aggregates_seed_level_delta_and_preserves_bootstrap_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {seed: self.write_seed(root, seed) for seed in (17, 29, 43)}
            result = aggregate_formal_seed_comparisons(paths)
        metric = result["families"]["retrieval"]["metrics"]["selection_hit"]
        self.assertAlmostEqual(metric["delta"]["mean"], 0.2)
        self.assertAlmostEqual(metric["delta"]["sample_std"], 0.1)
        self.assertEqual(metric["paired_bootstrap_95_ci_by_seed"]["29"], [0.2, 0.2])

    def test_rejects_cross_seed_reference_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                seed: self.write_seed(root, seed, changed_query=seed == 43)
                for seed in (17, 29, 43)
            }
            with self.assertRaisesRegex(ValueError, "reference fingerprints differ"):
                aggregate_formal_seed_comparisons(paths)

    def test_rejects_cross_seed_comparison_code_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                seed: self.write_seed(root, seed, changed_code=seed == 43)
                for seed in (17, 29, 43)
            }
            with self.assertRaisesRegex(ValueError, "code hashes differ"):
                aggregate_formal_seed_comparisons(paths)


if __name__ == "__main__":
    unittest.main()
