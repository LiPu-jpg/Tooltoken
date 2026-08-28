import json
import tempfile
import unittest
from pathlib import Path

from latent_register.aggregate_toolgen_retrieval import (
    aggregate_runs,
    compare_paper_table1_multi_domain,
    parse_tasks,
    validate_official_ndcg,
)


class AggregateToolGenRetrievalTest(unittest.TestCase):
    def test_parse_tasks_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.tsv"
            path.write_text("G1\tinstruction\tTrue\n" * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                parse_tasks(path)

    def test_rejects_metric_reproduction_mismatch(self):
        with self.assertRaisesRegex(ValueError, "differ at k=3"):
            validate_official_ndcg(
                {
                    "official_ndcg": {
                        "ndcg@1": 0.5,
                        "ndcg@3": 0.6,
                        "ndcg@5": 0.7,
                    },
                    "metrics": {
                        "ndcg_paper_compatible_at_1": 0.5,
                        "ndcg_paper_compatible_at_3": 0.61,
                        "ndcg_paper_compatible_at_5": 0.7,
                    },
                }
            )

    def test_aggregates_complete_runs_with_micro_and_macro_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = root / "tasks.tsv"
            task_file.write_text(
                "G1\tinstruction\tTrue\nG1\tinstruction\tFalse\n",
                encoding="utf-8",
            )
            audit = root / "audit.json"
            audit.write_text('{"version":2}\n', encoding="utf-8")
            for domain, count, score in (
                ("in_domain", 1, 1.0),
                ("multi_domain", 3, 0.0),
            ):
                run = root / domain / "G1_instruction"
                run.mkdir(parents=True)
                (run / "COMPLETE").touch()
                (run / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
                (run / "metrics.json").write_text(
                    json.dumps(
                        {
                            "count": count,
                            "official_ndcg": {
                                "ndcg@1": score,
                                "ndcg@3": score,
                                "ndcg@5": score,
                            },
                            "metrics": {
                                "hit_at_1": score,
                                "ndcg_paper_compatible_at_1": score,
                                "ndcg_paper_compatible_at_3": score,
                                "ndcg_paper_compatible_at_5": score,
                            },
                            "raw_prediction_count": count * 5,
                            "invalid_prediction_count": count,
                        }
                    ),
                    encoding="utf-8",
                )
            result = aggregate_runs(
                root=root,
                task_file=task_file,
                tokenizer_audit=audit,
                expected_task_count=2,
            )
            self.assertEqual(result["task_count"], 2)
            self.assertEqual(result["macro_metrics"]["hit_at_1"], 0.5)
            self.assertEqual(result["micro_metrics"]["hit_at_1"], 0.25)
            self.assertEqual(result["raw_prediction_count"], 20)
            self.assertEqual(result["invalid_prediction_count"], 4)
            self.assertEqual(result["invalid_prediction_rate"], 0.2)
            self.assertTrue(result["released_evaluator_recomputation_passed"])
            self.assertFalse(result["paper_table1_multi_domain"]["complete"])

    def test_compares_released_checkpoint_with_paper_multi_domain_rows(self):
        runs = []
        values = {
            "G1": (0.895, 0.9022590751349471, 0.9296549692730405),
            "G2": (0.84, 0.8626014331713436, 0.8916126839234185),
            "G3": (0.78, 0.7970465190950837, 0.8469445856294179),
        }
        for stage, scores in values.items():
            runs.append(
                {
                    "stage": stage,
                    "split": "instruction",
                    "domain": "multi_domain",
                    "metrics": {
                        f"ndcg_paper_compatible_at_{k}": score
                        for k, score in zip((1, 3, 5), scores)
                    },
                }
            )
        comparison = compare_paper_table1_multi_domain(runs)
        self.assertTrue(comparison["complete"])
        self.assertAlmostEqual(
            comparison["macro"]["ndcg_at_1"]["difference_points"],
            0.45666666666666233,
        )
        self.assertAlmostEqual(
            comparison["max_absolute_cell_difference_points"], 1.83
        )


if __name__ == "__main__":
    unittest.main()
