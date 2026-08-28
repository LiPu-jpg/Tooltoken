import math
import unittest

from latent_register.benchmark_metrics import (
    aggregate_records,
    canonical_json,
    flatten_json,
    ndcg_at_k,
    paired_bootstrap_interval,
    score_arguments,
    score_retrieval,
)


class BenchmarkMetricsTest(unittest.TestCase):
    def test_canonical_json_ignores_object_key_order(self):
        self.assertEqual(
            canonical_json({"unit": "C", "city": "Paris"}),
            canonical_json({"city": "Paris", "unit": "C"}),
        )

    def test_flatten_json_tracks_nested_arrays(self):
        self.assertEqual(
            flatten_json({"trip": {"cities": ["Paris", "Rome"]}}),
            {"trip.cities[0]": "Paris", "trip.cities[1]": "Rome"},
        )

    def test_argument_metrics_are_structural(self):
        scores = score_arguments(
            '{"unit":"C","city":"Paris"}',
            {"city": "Paris", "unit": "C"},
        )
        self.assertEqual(scores["json_valid"], 1.0)
        self.assertEqual(scores["argument_exact"], 1.0)
        self.assertEqual(scores["key_exact"], 1.0)
        self.assertEqual(scores["key_f1"], 1.0)
        self.assertEqual(scores["value_exact"], 1.0)

    def test_invalid_json_is_a_failure(self):
        scores = score_arguments("{broken", {"city": "Paris"})
        self.assertEqual(scores["json_valid"], 0.0)
        self.assertEqual(scores["argument_exact"], 0.0)
        self.assertEqual(scores["key_exact"], 0.0)
        self.assertEqual(scores["key_recall"], 0.0)

    def test_key_exact_separates_schema_shape_from_values(self):
        same_keys = score_arguments(
            {"city": "wrong", "unit": "F"},
            {"city": "Paris", "unit": "C"},
        )
        missing_key = score_arguments(
            {"city": "Paris"},
            {"city": "Paris", "unit": "C"},
        )
        self.assertEqual(same_keys["key_exact"], 1.0)
        self.assertEqual(same_keys["argument_exact"], 0.0)
        self.assertEqual(missing_key["key_exact"], 0.0)

    def test_schema_validation(self):
        schema = {
            "type": "object",
            "properties": {"unit": {"type": "string", "enum": ["C", "F"]}},
            "required": ["unit"],
            "additionalProperties": False,
        }
        valid = score_arguments({"unit": "C"}, {"unit": "C"}, schema)
        invalid = score_arguments({"unit": "K"}, {"unit": "C"}, schema)
        self.assertEqual(valid["schema_valid"], 1.0)
        self.assertEqual(invalid["schema_valid"], 0.0)

    def test_corrected_ndcg_penalizes_missing_relevant_tools(self):
        labels = ["a", "b", "c"]
        predictions = ["a", "x", "y"]
        corrected = ndcg_at_k(labels, predictions, 3)
        paper = ndcg_at_k(labels, predictions, 3, paper_compatible=True)
        self.assertLess(corrected, paper)
        self.assertEqual(paper, 1.0)

    def test_retrieval_metrics(self):
        scores = score_retrieval(["x", "b", "a"], ["a", "b"])
        self.assertEqual(scores["hit_at_1"], 0.0)
        self.assertEqual(scores["recall_at_5"], 1.0)
        self.assertEqual(scores["mrr"], 0.5)

    def test_aggregate_skips_inapplicable_schema_metric(self):
        result = aggregate_records(
            [
                {
                    "reference_tools": ["weather"],
                    "predicted_tools": ["weather"],
                    "reference_arguments": {"city": "Paris"},
                    "predicted_arguments": {"city": "Paris"},
                }
            ]
        )
        self.assertEqual(result["count"], 1)
        self.assertTrue(math.isnan(result["metrics"]["schema_valid"]))
        self.assertEqual(result["metrics"]["argument_exact"], 1.0)
        self.assertEqual(result["metrics"]["end_to_end_argument_exact"], 1.0)

    def test_end_to_end_metrics_require_correct_tool_and_arguments(self):
        result = aggregate_records(
            [
                {
                    "reference_tools": ["weather"],
                    "predicted_tools": ["calendar"],
                    "reference_arguments": {"city": "Paris"},
                    "predicted_arguments": {"city": "Paris"},
                }
            ]
        )["metrics"]
        self.assertEqual(result["argument_exact"], 1.0)
        self.assertEqual(result["key_exact"], 1.0)
        self.assertEqual(result["end_to_end_argument_exact"], 0.0)
        self.assertEqual(result["end_to_end_key_exact"], 0.0)
        self.assertEqual(result["end_to_end_key_recall"], 0.0)

    def test_paired_bootstrap_is_deterministic(self):
        first = paired_bootstrap_interval([1.0, 0.0, 1.0], samples=100, seed=4)
        second = paired_bootstrap_interval([1.0, 0.0, 1.0], samples=100, seed=4)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
