import tempfile
import unittest
from pathlib import Path

from latent_register.convert_toolgen_retrieval import (
    convert_result,
    load_valid_actions,
    recover_unique_logs,
)


class ConvertToolGenRetrievalTest(unittest.TestCase):
    def test_recovers_three_identical_blocks(self):
        block = [
            {"label": ["a", "b"], "pred": ["a", "x"]},
            {"label": ["c"], "pred": ["x", "c"]},
        ]
        self.assertEqual(recover_unique_logs(block * 3), block)

    def test_rejects_nonidentical_blocks(self):
        with self.assertRaisesRegex(ValueError, "not identical"):
            recover_unique_logs(
                [
                    {"label": ["a"], "pred": ["a"]},
                    {"label": ["a"], "pred": ["x"]},
                    {"label": ["a"], "pred": ["a"]},
                ]
            )

    def test_conversion_preserves_official_and_corrected_metrics(self):
        block = [{"label": ["a", "b"], "pred": ["a", "x", ""]}]
        records, summary = convert_result(
            {
                "model": "toolgen",
                "ndcg": {"ndcg@3": 1.0},
                "logs": block * 3,
            }
        )
        self.assertEqual(records[0]["predicted_tools"], ["a", "x"])
        self.assertEqual(summary["official_ndcg"], {"ndcg@3": 1.0})
        self.assertLess(summary["metrics"]["ndcg_corrected_at_3"], 1.0)
        self.assertEqual(summary["metrics"]["ndcg_paper_compatible_at_3"], 1.0)

    def test_filters_nonempty_predictions_outside_official_action_vocab(self):
        block = [
            {
                "label": ["<<valid&&target>>"],
                "pred": ["<<invalid&&decoded>>", "<<valid&&target>>", ""],
            }
        ]
        records, summary = convert_result(
            {
                "model": "toolgen",
                "ndcg": {"ndcg@1": 1.0, "ndcg@3": 1.0, "ndcg@5": 1.0},
                "logs": block * 3,
            },
            valid_actions={"<<valid&&target>>"},
        )
        self.assertEqual(records[0]["predicted_tools"], ["<<valid&&target>>"])
        self.assertEqual(records[0]["invalid_predictions"], ["<<invalid&&decoded>>"])
        self.assertEqual(summary["metrics"]["hit_at_1"], 1.0)
        self.assertEqual(summary["raw_prediction_count"], 2)
        self.assertEqual(summary["invalid_prediction_count"], 1)
        self.assertEqual(summary["invalid_prediction_rate"], 0.5)

    def test_load_valid_actions_accepts_mapping_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text('{"tool": "<<tool&&api>>"}\n', encoding="utf-8")
            self.assertEqual(load_valid_actions(path), {"<<tool&&api>>"})


if __name__ == "__main__":
    unittest.main()
