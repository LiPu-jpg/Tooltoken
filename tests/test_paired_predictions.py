import unittest

from latent_register.paired_predictions import (
    compare_predictions,
    exact_mcnemar_pvalue,
)


class PairedPredictionTests(unittest.TestCase):
    def test_reports_paired_improvements_and_regressions(self) -> None:
        baseline = [
            {
                "condition": "trajectory",
                "example_id": "1",
                "tool_name": "weather",
                "selected_tool": "weather",
                "target": {"city": "Suzhou"},
                "parsed": {"city": "Suzhou"},
            },
            {
                "condition": "trajectory",
                "example_id": "2",
                "tool_name": "maps",
                "selected_tool": "weather",
                "target": {"city": "Nanjing"},
                "parsed": {"city": "Nanjing"},
            },
        ]
        candidate = [
            {**baseline[0], "parsed": {"city": "Shanghai"}},
            {**baseline[1], "selected_tool": "maps"},
        ]

        result = compare_predictions(baseline, candidate)["trajectory"]

        self.assertEqual(result["arguments_exact"]["baseline_only"], 1)
        self.assertEqual(result["selection_correct"]["candidate_only"], 1)
        self.assertEqual(result["end_to_end_exact"]["baseline_only"], 1)
        self.assertEqual(result["end_to_end_exact"]["candidate_only"], 1)

    def test_exact_mcnemar_handles_no_disagreement(self) -> None:
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)

    def test_selection_outcome_uses_document_identity_when_present(self) -> None:
        baseline = [
            {
                "condition": "trajectory",
                "example_id": "1",
                "tool_name": "weather",
                "tool_id": "weather::schema-a",
                "selected_tool": "weather",
                "selected_tool_id": "weather::schema-b",
                "target": {"city": "Suzhou"},
                "parsed": {"city": "Suzhou"},
            }
        ]
        candidate = [
            {**baseline[0], "selected_tool_id": "weather::schema-a"}
        ]

        outcomes = compare_predictions(baseline, candidate)["trajectory"]

        self.assertEqual(outcomes["selection_correct"]["candidate_only"], 1)

    def test_rejects_mismatched_targets(self) -> None:
        baseline = [
            {
                "condition": "registered",
                "example_id": "1",
                "tool_name": "weather",
                "target": {"city": "Suzhou"},
                "parsed": None,
            }
        ]
        candidate = [{**baseline[0], "target": {"city": "Nanjing"}}]
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            compare_predictions(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
