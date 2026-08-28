import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from latent_register.compare_controlled_predictions import (
    compare_controlled_predictions,
    main,
)


def _row(example_id, predicted_tools, predicted_arguments, registry_hash="same"):
    return {
        "family": "arguments",
        "condition": "seen_tool_seen_token",
        "example_id": str(example_id),
        "query": "weather",
        "reference_tools": ["tool-a"],
        "reference_arguments": {"city": "Suzhou"},
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "predicted_tools": predicted_tools,
        "predicted_arguments": predicted_arguments,
        "registry_identity_sha256": registry_hash,
    }


class CompareControlledPredictionsTests(unittest.TestCase):
    def test_reports_paired_metric_deltas_and_registry_audit(self):
        baseline = [
            _row(1, ["tool-a"], '{"city":"Suzhou"}'),
            _row(2, ["tool-b"], "bad-json"),
        ]
        candidate = [
            {**_row(1, ["tool-a"], '{"city":"Suzhou"}'), "source_condition": "seen_tool_seen_token"},
            {**_row(2, ["tool-a"], '{"city":"Suzhou"}'), "source_condition": "seen_tool_seen_token"},
        ]
        result = compare_controlled_predictions(
            baseline,
            candidate,
            require_registry_match=True,
            bootstrap_samples=100,
        )
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["registry_pairs"], 2)
        self.assertEqual(result["registry_mismatches"], 0)
        metrics = result["families"]["arguments"]["paired_metrics"]
        self.assertEqual(metrics["argument_exact"]["delta"], 0.5)
        self.assertEqual(metrics["argument_exact"]["candidate_over_baseline"], 2.0)
        self.assertEqual(metrics["end_to_end_argument_exact"]["delta"], 0.5)
        self.assertEqual(metrics["end_to_end_key_exact"]["delta"], 0.5)
        self.assertEqual(metrics["hit_at_1"]["delta"], 0.5)
        self.assertEqual(metrics["json_valid"]["delta"], 0.5)

    def test_rejects_registry_mismatch(self):
        baseline = [_row(1, ["tool-a"], "{}", registry_hash="left")]
        candidate = [_row(1, ["tool-a"], "{}", registry_hash="right")]
        with self.assertRaisesRegex(ValueError, "Registry identity mismatch"):
            compare_controlled_predictions(
                baseline,
                candidate,
                require_registry_match=True,
                bootstrap_samples=10,
            )

    def test_rejects_reference_mismatch(self):
        baseline = [_row(1, ["tool-a"], "{}")]
        candidate = [{**_row(1, ["tool-a"], "{}"), "query": "maps"}]
        with self.assertRaisesRegex(ValueError, "Reference mismatch"):
            compare_controlled_predictions(
                baseline,
                candidate,
                require_registry_match=True,
                bootstrap_samples=10,
            )

    def test_rejects_different_agents_in_common_document_comparison(self):
        common = {
            "information_condition": "common_document",
            "common_document_agent_model_path": "/models/shared-agent",
            "common_document_agent_audit_sha256": "a" * 64,
        }
        baseline = [{**_row(1, ["tool-a"], "{}"), **common}]
        candidate = [
            {
                **_row(1, ["tool-a"], "{}"),
                **common,
                "common_document_agent_audit_sha256": "b" * 64,
            }
        ]
        with self.assertRaisesRegex(ValueError, "Agent mismatch"):
            compare_controlled_predictions(
                baseline,
                candidate,
                require_registry_match=True,
                bootstrap_samples=10,
            )

    def test_allows_same_agent_common_document_against_correct_document_oracle(self):
        shared = {
            "common_document_agent_model_path": "/models/shared-agent",
            "common_document_agent_audit_sha256": "a" * 64,
        }
        baseline = [
            {
                **_row(1, ["tool-a"], '{}'),
                **shared,
                "information_condition": "full_document_oracle",
            }
        ]
        candidate = [
            {
                **_row(1, ["tool-a"], '{}'),
                **shared,
                "information_condition": "common_document",
            }
        ]
        result = compare_controlled_predictions(
            baseline,
            candidate,
            require_registry_match=True,
            bootstrap_samples=10,
        )
        self.assertEqual(result["pair_count"], 1)

    def test_cli_embeds_runtime_comparison_code_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            output = root / "comparison.json"
            baseline.write_text(json.dumps(_row(1, ["tool-a"], "{}")) + "\n")
            candidate.write_text(json.dumps(_row(1, ["tool-a"], "{}")) + "\n")
            argv = [
                "compare_controlled_predictions",
                str(baseline),
                str(candidate),
                "--output",
                str(output),
                "--bootstrap-samples",
                "10",
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                io.StringIO()
            ):
                main()
            sources = json.loads(output.read_text(encoding="utf-8"))["sources"]

        self.assertEqual(
            len(sources["compare_controlled_predictions_code_sha256"]), 64
        )
        self.assertEqual(len(sources["benchmark_metrics_code_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
