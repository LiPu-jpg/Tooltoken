import json
import tempfile
import unittest
from pathlib import Path

from latent_register.assess_formal_readiness import (
    OUTPUT_CONTROLS,
    REQUIRED_CONTROLS,
    assess_formal_readiness,
)


class AssessFormalReadinessTests(unittest.TestCase):
    def write_fixture(self, root: Path, *, key_recall: float = 0.7):
        registered_dir = root / "registered"
        registered_dir.mkdir()
        (registered_dir / "COMPLETE").touch()
        registered = {
            "kind": "qwen_late_bound",
            "code_hashes": {
                "evaluate_late_bound": "a" * 64,
                "benchmark_metrics": "b" * 64,
                "model": "c" * 64,
                "physical_tokens": "d" * 64,
            },
            "condition": "unseen_tool_unseen_address",
            "source_condition": "unseen_tool_unseen_token",
            "address_status": "unseen",
            "registration_control": "registered",
            "registry_size": 1000,
            "information_condition": "common_document",
            "predictions_sha256": "registered-predictions",
            "aggregates": {
                "arguments:unseen_tool_unseen_address": {
                    "count": 1169,
                    "metrics": {
                        "end_to_end_key_exact": 0.4,
                        "end_to_end_key_recall": key_recall,
                    },
                },
                "retrieval:unseen_tool_unseen_address": {
                    "count": 2002,
                    "metrics": {"hit_at_1": 0.92},
                },
            },
            "registry_constrained_retrieval": {"metrics": {"hit_at_1": 0.96}},
            "registration_audit": {
                "optimizer_steps": 0,
                "model_or_table_parameters_changed": 0,
                "registration_forwards_per_tool": 1,
                "selection_emits_single_physical_id": True,
                "selected_physical_bindings_verified": True,
                "selected_physical_binding_count": 2500,
                "selected_physical_binding_verified_count": 2500,
                "selected_argument_payload_count": 1000,
                "selected_argument_payload_verified_count": 1000,
                "selected_id_dereferences_registered_memory": False,
                "selected_id_dereferences_full_document": True,
                "selected_id_uses_static_embedding": False,
                "ordinary_vocabulary_competes_at_selection": True,
                "inactive_reserved_ids_masked": True,
            },
            "evaluation_mutation_audit": {
                "all_reserved_input_row_max_change": 0.0,
                "all_reserved_output_row_max_change": 0.0,
            },
            "physical_token_audit": {
                "total_reserved_tokens": 8192,
                "atomic_tokenization_verified": True,
                "distinct_reserved_token_ids": 8192,
                "reserved_token_ids_contiguous": True,
                "reserved_token_id_mapping_sha256": "f" * 64,
                "evaluation_ids_seen_during_training": 0,
                "evaluation_input_row_max_change": 0.0,
                "evaluation_output_row_max_change": 0.0,
            },
        }
        registered_path = registered_dir / "results.json"
        registered_path.write_text(json.dumps(registered), encoding="utf-8")

        controls = {}
        for control in REQUIRED_CONTROLS:
            directory = root / control
            directory.mkdir()
            (directory / "COMPLETE").touch()
            families = {
                "arguments": {
                    "paired_metrics": {
                        "end_to_end_key_recall": {
                            "delta": 0.2,
                            "paired_bootstrap_95_ci": [0.1, 0.3],
                        }
                    }
                }
            }
            if control in OUTPUT_CONTROLS:
                families["retrieval"] = {
                    "paired_metrics": {
                        "hit_at_1": {
                            "count": 2002,
                            "delta": 0.3,
                            "paired_bootstrap_95_ci": [0.2, 0.4],
                        }
                    }
                }
            comparison = {
                "families": families,
                "sources": {
                    "candidate_sha256": "registered-predictions",
                    "compare_controlled_predictions_code_sha256": "e" * 64,
                    "benchmark_metrics_code_sha256": "b" * 64,
                },
            }
            path = directory / "comparison.json"
            path.write_text(json.dumps(comparison), encoding="utf-8")
            controls[control] = path
        return registered_path, controls

    def test_passes_complete_positive_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            result = assess_formal_readiness(registered, controls)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_checks"], [])
        self.assertEqual(len(result["controls"]), 7)

    def test_low_memory_only_key_recall_is_diagnostic_not_a_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(
                Path(directory), key_recall=0.01
            )
            result = assess_formal_readiness(registered, controls)
        self.assertTrue(result["passed"])

    def test_fails_when_registered_selection_is_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            payload = json.loads(registered.read_text(encoding="utf-8"))
            payload["aggregates"]["retrieval:unseen_tool_unseen_address"][
                "metrics"
            ]["hit_at_1"] = 0.0
            registered.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("registered_selection_is_nonzero", result["failed_checks"])

    def test_fails_when_selection_does_not_beat_a_control_significantly(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            path = controls["random"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            metric = payload["families"]["retrieval"]["paired_metrics"]["hit_at_1"]
            metric["delta"] = 0.01
            metric["paired_bootstrap_95_ci"] = [-0.01, 0.03]
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("control_random", result["failed_checks"])

    def test_fails_when_control_uses_a_smaller_retrieval_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            path = controls["random"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            metric = payload["families"]["retrieval"]["paired_metrics"]["hit_at_1"]
            metric["count"] = 1999
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("control_random", result["failed_checks"])
        self.assertFalse(
            result["controls"]["random"][
                "paired_retrieval_count_matches_registered"
            ]
        )

    def test_rejects_control_built_from_different_registered_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            path = controls["blank"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sources"]["candidate_sha256"] = "different"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not use"):
                assess_formal_readiness(registered, controls)

    def test_fails_when_control_comparison_code_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            path = controls["blank"]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sources"]["benchmark_metrics_code_sha256"] = "f" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("scoring_code_consistency", result["failed_checks"])

    def test_fails_when_reserved_tokens_are_not_atomic_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            payload = json.loads(registered.read_text(encoding="utf-8"))
            payload["physical_token_audit"]["distinct_reserved_token_ids"] = 8191
            registered.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("registration_contract", result["failed_checks"])

    def test_selected_id_must_dereference_its_full_document(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            payload = json.loads(registered.read_text(encoding="utf-8"))
            payload["registration_audit"][
                "selected_id_dereferences_full_document"
            ] = False
            registered.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("registration_contract", result["failed_checks"])

    def test_selected_id_must_verify_its_physical_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            payload = json.loads(registered.read_text(encoding="utf-8"))
            payload["registration_audit"][
                "selected_physical_bindings_verified"
            ] = False
            registered.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("registration_contract", result["failed_checks"])

    def test_selected_binding_counts_must_be_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            registered, controls = self.write_fixture(Path(directory))
            payload = json.loads(registered.read_text(encoding="utf-8"))
            payload["registration_audit"][
                "selected_argument_payload_verified_count"
            ] -= 1
            registered.write_text(json.dumps(payload), encoding="utf-8")
            result = assess_formal_readiness(registered, controls)
        self.assertFalse(result["passed"])
        self.assertIn("registration_contract", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
