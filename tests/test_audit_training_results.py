import json
import tempfile
import unittest
from pathlib import Path

from latent_register.audit_training_results import audit_training_results


def valid_results() -> dict:
    return {
        "config": {"seed": 29},
        "completed_steps": 1000,
        "world_size": 6,
        "recent_train_loss": 0.5,
        "physical_token_audit": {
            "total_reserved_tokens": 8192,
            "atomic_tokenization_verified": True,
            "distinct_reserved_token_ids": 8192,
            "reserved_token_ids_contiguous": True,
            "reserved_token_id_mapping_sha256": "a" * 64,
            "evaluation_ids_seen_during_training": 0,
            "evaluation_input_row_max_change": 0.0,
            "evaluation_output_row_max_change": 0.0,
        },
    }


class AuditTrainingResultsTests(unittest.TestCase):
    def write_results(self, directory: str, payload: dict) -> Path:
        path = Path(directory) / "results.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_accepts_exact_seed_step_world_and_isolation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = audit_training_results(
                self.write_results(directory, valid_results()),
                expected_seed=29,
                expected_steps=1000,
                expected_world_size=6,
                stage="retrieval",
            )
        self.assertTrue(audit["physical_token_isolation"])
        self.assertTrue(audit["physical_token_identity"]["passed"])
        self.assertTrue(audit["all_numeric_results_finite"])

    def test_rejects_configuration_or_isolation_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_results()
            payload["physical_token_audit"]["evaluation_ids_seen_during_training"] = 1
            with self.assertRaisesRegex(ValueError, "isolation failed"):
                audit_training_results(
                    self.write_results(directory, payload),
                    expected_seed=29,
                    expected_steps=1000,
                    expected_world_size=6,
                    stage="retrieval",
                )

    def test_rejects_non_finite_nested_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = valid_results()
            payload["validation"] = {"loss": float("nan")}
            with self.assertRaisesRegex(ValueError, "Non-finite value"):
                audit_training_results(
                    self.write_results(directory, payload),
                    expected_seed=29,
                    expected_steps=1000,
                    expected_world_size=6,
                    stage="retrieval",
                )


if __name__ == "__main__":
    unittest.main()
