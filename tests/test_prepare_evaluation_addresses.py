import json
import tempfile
import unittest
from pathlib import Path

from latent_register.prepare_evaluation_addresses import (
    prepare_evaluation_address_manifest,
)


class PrepareEvaluationAddressesTests(unittest.TestCase):
    def test_preserves_training_pool_and_expands_disjoint_evaluation_pools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "split_manifest.json"
            source.write_text(
                json.dumps(
                    {
                        "token_pools": {
                            "train": {"start_inclusive": 0, "end_exclusive": 8},
                            "validation": {"start_inclusive": 8, "end_exclusive": 12},
                            "test": {"start_inclusive": 12, "end_exclusive": 16},
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "expanded.json"
            result = prepare_evaluation_address_manifest(
                source, output, evaluation_addresses_per_split=47
            )

            self.assertEqual(
                result["token_pools"]["train"],
                {"start_inclusive": 0, "end_exclusive": 8, "count": 8},
            )
            self.assertEqual(
                result["token_pools"]["validation"]["start_inclusive"], 8
            )
            self.assertEqual(
                result["token_pools"]["test"]["start_inclusive"], 55
            )
            self.assertEqual(
                result["maximum_registry_size_without_address_reuse"], 47
            )
            self.assertTrue(all(result["audits"].values()))
            self.assertEqual(json.loads(output.read_text()), result)

    def test_rejects_zero_addresses(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare_evaluation_address_manifest(
                "unused.json", "unused-output.json", evaluation_addresses_per_split=0
            )


if __name__ == "__main__":
    unittest.main()
