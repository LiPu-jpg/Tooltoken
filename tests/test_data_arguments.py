import json
import tempfile
import unittest
from pathlib import Path

from latent_register.data import (
    ToolExample,
    canonical_target_arguments,
    load_function_call_jsonl,
    schema_target_arguments,
    tool_registry_key,
)


class DataArgumentsTests(unittest.TestCase):
    def test_canonicalizes_bfcl_possible_values_and_omits_optional_defaults(self) -> None:
        self.assertEqual(
            canonical_target_arguments(
                {
                    "base": [10],
                    "unit": ["units", ""],
                    "items": [[1, 2, 3]],
                }
            ),
            {"base": 10, "items": [1, 2, 3]},
        )

    def test_schema_target_contains_all_declared_properties(self) -> None:
        self.assertEqual(
            schema_target_arguments(
                {
                    "parameters": {
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                        }
                    }
                }
            ),
            {"city": None, "days": None},
        )

    def test_loader_preserves_ground_truth_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            answers = root / "answers.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "id": "x",
                        "question": "user: find rain",
                        "function": [
                            {
                                "name": "weather",
                                "description": "forecast",
                                "parameters": {"type": "object"},
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            answers.write_text(
                json.dumps({"id": "x", "ground_truth": {"weather": {"city": "Suzhou"}}})
                + "\n",
                encoding="utf-8",
            )
            loaded = load_function_call_jsonl(data, answers)
            self.assertEqual(loaded[0].target_arguments, {"city": "Suzhou"})
            self.assertEqual(loaded[0].schema_arguments, {})

    def test_document_identity_separates_same_name_overloads(self) -> None:
        first = ToolExample("1", "q1", "weather", "schema one")
        second = ToolExample("2", "q2", "weather", "schema two")

        self.assertEqual(tool_registry_key(first, "name"), "weather")
        self.assertNotEqual(
            tool_registry_key(first, "document"),
            tool_registry_key(second, "document"),
        )
        self.assertEqual(
            tool_registry_key(first, "document"),
            tool_registry_key(first, "document"),
        )


if __name__ == "__main__":
    unittest.main()
