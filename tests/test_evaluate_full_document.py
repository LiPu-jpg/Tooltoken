import unittest

from latent_register.evaluate_full_document import select_argument_rows


class EvaluateFullDocumentTests(unittest.TestCase):
    def test_selects_only_requested_argument_rows(self) -> None:
        rows = [
            {"family": "retrieval", "condition": "seen", "example_id": "r"},
            {"family": "arguments", "condition": "seen", "example_id": "b"},
            {"family": "arguments", "condition": "seen", "example_id": "a"},
            {"family": "arguments", "condition": "unseen", "example_id": "c"},
        ]
        selected = select_argument_rows(rows, conditions={"seen"})
        self.assertEqual([row["example_id"] for row in selected], ["a", "b"])

    def test_applies_limit_per_condition(self) -> None:
        rows = [
            {"family": "arguments", "condition": "seen", "example_id": "a"},
            {"family": "arguments", "condition": "seen", "example_id": "b"},
            {"family": "arguments", "condition": "unseen", "example_id": "c"},
            {"family": "arguments", "condition": "unseen", "example_id": "d"},
        ]
        selected = select_argument_rows(
            rows, conditions={"seen", "unseen"}, max_examples_per_condition=1
        )
        self.assertEqual([row["example_id"] for row in selected], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
