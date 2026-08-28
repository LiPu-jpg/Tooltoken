import json
import tempfile
import unittest
from pathlib import Path

from latent_register.prepare_incremental_fixed import prepare_incremental_fixed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class PrepareIncrementalFixedTests(unittest.TestCase):
    def test_prepares_document_only_test_registration_without_eval_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            controlled = root / "controlled"
            output = root / "output"
            prepared.mkdir()
            controlled.mkdir()
            _write_jsonl(
                prepared / "tools.jsonl",
                [
                    {
                        "identity_hash": "train-a",
                        "group_hash": "group-a",
                        "split": "train",
                        "document": "train document",
                        "tool_name": "train",
                        "endpoint_name": "call",
                        "source": "test",
                        "parameters": {},
                    },
                    {
                        "identity_hash": "test-b",
                        "group_hash": "group-b",
                        "split": "test",
                        "document": "test document",
                        "tool_name": "test",
                        "endpoint_name": "call",
                        "source": "test",
                        "parameters": {},
                    },
                ],
            )
            _write_jsonl(
                controlled / "fixed_tokens.jsonl",
                [
                    {
                        "identity_hash": "train-a",
                        "split": "train",
                        "fixed_token": "<<train&&call>>",
                    },
                    {
                        "identity_hash": "test-b",
                        "split": "test",
                        "fixed_token": "<<test&&call>>",
                    },
                ],
            )
            (controlled / "virtual_tokens_train.txt").write_text(
                "<<train&&call>>\n<<Finish>>\n", encoding="utf-8"
            )
            (controlled / "virtual_tokens_test.txt").write_text(
                "<<test&&call>>\n", encoding="utf-8"
            )
            (controlled / "benchmark_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )

            result = prepare_incremental_fixed(
                prepared_dir=prepared,
                controlled_dir=controlled,
                output_dir=output,
            )

            self.assertEqual(result["counts"]["incremental_tools"], 1)
            self.assertEqual(result["audits"]["evaluation_query_records_used"], 0)
            rows = json.loads(
                (output / "fixed_incremental_memorization.json").read_text()
            )
            self.assertEqual(rows[0]["conversations"][0]["content"], "test document")
            self.assertEqual(
                rows[0]["conversations"][1]["content"], "<<test&&call>>"
            )
            self.assertEqual(
                (output / "virtual_tokens_combined.txt").read_text(),
                "<<train&&call>>\n<<Finish>>\n<<test&&call>>\n",
            )


if __name__ == "__main__":
    unittest.main()
