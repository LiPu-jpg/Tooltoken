from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from latent_register.prepare_controlled_benchmark import (
    allocate_fixed_tokens,
    prepare_controlled_benchmark,
)
from latent_register.episodic_data import PreparedTool


def _tool(identity: str, split: str, token: str | None = None) -> dict:
    return {
        "identity_hash": identity,
        "group_hash": f"group-{identity}",
        "split": split,
        "document": f"Tool definition {identity}",
        "tool_name": "Meteo" if identity in {"a", "b"} else f"Tool {identity}",
        "endpoint_name": "prevision" if identity in {"a", "b"} else f"call {identity}",
        "source": "toolgen" if token else "toolace",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "token": token,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class PrepareControlledBenchmarkTests(unittest.TestCase):
    def test_allocator_resolves_unidecode_collisions(self) -> None:
        tools = {
            "a": PreparedTool(
                "a", "ga", "train", "doc", "Meteo", "prevision", "toolace"
            ),
            "b": PreparedTool(
                "b", "gb", "test", "doc", "Météo", "prévision", "toolace"
            ),
        }
        mapping, audit = allocate_fixed_tokens(tools)
        self.assertNotEqual(mapping["a"], mapping["b"])
        self.assertEqual(audit["base_collision_groups"], 1)
        self.assertEqual(audit["normalized_collisions"], 0)

    def test_materializer_holds_out_seen_queries_and_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            output = Path(directory) / "controlled"
            prepared.mkdir()
            tools = [
                _tool("a", "train", "<<Meteo&&prevision>>"),
                _tool("b", "train"),
                _tool("c", "test"),
            ]
            _write_jsonl(prepared / "tools.jsonl", tools)
            _write_jsonl(
                prepared / "retrieval.jsonl",
                [
                    {
                        "query_hash": "q-a-1",
                        "query": "weather one",
                        "split": "train",
                        "tool_identity_hashes": ["a"],
                    },
                    {
                        "query_hash": "q-a-2",
                        "query": "weather two",
                        "split": "train",
                        "tool_identity_hashes": ["a"],
                    },
                    {
                        "query_hash": "q-b-1",
                        "query": "other one",
                        "split": "train",
                        "tool_identity_hashes": ["b"],
                    },
                    {
                        "query_hash": "q-b-2",
                        "query": "other two",
                        "split": "train",
                        "tool_identity_hashes": ["b"],
                    },
                    {
                        "query_hash": "q-c",
                        "query": "unseen weather",
                        "split": "test",
                        "tool_identity_hashes": ["c"],
                    },
                ],
            )
            _write_jsonl(
                prepared / "readback.jsonl",
                [
                    {
                        "source_id": "a-1",
                        "query_hash": "ra1",
                        "query": "args one",
                        "split": "train",
                        "tool_identity_hash": "a",
                        "arguments": {"city": "Paris"},
                        "call_index": 0,
                        "call_count": 1,
                    },
                    {
                        "source_id": "a-2",
                        "query_hash": "ra2",
                        "query": "args two",
                        "split": "train",
                        "tool_identity_hash": "a",
                        "arguments": {"city": "Rome"},
                        "call_index": 0,
                        "call_count": 1,
                    },
                    {
                        "source_id": "c-1",
                        "query_hash": "rc1",
                        "query": "unseen args",
                        "split": "test",
                        "tool_identity_hash": "c",
                        "arguments": {"city": "Suzhou"},
                        "call_index": 0,
                        "call_count": 1,
                    },
                ],
            )
            _write_jsonl(
                prepared / "trajectories.jsonl",
                [
                    {
                        "split": "train",
                        "target_tokens": ["<<Meteo&&prevision>>"],
                        "conversations": [
                            {"from": "user", "value": "weather"},
                            {"from": "assistant", "value": "<<Meteo&&prevision>>"},
                        ],
                    }
                ],
            )
            (prepared / "split_manifest.json").write_text(
                json.dumps({"audits": {"tool_group_overlap_zero": True}}),
                encoding="utf-8",
            )

            manifest = prepare_controlled_benchmark(
                prepared_dir=prepared,
                output_dir=output,
                seen_retrieval_count=1,
                seen_readback_count=1,
                seed=17,
            )

            self.assertEqual(manifest["counts"]["retrieval_seen_eval"], 1)
            self.assertEqual(manifest["counts"]["retrieval_unseen_test"], 1)
            self.assertEqual(manifest["audits"]["seen_retrieval_queries_in_training"], 0)
            self.assertEqual(
                manifest["audits"][
                    "seen_retrieval_targets_without_remaining_training_query"
                ],
                0,
            )
            self.assertEqual(manifest["audits"]["seen_readback_examples_in_training"], 0)
            train_tokens = (output / "virtual_tokens_train.txt").read_text()
            test_tokens = (output / "virtual_tokens_test.txt").read_text()
            self.assertIn("<<Finish>>", train_tokens)
            self.assertNotEqual(train_tokens, test_tokens)
            eval_rows = [
                json.loads(line)
                for line in (output / "benchmark_eval.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(eval_rows), 4)
            self.assertEqual(
                {row["condition"] for row in eval_rows},
                {"seen_tool_seen_token", "unseen_tool_unseen_token"},
            )
            retrieval_train = json.loads(
                (output / "fixed_retrieval_train.json").read_text()
            )
            held_query = next(
                row["query"]
                for row in eval_rows
                if row["family"] == "retrieval"
                and row["condition"] == "seen_tool_seen_token"
            )
            self.assertNotIn(
                held_query,
                [row["conversations"][0]["content"] for row in retrieval_train],
            )
            full_document_train = json.loads(
                (output / "full_document_readback_train.json").read_text()
            )
            self.assertEqual(len(full_document_train), 1)
            self.assertTrue(
                full_document_train[0]["conversations"][1]["content"].startswith(
                    "Request: "
                )
            )
            self.assertIn(
                "Tool definition:\n",
                full_document_train[0]["conversations"][1]["content"],
            )


if __name__ == "__main__":
    unittest.main()
