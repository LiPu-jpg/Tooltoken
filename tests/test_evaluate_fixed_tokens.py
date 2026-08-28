import json
import tempfile
import unittest
from pathlib import Path

import torch

from latent_register.evaluate_fixed_tokens import (
    argument_prompt,
    atomic_candidate_ids,
    load_fixed_token_mapping,
    load_common_document_agent_audit,
    rank_candidate_logits,
    retrieval_prompt,
)


class _Encoding:
    def __init__(self, ids):
        self.input_ids = ids


class _FakeTokenizer:
    def __init__(self, values):
        self.values = values

    def __call__(self, value, add_special_tokens=False):
        return _Encoding(self.values[value])


class EvaluateFixedTokensTests(unittest.TestCase):
    def test_common_document_agent_audit_binds_one_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "Qwen3-8B-Full-Document"
            model.mkdir()
            audit = root / "checkpoint_audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "kind": "qwen_full_document_checkpoint_audit",
                        "model_path": str(model.resolve()),
                        "full_model_reloaded": True,
                        "embedding_tables_all_finite": True,
                    }
                ),
                encoding="utf-8",
            )
            (root / "COMPLETE").touch()
            result = load_common_document_agent_audit(model, audit)

        self.assertEqual(result["model_path"], str(model.resolve()))
        self.assertEqual(len(result["audit_sha256"]), 64)

    def test_chatml_matches_toolgen_qwen_template(self) -> None:
        self.assertEqual(
            retrieval_prompt("weather"),
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nweather<|im_end|>\n"
            "<|im_start|>assistant\n",
        )

    def test_native_argument_prompt_fetches_selected_document(self) -> None:
        prompt = argument_prompt(
            "weather",
            selected_token="<<Weather&&forecast>>",
            selected_document="city schema",
            information_condition="native",
        )
        self.assertIn("<<Weather&&forecast>><|im_end|>", prompt)
        self.assertIn("Tool definition:\ncity schema", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_token_only_prompt_continues_after_physical_token(self) -> None:
        prompt = argument_prompt(
            "weather",
            selected_token="<<Weather&&forecast>>",
            selected_document=None,
            information_condition="token_memory_only",
        )
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<<Weather&&forecast>>"))

    def test_common_document_does_not_consume_the_selected_token_embedding(self) -> None:
        prompt = argument_prompt(
            "weather",
            selected_token="<<Weather&&forecast>>",
            selected_document="city schema",
            information_condition="common_document",
        )
        self.assertIn("Request: weather\nTool definition:\ncity schema", prompt)
        self.assertNotIn("<<Weather&&forecast>>", prompt)

    def test_atomic_candidates_reject_multi_token_mapping(self) -> None:
        mapping = {
            "a": {"split": "train", "fixed_token": "A", "original_token": None},
            "b": {"split": "test", "fixed_token": "B", "original_token": None},
        }
        ids, identities, strings = atomic_candidate_ids(
            _FakeTokenizer({"A": [7], "B": [8, 9]}),
            mapping,
            splits={"train"},
        )
        self.assertEqual((ids, identities, strings), ([7], ["a"], ["A"]))
        with self.assertRaises(ValueError):
            atomic_candidate_ids(
                _FakeTokenizer({"A": [7], "B": [8, 9]}),
                mapping,
                splits={"test"},
            )

    def test_candidate_ranking_uses_only_fixed_token_columns(self) -> None:
        logits = torch.tensor([[100.0, 2.0, 5.0, 3.0]])
        ranked = rank_candidate_logits(
            logits,
            [1, 2, 3],
            ["a", "b", "c"],
            ["A", "B", "C"],
            k=2,
        )
        self.assertEqual(ranked[0]["identities"], ["b", "c"])
        self.assertEqual(ranked[0]["tokens"], ["B", "C"])

    def test_mapping_loader_rejects_duplicate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.jsonl"
            row = {
                "identity_hash": "a",
                "split": "train",
                "fixed_token": "A",
                "original_token": None,
            }
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_fixed_token_mapping(path)


if __name__ == "__main__":
    unittest.main()
