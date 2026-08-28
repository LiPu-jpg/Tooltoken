import json
import tempfile
import unittest
from pathlib import Path

from latent_register.episodic_data import (
    EpisodicRegistrySampler,
    PreparedRetrievalEpisode,
    PreparedTool,
    load_readback_examples,
    load_retrieval_episodes,
    load_token_pools,
)
from latent_register.audit_episodic_data import audit_prepared_episodes


def _tool(identity: str, split: str) -> PreparedTool:
    return PreparedTool(identity, identity, split, f"document {identity}", identity, "call", "test")


class EpisodicDataTests(unittest.TestCase):
    def test_binding_is_deterministic_per_epoch_and_rebinds_across_epochs(self) -> None:
        tools = {f"t{index}": _tool(f"t{index}", "train") for index in range(10)}
        sampler = EpisodicRegistrySampler(
            tools,
            {"train": range(0, 20), "validation": range(20, 30), "test": range(30, 40)},
            seed=17,
        )
        episode = PreparedRetrievalEpisode("q", "query", "train", ("t0", "t1"))
        first = sampler.bind(episode, registry_size=6, epoch=0)
        repeated = sampler.bind(episode, registry_size=6, epoch=0)
        rebound = sampler.bind(episode, registry_size=6, epoch=1)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.slot_indices, rebound.slot_indices)
        self.assertEqual(len(first.positive_positions), 2)
        self.assertTrue(set(first.positive_slot_indices).issubset(set(range(0, 20))))

    def test_binding_never_crosses_validation_token_pool(self) -> None:
        tools = {f"v{index}": _tool(f"v{index}", "validation") for index in range(5)}
        sampler = EpisodicRegistrySampler(
            tools,
            {"train": range(0, 10), "validation": range(10, 20), "test": range(20, 30)},
            seed=17,
        )
        episode = PreparedRetrievalEpisode("q", "query", "validation", ("v0",))
        bound = sampler.bind(episode, registry_size=4, epoch=0)
        self.assertTrue(all(10 <= slot < 20 for slot in bound.slot_indices))

    def test_loader_rejects_retrieval_target_from_another_split(self) -> None:
        tools = {"train": _tool("train", "train"), "test": _tool("test", "test")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retrieval.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_hash": "q",
                        "query": "query",
                        "split": "train",
                        "tool_identity_hashes": ["test"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_retrieval_episodes(path, tools)

    def test_manifest_loader_rejects_overlapping_pools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "token_pools": {
                            "train": {"start_inclusive": 0, "end_exclusive": 10},
                            "validation": {"start_inclusive": 9, "end_exclusive": 15},
                            "test": {"start_inclusive": 15, "end_exclusive": 20},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_token_pools(path)

    def test_readback_loader_rejects_cross_split_tool(self) -> None:
        tools = {"train": _tool("train", "train"), "test": _tool("test", "test")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readback.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "source_id": "row:1",
                        "query_hash": "q",
                        "query": "query",
                        "split": "train",
                        "tool_identity_hash": "test",
                        "arguments": {"city": "Suzhou"},
                        "call_index": 0,
                        "call_count": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_readback_examples(path, tools)

    def test_readback_loader_can_filter_multi_call_examples(self) -> None:
        tools = {"train": _tool("train", "train")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "readback.jsonl"
            rows = [
                {
                    "source_id": f"row:{count}",
                    "query_hash": f"q{count}",
                    "query": "query",
                    "split": "train",
                    "tool_identity_hash": "train",
                    "arguments": {"count": count},
                    "call_index": 0,
                    "call_count": count,
                }
                for count in (1, 2)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            examples = load_readback_examples(
                path, tools, single_call_only=True
            )
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].arguments, {"count": 1})

    def test_audit_samples_multiple_registry_sizes_without_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = []
            retrieval = []
            pools = {"train": (0, 10), "validation": (10, 20), "test": (20, 30)}
            for split, (start, _) in pools.items():
                for index in range(4):
                    identity = f"{split}-{index}"
                    tools.append(
                        {
                            "identity_hash": identity,
                            "group_hash": identity,
                            "split": split,
                            "document": f"document {identity}",
                            "tool_name": identity,
                            "endpoint_name": "call",
                            "source": "test",
                        }
                    )
                retrieval.append(
                    {
                        "query_hash": f"q-{split}",
                        "query": "query",
                        "split": split,
                        "tool_identity_hashes": [f"{split}-0"],
                    }
                )
            (root / "tools.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in tools), encoding="utf-8"
            )
            (root / "retrieval.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in retrieval), encoding="utf-8"
            )
            (root / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "token_pools": {
                            split: {
                                "start_inclusive": bounds[0],
                                "end_exclusive": bounds[1],
                            }
                            for split, bounds in pools.items()
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = audit_prepared_episodes(
                root, registry_sizes=(2, 4), sample_per_split=1
            )
            self.assertTrue(
                result["audits"]["cross_split_tool_or_token_violations_zero"]
            )
            self.assertEqual(result["splits"]["test"]["samples"]["4"]["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
