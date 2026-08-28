import unittest

import torch

from latent_register.data import ToolExample
from latent_register.evaluate_memory_oracle import (
    adaptation_example,
    assigned_tool_keys,
    normalize_memory_,
)


class MemoryOracleTests(unittest.TestCase):
    def test_distributed_assignment_partitions_tools(self) -> None:
        keys = [f"tool-{index}" for index in range(7)]
        partitions = [assigned_tool_keys(keys, 4, rank) for rank in range(4)]

        self.assertEqual(sorted(key for part in partitions for key in part), keys)
        self.assertEqual(sum(map(len, partitions)), len(keys))

    def test_normalization_preserves_requested_slot_norm(self) -> None:
        memory = torch.randn(3, 8)
        normalize_memory_(memory, 2.5)

        self.assertTrue(torch.allclose(memory.norm(dim=-1), torch.full((3,), 2.5)))

    def test_adaptation_prompt_does_not_reuse_evaluation_query_or_arguments(self) -> None:
        example = ToolExample(
            "x",
            "private evaluation query",
            "weather",
            "weather document",
            {"city": "Suzhou"},
            {"city": None},
        )

        adapted = adaptation_example(example)

        self.assertEqual(adapted.query, "Reconstruct the selected tool schema.")
        self.assertEqual(adapted.target_arguments, {})
        self.assertEqual(adapted.schema_arguments, {"city": None})


if __name__ == "__main__":
    unittest.main()
