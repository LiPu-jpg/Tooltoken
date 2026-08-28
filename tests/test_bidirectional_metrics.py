import unittest

from latent_register.train_bidirectional import (
    distributed_input_batches,
    generation_metrics,
    input_training_phase,
    key_paths,
    parse_json_object,
    pipeline_metrics,
    resolve_layerwise_memory_layers,
    select_registered_index,
    uses_layerwise_for_ordinary_readback,
    uses_prompt_slots,
)


class BidirectionalMetricTests(unittest.TestCase):
    def test_resolves_evenly_spaced_non_final_memory_layers(self) -> None:
        self.assertEqual(
            resolve_layerwise_memory_layers("auto:4", 36), (6, 13, 21, 28)
        )
        self.assertEqual(resolve_layerwise_memory_layers("2,7", 12), (2, 7))

    def test_trajectory_only_layerwise_interface_keeps_ordinary_prompt_path(self) -> None:
        interface = "prompt_slots_plus_trajectory_layerwise"

        self.assertTrue(uses_prompt_slots(interface))
        self.assertFalse(uses_layerwise_for_ordinary_readback(interface))
        self.assertTrue(
            uses_layerwise_for_ordinary_readback("prompt_slots_plus_layerwise")
        )

    def test_json_and_nested_schema_metrics(self) -> None:
        target = {"where": {"city": "Suzhou"}, "days": 2}
        parsed = parse_json_object(f"answer: {target!r}")
        self.assertIsNone(parsed)
        parsed = parse_json_object('answer: {"where":{"city":"Suzhou"},"days":2}')
        self.assertEqual(parsed, target)
        self.assertEqual(key_paths(parsed), {"where", "where.city", "days"})
        metrics = generation_metrics([{"parsed": parsed, "target": target}])
        self.assertEqual(metrics["exact_arguments"], 1.0)
        self.assertEqual(metrics["exact_schema_keys"], 1.0)
        self.assertEqual(metrics["key_precision"], 1.0)
        self.assertEqual(metrics["key_recall"], 1.0)
        self.assertEqual(metrics["shared_key_value_accuracy"], 1.0)

    def test_generation_metrics_separate_keys_from_values(self) -> None:
        metrics = generation_metrics(
            [
                {
                    "parsed": {"city": "Suzhou", "unit": "C"},
                    "target": {"city": "Suzhou", "days": 2},
                }
            ]
        )
        self.assertEqual(metrics["exact_arguments"], 0.0)
        self.assertEqual(metrics["key_precision"], 0.5)
        self.assertEqual(metrics["key_recall"], 0.5)
        self.assertEqual(metrics["shared_key_value_accuracy"], 1.0)

    def test_pipeline_requires_correct_selection_and_arguments(self) -> None:
        rows = [
            {
                "tool_name": "weather",
                "selected_tool": "weather",
                "parsed": {"city": "Suzhou"},
                "target": {"city": "Suzhou"},
            },
            {
                "tool_name": "maps",
                "selected_tool": "weather",
                "parsed": {"city": "Suzhou"},
                "target": {"city": "Suzhou"},
            },
        ]
        metrics = pipeline_metrics(rows)
        self.assertEqual(metrics["selection_accuracy"], 0.5)
        self.assertEqual(metrics["json_valid"], 1.0)
        self.assertEqual(
            metrics["readback_exact_arguments_given_correct_selection"], 1.0
        )
        self.assertEqual(metrics["end_to_end_exact_arguments"], 0.5)
        self.assertEqual(metrics["end_to_end_exact_schema_keys"], 0.5)

    def test_pipeline_distinguishes_same_name_tool_definitions(self) -> None:
        metrics = pipeline_metrics(
            [
                {
                    "tool_name": "weather",
                    "tool_id": "weather::schema-a",
                    "selected_tool": "weather",
                    "selected_tool_id": "weather::schema-b",
                    "parsed": {"city": "Suzhou"},
                    "target": {"city": "Suzhou"},
                }
            ]
        )

        self.assertEqual(metrics["selection_accuracy"], 0.0)
        self.assertEqual(metrics["end_to_end_exact_arguments"], 0.0)

    def test_schema_warmup_precedes_joint_training(self) -> None:
        self.assertEqual(input_training_phase(0, 3), "schema_warmup")
        self.assertEqual(input_training_phase(2, 3), "schema_warmup")
        self.assertEqual(input_training_phase(3, 3), "joint")

    def test_registered_selection_uses_dynamic_output_rows(self) -> None:
        import torch

        query = torch.tensor([0.0, 2.0, 0.0])
        rows = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(select_registered_index(query, rows), 1)

    def test_distributed_batches_cover_data_with_equal_rank_steps(self) -> None:
        order = list(range(10))
        rank_batches = [
            distributed_input_batches(order, 1, 4, rank) for rank in range(4)
        ]

        self.assertTrue(all(padding == 2 for _, padding in rank_batches))
        self.assertTrue(all(len(batches) == 3 for batches, _ in rank_batches))
        flattened = [
            index
            for step in range(3)
            for batches, _ in rank_batches
            for index in batches[step]
        ]
        self.assertEqual(flattened, order + order[:2])


if __name__ == "__main__":
    unittest.main()
