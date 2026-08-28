import unittest

import torch

from latent_register.data import ToolExample, stable_tool_split
from latent_register.model import (
    DualViewGeneratedMemory,
    GatedLayerwiseCrossAttentionMemory,
    GeneratedMemory,
    ReadoutCrossAttentionMemory,
    SchemaKeyGeneratedMemory,
    TokenResamplerMemory,
    multi_positive_contrastive_loss,
)
from latent_register.registry import DynamicRegistry


class RegistryTests(unittest.TestCase):
    def test_remapping_unseen_slots_preserves_semantics(self) -> None:
        tools = ["weather", "maps", "calendar"]
        vectors = torch.eye(3)
        memories = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        original = DynamicRegistry.from_vectors(
            tools, vectors, [1, 2, 3], memories
        )
        remapped = original.remap([8001, 9002, 10003])

        query = torch.tensor([0.1, 0.8, 0.1])
        self.assertEqual(original.select(query).tool_name, "maps")
        self.assertEqual(remapped.select(query).tool_name, "maps")
        self.assertEqual(original.score_by_tool(query), remapped.score_by_tool(query))
        self.assertTrue(torch.equal(remapped.expand(9002), memories[1]))

    def test_registry_rejects_unregistered_expansion(self) -> None:
        registry = DynamicRegistry.from_vectors(["weather"], torch.eye(1), [8001])
        with self.assertRaisesRegex(KeyError, "Unregistered slot"):
            registry.expand(9002)
        with self.assertRaisesRegex(ValueError, "no registered input memory"):
            registry.expand(8001)

    def test_tool_split_has_no_identity_leakage(self) -> None:
        examples = [
            ToolExample(str(index), f"query {index}", f"tool_{index % 5}", "doc")
            for index in range(20)
        ]
        train, evaluation = stable_tool_split(examples, eval_ratio=0.4, seed=7)
        self.assertFalse(
            {item.tool_name for item in train} & {item.tool_name for item in evaluation}
        )

    def test_multi_positive_loss_is_finite_with_duplicate_tools(self) -> None:
        logits = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        loss = multi_positive_contrastive_loss(
            logits,
            ["weather", "weather", "maps"],
            ["weather", "maps"],
        )
        self.assertTrue(torch.isfinite(loss))

    def test_generated_memory_has_requested_slots_and_norm(self) -> None:
        generator = GeneratedMemory(hidden_size=8, rank=4, num_slots=3, initial_norm=2.5)
        memory = generator(torch.randn(2, 8))
        self.assertEqual(memory.shape, (2, 3, 8))
        self.assertTrue(torch.allclose(memory.norm(dim=-1), torch.full((2, 3), 2.5)))

    def test_dual_view_memory_splits_slot_bases(self) -> None:
        generator = DualViewGeneratedMemory(
            hidden_size=8, rank=4, num_slots=4, initial_norm=2.5
        )
        views = torch.randn(2, 2, 8)
        memory = generator(views)

        self.assertEqual(memory.shape, (2, 4, 8))
        self.assertTrue(torch.allclose(memory.norm(dim=-1), torch.full((2, 4), 2.5)))
        with self.assertRaisesRegex(ValueError, r"\[batch, 2, hidden\]"):
            generator(torch.randn(2, 8))

    def test_schema_key_memory_preserves_one_anchor_per_slot(self) -> None:
        generator = SchemaKeyGeneratedMemory(
            hidden_size=8, rank=4, num_slots=3, initial_norm=2.5
        )
        views = torch.randn(2, 4, 8)
        memory = generator(views)

        self.assertEqual(memory.shape, (2, 3, 8))
        self.assertTrue(torch.allclose(memory.norm(dim=-1), torch.full((2, 3), 2.5)))
        with self.assertRaisesRegex(ValueError, r"\[batch, 4, hidden\]"):
            generator(torch.randn(2, 3, 8))

    def test_layerwise_memory_accepts_schema_key_views(self) -> None:
        module = GatedLayerwiseCrossAttentionMemory(
            hidden_size=8,
            rank=4,
            num_slots=3,
            initial_norm=1.5,
            layer_indices=(0,),
            num_views=4,
        )
        memory = module(torch.randn(2, 4, 8))
        self.assertEqual(memory.shape, (2, 3, 8))

    def test_readout_cross_attention_conditions_each_decoder_position(self) -> None:
        module = ReadoutCrossAttentionMemory(12, 4, 3, 2.5)
        latent = torch.randn(2, 12)
        hidden = torch.randn(2, 5, 12)
        initial = module(latent, hidden)
        self.assertEqual(initial.shape, hidden.shape)
        self.assertTrue(torch.allclose(initial, hidden.float()))

        with torch.no_grad():
            module.output.weight.normal_(std=0.02)
        adapted = module(latent, hidden)
        self.assertFalse(torch.allclose(adapted, hidden.float()))
        adapted.sum().backward()
        self.assertIsNotNone(module.memory.up.weight.grad)
        self.assertIsNotNone(module.query.weight.grad)

    def test_readout_cross_attention_rejects_mismatched_batches(self) -> None:
        module = ReadoutCrossAttentionMemory(8, 4, 2, 1.0)
        with self.assertRaisesRegex(ValueError, "batches differ"):
            module.adapt(torch.randn(2, 3, 8), torch.randn(1, 2, 8))

    def test_gated_layerwise_memory_is_identity_until_gate_opens(self) -> None:
        decoder_layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        module = GatedLayerwiseCrossAttentionMemory(
            hidden_size=8,
            rank=4,
            num_slots=2,
            initial_norm=1.5,
            layer_indices=(1, 3),
            max_gate=0.2,
        )
        module.install(decoder_layers)
        hidden = torch.randn(2, 3, 8)
        memory = module(torch.randn(2, 8))

        with module.activate(memory):
            initial = hidden
            for layer in decoder_layers:
                initial = layer(initial)
        self.assertTrue(torch.equal(initial, hidden))

        with torch.no_grad():
            for block in module.blocks.values():
                block.gate_logit.fill_(0.5)
        with module.activate(memory):
            adapted = hidden
            for layer in decoder_layers:
                adapted = layer(adapted)
        self.assertFalse(torch.allclose(adapted, hidden))
        adapted.sum().backward()
        self.assertIsNotNone(module.blocks["1"].gate_logit.grad)
        self.assertIsNotNone(module.memory.up.weight.grad)
        module.remove()

    def test_gated_layerwise_memory_can_be_suspended_for_selection(self) -> None:
        decoder_layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
        module = GatedLayerwiseCrossAttentionMemory(8, 4, 2, 1.0, (0,), 0.2)
        module.install(decoder_layers)
        with torch.no_grad():
            module.blocks["0"].gate_logit.fill_(1.0)
        hidden = torch.randn(1, 2, 8)
        memory = module(torch.randn(1, 8))
        with module.activate(memory), module.suspended():
            output = decoder_layers[0](hidden)
        self.assertTrue(torch.equal(output, hidden))
        module.remove()

    def test_token_resampler_ignores_masked_states(self) -> None:
        generator = TokenResamplerMemory(
            hidden_size=8, rank=4, num_slots=3, initial_norm=2.5
        )
        states = torch.randn(2, 5, 8)
        values = torch.randn(2, 5, 8)
        mask = torch.tensor([[True, True, False, False, False], [True] * 5])
        first = generator(states, mask, values)
        states[0, 2:] = 1000.0
        values[0, 2:] = -1000.0
        second = generator(states, mask, values)
        self.assertEqual(first.shape, (2, 3, 8))
        self.assertTrue(torch.allclose(first[0], second[0]))

    def test_token_resampler_value_head_receives_gradients(self) -> None:
        generator = TokenResamplerMemory(
            hidden_size=8, rank=4, num_slots=3, initial_norm=2.5
        )
        states = torch.randn(2, 5, 8)
        values = torch.randn(2, 5, 8)
        memory = generator(states, torch.ones(2, 5, dtype=torch.bool), values)

        memory[..., 0].sum().backward()

        self.assertIsNotNone(generator.value_up.weight.grad)
        self.assertGreater(float(generator.value_up.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
