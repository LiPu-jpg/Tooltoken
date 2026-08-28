import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from latent_register.episodic_data import (
    BoundRegistryEpisode,
    PreparedReadbackExample,
    PreparedRetrievalEpisode,
    PreparedTool,
)
from latent_register.model import PhysicalOutputGenerator, TokenResamplerMemory
from latent_register.train_meta_readback import (
    MetaReadbackModel,
    _epoch_batches,
    canonical_arguments,
    canonical_schema_keys,
    active_registry_full_logits,
    evaluate_full_vocabulary_selection,
    evaluate_same_stream_generation,
    full_vocabulary_selection_loss,
    generate_same_stream_one,
    SameStreamGeneration,
    logical_slot_for_example,
    schema_key_object,
    selection_registry_size_for_batch,
    wrong_memory_indices,
    wrong_tool_lookup,
)


class _CountingDecoder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)
        self.calls = 0

    def forward(self, input_ids, attention_mask, **_kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _Backbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.model = _CountingDecoder(hidden_size)


class _SelectionBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output = nn.Linear(2, 5, bias=False)
        with torch.no_grad():
            self.output.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [8.0, 0.0],
                        [0.0, 0.0],
                        [100.0, 0.0],
                        [100.0, 0.0],
                    ]
                )
            )

    def get_output_embeddings(self) -> nn.Module:
        return self.output


class _SelectionModel:
    def __init__(self) -> None:
        self.backbone = _SelectionBackbone()

    def eval(self) -> None:
        pass

    def selection_logits(self, *_args) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([[1.0, 0.0]]), torch.tensor([[6.0, 1.0]])


class _SameStreamTokenizer:
    chat_template = None
    eos_token_id = 9

    def __call__(self, _text, **_kwargs):
        return SimpleNamespace(input_ids=[1, 2])

    def decode(self, token_ids, **_kwargs) -> str:
        return '{"value":1}' if tuple(token_ids) == (7,) else ""


class _SameStreamBackbone(nn.Module):
    def __init__(self, ordinary_logit: float) -> None:
        super().__init__()
        self.input = nn.Embedding(10, 2)
        self.output = nn.Linear(2, 10, bias=False)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.weight[0, 0] = ordinary_logit
        self.calls: list[dict] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.input

    def get_output_embeddings(self) -> nn.Module:
        return self.output

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        past = kwargs.get("past_key_values")
        if past is None:
            hidden = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
            logits = torch.zeros(1, 2, 10)
            return SimpleNamespace(
                hidden_states=(hidden,),
                logits=logits,
                past_key_values="prompt-cache",
            )
        if "inputs_embeds" in kwargs:
            slots = kwargs["inputs_embeds"].shape[1]
            logits = torch.zeros(1, slots, 10)
            logits[:, -1, 7] = 5.0
            return SimpleNamespace(logits=logits, past_key_values="memory-cache")
        self.assert_cache(past, "memory-cache")
        logits = torch.zeros(1, 1, 10)
        logits[:, -1, 9] = 5.0
        return SimpleNamespace(logits=logits, past_key_values="token-cache")

    @staticmethod
    def assert_cache(actual, expected) -> None:
        if actual != expected:
            raise AssertionError(f"Expected {expected}, got {actual}")


class _SameStreamEvalModel:
    output_compiler = object()

    def __init__(self) -> None:
        self.bundle_calls = 0
        self.memory_calls = 0

    def eval(self) -> None:
        pass

    def register_bundle(self, _tokens):
        self.bundle_calls += 1
        return torch.ones(2, 2), torch.ones(2, 1, 2)

    def register(self, _tokens):
        self.memory_calls += 1
        return torch.ones(2, 1, 2)


def _example(index: int) -> PreparedReadbackExample:
    return PreparedReadbackExample(
        source_id=f"row:{index}",
        query_hash=f"q{index}",
        query="query",
        split="train",
        tool_identity_hash="tool",
        arguments={"z": index, "a": True},
        call_index=0,
        call_count=1,
    )


class MetaReadbackTests(unittest.TestCase):
    def test_same_stream_evaluator_counts_shared_and_dual_document_forwards(self) -> None:
        target = PreparedTool(
            identity_hash="target",
            group_hash="group-target",
            split="test",
            document="target document",
            tool_name="target",
            endpoint_name="target",
            source="test",
        )
        distractor = PreparedTool(
            identity_hash="distractor",
            group_hash="group-distractor",
            split="test",
            document="distractor document",
            tool_name="distractor",
            endpoint_name="distractor",
            source="test",
        )
        bound = BoundRegistryEpisode(
            query_hash="query",
            query="query",
            split="test",
            tools=(target, distractor),
            slot_indices=(0, 1),
            positive_positions=(0,),
        )
        sampler = SimpleNamespace(bind=lambda *_args, **_kwargs: bound)
        pool = SimpleNamespace(
            token_ids=[3, 4], physical_id=lambda slot: 3 + slot
        )
        example = PreparedReadbackExample(
            source_id="source",
            query_hash="query",
            query="query",
            split="test",
            tool_identity_hash="target",
            arguments={"value": 1},
            call_index=0,
            call_count=1,
        )
        generated = SameStreamGeneration(3, 0, (7,), '{"value":1}')

        with patch(
            "latent_register.train_meta_readback.tokenize_documents",
            return_value={"input_ids": torch.ones(2, 1, dtype=torch.long)},
        ), patch(
            "latent_register.train_meta_readback.generate_same_stream_one",
            return_value=generated,
        ):
            shared_model = _SameStreamEvalModel()
            shared = evaluate_same_stream_generation(
                shared_model,
                tokenizer=None,
                tools={"target": target, "distractor": distractor},
                examples=[example],
                sampler=sampler,
                physical_pool=pool,
                split="test",
                registry_size=2,
                sample_limit=1,
                max_document_length=8,
                max_prompt_length=8,
                max_new_tokens=8,
                document_instruction="shared",
                selection_document_instruction="shared",
                device=torch.device("cpu"),
                rank=0,
                world_size=1,
            )
            dual_model = _SameStreamEvalModel()
            dual = evaluate_same_stream_generation(
                dual_model,
                tokenizer=None,
                tools={"target": target, "distractor": distractor},
                examples=[example],
                sampler=sampler,
                physical_pool=pool,
                split="test",
                registry_size=2,
                sample_limit=1,
                max_document_length=8,
                max_prompt_length=8,
                max_new_tokens=8,
                document_instruction="memory",
                selection_document_instruction="selection",
                device=torch.device("cpu"),
                rank=0,
                world_size=1,
            )

        self.assertEqual(shared_model.bundle_calls, 1)
        self.assertEqual(shared_model.memory_calls, 0)
        self.assertEqual(shared["registration_document_forwards_per_tool"], 1)
        self.assertEqual(shared["physical_token_emission_rate"], 1.0)
        self.assertEqual(shared["end_to_end_exact_arguments"], 1.0)
        self.assertEqual(dual_model.bundle_calls, 1)
        self.assertEqual(dual_model.memory_calls, 1)
        self.assertEqual(dual["registration_document_forwards_per_tool"], 2)

    def test_same_stream_emits_physical_id_and_reuses_prompt_cache(self) -> None:
        backbone = _SameStreamBackbone(ordinary_logit=1.0)
        model = SimpleNamespace(backbone=backbone)
        result = generate_same_stream_one(
            model,
            _SameStreamTokenizer(),
            "query",
            registered_output_rows=torch.tensor([[4.0, 0.0], [2.0, 0.0]]),
            registered_memory=torch.ones(2, 3, 2),
            physical_ids=torch.tensor([5, 6]),
            reserved_token_ids=torch.tensor([5, 6]),
            max_prompt_length=16,
            max_new_tokens=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(result.selected_token_id, 5)
        self.assertEqual(result.selected_registry_index, 0)
        self.assertEqual(result.generated_token_ids, (7,))
        self.assertEqual(result.text, '{"value":1}')
        self.assertEqual(len(backbone.calls), 3)
        self.assertIsNone(backbone.calls[0].get("past_key_values"))
        self.assertEqual(backbone.calls[1]["past_key_values"], "prompt-cache")
        self.assertEqual(backbone.calls[1]["attention_mask"].shape[1], 5)
        self.assertEqual(backbone.calls[2]["past_key_values"], "memory-cache")
        self.assertEqual(backbone.calls[2]["attention_mask"].shape[1], 6)

    def test_same_stream_stops_when_an_ordinary_token_wins(self) -> None:
        backbone = _SameStreamBackbone(ordinary_logit=8.0)
        result = generate_same_stream_one(
            SimpleNamespace(backbone=backbone),
            _SameStreamTokenizer(),
            "query",
            registered_output_rows=torch.tensor([[4.0, 0.0], [2.0, 0.0]]),
            registered_memory=torch.ones(2, 3, 2),
            physical_ids=torch.tensor([5, 6]),
            reserved_token_ids=torch.tensor([5, 6]),
            max_prompt_length=16,
            max_new_tokens=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(result.selected_token_id, 0)
        self.assertIsNone(result.selected_registry_index)
        self.assertEqual(result.generated_token_ids, ())
        self.assertEqual(len(backbone.calls), 1)

    def test_full_vocabulary_evaluator_exposes_ordinary_token_competition(self) -> None:
        episode = PreparedRetrievalEpisode(
            query_hash="q1",
            query="query",
            split="test",
            target_identity_hashes=("target",),
        )
        bound = SimpleNamespace(slot_indices=(0, 1))
        sampler = SimpleNamespace(bind=lambda *_args, **_kwargs: bound)
        physical_pool = SimpleNamespace(
            token_ids=[3, 4],
            original_vocab_size=3,
            physical_id=lambda slot: 3 + slot,
        )
        tokenized = (
            {"input_ids": torch.ones(1, 1, dtype=torch.long)},
            {"input_ids": torch.ones(2, 1, dtype=torch.long)},
            torch.tensor([[True, False]]),
            2,
        )
        with patch(
            "latent_register.train_meta_readback._tokenize_bound_episodes",
            return_value=tokenized,
        ):
            metrics = evaluate_full_vocabulary_selection(
                _SelectionModel(),
                tokenizer=None,
                sampler=sampler,
                episodes=[episode],
                physical_pool=physical_pool,
                split="test",
                registry_size=2,
                sample_limit=1,
                max_query_length=8,
                max_document_length=8,
                document_instruction="Represent this tool for registration.",
                device=torch.device("cpu"),
                rank=0,
                world_size=1,
            )

        self.assertEqual(metrics["episodes"], 1)
        self.assertEqual(metrics["registry_hit_at_1"], 1.0)
        self.assertEqual(metrics["hit_at_1"], 0.0)
        self.assertEqual(metrics["ordinary_token_win_rate"], 1.0)
        self.assertEqual(metrics["mean_target_rank"], 2.0)
        self.assertLess(metrics["mean_active_registry_probability"], 0.2)

    def test_full_vocabulary_logits_keep_base_words_and_replace_active_rows(self) -> None:
        base = torch.tensor([[1.0, 5.0, 2.0, 100.0, 100.0, 100.0]])
        registry = torch.tensor([[7.0, 6.0]], requires_grad=True)
        physical = torch.tensor([[3, 5]])
        reserved = torch.tensor([3, 4, 5])
        logits = active_registry_full_logits(base, registry, physical, reserved)

        self.assertEqual(float(logits[0, 1].detach()), 5.0)
        self.assertEqual(float(logits[0, 3].detach()), 7.0)
        self.assertTrue(torch.isneginf(logits[0, 4]))
        self.assertEqual(float(logits[0, 5].detach()), 6.0)
        self.assertEqual(int(logits.argmax(dim=1).item()), 3)
        logits.sum().backward()
        self.assertTrue(torch.equal(registry.grad, torch.ones_like(registry)))

    def test_full_vocabulary_loss_penalizes_a_winning_base_token(self) -> None:
        physical = torch.tensor([[3, 4]])
        reserved = torch.tensor([3, 4])
        positive = torch.tensor([[True, False]])
        registry = torch.tensor([[6.0, 1.0]])
        good = full_vocabulary_selection_loss(
            torch.tensor([[0.0, 2.0, 1.0, 0.0, 0.0]]),
            registry,
            physical,
            reserved,
            positive,
        )
        bad = full_vocabulary_selection_loss(
            torch.tensor([[0.0, 8.0, 1.0, 0.0, 0.0]]),
            registry,
            physical,
            reserved,
            positive,
        )
        self.assertLess(float(good), float(bad))

    def test_canonical_arguments_are_stable_and_compact(self) -> None:
        self.assertEqual(canonical_arguments({"z": 2, "a": True}), '{"a":true,"z":2}')

    def test_schema_target_uses_every_declared_property(self) -> None:
        tool = PreparedTool(
            identity_hash="tool",
            group_hash="group",
            split="train",
            document="document",
            tool_name="tool",
            endpoint_name="endpoint",
            source="test",
            parameters={
                "properties": {
                    "required_key": {"type": "string"},
                    "optional_key": {"type": "integer", "default": 1},
                },
                "required": ["required_key"],
            },
        )
        self.assertEqual(
            schema_key_object(tool),
            {"optional_key": None, "required_key": None},
        )
        self.assertEqual(
            canonical_schema_keys(tool),
            '{"optional_key":null,"required_key":null}',
        )

    def test_schema_target_is_empty_without_properties(self) -> None:
        tool = PreparedTool(
            identity_hash="tool",
            group_hash="group",
            split="train",
            document="document",
            tool_name="tool",
            endpoint_name="endpoint",
            source="test",
            parameters=None,
        )
        self.assertEqual(schema_key_object(tool), {})

    def test_registration_bundle_uses_one_document_forward(self) -> None:
        hidden_size = 8
        backbone = _Backbone(hidden_size)
        memory_compiler = TokenResamplerMemory(
            hidden_size, rank=4, num_slots=3, initial_norm=2.0
        )
        output_compiler = PhysicalOutputGenerator(
            hidden_size, rank=4, initial_row_norm=2.0
        )
        model = MetaReadbackModel(backbone, memory_compiler, output_compiler)
        output_rows, memory = model.register_bundle(
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            }
        )
        self.assertEqual(backbone.model.calls, 1)
        self.assertEqual(tuple(output_rows.shape), (2, hidden_size))
        self.assertEqual(tuple(memory.shape), (2, 3, hidden_size))

    def test_selection_batch_uses_one_size_for_multi_positive_rows(self) -> None:
        episodes = [
            PreparedRetrievalEpisode(
                query_hash="q1",
                query="query one",
                split="train",
                target_identity_hashes=("a",),
            ),
            PreparedRetrievalEpisode(
                query_hash="q2",
                query="query two",
                split="train",
                target_identity_hashes=tuple(f"t{index}" for index in range(11)),
            ),
        ]
        self.assertEqual(selection_registry_size_for_batch(episodes, 8), 11)
        self.assertEqual(selection_registry_size_for_batch(episodes[:1], 8), 8)

    def test_slot_binding_is_deterministic_and_epoch_rebound(self) -> None:
        example = _example(1)
        first = logical_slot_for_example(example, range(10, 100), seed=17, epoch=0)
        repeated = logical_slot_for_example(example, range(10, 100), seed=17, epoch=0)
        rebound = logical_slot_for_example(example, range(10, 100), seed=17, epoch=1)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, rebound)
        self.assertTrue(10 <= first < 100)

    def test_epoch_batches_partition_equal_work_across_ranks(self) -> None:
        examples = [_example(index) for index in range(11)]
        rank_zero = list(
            _epoch_batches(
                examples,
                epoch=0,
                seed=17,
                batch_size=2,
                rank=0,
                world_size=2,
            )
        )
        rank_one = list(
            _epoch_batches(
                examples,
                epoch=0,
                seed=17,
                batch_size=2,
                rank=1,
                world_size=2,
            )
        )
        self.assertEqual(len(rank_zero), len(rank_one))
        zero_ids = {item.source_id for batch in rank_zero for item in batch}
        one_ids = {item.source_id for batch in rank_one for item in batch}
        self.assertFalse(zero_ids.intersection(one_ids))
        self.assertEqual(len(zero_ids | one_ids), 8)

    def test_wrong_memory_indices_always_choose_another_tool(self) -> None:
        identities = ["a", "a", "b"]
        indices = wrong_memory_indices(identities)
        self.assertIsNotNone(indices)
        assert indices is not None
        self.assertEqual(len(indices), len(identities))
        self.assertTrue(
            all(
                identities[index] != identities[wrong]
                for index, wrong in enumerate(indices)
            )
        )
        self.assertIsNone(wrong_memory_indices(["a", "a"]))

    def test_wrong_tool_lookup_is_deterministic_and_has_no_self_pairs(self) -> None:
        examples = [
            PreparedReadbackExample(
                source_id=f"row:{index}",
                query_hash=f"q{index}",
                query="query",
                split="validation",
                tool_identity_hash=identity,
                arguments={},
                call_index=0,
                call_count=1,
            )
            for index, identity in enumerate(["c", "a", "b", "a"])
        ]
        lookup = wrong_tool_lookup(examples)
        self.assertEqual(lookup, {"a": "b", "b": "c", "c": "a"})
        self.assertTrue(all(identity != wrong for identity, wrong in lookup.items()))


if __name__ == "__main__":
    unittest.main()
