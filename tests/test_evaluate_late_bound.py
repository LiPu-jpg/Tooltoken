import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from latent_register.controlled_registry import ControlledRegistry
from latent_register.evaluate_late_bound import (
    RegisteredBank,
    _binding_fields,
    _build_registries,
    _common_document_argument_predictions,
    _selected_payload_audit,
    apply_registration_control,
    apply_nearest_reference_control,
    address_pool_for,
    candidate_splits_for,
    latent_condition,
    load_evaluation_address_pools,
    load_shared_registry_binding,
    load_training_provenance,
)
from latent_register.episodic_data import PreparedTool


class EvaluateLateBoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pools = {
            "train": range(0, 8),
            "validation": range(8, 12),
            "test": range(12, 16),
        }

    def test_maps_all_four_tool_address_quadrants(self) -> None:
        self.assertEqual(
            latent_condition("seen_tool_seen_token", "seen"),
            "seen_tool_seen_address",
        )
        self.assertEqual(
            latent_condition("seen_tool_seen_token", "unseen"),
            "seen_tool_unseen_address",
        )
        self.assertEqual(
            latent_condition("unseen_tool_unseen_token", "seen"),
            "unseen_tool_seen_address",
        )
        self.assertEqual(
            latent_condition("unseen_tool_unseen_token", "unseen"),
            "unseen_tool_unseen_address",
        )

    def test_common_document_generation_fetches_the_selected_tool_document(self) -> None:
        tool = PreparedTool(
            identity_hash="selected-tool",
            group_hash="group",
            split="test",
            document="name: weather\nparameters: city, date",
            tool_name="weather",
            endpoint_name="forecast",
            source="fixture",
        )
        selection = {
            "predicted_tools": ["selected-tool"],
            "selected_physical_token_id": 151681,
            "selected_physical_binding_verified": True,
            "ordinary_token_win": False,
        }
        generated_prompts = []
        argument_model = object()
        argument_tokenizer = object()
        generated_with = []

        def fake_generate(model, tokenizer, prompts, **_kwargs):
            generated_with.append((model, tokenizer))
            generated_prompts.extend(prompts)
            return ['{"city":"Harbin"}']

        with (
            patch(
                "latent_register.evaluate_late_bound._retrieval_predictions",
                return_value=[selection],
            ),
            patch(
                "latent_register.evaluate_late_bound.generate_arguments",
                side_effect=fake_generate,
            ),
        ):
            predictions = _common_document_argument_predictions(
                SimpleNamespace(backbone=object()),
                object(),
                argument_model,
                argument_tokenizer,
                {tool.identity_hash: tool},
                [{"query": "Weather in Harbin?"}],
                [object()],
                object(),
                object(),
                selection_batch_size=2,
                generation_batch_size=3,
                max_query_length=128,
                max_prompt_length=512,
                max_new_tokens=32,
                device=torch.device("cpu"),
            )

        self.assertEqual(predictions[0]["predicted_arguments"], '{"city":"Harbin"}')
        self.assertIn(tool.document, generated_prompts[0])
        self.assertNotIn("151681", generated_prompts[0])
        self.assertEqual(generated_with, [(argument_model, argument_tokenizer)])
        self.assertTrue(predictions[0]["selected_document_dereference_verified"])
        self.assertEqual(
            predictions[0]["selected_document_identity"], tool.identity_hash
        )
        self.assertEqual(
            predictions[0]["selected_document_sha256"],
            hashlib.sha256(tool.document.encode("utf-8")).hexdigest(),
        )

    def test_selected_payload_audit_recomputes_binding_and_document_counts(self) -> None:
        audit = _selected_payload_audit(
            [
                {
                    "predicted_tools": ["tool-a"],
                    "selected_physical_token_id": 151681,
                    "selected_physical_binding_verified": True,
                }
            ],
            [
                {
                    "predicted_tools": ["tool-b"],
                    "selected_physical_token_id": 151682,
                    "selected_physical_binding_verified": True,
                    "selected_document_identity": "tool-b",
                    "selected_document_dereference_verified": True,
                }
            ],
            information_condition="common_document",
        )
        self.assertEqual(audit["selected_physical_binding_count"], 2)
        self.assertTrue(audit["selected_physical_bindings_verified"])
        self.assertEqual(audit["selected_argument_payload_count"], 1)
        self.assertTrue(audit["selected_argument_payloads_verified"])

    def test_selected_payload_audit_rejects_cross_identity_document(self) -> None:
        audit = _selected_payload_audit(
            [],
            [
                {
                    "predicted_tools": ["tool-a"],
                    "selected_physical_token_id": 151681,
                    "selected_physical_binding_verified": True,
                    "selected_document_identity": "tool-b",
                    "selected_document_dereference_verified": True,
                }
            ],
            information_condition="common_document",
        )
        self.assertTrue(audit["selected_physical_bindings_verified"])
        self.assertFalse(audit["selected_argument_payloads_verified"])

    def test_uses_only_heldout_addresses_for_unseen_conditions(self) -> None:
        seen_tool_pool = address_pool_for(
            self.pools, "seen_tool_seen_token", "unseen"
        )
        unseen_tool_pool = address_pool_for(
            self.pools, "unseen_tool_unseen_token", "unseen"
        )
        self.assertEqual(seen_tool_pool, self.pools["validation"])
        self.assertEqual(unseen_tool_pool, self.pools["test"])
        self.assertFalse(set(seen_tool_pool) & set(self.pools["train"]))
        self.assertFalse(set(unseen_tool_pool) & set(self.pools["train"]))

    def test_training_provenance_requires_strict_zero_leakage_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "completed_steps": 2000,
                        "world_size": 8,
                        "config": {"selection_softmax": "full_vocabulary"},
                        "physical_token_audit": {
                            "total_reserved_tokens": 8192,
                            "atomic_tokenization_verified": True,
                            "distinct_reserved_token_ids": 8192,
                            "reserved_token_ids_contiguous": True,
                            "reserved_token_id_mapping_sha256": "a" * 64,
                            "evaluation_ids_seen_during_training": 0,
                            "evaluation_input_row_max_change": 0.0,
                            "evaluation_output_row_max_change": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            provenance = load_training_provenance(path)
            self.assertEqual(provenance["completed_steps"], 2000)
            self.assertTrue(provenance["physical_token_identity"]["passed"])
            self.assertEqual(len(provenance["results_sha256"]), 64)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["physical_token_audit"][
                "evaluation_ids_seen_during_training"
            ] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "isolation failed"):
                load_training_provenance(path)

    def test_binding_fields_prove_atomic_identity_to_physical_id_mapping(self) -> None:
        registry = ControlledRegistry(
            identities=("tool-a", "tool-b"),
            positive_positions=(1,),
            address_slots=(12, 15),
            identity_sha256="identity-hash",
        )
        fields = _binding_fields(registry, torch.tensor([151681, 151684]))
        self.assertEqual(fields["registry_address_slots"], [12, 15])
        self.assertEqual(fields["registry_physical_token_ids"], [151681, 151684])
        self.assertEqual(fields["reference_physical_token_ids"], [151684])
        self.assertEqual(len(fields["registry_physical_binding_sha256"]), 64)

    def test_compact_binding_keeps_target_and_join_hash(self) -> None:
        registry = ControlledRegistry(
            identities=("tool-a", "tool-b"),
            positive_positions=(1,),
            address_slots=(12, 15),
            identity_sha256="identity-hash",
        )
        fields = _binding_fields(
            registry,
            torch.tensor([151681, 151684]),
            include_registry_arrays=False,
        )
        self.assertNotIn("registry_address_slots", fields)
        self.assertNotIn("registry_physical_token_ids", fields)
        self.assertEqual(fields["reference_physical_token_ids"], [151684])
        self.assertEqual(len(fields["registry_physical_binding_sha256"]), 64)

    def test_shared_registry_builder_reuses_one_mapping(self) -> None:
        rows = [
            {
                "family": "retrieval",
                "condition": "unseen_tool_unseen_token",
                "example_id": "one",
                "reference_tools": ["a"],
            },
            {
                "family": "arguments",
                "condition": "unseen_tool_unseen_token",
                "example_id": "two",
                "reference_tools": ["b"],
            },
        ]
        registries = _build_registries(
            rows,
            ["a", "b", "c"],
            registry_size=3,
            registry_seed=17,
            address_pool=range(12, 20),
            registry_scope="shared",
        )
        self.assertIs(registries[0].identities, registries[1].identities)
        self.assertIs(registries[0].address_slots, registries[1].address_slots)

    def test_expanded_manifest_must_preserve_training_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            expanded = root / "expanded.json"
            source.write_text(
                json.dumps(
                    {
                        "token_pools": {
                            "train": {"start_inclusive": 0, "end_exclusive": 8},
                            "validation": {"start_inclusive": 8, "end_exclusive": 12},
                            "test": {"start_inclusive": 12, "end_exclusive": 16},
                        }
                    }
                ),
                encoding="utf-8",
            )
            expanded.write_text(
                json.dumps(
                    {
                        "token_pools": {
                            "train": {"start_inclusive": 0, "end_exclusive": 8},
                            "validation": {"start_inclusive": 8, "end_exclusive": 55},
                            "test": {"start_inclusive": 55, "end_exclusive": 102},
                        }
                    }
                ),
                encoding="utf-8",
            )
            pools, selected = load_evaluation_address_pools(source, expanded)
            self.assertEqual(len(pools["test"]), 47)
            self.assertEqual(selected, expanded)

            payload = json.loads(expanded.read_text())
            payload["token_pools"]["train"]["end_exclusive"] = 7
            payload["token_pools"]["validation"]["start_inclusive"] = 7
            expanded.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed the training"):
                load_evaluation_address_pools(source, expanded)

    def test_candidate_split_all_is_explicit_mixed_universe(self) -> None:
        self.assertEqual(
            candidate_splits_for(["all"], "unseen_tool_unseen_token"),
            ("train", "validation", "test"),
        )
        self.assertEqual(
            candidate_splits_for([], "unseen_tool_unseen_token"), ("test",)
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            candidate_splits_for(["all", "test"], "unseen_tool_unseen_token")

    def test_registration_controls_preserve_shape_and_isolate_semantics(self) -> None:
        bank = RegisteredBank(
            identities=("a", "b", "c"),
            output_rows=torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            ),
            memory=torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
            wall_seconds=1.0,
            peak_memory_bytes=10,
        )
        blank = apply_registration_control(bank, "blank", seed=17)
        self.assertEqual(torch.count_nonzero(blank.output_rows).item(), 0)
        self.assertEqual(torch.count_nonzero(blank.memory).item(), 0)
        query_only = apply_registration_control(bank, "query_only", seed=17)
        self.assertEqual(torch.count_nonzero(query_only.output_rows).item(), 0)
        self.assertEqual(torch.count_nonzero(query_only.memory).item(), 0)

        shared = apply_registration_control(bank, "shared_vector", seed=17)
        self.assertTrue(torch.equal(shared.output_rows[0], shared.output_rows[2]))
        self.assertTrue(torch.equal(shared.memory[0], shared.memory[2]))

        wrong = apply_registration_control(bank, "wrong_memory", seed=17)
        self.assertTrue(torch.equal(wrong.output_rows, bank.output_rows))
        self.assertFalse(torch.equal(wrong.memory, bank.memory))

        permuted = apply_registration_control(bank, "permuted", seed=17)
        self.assertFalse(torch.equal(permuted.output_rows, bank.output_rows))
        self.assertFalse(torch.equal(permuted.memory, bank.memory))

        random = apply_registration_control(bank, "random", seed=17)
        repeated = apply_registration_control(bank, "random", seed=17)
        self.assertTrue(torch.equal(random.output_rows, repeated.output_rows))
        self.assertTrue(torch.equal(random.memory, repeated.memory))
        self.assertAlmostEqual(
            random.output_rows.norm(dim=-1).mean().item(),
            bank.output_rows.norm(dim=-1).mean().item(),
            places=5,
        )

    def test_nearest_reference_control_reuses_complete_reference_bundle(self) -> None:
        bank = RegisteredBank(
            identities=("new-a", "new-b"),
            output_rows=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            memory=torch.zeros(2, 1, 2),
            wall_seconds=1.0,
            peak_memory_bytes=10,
        )
        reference = RegisteredBank(
            identities=("known-x", "known-y", "known-z"),
            output_rows=torch.tensor(
                [[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]]
            ),
            memory=torch.tensor(
                [[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]]
            ),
            wall_seconds=2.0,
            peak_memory_bytes=20,
        )
        result = apply_nearest_reference_control(
            bank, reference, query_chunk_size=1, reference_chunk_size=2
        )
        self.assertEqual(result.nearest_reference_indices, (0, 1))
        self.assertTrue(
            torch.equal(result.bank.output_rows, reference.output_rows[:2])
        )
        self.assertTrue(torch.equal(result.bank.memory, reference.memory[:2]))
        self.assertGreater(result.cosine_similarities[0], 0.99)

    def test_loads_and_validates_a_precomputed_shared_binding(self) -> None:
        rows = [
            {"reference_tools": ["b"]},
            {"reference_tools": ["a", "c"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binding.json"
            identities = ["a", "b", "c"]
            path.write_text(
                json.dumps(
                    {
                        "identities": identities,
                        "address_slots": [12, 13, 15],
                        "identity_sha256": hashlib.sha256(
                            "\n".join(identities).encode("utf-8")
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            registries = load_shared_registry_binding(
                path, rows, ["a", "b", "c", "d"], range(12, 20)
            )
            self.assertIs(registries[0].identities, registries[1].identities)
            self.assertEqual(registries[0].positive_positions, (1,))
            self.assertEqual(registries[1].positive_positions, (0, 2))

            payload = json.loads(path.read_text())
            payload["address_slots"][2] = 13
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reused an address"):
                load_shared_registry_binding(
                    path, rows, ["a", "b", "c", "d"], range(12, 20)
                )


if __name__ == "__main__":
    unittest.main()
