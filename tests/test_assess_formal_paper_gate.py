from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from latent_register.aggregate_formal_pipeline import (
    CONTROLS,
    EXPECTED_CELL_COUNT,
    MATCHED_REGISTRY_SIZES,
    SCALE_REGISTRY_SIZES,
)
from latent_register.assess_formal_paper_gate import (
    OUTPUT_CONTROLS,
    assess_formal_paper_gate,
)


SEEDS = (17, 29, 43)


def summary(value):
    return {
        "values_by_seed": {str(seed): value for seed in SEEDS},
        "mean": value,
        "sample_std": 0.0,
        "seed_t_95_ci": [value, value],
        "seed_count": 3,
    }


def paired_metric(baseline, candidate, count=2500):
    delta = candidate - baseline
    lower = delta / 2 if delta > 0 else delta
    return {
        "count_per_seed": count,
        "baseline": summary(baseline),
        "candidate": summary(candidate),
        "delta": summary(delta),
        "paired_bootstrap_95_ci_by_seed": {
            str(seed): [lower, delta] for seed in SEEDS
        },
    }


def cell(families):
    return {
        "code_hashes": {
            "compare_controlled_predictions_code_sha256": "e" * 64,
            "benchmark_metrics_code_sha256": "b" * 64,
        },
        "families": {
            family: {"metrics": metrics} for family, metrics in families.items()
        }
    }


class AssessFormalPaperGateTests(unittest.TestCase):
    @staticmethod
    def write_simple_evaluation(
        root: Path,
        name: str,
        common_agent: dict,
        config_sha256: str,
        information_condition: str,
        checkpoint_hashes: dict | None = None,
        extra: dict | None = None,
    ) -> tuple[Path, str]:
        output = root / name
        output.mkdir()
        (output / "COMPLETE").touch()
        predictions = output / "predictions.jsonl"
        predictions.write_text("{}\n", encoding="utf-8")
        prediction_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
        result = {
            "predictions_sha256": prediction_hash,
            "information_condition": information_condition,
            "common_document_agent": common_agent,
            "common_document_agent_config_sha256": config_sha256,
        }
        if checkpoint_hashes is not None:
            result["checkpoint_hashes"] = checkpoint_hashes
        if extra is not None:
            result.update(extra)
        (output / "results.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return predictions, prediction_hash

    def write_fixture(self, root: Path) -> Path:
        cells = {}
        for system, candidate_value in (
            ("fixed_vs_latebound", 0.79),
            ("oracle_vs_latebound", 0.75),
        ):
            for address in ("seen", "unseen"):
                for size in MATCHED_REGISTRY_SIZES:
                    cells[f"{system}/{address}-address/registry-{size}"] = cell(
                        {
                            "arguments": {
                                "end_to_end_argument_exact": paired_metric(
                                    0.80, candidate_value
                                )
                            }
                        }
                    )
            for size in SCALE_REGISTRY_SIZES:
                cells[f"{system}/unseen-address/registry-{size}"] = cell(
                    {
                        "arguments": {
                            "end_to_end_argument_exact": paired_metric(
                                0.80, candidate_value
                            )
                        }
                    }
                )
        for size in MATCHED_REGISTRY_SIZES:
            cells[f"incremental_vs_latebound/unseen-address/registry-{size}"] = cell(
                {
                    "retrieval": {"hit_at_1": paired_metric(0.85, 0.80)},
                    "arguments": {
                        "end_to_end_argument_exact": paired_metric(0.78, 0.74)
                    },
                }
            )
        for control in CONTROLS:
            families = {
                "arguments": {
                    "end_to_end_argument_exact": paired_metric(0.20, 0.40)
                }
            }
            if control in OUTPUT_CONTROLS:
                families["retrieval"] = {"hit_at_1": paired_metric(0.30, 0.90)}
            cells[f"registered_vs_control/{control}"] = cell(families)
            cells[f"registered_vs_control/{control}"]["inputs"] = {}
        cells["append/registry-100-to-1000"] = cell(
            {
                "arguments": {
                    "end_to_end_argument_exact": paired_metric(0.70, 0.69, 64)
                },
                "retrieval": {"hit_at_1": paired_metric(0.95, 0.94, 64)},
            }
        )
        cells["append/registry-100-to-1000"]["inputs"] = {}
        self.assertEqual(len(cells), EXPECTED_CELL_COUNT)

        fixed_cell = cells["fixed_vs_latebound/seen-address/registry-1000"]
        incremental_cell = cells[
            "incremental_vs_latebound/unseen-address/registry-1000"
        ]
        audit_cell = cells["oracle_vs_latebound/unseen-address/registry-1000"]
        fixed_cell["inputs"] = {}
        incremental_cell["inputs"] = {}
        audit_cell["inputs"] = {}

        agent_root = root / "shared-agent"
        agent_model = agent_root / "Qwen3-8B-Full-Document"
        agent_model.mkdir(parents=True)
        agent_config = agent_model / "config.json"
        agent_config.write_text("{}\n", encoding="utf-8")
        agent_config_sha256 = hashlib.sha256(agent_config.read_bytes()).hexdigest()
        agent_audit = agent_root / "checkpoint_audit.json"
        agent_audit.write_text(
            json.dumps(
                {
                    "kind": "qwen_full_document_checkpoint_audit",
                    "model_path": str(agent_model.resolve()),
                    "full_model_reloaded": True,
                    "embedding_tables_all_finite": True,
                }
            ),
            encoding="utf-8",
        )
        (agent_root / "COMPLETE").touch()
        common_agent = {
            "model_path": str(agent_model.resolve()),
            "audit_path": str(agent_audit.resolve()),
            "audit_sha256": hashlib.sha256(agent_audit.read_bytes()).hexdigest(),
        }
        checkpoint_hashes = {
            "retrieval_adapter_config": "1" * 64,
            "retrieval_adapter_weights": "2" * 64,
            "retrieval_compiler": "3" * 64,
            "memory_compiler": "4" * 64,
            "training_results": "5" * 64,
        }
        for seed in SEEDS:
            train_root = root / f"train-{seed}"
            final_dir = train_root / "final"
            retrieval_dir = train_root / "retrieval"
            final_dir.mkdir(parents=True)
            retrieval_dir.mkdir()
            training_results = final_dir / "results.json"
            training_results.write_text("{}\n", encoding="utf-8")
            (final_dir / "training_audit.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "completed_steps": 3000,
                        "world_size": 6,
                        "stage": "full_vocabulary_joint",
                        "physical_token_isolation": True,
                    }
                ),
                encoding="utf-8",
            )
            (retrieval_dir / "training_audit.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "completed_steps": 1000,
                        "world_size": 6,
                        "stage": "retrieval",
                        "physical_token_isolation": True,
                    }
                ),
                encoding="utf-8",
            )

            eval_dir = root / f"eval-{seed}"
            eval_dir.mkdir()
            (eval_dir / "COMPLETE").touch()
            predictions = eval_dir / "predictions.jsonl"
            predictions.write_text("{}\n", encoding="utf-8")
            prediction_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
            evaluation = {
                "kind": "qwen_late_bound",
                "code_hashes": {
                    "evaluate_late_bound": "a" * 64,
                    "benchmark_metrics": "b" * 64,
                    "model": "c" * 64,
                    "physical_tokens": "d" * 64,
                },
                "source_condition": "unseen_tool_unseen_token",
                "condition": "unseen_tool_unseen_address",
                "address_status": "unseen",
                "registry_size": 1000,
                "information_condition": "common_document",
                "registration_control": "registered",
                "common_document_agent": common_agent,
                "common_document_agent_config_sha256": agent_config_sha256,
                "checkpoint_hashes": checkpoint_hashes,
                "predictions_sha256": prediction_hash,
                "training_provenance": {
                    "results_path": str(training_results),
                    "completed_steps": 3000,
                    "world_size": 6,
                    "config": {"seed": seed},
                },
                "registration_audit": {
                    "document_encoder_batch_forwards": 63,
                    "distinct_registered_tools": 1000,
                    "wall_seconds": 1.5,
                    "peak_memory_bytes": 1024,
                    "bytes_stored_per_tool": 4096,
                    "optimizer_steps": 0,
                    "model_or_table_parameters_changed": 0,
                    "registration_forwards_per_tool": 1,
                    "selection_emits_single_physical_id": True,
                    "selected_physical_bindings_verified": True,
                    "selected_physical_binding_count": 2500,
                    "selected_physical_binding_verified_count": 2500,
                    "selected_argument_payload_count": 1000,
                    "selected_argument_payload_verified_count": 1000,
                    "selected_id_dereferences_registered_memory": False,
                    "selected_id_dereferences_full_document": True,
                    "selected_id_uses_static_embedding": False,
                    "ordinary_vocabulary_competes_at_selection": True,
                    "inactive_reserved_ids_masked": True,
                },
                "evaluation_mutation_audit": {
                    "all_reserved_input_row_max_change": 0.0,
                    "all_reserved_output_row_max_change": 0.0,
                },
                "inference_seconds": 3.0,
                "latency": {
                    "model_loading_excluded": True,
                    "retrieval_selection": {
                        "count": 2500,
                        "total_seconds": 1.0,
                        "mean_seconds_per_example": 0.0004,
                    },
                    "argument_path": {
                        "count": 1000,
                        "total_seconds": 2.0,
                        "mean_seconds_per_example": 0.002,
                        "includes_tool_selection": True,
                        "includes_document_dereference_and_generation": True,
                    },
                },
                "ordinary_token_win_rate": 0.1,
                "physical_token_audit": {
                    "total_reserved_tokens": 8192,
                    "atomic_tokenization_verified": True,
                    "distinct_reserved_token_ids": 8192,
                    "reserved_token_ids_contiguous": True,
                    "reserved_token_id_mapping_sha256": "f" * 64,
                    "evaluation_ids_seen_during_training": 0,
                    "evaluation_input_row_max_change": 0.0,
                    "evaluation_output_row_max_change": 0.0,
                },
            }
            (eval_dir / "results.json").write_text(
                json.dumps(evaluation), encoding="utf-8"
            )
            fixed_predictions, fixed_hash = self.write_simple_evaluation(
                root,
                f"fixed-{seed}",
                common_agent,
                agent_config_sha256,
                "common_document",
                extra={
                    "kind": "qwen_toolgen_fixed",
                    "model_path": str((root / f"fixed-model-{seed}").resolve()),
                    "candidate_splits": ["train"],
                },
            )
            incremental_predictions, incremental_hash = self.write_simple_evaluation(
                root,
                f"incremental-{seed}",
                common_agent,
                agent_config_sha256,
                "common_document",
                extra={
                    "kind": "qwen_toolgen_fixed",
                    "model_path": str(
                        (root / f"incremental-model-{seed}").resolve()
                    ),
                    "candidate_splits": ["test"],
                },
            )
            oracle_predictions, oracle_hash = self.write_simple_evaluation(
                root,
                f"oracle-{seed}",
                common_agent,
                agent_config_sha256,
                "full_document_oracle",
                extra={
                    "kind": "qwen_full_document",
                    "model_path": common_agent["model_path"],
                },
            )
            latent_seen_predictions, latent_seen_hash = self.write_simple_evaluation(
                root,
                f"latent-seen-{seed}",
                common_agent,
                agent_config_sha256,
                "common_document",
                checkpoint_hashes,
                {
                    "kind": "qwen_late_bound",
                    "condition": "seen_tool_seen_address",
                    "source_condition": "seen_tool_seen_token",
                    "address_status": "seen",
                    "registry_size": 1000,
                    "registration_control": "registered",
                },
            )
            fixed_cell["inputs"][str(seed)] = {
                "baseline_predictions_path": str(fixed_predictions),
                "baseline_predictions_sha256": fixed_hash,
                "candidate_predictions_path": str(latent_seen_predictions),
                "candidate_predictions_sha256": latent_seen_hash,
            }
            incremental_cell["inputs"][str(seed)] = {
                "baseline_predictions_path": str(incremental_predictions),
                "baseline_predictions_sha256": incremental_hash,
                "candidate_predictions_path": str(predictions),
                "candidate_predictions_sha256": prediction_hash,
            }
            for control in CONTROLS:
                control_predictions, control_hash = self.write_simple_evaluation(
                    root,
                    f"control-{control}-{seed}",
                    common_agent,
                    agent_config_sha256,
                    "common_document",
                    checkpoint_hashes,
                    {
                        "kind": "qwen_late_bound",
                        "registration_control": control,
                        "source_condition": "unseen_tool_unseen_token",
                        "address_status": "unseen",
                        "registry_size": 1000,
                    },
                )
                cells[f"registered_vs_control/{control}"]["inputs"][str(seed)] = {
                    "baseline_predictions_path": str(control_predictions),
                    "baseline_predictions_sha256": control_hash,
                    "candidate_predictions_path": str(predictions),
                    "candidate_predictions_sha256": prediction_hash,
                }
            audit_cell["inputs"][str(seed)] = {
                "baseline_predictions_path": str(oracle_predictions),
                "baseline_predictions_sha256": oracle_hash,
                "candidate_predictions_path": str(predictions),
                "candidate_predictions_sha256": prediction_hash,
            }
            quadrant_specs = []
            for system, tool_status in (
                ("fixed_vs_latebound", "seen"),
                ("oracle_vs_latebound", "unseen"),
            ):
                for address in ("seen", "unseen"):
                    for size in MATCHED_REGISTRY_SIZES:
                        quadrant_specs.append((system, tool_status, address, size))
                for size in SCALE_REGISTRY_SIZES:
                    quadrant_specs.append((system, tool_status, "unseen", size))
            for size in MATCHED_REGISTRY_SIZES:
                quadrant_specs.append(
                    ("incremental_vs_latebound", "unseen", "unseen", size)
                )
            for system, tool_status, address, size in quadrant_specs:
                cell_name = f"{system}/{address}-address/registry-{size}"
                source = cells[cell_name].setdefault("inputs", {})
                if str(seed) in source:
                    continue
                candidate_predictions, candidate_hash = self.write_simple_evaluation(
                    root,
                    f"quadrant-{system}-{tool_status}-{address}-{size}-{seed}",
                    common_agent,
                    agent_config_sha256,
                    "common_document",
                    checkpoint_hashes,
                    {
                        "kind": "qwen_late_bound",
                        "condition": f"{tool_status}_tool_{address}_address",
                        "source_condition": (
                            "seen_tool_seen_token"
                            if tool_status == "seen"
                            else "unseen_tool_unseen_token"
                        ),
                        "address_status": address,
                        "registry_size": size,
                        "registration_control": "registered",
                    },
                )
                source[str(seed)] = {
                    "candidate_predictions_path": str(candidate_predictions),
                    "candidate_predictions_sha256": candidate_hash,
                }
            binding_root = root / f"append-bindings-{seed}"
            binding_root.mkdir()
            identities = [f"tool-{index:04d}" for index in range(1000)]
            slots = list(range(1000))
            initial_binding = binding_root / "registry_initial.json"
            extended_binding = binding_root / "registry_extended.json"
            initial_binding.write_text(
                json.dumps(
                    {
                        "registry_size": 100,
                        "identities": identities[:100],
                        "address_slots": slots[:100],
                    }
                ),
                encoding="utf-8",
            )
            extended_binding.write_text(
                json.dumps(
                    {
                        "registry_size": 1000,
                        "identities": identities,
                        "address_slots": slots,
                    }
                ),
                encoding="utf-8",
            )
            initial_binding_hash = hashlib.sha256(
                initial_binding.read_bytes()
            ).hexdigest()
            extended_binding_hash = hashlib.sha256(
                extended_binding.read_bytes()
            ).hexdigest()
            append_initial, append_initial_hash = self.write_simple_evaluation(
                root,
                f"append-initial-{seed}",
                common_agent,
                agent_config_sha256,
                "registered_memory",
                checkpoint_hashes,
                {
                    "kind": "qwen_late_bound",
                    "registration_control": "registered",
                    "source_condition": "unseen_tool_unseen_token",
                    "address_status": "unseen",
                    "registry_size": 100,
                    "prediction_count": 64,
                    "input_registry_binding": str(initial_binding.resolve()),
                    "input_registry_binding_sha256": initial_binding_hash,
                },
            )
            append_extended, append_extended_hash = self.write_simple_evaluation(
                root,
                f"append-extended-{seed}",
                common_agent,
                agent_config_sha256,
                "registered_memory",
                checkpoint_hashes,
                {
                    "kind": "qwen_late_bound",
                    "registration_control": "registered",
                    "source_condition": "unseen_tool_unseen_token",
                    "address_status": "unseen",
                    "registry_size": 1000,
                    "prediction_count": 64,
                    "input_registry_binding": str(extended_binding.resolve()),
                    "input_registry_binding_sha256": extended_binding_hash,
                },
            )
            cells["append/registry-100-to-1000"]["inputs"][str(seed)] = {
                "baseline_predictions_path": str(append_initial),
                "baseline_predictions_sha256": append_initial_hash,
                "candidate_predictions_path": str(append_extended),
                "candidate_predictions_sha256": append_extended_hash,
            }

        aggregate = {
            "kind": "formal_three_seed_pipeline_aggregate",
            "seeds": list(SEEDS),
            "cell_count": EXPECTED_CELL_COUNT,
            "all_cells_complete": True,
            "cells": cells,
        }
        output = root / "aggregate.json"
        output.write_text(json.dumps(aggregate), encoding="utf-8")
        self.write_complete(output)
        return output

    @staticmethod
    def write_complete(path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (path.parent / "COMPLETE").write_text(
            f"{digest}  aggregate.json\n", encoding="utf-8"
        )

    def test_passes_every_predeclared_paper_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            result = assess_formal_paper_gate(aggregate)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_checks"], [])
        self.assertEqual(len(result["checks"]), 12)
        self.assertIn("closed_set_registry_100", result["measurements"])
        self.assertIn("oracle_retention_registry_47000", result["measurements"])
        incremental = result["measurements"]["incremental_comparison_registry_100"]
        self.assertAlmostEqual(
            incremental["retrieval"]["late_bound_minus_incremental_mean"], -0.05
        )
        self.assertEqual(incremental["arguments"]["incremental_mean"], 0.78)

    def test_reports_closed_set_drop_without_turning_it_into_a_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            metric = payload["cells"][
                "fixed_vs_latebound/seen-address/registry-100"
            ]["families"]["arguments"]["metrics"]["end_to_end_argument_exact"]
            metric["delta"] = summary(-0.03)
            aggregate.write_text(json.dumps(payload), encoding="utf-8")
            self.write_complete(aggregate)
            result = assess_formal_paper_gate(aggregate)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["measurements"]["closed_set_registry_100"]["delta_mean"],
            -0.03,
        )
        self.assertTrue(result["checks"]["selection_control_blank"]["passed"])

    def test_fails_when_registered_selection_does_not_beat_a_control(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            metric = payload["cells"]["registered_vs_control/random"][
                "families"
            ]["retrieval"]["metrics"]["hit_at_1"]
            metric["delta"] = summary(0.0)
            metric["paired_bootstrap_95_ci_by_seed"] = {
                str(seed): [-0.01, 0.01] for seed in SEEDS
            }
            aggregate.write_text(json.dumps(payload), encoding="utf-8")
            self.write_complete(aggregate)
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn("selection_control_random", result["failed_checks"])

    def test_fails_when_selection_control_has_fewer_than_2000_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            metric = payload["cells"]["registered_vs_control/random"][
                "families"
            ]["retrieval"]["metrics"]["hit_at_1"]
            metric["count_per_seed"] = 1999
            aggregate.write_text(json.dumps(payload), encoding="utf-8")
            self.write_complete(aggregate)
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        evidence = result["checks"]["selection_control_random"]["retrieval"]
        self.assertFalse(evidence["count_requirement_passed"])
        self.assertEqual(evidence["minimum_count_per_seed"], 2000)

    def test_fails_when_one_system_uses_another_common_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            source = payload["cells"][
                "incremental_vs_latebound/unseen-address/registry-1000"
            ]["inputs"][str(SEEDS[0])]
            results_path = (
                Path(source["baseline_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["common_document_agent"]["audit_sha256"] = "9" * 64
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "shared_agent_and_latebound_checkpoint_identity",
            result["failed_checks"],
        )

    def test_fails_when_incremental_reuses_the_fixed_selection_model(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            fixed_source = payload["cells"][
                "fixed_vs_latebound/seen-address/registry-1000"
            ]["inputs"][str(SEEDS[0])]
            incremental_source = payload["cells"][
                "incremental_vs_latebound/unseen-address/registry-1000"
            ]["inputs"][str(SEEDS[0])]
            fixed_results = json.loads(
                (
                    Path(fixed_source["baseline_predictions_path"]).parent
                    / "results.json"
                ).read_text(encoding="utf-8")
            )
            incremental_results_path = (
                Path(incremental_source["baseline_predictions_path"]).parent
                / "results.json"
            )
            incremental_results = json.loads(
                incremental_results_path.read_text(encoding="utf-8")
            )
            incremental_results["model_path"] = fixed_results["model_path"]
            incremental_results_path.write_text(
                json.dumps(incremental_results), encoding="utf-8"
            )
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "shared_agent_and_latebound_checkpoint_identity",
            result["failed_checks"],
        )

    def test_fails_when_latebound_cells_mix_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            source = payload["cells"][
                "fixed_vs_latebound/seen-address/registry-1000"
            ]["inputs"][str(SEEDS[0])]
            results_path = (
                Path(source["candidate_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["checkpoint_hashes"]["memory_compiler"] = "9" * 64
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "shared_agent_and_latebound_checkpoint_identity",
            result["failed_checks"],
        )

    def test_fails_when_control_cell_has_another_treatment_label(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            source = payload["cells"]["registered_vs_control/blank"]["inputs"][
                str(SEEDS[0])
            ]
            results_path = (
                Path(source["baseline_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["registration_control"] = "random"
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn("control_treatment_identity", result["failed_checks"])

    def test_fails_when_cell_path_hides_another_quadrant(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            source = payload["cells"][
                "fixed_vs_latebound/unseen-address/registry-100"
            ]["inputs"][str(SEEDS[0])]
            results_path = (
                Path(source["candidate_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["condition"] = "unseen_tool_unseen_address"
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn("latebound_quadrant_identity", result["failed_checks"])

    def test_fails_when_append_rebinds_an_old_address(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            source = payload["cells"]["append/registry-100-to-1000"]["inputs"][
                str(SEEDS[0])
            ]
            results_path = (
                Path(source["candidate_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            binding_path = Path(results["input_registry_binding"])
            binding = json.loads(binding_path.read_text(encoding="utf-8"))
            binding["address_slots"][0] = 999
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            results["input_registry_binding_sha256"] = hashlib.sha256(
                binding_path.read_bytes()
            ).hexdigest()
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn("append_identity_prefix", result["failed_checks"])

    def test_rejects_tampered_aggregate_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            aggregate.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                assess_formal_paper_gate(aggregate)

    def test_fails_when_one_comparison_cell_uses_different_code(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            next(iter(payload["cells"].values()))["code_hashes"][
                "benchmark_metrics_code_sha256"
            ] = "f" * 64
            aggregate.write_text(json.dumps(payload), encoding="utf-8")
            self.write_complete(aggregate)
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn("comparison_code_consistency", result["failed_checks"])

    def test_fails_formal_audit_when_physical_ids_are_not_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            cell_payload = payload["cells"][
                "oracle_vs_latebound/unseen-address/registry-1000"
            ]
            first_seed = cell_payload["inputs"][str(SEEDS[0])]
            results_path = Path(first_seed["candidate_predictions_path"]).parent / "results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["physical_token_audit"]["distinct_reserved_token_ids"] = 8191
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "formal_training_and_registration_audits", result["failed_checks"]
        )

    def test_fails_formal_audit_when_static_embedding_replaces_dereference(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            cell_payload = payload["cells"][
                "oracle_vs_latebound/unseen-address/registry-1000"
            ]
            first_seed = cell_payload["inputs"][str(SEEDS[0])]
            results_path = Path(first_seed["candidate_predictions_path"]).parent / "results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["registration_audit"]["selected_id_uses_static_embedding"] = True
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "formal_training_and_registration_audits", result["failed_checks"]
        )

    def test_fails_formal_audit_when_selected_document_is_not_dereferenced(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            cell_payload = payload["cells"][
                "oracle_vs_latebound/unseen-address/registry-1000"
            ]
            first_seed = cell_payload["inputs"][str(SEEDS[0])]
            results_path = (
                Path(first_seed["candidate_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["registration_audit"][
                "selected_id_dereferences_full_document"
            ] = False
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "formal_training_and_registration_audits", result["failed_checks"]
        )

    def test_fails_formal_audit_when_selected_binding_is_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            cell_payload = payload["cells"][
                "oracle_vs_latebound/unseen-address/registry-1000"
            ]
            first_seed = cell_payload["inputs"][str(SEEDS[0])]
            results_path = (
                Path(first_seed["candidate_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["registration_audit"][
                "selected_physical_bindings_verified"
            ] = False
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "formal_training_and_registration_audits", result["failed_checks"]
        )

    def test_fails_formal_audit_when_binding_counts_disagree(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            cell_payload = payload["cells"][
                "oracle_vs_latebound/unseen-address/registry-1000"
            ]
            first_seed = cell_payload["inputs"][str(SEEDS[0])]
            results_path = (
                Path(first_seed["candidate_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["registration_audit"][
                "selected_physical_binding_verified_count"
            ] -= 1
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "formal_training_and_registration_audits", result["failed_checks"]
        )

    def test_fails_formal_audit_when_runtime_reporting_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self.write_fixture(Path(directory))
            payload = json.loads(aggregate.read_text(encoding="utf-8"))
            cell_payload = payload["cells"][
                "oracle_vs_latebound/unseen-address/registry-1000"
            ]
            first_seed = cell_payload["inputs"][str(SEEDS[0])]
            results_path = (
                Path(first_seed["candidate_predictions_path"]).parent / "results.json"
            )
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["registration_audit"].pop("bytes_stored_per_tool")
            results_path.write_text(json.dumps(results), encoding="utf-8")
            result = assess_formal_paper_gate(aggregate)
        self.assertFalse(result["passed"])
        self.assertIn(
            "formal_training_and_registration_audits", result["failed_checks"]
        )


if __name__ == "__main__":
    unittest.main()
