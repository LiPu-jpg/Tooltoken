import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from latent_register.summarize_shared_agent_experiment import (
    MATCHED_SIZES,
    SCALE_SIZES,
    SELECTION_CONTROLS,
    summarize_shared_agent_experiment,
)


def metric(baseline=0.1, candidate=0.5, *, count=2002):
    return {
        "count": count,
        "baseline": baseline,
        "candidate": candidate,
        "delta": candidate - baseline,
        "paired_bootstrap_95_ci": [0.2, 0.6],
    }


class SummarizeSharedAgentExperimentTests(unittest.TestCase):
    @staticmethod
    def write_common_agent(root: Path) -> dict[str, str]:
        model = root / "Qwen3-8B-Full-Document"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n")
        audit = root / "checkpoint_audit.json"
        audit.write_text(
            json.dumps(
                {
                    "kind": "qwen_full_document_checkpoint_audit",
                    "model_path": str(model.resolve()),
                    "full_model_reloaded": True,
                    "embedding_tables_all_finite": True,
                }
            )
            + "\n"
        )
        (root / "COMPLETE").write_text("job\n")
        return {
            "model_path": str(model.resolve()),
            "audit_path": str(audit.resolve()),
            "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        }

    @staticmethod
    def write_fixed_train(root: Path) -> Path:
        root.mkdir(parents=True)
        (root / "COMPLETE").write_text("job\n")
        common = {
            "kind": "qwen_toolgen_fixed_checkpoint_audit",
            "normalized_token_count": 51_895,
            "normalized_tokens_sha256": "e" * 64,
            "token_mapping_sha256": "f" * 64,
            "tokenizer_size": 203_564,
            "first_tool_token_id": 151_669,
            "last_tool_token_id": 203_563,
            "mapping_is_contiguous_suffix_range": True,
            "full_model_reloaded": False,
        }
        audits = []
        for stage, model_name in (
            (1, "Qwen3-8B-Fixed-Memorization"),
            (2, "Qwen3-8B-Fixed-Retriever"),
            (3, "Qwen3-8B-Fixed-Agent"),
        ):
            payload = {
                **common,
                "model_path": str((root / model_name).resolve()),
            }
            if stage > 1:
                payload["reference_mapping_match"] = True
            if stage == 3:
                payload.update(
                    {
                        "full_model_reloaded": True,
                        "input_embeddings_all_finite": True,
                        "output_embeddings_all_finite": True,
                    }
                )
            audit = root / f"checkpoint_audit_stage{stage}.json"
            audit.write_text(json.dumps(payload) + "\n")
            audits.append(audit)
        (root / "checkpoint_audits.sha256").write_text(
            "".join(
                f"{hashlib.sha256(audit.read_bytes()).hexdigest()}  {audit.resolve()}\n"
                for audit in audits
            )
        )
        return root

    @staticmethod
    def write_incremental_runtime(root: Path) -> Path:
        root.mkdir(parents=True)
        runtime = root / "registration_runtime.json"
        checkpoint = root / "checkpoint_audit_stage3.json"
        prefix = root / "base_token_prefix_audit.json"
        runtime.write_text(
            json.dumps(
                {
                    "kind": "qwen_toolgen_incremental_registration_runtime",
                    "registration_optimizer_updates_required": True,
                    "model_parameters_updated": True,
                    "optimizer_steps": 56,
                    "incremental_tools": 6335,
                    "wall_seconds": 1200,
                    "wall_seconds_per_tool": 1200 / 6335,
                    "training_query_records_used": 0,
                    "training_argument_records_used": 0,
                    "training_trajectory_records_used": 0,
                }
            )
            + "\n"
        )
        checkpoint.write_text(
            json.dumps(
                {
                    "kind": "qwen_toolgen_fixed_checkpoint_audit",
                    "full_model_reloaded": True,
                    "model_path": "/models/incremental",
                }
            )
            + "\n"
        )
        prefix.write_text(
            json.dumps(
                {
                    "kind": "qwen_toolgen_fixed_checkpoint_audit",
                    "reference_subset_mapping_match": True,
                }
            )
            + "\n"
        )
        data = root / "input-data"
        data.mkdir()
        memorization = data / "fixed_incremental_memorization.json"
        incremental_tokens = data / "virtual_tokens_incremental.txt"
        combined_tokens = data / "virtual_tokens_combined.txt"
        memorization.write_text("[]\n")
        incremental_tokens.write_text("token\n")
        combined_tokens.write_text("base\ntoken\n")
        manifest = data / "incremental_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "kind": "qwen_toolgen_incremental_document_registration",
                    "split": "test",
                    "training_information": ["tool_document", "fixed_token_label"],
                    "audits": {
                        "base_incremental_token_overlap": 0,
                        "evaluation_query_records_used": 0,
                        "evaluation_argument_records_used": 0,
                        "evaluation_trajectory_records_used": 0,
                        "incremental_tokens_unique": True,
                    },
                    "counts": {
                        "base_tokens": 51_895,
                        "incremental_tools": 6_335,
                        "memorization_records": 6_335,
                        "combined_tokens": 58_230,
                    },
                    "outputs": {
                        path.name: {
                            "path": str(path.resolve()),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                        for path in (memorization, incremental_tokens, combined_tokens)
                    },
                }
            )
            + "\n"
        )
        (root / "input.sha256").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n"
                for path in (manifest, memorization, incremental_tokens, combined_tokens)
            )
        )
        code = root / "code"
        code.mkdir()
        code_files = [
            code / "audit_fixed_checkpoint.py",
            code / "prepare_incremental_fixed.py",
            code / "a100_train_controlled_toolgen_incremental.sbatch",
        ]
        for path in code_files:
            path.write_text(f"{path.name}\n")
        (root / "code.sha256").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n"
                for path in code_files
            )
        )
        for path in (runtime, checkpoint, prefix):
            (path.parent / "COMPLETE").write_text("job\n")
        (root / "output.sha256").write_text(
            "".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n"
                for path in (checkpoint, prefix, runtime)
            )
        )
        return root

    @staticmethod
    def write_comparison(root: Path, suffix: str, *, retrieval: bool = True) -> Path:
        directory = root / suffix
        directory.mkdir(parents=True)
        families = {
            "arguments": {
                "paired_metrics": {
                    "end_to_end_argument_exact": metric(count=1169)
                }
            }
        }
        if retrieval:
            families["retrieval"] = {"paired_metrics": {"hit_at_1": metric()}}
        pair_count = 3171 if retrieval else 1169
        path = directory / "comparison.json"
        baseline_predictions = directory / "baseline_predictions.jsonl"
        baseline_predictions.write_text("{}\n")
        baseline_predictions_sha256 = hashlib.sha256(
            baseline_predictions.read_bytes()
        ).hexdigest()
        common_agent_root = root.parent / "shared-agent"
        common_agent_audit = common_agent_root / "checkpoint_audit.json"
        common_agent = {
            "model_path": str(
                (common_agent_root / "Qwen3-8B-Full-Document").resolve()
            ),
            "audit_path": str(common_agent_audit.resolve()),
            "audit_sha256": hashlib.sha256(common_agent_audit.read_bytes()).hexdigest(),
        }
        registration_control = None
        if root.name in {"fixed", "fixed-scale"}:
            baseline_model_path = str(
                (root.parent / "fixed-train" / "Qwen3-8B-Fixed-Agent").resolve()
            )
            baseline_kind = "qwen_toolgen_fixed"
        elif root.name == "incremental":
            baseline_model_path = "/models/incremental"
            baseline_kind = "qwen_toolgen_fixed"
        elif root.name in {"oracle", "oracle-scale"}:
            baseline_model_path = common_agent["model_path"]
            baseline_kind = "qwen_full_document"
        else:
            baseline_model_path = "/models/latent"
            baseline_kind = "qwen_late_bound"
            registration_control = suffix.removeprefix("control-")
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "kind": baseline_kind,
                    "model_path": baseline_model_path,
                    "condition": "control",
                    "registration_control": registration_control,
                    "predictions_sha256": baseline_predictions_sha256,
                    "code_hashes": {"benchmark_metrics": "b" * 64},
                    "common_document_agent": common_agent,
                }
            )
        )
        path.write_text(
            json.dumps(
                {
                    "pair_count": pair_count,
                    "require_registry_match": retrieval,
                    "registry_pairs": pair_count if retrieval else 0,
                    "registry_mismatches": 0,
                    "selected_families": ["all"] if retrieval else ["arguments"],
                    "families": families,
                    "sources": {
                        "baseline_path": str(baseline_predictions.resolve()),
                        "baseline_sha256": baseline_predictions_sha256,
                        "candidate_sha256": "d" * 64,
                        "compare_controlled_predictions_code_sha256": "c" * 64,
                        "benchmark_metrics_code_sha256": "b" * 64,
                    },
                }
            ),
            encoding="utf-8",
        )
        (directory / "COMPLETE").touch()
        return path

    @staticmethod
    def write_latebound_result(root: Path, condition: str, size: int) -> Path:
        tool_status, address_status = condition.split("_tool_", maxsplit=1)
        address_status = address_status.removesuffix("_address")
        directory = (
            root
            / f"{tool_status}-tool_{address_status}-address"
            / f"registry-{size}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        common_agent_root = root.parent / "shared-agent"
        common_agent_audit = common_agent_root / "checkpoint_audit.json"
        result = {
            "kind": "qwen_late_bound",
            "condition": condition,
            "registry_size": size,
            "registration_control": "registered",
            "information_condition": "common_document",
            "common_document_agent": {
                "model_path": str(
                    (common_agent_root / "Qwen3-8B-Full-Document").resolve()
                ),
                "audit_path": str(common_agent_audit.resolve()),
                "audit_sha256": hashlib.sha256(
                    common_agent_audit.read_bytes()
                ).hexdigest(),
            },
            "code_hashes": {"benchmark_metrics": "b" * 64},
            "predictions_sha256": "d" * 64,
            "checkpoint_hashes": {
                "retrieval_adapter_config": "1" * 64,
                "retrieval_adapter_weights": "2" * 64,
                "retrieval_compiler": "3" * 64,
                "memory_compiler": "4" * 64,
                "training_results": "5" * 64,
            },
            "ordinary_token_win_rate": 0.1,
            "registration_audit": {
                "distinct_registered_tools": size,
                "document_encoder_batch_forwards": 1,
                "wall_seconds": 0.25,
                "peak_memory_bytes": 1024,
                "bytes_stored_per_tool": 2048,
            },
            "latency": {
                "model_loading_excluded": True,
                "retrieval_selection": {
                    "count": 2002,
                    "total_seconds": 2.0,
                    "mean_seconds_per_example": 0.001,
                },
                "argument_path": {
                    "count": 1169,
                    "total_seconds": 4.0,
                    "mean_seconds_per_example": 0.003,
                    "includes_tool_selection": True,
                    "includes_document_dereference_and_generation": True,
                },
            },
            "aggregates": {
                f"retrieval:{condition}": {
                    "count": 2002,
                    "metrics": {"hit_at_1": 0.5},
                },
                f"arguments:{condition}": {
                    "count": 1169,
                    "metrics": {"end_to_end_argument_exact": 0.4},
                },
            },
        }
        path = directory / "results.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        (directory / "COMPLETE").touch()
        return path

    def write_fixture(self, root: Path):
        paths = {
            name: root / name
            for name in (
                "fixed",
                "fixed-scale",
                "incremental",
                "oracle",
                "oracle-scale",
                "controls",
                "latebound",
                "incremental-train",
                "fixed-train",
            )
        }
        self.write_incremental_runtime(paths["incremental-train"])
        self.write_fixed_train(paths["fixed-train"])
        common_agent = self.write_common_agent(root / "shared-agent")
        for size in MATCHED_SIZES:
            self.write_comparison(
                paths["fixed"], f"seen-tool_seen-address/registry-{size}"
            )
            self.write_comparison(
                paths["fixed"], f"seen-tool_unseen-address/registry-{size}"
            )
            self.write_comparison(
                paths["incremental"],
                f"unseen-tool_unseen-address/registry-{size}",
            )
            self.write_comparison(
                paths["oracle"],
                f"unseen-tool_seen-address/registry-{size}",
                retrieval=False,
            )
            self.write_comparison(
                paths["oracle"],
                f"unseen-tool_unseen-address/registry-{size}",
                retrieval=False,
            )
        for size in SCALE_SIZES:
            self.write_comparison(
                paths["fixed-scale"],
                f"seen-tool_unseen-address/registry-{size}",
            )
            self.write_comparison(
                paths["oracle-scale"],
                f"unseen-tool_unseen-address/registry-{size}",
                retrieval=False,
            )
        for control in SELECTION_CONTROLS:
            self.write_comparison(paths["controls"], f"control-{control}")

        for size in MATCHED_SIZES:
            self.write_latebound_result(
                paths["latebound"], "seen_tool_seen_address", size
            )
            self.write_latebound_result(
                paths["latebound"], "seen_tool_unseen_address", size
            )
            self.write_latebound_result(
                paths["latebound"], "unseen_tool_seen_address", size
            )
            self.write_latebound_result(
                paths["latebound"], "unseen_tool_unseen_address", size
            )
        for size in SCALE_SIZES:
            self.write_latebound_result(
                paths["latebound"], "seen_tool_unseen_address", size
            )
            self.write_latebound_result(
                paths["latebound"], "unseen_tool_unseen_address", size
            )

        late = (
            paths["latebound"]
            / "unseen-tool_unseen-address"
            / "registry-1000"
        )
        late.mkdir(parents=True, exist_ok=True)
        result = {
            "kind": "qwen_late_bound",
            "condition": "unseen_tool_unseen_address",
            "registry_size": 1000,
            "registration_control": "registered",
            "information_condition": "common_document",
            "common_document_agent": common_agent,
            "code_hashes": {"benchmark_metrics": "b" * 64},
            "predictions_sha256": "d" * 64,
            "checkpoint_hashes": {
                "retrieval_adapter_config": "1" * 64,
                "retrieval_adapter_weights": "2" * 64,
                "retrieval_compiler": "3" * 64,
                "memory_compiler": "4" * 64,
                "training_results": "5" * 64,
            },
            "aggregates": {
                "retrieval:unseen_tool_unseen_address": {
                    "count": 2002,
                    "metrics": {"hit_at_1": 0.5},
                },
                "arguments:unseen_tool_unseen_address": {
                    "count": 1169,
                    "metrics": {"end_to_end_argument_exact": 0.4},
                },
            },
            "registration_audit": {
                "optimizer_steps": 0,
                "model_or_table_parameters_changed": 0,
                "backbone_parameters_changed": 0,
                "embedding_table_parameters_changed": 0,
                "lm_head_parameters_changed": 0,
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
                "distinct_registered_tools": 1000,
                "document_encoder_batch_forwards": 1,
                "wall_seconds": 0.25,
                "peak_memory_bytes": 1024,
                "bytes_stored_per_tool": 2048,
            },
            "evaluation_mutation_audit": {
                "all_reserved_input_row_max_change": 0.0,
                "all_reserved_output_row_max_change": 0.0,
            },
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
            "ordinary_token_win_rate": 0.1,
            "latency": {
                "model_loading_excluded": True,
                "retrieval_selection": {
                    "count": 2002,
                    "total_seconds": 2.0,
                    "mean_seconds_per_example": 0.001,
                },
                "argument_path": {
                    "count": 1169,
                    "total_seconds": 4.0,
                    "mean_seconds_per_example": 0.003,
                    "includes_tool_selection": True,
                    "includes_document_dereference_and_generation": True,
                },
            },
        }
        (late / "results.json").write_text(json.dumps(result), encoding="utf-8")
        (late / "COMPLETE").touch()
        return paths

    def summarize(self, paths):
        return summarize_shared_agent_experiment(
            fixed_root=paths["fixed"],
            fixed_scale_root=paths["fixed-scale"],
            incremental_root=paths["incremental"],
            oracle_root=paths["oracle"],
            oracle_scale_root=paths["oracle-scale"],
            control_root=paths["controls"],
            latebound_root=paths["latebound"],
            incremental_train_root=paths["incremental-train"],
            fixed_train_root=paths["fixed-train"],
        )

    def test_supports_claim_only_from_relative_controls_and_strict_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result = self.summarize(paths)
        self.assertTrue(result["core_claim_supported"])
        self.assertTrue(result["common_document_end_to_end_supported"])
        self.assertTrue(result["strict_post_training_registration"])
        self.assertEqual(len(result["code_sha256"]), 64)
        self.assertEqual(len(result["runtime_and_storage"]["cells"]), 16)
        self.assertEqual(
            result["runtime_and_storage"]["cells"]
            ["unseen_tool_unseen_address.registry_1000"]
            ["bytes_stored_per_tool"],
            2048,
        )
        self.assertEqual(
            result["runtime_and_storage"]["incremental_toolgen_registration"]
            ["incremental_tools"],
            6335,
        )
        fixed_gap = result["baseline_comparisons"]["registry_1000"][
            "fixed_seen_address"
        ]["end_to_end_argument_exact"]
        self.assertEqual(fixed_gap["candidate_minus_baseline"], 0.4)
        self.assertEqual(fixed_gap["candidate_over_baseline"], 5.0)

    def test_failed_control_interval_blocks_selection_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            comparison = paths["controls"] / "control-random" / "comparison.json"
            payload = json.loads(comparison.read_text(encoding="utf-8"))
            payload["families"]["retrieval"]["paired_metrics"]["hit_at_1"][
                "paired_bootstrap_95_ci"
            ] = [-0.01, 0.3]
            comparison.write_text(json.dumps(payload), encoding="utf-8")
            result = self.summarize(paths)
        self.assertFalse(result["selection_supported"])
        self.assertFalse(result["core_claim_supported"])

    def test_wrong_selected_payload_dereference_blocks_core_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result_path = (
                paths["latebound"]
                / "unseen-tool_unseen-address"
                / "registry-1000"
                / "results.json"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["registration_audit"][
                "selected_id_dereferences_full_document"
            ] = False
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            result = self.summarize(paths)
        self.assertFalse(result["strict_post_training_registration"])
        self.assertFalse(result["core_claim_supported"])

    def test_unverified_physical_binding_blocks_core_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result_path = (
                paths["latebound"]
                / "unseen-tool_unseen-address"
                / "registry-1000"
                / "results.json"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["registration_audit"][
                "selected_physical_bindings_verified"
            ] = False
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            result = self.summarize(paths)
        self.assertFalse(result["strict_post_training_registration"])
        self.assertFalse(result["core_claim_supported"])

    def test_inconsistent_binding_counts_block_core_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result_path = (
                paths["latebound"]
                / "unseen-tool_unseen-address"
                / "registry-1000"
                / "results.json"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["registration_audit"][
                "selected_physical_binding_verified_count"
            ] -= 1
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            result = self.summarize(paths)
        self.assertFalse(result["registration_audit"]["binding_evidence"]["counts_valid"])
        self.assertFalse(result["core_claim_supported"])

    def test_smoke_subset_cannot_support_selection_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            latebound = (
                paths["latebound"]
                / "unseen-tool_unseen-address"
                / "registry-1000"
                / "results.json"
            )
            payload = json.loads(latebound.read_text(encoding="utf-8"))
            payload["aggregates"]["retrieval:unseen_tool_unseen_address"][
                "count"
            ] = 4
            latebound.write_text(json.dumps(payload), encoding="utf-8")
            incremental = (
                paths["incremental"]
                / "unseen-tool_unseen-address"
                / "registry-1000"
                / "comparison.json"
            )
            payload = json.loads(incremental.read_text(encoding="utf-8"))
            payload["pair_count"] = 1173
            payload["registry_pairs"] = 1173
            incremental.write_text(json.dumps(payload), encoding="utf-8")
            for control in SELECTION_CONTROLS:
                comparison = paths["controls"] / f"control-{control}" / "comparison.json"
                payload = json.loads(comparison.read_text(encoding="utf-8"))
                payload["families"]["retrieval"]["paired_metrics"]["hit_at_1"][
                    "count"
                ] = 4
                payload["pair_count"] = 1173
                payload["registry_pairs"] = 1173
                comparison.write_text(json.dumps(payload), encoding="utf-8")
            result = self.summarize(paths)
        self.assertEqual(result["registered_absolute_metrics"]["retrieval_count"], 4)
        self.assertFalse(result["selection_supported"])
        self.assertFalse(result["core_claim_supported"])

    def test_rejects_control_built_from_different_registered_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            comparison = paths["controls"] / "control-random" / "comparison.json"
            payload = json.loads(comparison.read_text(encoding="utf-8"))
            payload["sources"]["candidate_sha256"] = "e" * 64
            comparison.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "declared late-bound"):
                self.summarize(paths)

    def test_rejects_a_registry_cell_using_a_different_common_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            candidate = (
                paths["latebound"]
                / "seen-tool_seen-address"
                / "registry-10"
                / "results.json"
            )
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            payload["common_document_agent"]["audit_sha256"] = "9" * 64
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different common-document Agent"):
                self.summarize(paths)

    def test_rejects_fixed_baseline_result_using_another_model(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result_path = (
                paths["fixed"]
                / "seen-tool_seen-address"
                / "registry-10"
                / "results.json"
            )
            payload = json.loads(result_path.read_text())
            payload["model_path"] = "/models/different-fixed"
            result_path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "another selection model"):
                self.summarize(paths)

    def test_rejects_tampered_incremental_registration_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            runtime = paths["incremental-train"] / "registration_runtime.json"
            payload = json.loads(runtime.read_text())
            payload["wall_seconds"] = 1
            runtime.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.summarize(paths)

    def test_rejects_incremental_manifest_with_evaluation_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            root = paths["incremental-train"]
            manifest = root / "input-data" / "incremental_manifest.json"
            payload = json.loads(manifest.read_text())
            payload["audits"]["evaluation_query_records_used"] = 1
            manifest.write_text(json.dumps(payload) + "\n")
            inputs = [
                root / "input-data" / name
                for name in (
                    "incremental_manifest.json",
                    "fixed_incremental_memorization.json",
                    "virtual_tokens_incremental.txt",
                    "virtual_tokens_combined.txt",
                )
            ]
            (root / "input.sha256").write_text(
                "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.resolve()}\n"
                    for path in inputs
                )
            )
            with self.assertRaisesRegex(ValueError, "document-only isolation"):
                self.summarize(paths)

    def test_rejects_tampered_common_agent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            audit = Path(directory) / "shared-agent" / "checkpoint_audit.json"
            payload = json.loads(audit.read_text())
            payload["full_model_reloaded"] = False
            audit.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "audit digest mismatch"):
                self.summarize(paths)

    def test_rejects_fixed_checkpoint_without_reference_mapping_match(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            root = paths["fixed-train"]
            stage3 = root / "checkpoint_audit_stage3.json"
            payload = json.loads(stage3.read_text())
            payload["reference_mapping_match"] = False
            stage3.write_text(json.dumps(payload) + "\n")
            audits = [root / f"checkpoint_audit_stage{stage}.json" for stage in (1, 2, 3)]
            (root / "checkpoint_audits.sha256").write_text(
                "".join(
                    f"{hashlib.sha256(audit.read_bytes()).hexdigest()}  {audit.resolve()}\n"
                    for audit in audits
                )
            )
            with self.assertRaisesRegex(ValueError, "reference token mapping"):
                self.summarize(paths)

    def test_rejects_mislabeled_control_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result_path = paths["controls"] / "control-random" / "results.json"
            payload = json.loads(result_path.read_text())
            payload["registration_control"] = "blank"
            result_path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "another registration control"):
                self.summarize(paths)

    def test_rejects_candidate_from_another_late_bound_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result_path = (
                paths["latebound"]
                / "seen-tool_seen-address"
                / "registry-10"
                / "results.json"
            )
            payload = json.loads(result_path.read_text())
            payload["checkpoint_hashes"]["memory_compiler"] = "9" * 64
            result_path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "another late-bound checkpoint"):
                self.summarize(paths)

    def test_rejects_incomplete_runtime_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_fixture(Path(directory))
            result_path = (
                paths["latebound"]
                / "seen-tool_seen-address"
                / "registry-10"
                / "results.json"
            )
            payload = json.loads(result_path.read_text())
            payload["registration_audit"]["bytes_stored_per_tool"] = -1
            result_path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(ValueError, "runtime or storage reporting"):
                self.summarize(paths)


if __name__ == "__main__":
    unittest.main()
