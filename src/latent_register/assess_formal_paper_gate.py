"""Apply the predeclared paper-level gate to a three-seed aggregate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .aggregate_formal_pipeline import (
    ADDRESS_STATUSES,
    CONTROLS,
    EXPECTED_CELL_COUNT,
    MATCHED_REGISTRY_SIZES,
    SCALE_REGISTRY_SIZES,
)
from .aggregate_formal_seeds import FORMAL_SEEDS
from .physical_tokens import validate_token_identity_audit
from .validate_toolgen_tokenizer import sha256_file, write_json_atomic


OUTPUT_CONTROLS = tuple(control for control in CONTROLS if control != "wrong_memory")
MIN_RETRIEVAL_EXAMPLES = 2000
LATEBOUND_CHECKPOINT_HASH_KEYS = {
    "retrieval_adapter_config",
    "retrieval_adapter_weights",
    "retrieval_compiler",
    "memory_compiler",
    "training_results",
}


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value}")
    return result


def _metric(
    aggregate: Mapping[str, Any], cell: str, family: str, metric: str
) -> Mapping[str, Any]:
    try:
        value = aggregate["cells"][cell]["families"][family]["metrics"][metric]
    except KeyError as exc:
        raise ValueError(f"Missing aggregate metric {cell}:{family}.{metric}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Aggregate metric is not an object: {cell}:{family}.{metric}")
    return value


def _seed_values(summary: Mapping[str, Any], label: str) -> dict[int, float]:
    values = summary.get("values_by_seed", {})
    if set(values) != {str(seed) for seed in FORMAL_SEEDS}:
        raise ValueError(f"Incomplete seed values for {label}")
    return {
        seed: _finite(values[str(seed)], f"{label}.seed-{seed}")
        for seed in FORMAL_SEEDS
    }


def _record(checks: dict[str, Any], name: str, passed: bool, **evidence: Any) -> None:
    checks[name] = {"passed": bool(passed), **evidence}


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_cell_evaluation(
    aggregate: Mapping[str, Any],
    cell_name: str,
    seed: int,
    role: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    source = aggregate["cells"][cell_name]["inputs"][str(seed)]
    predictions_path = Path(str(source[f"{role}_predictions_path"])).resolve()
    declared_sha256 = str(source[f"{role}_predictions_sha256"])
    if not predictions_path.is_file() or sha256_file(predictions_path) != declared_sha256:
        raise ValueError(f"{cell_name} seed {seed} {role} predictions are missing or modified")
    results_path = predictions_path.parent / "results.json"
    if not results_path.is_file() or not (results_path.parent / "COMPLETE").is_file():
        raise FileNotFoundError(f"Incomplete evaluation artifact: {results_path.parent}")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("predictions_sha256") != declared_sha256:
        raise ValueError(f"{cell_name} seed {seed} {role} result/prediction mismatch")
    return result, {
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": declared_sha256,
    }


def _shared_agent_and_checkpoint_audit(
    aggregate: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    specifications = (
        (
            "fixed",
            "fixed_vs_latebound/seen-address/registry-1000",
            "baseline",
            "common_document",
            False,
            "qwen_toolgen_fixed",
            ("train",),
        ),
        (
            "latebound_seen",
            "fixed_vs_latebound/seen-address/registry-1000",
            "candidate",
            "common_document",
            True,
            "qwen_late_bound",
            None,
        ),
        (
            "incremental",
            "incremental_vs_latebound/unseen-address/registry-1000",
            "baseline",
            "common_document",
            False,
            "qwen_toolgen_fixed",
            ("test",),
        ),
        (
            "latebound_incremental",
            "incremental_vs_latebound/unseen-address/registry-1000",
            "candidate",
            "common_document",
            True,
            "qwen_late_bound",
            None,
        ),
        (
            "oracle",
            "oracle_vs_latebound/unseen-address/registry-1000",
            "baseline",
            "full_document_oracle",
            False,
            "qwen_full_document",
            None,
        ),
        (
            "latebound_unseen",
            "oracle_vs_latebound/unseen-address/registry-1000",
            "candidate",
            "common_document",
            True,
            "qwen_late_bound",
            None,
        ),
    )
    systems: dict[str, Any] = {}
    agent_identities: set[tuple[str, str, str]] = set()
    agent_config_hashes: set[str] = set()
    latebound_checkpoint_hashes: set[tuple[tuple[str, str], ...]] = set()
    model_paths: dict[str, str] = {}
    for (
        name,
        cell_name,
        role,
        information_condition,
        is_latebound,
        expected_kind,
        expected_splits,
    ) in specifications:
        result, sources = _load_cell_evaluation(
            aggregate, cell_name, seed, role
        )
        common_agent = result.get("common_document_agent")
        if not isinstance(common_agent, dict):
            raise ValueError(f"{name} seed {seed} lacks common Agent evidence")
        identity = tuple(
            str(common_agent.get(field, ""))
            for field in ("model_path", "audit_path", "audit_sha256")
        )
        agent_identities.add(identity)
        config_sha256 = str(result.get("common_document_agent_config_sha256", ""))
        agent_config_hashes.add(config_sha256)
        checkpoint_hashes = result.get("checkpoint_hashes")
        model_paths[name] = str(result.get("model_path", ""))
        checkpoint_valid = True
        if is_latebound:
            checkpoint_valid = (
                isinstance(checkpoint_hashes, dict)
                and set(checkpoint_hashes) == LATEBOUND_CHECKPOINT_HASH_KEYS
                and all(_hex_digest(value) for value in checkpoint_hashes.values())
            )
            if checkpoint_valid:
                latebound_checkpoint_hashes.add(
                    tuple(sorted((str(key), str(value)) for key, value in checkpoint_hashes.items()))
                )
        systems[name] = {
            "information_condition": result.get("information_condition"),
            "information_condition_valid": (
                result.get("information_condition") == information_condition
            ),
            "kind": result.get("kind"),
            "kind_valid": result.get("kind") == expected_kind,
            "candidate_splits": result.get("candidate_splits"),
            "candidate_splits_valid": (
                expected_splits is None
                or tuple(result.get("candidate_splits", ())) == expected_splits
            ),
            "model_path": result.get("model_path"),
            "common_document_agent": common_agent,
            "common_document_agent_config_sha256": config_sha256,
            "common_document_agent_config_sha256_valid": _hex_digest(config_sha256),
            "latebound_checkpoint_hashes": checkpoint_hashes if is_latebound else None,
            "latebound_checkpoint_hashes_valid": checkpoint_valid,
            "sources": sources,
        }

    model_path_text, audit_path_text, declared_audit_sha256 = next(iter(agent_identities))
    model_path = Path(model_path_text)
    audit_path = Path(audit_path_text)
    live_checks = {
        "one_agent_identity": len(agent_identities) == 1,
        "one_agent_config_hash": len(agent_config_hashes) == 1,
        "agent_config_hash_valid": (
            len(agent_config_hashes) == 1 and _hex_digest(next(iter(agent_config_hashes)))
        ),
        "agent_model_config_exists": (model_path / "config.json").is_file(),
        "agent_audit_complete": (
            audit_path.is_file() and (audit_path.parent / "COMPLETE").is_file()
        ),
        "agent_audit_digest_matches": (
            audit_path.is_file()
            and _hex_digest(declared_audit_sha256)
            and sha256_file(audit_path) == declared_audit_sha256
        ),
        "one_latebound_checkpoint": len(latebound_checkpoint_hashes) == 1,
        "all_information_conditions_valid": all(
            value["information_condition_valid"] for value in systems.values()
        ),
        "all_system_kinds_valid": all(
            value["kind_valid"] for value in systems.values()
        ),
        "all_candidate_splits_valid": all(
            value["candidate_splits_valid"] for value in systems.values()
        ),
        "fixed_model_path_present": bool(model_paths["fixed"]),
        "incremental_model_path_present": bool(model_paths["incremental"]),
        "fixed_and_incremental_models_distinct": (
            Path(model_paths["fixed"]).resolve()
            != Path(model_paths["incremental"]).resolve()
        ),
        "oracle_is_common_agent_model": (
            Path(model_paths["oracle"]).resolve() == model_path.resolve()
        ),
        "all_agent_config_hashes_valid": all(
            value["common_document_agent_config_sha256_valid"]
            for value in systems.values()
        ),
        "all_latebound_checkpoint_hashes_valid": all(
            value["latebound_checkpoint_hashes_valid"] for value in systems.values()
        ),
    }
    audit_payload: dict[str, Any] = {}
    if live_checks["agent_audit_complete"]:
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    live_checks.update(
        {
            "agent_audit_kind": (
                audit_payload.get("kind") == "qwen_full_document_checkpoint_audit"
            ),
            "agent_audit_model_path": (
                Path(str(audit_payload.get("model_path", ""))).resolve()
                == model_path.resolve()
            ),
            "agent_full_model_reloaded": audit_payload.get("full_model_reloaded") is True,
            "agent_embeddings_finite": (
                audit_payload.get("embedding_tables_all_finite") is True
            ),
            "agent_config_digest_matches": (
                (model_path / "config.json").is_file()
                and len(agent_config_hashes) == 1
                and sha256_file(model_path / "config.json")
                == next(iter(agent_config_hashes))
            ),
        }
    )
    return {
        "passed": all(live_checks.values()),
        "checks": live_checks,
        "systems": systems,
    }


def _control_treatment_audit(
    aggregate: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    canonical_cell = "oracle_vs_latebound/unseen-address/registry-1000"
    canonical, canonical_source = _load_cell_evaluation(
        aggregate, canonical_cell, seed, "candidate"
    )
    canonical_agent = canonical.get("common_document_agent")
    canonical_checkpoint = canonical.get("checkpoint_hashes")
    canonical_config = canonical.get("common_document_agent_config_sha256")
    controls: dict[str, Any] = {}
    for control in CONTROLS:
        cell_name = f"registered_vs_control/{control}"
        baseline, baseline_source = _load_cell_evaluation(
            aggregate, cell_name, seed, "baseline"
        )
        candidate, candidate_source = _load_cell_evaluation(
            aggregate, cell_name, seed, "candidate"
        )
        checks = {
            "control_label": baseline.get("registration_control") == control,
            "control_kind": baseline.get("kind") == "qwen_late_bound",
            "control_source_condition": (
                baseline.get("source_condition") == "unseen_tool_unseen_token"
            ),
            "control_address_status": baseline.get("address_status") == "unseen",
            "control_registry_size": int(baseline.get("registry_size", -1)) == 1000,
            "control_information_condition": (
                baseline.get("information_condition") == "common_document"
            ),
            "control_common_agent": (
                baseline.get("common_document_agent") == canonical_agent
            ),
            "control_agent_config": (
                baseline.get("common_document_agent_config_sha256")
                == canonical_config
            ),
            "control_checkpoint": (
                baseline.get("checkpoint_hashes") == canonical_checkpoint
            ),
            "candidate_is_registered": (
                candidate.get("registration_control") == "registered"
            ),
            "candidate_is_canonical_path": (
                candidate_source["predictions_path"]
                == canonical_source["predictions_path"]
            ),
            "candidate_is_canonical_digest": (
                candidate_source["predictions_sha256"]
                == canonical_source["predictions_sha256"]
            ),
            "candidate_common_agent": (
                candidate.get("common_document_agent") == canonical_agent
            ),
            "candidate_checkpoint": (
                candidate.get("checkpoint_hashes") == canonical_checkpoint
            ),
        }
        controls[control] = {
            "passed": all(checks.values()),
            "checks": checks,
            "baseline_sources": baseline_source,
            "candidate_sources": candidate_source,
        }
    return {
        "passed": all(value["passed"] for value in controls.values()),
        "canonical_candidate": canonical_source,
        "controls": controls,
    }


def _runtime_and_storage_audit(
    result: Mapping[str, Any], expected_registry_size: int
) -> dict[str, Any]:
    registration = result.get("registration_audit", {})
    latency = result.get("latency", {})
    retrieval = latency.get("retrieval_selection", {})
    arguments = latency.get("argument_path", {})
    values = {
        "registration_wall_seconds": _finite(
            registration.get("wall_seconds"), "registration.wall_seconds"
        ),
        "registration_peak_memory_bytes": int(
            registration.get("peak_memory_bytes", -1)
        ),
        "bytes_stored_per_tool": int(registration.get("bytes_stored_per_tool", -1)),
        "retrieval_count": int(retrieval.get("count", -1)),
        "retrieval_total_seconds": _finite(
            retrieval.get("total_seconds"), "latency.retrieval.total_seconds"
        ),
        "retrieval_mean_seconds_per_example": _finite(
            retrieval.get("mean_seconds_per_example"),
            "latency.retrieval.mean_seconds_per_example",
        ),
        "argument_count": int(arguments.get("count", -1)),
        "argument_total_seconds": _finite(
            arguments.get("total_seconds"), "latency.arguments.total_seconds"
        ),
        "argument_mean_seconds_per_example": _finite(
            arguments.get("mean_seconds_per_example"),
            "latency.arguments.mean_seconds_per_example",
        ),
        "inference_seconds": _finite(
            result.get("inference_seconds"), "inference_seconds"
        ),
        "ordinary_token_win_rate": _finite(
            result.get("ordinary_token_win_rate"), "ordinary_token_win_rate"
        ),
    }
    checks = {
        "registry_size_matches": int(result.get("registry_size", -1))
        == expected_registry_size,
        "distinct_registered_tools_match": int(
            registration.get("distinct_registered_tools", -1)
        )
        == expected_registry_size,
        "document_forwards_positive": int(
            registration.get("document_encoder_batch_forwards", -1)
        )
        > 0,
        "registration_wall_seconds_nonnegative": values[
            "registration_wall_seconds"
        ]
        >= 0.0,
        "registration_peak_memory_nonnegative": values[
            "registration_peak_memory_bytes"
        ]
        >= 0,
        "bytes_stored_per_tool_positive": values["bytes_stored_per_tool"] > 0,
        "model_loading_excluded": latency.get("model_loading_excluded") is True,
        "retrieval_count_positive": values["retrieval_count"] > 0,
        "retrieval_latency_nonnegative": (
            values["retrieval_total_seconds"] >= 0.0
            and values["retrieval_mean_seconds_per_example"] >= 0.0
        ),
        "argument_count_positive": values["argument_count"] > 0,
        "argument_latency_nonnegative": (
            values["argument_total_seconds"] >= 0.0
            and values["argument_mean_seconds_per_example"] >= 0.0
        ),
        "argument_latency_includes_selection": (
            arguments.get("includes_tool_selection") is True
        ),
        "argument_latency_includes_dereference_and_generation": (
            arguments.get("includes_document_dereference_and_generation") is True
        ),
        "inference_seconds_nonnegative": values["inference_seconds"] >= 0.0,
        "ordinary_token_win_rate_bounded": (
            0.0 <= values["ordinary_token_win_rate"] <= 1.0
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "values": values}


def _latebound_quadrant_audit(
    aggregate: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    canonical, _ = _load_cell_evaluation(
        aggregate,
        "oracle_vs_latebound/unseen-address/registry-1000",
        seed,
        "candidate",
    )
    canonical_agent = canonical.get("common_document_agent")
    canonical_checkpoint = canonical.get("checkpoint_hashes")
    specifications: list[tuple[str, str, str, int]] = []
    for system, tool_status in (
        ("fixed_vs_latebound", "seen"),
        ("oracle_vs_latebound", "unseen"),
    ):
        for address in ADDRESS_STATUSES:
            for size in MATCHED_REGISTRY_SIZES:
                specifications.append((system, tool_status, address, size))
        for size in SCALE_REGISTRY_SIZES:
            specifications.append((system, tool_status, "unseen", size))
    for size in MATCHED_REGISTRY_SIZES:
        specifications.append(
            ("incremental_vs_latebound", "unseen", "unseen", size)
        )

    cells: dict[str, Any] = {}
    for system, tool_status, address, size in specifications:
        cell_name = f"{system}/{address}-address/registry-{size}"
        candidate, sources = _load_cell_evaluation(
            aggregate, cell_name, seed, "candidate"
        )
        expected_condition = f"{tool_status}_tool_{address}_address"
        expected_source = (
            "seen_tool_seen_token"
            if tool_status == "seen"
            else "unseen_tool_unseen_token"
        )
        checks = {
            "kind": candidate.get("kind") == "qwen_late_bound",
            "condition": candidate.get("condition") == expected_condition,
            "source_condition": candidate.get("source_condition") == expected_source,
            "address_status": candidate.get("address_status") == address,
            "registry_size": int(candidate.get("registry_size", -1)) == size,
            "registration_control": (
                candidate.get("registration_control") == "registered"
            ),
            "information_condition": (
                candidate.get("information_condition") == "common_document"
            ),
            "common_agent": (
                candidate.get("common_document_agent") == canonical_agent
            ),
            "latebound_checkpoint": (
                candidate.get("checkpoint_hashes") == canonical_checkpoint
            ),
        }
        cells[cell_name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "sources": sources,
        }
    return {
        "passed": all(value["passed"] for value in cells.values()),
        "cell_count": len(cells),
        "cells": cells,
    }


def _append_identity_audit(
    aggregate: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    cell_name = "append/registry-100-to-1000"
    initial, initial_source = _load_cell_evaluation(
        aggregate, cell_name, seed, "baseline"
    )
    extended, extended_source = _load_cell_evaluation(
        aggregate, cell_name, seed, "candidate"
    )
    initial_binding_path = Path(str(initial.get("input_registry_binding", "")))
    extended_binding_path = Path(str(extended.get("input_registry_binding", "")))
    initial_declared_hash = str(initial.get("input_registry_binding_sha256", ""))
    extended_declared_hash = str(extended.get("input_registry_binding_sha256", ""))
    bindings_exist = initial_binding_path.is_file() and extended_binding_path.is_file()
    initial_binding: dict[str, Any] = {}
    extended_binding: dict[str, Any] = {}
    if bindings_exist:
        initial_binding = json.loads(initial_binding_path.read_text(encoding="utf-8"))
        extended_binding = json.loads(extended_binding_path.read_text(encoding="utf-8"))
    initial_size = int(initial_binding.get("registry_size", -1))
    extended_size = int(extended_binding.get("registry_size", -1))
    initial_identities = list(initial_binding.get("identities", []))
    extended_identities = list(extended_binding.get("identities", []))
    initial_slots = list(initial_binding.get("address_slots", []))
    extended_slots = list(extended_binding.get("address_slots", []))
    checks = {
        "bindings_exist": bindings_exist,
        "initial_binding_digest": (
            initial_binding_path.is_file()
            and _hex_digest(initial_declared_hash)
            and sha256_file(initial_binding_path) == initial_declared_hash
        ),
        "extended_binding_digest": (
            extended_binding_path.is_file()
            and _hex_digest(extended_declared_hash)
            and sha256_file(extended_binding_path) == extended_declared_hash
        ),
        "binding_sizes": initial_size == 100 and extended_size == 1000,
        "identity_lengths": (
            len(initial_identities) == initial_size
            and len(extended_identities) == extended_size
        ),
        "address_slot_lengths": (
            len(initial_slots) == initial_size and len(extended_slots) == extended_size
        ),
        "identity_prefix_preserved": (
            extended_identities[:initial_size] == initial_identities
        ),
        "address_prefix_preserved": extended_slots[:initial_size] == initial_slots,
        "extended_identities_unique": len(set(extended_identities)) == extended_size,
        "extended_slots_unique": len(set(extended_slots)) == extended_size,
        "result_registry_sizes": (
            int(initial.get("registry_size", -1)) == initial_size
            and int(extended.get("registry_size", -1)) == extended_size
        ),
        "prediction_counts": (
            int(initial.get("prediction_count", -1)) == 64
            and int(extended.get("prediction_count", -1)) == 64
        ),
        "same_latebound_checkpoint": (
            initial.get("checkpoint_hashes") == extended.get("checkpoint_hashes")
            and isinstance(initial.get("checkpoint_hashes"), dict)
        ),
        "registered_treatment": (
            initial.get("registration_control") == "registered"
            and extended.get("registration_control") == "registered"
        ),
        "unseen_address": (
            initial.get("address_status") == "unseen"
            and extended.get("address_status") == "unseen"
        ),
        "paired_retrieval_count": int(
            _metric(aggregate, cell_name, "retrieval", "hit_at_1").get(
                "count_per_seed", -1
            )
        )
        == 64,
        "paired_argument_count": int(
            _metric(
                aggregate, cell_name, "arguments", "end_to_end_argument_exact"
            ).get("count_per_seed", -1)
        )
        == 64,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "initial_sources": initial_source,
        "extended_sources": extended_source,
        "initial_binding_path": str(initial_binding_path.resolve()),
        "extended_binding_path": str(extended_binding_path.resolve()),
    }


def _significance_evidence(
    metric: Mapping[str, Any],
    label: str,
    *,
    minimum_count_per_seed: int = 1,
) -> dict[str, Any]:
    count_per_seed = int(metric.get("count_per_seed", -1))
    delta = metric["delta"]
    mean = _finite(delta["mean"], f"{label}.delta.mean")
    seed_ci = [
        _finite(value, f"{label}.delta.seed_t_95_ci")
        for value in delta["seed_t_95_ci"]
    ]
    bootstrap = metric["paired_bootstrap_95_ci_by_seed"]
    bootstrap_lower = {
        str(seed): _finite(
            bootstrap[str(seed)][0], f"{label}.seed-{seed}.bootstrap_lower"
        )
        for seed in FORMAL_SEEDS
    }
    return {
        "passed": count_per_seed >= minimum_count_per_seed
        and mean > 0.0
        and seed_ci[0] > 0.0
        and all(value > 0.0 for value in bootstrap_lower.values()),
        "count_per_seed": count_per_seed,
        "minimum_count_per_seed": minimum_count_per_seed,
        "count_requirement_passed": count_per_seed >= minimum_count_per_seed,
        "delta_mean": mean,
        "seed_t_95_ci": seed_ci,
        "paired_bootstrap_lower_by_seed": bootstrap_lower,
    }


def _validate_complete_aggregate(path: Path) -> dict[str, Any]:
    complete = path.parent / "COMPLETE"
    if not complete.is_file():
        raise FileNotFoundError(f"Formal aggregate is incomplete: {path.parent}")
    fields = complete.read_text(encoding="utf-8").split()
    if not fields or fields[0] != sha256_file(path):
        raise ValueError("Formal aggregate COMPLETE digest does not match aggregate.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "formal_three_seed_pipeline_aggregate":
        raise ValueError("Unexpected formal aggregate kind")
    if payload.get("seeds") != list(FORMAL_SEEDS):
        raise ValueError("Formal aggregate seeds differ from the predeclared seeds")
    if int(payload.get("cell_count", -1)) != EXPECTED_CELL_COUNT:
        raise ValueError("Formal aggregate does not contain all comparison cells")
    if len(payload.get("cells", {})) != EXPECTED_CELL_COUNT:
        raise ValueError("Formal aggregate cell payload count is inconsistent")
    if payload.get("all_cells_complete") is not True:
        raise ValueError("Formal aggregate reports incomplete cells")
    return payload


def _audit_formal_seed(cell: Mapping[str, Any], seed: int) -> dict[str, Any]:
    source = cell["inputs"][str(seed)]
    predictions_path = Path(str(source["candidate_predictions_path"]))
    results_path = predictions_path.parent / "results.json"
    if not (predictions_path.parent / "COMPLETE").is_file():
        raise FileNotFoundError(f"Formal evaluation is incomplete: {predictions_path.parent}")
    result = json.loads(results_path.read_text(encoding="utf-8"))
    if result.get("predictions_sha256") != source["candidate_predictions_sha256"]:
        raise ValueError(f"Seed {seed} evaluation/result prediction hash mismatch")
    if result.get("source_condition") != "unseen_tool_unseen_token":
        raise ValueError(f"Seed {seed} audit is not an unseen-tool evaluation")
    if result.get("address_status") != "unseen" or result.get("registry_size") != 1000:
        raise ValueError(f"Seed {seed} audit is not the unseen-address 1K cell")

    provenance = result["training_provenance"]
    config = provenance.get("config", {})
    training_results = Path(str(provenance["results_path"]))
    final_audit_path = training_results.parent / "training_audit.json"
    retrieval_audit_path = (
        training_results.parent.parent / "retrieval" / "training_audit.json"
    )
    final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
    retrieval_audit = json.loads(retrieval_audit_path.read_text(encoding="utf-8"))

    registration = result["registration_audit"]
    mutation = result["evaluation_mutation_audit"]
    physical = result["physical_token_audit"]
    physical_identity = validate_token_identity_audit(physical)
    code_hashes = result.get("code_hashes", {})
    expected_code_keys = {
        "evaluate_late_bound",
        "benchmark_metrics",
        "model",
        "physical_tokens",
    }
    code_hashes_valid = set(code_hashes) == expected_code_keys and all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in code_hashes.values()
    )
    zero_values = {
        "optimizer_steps": registration.get("optimizer_steps"),
        "model_or_table_parameters_changed": registration.get(
            "model_or_table_parameters_changed"
        ),
        "all_reserved_input_row_max_change": mutation.get(
            "all_reserved_input_row_max_change"
        ),
        "all_reserved_output_row_max_change": mutation.get(
            "all_reserved_output_row_max_change"
        ),
        "evaluation_ids_seen_during_training": physical.get(
            "evaluation_ids_seen_during_training"
        ),
        "evaluation_input_row_max_change": physical.get(
            "evaluation_input_row_max_change"
        ),
        "evaluation_output_row_max_change": physical.get(
            "evaluation_output_row_max_change"
        ),
    }
    selection_contract = {
        "selection_emits_single_physical_id": registration.get(
            "selection_emits_single_physical_id"
        ),
        "selected_physical_bindings_verified": registration.get(
            "selected_physical_bindings_verified"
        ),
        "selected_id_dereferences_registered_memory": registration.get(
            "selected_id_dereferences_registered_memory"
        ),
        "selected_id_dereferences_full_document": registration.get(
            "selected_id_dereferences_full_document"
        ),
        "selected_id_uses_static_embedding": registration.get(
            "selected_id_uses_static_embedding"
        ),
        "ordinary_vocabulary_competes_at_selection": registration.get(
            "ordinary_vocabulary_competes_at_selection"
        ),
        "inactive_reserved_ids_masked": registration.get(
            "inactive_reserved_ids_masked"
        ),
    }
    selection_contract_passed = selection_contract == {
        "selection_emits_single_physical_id": True,
        "selected_physical_bindings_verified": True,
        "selected_id_dereferences_registered_memory": False,
        "selected_id_dereferences_full_document": True,
        "selected_id_uses_static_embedding": False,
        "ordinary_vocabulary_competes_at_selection": True,
        "inactive_reserved_ids_masked": True,
    }
    binding_evidence = {
        "selected_physical_binding_count": int(
            registration.get("selected_physical_binding_count", -1)
        ),
        "selected_physical_binding_verified_count": int(
            registration.get("selected_physical_binding_verified_count", -1)
        ),
        "selected_argument_payload_count": int(
            registration.get("selected_argument_payload_count", -1)
        ),
        "selected_argument_payload_verified_count": int(
            registration.get("selected_argument_payload_verified_count", -1)
        ),
    }
    binding_counts_valid = (
        binding_evidence["selected_physical_binding_count"] > 0
        and binding_evidence["selected_physical_binding_verified_count"]
        == binding_evidence["selected_physical_binding_count"]
        and binding_evidence["selected_argument_payload_count"] > 0
        and binding_evidence["selected_argument_payload_verified_count"]
        == binding_evidence["selected_argument_payload_count"]
    )
    runtime_and_storage = _runtime_and_storage_audit(result, 1000)
    training_checks = {
        "final_seed": int(final_audit.get("seed", -1)) == seed,
        "final_steps": int(final_audit.get("completed_steps", -1)) == 3000,
        "final_world_size": int(final_audit.get("world_size", -1)) == 6,
        "final_stage": final_audit.get("stage") == "full_vocabulary_joint",
        "final_isolation": final_audit.get("physical_token_isolation") is True,
        "retrieval_seed": int(retrieval_audit.get("seed", -1)) == seed,
        "retrieval_steps": int(retrieval_audit.get("completed_steps", -1)) == 1000,
        "retrieval_world_size": int(retrieval_audit.get("world_size", -1)) == 6,
        "retrieval_stage": retrieval_audit.get("stage") == "retrieval",
        "retrieval_isolation": retrieval_audit.get("physical_token_isolation") is True,
        "evaluation_seed": int(config.get("seed", -1)) == seed,
        "evaluation_completed_steps": int(provenance.get("completed_steps", -1)) == 3000,
        "evaluation_world_size": int(provenance.get("world_size", -1)) == 6,
        "common_document_information_condition": (
            result.get("information_condition") == "common_document"
        ),
        "common_document_agent_audited": (
            isinstance(result.get("common_document_agent"), dict)
            and len(
                str(result.get("common_document_agent", {}).get("audit_sha256", ""))
            )
            == 64
        ),
    }
    zero_passed = all(
        _finite(value, f"seed-{seed}.{name}") == 0.0
        for name, value in zero_values.items()
    )
    passed = (
        zero_passed
        and code_hashes_valid
        and bool(physical_identity["passed"])
        and selection_contract_passed
        and binding_counts_valid
        and runtime_and_storage["passed"]
        and int(registration.get("registration_forwards_per_tool", -1)) == 1
        and all(training_checks.values())
    )
    return {
        "passed": passed,
        "zero_values": zero_values,
        "registration_forwards_per_tool": registration.get(
            "registration_forwards_per_tool"
        ),
        "training_checks": training_checks,
        "code_hashes": code_hashes,
        "code_hashes_valid": code_hashes_valid,
        "physical_token_identity": physical_identity,
        "selection_contract": selection_contract,
        "selection_contract_passed": selection_contract_passed,
        "binding_evidence": binding_evidence,
        "binding_counts_valid": binding_counts_valid,
        "runtime_and_storage": runtime_and_storage,
        "sources": {
            "evaluation_results_path": str(results_path.resolve()),
            "evaluation_results_sha256": sha256_file(results_path),
            "final_training_audit_path": str(final_audit_path.resolve()),
            "final_training_audit_sha256": sha256_file(final_audit_path),
            "retrieval_training_audit_path": str(retrieval_audit_path.resolve()),
            "retrieval_training_audit_sha256": sha256_file(retrieval_audit_path),
        },
    }


def assess_formal_paper_gate(aggregate_path: Path) -> dict[str, Any]:
    aggregate = _validate_complete_aggregate(aggregate_path)
    checks: dict[str, Any] = {}
    measurements: dict[str, Any] = {}

    comparison_code_hashes = {
        name: value.get("code_hashes", {})
        for name, value in aggregate["cells"].items()
    }
    expected_comparison_code_keys = {
        "compare_controlled_predictions_code_sha256",
        "benchmark_metrics_code_sha256",
    }
    code_hashes_valid = all(
        set(value) == expected_comparison_code_keys
        and all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in value.values()
        )
        for value in comparison_code_hashes.values()
    )
    code_hashes_match = code_hashes_valid and len(
        {
            tuple(sorted(value.items()))
            for value in comparison_code_hashes.values()
        }
    ) == 1
    _record(
        checks,
        "comparison_code_consistency",
        code_hashes_match,
        code_hashes=comparison_code_hashes,
    )

    shared_agent_audits = {
        str(seed): _shared_agent_and_checkpoint_audit(aggregate, seed)
        for seed in FORMAL_SEEDS
    }
    _record(
        checks,
        "shared_agent_and_latebound_checkpoint_identity",
        all(value["passed"] for value in shared_agent_audits.values()),
        seeds=shared_agent_audits,
    )
    control_treatment_audits = {
        str(seed): _control_treatment_audit(aggregate, seed)
        for seed in FORMAL_SEEDS
    }
    _record(
        checks,
        "control_treatment_identity",
        all(value["passed"] for value in control_treatment_audits.values()),
        seeds=control_treatment_audits,
    )
    quadrant_audits = {
        str(seed): _latebound_quadrant_audit(aggregate, seed)
        for seed in FORMAL_SEEDS
    }
    _record(
        checks,
        "latebound_quadrant_identity",
        all(value["passed"] for value in quadrant_audits.values()),
        seeds=quadrant_audits,
    )
    append_audits = {
        str(seed): _append_identity_audit(aggregate, seed)
        for seed in FORMAL_SEEDS
    }
    _record(
        checks,
        "append_identity_prefix",
        all(value["passed"] for value in append_audits.values()),
        seeds=append_audits,
    )

    for size in MATCHED_REGISTRY_SIZES:
        cell_name = f"fixed_vs_latebound/seen-address/registry-{size}"
        metric = _metric(
            aggregate, cell_name, "arguments", "end_to_end_argument_exact"
        )
        deltas = _seed_values(metric["delta"], f"{cell_name}.delta")
        mean_delta = _finite(metric["delta"]["mean"], f"{cell_name}.delta.mean")
        measurements[f"closed_set_registry_{size}"] = {
            "delta_mean": mean_delta,
            "delta_by_seed": {str(seed): value for seed, value in deltas.items()},
        }

    for size in (*MATCHED_REGISTRY_SIZES, *SCALE_REGISTRY_SIZES):
        cell_name = f"oracle_vs_latebound/unseen-address/registry-{size}"
        metric = _metric(
            aggregate, cell_name, "arguments", "end_to_end_argument_exact"
        )
        baseline = _seed_values(metric["baseline"], f"{cell_name}.baseline")
        candidate = _seed_values(metric["candidate"], f"{cell_name}.candidate")
        ratios = {
            seed: candidate[seed] / baseline[seed] if baseline[seed] > 0.0 else 0.0
            for seed in FORMAL_SEEDS
        }
        baseline_mean = _finite(metric["baseline"]["mean"], f"{cell_name}.baseline.mean")
        candidate_mean = _finite(metric["candidate"]["mean"], f"{cell_name}.candidate.mean")
        mean_ratio = candidate_mean / baseline_mean if baseline_mean > 0.0 else 0.0
        measurements[f"oracle_retention_registry_{size}"] = {
            "ratio_of_seed_means": mean_ratio,
            "ratio_by_seed": {str(seed): value for seed, value in ratios.items()},
        }

    for size in MATCHED_REGISTRY_SIZES:
        cell_name = f"incremental_vs_latebound/unseen-address/registry-{size}"
        report: dict[str, Any] = {}
        for family, metric_name in (
            ("retrieval", "hit_at_1"),
            ("arguments", "end_to_end_argument_exact"),
        ):
            metric = _metric(aggregate, cell_name, family, metric_name)
            baseline = _seed_values(metric["baseline"], f"{cell_name}.{family}.baseline")
            candidate = _seed_values(
                metric["candidate"], f"{cell_name}.{family}.candidate"
            )
            deltas = _seed_values(metric["delta"], f"{cell_name}.{family}.delta")
            baseline_mean = _finite(
                metric["baseline"]["mean"], f"{cell_name}.{family}.baseline.mean"
            )
            candidate_mean = _finite(
                metric["candidate"]["mean"], f"{cell_name}.{family}.candidate.mean"
            )
            report[family] = {
                "incremental_mean": baseline_mean,
                "late_bound_mean": candidate_mean,
                "late_bound_minus_incremental_mean": _finite(
                    metric["delta"]["mean"], f"{cell_name}.{family}.delta.mean"
                ),
                "late_bound_minus_incremental_by_seed": {
                    str(seed): value for seed, value in deltas.items()
                },
                "late_bound_over_incremental_ratio_of_seed_means": (
                    candidate_mean / baseline_mean if baseline_mean > 0.0 else 0.0
                ),
                "late_bound_over_incremental_ratio_by_seed": {
                    str(seed): (
                        candidate[seed] / baseline[seed]
                        if baseline[seed] > 0.0
                        else 0.0
                    )
                    for seed in FORMAL_SEEDS
                },
            }
        measurements[f"incremental_comparison_registry_{size}"] = report

    for control in CONTROLS:
        cell_name = f"registered_vs_control/{control}"
        argument = _significance_evidence(
            _metric(
                aggregate,
                cell_name,
                "arguments",
                "end_to_end_argument_exact",
            ),
            f"{control}.end_to_end_argument_exact",
        )
        evidence: dict[str, Any] = {"argument_diagnostic": argument}
        if control in OUTPUT_CONTROLS:
            retrieval = _significance_evidence(
                _metric(aggregate, cell_name, "retrieval", "hit_at_1"),
                f"{control}.hit_at_1",
                minimum_count_per_seed=MIN_RETRIEVAL_EXAMPLES,
            )
            evidence["retrieval"] = retrieval
            _record(
                checks,
                f"selection_control_{control}",
                bool(retrieval["passed"]),
                **evidence,
            )
        else:
            measurements[f"control_{control}"] = evidence

    append_cell = "append/registry-100-to-1000"
    for family, metric_name in (
        ("arguments", "end_to_end_argument_exact"),
        ("retrieval", "hit_at_1"),
    ):
        metric = _metric(aggregate, append_cell, family, metric_name)
        deltas = _seed_values(metric["delta"], f"append.{family}.delta")
        mean_delta = _finite(metric["delta"]["mean"], f"append.{family}.delta.mean")
        measurements[f"append_{family}"] = {
            "delta_mean": mean_delta,
            "delta_by_seed": {str(seed): value for seed, value in deltas.items()},
        }

    audit_cell = aggregate["cells"][
        "oracle_vs_latebound/unseen-address/registry-1000"
    ]
    formal_audits = {
        str(seed): _audit_formal_seed(audit_cell, seed) for seed in FORMAL_SEEDS
    }
    formal_code_hashes_match = len(
        {
            tuple(sorted(value["code_hashes"].items()))
            for value in formal_audits.values()
        }
    ) == 1
    comparison_metric_hash = comparison_code_hashes[
        "oracle_vs_latebound/unseen-address/registry-1000"
    ].get("benchmark_metrics_code_sha256")
    evaluator_metric_hashes = {
        seed: formal_audits[str(seed)]["code_hashes"].get("benchmark_metrics")
        for seed in FORMAL_SEEDS
    }
    evaluator_and_comparison_metrics_match = all(
        value == comparison_metric_hash for value in evaluator_metric_hashes.values()
    )
    _record(
        checks,
        "formal_training_and_registration_audits",
        all(value["passed"] for value in formal_audits.values())
        and formal_code_hashes_match
        and evaluator_and_comparison_metrics_match,
        seeds=formal_audits,
        code_hashes_match=formal_code_hashes_match,
        evaluator_and_comparison_metrics_match=(
            evaluator_and_comparison_metrics_match
        ),
    )

    passed = all(bool(value["passed"]) for value in checks.values())
    return {
        "kind": "formal_three_seed_paper_gate",
        "version": 3,
        "passed": passed,
        "failed_checks": sorted(
            name for name, value in checks.items() if not value["passed"]
        ),
        "checks": checks,
        "measurements": measurements,
        "sources": {
            "aggregate_path": str(aggregate_path.resolve()),
            "aggregate_sha256": sha256_file(aggregate_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the predeclared paper gate to all formal seed results"
    )
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_formal_paper_gate(args.aggregate)
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
