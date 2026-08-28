"""Summarize the predeclared shared-Agent exploratory comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .physical_tokens import validate_token_identity_audit
from .validate_toolgen_tokenizer import sha256_file, write_json_atomic


MATCHED_SIZES = (10, 100, 1000)
SCALE_SIZES = (10000, 47000)
SELECTION_CONTROLS = (
    "blank",
    "random",
    "permuted",
    "shared_vector",
    "query_only",
    "nearest_trained",
)
MIN_RETRIEVAL_EXAMPLES = 2000


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value}")
    return result


def _load_complete(path: Path) -> dict[str, Any]:
    if not path.is_file() or not (path.parent / "COMPLETE").is_file():
        raise FileNotFoundError(f"Incomplete experiment artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _paired_metric(
    comparison: Mapping[str, Any], family: str, metric: str
) -> Mapping[str, Any]:
    try:
        value = comparison["families"][family]["paired_metrics"][metric]
    except KeyError as exc:
        raise ValueError(f"Missing paired metric {family}.{metric}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Paired metric is not an object: {family}.{metric}")
    return value


def _paired_report(metric: Mapping[str, Any], label: str) -> dict[str, Any]:
    baseline = _finite(metric["baseline"], f"{label}.baseline")
    candidate = _finite(metric["candidate"], f"{label}.candidate")
    result = dict(metric)
    result["candidate_minus_baseline"] = candidate - baseline
    result["candidate_over_baseline"] = (
        candidate / baseline if baseline > 0.0 else None
    )
    return result


def _significance(
    metric: Mapping[str, Any],
    label: str,
    *,
    expected_count: int,
    minimum_count: int,
) -> dict[str, Any]:
    delta = _finite(metric["delta"], f"{label}.delta")
    interval = [
        _finite(value, f"{label}.paired_bootstrap_95_ci")
        for value in metric["paired_bootstrap_95_ci"]
    ]
    count = int(metric["count"])
    count_valid = count == expected_count and count >= minimum_count
    return {
        "passed": delta > 0.0 and interval[0] > 0.0 and count_valid,
        "count": count,
        "expected_count": expected_count,
        "minimum_count": minimum_count,
        "count_valid": count_valid,
        "baseline": _finite(metric["baseline"], f"{label}.baseline"),
        "candidate": _finite(metric["candidate"], f"{label}.candidate"),
        "delta": delta,
        "paired_bootstrap_95_ci": interval,
    }


def _comparison_cell(root: Path, suffix: str) -> tuple[dict[str, Any], Path]:
    path = root / suffix / "comparison.json"
    return _load_complete(path), path


def _registration_audit(result: Mapping[str, Any]) -> dict[str, Any]:
    registration = result["registration_audit"]
    mutation = result["evaluation_mutation_audit"]
    physical = result["physical_token_audit"]
    identity = validate_token_identity_audit(physical)
    zero_values = {
        "optimizer_steps": registration.get("optimizer_steps"),
        "model_or_table_parameters_changed": registration.get(
            "model_or_table_parameters_changed"
        ),
        "backbone_parameters_changed": registration.get(
            "backbone_parameters_changed"
        ),
        "embedding_table_parameters_changed": registration.get(
            "embedding_table_parameters_changed"
        ),
        "lm_head_parameters_changed": registration.get(
            "lm_head_parameters_changed"
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
    contract = {
        "registration_forwards_per_tool": registration.get(
            "registration_forwards_per_tool"
        ),
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
    common_agent = result.get("common_document_agent")
    passed = (
        all(_finite(value, name) == 0.0 for name, value in zero_values.items())
        and bool(identity["passed"])
        and binding_counts_valid
        and contract
        == {
            "registration_forwards_per_tool": 1,
            "selection_emits_single_physical_id": True,
            "selected_physical_bindings_verified": True,
            "selected_id_dereferences_registered_memory": False,
            "selected_id_dereferences_full_document": True,
            "selected_id_uses_static_embedding": False,
            "ordinary_vocabulary_competes_at_selection": True,
            "inactive_reserved_ids_masked": True,
        }
        and result.get("information_condition") == "common_document"
        and isinstance(common_agent, dict)
        and len(str(common_agent.get("audit_sha256", ""))) == 64
    )
    return {
        "passed": passed,
        "zero_values": zero_values,
        "physical_token_identity": identity,
        "selection_contract": contract,
        "binding_evidence": {
            **binding_evidence,
            "counts_valid": binding_counts_valid,
        },
        "common_document_agent": common_agent,
    }


def _common_agent_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Shared comparison is missing common Agent evidence")
    model_path = Path(str(value.get("model_path", ""))).resolve()
    audit_path = Path(str(value.get("audit_path", ""))).resolve()
    declared_sha256 = str(value.get("audit_sha256", ""))
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Common Agent model is missing: {model_path}")
    if not audit_path.is_file() or not (audit_path.parent / "COMPLETE").is_file():
        raise FileNotFoundError(f"Common Agent audit is incomplete: {audit_path}")
    observed_sha256 = sha256_file(audit_path)
    if len(declared_sha256) != 64 or observed_sha256 != declared_sha256:
        raise ValueError("Common Agent audit digest mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("kind") != "qwen_full_document_checkpoint_audit"
        or audit.get("full_model_reloaded") is not True
        or audit.get("embedding_tables_all_finite") is not True
        or Path(str(audit.get("model_path", ""))).resolve() != model_path
    ):
        raise ValueError("Common Agent checkpoint audit did not pass full reload")
    return {
        "model_path": str(model_path),
        "audit_path": str(audit_path),
        "audit_sha256": observed_sha256,
        "full_model_reloaded": True,
        "embedding_tables_all_finite": True,
    }


def _runtime_cell(
    result: Mapping[str, Any],
    label: str,
    *,
    expected_condition: str,
    expected_registry_size: int,
) -> dict[str, Any]:
    registration = result.get("registration_audit")
    latency = result.get("latency")
    if not isinstance(registration, dict) or not isinstance(latency, dict):
        raise ValueError(f"{label} is missing registration or latency reporting")
    retrieval = latency.get("retrieval_selection")
    arguments = latency.get("argument_path")
    if not isinstance(retrieval, dict) or not isinstance(arguments, dict):
        raise ValueError(f"{label} has incomplete latency reporting")
    if latency.get("model_loading_excluded") is not True:
        raise ValueError(f"{label} includes model loading in inference latency")
    registry_size = int(result.get("registry_size", -1))
    distinct_tools = int(registration.get("distinct_registered_tools", -1))
    document_forwards = int(registration.get("document_encoder_batch_forwards", -1))
    wall_seconds = _finite(
        registration.get("wall_seconds"), f"{label}.registration_wall_seconds"
    )
    peak_memory = int(registration.get("peak_memory_bytes", -1))
    bytes_per_tool = int(registration.get("bytes_stored_per_tool", -1))
    retrieval_count = int(retrieval.get("count", -1))
    retrieval_total = _finite(
        retrieval.get("total_seconds"), f"{label}.retrieval.total_seconds"
    )
    retrieval_mean = _finite(
        retrieval.get("mean_seconds_per_example"),
        f"{label}.retrieval.mean_seconds_per_example",
    )
    argument_count = int(arguments.get("count", -1))
    argument_total = _finite(
        arguments.get("total_seconds"), f"{label}.arguments.total_seconds"
    )
    argument_mean = _finite(
        arguments.get("mean_seconds_per_example"),
        f"{label}.arguments.mean_seconds_per_example",
    )
    ordinary_token_win_rate = _finite(
        result.get("ordinary_token_win_rate"), f"{label}.ordinary_token_win_rate"
    )
    if (
        result.get("condition") != expected_condition
        or registry_size != expected_registry_size
        or distinct_tools != expected_registry_size
        or document_forwards <= 0
        or wall_seconds < 0.0
        or peak_memory < 0
        or bytes_per_tool <= 0
        or retrieval_count <= 0
        or retrieval_total < 0.0
        or retrieval_mean < 0.0
        or argument_count <= 0
        or argument_total < 0.0
        or argument_mean < 0.0
        or arguments.get("includes_tool_selection") is not True
        or arguments.get("includes_document_dereference_and_generation") is not True
        or not 0.0 <= ordinary_token_win_rate <= 1.0
    ):
        raise ValueError(f"{label} has invalid runtime or storage reporting")
    return {
        "condition": result.get("condition"),
        "registry_size": registry_size,
        "distinct_registered_tools": distinct_tools,
        "document_encoder_batch_forwards": document_forwards,
        "registration_wall_seconds": wall_seconds,
        "registration_peak_memory_bytes": peak_memory,
        "bytes_stored_per_tool": bytes_per_tool,
        "retrieval_selection": {
            "count": retrieval_count,
            "total_seconds": retrieval_total,
            "mean_seconds_per_example": retrieval_mean,
        },
        "argument_path": {
            "count": argument_count,
            "total_seconds": argument_total,
            "mean_seconds_per_example": argument_mean,
            "includes_tool_selection": arguments.get("includes_tool_selection"),
            "includes_document_dereference_and_generation": arguments.get(
                "includes_document_dereference_and_generation"
            ),
        },
        "ordinary_token_win_rate": ordinary_token_win_rate,
    }


def _incremental_registration_cost(root: Path) -> dict[str, Any]:
    root = root.resolve()
    runtime_path = root / "registration_runtime.json"
    checkpoint_path = root / "checkpoint_audit_stage3.json"
    prefix_path = root / "base_token_prefix_audit.json"
    runtime = _load_complete(runtime_path)
    checkpoint = _load_complete(checkpoint_path)
    prefix = _load_complete(prefix_path)

    catalog_path = root / "output.sha256"
    expected = {path.resolve() for path in (runtime_path, checkpoint_path, prefix_path)}
    observed: dict[Path, str] = {}
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("Malformed incremental output checksum catalog")
        path = Path(fields[1].lstrip("* ")).resolve()
        if path in observed:
            raise ValueError(f"Duplicate incremental output checksum: {path}")
        observed[path] = fields[0]
    if set(observed) != expected:
        raise ValueError("Incremental output checksum catalog has another file set")
    for path, digest in observed.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Incremental output checksum mismatch: {path}")

    input_catalog_path = root / "input.sha256"
    input_catalog: dict[Path, str] = {}
    for line in input_catalog_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("Malformed incremental input checksum catalog")
        path = Path(fields[1].lstrip("* ")).resolve()
        if path in input_catalog:
            raise ValueError(f"Duplicate incremental input checksum: {path}")
        input_catalog[path] = fields[0]
    for path, digest in input_catalog.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Incremental input checksum mismatch: {path}")

    required_inputs = {}
    for name in (
        "incremental_manifest.json",
        "fixed_incremental_memorization.json",
        "virtual_tokens_incremental.txt",
        "virtual_tokens_combined.txt",
    ):
        matches = [path for path in input_catalog if path.name == name]
        if len(matches) != 1:
            raise ValueError(f"Incremental input catalog must bind exactly one {name}")
        required_inputs[name] = matches[0]
    manifest_path = required_inputs["incremental_manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    code_catalog_path = root / "code.sha256"
    code_catalog: dict[Path, str] = {}
    for line in code_catalog_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("Malformed incremental code checksum catalog")
        path = Path(fields[1].lstrip("* ")).resolve()
        if path in code_catalog:
            raise ValueError(f"Duplicate incremental code checksum: {path}")
        code_catalog[path] = fields[0]
    expected_code_names = {
        "audit_fixed_checkpoint.py",
        "prepare_incremental_fixed.py",
        "a100_train_controlled_toolgen_incremental.sbatch",
    }
    if {path.name for path in code_catalog} != expected_code_names:
        raise ValueError("Incremental code catalog has another file set")
    for path, digest in code_catalog.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Incremental code checksum mismatch: {path}")

    if runtime.get("kind") != "qwen_toolgen_incremental_registration_runtime":
        raise ValueError("Unexpected incremental registration runtime kind")
    if (
        runtime.get("registration_optimizer_updates_required") is not True
        or runtime.get("model_parameters_updated") is not True
        or int(runtime.get("optimizer_steps", 0)) <= 0
        or int(runtime.get("incremental_tools", 0)) <= 0
    ):
        raise ValueError("Incremental registration did not perform declared updates")
    leakage = {
        field: int(runtime.get(field, -1))
        for field in (
            "training_query_records_used",
            "training_argument_records_used",
            "training_trajectory_records_used",
        )
    }
    if any(value != 0 for value in leakage.values()):
        raise ValueError(f"Incremental registration used evaluation labels: {leakage}")
    if (
        checkpoint.get("kind") != "qwen_toolgen_fixed_checkpoint_audit"
        or checkpoint.get("full_model_reloaded") is not True
        or prefix.get("kind") != "qwen_toolgen_fixed_checkpoint_audit"
        or prefix.get("reference_subset_mapping_match") is not True
    ):
        raise ValueError("Incremental checkpoint or old-token prefix audit failed")

    tools = int(runtime["incremental_tools"])
    manifest_audits = manifest.get("audits", {})
    manifest_counts = manifest.get("counts", {})
    if (
        manifest.get("kind") != "qwen_toolgen_incremental_document_registration"
        or manifest.get("split") != "test"
        or manifest.get("training_information")
        != ["tool_document", "fixed_token_label"]
        or manifest_audits.get("base_incremental_token_overlap") != 0
        or manifest_audits.get("evaluation_query_records_used") != 0
        or manifest_audits.get("evaluation_argument_records_used") != 0
        or manifest_audits.get("evaluation_trajectory_records_used") != 0
        or manifest_audits.get("incremental_tokens_unique") is not True
        or int(manifest_counts.get("base_tokens", -1)) != 51_895
        or int(manifest_counts.get("incremental_tools", -1)) != tools
        or int(manifest_counts.get("memorization_records", -1)) != tools
        or int(manifest_counts.get("combined_tokens", -1)) != 51_895 + tools
    ):
        raise ValueError("Incremental registration manifest violates document-only isolation")
    for name, path in required_inputs.items():
        if name == "incremental_manifest.json":
            continue
        declared = manifest.get("outputs", {}).get(name, {})
        if (
            Path(str(declared.get("path", ""))).resolve() != path
            or declared.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"Incremental manifest output mismatch: {name}")
    wall_seconds = _finite(runtime.get("wall_seconds"), "incremental.wall_seconds")
    wall_per_tool = _finite(
        runtime.get("wall_seconds_per_tool"), "incremental.wall_seconds_per_tool"
    )
    if wall_seconds < 0.0 or wall_per_tool < 0.0:
        raise ValueError("Incremental registration reported negative runtime")
    return {
        "incremental_tools": tools,
        "optimizer_steps": int(runtime["optimizer_steps"]),
        "model_parameters_updated": True,
        "registration_optimizer_updates_required": True,
        "wall_seconds": wall_seconds,
        "wall_seconds_per_tool": wall_per_tool,
        "training_label_leakage": leakage,
        "old_token_physical_ids_unchanged": True,
        "full_model_reloaded": True,
        "runtime_sha256": sha256_file(runtime_path),
        "checkpoint_audit_sha256": sha256_file(checkpoint_path),
        "base_token_prefix_audit_sha256": sha256_file(prefix_path),
        "output_catalog_sha256": sha256_file(catalog_path),
        "input_catalog_sha256": sha256_file(input_catalog_path),
        "code_catalog_sha256": sha256_file(code_catalog_path),
        "incremental_manifest_sha256": sha256_file(manifest_path),
        "training_information": list(manifest["training_information"]),
        "model_path": str(Path(str(checkpoint.get("model_path", ""))).resolve()),
    }


def _fixed_checkpoint_evidence(root: Path) -> dict[str, Any]:
    root = root.resolve()
    audit_paths = {
        stage: root / f"checkpoint_audit_stage{stage}.json" for stage in (1, 2, 3)
    }
    audits = {stage: _load_complete(path) for stage, path in audit_paths.items()}
    audit_path = audit_paths[3]
    audit = audits[3]
    if (
        audit.get("kind") != "qwen_toolgen_fixed_checkpoint_audit"
        or audit.get("full_model_reloaded") is not True
        or audit.get("input_embeddings_all_finite") is not True
        or audit.get("output_embeddings_all_finite") is not True
    ):
        raise ValueError("Fixed-token checkpoint audit did not pass full reload")
    expected_model = (root / "Qwen3-8B-Fixed-Agent").resolve()
    if Path(str(audit.get("model_path", ""))).resolve() != expected_model:
        raise ValueError("Fixed-token checkpoint audit refers to another model")
    catalog_path = root / "checkpoint_audits.sha256"
    observed: dict[Path, str] = {}
    for line in catalog_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("Malformed fixed checkpoint audit catalog")
        path = Path(fields[1].lstrip("* ")).resolve()
        if path in observed:
            raise ValueError(f"Duplicate fixed checkpoint audit checksum: {path}")
        observed[path] = fields[0]
    expected_paths = {path.resolve() for path in audit_paths.values()}
    if set(observed) != expected_paths:
        raise ValueError("Fixed checkpoint catalog must bind exactly stages 1--3")
    for path, digest in observed.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Fixed checkpoint audit digest mismatch: {path}")

    mapping_fields = (
        "normalized_token_count",
        "normalized_tokens_sha256",
        "token_mapping_sha256",
        "tokenizer_size",
        "first_tool_token_id",
        "last_tool_token_id",
    )
    reference = audits[1]
    for stage, stage_audit in audits.items():
        if stage_audit.get("kind") != "qwen_toolgen_fixed_checkpoint_audit":
            raise ValueError(f"Unexpected fixed checkpoint stage-{stage} audit kind")
        if any(stage_audit.get(field) != reference.get(field) for field in mapping_fields):
            raise ValueError(f"Fixed checkpoint stage-{stage} token mapping changed")
        if stage_audit.get("mapping_is_contiguous_suffix_range") is not True:
            raise ValueError(f"Fixed checkpoint stage-{stage} token IDs are not contiguous")
    if (
        audits[2].get("reference_mapping_match") is not True
        or audits[3].get("reference_mapping_match") is not True
    ):
        raise ValueError("Fixed checkpoint did not preserve its reference token mapping")
    token_count = int(audit.get("normalized_token_count", -1))
    first_token_id = int(audit.get("first_tool_token_id", -1))
    last_token_id = int(audit.get("last_tool_token_id", -1))
    if token_count != 51_895 or last_token_id - first_token_id + 1 != token_count:
        raise ValueError("Fixed checkpoint does not contain the expected 51,895-token range")
    return {
        "model_path": str(expected_model),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "catalog_sha256": sha256_file(catalog_path),
        "token_mapping_sha256": audit.get("token_mapping_sha256"),
        "token_count": token_count,
        "first_tool_token_id": first_token_id,
        "last_tool_token_id": last_token_id,
        "reference_mapping_match": True,
        "full_model_reloaded": True,
    }


def _baseline_result(
    comparison: Mapping[str, Any],
    *,
    label: str,
    expected_common_agent: Mapping[str, Any],
    expected_model_path: str | None = None,
    expected_kind: str | None = None,
    expected_registration_control: str | None = None,
) -> tuple[dict[str, Any], Path]:
    sources = comparison.get("sources", {})
    predictions_path = Path(str(sources.get("baseline_path", ""))).resolve()
    predictions_sha256 = str(sources.get("baseline_sha256", ""))
    if not predictions_path.is_file() or sha256_file(predictions_path) != predictions_sha256:
        raise ValueError(f"{label} baseline predictions are missing or modified")
    result_path = predictions_path.parent / "results.json"
    result = _load_complete(result_path)
    if result.get("predictions_sha256") != predictions_sha256:
        raise ValueError(f"{label} baseline result does not bind its predictions")
    metric_hash = str(result.get("code_hashes", {}).get("benchmark_metrics", ""))
    if metric_hash != sources.get("benchmark_metrics_code_sha256"):
        raise ValueError(f"{label} baseline result uses another metric implementation")
    common_agent = result.get("common_document_agent")
    if not isinstance(common_agent, dict):
        raise ValueError(f"{label} baseline result lacks the common Agent audit")
    for field in ("model_path", "audit_path", "audit_sha256"):
        if common_agent.get(field) != expected_common_agent.get(field):
            raise ValueError(f"{label} baseline uses another common Agent: {field}")
    if expected_kind is not None and result.get("kind") != expected_kind:
        raise ValueError(f"{label} baseline has another system kind")
    if (
        expected_registration_control is not None
        and result.get("registration_control") != expected_registration_control
    ):
        raise ValueError(f"{label} baseline has another registration control")
    if expected_model_path is not None and Path(
        str(result.get("model_path", ""))
    ).resolve() != Path(expected_model_path).resolve():
        raise ValueError(f"{label} baseline result uses another selection model")
    return result, result_path


def _validate_candidate_binding(
    comparison: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    label: str,
    expected_common_agent: Mapping[str, Any],
    expected_condition: str,
    expected_checkpoint_hashes: Mapping[str, Any],
) -> str:
    sources = comparison.get("sources", {})
    predictions_sha256 = str(candidate.get("predictions_sha256", ""))
    if len(predictions_sha256) != 64 or sources.get("candidate_sha256") != predictions_sha256:
        raise ValueError(f"{label} does not use the declared late-bound predictions")

    metric_code_sha256 = str(
        candidate.get("code_hashes", {}).get("benchmark_metrics", "")
    )
    if (
        len(metric_code_sha256) != 64
        or sources.get("benchmark_metrics_code_sha256") != metric_code_sha256
    ):
        raise ValueError(f"{label} uses mismatched metric code")

    comparison_code_sha256 = str(
        sources.get("compare_controlled_predictions_code_sha256", "")
    )
    if len(comparison_code_sha256) != 64:
        raise ValueError(f"{label} has an invalid comparison code hash")

    common_agent = candidate.get("common_document_agent")
    if not isinstance(common_agent, dict):
        raise ValueError(f"{label} is missing the common-document Agent audit")
    for field in ("model_path", "audit_path", "audit_sha256"):
        if not common_agent.get(field) or common_agent.get(field) != expected_common_agent.get(
            field
        ):
            raise ValueError(f"{label} uses a different common-document Agent: {field}")
    if candidate.get("registration_control") != "registered":
        raise ValueError(f"{label} candidate is not registered late binding")
    if (
        candidate.get("kind") != "qwen_late_bound"
        or candidate.get("information_condition") != "common_document"
        or candidate.get("condition") != expected_condition
    ):
        raise ValueError(f"{label} candidate has another system or condition")
    if candidate.get("checkpoint_hashes") != expected_checkpoint_hashes:
        raise ValueError(f"{label} candidate uses another late-bound checkpoint")

    selected_families = comparison.get("selected_families", ["all"])
    if selected_families == ["all"]:
        families = set(comparison.get("families", {}))
    else:
        families = {str(family) for family in selected_families}
        if families != set(comparison.get("families", {})):
            raise ValueError(f"{label} has inconsistent selected families")
    condition = str(candidate["condition"])
    expected_pair_count = sum(
        int(candidate["aggregates"][f"{family}:{condition}"]["count"])
        for family in families
    )
    if int(comparison.get("pair_count", -1)) != expected_pair_count:
        raise ValueError(f"{label} has an incomplete prediction pairing")
    if int(comparison.get("registry_mismatches", -1)) != 0:
        raise ValueError(f"{label} has a registry pairing mismatch")
    if bool(comparison.get("require_registry_match")) and int(
        comparison.get("registry_pairs", -1)
    ) != expected_pair_count:
        raise ValueError(f"{label} has incomplete registry identity evidence")
    return comparison_code_sha256


def summarize_shared_agent_experiment(
    *,
    fixed_root: Path,
    fixed_scale_root: Path,
    incremental_root: Path,
    oracle_root: Path,
    oracle_scale_root: Path,
    control_root: Path,
    latebound_root: Path,
    incremental_train_root: Path,
    fixed_train_root: Path,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    baseline_comparisons: dict[str, Any] = {}
    runtime_cells: dict[str, Any] = {}
    incremental_registration = _incremental_registration_cost(
        incremental_train_root
    )
    fixed_checkpoint = _fixed_checkpoint_evidence(fixed_train_root)

    latebound_path = (
        latebound_root
        / "unseen-tool_unseen-address"
        / "registry-1000"
        / "results.json"
    )
    latebound = _load_complete(latebound_path)
    condition = str(latebound["condition"])
    retrieval = latebound["aggregates"][f"retrieval:{condition}"]
    arguments = latebound["aggregates"][f"arguments:{condition}"]
    audit = _registration_audit(latebound)
    common_agent = _common_agent_evidence(latebound.get("common_document_agent"))
    checkpoint_hashes = latebound.get("checkpoint_hashes")
    required_checkpoint_hashes = {
        "retrieval_adapter_config",
        "retrieval_adapter_weights",
        "retrieval_compiler",
        "memory_compiler",
        "training_results",
    }
    if (
        not isinstance(checkpoint_hashes, dict)
        or set(checkpoint_hashes) != required_checkpoint_hashes
        or any(len(str(value)) != 64 for value in checkpoint_hashes.values())
    ):
        raise ValueError("Registered result has incomplete checkpoint hashes")
    absolute = {
        "retrieval_count": int(retrieval["count"]),
        "minimum_retrieval_count": MIN_RETRIEVAL_EXAMPLES,
        "full_vocabulary_hit_at_1": _finite(
            retrieval["metrics"]["hit_at_1"], "registered.hit_at_1"
        ),
        "argument_count": int(arguments["count"]),
        "common_document_end_to_end_argument_exact": _finite(
            arguments["metrics"]["end_to_end_argument_exact"],
            "registered.end_to_end_argument_exact",
        ),
    }

    comparison_code_hashes: set[str] = set()
    for size in MATCHED_SIZES:
        cells = {
            "fixed_seen_address": (
                *_comparison_cell(
                    fixed_root, f"seen-tool_seen-address/registry-{size}"
                ),
                "seen-tool_seen-address",
            ),
            "fixed_unseen_address": (
                *_comparison_cell(
                    fixed_root, f"seen-tool_unseen-address/registry-{size}"
                ),
                "seen-tool_unseen-address",
            ),
            "incremental": (
                *_comparison_cell(
                    incremental_root, f"unseen-tool_unseen-address/registry-{size}"
                ),
                "unseen-tool_unseen-address",
            ),
            "oracle_seen_address": (
                *_comparison_cell(
                    oracle_root, f"unseen-tool_seen-address/registry-{size}"
                ),
                "unseen-tool_seen-address",
            ),
            "oracle_unseen_address": (
                *_comparison_cell(
                    oracle_root, f"unseen-tool_unseen-address/registry-{size}"
                ),
                "unseen-tool_unseen-address",
            ),
        }
        baseline_comparisons[f"registry_{size}"] = {}
        for name, (comparison, path, candidate_condition) in cells.items():
            candidate_path = (
                latebound_root
                / candidate_condition
                / f"registry-{size}"
                / "results.json"
            )
            candidate = _load_complete(candidate_path)
            if name.startswith("fixed"):
                expected_baseline_model = fixed_checkpoint["model_path"]
                expected_baseline_kind = "qwen_toolgen_fixed"
            elif name == "incremental":
                expected_baseline_model = incremental_registration["model_path"]
                expected_baseline_kind = "qwen_toolgen_fixed"
            else:
                expected_baseline_model = common_agent["model_path"]
                expected_baseline_kind = "qwen_full_document"
            baseline_result, baseline_result_path = _baseline_result(
                comparison,
                label=f"{name}.registry_{size}",
                expected_common_agent=common_agent,
                expected_model_path=expected_baseline_model,
                expected_kind=expected_baseline_kind,
            )
            runtime_label = f"{candidate['condition']}.registry_{size}"
            expected_condition = candidate_condition.replace("-", "_")
            runtime_cells[runtime_label] = _runtime_cell(
                candidate,
                runtime_label,
                expected_condition=expected_condition,
                expected_registry_size=size,
            )
            comparison_code_hashes.add(
                _validate_candidate_binding(
                    comparison,
                    candidate,
                    label=f"{name}.registry_{size}",
                    expected_common_agent=common_agent,
                    expected_condition=expected_condition,
                    expected_checkpoint_hashes=checkpoint_hashes,
                )
            )
            baseline_comparisons[f"registry_{size}"][name] = {
                "retrieval_hit_at_1": _paired_report(
                    _paired_metric(comparison, "retrieval", "hit_at_1"),
                    f"{name}.registry_{size}.retrieval.hit_at_1",
                )
                if "retrieval" in comparison.get("families", {})
                else None,
                "end_to_end_argument_exact": _paired_report(
                    _paired_metric(
                        comparison, "arguments", "end_to_end_argument_exact"
                    ),
                    f"{name}.registry_{size}.arguments.end_to_end_argument_exact",
                ),
            }
            sources[f"{name}_registry_{size}"] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "candidate_results_path": str(candidate_path.resolve()),
                "candidate_results_sha256": sha256_file(candidate_path),
                "baseline_results_path": str(baseline_result_path.resolve()),
                "baseline_results_sha256": sha256_file(baseline_result_path),
                "baseline_model_path": baseline_result.get("model_path"),
            }

    for size in SCALE_SIZES:
        fixed, fixed_path = _comparison_cell(
            fixed_scale_root, f"seen-tool_unseen-address/registry-{size}"
        )
        oracle, oracle_path = _comparison_cell(
            oracle_scale_root, f"unseen-tool_unseen-address/registry-{size}"
        )
        fixed_candidate_path = (
            latebound_root
            / "seen-tool_unseen-address"
            / f"registry-{size}"
            / "results.json"
        )
        oracle_candidate_path = (
            latebound_root
            / "unseen-tool_unseen-address"
            / f"registry-{size}"
            / "results.json"
        )
        fixed_candidate = _load_complete(fixed_candidate_path)
        oracle_candidate = _load_complete(oracle_candidate_path)
        fixed_baseline_result, fixed_baseline_result_path = _baseline_result(
            fixed,
            label=f"fixed.registry_{size}",
            expected_common_agent=common_agent,
            expected_model_path=fixed_checkpoint["model_path"],
            expected_kind="qwen_toolgen_fixed",
        )
        oracle_baseline_result, oracle_baseline_result_path = _baseline_result(
            oracle,
            label=f"oracle.registry_{size}",
            expected_common_agent=common_agent,
            expected_model_path=common_agent["model_path"],
            expected_kind="qwen_full_document",
        )
        fixed_runtime_label = f"seen_tool_unseen_address.registry_{size}"
        oracle_runtime_label = f"unseen_tool_unseen_address.registry_{size}"
        runtime_cells[fixed_runtime_label] = _runtime_cell(
            fixed_candidate,
            fixed_runtime_label,
            expected_condition="seen_tool_unseen_address",
            expected_registry_size=size,
        )
        runtime_cells[oracle_runtime_label] = _runtime_cell(
            oracle_candidate,
            oracle_runtime_label,
            expected_condition="unseen_tool_unseen_address",
            expected_registry_size=size,
        )
        comparison_code_hashes.add(
            _validate_candidate_binding(
                fixed,
                fixed_candidate,
                label=f"fixed.registry_{size}",
                expected_common_agent=common_agent,
                expected_condition="seen_tool_unseen_address",
                expected_checkpoint_hashes=checkpoint_hashes,
            )
        )
        comparison_code_hashes.add(
            _validate_candidate_binding(
                oracle,
                oracle_candidate,
                label=f"oracle.registry_{size}",
                expected_common_agent=common_agent,
                expected_condition="unseen_tool_unseen_address",
                expected_checkpoint_hashes=checkpoint_hashes,
            )
        )
        baseline_comparisons[f"registry_{size}"] = {
            "fixed": {
                "retrieval_hit_at_1": _paired_report(
                    _paired_metric(fixed, "retrieval", "hit_at_1"),
                    f"fixed.registry_{size}.retrieval.hit_at_1",
                ),
                "end_to_end_argument_exact": _paired_report(
                    _paired_metric(fixed, "arguments", "end_to_end_argument_exact"),
                    f"fixed.registry_{size}.arguments.end_to_end_argument_exact",
                ),
            },
            "oracle": {
                "end_to_end_argument_exact": _paired_report(
                    _paired_metric(
                        oracle, "arguments", "end_to_end_argument_exact"
                    ),
                    f"oracle.registry_{size}.arguments.end_to_end_argument_exact",
                )
            },
        }
        sources[f"fixed_registry_{size}"] = {
            "path": str(fixed_path.resolve()),
            "sha256": sha256_file(fixed_path),
            "candidate_results_path": str(fixed_candidate_path.resolve()),
            "candidate_results_sha256": sha256_file(fixed_candidate_path),
            "baseline_results_path": str(fixed_baseline_result_path.resolve()),
            "baseline_results_sha256": sha256_file(fixed_baseline_result_path),
            "baseline_model_path": fixed_baseline_result.get("model_path"),
        }
        sources[f"oracle_registry_{size}"] = {
            "path": str(oracle_path.resolve()),
            "sha256": sha256_file(oracle_path),
            "candidate_results_path": str(oracle_candidate_path.resolve()),
            "candidate_results_sha256": sha256_file(oracle_candidate_path),
            "baseline_results_path": str(oracle_baseline_result_path.resolve()),
            "baseline_results_sha256": sha256_file(oracle_baseline_result_path),
            "baseline_model_path": oracle_baseline_result.get("model_path"),
        }

    controls: dict[str, Any] = {}
    for control in SELECTION_CONTROLS:
        comparison, path = _comparison_cell(control_root, f"control-{control}")
        comparison_code_sha256 = _validate_candidate_binding(
            comparison,
            latebound,
            label=f"control.{control}",
            expected_common_agent=common_agent,
            expected_condition="unseen_tool_unseen_address",
            expected_checkpoint_hashes=checkpoint_hashes,
        )
        comparison_code_hashes.add(comparison_code_sha256)
        control_baseline, control_baseline_path = _baseline_result(
            comparison,
            label=f"control.{control}",
            expected_common_agent=common_agent,
            expected_kind="qwen_late_bound",
            expected_registration_control=control,
        )
        controls[control] = {
            "selection": _significance(
                _paired_metric(comparison, "retrieval", "hit_at_1"),
                f"{control}.retrieval.hit_at_1",
                expected_count=absolute["retrieval_count"],
                minimum_count=MIN_RETRIEVAL_EXAMPLES,
            ),
            "common_document_end_to_end": _significance(
                _paired_metric(
                    comparison, "arguments", "end_to_end_argument_exact"
                ),
                f"{control}.arguments.end_to_end_argument_exact",
                expected_count=absolute["argument_count"],
                minimum_count=1,
            ),
        }
        sources[f"control_{control}"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "baseline_results_path": str(control_baseline_path.resolve()),
            "baseline_results_sha256": sha256_file(control_baseline_path),
            "baseline_condition": control_baseline.get("condition"),
        }
    if len(comparison_code_hashes) != 1:
        raise ValueError("Shared-Agent cells use different comparison code")

    selection_supported = (
        absolute["retrieval_count"] >= MIN_RETRIEVAL_EXAMPLES
        and absolute["full_vocabulary_hit_at_1"] > 0.0
        and all(value["selection"]["passed"] for value in controls.values())
    )
    end_to_end_supported = (
        absolute["common_document_end_to_end_argument_exact"] > 0.0
        and all(
            value["common_document_end_to_end"]["passed"]
            for value in controls.values()
        )
    )
    strict_post_training_registration = bool(audit["passed"])
    return {
        "kind": "shared_agent_exploratory_comparison",
        "version": 1,
        "code_sha256": sha256_file(Path(__file__)),
        "core_claim_supported": (
            selection_supported and strict_post_training_registration
        ),
        "common_document_end_to_end_supported": end_to_end_supported,
        "selection_supported": selection_supported,
        "strict_post_training_registration": strict_post_training_registration,
        "registered_absolute_metrics": absolute,
        "registration_audit": audit,
        "controls": controls,
        "baseline_comparisons": baseline_comparisons,
        "runtime_and_storage": {
            "latency_definition": (
                "retrieval_selection times full-vocabulary tool selection; "
                "argument_path times selection, selected-document dereference, "
                "and shared-Agent generation; model loading is excluded"
            ),
            "cells": runtime_cells,
            "same_stream_generation": latebound.get("training_provenance", {}).get(
                "same_stream_generation", {}
            ),
            "common_document_agent": common_agent,
            "incremental_toolgen_registration": incremental_registration,
            "fixed_toolgen_checkpoint": fixed_checkpoint,
        },
        "sources": {
            **sources,
            "latebound_results": {
                "path": str(latebound_path.resolve()),
                "sha256": sha256_file(latebound_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the shared-Agent exploratory comparison"
    )
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--fixed-scale-root", type=Path, required=True)
    parser.add_argument("--incremental-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--oracle-scale-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--latebound-root", type=Path, required=True)
    parser.add_argument("--incremental-train-root", type=Path, required=True)
    parser.add_argument("--fixed-train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_shared_agent_experiment(
        fixed_root=args.fixed_root,
        fixed_scale_root=args.fixed_scale_root,
        incremental_root=args.incremental_root,
        oracle_root=args.oracle_root,
        oracle_scale_root=args.oracle_scale_root,
        control_root=args.control_root,
        latebound_root=args.latebound_root,
        incremental_train_root=args.incremental_train_root,
        fixed_train_root=args.fixed_train_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    write_json_atomic(output, result)
    (args.output_dir / "COMPLETE").write_text(
        f"{sha256_file(output)}  summary.json\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
