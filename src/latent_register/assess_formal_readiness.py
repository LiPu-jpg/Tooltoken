"""Assess whether an exploratory late-bound run justifies formal seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .physical_tokens import validate_token_identity_audit
from .validate_toolgen_tokenizer import sha256_file, write_json_atomic


REQUIRED_CONTROLS = (
    "blank",
    "random",
    "wrong_memory",
    "permuted",
    "shared_vector",
    "query_only",
    "nearest_trained",
)
SELECTION_CONTROLS = tuple(
    control for control in REQUIRED_CONTROLS if control != "wrong_memory"
)
OUTPUT_CONTROLS = SELECTION_CONTROLS
MIN_RETRIEVAL_EXAMPLES = 2000


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value}")
    return result


def _metric(result: Mapping[str, Any], family: str, name: str) -> float:
    condition = str(result["condition"])
    return _finite(
        result["aggregates"][f"{family}:{condition}"]["metrics"][name],
        f"{family}.{name}",
    )


def _comparison_metric(
    comparison: Mapping[str, Any], family: str, name: str
) -> Mapping[str, Any]:
    try:
        value = comparison["families"][family]["paired_metrics"][name]
    except KeyError as exc:
        raise ValueError(f"Comparison is missing {family}.{name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Comparison metric is not an object: {family}.{name}")
    return value


def _check(
    checks: dict[str, Any], name: str, passed: bool, **evidence: Any
) -> None:
    checks[name] = {"passed": bool(passed), **evidence}


def assess_formal_readiness(
    registered_results_path: Path,
    control_comparisons: Mapping[str, Path],
) -> dict[str, Any]:
    if set(control_comparisons) != set(REQUIRED_CONTROLS):
        raise ValueError(f"Controls must contain exactly {REQUIRED_CONTROLS}")
    if not (registered_results_path.parent / "COMPLETE").is_file():
        raise FileNotFoundError(
            f"Registered evaluation is incomplete: {registered_results_path.parent}"
        )
    registered = json.loads(registered_results_path.read_text(encoding="utf-8"))
    if registered.get("kind") != "qwen_late_bound":
        raise ValueError("Unexpected registered evaluation kind")
    expected_identity = {
        "source_condition": "unseen_tool_unseen_token",
        "address_status": "unseen",
        "registration_control": "registered",
        "registry_size": 1000,
        "information_condition": "common_document",
    }
    observed_identity = {
        key: registered.get(key) for key in expected_identity
    }
    if observed_identity != expected_identity:
        raise ValueError(
            f"Registered readiness cell mismatch: {observed_identity} != "
            f"{expected_identity}"
        )

    condition = str(registered["condition"])
    retrieval_group = registered["aggregates"][f"retrieval:{condition}"]
    retrieval_count = int(retrieval_group["count"])
    full_hit = _metric(registered, "retrieval", "hit_at_1")
    constrained_hit = _finite(
        registered["registry_constrained_retrieval"]["metrics"]["hit_at_1"],
        "registry_constrained_retrieval.hit_at_1",
    )

    checks: dict[str, Any] = {}
    _check(
        checks,
        "sample_size",
        retrieval_count >= MIN_RETRIEVAL_EXAMPLES,
        retrieval_count=retrieval_count,
        minimum_retrieval_count=MIN_RETRIEVAL_EXAMPLES,
    )
    _check(
        checks,
        "registered_selection_is_nonzero",
        full_hit > 0.0,
        full_vocabulary_hit_at_1=full_hit,
        registry_constrained_hit_at_1=constrained_hit,
    )

    registration_audit = registered["registration_audit"]
    mutation_audit = registered["evaluation_mutation_audit"]
    physical_audit = registered["physical_token_audit"]
    physical_identity = validate_token_identity_audit(physical_audit)
    registered_code_hashes = registered.get("code_hashes", {})
    expected_code_keys = {
        "evaluate_late_bound",
        "benchmark_metrics",
        "model",
        "physical_tokens",
    }
    code_hashes_valid = set(registered_code_hashes) == expected_code_keys and all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in registered_code_hashes.values()
    )
    audit_values = {
        "optimizer_steps": registration_audit.get("optimizer_steps"),
        "model_or_table_parameters_changed": registration_audit.get(
            "model_or_table_parameters_changed"
        ),
        "all_reserved_input_row_max_change": mutation_audit.get(
            "all_reserved_input_row_max_change"
        ),
        "all_reserved_output_row_max_change": mutation_audit.get(
            "all_reserved_output_row_max_change"
        ),
        "evaluation_ids_seen_during_training": physical_audit.get(
            "evaluation_ids_seen_during_training"
        ),
        "evaluation_input_row_max_change": physical_audit.get(
            "evaluation_input_row_max_change"
        ),
        "evaluation_output_row_max_change": physical_audit.get(
            "evaluation_output_row_max_change"
        ),
    }
    selection_contract = {
        "selection_emits_single_physical_id": registration_audit.get(
            "selection_emits_single_physical_id"
        ),
        "selected_physical_bindings_verified": registration_audit.get(
            "selected_physical_bindings_verified"
        ),
        "selected_id_dereferences_registered_memory": registration_audit.get(
            "selected_id_dereferences_registered_memory"
        ),
        "selected_id_dereferences_full_document": registration_audit.get(
            "selected_id_dereferences_full_document"
        ),
        "selected_id_uses_static_embedding": registration_audit.get(
            "selected_id_uses_static_embedding"
        ),
        "ordinary_vocabulary_competes_at_selection": registration_audit.get(
            "ordinary_vocabulary_competes_at_selection"
        ),
        "inactive_reserved_ids_masked": registration_audit.get(
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
            registration_audit.get("selected_physical_binding_count", -1)
        ),
        "selected_physical_binding_verified_count": int(
            registration_audit.get("selected_physical_binding_verified_count", -1)
        ),
        "selected_argument_payload_count": int(
            registration_audit.get("selected_argument_payload_count", -1)
        ),
        "selected_argument_payload_verified_count": int(
            registration_audit.get("selected_argument_payload_verified_count", -1)
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
    _check(
        checks,
        "registration_contract",
        all(_finite(value, name) == 0.0 for name, value in audit_values.items())
        and code_hashes_valid
        and bool(physical_identity["passed"])
        and selection_contract_passed
        and binding_counts_valid
        and int(registration_audit.get("registration_forwards_per_tool", -1)) == 1,
        **audit_values,
        registration_forwards_per_tool=registration_audit.get(
            "registration_forwards_per_tool"
        ),
        code_hashes=registered_code_hashes,
        code_hashes_valid=code_hashes_valid,
        physical_token_identity=physical_identity,
        selection_contract=selection_contract,
        selection_contract_passed=selection_contract_passed,
        binding_evidence=binding_evidence,
        binding_counts_valid=binding_counts_valid,
    )

    control_evidence: dict[str, Any] = {}
    comparison_code_hashes: dict[str, dict[str, str]] = {}
    registered_predictions_sha256 = str(registered["predictions_sha256"])
    for control in REQUIRED_CONTROLS:
        path = control_comparisons[control]
        if not (path.parent / "COMPLETE").is_file():
            raise FileNotFoundError(f"Control comparison is incomplete: {path.parent}")
        comparison = json.loads(path.read_text(encoding="utf-8"))
        sources = comparison.get("sources", {})
        if sources.get("candidate_sha256") != registered_predictions_sha256:
            raise ValueError(
                f"Control {control} does not use the registered readiness predictions"
            )
        comparison_code = {
            "compare_controlled_predictions": str(
                sources.get("compare_controlled_predictions_code_sha256", "")
            ),
            "benchmark_metrics": str(
                sources.get("benchmark_metrics_code_sha256", "")
            ),
        }
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in comparison_code.values()
        ):
            raise ValueError(f"Control {control} has invalid comparison code hashes")
        comparison_code_hashes[control] = comparison_code
        evidence: dict[str, Any] = {"diagnostic_only": control == "wrong_memory"}
        if "arguments" in comparison.get("families", {}):
            argument_metric = _comparison_metric(
                comparison, "arguments", "end_to_end_key_recall"
            )
            argument_ci = argument_metric["paired_bootstrap_95_ci"]
            argument_delta = _finite(
                argument_metric["delta"], f"{control}.key_recall"
            )
            evidence.update(
                {
                    "end_to_end_key_recall_delta": argument_delta,
                    "end_to_end_key_recall_ci": argument_ci,
                }
            )
        if control in SELECTION_CONTROLS:
            retrieval_metric = _comparison_metric(comparison, "retrieval", "hit_at_1")
            paired_retrieval_count = int(retrieval_metric.get("count", -1))
            retrieval_ci = retrieval_metric["paired_bootstrap_95_ci"]
            retrieval_delta = _finite(
                retrieval_metric["delta"], f"{control}.hit_at_1"
            )
            retrieval_lower = _finite(
                retrieval_ci[0], f"{control}.hit_at_1.ci_lower"
            )
            evidence.update(
                {
                    "retrieval_hit_at_1_delta": retrieval_delta,
                    "retrieval_hit_at_1_ci": retrieval_ci,
                    "paired_retrieval_count": paired_retrieval_count,
                    "minimum_paired_retrieval_count": MIN_RETRIEVAL_EXAMPLES,
                    "paired_retrieval_count_matches_registered": (
                        paired_retrieval_count == retrieval_count
                    ),
                    "retrieval_passed": (
                        paired_retrieval_count == retrieval_count
                        and paired_retrieval_count >= MIN_RETRIEVAL_EXAMPLES
                        and retrieval_delta > 0.0
                        and retrieval_lower > 0.0
                    ),
                }
            )
        control_evidence[control] = evidence
        _check(
            checks,
            f"control_{control}",
            bool(evidence.get("retrieval_passed", True)),
            **evidence,
        )

    code_hashes_match = len(
        {
            tuple(sorted(value.items()))
            for value in comparison_code_hashes.values()
        }
    ) == 1
    comparison_metric_hash = comparison_code_hashes[REQUIRED_CONTROLS[0]][
        "benchmark_metrics"
    ]
    evaluator_metric_hash = str(registered_code_hashes.get("benchmark_metrics", ""))
    _check(
        checks,
        "scoring_code_consistency",
        code_hashes_match and comparison_metric_hash == evaluator_metric_hash,
        comparison_code_hashes=comparison_code_hashes,
        evaluator_benchmark_metrics_sha256=evaluator_metric_hash,
    )

    passed = all(bool(value["passed"]) for value in checks.values())
    return {
        "kind": "formal_latebound_readiness_gate",
        "version": 3,
        "passed": passed,
        "checks": checks,
        "failed_checks": sorted(
            name for name, value in checks.items() if not value["passed"]
        ),
        "controls": control_evidence,
        "sources": {
            "registered_results_path": str(registered_results_path.resolve()),
            "registered_results_sha256": sha256_file(registered_results_path),
            "control_comparisons": {
                control: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for control, path in sorted(control_comparisons.items())
            },
        },
    }


def _parse_control(value: str) -> tuple[str, Path]:
    control, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(
            "Control comparison must be NAME=/path/to/comparison.json"
        )
    return control, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the predeclared gate before formal late-bound seeds"
    )
    parser.add_argument("--registered-results", type=Path, required=True)
    parser.add_argument("--control-comparison", action="append", type=_parse_control, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controls = dict(args.control_comparison)
    if len(controls) != len(args.control_comparison):
        raise ValueError("Duplicate control comparison")
    result = assess_formal_readiness(args.registered_results, controls)
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
