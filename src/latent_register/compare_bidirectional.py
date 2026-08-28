from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from .paired_predictions import compare_predictions, load_predictions


AUDIT_EXPECTATIONS = {
    "heldout_ids_seen_during_training": 0,
    "heldout_input_row_max_change": 0.0,
    "heldout_output_row_max_change": 0.0,
    "all_backbone_parameters_frozen": True,
    "dynamic_output_full_head_max_delta": 0.0,
}

LAYERWISE_INTERFACE_CONFIG_DIFFERENCES = {
    "input_memory_interface",
    "layerwise_memory_layers",
    "layerwise_memory_max_gate",
}

METRICS = {
    "selection_top1": ("selection", "generated_physical_rows", "top1"),
    "selection_top5": ("selection", "generated_physical_rows", "top5"),
    "registered_argument_nll": (
        "input_readback_teacher_forced",
        "registered",
        "nll",
    ),
    "registered_schema_nll": (
        "schema_readback_teacher_forced",
        "registered",
        "nll",
    ),
    "trajectory_nll": ("trajectory_readback_teacher_forced", "nll"),
    "registered_exact_arguments": (
        "input_readback_generation",
        "registered",
        "exact_arguments",
    ),
    "registered_exact_schema": (
        "input_readback_generation",
        "registered",
        "exact_schema_keys",
    ),
    "registered_key_precision": (
        "input_readback_generation",
        "registered",
        "key_precision",
    ),
    "registered_key_recall": (
        "input_readback_generation",
        "registered",
        "key_recall",
    ),
    "registered_shared_value_accuracy": (
        "input_readback_generation",
        "registered",
        "shared_key_value_accuracy",
    ),
    "staged_end_to_end_exact": (
        "constrained_registered_pipeline",
        "end_to_end_exact_arguments",
    ),
    "trajectory_json_valid": (
        "autoregressive_registered_trajectory",
        "json_valid",
    ),
    "trajectory_readback_given_correct_selection": (
        "autoregressive_registered_trajectory",
        "readback_exact_arguments_given_correct_selection",
    ),
    "trajectory_end_to_end_exact": (
        "autoregressive_registered_trajectory",
        "end_to_end_exact_arguments",
    ),
}

INPUT_EPOCH_PATTERN = re.compile(
    r"input_epoch=(?P<epoch>\d+)/(?P<total>\d+) "
    r"phase=(?P<phase>\w+) "
    r"loss=(?P<loss>\S+) "
    r"execution=(?P<execution>\S+) "
    r"schema=(?P<schema>\S+) "
    r"distill=(?P<distill>\S+) "
    r"trajectory=(?P<trajectory>\S+)"
)


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing result field: {'.'.join(path)}")
        value = value[key]
    return value


def validate_audits(payload: dict[str, Any], label: str) -> None:
    audit = payload.get("physical_token_audit")
    if not isinstance(audit, dict):
        raise ValueError(f"{label} has no physical_token_audit")
    failures = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in AUDIT_EXPECTATIONS.items()
        if audit.get(key) != expected
    }
    if failures:
        raise ValueError(f"{label} failed physical-token audits: {failures}")


def validate_training_log(
    text: str,
    schema_warmup_epochs: int,
    trajectory_loss_weight: float,
    schema_loss_weight: float = 0.0,
    distill_loss_weight: float = 0.0,
    expected_joint_epochs: int | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for match in INPUT_EPOCH_PATTERN.finditer(text):
        record: dict[str, Any] = {
            "epoch": int(match.group("epoch")),
            "total": int(match.group("total")),
            "phase": match.group("phase"),
        }
        for name in ("loss", "execution", "schema", "distill", "trajectory"):
            record[name] = float(match.group(name))
            if not math.isfinite(record[name]):
                raise ValueError(f"Non-finite {name} at input epoch {record['epoch']}")
        records.append(record)
    if not records:
        raise ValueError("Training log contains no input epoch records")

    total = records[0]["total"]
    if (
        expected_joint_epochs is not None
        and total != schema_warmup_epochs + expected_joint_epochs
    ):
        raise ValueError(
            f"Training log reports {total} input epochs, expected "
            f"{schema_warmup_epochs + expected_joint_epochs}"
        )
    if len(records) != total or [row["epoch"] for row in records] != list(
        range(1, total + 1)
    ):
        raise ValueError(
            f"Training log has incomplete input epochs: found {len(records)}, expected {total}"
        )
    if any(row["total"] != total for row in records):
        raise ValueError("Training log reports inconsistent total input epochs")

    for row in records:
        warmup = row["epoch"] <= schema_warmup_epochs
        expected_phase = "schema_warmup" if warmup else "joint"
        if row["phase"] != expected_phase:
            raise ValueError(
                f"Input epoch {row['epoch']} has phase {row['phase']}, "
                f"expected {expected_phase}"
            )
        if warmup and any(row[name] != 0.0 for name in ("execution", "distill", "trajectory")):
            raise ValueError(f"Warmup losses leaked at input epoch {row['epoch']}")
        if schema_loss_weight > 0 and row["schema"] <= 0.0:
            raise ValueError(f"Schema loss is inactive at input epoch {row['epoch']}")
        if not warmup:
            if row["execution"] <= 0.0:
                raise ValueError(f"Execution loss is inactive at joint epoch {row['epoch']}")
            if distill_loss_weight > 0 and row["distill"] <= 0.0:
                raise ValueError(f"Distill loss is inactive at joint epoch {row['epoch']}")
            if trajectory_loss_weight > 0 and row["trajectory"] <= 0.0:
                raise ValueError(f"Trajectory loss is inactive at joint epoch {row['epoch']}")

    joint_records = records[schema_warmup_epochs:]
    if trajectory_loss_weight > 0 and not joint_records:
        raise ValueError("Trajectory loss is configured but the log has no joint epochs")
    return {
        "epochs": total,
        "schema_warmup_epochs": schema_warmup_epochs,
        "joint_epochs": len(joint_records),
        "finite_losses": True,
        "warmup_auxiliary_losses_zero": True,
        "joint_trajectory_loss_active": bool(
            trajectory_loss_weight > 0
            and joint_records
            and all(row["trajectory"] > 0 for row in joint_records)
        ),
        "first_joint_trajectory_loss": (
            joint_records[0]["trajectory"] if joint_records else None
        ),
        "last_joint_trajectory_loss": (
            joint_records[-1]["trajectory"] if joint_records else None
        ),
    }


def compare_bidirectional(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    expected_config_differences: set[str],
) -> dict[str, Any]:
    if baseline.get("split") != candidate.get("split"):
        raise ValueError("Baseline and candidate data splits differ")
    validate_audits(baseline, "baseline")
    validate_audits(candidate, "candidate")

    baseline_config = baseline.get("config")
    candidate_config = candidate.get("config")
    if not isinstance(baseline_config, dict) or not isinstance(candidate_config, dict):
        raise ValueError("Both results must contain config objects")
    config_keys = set(baseline_config) | set(candidate_config)
    actual_differences = {
        key
        for key in config_keys
        if key != "output_dir" and baseline_config.get(key) != candidate_config.get(key)
    }
    if actual_differences != expected_config_differences:
        raise ValueError(
            "Unexpected config differences: "
            f"expected {sorted(expected_config_differences)}, "
            f"found {sorted(actual_differences)}"
        )

    metrics: dict[str, dict[str, float]] = {}
    for name, path in METRICS.items():
        left = float(nested_value(baseline, path))
        right = float(nested_value(candidate, path))
        metrics[name] = {
            "baseline": left,
            "candidate": right,
            "delta": right - left,
        }

    comparison = {
        "split": baseline["split"],
        "audits_passed": True,
        "config_differences": {
            key: {
                "baseline": baseline_config.get(key),
                "candidate": candidate_config.get(key),
            }
            for key in sorted(actual_differences)
        },
        "metrics": metrics,
    }
    comparison["gates"] = evaluate_gates(metrics)
    if expected_config_differences == {"input_memory_source"}:
        if (
            baseline_config.get("input_memory_source") != "pooled"
            or candidate_config.get("input_memory_source") != "token_resampler"
        ):
            raise ValueError(
                "Token-resampler comparison must run from pooled to token_resampler"
            )
        comparison["gates"]["token_resampler_advancement"] = (
            evaluate_token_resampler_gate(metrics)
        )
    if expected_config_differences == {"input_memory_slots"}:
        if (
            baseline_config.get("input_memory_slots") != 8
            or candidate_config.get("input_memory_slots") != 32
        ):
            raise ValueError("Memory-capacity comparison must run from 8 to 32 slots")
        comparison["gates"]["memory_capacity_advancement"] = (
            evaluate_memory_capacity_gate(metrics)
        )
    if expected_config_differences == {"input_memory_interface"}:
        baseline_interface = baseline_config.get(
            "input_memory_interface", "prompt_slots"
        )
        candidate_interface = candidate_config.get("input_memory_interface")
        if baseline_interface != "prompt_slots" or candidate_interface not in {
            "readout_cross_attention",
            "gated_layerwise_cross_attention",
            "prompt_slots_plus_layerwise",
            "prompt_slots_plus_trajectory_layerwise",
        }:
            raise ValueError(
                "Interface comparison must run from prompt_slots to a persistent interface"
            )
    if (
        expected_config_differences == {"input_memory_interface"}
        and candidate_config.get("input_memory_interface")
        == "readout_cross_attention"
    ):
        baseline_interface = baseline_config.get(
            "input_memory_interface", "prompt_slots"
        )
        candidate_interface = candidate_config.get("input_memory_interface")
        if (
            baseline_interface != "prompt_slots"
            or candidate_interface != "readout_cross_attention"
        ):
            raise ValueError(
                "Readout-interface comparison must run from prompt_slots "
                "to readout_cross_attention"
            )
        interface_audit = candidate.get("input_interface_audit")
        if not isinstance(interface_audit, dict):
            raise ValueError("Readout candidate has no input_interface_audit")
        audit_values = (
            interface_audit.get("readout_adapter_nonzero_gradient_steps"),
            interface_audit.get("readout_adapter_max_gradient_norm"),
            interface_audit.get("readout_output_max_change"),
        )
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in audit_values
        ):
            raise ValueError(f"Readout candidate failed interface audit: {interface_audit}")
        controls_reproduced = all(
            baseline.get(section, {}).get(condition)
            == candidate.get(section, {}).get(condition)
            for section in (
                "input_readback_teacher_forced",
                "schema_readback_teacher_forced",
                "input_readback_generation",
            )
            for condition in ("query_only", "full_document")
        )
        comparison["gates"]["readout_interface_advancement"] = (
            evaluate_readout_interface_gate(metrics, controls_reproduced)
        )
        comparison["input_interface_audit"] = interface_audit
    if (
        expected_config_differences
        in (
            LAYERWISE_INTERFACE_CONFIG_DIFFERENCES,
            {"input_memory_interface"},
        )
        and candidate_config.get("input_memory_interface")
        in {
            "gated_layerwise_cross_attention",
            "prompt_slots_plus_layerwise",
            "prompt_slots_plus_trajectory_layerwise",
        }
    ):
        baseline_interface = baseline_config.get(
            "input_memory_interface", "prompt_slots"
        )
        candidate_interface = candidate_config.get("input_memory_interface")
        if (
            baseline_interface != "prompt_slots"
            or candidate_interface
            not in {
                "gated_layerwise_cross_attention",
                "prompt_slots_plus_layerwise",
                "prompt_slots_plus_trajectory_layerwise",
            }
        ):
            raise ValueError(
                "Layerwise-interface comparison must run from prompt_slots "
                "to gated_layerwise_cross_attention"
            )
        interface_audit = candidate.get("input_interface_audit")
        if not isinstance(interface_audit, dict):
            raise ValueError("Layerwise candidate has no input_interface_audit")
        initial_gates = interface_audit.get("layerwise_initial_gates")
        final_gates = interface_audit.get("layerwise_final_gates")
        max_abs_gate = interface_audit.get("layerwise_max_abs_gate")
        gradient_steps = interface_audit.get(
            "readout_adapter_nonzero_gradient_steps"
        )
        gradient_norm = interface_audit.get("readout_adapter_max_gradient_norm")
        configured_max = candidate_config.get("layerwise_memory_max_gate")
        valid_gates = (
            isinstance(initial_gates, dict)
            and bool(initial_gates)
            and all(float(value) == 0.0 for value in initial_gates.values())
            and isinstance(final_gates, dict)
            and set(final_gates) == set(initial_gates)
            and all(math.isfinite(float(value)) for value in final_gates.values())
            and isinstance(max_abs_gate, (int, float))
            and math.isfinite(float(max_abs_gate))
            and float(max_abs_gate) > 0.0
            and isinstance(configured_max, (int, float))
            and float(max_abs_gate) <= float(configured_max) + 1e-12
        )
        valid_gradients = all(
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in (gradient_steps, gradient_norm)
        )
        if not valid_gates or not valid_gradients:
            raise ValueError(
                f"Layerwise candidate failed interface audit: {interface_audit}"
            )
        controls_reproduced = all(
            baseline.get(section, {}).get(condition)
            == candidate.get(section, {}).get(condition)
            for section in (
                "input_readback_teacher_forced",
                "schema_readback_teacher_forced",
                "input_readback_generation",
            )
            for condition in ("query_only", "full_document")
        )
        comparison["gates"]["layerwise_interface_advancement"] = (
            evaluate_readout_interface_gate(metrics, controls_reproduced)
        )
        comparison["input_interface_audit"] = interface_audit
    return comparison


def evaluate_gates(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    baseline_trajectory_nll = metrics["trajectory_nll"]["baseline"]
    candidate_trajectory_nll = metrics["trajectory_nll"]["candidate"]
    relative_nll_improvement = (
        (baseline_trajectory_nll - candidate_trajectory_nll)
        / baseline_trajectory_nll
        if baseline_trajectory_nll > 0
        else float("-inf")
    )
    advancement_checks = {
        "trajectory_nll_relative_improvement_at_least_15pct": (
            relative_nll_improvement >= 0.15
        ),
        "trajectory_json_valid_at_least_90pct": (
            metrics["trajectory_json_valid"]["candidate"] >= 0.90
        ),
        "trajectory_conditional_exact_at_least_20pct": (
            metrics["trajectory_readback_given_correct_selection"]["candidate"]
            >= 0.20
        ),
        "registered_exact_arguments_drop_at_most_2pt": (
            metrics["registered_exact_arguments"]["delta"] >= -0.02
        ),
        "registered_key_recall_drop_at_most_2pt": (
            metrics["registered_key_recall"]["delta"] >= -0.02
        ),
    }
    semantic_prefix_checks = {
        "trajectory_conditional_exact_at_least_30pct": (
            metrics["trajectory_readback_given_correct_selection"]["candidate"]
            >= 0.30
        ),
        "registered_exact_schema_at_least_30pct": (
            metrics["registered_exact_schema"]["candidate"] >= 0.30
        ),
        "registered_key_recall_at_least_60pct": (
            metrics["registered_key_recall"]["candidate"] >= 0.60
        ),
    }
    return {
        "trajectory_loss_advancement": {
            "passed": all(advancement_checks.values()),
            "trajectory_nll_relative_improvement": relative_nll_improvement,
            "checks": advancement_checks,
        },
        "semantic_prefix_readiness": {
            "passed": all(semantic_prefix_checks.values()),
            "checks": semantic_prefix_checks,
        },
    }


def evaluate_token_resampler_gate(
    metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    def relative_nll_improvement(metric: str) -> float:
        baseline = metrics[metric]["baseline"]
        candidate = metrics[metric]["candidate"]
        return (baseline - candidate) / baseline if baseline > 0 else float("-inf")

    argument_nll_improvement = relative_nll_improvement("registered_argument_nll")
    schema_nll_improvement = relative_nll_improvement("registered_schema_nll")
    tolerance = 1e-12
    checks = {
        "registered_argument_nll_improves_at_least_10pct": (
            argument_nll_improvement >= 0.10 - tolerance
        ),
        "registered_schema_nll_improves_at_least_10pct": (
            schema_nll_improvement >= 0.10 - tolerance
        ),
        "registered_exact_arguments_drop_at_most_2pt": (
            metrics["registered_exact_arguments"]["delta"] >= -0.02 - tolerance
        ),
        "registered_exact_schema_improves_at_least_3pt": (
            metrics["registered_exact_schema"]["delta"] >= 0.03 - tolerance
        ),
        "registered_key_recall_improves_at_least_5pt": (
            metrics["registered_key_recall"]["delta"] >= 0.05 - tolerance
        ),
        "trajectory_json_valid_at_least_90pct": (
            metrics["trajectory_json_valid"]["candidate"] >= 0.90
        ),
        "trajectory_conditional_exact_at_least_20pct": (
            metrics["trajectory_readback_given_correct_selection"]["candidate"]
            >= 0.20
        ),
    }
    return {
        "passed": all(checks.values()),
        "registered_argument_nll_relative_improvement": argument_nll_improvement,
        "registered_schema_nll_relative_improvement": schema_nll_improvement,
        "checks": checks,
    }


def evaluate_memory_capacity_gate(
    metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    tolerance = 1e-12

    def relative_nll_improvement(metric_name: str) -> float:
        baseline = metrics[metric_name]["baseline"]
        candidate = metrics[metric_name]["candidate"]
        return (baseline - candidate) / baseline if baseline > 0 else float("-inf")

    argument_nll_improvement = relative_nll_improvement("registered_argument_nll")
    schema_nll_improvement = relative_nll_improvement("registered_schema_nll")
    checks = {
        "registered_argument_nll_improves_at_least_10pct": (
            argument_nll_improvement >= 0.10 - tolerance
        ),
        "registered_schema_nll_improves_at_least_10pct": (
            schema_nll_improvement >= 0.10 - tolerance
        ),
        "registered_exact_arguments_drop_at_most_2pt": (
            metrics["registered_exact_arguments"]["delta"] >= -0.02 - tolerance
        ),
        "registered_exact_schema_improves_at_least_5pt": (
            metrics["registered_exact_schema"]["delta"] >= 0.05 - tolerance
        ),
        "registered_key_recall_improves_at_least_10pt": (
            metrics["registered_key_recall"]["delta"] >= 0.10 - tolerance
        ),
        "trajectory_json_valid_at_least_90pct": (
            metrics["trajectory_json_valid"]["candidate"] >= 0.90
        ),
        "trajectory_conditional_exact_at_least_15pct": (
            metrics["trajectory_readback_given_correct_selection"]["candidate"] >= 0.15
        ),
    }
    return {
        "passed": all(checks.values()),
        "registered_argument_nll_relative_improvement": argument_nll_improvement,
        "registered_schema_nll_relative_improvement": schema_nll_improvement,
        "checks": checks,
    }


def evaluate_readout_interface_gate(
    metrics: dict[str, dict[str, float]],
    controls_reproduced_exactly: bool,
) -> dict[str, Any]:
    tolerance = 1e-12
    argument_baseline = metrics["registered_argument_nll"]["baseline"]
    schema_baseline = metrics["registered_schema_nll"]["baseline"]
    argument_improvement = (
        (argument_baseline - metrics["registered_argument_nll"]["candidate"])
        / argument_baseline
        if argument_baseline > 0
        else float("-inf")
    )
    schema_improvement = (
        (schema_baseline - metrics["registered_schema_nll"]["candidate"])
        / schema_baseline
        if schema_baseline > 0
        else float("-inf")
    )
    checks = {
        "registered_argument_nll_improves_at_least_10pct": (
            argument_improvement >= 0.10 - tolerance
        ),
        "registered_schema_nll_improves_at_least_10pct": (
            schema_improvement >= 0.10 - tolerance
        ),
        "registered_exact_arguments_drop_at_most_2pt": (
            metrics["registered_exact_arguments"]["delta"] >= -0.02 - tolerance
        ),
        "registered_exact_schema_improves_at_least_5pt": (
            metrics["registered_exact_schema"]["delta"] >= 0.05 - tolerance
        ),
        "registered_key_recall_improves_at_least_10pt": (
            metrics["registered_key_recall"]["delta"] >= 0.10 - tolerance
        ),
        "trajectory_json_valid_at_least_90pct": (
            metrics["trajectory_json_valid"]["candidate"] >= 0.90
        ),
        "trajectory_conditional_exact_at_least_20pct": (
            metrics["trajectory_readback_given_correct_selection"]["candidate"]
            >= 0.20
        ),
        "query_and_full_document_controls_reproduce_exactly": (
            controls_reproduced_exactly
        ),
    }
    return {
        "passed": all(checks.values()),
        "registered_argument_nll_relative_improvement": argument_improvement,
        "registered_schema_nll_relative_improvement": schema_improvement,
        "checks": checks,
    }


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly compare two bidirectional registration runs."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--expected-config-difference",
        action="append",
        default=[],
        help="Config key expected to differ; repeat for multiple keys.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--candidate-log",
        type=Path,
        help="Completed stdout log to audit curriculum and trajectory activation.",
    )
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--candidate-predictions", type=Path)
    args = parser.parse_args()
    candidate = load_result(args.candidate)
    comparison = compare_bidirectional(
        load_result(args.baseline),
        candidate,
        set(args.expected_config_difference),
    )
    if args.candidate_log is not None:
        config = candidate["config"]
        comparison["candidate_training_log"] = validate_training_log(
            args.candidate_log.read_text(encoding="utf-8"),
            int(config.get("schema_warmup_epochs", 0)),
            float(config.get("trajectory_loss_weight", 0.0)),
            float(config.get("schema_loss_weight", 0.0)),
            float(config.get("distill_loss_weight", 0.0)),
            int(config.get("input_epochs", 0)),
        )
    if (args.baseline_predictions is None) != (args.candidate_predictions is None):
        parser.error(
            "--baseline-predictions and --candidate-predictions must be provided together"
        )
    if args.baseline_predictions is not None and args.candidate_predictions is not None:
        comparison["paired_predictions"] = compare_predictions(
            load_predictions(args.baseline_predictions),
            load_predictions(args.candidate_predictions),
        )
    rendered = json.dumps(comparison, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
