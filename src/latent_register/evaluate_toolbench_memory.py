"""Explicit development-only memory diagnostics; no training or external APIs.

Schema probes are query-free. Argument probes use oracle tool and recorded
past history, but a model-generated current plan fixed across conditions.
These are mechanism diagnostics, NOT real-retrieval end-to-end scores.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft7Validator

from .toolbench_checkpoint import load_agent, sha256
from .toolbench_data import FINISH, ToolSpec, compact, load_steps, load_tools, strict_json
from .toolbench_schema import SCHEMA_TASKS, canonical, schema_targets, target_facts

CONDITIONS = ("correct", "blank", "wrong")


def wrong_mapping(tools: dict[str, ToolSpec]) -> dict[str, str]:
    identities = sorted(set(tools) - {FINISH})
    contracts = {identity: canonical(tools[identity].parameters) for identity in identities}
    mapping = {}
    for index, identity in enumerate(identities):
        other = next((identities[(index + shift) % len(identities)] for shift in range(1, len(identities))
                      if contracts[identities[(index + shift) % len(identities)]] != contracts[identity]), None)
        if other is None:
            raise ValueError("Wrong-memory probe needs at least two different schema contracts")
        mapping[identity] = other
    return mapping


def schema_metrics(task: str, target: Any, text: str, truncated: bool) -> dict:
    expected = target_facts(task, target)
    try:
        if truncated:
            raise ValueError("Truncated generation")
        parsed = strict_json(text)
        predicted = target_facts(task, parsed)
        valid = True
    except (ValueError, TypeError):
        predicted, parsed, valid = set(), None, False
    matched = len(expected & predicted)
    return dict(exact=bool(valid and canonical(parsed) == canonical(target)), parsed=valid,
                expected_facts=len(expected), predicted_facts=len(predicted), matched_facts=matched,
                fact_recall=matched / len(expected) if expected else None,
                fact_precision=matched / len(predicted) if predicted else None)


def argument_metrics(tool: ToolSpec, target: dict, text: str, truncated: bool) -> dict:
    try:
        if truncated:
            raise ValueError("Truncated generation")
        parsed = strict_json(text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected object")
    except (ValueError, TypeError):
        return dict(exact=False, parsed=False, schema_valid=False, missing_required_count=None,
                    type_error_count=None, enum_const_error_count=None)
    errors = list(Draft7Validator(tool.parameters).iter_errors(parsed))
    leaves = []
    def visit(error: Any) -> None:
        leaves.append(error)
        for child in error.context:
            visit(child)
    for error in errors:
        visit(error)
    missing = set()
    for error in leaves:
        if error.validator == "required" and isinstance(error.instance, dict):
            missing.update((tuple(map(str, error.absolute_path)), field) for field in error.validator_value if field not in error.instance)
    return dict(exact=canonical(parsed) == canonical(target), parsed=True, schema_valid=not errors,
        missing_required_count=len(missing), type_error_count=sum(error.validator == "type" for error in leaves),
        enum_const_error_count=sum(error.validator in {"enum", "const"} for error in leaves))


def paired_api_interval(values: list[float], *, seed: int = 17, draws: int = 2000) -> dict:
    if not values:
        return {"apis": 0, "mean": None, "ci95": None}
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(values) < 2:
        return {"apis": len(values), "mean": mean, "ci95": None}
    rng = np.random.default_rng(seed)
    samples = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(draws)]
    return {"apis": len(values), "mean": mean, "ci95": np.quantile(samples, [.025, .975]).tolist()}


def summarize(records: list[dict]) -> dict:
    summary: dict = {"schema": {}, "arguments": {}, "paired_schema_fact_recall": {}}
    for kind in ("schema", "arguments"):
        for condition in CONDITIONS:
            group = [row for row in records if row["kind"] == kind and row["condition"] == condition]
            metrics = [row["metrics"] for row in group]
            item = dict(presentations=len(group), parsed=sum(m["parsed"] for m in metrics),
                        exact=sum(m["exact"] for m in metrics) / len(group) if group else None,
                        truncated=sum(row["truncated"] for row in group))
            if kind == "schema":
                item["per_task"] = {}
                for task in SCHEMA_TASKS:
                    task_metrics = [row["metrics"] for row in group if row["task"] == task]
                    expected = sum(m["expected_facts"] for m in task_metrics)
                    item["per_task"][task] = dict(presentations=len(task_metrics), expected_facts=expected,
                        nonempty_targets=sum(m["expected_facts"] > 0 for m in task_metrics),
                        fact_recall=sum(m["matched_facts"] for m in task_metrics) / expected if expected else None)
            else:
                parsed = [m for m in metrics if m["parsed"]]
                item.update(schema_valid=sum(m["schema_valid"] for m in metrics) / len(metrics) if metrics else None,
                    missing_required_rate_among_parsed=sum(m["missing_required_count"] > 0 for m in parsed) / len(parsed) if parsed else None,
                    type_error_rate_among_parsed=sum(m["type_error_count"] > 0 for m in parsed) / len(parsed) if parsed else None,
                    enum_const_error_rate_among_parsed=sum(m["enum_const_error_count"] > 0 for m in parsed) / len(parsed) if parsed else None)
            summary[kind][condition] = item
    panels: dict = defaultdict(dict)
    for row in records:
        if row["kind"] == "schema" and row["wrong_changes_target"] and row["metrics"]["expected_facts"]:
            panels[(row["api_identity"], row["task"])][row["condition"]] = row["metrics"]["fact_recall"]
    for other in ("blank", "wrong"):
        api_values: dict = defaultdict(list)
        for (identity, _), conditions in panels.items():
            if set(conditions) != set(CONDITIONS):
                raise ValueError("Incomplete paired memory conditions")
            api_values[identity].append(conditions["correct"] - conditions[other])
        summary["paired_schema_fact_recall"]["correct_minus_" + other] = paired_api_interval(
            [float(np.mean(items)) for _, items in sorted(api_values.items())])
    intervals = list(summary["paired_schema_fact_recall"].values())
    summary["content_signal"] = ("positive_on_this_development_checkpoint" if intervals and
        all(item["ci95"] is not None and item["ci95"][0] > 0 for item in intervals)
        else "no_positive_signal_or_insufficient_evidence")
    summary["pairing_scope"] = "API-clustered, paired conditions; nonempty facts whose target changes under wrong memory; single checkpoint"
    return summary


def evaluate(agent: Any, tools: dict[str, ToolSpec], steps: list, *, max_new_tokens: int = 512,
             max_thought_tokens: int = 128) -> tuple[list[dict], dict]:
    if agent.condition != "memory":
        raise ValueError("Content perturbations require a memory-trained checkpoint")
    mapping = wrong_mapping(tools)
    agent.register_tools(list(tools.values()), profile=True)
    records: list[dict] = []
    def generation(history, task, thought, identity, condition):
        own = agent._registry[identity][2]
        memory = own if condition == "correct" else own.new_zeros(own.shape) if condition == "blank" else agent._registry[mapping[identity]][2]
        prefix = agent.prefix(history, task, thought, memory=memory)
        start = agent._clock(True)
        generated = agent.generate(prefix, max_new_tokens=max_new_tokens)
        elapsed = agent._clock(True) - start
        return generated, dict(input_positions=prefix.shape[1], memory_positions=memory.shape[0],
            generated_tokens=len(generated.token_ids) + int(generated.stopped_on_eos), generation_seconds=elapsed)
    for identity in sorted(mapping):
        targets = schema_targets(tools[identity].parameters)
        alternate = schema_targets(tools[mapping[identity]].parameters)
        for task, target in targets.items():
            for condition in CONDITIONS:
                generated, cost = generation([], task, "", identity, condition)
                records.append(dict(kind="schema", api_identity=identity, task=task, condition=condition,
                    wrong_api_identity=mapping[identity], wrong_changes_target=canonical(target) != canonical(alternate[task]),
                    text=generated.text, target=target, truncated=generated.truncated, cost=cost,
                    metrics=schema_metrics(task, target, generated.text, generated.truncated)))
    for step in steps:
        if step.api_identity == FINISH:
            continue
        prefix = agent.prefix(step.history, "thought")
        start = agent._clock(True)
        plan = agent.generate(prefix, max_new_tokens=max_thought_tokens)
        plan_cost = dict(input_positions=prefix.shape[1], generated_tokens=len(plan.token_ids) + int(plan.stopped_on_eos),
                         generation_seconds=agent._clock(True) - start)
        if plan.truncated:
            # Keep failed planning in the denominator, with three failed paired
            # outcomes. Do not silently replace it with the gold thought.
            for condition in CONDITIONS:
                records.append(dict(kind="arguments", api_identity=step.api_identity, source_id=step.source_id,
                    step_index=step.step_index, condition=condition, text="", target=step.arguments,
                    truncated=True, planning_failed=True, planning_cost=plan_cost, cost=None,
                    metrics=argument_metrics(tools[step.api_identity], step.arguments, "", True)))
            continue
        for condition in CONDITIONS:
            generated, cost = generation(step.history, "arguments", plan.text, step.api_identity, condition)
            records.append(dict(kind="arguments", api_identity=step.api_identity, source_id=step.source_id,
                step_index=step.step_index, condition=condition, wrong_api_identity=mapping[step.api_identity],
                text=generated.text, target=step.arguments, truncated=generated.truncated,
                planning_failed=False, planning_cost=plan_cost, fixed_generated_plan=plan.text, cost=cost,
                metrics=argument_metrics(tools[step.api_identity], step.arguments, generated.text, generated.truncated)))
    report = summarize(records)
    report["registration"] = agent.registration_profiles
    report["wrong_mapping"] = mapping
    report["wrong_mapping_note"] = "Deterministic different-schema assignment, not necessarily bijective"
    report["scope"] = "query-free schema and oracle-tool/oracle-past-history argument diagnostics; not end-to-end execution"
    report["cost_note"] = "Registration is one-time per API; each argument plan is shared across three conditions, count it once"
    return records, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tools", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, help="Optional explicit development-only argument trajectories")
    p.add_argument("--source-format", choices=["toolbench", "toolgen"], default="toolbench")
    p.add_argument("--split", choices=["dev", "validation"], required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-thought-tokens", type=int, default=128)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Diagnostic output must be new")
    tools = load_tools(args.tools, split=args.split)
    config = json.loads((args.checkpoint / "agent.json").read_text())
    trained = config.get("metadata", {}).get("train_api_identities")
    if not isinstance(trained, list):
        raise ValueError("Checkpoint lacks training API lineage; cannot certify unseen development tools")
    if (set(tools) - {FINISH}) & set(trained):
        raise ValueError("Development APIs overlap checkpoint training APIs")
    wrong_mapping(tools)
    steps = load_steps(args.trajectories, tools, source_format=args.source_format, split=args.split) if args.trajectories else []
    agent = load_agent(args.checkpoint, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        records, report = evaluate(agent, tools, steps, max_new_tokens=args.max_new_tokens, max_thought_tokens=args.max_thought_tokens)
        report.update(checkpoint_manifest_sha256=sha256(args.checkpoint / "SHA256.json"), tools_sha256=sha256(args.tools),
            trajectories_sha256=sha256(args.trajectories) if args.trajectories else None,
            optimizer_updates=0, split=args.split, tested_api_identities=sorted(set(tools) - {FINISH}),
            generation_limits=dict(max_new_tokens=args.max_new_tokens, max_thought_tokens=args.max_thought_tokens))
        (args.output_dir / "outputs.jsonl").write_text("".join(compact(row) + "\n" for row in records))
        (args.output_dir / "REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    except Exception as exc:
        (args.output_dir / "FAILED.json").write_text(json.dumps({"error": str(exc), "status": "incomplete"}) + "\n")
        raise


if __name__ == "__main__":
    main()
