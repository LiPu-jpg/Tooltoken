"""Train-only, auditable curriculum records. Reference actions are not truth labels.

No test data, learned checkpoints, teacher API, or simulated observation is read.
Unsupported argument/answer targets receive masks; original records are retained.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .toolbench_data import FINISH, compact, load_training_steps, load_training_tools
from .toolbench_checkpoint import sha256

CONTROL = {"tool": "<tool_request>", "final": "<final>", "give_up": "<give_up>"}


def clean_history(history, history_format="legacy"):
    # Old thoughts often describe a different next action; do not replay them.
    if history_format not in {"legacy", "observation_envelope_v1"}:
        raise ValueError("Unknown causal history format")
    result = [{k: copy.deepcopy(v) for k, v in item.items() if k != "thought"}
              for item in history if item.get("type") != "validation_error"]
    if history_format == "observation_envelope_v1":
        for item in result:
            if item.get("type") == "observation" and isinstance(item.get("content"), str):
                try:
                    value = strict_envelope(item["content"])
                except ValueError:
                    continue
                if isinstance(value, dict) and "error" in value and "response" in value:
                    item["content"] = value
    return result


def strict_envelope(text):
    # Normalize only the transport envelope, never recursively reinterpret
    # response strings or execute Python representations from tool results.
    from .toolbench_data import strict_json
    return strict_json(text)


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).casefold().strip()


def visible_values(history):
    # A previous model's guessed argument is NOT independent evidence.
    return "\n".join(str(item.get("content")) if isinstance(item.get("content"), str)
                     else compact(item.get("content")) for item in history
                     if item.get("type") in {"user", "observation"})


def argument_evidence(arguments, schema, history):
    missing = []
    text = normalize(visible_values(history))
    def walk(value, rule, path):
        if isinstance(value, dict):
            for key, child in value.items():
                properties = rule.get("properties", {})
                sub = properties.get(key)
                if sub is None:
                    patterns = [v for p, v in rule.get("patternProperties", {}).items() if re.search(p, key)]
                    sub = patterns[0] if patterns else rule.get("additionalProperties")
                    if not isinstance(sub, dict):
                        missing.append({"path": path + [key], "reason": "undeclared_field_not_certified"})
                        sub = {}
                walk(child, sub, path + [key])
        elif isinstance(value, list):
            for i, child in enumerate(value):
                rule_item = rule.get("items", {})
                rule_item = rule_item[i] if isinstance(rule_item, list) and i < len(rule_item) else rule_item
                walk(child, rule_item if isinstance(rule_item, dict) else {}, path + [i])
        else:
            literal = normalize(value)
            # Schema permits these values; this does not prove they fit the task.
            permitted = value in rule.get("enum", []) or ("const" in rule and value == rule["const"])
            present = bool(literal) and literal in text
            if not present and not permitted:
                missing.append({"path": path, "reason": "value_source_not_certified"})
    walk(arguments, schema, [])
    return {"eligible": not missing, "unresolved": missing,
            "scope": "schema and literal/schema-constraint provenance; not task-success certification"}


def schema_constraints(value):
    """Remove schema annotations without deleting user fields or enum values."""
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if key in {'description', 'x-toolbench-original-name', 'x-toolbench-original-type'}:
            continue
        if key in {'properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'}:
            out[key] = {name: schema_constraints(rule) for name, rule in item.items()}
        elif key in {'items', 'additionalProperties', 'contains', 'not', 'if', 'then', 'else', 'propertyNames'}:
            out[key] = [schema_constraints(rule) for rule in item] if isinstance(item, list) else schema_constraints(item)
        elif key in {'allOf', 'anyOf', 'oneOf', 'prefixItems'}:
            out[key] = [schema_constraints(rule) for rule in item]
        else:
            out[key] = copy.deepcopy(item)
    return out


def information_targets(tool):
    """Only nonempty facts provided by the actual document, including nested fields."""
    targets = []
    def visit(rule, path):
        for name, child in rule.get("properties", {}).items():
            info = {k: v for k, v in child.items() if k not in {"properties", "x-toolbench-original-name", "x-toolbench-original-type"}}
            if len(str(info.get("description", ""))) > 4096:
                info.pop("description", None)  # Long catalog remains in registration, not rote readback.
            targets.append({"path": path + [name], "required": name in rule.get("required", []), "rule": info})
            visit(child, path + [name])
        if isinstance(rule.get("items"), dict):
            visit(rule["items"], path + ["[]"])
    visit(tool.parameters, [])
    # Capability readback also covers genuinely parameterless APIs.
    # Full schema supplies nested/conditional rules; required and allowed keys are
    # taught jointly rather than relying exclusively on isolated field recall.
    signature = {"signature": {"declared_keys": sorted(tool.parameters.get("properties", {})),
        "required": tool.parameters.get("required", []), "schema": schema_constraints(tool.parameters),
        "fixed_top_level": tool.executable_contract is not None or tool.parameters.get("additionalProperties") is False}}
    # Entire oversized capability copy task is omitted; registration retains all text.
    capability = [{"capability": tool.document}] if len(tool.document.encode('utf-8')) <= 16000 else []
    return capability + targets + [signature]


def corrupt_arguments(arguments, tool, offset=0):
    """Same-API supervised repair, only for corruption rejected by the real schema."""
    try:
        tool.validate_arguments(arguments)
    except ValueError:
        return None  # Never synthesize a repair whose target itself is invalid.
    candidates = []
    extra = "__unexpected_input__"
    while extra in tool.parameters.get("properties", {}) or extra in arguments:
        extra += "_"
    changed = copy.deepcopy(arguments)
    changed[extra] = "unsupported"
    try:
        tool.validate_arguments(changed)
    except ValueError as exc:
        candidates.append({"previous": compact(changed), "error": str(exc), "kind": "extra",
                           "changed_path": [extra], "target": copy.deepcopy(arguments)})
    def visit(value, path):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            for kind in ("delete", "type"):
                changed = copy.deepcopy(arguments)
                cursor = changed
                for part in path:
                    cursor = cursor[part]
                if kind == "delete":
                    del cursor[key]
                else:
                    cursor[key] = {"invalid_type": True} if not isinstance(child, dict) else "invalid_type"
                try:
                    tool.validate_arguments(changed)
                except ValueError as exc:
                    candidates.append({"previous": compact(changed), "error": str(exc), "kind": kind,
                                       "changed_path": path + [key], "target": copy.deepcopy(arguments)})
            visit(child, path + [key])
    visit(arguments, [])
    return candidates[offset % len(candidates)] if candidates else None


def hard_negative_index(tools, count=32):
    """Deterministic lexical candidates; source docs only, never held-out qrels."""
    ids = sorted(set(tools) - {FINISH})
    words = {key: set(re.findall(r"[a-z0-9]+", tools[key].registration_document.casefold())) for key in ids}
    postings = {}
    for key in ids:
        for word in words[key]:
            postings.setdefault(word, []).append(key)
    hashes = {key: tools[key].document_hash for key in ids}
    result = {}
    for key in ids:
        score = Counter()
        for word in words[key]:
            members = postings[word]
            if len(members) > len(ids) // 3:
                continue
            for other in members:
                score[other] += 1 / len(members)
        ranked = sorted(score, key=lambda other: (-score[other], other))
        result[key] = [other for other in ranked if other != key
                       and hashes[other] != hashes[key]][:count]
    return result


def build_records(steps, tools):
    ordinary = sorted(set(tools) - {FINISH})
    records, counts = [], Counter()
    for index, step in enumerate(steps):
        history = clean_history(step.history)
        record = {"id": [step.source_id, step.step_index], "source_identity": step.source_id,
                  "api_identity": step.api_identity, "history": history,
                  "original_thought": step.thought, "reference_arguments": step.arguments,
                  "document_hash": tools[step.api_identity].document_hash,
                  "next_intent": "", "intent_supervision": False,
                  "label_status": "reference_trajectory_not_semantically_verified"}
        if step.api_identity != FINISH:
            tool = tools[step.api_identity]
            evidence = argument_evidence(step.arguments, tool.parameters, history)
            tool.validate_arguments(step.arguments)
            record.update(mode="tool", selected=step.api_identity, arguments=step.arguments,
                          masks={"control": 1., "selection": 1., "arguments": float(evidence["eligible"]), "answer": 0.},
                          argument_evidence=evidence, answer="", repair=corrupt_arguments(step.arguments, tool, index)
                          if evidence["eligible"] else None)
        else:
            mode = "give_up" if step.arguments["return_type"] == "give_up_and_restart" else "final"
            answer = step.arguments.get("final_answer", "")
            observations = [item for item in history if item.get("type") == "observation"]
            # Conservative terminal subset: an extractive answer is supported
            # by an actual observation, but even this does not certify completion.
            extractive = bool(observations) and len(answer.strip()) >= 4 and normalize(answer) in normalize(visible_values(observations))
            eligible = mode == "give_up" or extractive
            record.update(mode=mode, selected=ordinary[index % len(ordinary)], arguments={}, answer=answer if extractive else "",
                          masks={"control": float(eligible), "selection": 0., "arguments": 0., "answer": float(extractive)},
                          terminal_evidence={"extractive_observation_support": extractive,
                              "task_completion_certified": False, "reference_give_up": mode == "give_up"}, repair=None)
        record["synthetic"] = False
        records.append(record)
        counts["source_records"] += 1
        counts[record["mode"]] += 1
        for name, enabled in record["masks"].items():
            counts[name + "_supervised"] += int(enabled)
        counts["repair_supervised"] += record["repair"] is not None
        counts["empty_ordinary_arguments"] += int(record["mode"] == "tool" and not record["arguments"])
    return records, dict(counts)


def add_explicit_completion_examples(records):
    """A small, separately marked terminal curriculum with a verifiable objective.

This teaches ending/answer syntax, not general ToolBench task completion. It
never labels an unsupported original final answer as correct.
"""
    result = []
    seen = set()
    for row in records:
        observations = [item for item in row["history"] if item.get("type") == "observation"]
        if not observations or row["source_identity"] in seen:
            continue
        obs = observations[-1]
        content = obs["content"]
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except ValueError:
                continue
        if not isinstance(content, dict):
            continue
        fields = [(key, value) for key, value in sorted(content.items())
                  if isinstance(value, (str, int, float, bool)) and value is not None
                  and len(compact(value)) <= 160 and key.casefold() not in {"error", "message", "status", "code"}]
        if not fields:
            continue
        key, value = fields[0]
        seen.add(row["source_identity"])
        item = copy.deepcopy(row)
        item.update(id=[row["source_identity"], "explicit_completion"], synthetic=True,
                    history=[{"type": "user", "content": "The tool result has already been obtained. Return exactly the JSON value of its top-level field " + compact(key) + ". Do not call another tool."}, obs],
                    mode="final", answer=compact(value), original_thought="", arguments={}, repair=None,
                    masks={"control": 1., "selection": 0., "arguments": 0., "answer": 1.},
                    label_status="synthetic_extractive_completion_from_train_observation",
                    terminal_evidence={"constructed_objective": key, "exact_value": value})
        result.append(item)
        if len(result) >= 256:
            break
    return result


def candidate_ids(record, tools, hard, count, seed):
    gold = record["selected"]
    rng = random.Random(f"{seed}:{record['id']}")
    hard_pool = [x for x in hard.get(gold, []) if x != gold]
    selected = {gold, *rng.sample(hard_pool, min(len(hard_pool), (count - 1) // 2))}
    rest = sorted(set(tools) - {FINISH} - selected)
    selected.update(rng.sample(rest, min(len(rest), count - len(selected))))
    result = sorted(selected)
    rng.shuffle(result)
    return result


def epoch_order(records, stage, seed):
    """Full source coverage plus controlled nonempty-argument replay, no deletion."""
    result = [i for i, row in enumerate(records) if row["mode"] == "tool"] if stage == 1 else list(range(len(records)))
    empty = sum(records[i]["mode"] == "tool" and not records[i]["arguments"] for i in result)
    nonempty = [i for i in result if records[i]["mode"] == "tool" and records[i]["arguments"] and records[i]["masks"]["arguments"]]
    rng = random.Random(seed)
    if nonempty:
        result.extend(rng.choices(nonempty, k=max(0, empty - len(nonempty))))
    rng.shuffle(result)
    return result


def prepare(tools_path, trajectories_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    tools = load_training_tools(tools_path)
    steps = load_training_steps(trajectories_path, tools, source_format="toolgen")
    records, counts = build_records(steps, tools)
    synthetic = add_explicit_completion_examples(records)
    records += synthetic
    output.mkdir(parents=True)
    (output / "records.jsonl").write_text("".join(compact(row) + "\n" for row in records))
    (output / "hard-negatives.json").write_text(compact(hard_negative_index(tools)) + "\n")
    audit = {"source_sha256": {"tools": sha256(Path(tools_path)), "trajectories": sha256(Path(trajectories_path))},
             "train_api_identities": sorted(set(tools) - {FINISH}), "counts": counts,
             "synthetic_terminal_examples": len(synthetic), "records": len(records),
             "stage1_exposures_per_epoch": len(epoch_order(records, 1, 17)),
             "stage2_exposures_per_epoch": len(epoch_order(records, 2, 17)),
             "raw_thought_loss": 0, "test_files_opened": False, "stage3_rollouts_included": False,
             "semantic_completion_certified": False,
             "masked_labels_retained": True, "schema_defaults_fabricated": False}
    (output / "AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    manifest = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "READY.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tools", type=Path, required=True)
    p.add_argument("--trajectories", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = prepare(a.tools, a.trajectories, a.output)
    print(json.dumps({k: v for k, v in report.items() if k != "train_api_identities"}, indent=2))
