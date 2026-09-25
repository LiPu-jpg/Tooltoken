"""Train-split, snapshot-generated conditions for the second training stage.

Each attempt starts at a labeled causal training history. The actor sees only
that history and the full training registry, never the next gold action. We do
not execute arbitrary calls or attach reference observations to a wrong call.
This is supervised self-conditioning, not policy-gradient RL or a live rollout.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .toolbench_agent import GenerationFailure
from .toolbench_checkpoint import sha256
from .toolbench_data import FINISH, compact, strict_json

VERSION = 1
SCOPE = "train_history_anchored_self_conditioning"
STATUSES = {"valid", "invalid_arguments", "truncated_thought", "truncated_arguments", "context_error"}


def fingerprint(value) -> str:
    return hashlib.sha256(compact(value).encode()).hexdigest()


def step_key(step) -> tuple[str, int]:
    return step.source_id, step.step_index


def shard_for(step, shards: int) -> int:
    return int(fingerprint(list(step_key(step))), 16) % shards


def feedback(attempt: dict, retry: int) -> dict:
    return {"type": "validation_error", "retry": retry,
            "api_identity": attempt["selected_identity"], "thought": attempt["thought"],
            "generated_arguments": attempt["argument_text"], "error": attempt["error"]}


def collect_attempts(agent, history, tools, *, thought_tokens=1024, argument_tokens=1024,
                     validation_retries=2) -> list[dict]:
    """The actor has no gold action/arguments/next-observation argument."""
    if type(validation_retries) is not int or validation_retries < 0:
        raise ValueError("Invalid retry count")
    if min(thought_tokens, argument_tokens) < 1:
        raise ValueError("Generation budgets must be positive")
    current = json.loads(compact(history))
    attempts = []
    for index in range(validation_retries + 1):
        item = dict(history=json.loads(compact(current)), thought=None, selected_identity=None,
                    argument_text=None, error=None, status="context_error")
        try:
            decision = agent.decide(current, tools, max_thought_tokens=thought_tokens,
                                    max_argument_tokens=argument_tokens)
            item.update(status="valid", thought=decision.thought,
                        selected_identity=decision.api_identity, argument_text=compact(decision.arguments))
        except GenerationFailure as exc:
            details = exc.details
            status = ("invalid_arguments" if details.get("reason") == "invalid_arguments"
                      else "truncated_" + details["stage"])
            if status not in STATUSES:
                raise ValueError("Unsupported actor failure") from exc
            item.update(status=status, error=str(exc),
                        thought=details.get("thought") if details["stage"] != "thought" else details["generated_text"],
                        selected_identity=details.get("api_identity"),
                        argument_text=details["generated_text"] if details["stage"] == "arguments" else None)
        except ValueError as exc:
            # Only the known bounded-context errors are sample failures. Broken
            # checkpoints, identity bindings and infrastructure errors propagate.
            if not str(exc).startswith(("Prompt including memory exceeds context limit:",
                                        "No generation positions remain")):
                raise
            item["error"] = str(exc)
        attempts.append(item)
        if item["status"] != "invalid_arguments" or index == validation_retries:
            break
        current.append(feedback(item, index + 1))
    return attempts


def make_record(step, attempts):
    return dict(source_id=step.source_id, step_index=step.step_index,
                history_sha256=fingerprint(step.history), attempts=attempts)


def validate_record(record, step, tools, *, validation_retries):
    if set(record) != {"source_id", "step_index", "history_sha256", "attempts"}:
        raise ValueError("Unexpected self-conditioning record fields")
    if (record["source_id"], record["step_index"]) != step_key(step):
        raise ValueError("Training step identity mismatch")
    if record["history_sha256"] != fingerprint(step.history):
        raise ValueError("Training history changed")
    attempts = record["attempts"]
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= validation_retries + 1:
        raise ValueError("Invalid number of actor attempts")
    history = json.loads(compact(step.history))
    for index, item in enumerate(attempts):
        if set(item) != {"history", "thought", "selected_identity", "argument_text", "error", "status"}:
            raise ValueError("Unexpected actor attempt fields")
        if compact(item["history"]) != compact(history):
            raise ValueError("Actor history contains an unbound observation or altered feedback")
        status = item["status"]
        if status not in STATUSES:
            raise ValueError("Invalid actor status")
        if status in {"valid", "invalid_arguments", "truncated_arguments"}:
            if (item["selected_identity"] not in tools or not isinstance(item["thought"], str)
                    or not isinstance(item["argument_text"], str)):
                raise ValueError("Incomplete generated condition or held-out selection")
        if status == "valid":
            tools[item["selected_identity"]].validate_arguments(strict_json(item["argument_text"]))
            if item["error"] is not None:
                raise ValueError("A valid attempt cannot carry a validation error")
        elif not isinstance(item["error"], str) or not item["error"]:
            raise ValueError("Failure evidence is missing")
        if status == "invalid_arguments":
            try:
                tools[item["selected_identity"]].validate_arguments(strict_json(item["argument_text"]))
            except ValueError as exc:
                if str(exc) != item["error"]:
                    raise ValueError("Validation feedback does not match actual schema/JSON error") from exc
            else:
                raise ValueError("Valid arguments were labeled invalid")
        if index + 1 < len(attempts):
            if status != "invalid_arguments":
                raise ValueError("Only local invalid-argument feedback may extend an anchored history")
            history.append(feedback(item, index + 1))
    return attempts


def usable_condition(item):
    return item["status"] in {"valid", "invalid_arguments", "truncated_arguments"}


def inspect_actor_source(checkpoint, tools_path, trajectories_path, tools):
    """Bind a rollout to the exact training corpus and parent, before loading 8B."""
    config = json.loads((checkpoint / "agent.json").read_text())
    manifest = json.loads((checkpoint / "SHA256.json").read_text())
    if manifest.get("agent.json") != sha256(checkpoint / "agent.json"):
        raise ValueError("Unauthenticated actor metadata")
    metadata = config["metadata"]
    source = {"tools": sha256(tools_path), "trajectories": sha256(trajectories_path)}
    if metadata.get("source_sha256") != source:
        raise ValueError("Actor source differs from the training corpus")
    if set(metadata.get("train_api_identities", [])) != set(tools) - {FINISH}:
        raise ValueError("Actor training API lineage differs from the registry")
    return source


def read_rollouts(directories, steps, tools, *, source_sha256, checkpoint, source_format,
                  allow_partial=False):
    """Authenticate completed shards and rebuild conditions against TRAIN labels."""
    by_key = {step_key(step): step for step in steps}
    conditions, counts, contracts = {}, Counter(), []
    attempt_slots = None
    expected_parent = sha256(checkpoint / "SHA256.json")
    for directory in directories:
        manifest = json.loads((directory / "MANIFEST.json").read_text())
        if set(manifest) != {"metadata.json", "attempts.jsonl", "REPORT.json"}:
            raise ValueError("Incomplete rollout artifact")
        for relative, expected in manifest.items():
            if sha256(directory / relative) != expected:
                raise ValueError("Rollout checksum mismatch")
        meta = json.loads((directory / "metadata.json").read_text())
        expected = dict(version=VERSION, scope=SCOPE, split="train", optimizer_updates=0,
                        source_sha256=source_sha256, checkpoint_manifest_sha256=expected_parent,
                        source_format=source_format, registry_identities=sorted(tools),
                        document_hashes={key: tool.document_hash for key, tool in tools.items()},
                        external_calls=0, reference_observations_replayed_after_generated_calls=False,
                        actor_source_sha256=sha256(Path(__file__).with_name("toolbench_agent.py")),
                        conditions_source_sha256=sha256(Path(__file__)))
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Self-conditioning contract mismatch: {key}")
        shards, shard = meta["num_shards"], meta["shard_index"]
        if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
            raise ValueError("Invalid rollout shard")
        retry_count = meta["validation_retries"]
        if type(retry_count) is not int or retry_count < 0:
            raise ValueError("Invalid rollout feedback budget")
        if attempt_slots is not None and attempt_slots != retry_count + 1:
            raise ValueError("All shards must share the same fixed attempt-slot schedule")
        attempt_slots = retry_count + 1
        actual_keys = set()
        with (directory / "attempts.jsonl").open() as handle:
            for line in handle:
                record = strict_json(line)
                key = (record["source_id"], record["step_index"])
                if key not in by_key or key in conditions:
                    raise ValueError("Unknown or duplicate rollout training step")
                step = by_key[key]
                if shard_for(step, shards) != shard:
                    raise ValueError("Record belongs to a different shard")
                attempts = validate_record(record, step, tools, validation_retries=retry_count)
                conditions[key] = attempts
                actual_keys.add(key)
                counts.update(item["status"] for item in attempts)
                counts["usable_conditions"] += sum(usable_condition(item) for item in attempts)
                counts["argument_supervision_conditions"] += sum(usable_condition(item)
                    and item["selected_identity"] == step.api_identity for item in attempts)
        expected_keys = [step_key(step) for step in steps if shard_for(step, shards) == shard]
        if meta.get("max_steps") is not None:
            if type(meta["max_steps"]) is not int or meta["max_steps"] < 1:
                raise ValueError("Invalid rollout step cap")
            expected_keys = expected_keys[:meta["max_steps"]]
        if actual_keys != set(expected_keys):
            raise ValueError("Rollout dropped or added records within its declared shard")
        contracts.append(dict(path=str(directory), manifest_sha256=sha256(directory / "MANIFEST.json"),
                              records=len(actual_keys), metadata=meta))
    missing = len(steps) - len(conditions)
    if missing and not allow_partial:
        raise ValueError(f"Missing self-conditioning coverage for {missing} training steps")
    if not conditions:
        raise ValueError("Empty self-conditioning dataset")
    if not counts["usable_conditions"]:
        raise ValueError("No usable generated plans/selections; refusing teacher-only training labeled Stage 2")
    return conditions, dict(scope=SCOPE, actor_checkpoint_manifest_sha256=expected_parent,
        train_steps=len(steps), covered_steps=len(conditions), uncovered_steps=missing,
        all_failures_retained=True, statuses=dict(counts), shards=contracts,
        attempt_slots=attempt_slots, distributed_schedule="fixed_compile_select_argument_forward_count",
        missing_steps_use_teacher_loss_only=bool(missing))


def self_conditioned_loss(agent, steps, tools, *, conditions, candidate_count, seed,
                          teacher_losses, teacher_weight, attempt_slots=3):
    """Recompile with current trainable weights; never train on cached latents.

    Sampled identities/tokens are discrete conditions, not differentiable paths.
    Gold selection is corrective supervision, not an actor input. Argument CE
    is gated on exact identity; no wrong tool receives another tool's arguments.
    """
    import random
    import torch
    from torch.nn import functional as F
    if len(conditions) != len(steps) or not 0 < teacher_weight <= 1 or candidate_count < 3:
        raise ValueError("Invalid second-stage batch/teacher weight/candidate count")
    if type(attempt_slots) is not int or attempt_slots < 1 or any(len(x) > attempt_slots for x in conditions):
        raise ValueError("Invalid fixed attempt-slot schedule")
    zero = teacher_losses["loss"] * 0
    selection, arguments = [], []
    usable = matched = failed = attempts_total = 0
    for step, attempts in zip(steps, conditions):
        sl, al = [], []
        active = [item for item in attempts if usable_condition(item)]
        # Every rank executes the same module traversal, even on an entirely
        # failed sample. ZeRO-3 cannot safely skip arbitrary backbone forwards.
        candidates = {step.api_identity, FINISH, *(item["selected_identity"] for item in active)}
        if not candidates.issubset(tools):
            raise ValueError("Self-selected API is outside the training registry")
        rng = random.Random(f"self:{seed}:{step.source_id}:{step.step_index}")
        others = sorted(set(tools) - candidates)
        candidates.update(rng.sample(others, min(len(others), max(0, candidate_count - len(candidates)))))
        candidates = sorted(candidates); rng.shuffle(candidates)
        rows, memory = agent.compile([tools[key] for key in candidates])
        gold_index = candidates.index(step.api_identity)
        for index in range(attempt_slots):
            item = attempts[index] if index < len(attempts) else None
            valid = item is not None and usable_condition(item)
            match = valid and item["selected_identity"] == step.api_identity
            attempts_total += int(item is not None)
            failed += int(item is not None and not valid)
            usable += int(valid); matched += int(match)
            history = item["history"] if valid else []
            thought = item["thought"] if valid else ""
            scores = agent.selection_scores(history, thought, rows)
            sl.append(F.cross_entropy(scores, agent._ids([gold_index])) * int(valid))
            # The zero-weight traversal uses GOLD memory and a valid GOLD
            # target, never a wrong selected memory with another API's target.
            # It is padding for collectives, not an additional actor attempt.
            prefix = agent.prefix(history if match else [], "arguments", thought if match else "",
                memory=memory[gold_index], document=tools[step.api_identity].registration_document)
            al.append(agent.nll(prefix, compact(step.arguments)) * int(match))
        # Missing or failed attempts retain their denominator and the teacher
        # branch; successful actor samples are not silently oversampled.
        selection.append(torch.stack(sl).sum() / max(1, len(attempts)))
        arguments.append(torch.stack(al).sum() / max(1, len(attempts)))
    self_selection = torch.stack(selection).mean()
    self_arguments = torch.stack(arguments).mean()
    return {**teacher_losses, "self_selection": self_selection, "self_arguments": self_arguments,
            "actor_attempts": zero.detach() + attempts_total, "actor_usable": zero.detach() + usable,
            "actor_identity_matches": zero.detach() + matched, "actor_unusable": zero.detach() + failed,
            "loss": teacher_weight * teacher_losses["loss"] + self_selection + self_arguments}
