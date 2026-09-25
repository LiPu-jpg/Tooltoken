"""Train-only ToolBench agent entry point. No automatic test scoring or jobs.

Run with an explicitly selected Accelerator configuration for distributed
training. Full backbone training is the default; no frozen E0 cache is used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import re
from collections import Counter
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .toolbench_agent import Limits, ToolBenchAgent, prompt_parts
from .toolbench_checkpoint import save_agent, sha256
from .toolbench_data import FINISH, load_training_steps, load_training_tools
from .toolbench_schema import SCHEMA_TASKS, schema_targets, target_facts


def accumulation_window_size(index: int, batches: int, accumulation: int) -> int:
    return min(accumulation, batches - (index // accumulation) * accumulation)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tools", type=Path, required=True, help="Training-only API registry with actual JSON schemas")
    p.add_argument("--recipe", type=Path, help="Versioned hyperparameter JSON; explicit command-line flags override it")
    p.add_argument("--trajectories", type=Path, required=True, help="Training-only labeled agent trajectories")
    p.add_argument("--source-format", choices=["toolbench", "toolgen"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true", help="Audit data without loading model weights")
    p.add_argument("--training-stage", choices=["teacher", "self_conditioned"], default="teacher")
    p.add_argument("--rollout-dir", type=Path, action="append", default=[],
                   help="Completed train-only actor shard; repeat for independently collected GPU shards")
    p.add_argument("--teacher-mix-weight", type=float, default=0.5,
                   help="Stage 2 teacher loss weight; sampled-selection/argument losses have weight 1")
    p.add_argument("--allow-partial-rollouts", action="store_true",
                   help="Explicitly retain teacher-only loss on uncovered training steps")
    p.add_argument("--model-path")
    p.add_argument("--continue-from-export", type=Path,
                   help="Load all parent model/compiler weights; explicitly reset optimizer and data order")
    p.add_argument("--save-training-state-every", type=int, default=0,
                   help="Save full Accelerator state every N new updates and at the end; 0 disables")
    p.add_argument("--model-role", choices=["base", "warm-start"])
    p.add_argument("--revision", help="Pin the base model revision when loading from the Hub")
    p.add_argument("--warm-start-manifest", type=Path, help="JSON with exact train_api_identities for ALL prior stages")
    p.add_argument("--retrieval-adapter", type=Path, help="Optional stage-1 PEFT adapter; merged before joint training")
    p.add_argument("--retrieval-compiler", type=Path, help="Optional upstream compiler.pt state_dict")
    p.add_argument("--condition", choices=["memory", "full_document", "query_only"], default="memory")
    p.add_argument("--train-mode", choices=["full", "compiler_only"], default="full")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-updates", type=int)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--compiler-learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--candidate-count", type=int, default=8)
    p.add_argument("--compiler-rank", type=int, default=128)
    p.add_argument("--memory-slots", type=int, default=8)
    p.add_argument("--memory-kind", choices=["structured", "legacy"], default="structured")
    p.add_argument("--memory-width", type=int, default=512)
    p.add_argument("--memory-depth", type=int, default=2)
    p.add_argument("--memory-heads", type=int, default=8)
    p.add_argument("--schema-weight", type=float, default=0.5)
    p.add_argument("--schema-tasks-per-step", type=int, default=1)
    p.add_argument("--max-train-seconds", type=float, help="Stop after a synchronized update and export before the scheduler deadline")
    p.add_argument("--expected-world-size", type=int, help="Require this actual distributed world size")
    p.add_argument("--max-context-length", type=int, default=6144)
    p.add_argument("--max-document-length", type=int, default=2048)
    p.add_argument("--max-target-length", type=int, default=1024)
    p.add_argument("--thought-weight", type=float, default=0.2)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    return p


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--recipe", type=Path)
    initial, _ = preliminary.parse_known_args(argv)
    p = parser()
    if initial.recipe:
        recipe = json.loads(initial.recipe.read_text())
        allowed = {"memory_kind", "memory_width", "memory_depth", "memory_heads", "memory_slots",
                   "schema_weight", "schema_tasks_per_step", "train_mode", "condition", "candidate_count",
                   "compiler_rank", "thought_weight", "learning_rate", "compiler_learning_rate",
                   "training_stage", "teacher_mix_weight"}
        if not isinstance(recipe, dict) or set(recipe) - allowed:
            raise ValueError("Recipe contains unsupported fields")
        actions = {action.dest: action for action in p._actions}
        for key, value in recipe.items():
            action = actions[key]
            if action.type:
                recipe[key] = action.type(value)
            if action.choices and recipe[key] not in action.choices:
                raise ValueError(f"Invalid recipe choice for {key}")
        p.set_defaults(**recipe)
    return p.parse_args(argv)


def audit_token_lengths(tokenizer, tools, steps, *, limits, slots, condition, schema_enabled):
    from .toolbench_data import compact
    maxima = Counter()
    def tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    for identity, tool in tools.items():
        count = tokens("Represent this tool for registration.\n" + tool.registration_document)
        maxima["document"] = max(maxima["document"], count)
        if count > limits.document:
            raise ValueError(f"Document/schema overflow before model loading: {identity}, {count} tokens")
    def check(history, thought, task, target, tool, label):
        parts = prompt_parts(tokenizer, history, task, thought, condition=condition,
                             document=tool.registration_document)
        prefix_count = sum(map(len, parts)) + (slots if len(parts) == 2 else 0)
        target_count = tokens(target) + 1 if target is not None else 0
        maxima[task + "_prefix"] = max(maxima[task + "_prefix"], prefix_count)
        maxima[task + "_target"] = max(maxima[task + "_target"], target_count)
        if target_count > limits.target or prefix_count + target_count > limits.context:
            raise ValueError(f"Complete {task} overflow before model loading: {label}; prefix={prefix_count}, target={target_count}")
    for step in steps:
        tool = tools[step.api_identity]
        label = f"{step.source_id}:{step.step_index}"
        check(step.history, "", "thought", step.thought, tool, label)
        check(step.history, step.thought, "select", None, tool, label)
        check(step.history, step.thought, "arguments", compact(step.arguments), tool, label)
    if schema_enabled:
        for identity in sorted({step.api_identity for step in steps}):
            tool = tools[identity]
            for task, target in schema_targets(tool.parameters).items():
                check([], "", task, compact(target), tool, identity)
    return dict(maxima)


def audit_self_condition_lengths(tokenizer, tools, steps, conditions, *, limits, slots, condition):
    from .toolbench_data import compact
    from .toolbench_self_conditioning import step_key, usable_condition
    maxima = Counter()
    for step in steps:
        for item in conditions.get(step_key(step), []):
            if not usable_condition(item):
                continue
            for task in ("select", "arguments"):
                selected = item["selected_identity"]
                if task == "arguments" and selected != step.api_identity:
                    continue
                parts = prompt_parts(tokenizer, item["history"], task, item["thought"],
                    condition=condition, document=tools[selected].registration_document)
                prefix = sum(map(len, parts)) + (slots if len(parts) == 2 else 0)
                target = (len(tokenizer.encode(compact(step.arguments), add_special_tokens=False)) + 1
                          if task == "arguments" else 0)
                if target > limits.target or prefix + target > limits.context:
                    raise ValueError(f"Self-conditioned {task} exceeds length contract: {step_key(step)}")
                maxima[task + "_prefix"] = max(maxima[task + "_prefix"], prefix)
                maxima[task + "_target"] = max(maxima[task + "_target"], target)
    return dict(maxima)


def main() -> None:
    args = parse_args()
    if (args.trajectories.parent / "FAILED.json").exists():
        raise ValueError("Refusing a partially converted training source; resolve its conversion failure first")
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps) < 1:
        raise ValueError("Training counts must be positive")
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("max-updates must be positive")
    if args.candidate_count < 2 or args.thought_weight <= 0:
        raise ValueError("Keep selection and planning supervision active")
    if args.schema_weight < 0 or not 1 <= args.schema_tasks_per_step <= 5:
        raise ValueError("Invalid schema supervision settings")
    if args.max_train_seconds is not None and args.max_train_seconds <= 0:
        raise ValueError("Training time budget must be positive")
    if args.save_training_state_every < 0:
        raise ValueError("Training-state interval must be nonnegative")
    if args.training_stage == "self_conditioned":
        if not args.rollout_dir or not args.continue_from_export or args.candidate_count < 3:
            raise ValueError("Stage 2 needs actor shards, their exact parent export, and >=3 candidates")
        if not 0 < args.teacher_mix_weight <= 1:
            raise ValueError("Stage 2 must retain a positive teacher anchor, at most 1")
    elif args.rollout_dir or args.allow_partial_rollouts:
        raise ValueError("Actor shards are only used by the self_conditioned stage")
    if args.continue_from_export and (args.retrieval_adapter or args.retrieval_compiler or args.model_path or args.model_role):
        raise ValueError("A serving export supplies all model/compiler weights; do not mix other warm starts")
    tools = load_training_tools(args.tools)
    steps = load_training_steps(args.trajectories, tools, source_format=args.source_format)
    audit = {
        "source_format": args.source_format,
        "source_sha256": {"tools": sha256(args.tools), "trajectories": sha256(args.trajectories)},
        "train_api_identities": sorted(set(tools) - {FINISH}),
        "train_document_hashes": {key: tool.document_hash for key, tool in tools.items()},
        "trajectories": len({step.source_id for step in steps}), "steps": len(steps),
        "steps_by_api": dict(Counter(step.api_identity for step in steps)),
        "observations_in_prefix": sum(any(item["type"] == "observation" for item in step.history) for step in steps),
        "terminal_steps": sum(step.api_identity == FINISH for step in steps),
        "test_files_opened": False,
        "schema_fact_counts": dict(Counter({task: sum(len(target_facts(task, schema_targets(tool.parameters)[task]))
            for tool in tools.values()) for task in SCHEMA_TASKS})),
    }
    if args.recipe:
        audit["recipe_sha256"] = sha256(args.recipe)
    continuation = None
    if args.continue_from_export:
        from .toolbench_continuation import inspect_continuation
        continuation = inspect_continuation(args.continue_from_export, args, audit)
        audit["continuation"] = continuation
    self_conditions = {}
    if args.training_stage == "self_conditioned":
        from .toolbench_self_conditioning import read_rollouts
        self_conditions, rollout_audit = read_rollouts(args.rollout_dir, steps, tools,
            source_sha256=audit["source_sha256"], checkpoint=args.continue_from_export,
            source_format=args.source_format, allow_partial=args.allow_partial_rollouts)
        audit["self_conditioning"] = rollout_audit
    audit["training_stage"] = args.training_stage
    audit["teacher_mix_weight"] = args.teacher_mix_weight if self_conditions else 1.0
    if args.retrieval_adapter or args.retrieval_compiler or args.model_role == "warm-start":
        if args.warm_start_manifest is None or args.model_role != "warm-start":
            raise ValueError("Warm-start artifacts require their combined training API lineage")
        lineage = json.loads(args.warm_start_manifest.read_text())
        trained_apis = lineage.get("train_api_identities")
        if not isinstance(trained_apis, list) or not set(trained_apis).issubset(tools):
            raise ValueError("Warm start contains unknown/held-out training APIs")
        audit["warm_start_manifest_sha256"] = sha256(args.warm_start_manifest)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    # Synchronize this read before rank 0 creates the directory.
    output_exists = torch.tensor([int(args.output_dir.exists())], device=accelerator.device)
    if accelerator.reduce(output_exists, reduction="sum").item():
        raise FileExistsError("Output directory must not already exist; never overwrite prior evidence")
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    accelerator.wait_for_everyone()
    if args.audit_only:
        accelerator.print(json.dumps({"status": "data_audited", "steps": len(steps)}))
        return
    if not args.continue_from_export and (not args.model_path or not args.model_role):
        raise ValueError("Training requires an explicit model-path and model-role")
    if not args.continue_from_export and not Path(args.model_path).exists() and not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        raise ValueError("Pin the remote model revision")
    set_seed(args.seed)
    tokenizer_path = args.continue_from_export / "backbone" if args.continue_from_export else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=args.revision)
    lengths = audit_token_lengths(tokenizer, tools, steps, limits=Limits(args.max_context_length,
        args.max_document_length, args.max_target_length), slots=args.memory_slots,
        condition=args.condition, schema_enabled=args.schema_weight > 0)
    if self_conditions:
        lengths["self_conditioned"] = audit_self_condition_lengths(tokenizer, tools, steps, self_conditions,
            limits=Limits(args.max_context_length, args.max_document_length, args.max_target_length),
            slots=args.memory_slots, condition=args.condition)
    if accelerator.is_main_process:
        (args.output_dir / "token_length_audit.json").write_text(json.dumps(lengths, indent=2) + "\n")
    if args.continue_from_export:
        from .toolbench_continuation import load_trainable_agent
        model = load_trainable_agent(args.continue_from_export, train_mode=args.train_mode,
                                    gradient_checkpointing=args.gradient_checkpointing)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(args.model_path, revision=args.revision)
        if args.retrieval_adapter:
            from peft import PeftModel
            backbone = PeftModel.from_pretrained(backbone, args.retrieval_adapter).merge_and_unload()
        backbone.requires_grad_(args.train_mode == "full")
        if args.gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = ToolBenchAgent(backbone, tokenizer, rank=args.compiler_rank, slots=args.memory_slots,
                               condition=args.condition, limits=Limits(args.max_context_length,
                               args.max_document_length, args.max_target_length), memory_kind=args.memory_kind,
                               memory_width=args.memory_width, memory_depth=args.memory_depth, memory_heads=args.memory_heads)
    if args.retrieval_compiler:
        model.output_compiler.load_state_dict(torch.load(args.retrieval_compiler, map_location="cpu", weights_only=True))
    groups = [{"params": [parameter for name, parameter in model.named_parameters()
                          if parameter.requires_grad and name.startswith("backbone.")], "lr": args.learning_rate},
              {"params": [parameter for name, parameter in model.named_parameters()
                          if parameter.requires_grad and not name.startswith("backbone.")], "lr": args.compiler_learning_rate}]
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=args.weight_decay)
    loader = DataLoader(steps, batch_size=args.batch_size, shuffle=True, collate_fn=list,
                        generator=torch.Generator().manual_seed(args.seed))
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if args.expected_world_size is not None and accelerator.num_processes != args.expected_world_size:
        raise ValueError("Actual distributed world size differs from the requested allocation")
    runtime = {"rank": accelerator.process_index, "world_size": accelerator.num_processes,
               "compiler_scale_initialization": accelerator.unwrap_model(model).scale_initialization,
               "device": str(accelerator.device), "distributed_type": str(accelerator.distributed_type),
               "mixed_precision": accelerator.mixed_precision,
               "trainable_backbone_parameters": sum(p.numel() if not hasattr(p, "ds_numel") else p.ds_numel
                   for name, p in accelerator.unwrap_model(model).named_parameters() if name.startswith("backbone.") and p.requires_grad)}
    if accelerator.device.type == "cuda":
        runtime.update(gpu_name=torch.cuda.get_device_name(accelerator.device),
                       gpu_memory_bytes=torch.cuda.get_device_properties(accelerator.device).total_memory)
    if hasattr(model, "train_micro_batch_size_per_gpu"):
        runtime.update(engine_microbatch=model.train_micro_batch_size_per_gpu(),
                       engine_accumulation=model.gradient_accumulation_steps(), engine_global_batch=model.train_batch_size())
        if (runtime["engine_microbatch"] != args.batch_size or runtime["engine_accumulation"] != args.gradient_accumulation_steps
                or runtime["engine_global_batch"] != args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes):
            raise ValueError("Prepared DeepSpeed engine violates the training batch contract")
    (args.output_dir / f"runtime-rank-{accelerator.process_index}.json").write_text(json.dumps(runtime, indent=2) + "\n")
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    initial_updates = continuation["parent_updates"] if continuation else 0
    total_updates = initial_updates + updates_per_epoch * args.epochs
    if args.max_updates is not None:
        total_updates = min(total_updates, args.max_updates)
    remaining_updates = total_updates - initial_updates
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * remaining_updates), remaining_updates)
    if continuation:
        if (runtime["world_size"] != continuation["parent_world_size"]
                or args.batch_size * args.gradient_accumulation_steps * runtime["world_size"] != continuation["parent_effective_full_batch"]):
            raise ValueError("Continuation world size or effective batch differs from parent")
    from .toolbench_continuation import RuntimeTrainingState
    training_state = RuntimeTrainingState(accelerator.unwrap_model(model), {
        "data_sha256": audit["source_sha256"], "continuation": continuation,
        "seed": args.seed, "world_size": accelerator.num_processes,
        "batch_size": args.batch_size, "accumulation": args.gradient_accumulation_steps,
        "new_updates_planned": remaining_updates,
        "training_stage": args.training_stage,
        "self_conditioning": audit.get("self_conditioning"),
    })
    accelerator.register_for_checkpointing(scheduler, training_state)
    updates = initial_updates
    def save_training_state(epoch, next_batch):
        training_state.cursor = {"epoch": epoch, "next_microbatch": next_batch,
                                 "cumulative_updates": updates, "new_updates": updates - initial_updates}
        output = args.output_dir / "training-state" / f"update-{updates}"
        exists = torch.tensor(int(output.exists()), device=accelerator.device)
        if accelerator.reduce(exists, reduction="sum").item():
            raise FileExistsError("Never overwrite an earlier training state")
        accelerator.save_state(str(output))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Large parameter/optimizer shards are authenticated after saving.
            files = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
            (output / "READY.json").write_text(json.dumps({"cursor": training_state.cursor, "files": files,
                "includes_model_optimizer_scheduler_rng_and_buffers": True}, indent=2) + "\n")
        accelerator.wait_for_everyone()
    started = time.monotonic()
    stop_reason = "requested_updates_or_epochs"
    time_exhausted = False
    schema_exposures: Counter = Counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(args.epochs):
        for index, batch in enumerate(loader):
            with accelerator.accumulate(model):
                extra = {}
                if self_conditions:
                    extra = dict(self_conditions=[self_conditions.get((step.source_id, step.step_index), [])
                                                  for step in batch],
                                 teacher_mix_weight=args.teacher_mix_weight,
                                 self_attempt_slots=audit["self_conditioning"]["attempt_slots"])
                losses = model(batch, tools, candidate_count=args.candidate_count,
                               seed=args.seed + epoch, thought_weight=args.thought_weight,
                               schema_weight=args.schema_weight, schema_tasks_per_step=args.schema_tasks_per_step,
                               schema_step=epoch * len(loader) * args.batch_size + index * args.batch_size, **extra)
                if args.schema_weight:
                    for sample_index in range(len(batch)):
                        offset = epoch * len(loader) * args.batch_size + index * args.batch_size + sample_index
                        schema_exposures.update(list(SCHEMA_TASKS)[(offset + j) % 5] for j in range(args.schema_tasks_per_step))
                # Accelerator divides by accumulation even for the last partial
                # window. Correct that denominator; flush the final update.
                window = accumulation_window_size(index, len(loader), args.gradient_accumulation_steps)
                accelerator.backward(losses["loss"] * args.gradient_accumulation_steps / window)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                    updates += 1
                    elapsed = time.monotonic() - started
                    if args.max_train_seconds is not None:
                        flag = torch.tensor(int(elapsed >= args.max_train_seconds), device=accelerator.device)
                        time_exhausted = bool(accelerator.reduce(flag, reduction="sum").item())
                    if time_exhausted:
                        stop_reason = "training_time_budget_export"
                    record = {"update": updates, "epoch": epoch, "rank0_last_microbatch_losses":
                              {key: float(value.detach()) for key, value in losses.items()}}
                    if accelerator.is_main_process:
                        (args.output_dir / "progress.json").write_text(json.dumps({"updates": updates,
                            "target_updates": total_updates, "training_elapsed_seconds": elapsed,
                            "stop_reason": stop_reason if time_exhausted else None}) + "\n")
                        with (args.output_dir / "training.jsonl").open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                    if args.save_training_state_every and (
                            (updates - initial_updates) % args.save_training_state_every == 0
                            or updates >= total_updates or time_exhausted):
                        save_training_state(epoch, index + 1)
                    if updates >= total_updates or time_exhausted:
                        break
        if updates >= total_updates or time_exhausted:
            break
    accelerator.wait_for_everyone()
    exposures = accelerator.reduce(torch.tensor([schema_exposures[task] for task in SCHEMA_TASKS],
        dtype=torch.long, device=accelerator.device), reduction="sum").cpu().tolist()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_agent(accelerator.unwrap_model(model), args.output_dir / "checkpoint", metadata={
            **audit, "updates": updates, "world_size": accelerator.num_processes,
            "effective_full_batch": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
            "train_mode": args.train_mode, "development_or_test_evaluated": False,
            "schema_weight": args.schema_weight, "schema_task_exposures": dict(zip(SCHEMA_TASKS, exposures)),
            "token_length_maxima": lengths,
            "stop_reason": stop_reason, "target_updates": total_updates,
            "training_elapsed_seconds": time.monotonic() - started,
        }, state_dict=state)
        (args.output_dir / "TRAINING_COMPLETE.json").write_text(json.dumps({"updates": updates,
            "target_updates": total_updates, "stop_reason": stop_reason}) + "\n")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
