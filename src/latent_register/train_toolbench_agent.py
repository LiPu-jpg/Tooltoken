"""Train-only ToolBench agent entry point. No automatic test scoring or jobs.

Run with an explicitly selected Accelerator configuration for distributed
training. Full backbone training is the default; no frozen E0 cache is used.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .toolbench_agent import Limits, ToolBenchAgent
from .toolbench_checkpoint import save_agent, sha256
from .toolbench_data import FINISH, load_training_steps, load_training_tools


def accumulation_window_size(index: int, batches: int, accumulation: int) -> int:
    return min(accumulation, batches - (index // accumulation) * accumulation)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tools", type=Path, required=True, help="Training-only API registry with actual JSON schemas")
    p.add_argument("--trajectories", type=Path, required=True, help="Training-only labeled agent trajectories")
    p.add_argument("--source-format", choices=["toolbench", "toolgen"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--audit-only", action="store_true", help="Audit data without loading model weights")
    p.add_argument("--model-path")
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
    p.add_argument("--max-context-length", type=int, default=6144)
    p.add_argument("--max-document-length", type=int, default=2048)
    p.add_argument("--max-target-length", type=int, default=1024)
    p.add_argument("--thought-weight", type=float, default=0.2)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=17)
    return p


def main() -> None:
    args = parser().parse_args()
    if (args.trajectories.parent / "FAILED.json").exists():
        raise ValueError("Refusing a partially converted training source; resolve its conversion failure first")
    if min(args.epochs, args.batch_size, args.gradient_accumulation_steps) < 1:
        raise ValueError("Training counts must be positive")
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError("max-updates must be positive")
    if args.candidate_count < 2 or args.thought_weight <= 0:
        raise ValueError("Keep selection and planning supervision active")
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
    }
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
    if not args.model_path or not args.model_role:
        raise ValueError("Training requires an explicit model-path and model-role")
    if not Path(args.model_path).exists() and not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        raise ValueError("Pin the remote model revision")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision)
    backbone = AutoModelForCausalLM.from_pretrained(args.model_path, revision=args.revision)
    if args.retrieval_adapter:
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, args.retrieval_adapter).merge_and_unload()
    backbone.requires_grad_(args.train_mode == "full")
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = ToolBenchAgent(backbone, tokenizer, rank=args.compiler_rank, slots=args.memory_slots,
                           condition=args.condition, limits=Limits(args.max_context_length,
                           args.max_document_length, args.max_target_length))
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
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    if args.max_updates is not None:
        total_updates = min(total_updates, args.max_updates)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_updates), total_updates)
    updates = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(args.epochs):
        for index, batch in enumerate(loader):
            with accelerator.accumulate(model):
                losses = model(batch, tools, candidate_count=args.candidate_count,
                               seed=args.seed + epoch, thought_weight=args.thought_weight)
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
                    record = {"update": updates, "epoch": epoch, "rank0_last_microbatch_losses":
                              {key: float(value.detach()) for key, value in losses.items()}}
                    if accelerator.is_main_process:
                        with (args.output_dir / "training.jsonl").open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                    if updates >= total_updates:
                        break
        if updates >= total_updates:
            break
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_agent(accelerator.unwrap_model(model), args.output_dir / "checkpoint", metadata={
            **audit, "updates": updates, "world_size": accelerator.num_processes,
            "effective_full_batch": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
            "train_mode": args.train_mode, "development_or_test_evaluated": False,
        }, state_dict=state)
        (args.output_dir / "TRAINING_COMPLETE.json").write_text(json.dumps({"updates": updates}) + "\n")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
