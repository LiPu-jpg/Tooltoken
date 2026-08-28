from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ToolExample, stable_tool_split, tool_registry_key
from .model import GeneratedMemory
from .physical_tokens import ReservedTokenPool
from .train import seed_everything
from .train_bidirectional import (
    build_frozen_features,
    generation_metrics,
    greedy_decode,
    load_examples,
    parse_json_object,
    prepare_batch,
    prepare_trajectory_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--total-slots", type=int, default=1024)
    parser.add_argument("--train-slots", type=int, default=768)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--input-memory-slots", type=int, default=8)
    parser.add_argument("--adaptation-steps", type=int, default=20)
    parser.add_argument("--adaptation-learning-rate", type=float, default=0.05)
    parser.add_argument("--encode-batch-size", type=int, default=2)
    parser.add_argument("--max-encode-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=640)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-limit", type=int, default=100)
    return parser.parse_args()


def assigned_tool_keys(tool_keys: list[str], world_size: int, rank: int) -> list[str]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed assignment")
    return [key for index, key in enumerate(tool_keys) if index % world_size == rank]


def normalize_memory_(memory: torch.Tensor, target_norm: float) -> None:
    with torch.no_grad():
        memory.copy_(F.normalize(memory.float(), dim=-1) * target_norm)


def adaptation_example(example: ToolExample) -> ToolExample:
    return ToolExample(
        example_id=f"adapt::{example.example_id}",
        query="Reconstruct the selected tool schema.",
        tool_name=example.tool_name,
        tool_document=example.tool_document,
        target_arguments={},
        schema_arguments=example.schema_arguments,
    )


def adapt_memory(
    model,
    tokenizer,
    example: ToolExample,
    slot_id: int,
    initial_memory: torch.Tensor,
    steps: int,
    learning_rate: float,
    target_norm: float,
    max_prompt_length: int,
    max_target_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    memory = nn.Parameter(initial_memory.detach().clone())
    optimizer = torch.optim.AdamW([memory], lr=learning_rate, weight_decay=0.0)
    losses: list[float] = []
    registration_example = adaptation_example(example)
    for _ in range(steps):
        embeds, attention, labels = prepare_batch(
            model,
            tokenizer,
            [registration_example],
            "registered",
            [slot_id],
            memory.unsqueeze(0),
            max_prompt_length,
            max_target_length,
            device,
            target_kind="schema",
        )
        loss = model(
            inputs_embeds=embeds,
            attention_mask=attention,
            labels=labels,
            use_cache=False,
        ).loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite memory adaptation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        normalize_memory_(memory, target_norm)
        losses.append(float(loss.detach()))
    return memory.detach(), losses


@torch.inference_mode()
def generate_with_memory(
    model,
    tokenizer,
    example: ToolExample,
    slot_id: int,
    memory: torch.Tensor,
    trajectory: bool,
    max_prompt_length: int,
    max_target_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    if trajectory:
        embeds, attention, _ = prepare_trajectory_batch(
            model,
            tokenizer,
            [example],
            memory.unsqueeze(0),
            max_prompt_length,
            max_target_length,
            device,
            include_targets=False,
        )
    else:
        embeds, attention, _ = prepare_batch(
            model,
            tokenizer,
            [example],
            "registered",
            [slot_id],
            memory.unsqueeze(0),
            max_prompt_length,
            max_target_length,
            device,
            include_targets=False,
        )
    return greedy_decode(model, embeds, attention, tokenizer, max_new_tokens)


def main() -> None:
    args = parse_args()
    if args.adaptation_steps < 1 or args.adaptation_learning_rate <= 0:
        raise ValueError("Adaptation steps and learning rate must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.dataset, args.max_examples)
    train_examples, eval_examples = stable_tool_split(examples, args.eval_ratio, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    token_pool = ReservedTokenPool.create(
        tokenizer, model, args.total_slots, args.train_slots
    )
    features = build_frozen_features(
        model,
        tokenizer,
        examples,
        args.encode_batch_size,
        args.max_encode_length,
        device,
        tool_identity="document",
    )
    tool_index = {key: index for index, key in enumerate(features["tool_keys"])}
    tool_features = {
        key: features["tool"][index] for key, index in tool_index.items()
    }
    eval_tool_keys = sorted(
        {tool_registry_key(example, "document") for example in eval_examples}
    )
    if len(eval_tool_keys) > len(token_pool.heldout_token_ids):
        raise ValueError("Not enough strictly held-out physical slots")
    slot_by_tool = {
        key: token_pool.heldout_token_ids[index]
        for index, key in enumerate(eval_tool_keys)
    }
    representative = {
        tool_registry_key(example, "document"): example for example in eval_examples
    }

    input_weight = model.get_input_embeddings().weight[: token_pool.original_vocab_size]
    input_norm = float(input_weight.detach().float().norm(dim=-1).mean())
    generator = GeneratedMemory(
        model.config.hidden_size,
        args.rank,
        args.input_memory_slots,
        input_norm,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    generator.load_state_dict(checkpoint["input_generator"])
    generator.requires_grad_(False)
    generator.eval()

    local_keys = assigned_tool_keys(eval_tool_keys, world_size, rank)
    initial_memories: dict[str, torch.Tensor] = {}
    adapted_memories: dict[str, torch.Tensor] = {}
    loss_rows: list[dict[str, Any]] = []
    model.gradient_checkpointing_enable()
    model.train()
    for key in local_keys:
        with torch.no_grad():
            initial = generator(tool_features[key].to(device).unsqueeze(0))[0]
        adapted, losses = adapt_memory(
            model,
            tokenizer,
            representative[key],
            slot_by_tool[key],
            initial,
            args.adaptation_steps,
            args.adaptation_learning_rate,
            input_norm,
            args.max_prompt_length,
            args.max_target_length,
            device,
        )
        initial_memories[key] = initial.detach()
        adapted_memories[key] = adapted
        loss_rows.append(
            {
                "tool_id": key,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "finite": all(math.isfinite(value) for value in losses),
            }
        )
    model.gradient_checkpointing_disable()
    model.eval()
    print(
        f"rank={rank} adapted_tools={len(local_keys)} "
        f"mean_first_loss={sum(row['first_loss'] for row in loss_rows) / max(1, len(loss_rows)):.6f} "
        f"mean_last_loss={sum(row['last_loss'] for row in loss_rows) / max(1, len(loss_rows)):.6f}",
        flush=True,
    )

    generation_examples = eval_examples[: args.generation_limit or None]
    local_rows: list[dict[str, Any]] = []
    conditions = (
        ("pooled_registered", False, initial_memories),
        ("adapted_registered", False, adapted_memories),
        ("pooled_correct_tool_trajectory", True, initial_memories),
        ("adapted_correct_tool_trajectory", True, adapted_memories),
    )
    for example in generation_examples:
        key = tool_registry_key(example, "document")
        if key not in adapted_memories:
            continue
        for condition, trajectory, memories in conditions:
            text = generate_with_memory(
                model,
                tokenizer,
                example,
                slot_by_tool[key],
                memories[key],
                trajectory,
                args.max_prompt_length,
                args.max_target_length,
                args.max_new_tokens,
                device,
            )
            local_rows.append(
                {
                    "condition": condition,
                    "example_id": example.example_id,
                    "tool_name": example.tool_name,
                    "tool_id": key,
                    "target": example.target_arguments,
                    "prediction": text,
                    "parsed": parse_json_object(text),
                }
            )

    gathered_rows: list[list[dict[str, Any]] | None] | None = (
        [None] * world_size if is_main else None
    )
    gathered_losses: list[list[dict[str, Any]] | None] | None = (
        [None] * world_size if is_main else None
    )
    if dist.is_initialized():
        dist.gather_object(local_rows, gathered_rows, dst=0)
        dist.gather_object(loss_rows, gathered_losses, dst=0)
    else:
        gathered_rows = [local_rows]
        gathered_losses = [loss_rows]

    if is_main:
        rows = [row for group in gathered_rows or [] for row in group or []]
        losses = [row for group in gathered_losses or [] for row in group or []]
        order = {example.example_id: index for index, example in enumerate(generation_examples)}
        rows.sort(key=lambda row: (str(row["condition"]), order[str(row["example_id"])]))
        summaries = {
            condition: generation_metrics(
                [row for row in rows if row["condition"] == condition]
            )
            for condition, _, _ in conditions
        }
        first_mean = sum(row["first_loss"] for row in losses) / max(1, len(losses))
        last_mean = sum(row["last_loss"] for row in losses) / max(1, len(losses))
        results = {
            "split": {
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "train_tools": len(
                    {tool_registry_key(example, "document") for example in train_examples}
                ),
                "eval_tools": len(eval_tool_keys),
                "tool_overlap": 0,
                "tool_name_overlap": len(
                    {example.tool_name for example in train_examples}
                    & {example.tool_name for example in eval_examples}
                ),
            },
            "physical_token_audit": token_pool.audit(model, set()),
            "adaptation_audit": {
                "adapted_tools": len(losses),
                "uses_tool_schema": True,
                "uses_evaluation_queries": False,
                "uses_target_arguments": False,
                "zero_gradient_registration_claim": False,
                "all_losses_finite": all(row["finite"] for row in losses),
                "mean_first_loss": first_mean,
                "mean_last_loss": last_mean,
                "loss_decreased": last_mean < first_mean,
            },
            "generation": summaries,
            "config": {**vars(args), "world_size": world_size},
        }
        (output_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (output_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
