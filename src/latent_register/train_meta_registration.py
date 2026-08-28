from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .episodic_data import (
    BoundRegistryEpisode,
    EpisodicRegistrySampler,
    PreparedRetrievalEpisode,
    load_prepared_tools,
    load_retrieval_episodes,
    load_token_pools,
)
from .model import PhysicalOutputGenerator
from .physical_tokens import SplitReservedTokenPool
from .train import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-train post-training tool-token registration"
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--registry-size", type=int, default=8)
    parser.add_argument("--eval-registry-size", type=int, action="append")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-path")
    parser.add_argument("--compiler-path")
    parser.add_argument("--max-query-length", type=int, default=256)
    parser.add_argument("--max-document-length", type=int, default=256)
    parser.add_argument(
        "--document-instruction",
        default="Represent this tool for registration.",
    )
    parser.add_argument("--compiler-rank", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=1000)
    return parser.parse_args()


def render_query(tokenizer, query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Select every registered tool required to satisfy the request.",
        },
        {"role": "user", "content": query},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"System: {messages[0]['content']}\nUser: {query}\nAssistant:"


def multi_positive_selection_loss(
    logits: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("Logits and positive mask must have the same [batch, registry] shape")
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every registry episode must have at least one positive")
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    return (
        torch.logsumexp(logits.float(), dim=1)
        - torch.logsumexp(positive_logits.float(), dim=1)
    ).mean()


def selection_statistics(
    logits: torch.Tensor, positive_mask: torch.Tensor
) -> tuple[int, float]:
    order = logits.float().argsort(dim=1, descending=True)
    ranked_positive = positive_mask.gather(1, order)
    first_positive = ranked_positive.float().argmax(dim=1)
    hits = int((first_positive == 0).sum().item())
    reciprocal_rank = float((1.0 / (first_positive.float() + 1.0)).sum().item())
    return hits, reciprocal_rank


def _last_state(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.sum(dim=1) - 1
    return hidden[
        torch.arange(len(hidden), device=hidden.device), positions
    ]


class MetaRegistrationModel(nn.Module):
    def __init__(self, backbone: nn.Module, compiler: PhysicalOutputGenerator):
        super().__init__()
        self.backbone = backbone
        self.compiler = compiler

    def _hidden(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        causal_lm = (
            self.backbone.get_base_model()
            if hasattr(self.backbone, "get_base_model")
            else self.backbone
        )
        decoder = causal_lm.model
        output = decoder(
            **tokens,
            use_cache=False,
            return_dict=True,
        )
        return output.last_hidden_state

    def forward(
        self,
        query_tokens: dict[str, torch.Tensor],
        document_tokens: dict[str, torch.Tensor],
        positive_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, registry_size = positive_mask.shape
        query_hidden = self._hidden(query_tokens)
        query_states = _last_state(query_hidden, query_tokens["attention_mask"])

        document_hidden = self._hidden(document_tokens)
        document_mask = document_tokens["attention_mask"].unsqueeze(-1)
        document_states = (
            (document_hidden * document_mask).sum(dim=1)
            / document_mask.sum(dim=1).clamp_min(1)
        )
        output_rows = self.compiler.output_rows(document_states)
        output_rows = output_rows.reshape(batch_size, registry_size, -1)
        logits = torch.einsum("bh,brh->br", query_states.float(), output_rows.float())
        return multi_positive_selection_loss(logits, positive_mask), logits


def _tokenize_bound_episodes(
    tokenizer,
    episodes: list[BoundRegistryEpisode],
    *,
    max_query_length: int,
    max_document_length: int,
    device: torch.device,
    document_instruction: str = "Represent this tool for registration.",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, int]:
    registry_size = len(episodes[0].tools)
    if any(len(episode.tools) != registry_size for episode in episodes):
        raise ValueError("A meta-training batch mixed registry sizes")
    query_texts = [render_query(tokenizer, episode.query) for episode in episodes]
    document_texts = [
        f"{document_instruction}\n{tool.document}"
        for episode in episodes
        for tool in episode.tools
    ]
    query_tokens = tokenizer(
        query_texts,
        padding=True,
        truncation=True,
        max_length=max_query_length,
        return_tensors="pt",
    ).to(device)
    document_tokens = tokenizer(
        document_texts,
        padding=True,
        truncation=True,
        max_length=max_document_length,
        return_tensors="pt",
    ).to(device)
    positive_mask = torch.zeros(
        len(episodes), registry_size, dtype=torch.bool, device=device
    )
    for row, episode in enumerate(episodes):
        positive_mask[row, list(episode.positive_positions)] = True
    non_padding_tokens = int(query_tokens["attention_mask"].sum().item()) + int(
        document_tokens["attention_mask"].sum().item()
    )
    return query_tokens, document_tokens, positive_mask, non_padding_tokens


def _epoch_batches(
    episodes: list[PreparedRetrievalEpisode],
    *,
    epoch: int,
    seed: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> Iterable[list[PreparedRetrievalEpisode]]:
    indices = list(range(len(episodes)))
    random.Random(seed + epoch).shuffle(indices)
    global_batch_size = batch_size * world_size
    usable = len(indices) - len(indices) % global_batch_size
    indices = indices[:usable]
    local = indices[rank:usable:world_size]
    for start in range(0, len(local), batch_size):
        yield [episodes[index] for index in local[start : start + batch_size]]


def _reduce_metrics(values: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def _disable_incompatible_optional_torchao() -> bool:
    """Keep PEFT on its standard Linear path when an old optional torchao is installed."""
    try:
        from peft.tuners.lora import torchao as peft_torchao
    except ImportError:
        return False
    try:
        peft_torchao.is_torchao_available()
    except ImportError:
        peft_torchao.is_torchao_available = lambda: False
        return True
    return False


def _provide_optional_tensor_parallel_compat() -> bool:
    """Let PEFT load non-TP adapters with older Transformers releases."""
    from transformers.integrations import tensor_parallel

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        return False

    class _UnusedEmbeddingParallel:
        pass

    tensor_parallel.EmbeddingParallel = _UnusedEmbeddingParallel
    return True


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    tokenizer,
    sampler: EpisodicRegistrySampler,
    episodes: list[PreparedRetrievalEpisode],
    *,
    split: str,
    registry_size: int,
    sample_limit: int,
    max_query_length: int,
    max_document_length: int,
    document_instruction: str = "Represent this tool for registration.",
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float | int]:
    model.eval()
    selected = sorted(episodes, key=lambda item: item.query_hash)[:sample_limit]
    local = selected[rank::world_size]
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for episode in local:
        bound = sampler.bind(episode, registry_size=registry_size, epoch=0)
        query_tokens, document_tokens, positive_mask, _ = _tokenize_bound_episodes(
            tokenizer,
            [bound],
            max_query_length=max_query_length,
            max_document_length=max_document_length,
            device=device,
            document_instruction=document_instruction,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss, logits = model(query_tokens, document_tokens, positive_mask)
        hits, reciprocal_rank = selection_statistics(logits, positive_mask)
        totals += torch.tensor(
            [
                float(loss),
                hits,
                reciprocal_rank,
                float(positive_mask.sum().item()) / positive_mask.shape[1],
                1.0,
            ],
            device=device,
            dtype=torch.float64,
        )
    totals = _reduce_metrics(totals)
    count = max(1.0, float(totals[4].item()))
    return {
        "split": split,
        "episodes": int(totals[4].item()),
        "loss": float(totals[0].item() / count),
        "hit_at_1": float(totals[1].item() / count),
        "mean_reciprocal_rank": float(totals[2].item() / count),
        "random_hit_at_1": float(totals[3].item() / count),
    }


def main() -> None:
    args = parse_args()
    if bool(args.adapter_path) != bool(args.compiler_path):
        raise ValueError("--adapter-path and --compiler-path must be provided together")
    if args.adapter_path and not args.eval_only:
        raise ValueError("Checkpoint loading is currently evaluation-only")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    # All ranks must create identical frozen reserved rows before DDP synchronizes them.
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    prepared_dir = Path(args.prepared_dir)
    tools = load_prepared_tools(prepared_dir / "tools.jsonl")
    all_episodes = load_retrieval_episodes(prepared_dir / "retrieval.jsonl", tools)
    episodes_by_split = {
        split: [episode for episode in all_episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }
    logical_pools = load_token_pools(prepared_dir / "split_manifest.json")
    sampler = EpisodicRegistrySampler(tools, logical_pools, seed=args.seed)

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Meta-registration training requires peft") from exc
    disabled_torchao = _disable_incompatible_optional_torchao()
    if is_main and disabled_torchao:
        print("runtime disabled_incompatible_optional_torchao=true", flush=True)
    provided_tp_compat = _provide_optional_tensor_parallel_compat()
    if is_main and provided_tp_compat:
        print("runtime provided_optional_tensor_parallel_compat=true", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    physical_pool = SplitReservedTokenPool.create(
        tokenizer, backbone, logical_pools
    )

    if args.adapter_path:
        backbone = PeftModel.from_pretrained(
            backbone, args.adapter_path, is_trainable=False
        )
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        backbone.enable_input_require_grads()
    backbone.config.use_cache = False
    backbone.to(device)

    static_rows = backbone.get_output_embeddings().weight[
        : physical_pool.original_vocab_size
    ]
    initial_row_norm = float(static_rows.detach().float().norm(dim=-1).mean())
    compiler = PhysicalOutputGenerator(
        backbone.config.hidden_size, args.compiler_rank, initial_row_norm
    ).to(device)
    if args.compiler_path:
        compiler.load_state_dict(
            torch.load(args.compiler_path, map_location=device, weights_only=True)
        )
    model: nn.Module = MetaRegistrationModel(backbone, compiler).to(device)
    if world_size > 1 and not args.eval_only:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    # Training data are already disjoint by rank; use a distinct RNG stream for dropout.
    seed_everything(args.seed + rank)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batches_per_epoch = len(episodes_by_split["train"]) // (
        args.batch_size * world_size
    )
    updates_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_steps = 0 if args.eval_only else (
        args.max_steps or args.epochs * updates_per_epoch
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max(1, total_steps)
    )
    if is_main:
        trainable_count = sum(parameter.numel() for parameter in trainable)
        print(
            f"runtime world_size={world_size} registry_size={args.registry_size} "
            f"global_batch_size={args.batch_size * world_size * args.gradient_accumulation_steps} "
            f"trainable_parameters={trainable_count} total_steps={total_steps}",
            flush=True,
        )

    training_seen_ids: set[int] = set()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    recent_losses: list[float] = []
    interval_tokens = 0
    interval_started = time.monotonic()
    completed_epochs = 0
    epoch_limit = 0 if args.eval_only else (
        args.epochs if not args.max_steps else 10**9
    )
    for epoch in range(epoch_limit):
        model.train()
        epoch_losses: list[float] = []
        for raw_batch in _epoch_batches(
            episodes_by_split["train"],
            epoch=epoch,
            seed=args.seed,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        ):
            bound_batch = [
                sampler.bind(item, registry_size=args.registry_size, epoch=epoch)
                for item in raw_batch
            ]
            for bound in bound_batch:
                training_seen_ids.update(
                    physical_pool.physical_id(slot) for slot in bound.slot_indices
                )
            query_tokens, document_tokens, positive_mask, token_count = (
                _tokenize_bound_episodes(
                    tokenizer,
                    bound_batch,
                    max_query_length=args.max_query_length,
                    max_document_length=args.max_document_length,
                    device=device,
                    document_instruction=args.document_instruction,
                )
            )
            sync_step = (micro_step + 1) % args.gradient_accumulation_steps == 0
            sync_scope = (
                nullcontext()
                if sync_step or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            with sync_scope:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    loss, _ = model(query_tokens, document_tokens, positive_mask)
                    scaled_loss = loss / args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at micro-step {micro_step + 1}")
                scaled_loss.backward()
            loss_value = float(loss.detach())
            epoch_losses.append(loss_value)
            recent_losses.append(loss_value)
            recent_losses = recent_losses[-100:]
            interval_tokens += token_count
            micro_step += 1
            if not sync_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step + 1}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            should_log = step <= 5 or step % args.log_every == 0
            if should_log:
                elapsed = max(1e-6, time.monotonic() - interval_started)
                tokens_per_second = interval_tokens * world_size / elapsed
                if is_main:
                    allocated = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    )
                    print(
                        f"step={step} epoch={epoch + 1} batch_loss={loss_value:.6f} "
                        f"recent_loss={sum(recent_losses) / len(recent_losses):.6f} "
                        f"grad_norm={float(grad_norm):.4f} tokens_per_second={tokens_per_second:.1f} "
                        f"max_memory_gib={allocated:.2f}",
                        flush=True,
                    )
                interval_tokens = 0
                interval_started = time.monotonic()
            if step >= total_steps:
                break
        completed_epochs = epoch + 1
        if is_main and epoch_losses:
            print(
                f"epoch={epoch + 1} mean_loss={sum(epoch_losses) / len(epoch_losses):.6f}",
                flush=True,
            )
        if step >= total_steps:
            break

    evaluation_sizes = args.eval_registry_size or [args.registry_size]
    evaluations: dict[str, dict[str, dict[str, float | int]]] = {}
    for registry_size in evaluation_sizes:
        evaluations[str(registry_size)] = {}
        for split in ("validation", "test"):
            evaluations[str(registry_size)][split] = evaluate(
                model,
                tokenizer,
                sampler,
                episodes_by_split[split],
                split=split,
                registry_size=registry_size,
                sample_limit=args.eval_samples,
                max_query_length=args.max_query_length,
                max_document_length=args.max_document_length,
                document_instruction=args.document_instruction,
                device=device,
                rank=rank,
                world_size=world_size,
            )
    primary_evaluation = evaluations[str(evaluation_sizes[0])]

    if dist.is_initialized():
        gathered: list[set[int] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, training_seen_ids)
        training_seen_ids = set().union(*(item or set() for item in gathered))
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    audit = physical_pool.audit(unwrapped.backbone, training_seen_ids)
    if audit["evaluation_ids_seen_during_training"] != 0:
        raise AssertionError("Validation or test physical IDs leaked into training")
    if audit["evaluation_input_row_max_change"] != 0.0:
        raise AssertionError("Held-out input embedding rows changed")
    if audit["evaluation_output_row_max_change"] != 0.0:
        raise AssertionError("Held-out output vocabulary rows changed")

    if is_main:
        if not args.eval_only:
            unwrapped.backbone.save_pretrained(
                output_dir / "lora", save_embedding_layers=False
            )
            tokenizer.save_pretrained(output_dir / "tokenizer")
            torch.save(unwrapped.compiler.state_dict(), output_dir / "compiler.pt")
        results: dict[str, Any] = {
            "config": vars(args),
            "world_size": world_size,
            "completed_steps": step,
            "completed_epochs": completed_epochs,
            "recent_train_loss": sum(recent_losses) / max(1, len(recent_losses)),
            "validation": primary_evaluation["validation"],
            "test": primary_evaluation["test"],
            "evaluations_by_registry_size": evaluations,
            "physical_token_audit": audit,
        }
        (output_dir / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
