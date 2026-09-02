#!/usr/bin/env python3
"""Run a real full-parameter Native Late-Bound sequence-SFT smoke or train."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latebound_sequence_sft.data import render_sequence, sha256_file  # noqa: E402
from latebound_sequence_sft.model import (  # noqa: E402
    NativeBundleCompiler,
    NativeSequenceSFT,
)


# Tokenization is independent of the trainable backbone.  Keep the integer
# sequences in host memory so repeated ToolBench documents/queries do not
# invoke the Python tokenizer on every optimizer step.
_DOCUMENT_TOKEN_CACHE: dict[tuple[str, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
_QUERY_TOKEN_CACHE: dict[tuple[str, int], tuple[int, ...]] = {}


@dataclass(frozen=True)
class Example:
    query: str
    document_id: str
    loss_weight: float


def _load_manifest(path: Path) -> tuple[list[Example], dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "native_latebound_standard_sequence_sft_manifest":
        raise ValueError("Unexpected manifest kind")
    documents = {str(row["docid"]): row for row in payload["documents"]}
    if len(documents) != payload["exact_identity_count"]:
        raise ValueError("Manifest document identities are not one-to-one")
    examples = [
        Example(
            query=str(row["query"]),
            document_id=str(row["document_id"]),
            loss_weight=1.0 / float(row["positive_document_count"]),
        )
        for row in payload["examples"]
    ]
    if not examples:
        raise ValueError("Manifest contains no training examples")
    for item in examples:
        if item.document_id not in documents:
            raise ValueError(f"Unknown document id in manifest: {item.document_id}")
    return examples, documents, payload


def _tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"local_files_only": True}
    config_path = model_path / "tokenizer_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config.get("extra_special_tokens"), list):
            kwargs["extra_special_tokens"] = {}
            kwargs["additional_special_tokens"] = config["extra_special_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_backbone(model_path: Path, device: torch.device, *, gradient_checkpointing: bool):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("Backbone has no gradient-checkpointing support")
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError as exc:
            raise RuntimeError("Non-reentrant gradient checkpointing is required") from exc
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model.to(device)


def _query_texts(examples: list[Example], query_root: Path | None) -> list[str]:
    values = {item.query for item in examples}
    if query_root is not None:
        for path in sorted(query_root.glob("G[123]/test_G*_*.query.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    values.add(line.split("\t", 1)[-1].strip())
    return sorted(values)


def _reserved_pool(tokenizer, examples: list[Example], query_root: Path | None, count: int) -> list[int]:
    if count < 1:
        raise ValueError("reserved pool size must be positive")
    observed: set[int] = set()
    for text in _query_texts(examples, query_root):
        observed.update(tokenizer(text, add_special_tokens=False)["input_ids"])
    special = {int(value) for value in tokenizer.all_special_ids if value is not None}
    vocab_size = int(tokenizer.vocab_size)
    available = [
        value
        for value in range(vocab_size - 1, -1, -1)
        if value not in observed and value not in special
    ]
    if len(available) < count:
        raise ValueError(
            f"Only {len(available)} collision-free vocabulary rows remain; need {count}"
        )
    return sorted(available[:count])


def _batch(
    rows: list[Example],
    documents: dict[str, dict[str, Any]],
    tokenizer,
    physical_by_doc: dict[str, int],
    reserved_ids: list[int],
    *,
    max_length: int,
    max_document_length: int,
    device: torch.device,
) -> dict[str, Any]:
    unique_ids = sorted({row.document_id for row in rows}, key=lambda value: (int(value), value))
    document_rows = []
    for value in unique_ids:
        key = (value, max_document_length)
        cached = _DOCUMENT_TOKEN_CACHE.get(key)
        if cached is None:
            text = json.dumps(documents[value]["document"], ensure_ascii=False, sort_keys=True)
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_document_length,
                return_attention_mask=True,
            )
            cached = (
                tuple(int(item) for item in encoded["input_ids"]),
                tuple(int(item) for item in encoded["attention_mask"]),
            )
            _DOCUMENT_TOKEN_CACHE[key] = cached
        document_rows.append(cached)
    document_width = max(len(ids) for ids, _ in document_rows)
    document_input_ids = torch.full(
        (len(document_rows), document_width), int(tokenizer.pad_token_id), dtype=torch.long
    )
    document_attention = torch.zeros_like(document_input_ids)
    for row_index, (ids, mask) in enumerate(document_rows):
        document_input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        document_attention[row_index, : len(mask)] = torch.tensor(mask, dtype=torch.long)
    document_tokens = {
        "input_ids": document_input_ids.to(device),
        "attention_mask": document_attention.to(device),
    }
    document_index = torch.tensor(
        [unique_ids.index(row.document_id) for row in rows], dtype=torch.long, device=device
    )
    physical_ids = torch.tensor(
        [physical_by_doc[row.document_id] for row in rows], dtype=torch.long, device=device
    )
    prompt_ids: list[list[int]] = []
    targets: list[tuple[int, int]] = []
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is None or eot == tokenizer.unk_token_id:
        eot = tokenizer.eos_token_id
    if eot is None:
        raise ValueError("Tokenizer has neither <|eot_id|> nor eos token")
    for row, physical_id in zip(rows, physical_ids.tolist()):
        prompt, _ = render_sequence(row.query, "")
        query_key = (row.query, max_length)
        values = _QUERY_TOKEN_CACHE.get(query_key)
        if values is None:
            values = tuple(
                int(item) for item in tokenizer(prompt, add_special_tokens=False)["input_ids"]
            )
            _QUERY_TOKEN_CACHE[query_key] = values
        if len(values) + 2 > max_length:
            values = values[-(max_length - 2) :]
        prompt_ids.append(values)
        targets.append((physical_id, int(eot)))
    width = max(len(value) for value in prompt_ids) + 2
    input_ids = torch.full(
        (len(rows), width), int(tokenizer.pad_token_id), dtype=torch.long, device=device
    )
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for index, (values, target) in enumerate(zip(prompt_ids, targets)):
        length = len(values)
        input_ids[index, :length] = torch.tensor(values, dtype=torch.long, device=device)
        input_ids[index, length : length + 2] = torch.tensor(target, dtype=torch.long, device=device)
        labels[index, length : length + 2] = input_ids[index, length : length + 2]
        attention[index, : length + 2] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "document_tokens": document_tokens,
        "physical_ids": physical_ids,
        "document_index": document_index,
        "loss_weights": torch.tensor(
            [row.loss_weight for row in rows], dtype=torch.float32, device=device
        ),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _epoch_step_groups(
    example_count: int,
    batch_size: int,
    gradient_accumulation: int,
    epochs: int,
    seed: int,
) -> list[tuple[int, list[list[int]]]]:
    """Build deterministic optimizer-step groups that really consume epochs.

    The final micro-batch of an epoch may be short, but no sample is silently
    discarded or repeated before the next epoch begins.  A positive max-step
    limit is applied by the caller for smoke runs.
    """
    if example_count < 1 or batch_size < 1 or gradient_accumulation < 1 or epochs < 1:
        raise ValueError("schedule dimensions must be positive")
    groups: list[tuple[int, list[list[int]]]] = []
    for epoch in range(epochs):
        order = list(range(example_count))
        random.Random(seed + epoch).shuffle(order)
        batches = [
            order[start : start + batch_size]
            for start in range(0, example_count, batch_size)
        ]
        for start in range(0, len(batches), gradient_accumulation):
            groups.append((epoch, batches[start : start + gradient_accumulation]))
    return groups


def _distributed_epoch_step_groups(
    example_count: int,
    batch_size: int,
    gradient_accumulation: int,
    epochs: int,
    seed: int,
    world_size: int,
    rank: int,
) -> list[tuple[int, list[list[int | None]]]]:
    """Shard global batches while keeping every rank on the same step count.

    A final incomplete global batch is padded with ``None`` entries.  The
    caller turns those entries into zero-weight examples, so every source
    example is consumed once per epoch without an unequal-rank deadlock.
    """
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("rank must be within a positive world size")
    groups: list[tuple[int, list[list[int | None]]]] = []
    global_batch_size = batch_size * world_size
    for epoch in range(epochs):
        order = list(range(example_count))
        random.Random(seed + epoch).shuffle(order)
        global_batches: list[list[int | None]] = []
        for start in range(0, example_count, global_batch_size):
            batch: list[int | None] = order[start : start + global_batch_size]
            batch.extend([None] * (global_batch_size - len(batch)))
            global_batches.append(batch)
        local_batches = [
            batch[rank * batch_size : (rank + 1) * batch_size]
            for batch in global_batches
        ]
        for start in range(0, len(local_batches), gradient_accumulation):
            groups.append((epoch, local_batches[start : start + gradient_accumulation]))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="maximum optimizer steps; 0 means all steps implied by epochs",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-document-length", type=int, default=512)
    parser.add_argument("--compiler-rank", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--reserved-pool-size", type=int, default=49936)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--deepspeed-config",
        type=Path,
        help="explicit ZeRO-3 config for multi-GPU full-parameter training",
    )
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 0 or args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("steps must be non-negative; epochs, batch-size and gradient-accumulation must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.native_root.is_dir():
        raise FileNotFoundError(args.native_root)
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.deepspeed_config is not None
    if world_size_env > 1 and not distributed:
        raise RuntimeError(
            "WORLD_SIZE>1 requires --deepspeed-config; refusing duplicate single-process training"
        )
    if distributed and args.expected_world_size is not None and world_size_env != args.expected_world_size:
        raise RuntimeError(
            f"expected {args.expected_world_size} ranks, got WORLD_SIZE={world_size_env}"
        )
    if distributed and not args.deepspeed_config.is_file():
        raise FileNotFoundError(args.deepspeed_config)
    sys.path.insert(0, str(args.native_root / "src"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world_size_env = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entry")
    examples, documents, manifest = _load_manifest(args.manifest)
    tokenizer = _tokenizer(args.model_path)
    backbone = _load_backbone(
        args.model_path, device, gradient_checkpointing=args.gradient_checkpointing
    )
    reserved_ids = _reserved_pool(
        tokenizer, examples, args.query_root, args.reserved_pool_size
    )
    ordered_docs = sorted(documents, key=lambda value: (int(value), value))
    physical_by_doc = {
        docid: address for docid, address in zip(ordered_docs, reserved_ids)
    }
    output_norm = float(
        backbone.get_output_embeddings().weight.detach().float().norm(dim=-1).mean()
    )
    compiler = NativeBundleCompiler(
        int(backbone.config.hidden_size),
        args.compiler_rank,
        output_norm,
        args.memory_slots,
    ).to(device)
    model = NativeSequenceSFT(backbone, compiler, reserved_ids).to(device)
    # DeepSpeed ZeRO-3 replaces parameter storage with local shards during
    # initialization. Capture the full-model audit before that transformation;
    # counting partitioned tensors afterwards would incorrectly report zero.
    full_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameter_count = full_parameter_count - trainable_parameter_count
    backbone_requires_grad = all(parameter.requires_grad for parameter in backbone.parameters())
    compiler_requires_grad = all(parameter.requires_grad for parameter in compiler.parameters())
    if full_parameter_count <= 0 or trainable_parameter_count <= 0:
        raise RuntimeError("full-parameter audit produced an empty model")
    if not backbone_requires_grad or not compiler_requires_grad:
        raise RuntimeError("backbone and compiler must both remain trainable")
    optimizer = None
    engine = None
    optimizer_parameter_group_count = 0
    if distributed:
        import deepspeed

        config = json.loads(args.deepspeed_config.read_text(encoding="utf-8"))
        expected_micro = int(args.batch_size)
        expected_global = expected_micro * world_size_env
        for key, expected in (
            ("train_micro_batch_size_per_gpu", expected_micro),
            ("gradient_accumulation_steps", 1),
            ("train_batch_size", expected_global),
        ):
            value = config.get(key)
            if value in (None, "auto") or int(value) != expected:
                raise ValueError(
                    f"DeepSpeed config must pin {key}={expected}; got {value!r}"
                )
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=str(args.deepspeed_config),
        )
        model = engine.module
        optimizer_parameter_group_count = len(optimizer.param_groups)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        optimizer_parameter_group_count = len(optimizer.param_groups)
    model.train()
    schedule = (
        _distributed_epoch_step_groups(
            len(examples),
            args.batch_size,
            args.gradient_accumulation,
            args.epochs,
            args.seed,
            world_size_env,
            rank,
        )
        if distributed
        else _epoch_step_groups(
            len(examples), args.batch_size, args.gradient_accumulation, args.epochs, args.seed
        )
    )
    if args.steps:
        schedule = schedule[: args.steps]
    if not schedule:
        raise ValueError("the requested epoch/step schedule is empty")
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    examples_seen = 0
    padded_examples_seen = 0
    for step, (epoch, micro_batches) in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        loss_total = 0.0
        micro_count = len(micro_batches)
        step_tokens = 0
        step_sequence_tokens = 0
        step_document_tokens = 0
        step_sequence_width = 0
        step_document_width = 0
        diagnostics = None
        for indices in micro_batches:
            padded_examples_seen += sum(index is None for index in indices)
            rows = [
                examples[index]
                if index is not None
                else dataclasses.replace(examples[0], loss_weight=0.0)
                for index in indices
            ]
            examples_seen += sum(index is not None for index in indices)
            batch = _batch(
                rows,
                documents,
                tokenizer,
                physical_by_doc,
                reserved_ids,
                max_length=args.max_length,
                max_document_length=args.max_document_length,
                device=device,
            )
            train_model = engine if engine is not None else model
            loss, diagnostics = train_model(**batch)
            if engine is not None:
                engine.backward(loss / micro_count)
            else:
                (loss / micro_count).backward()
            loss_total += float(loss.detach())
            positive_rows = batch["loss_weights"].gt(0).unsqueeze(1)
            step_tokens += int(
                (batch["labels"].ne(-100) & positive_rows).sum().item()
            )
            step_sequence_tokens += int(batch["attention_mask"].sum().item())
            step_document_tokens += int(
                batch["document_tokens"]["attention_mask"].sum().item()
            )
            step_sequence_width = max(
                step_sequence_width, int(batch["input_ids"].shape[1])
            )
            step_document_width = max(
                step_document_width,
                int(batch["document_tokens"]["input_ids"].shape[1]),
            )
        if diagnostics is None:
            raise AssertionError("optimizer step had no micro-batch")
        if engine is not None:
            engine.step()
            grad_norm = engine.get_global_grad_norm()
            if grad_norm is not None and not bool(
                torch.isfinite(torch.as_tensor(grad_norm, device=device))
            ):
                raise FloatingPointError("DeepSpeed gradient norm is non-finite")
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("Gradient norm is non-finite")
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            elapsed = time.monotonic() - step_started
            peak = torch.cuda.max_memory_allocated(device) / (1024**3)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            elapsed = time.monotonic() - step_started
            peak = 0.0
        global_examples_seen = examples_seen
        global_padded_examples_seen = padded_examples_seen
        if distributed:
            import torch.distributed as dist

            maximums = torch.tensor(
                [elapsed, peak, step_sequence_width, step_document_width],
                dtype=torch.float64,
                device=device,
            )
            totals = torch.tensor(
                [
                    step_tokens,
                    step_sequence_tokens,
                    step_document_tokens,
                    examples_seen,
                    padded_examples_seen,
                ],
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(maximums, op=dist.ReduceOp.MAX)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            elapsed, peak = (float(value) for value in maximums[:2].tolist())
            step_sequence_width = int(maximums[2].item())
            step_document_width = int(maximums[3].item())
            (
                step_tokens,
                step_sequence_tokens,
                step_document_tokens,
                global_examples_seen,
                global_padded_examples_seen,
            ) = (int(value) for value in totals.tolist())
        processed_tokens = step_sequence_tokens + step_document_tokens
        records.append(
            {
                "optimizer_step": step + 1,
                "loss": loss_total / micro_count,
                "step_seconds": elapsed,
                "peak_memory_gib": peak,
                "document_forward_count": diagnostics.document_forward_count,
                "distinct_document_count": diagnostics.distinct_document_count,
                "world_size": world_size_env,
                "rank": rank,
                "epoch": epoch + 1,
                "micro_batches": micro_count,
                "examples_seen_cumulative": global_examples_seen,
                "padded_examples_seen_cumulative": global_padded_examples_seen,
                "effective_target_tokens": step_tokens,
                "sequence_input_tokens": step_sequence_tokens,
                "document_input_tokens": step_document_tokens,
                "processed_input_tokens": processed_tokens,
                "tokens_per_second": processed_tokens / elapsed,
                "actual_sequence_length": step_sequence_width,
                "actual_document_length": step_document_width,
            }
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)
    if engine is not None:
        import torch.distributed as dist

        if rank == 0:
            args.output_dir.mkdir(parents=True)
        dist.barrier()
        engine.save_checkpoint(str(args.output_dir), tag="final")
        dist.barrier()
    else:
        args.output_dir.mkdir(parents=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            args.output_dir / "checkpoint.pt",
        )
    metadata = {
        "kind": "native_latebound_sequence_sft_training",
        "passed": True,
        "manifest_sha256": sha256_file(args.manifest),
        "model_path": str(args.model_path),
        "steps": len(records),
        "requested_steps": args.steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "global_batch_size": args.batch_size * world_size_env * args.gradient_accumulation,
        "configured_max_sequence_length": args.max_length,
        "configured_max_document_length": args.max_document_length,
        "compiler_rank": args.compiler_rank,
        "memory_slots": args.memory_slots,
        "full_parameter_count": full_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": frozen_parameter_count,
        "backbone_requires_grad": backbone_requires_grad,
        "compiler_requires_grad": compiler_requires_grad,
        "optimizer_parameter_group_count": optimizer_parameter_group_count,
        "optimizer_steps": len(records),
        "registration_optimizer_steps": 0,
        "reserved_pool_size": len(reserved_ids),
        "reserved_ids_sha256": hashlib.sha256(
            json.dumps(reserved_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "records": records,
        "distributed": distributed,
        "expected_world_size": args.expected_world_size,
        "deepspeed_config_sha256": (
            sha256_file(args.deepspeed_config) if args.deepspeed_config is not None else None
        ),
        "wall_seconds": time.monotonic() - started,
    }
    if rank == 0:
        _save_json(
            args.output_dir / "reserved_ids.json",
            {
                "kind": "native_latebound_reserved_physical_ids",
                "ids": reserved_ids,
                "sha256": hashlib.sha256(
                    json.dumps(reserved_ids, separators=(",", ":")).encode()
                ).hexdigest(),
            },
        )
        _save_json(args.output_dir / "training_metadata.json", metadata)
        digest = hashlib.sha256(
            (args.output_dir / "training_metadata.json").read_bytes()
        ).hexdigest()
        (args.output_dir / "COMPLETE").write_text(digest + "\n", encoding="ascii")
    if distributed:
        import torch.distributed as dist

        dist.barrier()


if __name__ == "__main__":
    main()
