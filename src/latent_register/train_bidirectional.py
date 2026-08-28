from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import (
    ToolExample,
    load_function_call_jsonl,
    stable_tool_split,
    tool_registry_key,
)
from .model import (
    GatedLayerwiseCrossAttentionMemory,
    GeneratedMemory,
    PhysicalOutputGenerator,
    ReadoutCrossAttentionMemory,
    TokenResamplerMemory,
)
from .physical_tokens import ReservedTokenPool
from .registry import DynamicRegistry
from .train import metrics_from_scores, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--total-slots", type=int, default=512)
    parser.add_argument("--train-slots", type=int, default=384)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--selection-epochs", type=int, default=30)
    parser.add_argument("--input-epochs", type=int, default=10)
    parser.add_argument("--schema-warmup-epochs", type=int, default=0)
    parser.add_argument("--input-memory-slots", type=int, default=1)
    parser.add_argument(
        "--tool-identity",
        choices=("name", "document"),
        default="name",
        help="Registry key. Document mode separates same-name tools with different APIs.",
    )
    parser.add_argument(
        "--input-memory-source",
        choices=("pooled", "token_resampler"),
        default="pooled",
    )
    parser.add_argument(
        "--tool-pooling",
        choices=("mean", "last", "mean_last_slots", "mean_schema_key_slots"),
        default="mean",
        help="Frozen document-state pooling used by registration generators.",
    )
    parser.add_argument(
        "--input-memory-interface",
        choices=(
            "prompt_slots",
            "readout_cross_attention",
            "gated_layerwise_cross_attention",
            "prompt_slots_plus_layerwise",
            "prompt_slots_plus_trajectory_layerwise",
        ),
        default="prompt_slots",
    )
    parser.add_argument(
        "--layerwise-memory-layers",
        default="auto:4",
        help="Comma-separated decoder indices or auto:N evenly spaced non-final layers.",
    )
    parser.add_argument("--layerwise-memory-max-gate", type=float, default=0.25)
    parser.add_argument("--selection-batch-size", type=int, default=32)
    parser.add_argument("--input-batch-size", type=int, default=2)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--selection-learning-rate", type=float, default=3e-4)
    parser.add_argument("--input-learning-rate", type=float, default=1e-3)
    parser.add_argument("--schema-loss-weight", type=float, default=0.0)
    parser.add_argument("--distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--max-encode-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=640)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-limit", type=int, default=60)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_examples(specifications: list[str], max_examples: int) -> list[ToolExample]:
    examples: list[ToolExample] = []
    for specification in specifications:
        data_path, answer_path = specification.split("::", maxsplit=1)
        examples.extend(load_function_call_jsonl(data_path, answer_path))
    examples = list({item.example_id: item for item in examples}.values())
    examples = [item for item in examples if item.target_arguments is not None]
    if max_examples:
        examples = examples[:max_examples]
    if len({item.tool_name for item in examples}) < 2:
        raise ValueError("At least two tools with argument targets are required")
    return examples


def resolve_layerwise_memory_layers(specification: str, depth: int) -> tuple[int, ...]:
    if depth < 2:
        raise ValueError("Layerwise memory requires at least two decoder layers")
    if specification.startswith("auto:"):
        count = int(specification.split(":", maxsplit=1)[1])
        if count < 1 or count >= depth:
            raise ValueError("auto layer count must be between one and depth - 1")
        indices = tuple(
            max(0, round((position + 1) * depth / (count + 1)) - 1)
            for position in range(count)
        )
    else:
        indices = tuple(int(value.strip()) for value in specification.split(","))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Layerwise memory layers must be unique and non-empty")
    if min(indices) < 0 or max(indices) >= depth:
        raise ValueError(f"Layerwise memory layers must be within [0, {depth - 1}]")
    return indices


def is_layerwise_interface(input_memory_interface: str) -> bool:
    return input_memory_interface in {
        "gated_layerwise_cross_attention",
        "prompt_slots_plus_layerwise",
        "prompt_slots_plus_trajectory_layerwise",
    }


def uses_prompt_slots(input_memory_interface: str) -> bool:
    return input_memory_interface in {
        "prompt_slots",
        "prompt_slots_plus_layerwise",
        "prompt_slots_plus_trajectory_layerwise",
    }


def uses_layerwise_for_ordinary_readback(input_memory_interface: str) -> bool:
    return input_memory_interface in {
        "gated_layerwise_cross_attention",
        "prompt_slots_plus_layerwise",
    }


def is_persistent_memory_interface(input_memory_interface: str) -> bool:
    return input_memory_interface in {
        "readout_cross_attention",
        "gated_layerwise_cross_attention",
        "prompt_slots_plus_layerwise",
        "prompt_slots_plus_trajectory_layerwise",
    }


def uses_persistent_memory_for_ordinary_readback(
    input_memory_interface: str,
) -> bool:
    return (
        input_memory_interface == "readout_cross_attention"
        or uses_layerwise_for_ordinary_readback(input_memory_interface)
    )


def render_chat(tokenizer, user_text: str, system_text: str | None = None) -> str:
    if system_text is None:
        system_text = "You produce tool-call arguments. Return only one compact JSON object."
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return f"System: {messages[0]['content']}\nUser: {user_text}\nAssistant:"


def selection_user_text(query: str) -> str:
    return f"Request: {query}\nSelect the tool token."


@torch.inference_mode()
def encode_texts(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    pooling: str,
    device: torch.device,
) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model.model(**tokens, use_cache=False).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        if pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            indices = tokens["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(len(hidden), device=device), indices]
        encoded.append(pooled.float().cpu())
    return torch.cat(encoded)


@torch.inference_mode()
def encode_text_views(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    means: list[torch.Tensor] = []
    lasts: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model.model(**tokens, use_cache=False).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        means.append(
            ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
            .float()
            .cpu()
        )
        indices = tokens["attention_mask"].sum(dim=1) - 1
        lasts.append(
            hidden[torch.arange(len(hidden), device=device), indices].float().cpu()
        )
    return torch.cat(means), torch.cat(lasts)


@torch.inference_mode()
def encode_schema_key_views(
    model,
    tokenizer,
    texts: list[str],
    schema_keys: list[tuple[str, ...]],
    num_slots: int,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(texts) != len(schema_keys):
        raise ValueError("Tool texts and schema-key lists differ in length")
    if any(len(keys) > num_slots for keys in schema_keys):
        raise ValueError("A tool schema has more keys than registered memory slots")

    means: list[torch.Tensor] = []
    anchors: list[torch.Tensor] = []
    input_embeddings = model.get_input_embeddings()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model.model(**tokens, use_cache=False).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        batch_means = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        means.append(batch_means.float().cpu())

        for row, keys in enumerate(schema_keys[start : start + batch_size]):
            row_anchors: list[torch.Tensor] = []
            for key in keys:
                key_tokens = tokenizer(
                    key,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].to(device)
                if key_tokens.numel() == 0:
                    raise ValueError(f"Schema key tokenized to an empty sequence: {key!r}")
                row_anchors.append(
                    input_embeddings(key_tokens).mean(dim=1).squeeze(0).float()
                )
            row_anchors.extend(
                batch_means[row].float() for _ in range(num_slots - len(row_anchors))
            )
            anchors.append(torch.stack(row_anchors).cpu())
    return torch.cat(means), torch.stack(anchors)


def build_frozen_features(
    model,
    tokenizer,
    examples: list[ToolExample],
    batch_size: int,
    max_length: int,
    device: torch.device,
    include_tool_sequences: bool = False,
    tool_identity: str = "name",
    tool_pooling: str = "mean",
    input_memory_slots: int = 8,
) -> dict[str, Any]:
    query_prompts = [
        render_chat(tokenizer, selection_user_text(item.query))
        for item in examples
    ]
    documents = {
        tool_registry_key(item, tool_identity): item.tool_document for item in examples
    }
    schema_keys_by_tool: dict[str, tuple[str, ...]] = {}
    for item in examples:
        key = tool_registry_key(item, tool_identity)
        schema_keys = tuple(sorted(str(name) for name in (item.schema_arguments or {})))
        previous = schema_keys_by_tool.setdefault(key, schema_keys)
        if previous != schema_keys:
            raise ValueError(f"Conflicting schema keys for registered tool {key}")
    tool_keys = sorted(documents)
    tool_texts = [
        f"Represent this tool for registration.\n{documents[key]}" for key in tool_keys
    ]
    tool_sequences: list[torch.Tensor] | None = None
    tool_values: list[torch.Tensor] | None = None
    input_tool_features: torch.Tensor
    if include_tool_sequences:
        pooled_tools: list[torch.Tensor] = []
        tool_sequences = []
        tool_values = []
        for start in range(0, len(tool_texts), batch_size):
            tokens = tokenizer(
                tool_texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model.model(**tokens, use_cache=False).last_hidden_state
            input_values = model.get_input_embeddings()(tokens["input_ids"]).detach()
            lengths = tokens["attention_mask"].sum(dim=1).tolist()
            for row, length in enumerate(lengths):
                sequence = hidden[row, : int(length)].detach().cpu()
                tool_sequences.append(sequence)
                tool_values.append(input_values[row, : int(length)].cpu())
                pooled_tools.append(
                    sequence.float().mean(dim=0)
                    if tool_pooling == "mean"
                    else sequence[-1].float()
                )
        tool_features = torch.stack(pooled_tools)
        input_tool_features = tool_features
    elif tool_pooling == "mean_last_slots":
        mean_features, last_features = encode_text_views(
            model,
            tokenizer,
            tool_texts,
            batch_size,
            max_length,
            device,
        )
        tool_features = mean_features
        input_tool_features = torch.stack(
            [mean_features, last_features], dim=1
        )
    elif tool_pooling == "mean_schema_key_slots":
        mean_features, schema_anchors = encode_schema_key_views(
            model,
            tokenizer,
            tool_texts,
            [schema_keys_by_tool[key] for key in tool_keys],
            input_memory_slots,
            batch_size,
            max_length,
            device,
        )
        tool_features = mean_features
        input_tool_features = torch.cat(
            [mean_features.unsqueeze(1), schema_anchors], dim=1
        )
    else:
        tool_features = encode_texts(
            model,
            tokenizer,
            tool_texts,
            batch_size,
            max_length,
            tool_pooling,
            device,
        )
        input_tool_features = tool_features
    return {
        "query": encode_texts(
            model, tokenizer, query_prompts, batch_size, max_length, "last", device
        ),
        "tool": tool_features,
        "input_tool": input_tool_features,
        "tool_sequences": tool_sequences,
        "tool_values": tool_values,
        "tool_keys": tool_keys,
        "schema_key_counts": [len(schema_keys_by_tool[key]) for key in tool_keys],
    }


def pad_tool_sequences(
    sequences: list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(sequence) for sequence in sequences)
    hidden_size = sequences[0].shape[-1]
    states = torch.zeros(
        len(sequences),
        max_length,
        hidden_size,
        dtype=sequences[0].dtype,
        device=device,
    )
    mask = torch.zeros(len(sequences), max_length, dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        states[row, : len(sequence)] = sequence.to(device)
        mask[row, : len(sequence)] = True
    return states, mask


def make_input_overrides(
    condition: str,
    tool_names: list[str],
    tool_features: dict[str, torch.Tensor],
    tool_sequences: dict[str, torch.Tensor] | None,
    tool_values: dict[str, torch.Tensor] | None,
    input_generator: GeneratedMemory
    | TokenResamplerMemory
    | ReadoutCrossAttentionMemory
    | GatedLayerwiseCrossAttentionMemory
    | DistributedDataParallel,
    input_memory_source: str,
    input_memory_interface: str,
    input_embedding_norm: float,
    device: torch.device,
) -> torch.Tensor | None:
    base_generator = unwrap_module(input_generator)
    if condition not in {"registered", "raw_latent"}:
        return None
    if is_persistent_memory_interface(input_memory_interface):
        expected = (
            ReadoutCrossAttentionMemory
            if input_memory_interface == "readout_cross_attention"
            else GatedLayerwiseCrossAttentionMemory
        )
        if not isinstance(base_generator, expected):
            raise ValueError("Persistent interface has an incompatible memory generator")
        if not uses_prompt_slots(input_memory_interface):
            return None
    latents = torch.stack([tool_features[name] for name in tool_names]).to(device)
    if condition == "raw_latent":
        return raw_registered_memory(
            latents, base_generator.num_slots, input_embedding_norm
        )
    if uses_prompt_slots(input_memory_interface):
        return input_generator(latents)
    if input_memory_source == "token_resampler":
        if (
            tool_sequences is None
            or tool_values is None
            or not isinstance(base_generator, TokenResamplerMemory)
        ):
            raise ValueError("Token-resampler features and generator are required")
        states, mask = pad_tool_sequences([tool_sequences[name] for name in tool_names], device)
        values, value_mask = pad_tool_sequences(
            [tool_values[name] for name in tool_names], device
        )
        if not torch.equal(mask, value_mask):
            raise ValueError("Token-resampler keys and values have different lengths")
        return input_generator(states, mask, values)
    if not isinstance(base_generator, GeneratedMemory):
        raise ValueError("Pooled memory source requires GeneratedMemory")
    return input_generator(latents)


def stack_tool_latents(
    tool_names: list[str],
    tool_features: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([tool_features[name] for name in tool_names]).to(device)


def raw_registered_memory(
    latents: torch.Tensor,
    num_slots: int,
    input_embedding_norm: float,
) -> torch.Tensor:
    if latents.ndim == 3:
        if latents.shape[1] < 2:
            raise ValueError("Structured raw memory expects multiple views")
        latents = latents[:, 0]
    raw = F.normalize(latents.float(), dim=-1) * input_embedding_norm
    return raw.unsqueeze(1).expand(-1, num_slots, -1)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    if not mask.any():
        raise ValueError("Registered readout batch has no target labels")
    return F.cross_entropy(shifted_logits[mask], shifted_labels[mask])


def registered_readout_logits(
    model,
    input_generator: ReadoutCrossAttentionMemory | DistributedDataParallel,
    tool_latents: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    memory_override: torch.Tensor | None = None,
) -> torch.Tensor:
    base_generator = unwrap_module(input_generator)
    if not isinstance(base_generator, ReadoutCrossAttentionMemory):
        raise ValueError("Registered readout requires ReadoutCrossAttentionMemory")
    with torch.no_grad():
        hidden = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
    if memory_override is None:
        adapted = input_generator(tool_latents, hidden)
    else:
        adapted = base_generator.adapt(hidden, memory_override)
    return model.get_output_embeddings()(adapted.to(model.dtype))


def registered_forward(
    model,
    input_generator,
    input_memory_interface: str,
    tool_latents: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> SimpleNamespace:
    if input_memory_interface in {
        "prompt_slots",
        "gated_layerwise_cross_attention",
        "prompt_slots_plus_layerwise",
        "prompt_slots_plus_trajectory_layerwise",
    }:
        if (
            is_layerwise_interface(input_memory_interface)
            and torch.is_grad_enabled()
            and not inputs_embeds.requires_grad
        ):
            # Reentrant checkpointing needs one differentiable input to replay
            # decoder layers that contain the memory hooks.
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        return model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    logits = registered_readout_logits(
        model,
        input_generator,
        tool_latents,
        inputs_embeds,
        attention_mask,
    )
    return SimpleNamespace(logits=logits, loss=causal_lm_loss(logits, labels))


def layerwise_memory_for_condition(
    input_generator: GatedLayerwiseCrossAttentionMemory,
    condition: str,
    tool_latents: torch.Tensor,
    input_embedding_norm: float,
) -> torch.Tensor:
    if condition == "registered":
        return input_generator.registered_memory(tool_latents)
    if condition == "raw_latent":
        return raw_registered_memory(
            tool_latents, input_generator.num_slots, input_embedding_norm
        )
    raise ValueError(f"Condition {condition} does not use persistent memory")


def tool_target_json(example: ToolExample, target_kind: str = "arguments") -> str:
    value = (
        example.schema_arguments if target_kind == "schema" else example.target_arguments
    )
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def condition_user_text(
    example: ToolExample,
    condition: str,
    token_string: str | None,
    target_kind: str = "arguments",
) -> str:
    if target_kind == "schema":
        instruction = (
            "Return only one JSON object containing every parameter name for the selected "
            "tool, with null as each value."
        )
    else:
        instruction = "Return only the JSON object containing arguments for the selected tool."
    if condition == "full_document":
        return f"Tool documentation:\n{example.tool_document}\nRequest: {example.query}\n{instruction}"
    if condition == "query_only":
        return f"Request: {example.query}\n{instruction}"
    if token_string is None:
        raise ValueError(f"Condition {condition} requires a physical token")
    return f"Request: {example.query}\nSelected tool token: {token_string}\n{instruction}"


def prepare_batch(
    model,
    tokenizer,
    examples: list[ToolExample],
    condition: str,
    slot_ids: list[int | None],
    overrides: torch.Tensor | None,
    max_prompt_length: int,
    max_target_length: int,
    device: torch.device,
    include_targets: bool = True,
    target_kind: str = "arguments",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding = model.get_input_embeddings()
    sequences: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for index, (example, slot_id) in enumerate(zip(examples, slot_ids)):
        token_string = tokenizer.convert_ids_to_tokens(slot_id) if slot_id is not None else None
        system_text = None
        if target_kind == "schema":
            system_text = "You reproduce tool parameter schemas as compact JSON objects."
        prompt = render_chat(
            tokenizer,
            condition_user_text(example, condition, token_string, target_kind),
            system_text,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
        prompt_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long)
        prompt_embeddings = embedding(prompt_tensor).detach()
        if slot_id is not None and overrides is not None:
            positions = (prompt_tensor == slot_id).nonzero(as_tuple=False).flatten()
            if len(positions) != 1:
                raise ValueError(f"Expected one occurrence of slot {slot_id}, found {len(positions)}")
            position = int(positions.item())
            override_rows = overrides[index]
            if override_rows.ndim == 1:
                override_rows = override_rows.unsqueeze(0)
            prompt_embeddings = torch.cat(
                [
                    prompt_embeddings[:position],
                    override_rows.to(prompt_embeddings.dtype),
                    prompt_embeddings[position + 1 :],
                ]
            )

        if include_targets:
            target_ids = tokenizer(
                tool_target_json(example, target_kind),
                add_special_tokens=False,
                truncation=True,
                max_length=max_target_length - 1,
            ).input_ids + [tokenizer.eos_token_id]
            target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
            target_embeddings = embedding(target_tensor).detach()
            sequences.append(torch.cat([prompt_embeddings, target_embeddings]))
            labels.append(
                torch.cat(
                    [
                        torch.full(
                            (len(prompt_embeddings),), -100, device=device, dtype=torch.long
                        ),
                        target_tensor,
                    ]
                )
            )
        else:
            sequences.append(prompt_embeddings)
            labels.append(
                torch.full((len(prompt_embeddings),), -100, device=device, dtype=torch.long)
            )

    max_length = max(len(sequence) for sequence in sequences)
    hidden_size = sequences[0].shape[-1]
    dtype = sequences[0].dtype
    padded = torch.zeros(len(sequences), max_length, hidden_size, device=device, dtype=dtype)
    attention = torch.zeros(len(sequences), max_length, device=device, dtype=torch.long)
    padded_labels = torch.full(
        (len(sequences), max_length), -100, device=device, dtype=torch.long
    )
    for row, (sequence, target_labels) in enumerate(zip(sequences, labels)):
        padded[row, : len(sequence)] = sequence
        attention[row, : len(sequence)] = 1
        padded_labels[row, : len(target_labels)] = target_labels
    return padded, attention, padded_labels


def prepare_trajectory_batch(
    model,
    tokenizer,
    examples: list[ToolExample],
    memory: torch.Tensor,
    max_prompt_length: int,
    max_target_length: int,
    device: torch.device,
    include_targets: bool = True,
    target_kind: str = "arguments",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place registered memory at the first assistant action position."""
    embedding = model.get_input_embeddings()
    sequences: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for index, example in enumerate(examples):
        system_text = None
        if target_kind == "schema":
            system_text = "You select a tool and reproduce its parameter schema as JSON."
        prompt = render_chat(tokenizer, selection_user_text(example.query), system_text)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
        prompt_tensor = torch.tensor(prompt_ids, device=device, dtype=torch.long)
        prompt_embeddings = embedding(prompt_tensor).detach()
        memory_rows = memory[index]
        if memory_rows.ndim == 1:
            memory_rows = memory_rows.unsqueeze(0)
        prefix_embeddings = torch.cat(
            [prompt_embeddings, memory_rows.to(prompt_embeddings.dtype)]
        )

        if include_targets:
            target_ids = tokenizer(
                tool_target_json(example, target_kind),
                add_special_tokens=False,
                truncation=True,
                max_length=max_target_length - 1,
            ).input_ids + [tokenizer.eos_token_id]
            target_tensor = torch.tensor(target_ids, device=device, dtype=torch.long)
            target_embeddings = embedding(target_tensor).detach()
            sequences.append(torch.cat([prefix_embeddings, target_embeddings]))
            labels.append(
                torch.cat(
                    [
                        torch.full(
                            (len(prefix_embeddings),),
                            -100,
                            device=device,
                            dtype=torch.long,
                        ),
                        target_tensor,
                    ]
                )
            )
        else:
            sequences.append(prefix_embeddings)
            labels.append(
                torch.full(
                    (len(prefix_embeddings),), -100, device=device, dtype=torch.long
                )
            )

    max_length = max(len(sequence) for sequence in sequences)
    hidden_size = sequences[0].shape[-1]
    dtype = sequences[0].dtype
    padded = torch.zeros(len(sequences), max_length, hidden_size, device=device, dtype=dtype)
    attention = torch.zeros(len(sequences), max_length, device=device, dtype=torch.long)
    padded_labels = torch.full(
        (len(sequences), max_length), -100, device=device, dtype=torch.long
    )
    for row, (sequence, target_labels) in enumerate(zip(sequences, labels)):
        padded[row, : len(sequence)] = sequence
        attention[row, : len(sequence)] = 1
        padded_labels[row, : len(target_labels)] = target_labels
    return padded, attention, padded_labels


def teacher_forced_metrics(
    model,
    tokenizer,
    examples: list[ToolExample],
    condition: str,
    slot_by_tool: dict[str, int],
    tool_features: dict[str, torch.Tensor],
    tool_sequences: dict[str, torch.Tensor] | None,
    tool_values: dict[str, torch.Tensor] | None,
    input_generator: GeneratedMemory
    | TokenResamplerMemory
    | ReadoutCrossAttentionMemory
    | GatedLayerwiseCrossAttentionMemory,
    input_memory_source: str,
    input_memory_interface: str,
    input_embedding_norm: float,
    batch_size: int,
    max_prompt_length: int,
    max_target_length: int,
    device: torch.device,
    target_kind: str = "arguments",
    tool_identity: str = "name",
) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tool_keys = [tool_registry_key(item, tool_identity) for item in batch]
        tool_latents = stack_tool_latents(tool_keys, tool_features, device)
        slots = [slot_by_tool[key] for key in tool_keys]
        overrides = make_input_overrides(
            condition,
            tool_keys,
            tool_features,
            tool_sequences,
            tool_values,
            input_generator,
            input_memory_source,
            input_memory_interface,
            input_embedding_norm,
            device,
        )
        effective_slots: list[int | None]
        if condition in {"full_document", "query_only"}:
            effective_slots = [None] * len(batch)
        else:
            effective_slots = slots
        embeds, attention, labels = prepare_batch(
            model,
            tokenizer,
            batch,
            condition,
            effective_slots,
            overrides,
            max_prompt_length,
            max_target_length,
            device,
            target_kind=target_kind,
        )
        with torch.inference_mode():
            if (
                input_memory_interface == "readout_cross_attention"
                and condition in {"registered", "raw_latent"}
            ):
                base_generator = unwrap_module(input_generator)
                memory_override = None
                if condition == "raw_latent":
                    memory_override = raw_registered_memory(
                        tool_latents,
                        base_generator.num_slots,
                        input_embedding_norm,
                    )
                logits = registered_readout_logits(
                    model,
                    input_generator,
                    tool_latents,
                    embeds,
                    attention,
                    memory_override,
                )
            elif (
                uses_layerwise_for_ordinary_readback(input_memory_interface)
                and condition in {"registered", "raw_latent"}
            ):
                if not isinstance(
                    input_generator, GatedLayerwiseCrossAttentionMemory
                ):
                    raise RuntimeError("Layerwise memory generator is unavailable")
                memory = (
                    overrides
                    if uses_prompt_slots(input_memory_interface)
                    and overrides is not None
                    else layerwise_memory_for_condition(
                        input_generator,
                        condition,
                        tool_latents,
                        input_embedding_norm,
                    )
                )
                with input_generator.activate(memory):
                    logits = model(
                        inputs_embeds=embeds,
                        attention_mask=attention,
                        use_cache=False,
                    ).logits
            else:
                logits = model(
                    inputs_embeds=embeds,
                    attention_mask=attention,
                    use_cache=False,
                ).logits
        shifted_logits = logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100
        total_nll += float(
            F.cross_entropy(
                shifted_logits[mask], shifted_labels[mask], reduction="sum"
            )
        )
        correct += int((shifted_logits[mask].argmax(dim=-1) == shifted_labels[mask]).sum())
        total_tokens += int(mask.sum())
    mean_nll = total_nll / max(1, total_tokens)
    return {
        "nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
        "token_accuracy": correct / max(1, total_tokens),
        "target_tokens": total_tokens,
    }


def trajectory_teacher_forced_metrics(
    model,
    tokenizer,
    examples: list[ToolExample],
    tool_features: dict[str, torch.Tensor],
    tool_sequences: dict[str, torch.Tensor] | None,
    tool_values: dict[str, torch.Tensor] | None,
    input_generator: GeneratedMemory
    | TokenResamplerMemory
    | ReadoutCrossAttentionMemory
    | GatedLayerwiseCrossAttentionMemory,
    input_memory_source: str,
    input_memory_interface: str,
    input_embedding_norm: float,
    batch_size: int,
    max_prompt_length: int,
    max_target_length: int,
    device: torch.device,
    tool_identity: str = "name",
) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tool_keys = [tool_registry_key(item, tool_identity) for item in batch]
        tool_latents = stack_tool_latents(tool_keys, tool_features, device)
        if input_memory_interface in {
            "readout_cross_attention",
            "gated_layerwise_cross_attention",
        }:
            memory = torch.empty(
                (len(batch), 0, model.config.hidden_size),
                device=device,
                dtype=model.dtype,
            )
        else:
            memory = make_input_overrides(
                "registered",
                tool_keys,
                tool_features,
                tool_sequences,
                tool_values,
                input_generator,
                input_memory_source,
                input_memory_interface,
                input_embedding_norm,
                device,
            )
            if memory is None:
                raise RuntimeError("Trajectory readback requires registered memory")
        embeds, attention, labels = prepare_trajectory_batch(
            model,
            tokenizer,
            batch,
            memory,
            max_prompt_length,
            max_target_length,
            device,
        )
        with torch.inference_mode():
            if input_memory_interface == "readout_cross_attention":
                logits = registered_readout_logits(
                    model,
                    input_generator,
                    tool_latents,
                    embeds,
                    attention,
                )
            elif is_layerwise_interface(input_memory_interface):
                if not isinstance(
                    input_generator, GatedLayerwiseCrossAttentionMemory
                ):
                    raise RuntimeError("Layerwise memory generator is unavailable")
                registered_memory = (
                    memory
                    if uses_prompt_slots(input_memory_interface)
                    else input_generator.registered_memory(tool_latents)
                )
                with input_generator.activate(registered_memory):
                    logits = model(
                        inputs_embeds=embeds,
                        attention_mask=attention,
                        use_cache=False,
                    ).logits
            else:
                logits = model(
                    inputs_embeds=embeds,
                    attention_mask=attention,
                    use_cache=False,
                ).logits
        shifted_logits = logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        mask = shifted_labels != -100
        total_nll += float(
            F.cross_entropy(shifted_logits[mask], shifted_labels[mask], reduction="sum")
        )
        correct += int((shifted_logits[mask].argmax(dim=-1) == shifted_labels[mask]).sum())
        total_tokens += int(mask.sum())
    mean_nll = total_nll / max(1, total_tokens)
    return {
        "nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
        "token_accuracy": correct / max(1, total_tokens),
        "target_tokens": total_tokens,
    }


@torch.inference_mode()
def greedy_decode(
    model,
    prompt_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    max_new_tokens: int,
    readout_generator: ReadoutCrossAttentionMemory
    | GatedLayerwiseCrossAttentionMemory
    | None = None,
    readout_memory: torch.Tensor | None = None,
) -> str:
    if (readout_generator is None) != (readout_memory is None):
        raise ValueError("Readout generator and memory must be provided together")
    layerwise = isinstance(readout_generator, GatedLayerwiseCrossAttentionMemory)
    activation = (
        readout_generator.activate(readout_memory)
        if layerwise and readout_generator is not None and readout_memory is not None
        else nullcontext()
    )
    with activation:
        output = model(
            inputs_embeds=prompt_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=isinstance(
                readout_generator, ReadoutCrossAttentionMemory
            ),
            use_cache=True,
        )
        if not isinstance(readout_generator, ReadoutCrossAttentionMemory):
            next_token = output.logits[:, -1].argmax(dim=-1)
        else:
            adapted = readout_generator.adapt(
                output.hidden_states[-1][:, -1:], readout_memory
            )
            next_token = model.get_output_embeddings()(
                adapted.to(model.dtype)
            )[:, -1].argmax(dim=-1)
        past = output.past_key_values
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            output = model(
                input_ids=next_token[:, None],
                past_key_values=past,
                output_hidden_states=isinstance(
                    readout_generator, ReadoutCrossAttentionMemory
                ),
                use_cache=True,
            )
            past = output.past_key_values
            if not isinstance(readout_generator, ReadoutCrossAttentionMemory):
                next_token = output.logits[:, -1].argmax(dim=-1)
            else:
                adapted = readout_generator.adapt(
                    output.hidden_states[-1][:, -1:], readout_memory
                )
                next_token = model.get_output_embeddings()(
                    adapted.to(model.dtype)
                )[:, -1].argmax(dim=-1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def select_registered_index(
    query_state: torch.Tensor,
    output_rows: torch.Tensor,
) -> int:
    if query_state.ndim != 1 or output_rows.ndim != 2:
        raise ValueError("Expected one query vector and a matrix of registered rows")
    if query_state.shape[0] != output_rows.shape[1] or not len(output_rows):
        raise ValueError("Query and registered output rows are incompatible")
    return int((query_state.float() @ output_rows.float().T).argmax().item())


@torch.inference_mode()
def greedy_registered_trajectory(
    model,
    prompt_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    registered_output_rows: torch.Tensor,
    registered_memory: torch.Tensor,
    registered_tool_names: list[str],
    slot_by_tool: dict[str, int],
    tokenizer,
    max_new_tokens: int,
    input_memory_interface: str = "prompt_slots",
    readout_generator: ReadoutCrossAttentionMemory
    | GatedLayerwiseCrossAttentionMemory
    | None = None,
) -> tuple[str, int, str]:
    """Select a dynamic token, expand it to memory, then continue one KV stream."""
    if len(prompt_embeddings) != 1 or len(attention_mask) != 1:
        raise ValueError("Trajectory decoding currently requires batch size one")
    if len(registered_tool_names) != len(registered_output_rows):
        raise ValueError("Registered names and output rows must have equal length")
    if registered_memory.shape[0] != len(registered_tool_names):
        raise ValueError("Registered names and input memory must have equal length")

    registry = DynamicRegistry.from_vectors(
        registered_tool_names,
        registered_output_rows,
        [slot_by_tool[name] for name in registered_tool_names],
        registered_memory,
    )

    selection_scope = (
        readout_generator.suspended()
        if isinstance(readout_generator, GatedLayerwiseCrossAttentionMemory)
        else nullcontext()
    )
    with selection_scope:
        output = model(
            inputs_embeds=prompt_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )
    selected_entry = registry.select(output.hidden_states[-1][0, -1].float())
    selected_tool = selected_entry.tool_name
    selected_slot = selected_entry.slot_id

    memory = registry.expand(selected_slot).unsqueeze(0).to(prompt_embeddings.dtype)
    layerwise = is_layerwise_interface(input_memory_interface)
    if layerwise and not isinstance(
        readout_generator, GatedLayerwiseCrossAttentionMemory
    ):
        raise ValueError("Layerwise trajectory requires a layerwise generator")
    generation_scope = (
        readout_generator.activate(memory)
        if layerwise and isinstance(
            readout_generator, GatedLayerwiseCrossAttentionMemory
        )
        else nullcontext()
    )
    with generation_scope:
        if input_memory_interface == "readout_cross_attention":
            if not isinstance(readout_generator, ReadoutCrossAttentionMemory):
                raise ValueError("Readout trajectory requires a readout generator")
            adapted = readout_generator.adapt(
                output.hidden_states[-1][:, -1:], memory
            )
            next_token = model.get_output_embeddings()(
                adapted.to(model.dtype)
            )[:, -1].argmax(dim=-1)
            running_attention = attention_mask
            past = output.past_key_values
        elif input_memory_interface == "gated_layerwise_cross_attention":
            running_attention = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            selected_token = torch.tensor(
                [[selected_slot]], device=prompt_embeddings.device, dtype=torch.long
            )
            output = model(
                input_ids=selected_token,
                attention_mask=running_attention,
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            next_token = output.logits[:, -1].argmax(dim=-1)
            past = output.past_key_values
        else:
            running_attention = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, memory.shape[1]),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
            output = model(
                inputs_embeds=memory,
                attention_mask=running_attention,
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            next_token = output.logits[:, -1].argmax(dim=-1)
            past = output.past_key_values
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            running_attention = torch.cat(
                [
                    running_attention,
                    torch.ones(
                        (1, 1),
                        device=running_attention.device,
                        dtype=running_attention.dtype,
                    ),
                ],
                dim=1,
            )
            output = model(
                input_ids=next_token[:, None],
                attention_mask=running_attention,
                past_key_values=past,
                output_hidden_states=input_memory_interface
                == "readout_cross_attention",
                use_cache=True,
            )
            past = output.past_key_values
            if input_memory_interface == "readout_cross_attention":
                adapted = readout_generator.adapt(
                    output.hidden_states[-1][:, -1:], memory
                )
                next_token = model.get_output_embeddings()(
                    adapted.to(model.dtype)
                )[:, -1].argmax(dim=-1)
            else:
                next_token = output.logits[:, -1].argmax(dim=-1)
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return selected_tool, selected_slot, text


def parse_json_object(text: str) -> Any | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def key_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        paths: set[str] = set()
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= key_paths(child, path)
        return paths
    if isinstance(value, list):
        paths: set[str] = set()
        for child in value:
            paths |= key_paths(child, f"{prefix}[]")
        return paths
    return set()


def leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(leaf_values(child, path))
        return flattened
    return {prefix: value} if prefix else {}


def generation_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    valid = exact = schema = 0
    predicted_keys = target_keys = correct_keys = 0
    shared_values = correct_values = 0
    for row in rows:
        parsed = row["parsed"]
        target = row["target"]
        valid += parsed is not None
        exact += parsed == target
        schema += parsed is not None and key_paths(parsed) == key_paths(target)
        target_paths = key_paths(target)
        target_keys += len(target_paths)
        if parsed is not None:
            parsed_paths = key_paths(parsed)
            predicted_keys += len(parsed_paths)
            correct_keys += len(parsed_paths & target_paths)
            parsed_leaves = leaf_values(parsed)
            target_leaves = leaf_values(target)
            shared = parsed_leaves.keys() & target_leaves.keys()
            shared_values += len(shared)
            correct_values += sum(parsed_leaves[path] == target_leaves[path] for path in shared)
    count = len(rows)
    return {
        "examples": count,
        "json_valid": valid / max(1, count),
        "exact_arguments": exact / max(1, count),
        "exact_schema_keys": schema / max(1, count),
        "key_precision": correct_keys / max(1, predicted_keys),
        "key_recall": correct_keys / max(1, target_keys),
        "shared_key_value_accuracy": correct_values / max(1, shared_values),
    }


def pipeline_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Score selection and readback as one constrained tool-call pipeline."""
    count = len(rows)
    selected = valid = exact = schema = 0
    for row in rows:
        selected_key = row.get("selected_tool_id", row.get("selected_tool"))
        target_key = row.get("tool_id", row.get("tool_name"))
        selection_correct = selected_key == target_key
        selected += selection_correct
        parsed = row["parsed"]
        valid += parsed is not None
        exact += selection_correct and parsed == row["target"]
        schema += (
            selection_correct
            and parsed is not None
            and key_paths(parsed) == key_paths(row["target"])
        )
    return {
        "examples": count,
        "selection_accuracy": selected / max(1, count),
        "json_valid": valid / max(1, count),
        "readback_exact_arguments_given_correct_selection": exact / max(1, selected),
        "readback_exact_schema_keys_given_correct_selection": schema / max(1, selected),
        "end_to_end_exact_arguments": exact / max(1, count),
        "end_to_end_exact_schema_keys": schema / max(1, count),
    }


def input_training_phase(epoch: int, schema_warmup_epochs: int) -> str:
    return "schema_warmup" if epoch < schema_warmup_epochs else "joint"


def distributed_input_batches(
    order: list[int],
    local_batch_size: int,
    world_size: int,
    rank: int,
) -> tuple[list[list[int]], int]:
    if local_batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed batch configuration")
    global_batch_size = local_batch_size * world_size
    padding = (-len(order)) % global_batch_size
    padded = order + order[:padding]
    batches = []
    for start in range(0, len(padded), global_batch_size):
        global_batch = padded[start : start + global_batch_size]
        local_start = rank * local_batch_size
        batches.append(global_batch[local_start : local_start + local_batch_size])
    return batches, padding


def unwrap_module(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def distributed_loss_means(
    value_lists: list[list[float]], device: torch.device
) -> list[float]:
    totals = torch.tensor(
        [sum(values) for values in value_lists] + [len(value_lists[0])],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(1.0, float(totals[-1].item()))
    return [float(value.item()) / count for value in totals[:-1]]


def main() -> None:
    args = parse_args()
    if (
        args.schema_loss_weight < 0
        or args.distill_loss_weight < 0
        or args.trajectory_loss_weight < 0
    ):
        raise ValueError("Auxiliary loss weights must be non-negative")
    if args.schema_warmup_epochs < 0:
        raise ValueError("Schema warmup epochs must be non-negative")
    if args.schema_warmup_epochs and not args.schema_loss_weight:
        raise ValueError("Schema warmup requires a positive schema loss weight")
    if args.distill_temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    if (
        is_persistent_memory_interface(args.input_memory_interface)
        and args.input_memory_source != "pooled"
    ):
        raise ValueError("Persistent cross-attention currently requires pooled memory")
    if args.tool_pooling in {"mean_last_slots", "mean_schema_key_slots"}:
        if args.input_memory_source != "pooled":
            raise ValueError("Structured tool views require pooled memory")
        if not is_layerwise_interface(args.input_memory_interface):
            raise ValueError("Structured tool views currently require a layerwise interface")
    if args.layerwise_memory_max_gate <= 0:
        raise ValueError("Layerwise memory max gate must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    is_main = rank == 0
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
    if is_main:
        print(
            f"runtime world_size={world_size} "
            f"global_input_batch_size={args.input_batch_size * world_size}",
            flush=True,
        )
    examples = load_examples(args.dataset, args.max_examples)
    train_examples, eval_examples = stable_tool_split(examples, args.eval_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if device.type == "cuda":
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=model_dtype,
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
        include_tool_sequences=args.input_memory_source == "token_resampler",
        tool_identity=args.tool_identity,
        tool_pooling=args.tool_pooling,
        input_memory_slots=args.input_memory_slots,
    )
    example_index = {item.example_id: index for index, item in enumerate(examples)}
    tool_index = {key: index for index, key in enumerate(features["tool_keys"])}
    tool_feature_map = {
        key: features["input_tool"][index] for key, index in tool_index.items()
    }
    schema_key_count_map = {
        key: features["schema_key_counts"][index] for key, index in tool_index.items()
    }
    tool_sequence_map = None
    tool_value_map = None
    if features["tool_sequences"] is not None:
        tool_sequence_map = {
            key: features["tool_sequences"][index]
            for key, index in tool_index.items()
        }
        tool_value_map = {
            key: features["tool_values"][index]
            for key, index in tool_index.items()
        }
    train_query = features["query"][[example_index[item.example_id] for item in train_examples]]
    train_query_keys = [tool_registry_key(item, args.tool_identity) for item in train_examples]
    train_tool_keys = sorted(set(train_query_keys))
    train_tool = features["tool"][[tool_index[key] for key in train_tool_keys]]
    eval_query = features["query"][[example_index[item.example_id] for item in eval_examples]]
    eval_query_keys = [tool_registry_key(item, args.tool_identity) for item in eval_examples]
    eval_tool_keys = sorted(set(eval_query_keys))
    eval_tool = features["tool"][[tool_index[key] for key in eval_tool_keys]]
    display_name_by_key = {
        tool_registry_key(item, args.tool_identity): item.tool_name for item in examples
    }
    definition_count_by_name: dict[str, int] = {}
    for item in examples:
        definition_count_by_name.setdefault(item.tool_name, 0)
    for name in definition_count_by_name:
        definition_count_by_name[name] = len(
            {item.tool_document for item in examples if item.tool_name == name}
        )

    original_output = model.get_output_embeddings().weight[: token_pool.original_vocab_size]
    output_row_norm = float(original_output.detach().float().norm(dim=-1).mean())
    output_generator = PhysicalOutputGenerator(
        model.config.hidden_size, args.rank, output_row_norm
    ).to(device)
    selection_optimizer = torch.optim.AdamW(
        output_generator.parameters(), lr=args.selection_learning_rate, weight_decay=0.01
    )
    train_tool_lookup = {key: index for index, key in enumerate(train_tool_keys)}
    training_seen_ids: set[int] = set()
    if is_main:
        for epoch in range(args.selection_epochs):
            order = torch.randperm(len(train_query))
            losses = []
            for start in range(0, len(order), args.selection_batch_size):
                indices = order[start : start + args.selection_batch_size]
                keys = [train_query_keys[index] for index in indices.tolist()]
                candidate_keys = sorted(set(keys))
                random.shuffle(candidate_keys)
                candidate_slots = random.sample(
                    token_pool.train_token_ids, len(candidate_keys)
                )
                training_seen_ids.update(candidate_slots)
                candidate_indices = [train_tool_lookup[key] for key in candidate_keys]
                logits = output_generator(
                    train_query[indices].to(device), train_tool[candidate_indices].to(device)
                )
                physical_logits = torch.full(
                    (len(indices), len(tokenizer)),
                    float("-inf"),
                    device=device,
                    dtype=logits.dtype,
                )
                physical_logits[:, candidate_slots] = logits
                target_slots = torch.tensor(
                    [candidate_slots[candidate_keys.index(key)] for key in keys],
                    device=device,
                    dtype=torch.long,
                )
                loss = F.cross_entropy(physical_logits, target_slots)
                selection_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                selection_optimizer.step()
                losses.append(float(loss.detach()))
            print(
                f"selection_epoch={epoch + 1} loss={sum(losses) / len(losses):.6f}",
                flush=True,
            )
    if dist.is_initialized():
        for parameter in output_generator.parameters():
            dist.broadcast(parameter.data, src=0)
        dist.barrier()

    input_weight = model.get_input_embeddings().weight[: token_pool.original_vocab_size]
    input_embedding_norm = float(input_weight.detach().float().norm(dim=-1).mean())
    if is_layerwise_interface(args.input_memory_interface):
        decoder_layers = model.model.layers
        layer_indices = resolve_layerwise_memory_layers(
            args.layerwise_memory_layers, len(decoder_layers)
        )
        num_views = {
            "mean_last_slots": 2,
            "mean_schema_key_slots": args.input_memory_slots + 1,
        }.get(args.tool_pooling, 1)
        input_generator = GatedLayerwiseCrossAttentionMemory(
            model.config.hidden_size,
            args.rank,
            args.input_memory_slots,
            input_embedding_norm,
            layer_indices,
            args.layerwise_memory_max_gate,
            num_views=num_views,
        ).to(device)
        input_generator.install(decoder_layers)
        if is_main:
            print(f"layerwise_memory layers={list(layer_indices)}", flush=True)
    elif args.input_memory_interface == "readout_cross_attention":
        input_generator = ReadoutCrossAttentionMemory(
            model.config.hidden_size,
            args.rank,
            args.input_memory_slots,
            input_embedding_norm,
        ).to(device)
    elif args.input_memory_source == "token_resampler":
        input_generator = TokenResamplerMemory(
            model.config.hidden_size,
            args.rank,
            args.input_memory_slots,
            input_embedding_norm,
        ).to(device)
    else:
        input_generator = GeneratedMemory(
            model.config.hidden_size,
            args.rank,
            args.input_memory_slots,
            input_embedding_norm,
        ).to(device)
    base_input_generator = unwrap_module(input_generator)
    readout_initial_output = None
    layerwise_initial_gates = None
    if isinstance(base_input_generator, ReadoutCrossAttentionMemory):
        readout_initial_output = base_input_generator.output.weight.detach().clone()
    elif isinstance(base_input_generator, GatedLayerwiseCrossAttentionMemory):
        layerwise_initial_gates = base_input_generator.gate_values()
    if dist.is_initialized():
        input_generator = DistributedDataParallel(
            input_generator,
            device_ids=[local_rank],
            broadcast_buffers=False,
            static_graph=isinstance(
                base_input_generator, GatedLayerwiseCrossAttentionMemory
            ),
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
    input_optimizer = torch.optim.AdamW(
        input_generator.parameters(), lr=args.input_learning_rate, weight_decay=0.01
    )
    total_input_epochs = args.schema_warmup_epochs + args.input_epochs
    distributed_padding = 0
    readout_nonzero_gradient_steps = 0
    readout_max_gradient_norm = 0.0
    for epoch in range(total_input_epochs):
        phase = input_training_phase(epoch, args.schema_warmup_epochs)
        order = list(range(len(train_examples)))
        if world_size == 1:
            random.shuffle(order)
            local_batches = [
                order[start : start + args.input_batch_size]
                for start in range(0, len(order), args.input_batch_size)
            ]
            slot_random = random
        else:
            random.Random(args.seed + 10_000 + epoch).shuffle(order)
            local_batches, distributed_padding = distributed_input_batches(
                order,
                args.input_batch_size,
                world_size,
                rank,
            )
            if is_main and epoch == 0:
                print(
                    f"distributed_input steps_per_rank={len(local_batches)} "
                    f"padding_examples_per_epoch={distributed_padding}",
                    flush=True,
                )
            slot_random = random.Random(
                args.seed + 1_000_000 * (epoch + 1) + rank
            )
        losses = []
        execution_losses = []
        schema_losses = []
        distill_losses = []
        trajectory_losses = []
        for indices in local_batches:
            batch = [train_examples[index] for index in indices]
            slot_ids = slot_random.sample(token_pool.train_token_ids, len(batch))
            training_seen_ids.update(slot_ids)
            batch_tool_keys = [
                tool_registry_key(item, args.tool_identity) for item in batch
            ]
            batch_tool_latents = stack_tool_latents(
                batch_tool_keys, tool_feature_map, device
            )
            base_input_generator = unwrap_module(input_generator)
            active_layerwise_memory = None
            if isinstance(
                base_input_generator, GatedLayerwiseCrossAttentionMemory
            ):
                active_layerwise_memory = input_generator(batch_tool_latents)
                if uses_layerwise_for_ordinary_readback(
                    args.input_memory_interface
                ):
                    base_input_generator.bind(active_layerwise_memory)
            overrides = (
                active_layerwise_memory
                if uses_prompt_slots(args.input_memory_interface)
                and active_layerwise_memory is not None
                else make_input_overrides(
                    "registered",
                    batch_tool_keys,
                    tool_feature_map,
                    tool_sequence_map,
                    tool_value_map,
                    input_generator,
                    args.input_memory_source,
                    args.input_memory_interface,
                    input_embedding_norm,
                    device,
                )
            )
            output = None
            labels = None
            execution_loss = torch.zeros((), device=device)
            loss = torch.zeros((), device=device)
            if phase == "joint":
                embeds, attention, labels = prepare_batch(
                    model,
                    tokenizer,
                    batch,
                    "registered",
                    slot_ids,
                    overrides,
                    args.max_prompt_length,
                    args.max_target_length,
                    device,
                )
                output = registered_forward(
                    model,
                    input_generator,
                    args.input_memory_interface,
                    batch_tool_latents,
                    embeds,
                    attention,
                    labels,
                )
                execution_loss = output.loss
                loss = execution_loss

            trajectory_loss = torch.zeros((), device=device)
            if args.trajectory_loss_weight and phase == "joint":
                trajectory_memory = overrides
                if args.input_memory_interface in {
                    "readout_cross_attention",
                    "gated_layerwise_cross_attention",
                }:
                    trajectory_memory = torch.empty(
                        (len(batch), 0, model.config.hidden_size),
                        device=device,
                        dtype=model.dtype,
                    )
                trajectory_embeds, trajectory_attention, trajectory_labels = (
                    prepare_trajectory_batch(
                        model,
                        tokenizer,
                        batch,
                        trajectory_memory,
                        args.max_prompt_length,
                        args.max_target_length,
                        device,
                    )
                )
                trajectory_scope = (
                    base_input_generator.activate(active_layerwise_memory)
                    if args.input_memory_interface
                    == "prompt_slots_plus_trajectory_layerwise"
                    and isinstance(
                        base_input_generator,
                        GatedLayerwiseCrossAttentionMemory,
                    )
                    and active_layerwise_memory is not None
                    else nullcontext()
                )
                uncheckpointed_trajectory = (
                    args.gradient_checkpointing
                    and args.input_memory_interface
                    == "prompt_slots_plus_trajectory_layerwise"
                )
                if uncheckpointed_trajectory:
                    model.gradient_checkpointing_disable()
                try:
                    with trajectory_scope:
                        trajectory_output = registered_forward(
                            model,
                            input_generator,
                            args.input_memory_interface,
                            batch_tool_latents,
                            trajectory_embeds,
                            trajectory_attention,
                            trajectory_labels,
                        )
                finally:
                    if uncheckpointed_trajectory:
                        model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={
                                "use_reentrant": False
                            }
                        )
                trajectory_loss = trajectory_output.loss
                loss = loss + args.trajectory_loss_weight * trajectory_loss
            schema_loss = torch.zeros((), device=device)
            if args.schema_loss_weight:
                schema_embeds, schema_attention, schema_labels = prepare_batch(
                    model,
                    tokenizer,
                    batch,
                    "registered",
                    slot_ids,
                    overrides,
                    args.max_prompt_length,
                    args.max_target_length,
                    device,
                    target_kind="schema",
                )
                schema_output = registered_forward(
                    model,
                    input_generator,
                    args.input_memory_interface,
                    batch_tool_latents,
                    schema_embeds,
                    schema_attention,
                    schema_labels,
                )
                schema_loss = schema_output.loss
                loss = loss + args.schema_loss_weight * schema_loss

            distill_loss = torch.zeros((), device=device)
            if args.distill_loss_weight and phase == "joint":
                if output is None or labels is None:
                    raise RuntimeError("Joint output is required for distillation")
                teacher_embeds, teacher_attention, teacher_labels = prepare_batch(
                    model,
                    tokenizer,
                    batch,
                    "full_document",
                    [None] * len(batch),
                    None,
                    args.max_prompt_length,
                    args.max_target_length,
                    device,
                )
                student_training = model.training
                model.eval()
                teacher_scope = (
                    base_input_generator.suspended()
                    if isinstance(
                        base_input_generator,
                        GatedLayerwiseCrossAttentionMemory,
                    )
                    else nullcontext()
                )
                with teacher_scope, torch.no_grad():
                    teacher_logits = model(
                        inputs_embeds=teacher_embeds,
                        attention_mask=teacher_attention,
                        use_cache=False,
                    ).logits
                model.train(student_training)
                student_mask = labels[:, 1:] != -100
                teacher_mask = teacher_labels[:, 1:] != -100
                student_target_logits = output.logits[:, :-1][student_mask].float()
                teacher_target_logits = teacher_logits[:, :-1][teacher_mask].float()
                if student_target_logits.shape != teacher_target_logits.shape:
                    raise ValueError("Teacher and student target logits are not aligned")
                temperature = args.distill_temperature
                distill_loss = F.kl_div(
                    F.log_softmax(student_target_logits / temperature, dim=-1),
                    F.softmax(teacher_target_logits / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature**2
                loss = loss + args.distill_loss_weight * distill_loss
            if (
                args.input_memory_interface
                == "prompt_slots_plus_trajectory_layerwise"
                and phase != "joint"
                and isinstance(
                    base_input_generator,
                    GatedLayerwiseCrossAttentionMemory,
                )
            ):
                # Keep DDP's parameter-use set static while the warmup behavior
                # remains exactly prompt-only.
                loss = loss + sum(
                    parameter.float().sum() * 0.0
                    for parameter in base_input_generator.blocks.parameters()
                )
            input_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            base_input_generator = unwrap_module(input_generator)
            audit_adapter_gradients = not (
                args.input_memory_interface
                == "prompt_slots_plus_trajectory_layerwise"
                and phase != "joint"
            )
            if audit_adapter_gradients and isinstance(
                base_input_generator,
                (ReadoutCrossAttentionMemory, GatedLayerwiseCrossAttentionMemory),
            ):
                squared_norm = torch.zeros((), device=device)
                finite = True
                for name, parameter in base_input_generator.named_parameters():
                    adapter_parameter = (
                        name.startswith(("query.", "key.", "value.", "output."))
                        if isinstance(
                            base_input_generator, ReadoutCrossAttentionMemory
                        )
                        else name.startswith("blocks.")
                    )
                    if adapter_parameter:
                        if parameter.grad is None:
                            finite = False
                            continue
                        finite = finite and bool(torch.isfinite(parameter.grad).all())
                        squared_norm = squared_norm + parameter.grad.float().square().sum()
                gradient_norm = float(squared_norm.sqrt().detach())
                if not finite or not math.isfinite(gradient_norm):
                    raise RuntimeError("Readout adapter produced non-finite gradients")
                if gradient_norm > 0.0:
                    readout_nonzero_gradient_steps += 1
                readout_max_gradient_norm = max(
                    readout_max_gradient_norm, gradient_norm
                )
            if isinstance(
                base_input_generator, GatedLayerwiseCrossAttentionMemory
            ):
                base_input_generator.clear()
            torch.nn.utils.clip_grad_norm_(input_generator.parameters(), 1.0)
            input_optimizer.step()
            losses.append(float(loss.detach()))
            execution_losses.append(float(execution_loss.detach()))
            schema_losses.append(float(schema_loss.detach()))
            distill_losses.append(float(distill_loss.detach()))
            trajectory_losses.append(float(trajectory_loss.detach()))
        means = distributed_loss_means(
            [
                losses,
                execution_losses,
                schema_losses,
                distill_losses,
                trajectory_losses,
            ],
            device,
        )
        if is_main:
            print(
                f"input_epoch={epoch + 1}/{total_input_epochs} phase={phase} "
                f"loss={means[0]:.6f} execution={means[1]:.6f} "
                f"schema={means[2]:.6f} distill={means[3]:.6f} "
                f"trajectory={means[4]:.6f}",
                flush=True,
            )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
        model.eval()

    input_generator = unwrap_module(input_generator)
    interface_audit = {
        "readout_adapter_nonzero_gradient_steps": readout_nonzero_gradient_steps,
        "readout_adapter_max_gradient_norm": readout_max_gradient_norm,
        "readout_output_max_change": None,
        "layerwise_initial_gates": layerwise_initial_gates,
        "layerwise_final_gates": None,
        "layerwise_max_abs_gate": None,
    }
    if isinstance(input_generator, ReadoutCrossAttentionMemory):
        if readout_initial_output is None:
            raise RuntimeError("Readout adapter initialization audit is missing")
        interface_audit["readout_output_max_change"] = float(
            (input_generator.output.weight.detach() - readout_initial_output).abs().max()
        )
    elif isinstance(input_generator, GatedLayerwiseCrossAttentionMemory):
        final_gates = input_generator.gate_values()
        interface_audit["layerwise_final_gates"] = final_gates
        interface_audit["layerwise_max_abs_gate"] = max(
            abs(value) for value in final_gates.values()
        )
    if dist.is_initialized():
        seen_by_rank: list[set[int] | None] = [None] * world_size
        dist.all_gather_object(seen_by_rank, training_seen_ids)
        training_seen_ids = set().union(
            *(seen or set() for seen in seen_by_rank)
        )
        dist.barrier()
        dist.destroy_process_group()
        if not is_main:
            return

    with torch.inference_mode():
        eval_scores = output_generator(eval_query.to(device), eval_tool.to(device)).cpu()
        raw_scores = eval_query.float() @ (
            F.normalize(eval_tool.float(), dim=-1) * output_row_norm
        ).T
        generated_rows = output_generator.output_rows(eval_tool.to(device))
        first_query = eval_query[:1].to(device).float()
        direct = first_query @ generated_rows.T
        full_logits = model.get_output_embeddings()(first_query.to(model.dtype)).float()
        physical_slots = token_pool.heldout_token_ids[: len(eval_tool_keys)]
        full_logits[:, physical_slots] = direct
        physical_delta = float((full_logits[:, physical_slots] - direct).abs().max())

    if len(eval_tool_keys) > len(token_pool.heldout_token_ids):
        raise ValueError("Not enough strictly held-out slots for evaluation tools")
    slot_by_tool = {
        key: token_pool.heldout_token_ids[index]
        for index, key in enumerate(eval_tool_keys)
    }
    teacher_forced: dict[str, Any] = {}
    for condition in (
        "query_only",
        "full_document",
        "static_blank",
        "raw_latent",
        "registered",
    ):
        teacher_forced[condition] = teacher_forced_metrics(
            model,
            tokenizer,
            eval_examples,
            condition,
            slot_by_tool,
            tool_feature_map,
            tool_sequence_map,
            tool_value_map,
            input_generator,
            args.input_memory_source,
            args.input_memory_interface,
            input_embedding_norm,
            args.input_batch_size,
            args.max_prompt_length,
            args.max_target_length,
            device,
            tool_identity=args.tool_identity,
        )
        print(f"teacher_forced[{condition}]={teacher_forced[condition]}", flush=True)

    schema_teacher_forced: dict[str, Any] = {}
    for condition in (
        "query_only",
        "full_document",
        "static_blank",
        "raw_latent",
        "registered",
    ):
        schema_teacher_forced[condition] = teacher_forced_metrics(
            model,
            tokenizer,
            eval_examples,
            condition,
            slot_by_tool,
            tool_feature_map,
            tool_sequence_map,
            tool_value_map,
            input_generator,
            args.input_memory_source,
            args.input_memory_interface,
            input_embedding_norm,
            args.input_batch_size,
            args.max_prompt_length,
            args.max_target_length,
            device,
            target_kind="schema",
            tool_identity=args.tool_identity,
        )
        print(
            f"schema_teacher_forced[{condition}]={schema_teacher_forced[condition]}",
            flush=True,
        )

    trajectory_teacher_forced = trajectory_teacher_forced_metrics(
        model,
        tokenizer,
        eval_examples,
        tool_feature_map,
        tool_sequence_map,
        tool_value_map,
        input_generator,
        args.input_memory_source,
        args.input_memory_interface,
        input_embedding_norm,
        args.input_batch_size,
        args.max_prompt_length,
        args.max_target_length,
        device,
        tool_identity=args.tool_identity,
    )
    print(
        f"trajectory_teacher_forced[registered]={trajectory_teacher_forced}",
        flush=True,
    )

    generation_examples = eval_examples[: args.generation_limit or None]
    prediction_rows: list[dict[str, Any]] = []
    generation_summary: dict[str, Any] = {}
    for condition in (
        "query_only",
        "full_document",
        "static_blank",
        "raw_latent",
        "registered",
    ):
        condition_rows = []
        for example in generation_examples:
            tool_key = tool_registry_key(example, args.tool_identity)
            slot_id = slot_by_tool[tool_key]
            tool_latent = stack_tool_latents([tool_key], tool_feature_map, device)
            override = make_input_overrides(
                condition,
                [tool_key],
                tool_feature_map,
                tool_sequence_map,
                tool_value_map,
                input_generator,
                args.input_memory_source,
                args.input_memory_interface,
                input_embedding_norm,
                device,
            )
            effective_slot = None if condition in {"query_only", "full_document"} else slot_id
            embeds, attention, _ = prepare_batch(
                model,
                tokenizer,
                [example],
                condition,
                [effective_slot],
                override,
                args.max_prompt_length,
                args.max_target_length,
                device,
                include_targets=False,
            )
            readout_generator = None
            readout_memory = None
            if (
                uses_persistent_memory_for_ordinary_readback(
                    args.input_memory_interface
                )
                and condition in {"registered", "raw_latent"}
            ):
                if not isinstance(
                    input_generator,
                    (
                        ReadoutCrossAttentionMemory,
                        GatedLayerwiseCrossAttentionMemory,
                    ),
                ):
                    raise RuntimeError("Persistent generator is unavailable at evaluation")
                readout_generator = input_generator
                if condition == "registered":
                    readout_memory = input_generator.registered_memory(tool_latent)
                else:
                    readout_memory = raw_registered_memory(
                        tool_latent,
                        input_generator.num_slots,
                        input_embedding_norm,
                    )
            text = greedy_decode(
                model,
                embeds,
                attention,
                tokenizer,
                args.max_new_tokens,
                readout_generator,
                readout_memory,
            )
            row = {
                "condition": condition,
                "example_id": example.example_id,
                "tool_name": example.tool_name,
                "tool_id": tool_key,
                "same_name_definition_count": definition_count_by_name[example.tool_name],
                "slot_id": effective_slot,
                "target": example.target_arguments,
                "prediction": text,
                "parsed": parse_json_object(text),
            }
            prediction_rows.append(row)
            condition_rows.append(row)
        generation_summary[condition] = generation_metrics(condition_rows)
        print(f"generation[{condition}]={generation_summary[condition]}", flush=True)

    selected_eval_indices = eval_scores.argmax(dim=1).tolist()
    selected_tool_by_example = {
        example.example_id: eval_tool_keys[selected_eval_indices[index]]
        for index, example in enumerate(eval_examples)
    }
    pipeline_rows: list[dict[str, Any]] = []
    for example in generation_examples:
        selected_tool_id = selected_tool_by_example[example.example_id]
        selected_slot = slot_by_tool[selected_tool_id]
        selected_tool_latent = stack_tool_latents(
            [selected_tool_id], tool_feature_map, device
        )
        override = make_input_overrides(
            "registered",
            [selected_tool_id],
            tool_feature_map,
            tool_sequence_map,
            tool_value_map,
            input_generator,
            args.input_memory_source,
            args.input_memory_interface,
            input_embedding_norm,
            device,
        )
        embeds, attention, _ = prepare_batch(
            model,
            tokenizer,
            [example],
            "registered",
            [selected_slot],
            override,
            args.max_prompt_length,
            args.max_target_length,
            device,
            include_targets=False,
        )
        readout_generator = None
        readout_memory = None
        if uses_persistent_memory_for_ordinary_readback(
            args.input_memory_interface
        ):
            if not isinstance(
                input_generator,
                (ReadoutCrossAttentionMemory, GatedLayerwiseCrossAttentionMemory),
            ):
                raise RuntimeError("Persistent generator is unavailable at evaluation")
            readout_generator = input_generator
            readout_memory = input_generator.registered_memory(selected_tool_latent)
        text = greedy_decode(
            model,
            embeds,
            attention,
            tokenizer,
            args.max_new_tokens,
            readout_generator,
            readout_memory,
        )
        row = {
            "condition": "constrained_registered_pipeline",
            "example_id": example.example_id,
            "tool_name": example.tool_name,
            "tool_id": tool_registry_key(example, args.tool_identity),
            "same_name_definition_count": definition_count_by_name[example.tool_name],
            "selected_tool": display_name_by_key[selected_tool_id],
            "selected_tool_id": selected_tool_id,
            "slot_id": selected_slot,
            "target": example.target_arguments,
            "prediction": text,
            "parsed": parse_json_object(text),
        }
        prediction_rows.append(row)
        pipeline_rows.append(row)
    registered_pipeline = pipeline_metrics(pipeline_rows)
    print(f"pipeline[constrained_registered]={registered_pipeline}", flush=True)

    with torch.inference_mode():
        if is_persistent_memory_interface(args.input_memory_interface):
            if not isinstance(
                input_generator,
                (ReadoutCrossAttentionMemory, GatedLayerwiseCrossAttentionMemory),
            ):
                raise RuntimeError("Persistent generator is unavailable at evaluation")
            registered_eval_memory = input_generator.registered_memory(
                stack_tool_latents(eval_tool_keys, tool_feature_map, device)
            )
        else:
            registered_eval_memory = make_input_overrides(
                "registered",
                eval_tool_keys,
                tool_feature_map,
                tool_sequence_map,
                tool_value_map,
                input_generator,
                args.input_memory_source,
                args.input_memory_interface,
                input_embedding_norm,
                device,
            )
    if registered_eval_memory is None:
        raise RuntimeError("Registered trajectory requires generated input memory")
    empty_memory = torch.empty(
        (1, 0, model.config.hidden_size), device=device, dtype=model.dtype
    )
    trajectory_rows: list[dict[str, Any]] = []
    for example in generation_examples:
        prompt_embeds, prompt_attention, _ = prepare_trajectory_batch(
            model,
            tokenizer,
            [example],
            empty_memory,
            args.max_prompt_length,
            args.max_target_length,
            device,
            include_targets=False,
        )
        selected_tool_id, selected_slot, text = greedy_registered_trajectory(
            model,
            prompt_embeds,
            prompt_attention,
            generated_rows,
            registered_eval_memory,
            eval_tool_keys,
            slot_by_tool,
            tokenizer,
            args.max_new_tokens,
            args.input_memory_interface,
            input_generator
            if isinstance(
                input_generator,
                (ReadoutCrossAttentionMemory, GatedLayerwiseCrossAttentionMemory),
            )
            else None,
        )
        row = {
            "condition": "autoregressive_registered_trajectory",
            "example_id": example.example_id,
            "tool_name": example.tool_name,
            "tool_id": tool_registry_key(example, args.tool_identity),
            "same_name_definition_count": definition_count_by_name[example.tool_name],
            "selected_tool": display_name_by_key[selected_tool_id],
            "selected_tool_id": selected_tool_id,
            "slot_id": selected_slot,
            "target": example.target_arguments,
            "prediction": text,
            "parsed": parse_json_object(text),
        }
        prediction_rows.append(row)
        trajectory_rows.append(row)
    autoregressive_trajectory = pipeline_metrics(trajectory_rows)
    print(
        f"pipeline[autoregressive_registered]={autoregressive_trajectory}",
        flush=True,
    )

    results = {
        "split": {
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "train_tools": len(train_tool_keys),
            "eval_tools": len(eval_tool_keys),
            "tool_overlap": len(set(train_tool_keys) & set(eval_tool_keys)),
            "train_tool_names": len({item.tool_name for item in train_examples}),
            "eval_tool_names": len({item.tool_name for item in eval_examples}),
            "tool_name_overlap": len(
                {item.tool_name for item in train_examples}
                & {item.tool_name for item in eval_examples}
            ),
            "train_definition_excess_over_names": len(train_tool_keys)
            - len({item.tool_name for item in train_examples}),
            "eval_definition_excess_over_names": len(eval_tool_keys)
            - len({item.tool_name for item in eval_examples}),
        },
        "physical_token_audit": {
            **token_pool.audit(model, training_seen_ids),
            "dynamic_output_full_head_max_delta": physical_delta,
        },
        "input_interface_audit": interface_audit,
        "schema_anchor_audit": {
            "active": args.tool_pooling == "mean_schema_key_slots",
            "memory_slots": args.input_memory_slots,
            "max_train_schema_keys": max(
                schema_key_count_map[key] for key in train_tool_keys
            ),
            "max_eval_schema_keys": max(
                schema_key_count_map[key] for key in eval_tool_keys
            ),
            "truncated_train_tools": sum(
                schema_key_count_map[key] > args.input_memory_slots
                for key in train_tool_keys
            ),
            "truncated_eval_tools": sum(
                schema_key_count_map[key] > args.input_memory_slots
                for key in eval_tool_keys
            ),
        },
        "selection": {
            "raw_physical_rows": metrics_from_scores(
                raw_scores, eval_query_keys, eval_tool_keys
            ),
            "generated_physical_rows": metrics_from_scores(
                eval_scores, eval_query_keys, eval_tool_keys
            ),
        },
        "input_readback_teacher_forced": teacher_forced,
        "schema_readback_teacher_forced": schema_teacher_forced,
        "trajectory_readback_teacher_forced": trajectory_teacher_forced,
        "input_readback_generation": generation_summary,
        "constrained_registered_pipeline": registered_pipeline,
        "autoregressive_registered_trajectory": autoregressive_trajectory,
        "config": {
            **vars(args),
            "world_size": world_size,
            "global_input_batch_size": args.input_batch_size * world_size,
            "distributed_padding_examples_per_epoch": distributed_padding,
        },
    }
    torch.save(
        {
            "output_generator": output_generator.state_dict(),
            "input_generator": input_generator.state_dict(),
        },
        output_dir / "bidirectional_generators.pt",
    )
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
