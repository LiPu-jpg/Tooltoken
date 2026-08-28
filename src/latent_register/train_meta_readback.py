from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .episodic_data import (
    EpisodicRegistrySampler,
    PreparedReadbackExample,
    PreparedRetrievalEpisode,
    PreparedTool,
    load_prepared_tools,
    load_readback_examples,
    load_retrieval_episodes,
    load_token_pools,
)
from .model import PhysicalOutputGenerator, TokenResamplerMemory
from .physical_tokens import SplitReservedTokenPool
from .train import seed_everything
from .train_bidirectional import generation_metrics, parse_json_object, pipeline_metrics
from .train_meta_registration import (
    MetaRegistrationModel,
    _disable_incompatible_optional_torchao,
    _epoch_batches as _retrieval_epoch_batches,
    _provide_optional_tensor_parallel_compat,
    _tokenize_bound_episodes,
    evaluate as evaluate_selection,
    multi_positive_selection_loss,
    render_query,
    _last_state,
)


DEFAULT_DOCUMENT_INSTRUCTION = "Compile this tool for runtime registration."
DEFAULT_SELECTION_DOCUMENT_INSTRUCTION = "Represent this tool for registration."


@dataclass(frozen=True)
class SameStreamGeneration:
    selected_token_id: int
    selected_registry_index: int | None
    generated_token_ids: tuple[int, ...]
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train input-side latent memory for registered tool-token readback"
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--retrieval-adapter-path", required=True)
    parser.add_argument("--retrieval-compiler-path")
    parser.add_argument("--memory-compiler-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--compiler-rank", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-document-length", type=int, default=384)
    parser.add_argument(
        "--document-instruction",
        default=DEFAULT_DOCUMENT_INSTRUCTION,
    )
    parser.add_argument(
        "--selection-document-instruction",
        default=None,
        help="Selection view; defaults to --document-instruction for a shared bundle.",
    )
    parser.add_argument("--max-query-length", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument(
        "--selection-eval-samples",
        type=int,
        help="Selection/full-vocabulary sample limit; defaults to --eval-samples.",
    )
    parser.add_argument("--generation-samples", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--wrong-memory-margin", type=float, default=0.0)
    parser.add_argument("--wrong-memory-weight", type=float, default=0.0)
    parser.add_argument("--schema-weight", type=float, default=0.0)
    parser.add_argument("--selection-weight", type=float, default=0.0)
    parser.add_argument("--selection-registry-size", type=int, default=8)
    parser.add_argument(
        "--selection-softmax",
        choices=("registry", "full_vocabulary"),
        default="registry",
    )
    return parser.parse_args()


def canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def schema_key_object(tool: PreparedTool) -> dict[str, None]:
    parameters = tool.parameters or {}
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return {}
    return {str(key): None for key in sorted(properties)}


def canonical_schema_keys(tool: PreparedTool) -> str:
    return canonical_arguments(schema_key_object(tool))


def render_readback_prompt(
    tokenizer,
    query: str,
    *,
    selected_token: str | None = None,
    document: str | None = None,
    target_kind: str = "arguments",
) -> str:
    if target_kind == "arguments":
        instruction = "Return only one compact JSON object containing the selected tool arguments."
    elif target_kind == "schema":
        instruction = (
            "Return only one compact JSON object mapping every declared parameter "
            "name of the selected tool to null."
        )
    else:
        raise ValueError(f"Unknown readback target kind: {target_kind}")
    user = f"Request: {query}"
    if document is not None:
        user += f"\nTool definition:\n{document}"
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt = f"System: {messages[0]['content']}\nUser: {user}\nAssistant:"
    return prompt + (selected_token or "")


def logical_slot_for_example(
    example: PreparedReadbackExample,
    token_pool: range,
    *,
    seed: int,
    epoch: int,
) -> int:
    digest = hashlib.sha256(
        f"{seed}:{epoch}:{example.source_id}:{example.call_index}".encode("utf-8")
    ).digest()
    return token_pool.start + int.from_bytes(digest[:8], "big") % len(token_pool)


def _epoch_batches(
    examples: list[PreparedReadbackExample],
    *,
    epoch: int,
    seed: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> Iterable[list[PreparedReadbackExample]]:
    indices = list(range(len(examples)))
    random.Random(seed + epoch).shuffle(indices)
    global_batch_size = batch_size * world_size
    usable = len(indices) - len(indices) % global_batch_size
    local = indices[rank:usable:world_size]
    for start in range(0, len(local), batch_size):
        yield [examples[index] for index in local[start : start + batch_size]]


def wrong_memory_indices(tool_identities: list[str]) -> list[int] | None:
    """Choose a different tool memory for every row, reusing rows if needed."""
    if len(set(tool_identities)) < 2:
        return None
    result: list[int] = []
    for index, identity in enumerate(tool_identities):
        for offset in range(1, len(tool_identities) + 1):
            candidate = (index + offset) % len(tool_identities)
            if tool_identities[candidate] != identity:
                result.append(candidate)
                break
    return result


def wrong_tool_lookup(
    examples: list[PreparedReadbackExample],
) -> dict[str, str]:
    identities = sorted({example.tool_identity_hash for example in examples})
    if len(identities) < 2:
        raise ValueError("Wrong-memory evaluation requires at least two tools")
    return {
        identity: identities[(index + 1) % len(identities)]
        for index, identity in enumerate(identities)
    }


def selection_registry_size_for_batch(
    episodes: list[PreparedRetrievalEpisode], minimum_size: int
) -> int:
    if minimum_size < 1:
        raise ValueError("Minimum selection registry size must be positive")
    if not episodes:
        raise ValueError("Selection batch must not be empty")
    largest_target_set = max(
        len(set(episode.target_identity_hashes)) for episode in episodes
    )
    return max(minimum_size, largest_target_set)


def active_registry_full_logits(
    base_logits: torch.Tensor,
    registry_logits: torch.Tensor,
    physical_ids: torch.Tensor,
    reserved_token_ids: torch.Tensor,
    *,
    validate_ids: bool = True,
) -> torch.Tensor:
    """Expose active dynamic rows while keeping ordinary vocabulary competition."""
    if base_logits.ndim != 2:
        raise ValueError("Base vocabulary logits must have shape [batch, vocabulary]")
    if registry_logits.ndim != 2 or physical_ids.shape != registry_logits.shape:
        raise ValueError("Registry logits and physical IDs must share [batch, registry]")
    if len(base_logits) != len(registry_logits):
        raise ValueError("Base and registry logits must use the same batch size")
    if physical_ids.dtype != torch.long or reserved_token_ids.dtype != torch.long:
        raise ValueError("Physical and reserved token IDs must be integer tensors")
    if reserved_token_ids.ndim != 1:
        raise ValueError("Reserved token IDs must be one-dimensional")
    if validate_ids:
        if physical_ids.numel() and (
            int(physical_ids.min()) < 0
            or int(physical_ids.max()) >= base_logits.shape[1]
        ):
            raise ValueError("A physical token ID is outside the vocabulary")
        if reserved_token_ids.numel() and (
            int(reserved_token_ids.min()) < 0
            or int(reserved_token_ids.max()) >= base_logits.shape[1]
        ):
            raise ValueError("A reserved token ID is outside the vocabulary")
        reserved = set(reserved_token_ids.detach().cpu().tolist())
        active = physical_ids.detach().cpu().tolist()
        if any(token_id not in reserved for row in active for token_id in row):
            raise ValueError("Every active physical ID must belong to the reserved pool")
        if any(len(set(row)) != len(row) for row in active):
            raise ValueError("An active registry cannot reuse a physical token ID")

    logits = base_logits.float().clone()
    logits[:, reserved_token_ids] = float("-inf")
    return logits.scatter(1, physical_ids, registry_logits.float())


def full_vocabulary_selection_loss(
    base_logits: torch.Tensor,
    registry_logits: torch.Tensor,
    physical_ids: torch.Tensor,
    reserved_token_ids: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    validate_ids: bool = True,
) -> torch.Tensor:
    if positive_mask.shape != registry_logits.shape:
        raise ValueError("Positive mask and registry logits must have the same shape")
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every registry episode must have at least one positive")
    full_logits = active_registry_full_logits(
        base_logits,
        registry_logits,
        physical_ids,
        reserved_token_ids,
        validate_ids=validate_ids,
    )
    positive_logits = registry_logits.float().masked_fill(
        ~positive_mask, float("-inf")
    )
    return (
        torch.logsumexp(full_logits, dim=1)
        - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def _decoder(backbone: nn.Module) -> nn.Module:
    causal_lm = (
        backbone.get_base_model()
        if hasattr(backbone, "get_base_model")
        else backbone
    )
    return causal_lm.model


class MetaReadbackModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        memory_compiler: TokenResamplerMemory,
        output_compiler: PhysicalOutputGenerator | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.memory_compiler = memory_compiler
        self.output_compiler = output_compiler

    def encode_documents(
        self, document_tokens: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return _decoder(self.backbone)(
            **document_tokens, use_cache=False, return_dict=True
        ).last_hidden_state

    def register_from_states(
        self,
        states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.memory_compiler(
            states, attention_mask.bool()
        )

    def register(
        self, document_tokens: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        states = self.encode_documents(document_tokens)
        return self.register_from_states(states, document_tokens["attention_mask"])

    def register_bundle(
        self, document_tokens: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one tool's output row and input memory from one document pass."""
        if self.output_compiler is None:
            raise ValueError("A complete registration bundle requires an output compiler")
        states = self.encode_documents(document_tokens)
        attention_mask = document_tokens["attention_mask"]
        memory = self.register_from_states(states, attention_mask)
        document_mask = attention_mask.unsqueeze(-1)
        document_states = (
            (states * document_mask).sum(dim=1)
            / document_mask.sum(dim=1).clamp_min(1)
        )
        output_rows = self.output_compiler.output_rows(document_states)
        return output_rows, memory

    def readback_loss(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss, _ = self.readback_nll(inputs_embeds, attention_mask, labels)
        return loss

    def readback_nll(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = _decoder(self.backbone)(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        logits = self.backbone.get_output_embeddings()(hidden[:, :-1])
        shifted_labels = labels[:, 1:]
        token_nll = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shifted_labels)
        target_mask = shifted_labels.ne(-100)
        global_nll = token_nll.sum() / target_mask.sum().clamp_min(1)
        example_nll = token_nll.sum(dim=1) / target_mask.sum(dim=1).clamp_min(1)
        return global_nll, example_nll

    def selection_logits(
        self,
        query_tokens: dict[str, torch.Tensor],
        document_tokens: dict[str, torch.Tensor],
        positive_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.output_compiler is None:
            raise ValueError("Selection logits require an output compiler")
        batch_size, registry_size = positive_mask.shape
        query_hidden = _decoder(self.backbone)(
            **query_tokens, use_cache=False, return_dict=True
        ).last_hidden_state
        query_states = _last_state(query_hidden, query_tokens["attention_mask"])
        document_hidden = _decoder(self.backbone)(
            **document_tokens, use_cache=False, return_dict=True
        ).last_hidden_state
        document_mask = document_tokens["attention_mask"].unsqueeze(-1)
        document_states = (
            (document_hidden * document_mask).sum(dim=1)
            / document_mask.sum(dim=1).clamp_min(1)
        )
        output_rows = self.output_compiler.output_rows(document_states)
        output_rows = output_rows.reshape(batch_size, registry_size, -1)
        registry_logits = torch.einsum(
            "bh,brh->br", query_states.float(), output_rows.float()
        )
        return query_states, registry_logits

    def selection_loss(
        self,
        query_tokens: dict[str, torch.Tensor],
        document_tokens: dict[str, torch.Tensor],
        positive_mask: torch.Tensor,
        physical_ids: torch.Tensor | None,
        reserved_token_ids: torch.Tensor | None,
        softmax_mode: str,
    ) -> torch.Tensor:
        query_states, registry_logits = self.selection_logits(
            query_tokens, document_tokens, positive_mask
        )
        if softmax_mode == "registry":
            return multi_positive_selection_loss(registry_logits, positive_mask)
        if softmax_mode != "full_vocabulary":
            raise ValueError(f"Unknown selection softmax mode: {softmax_mode}")
        if physical_ids is None or reserved_token_ids is None:
            raise ValueError("Full-vocabulary selection requires physical token IDs")
        base_logits = self.backbone.get_output_embeddings()(
            query_states.to(self.backbone.get_output_embeddings().weight.dtype)
        )
        return full_vocabulary_selection_loss(
            base_logits,
            registry_logits,
            physical_ids,
            reserved_token_ids,
            positive_mask,
            validate_ids=False,
        )

    def forward(
        self,
        document_tokens: dict[str, torch.Tensor],
        tokenizer,
        examples: list[PreparedReadbackExample],
        tools: list[PreparedTool],
        physical_ids: list[int],
        *,
        selection_query_tokens: dict[str, torch.Tensor] | None,
        selection_document_tokens: dict[str, torch.Tensor] | None,
        selection_positive_mask: torch.Tensor | None,
        selection_physical_ids: torch.Tensor | None,
        selection_reserved_token_ids: torch.Tensor | None,
        selection_softmax: str,
        max_prompt_length: int,
        max_target_length: int,
        wrong_memory_margin: float,
        wrong_memory_weight: float,
        schema_weight: float,
        selection_weight: float,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.output_compiler is None:
            memory = self.register(document_tokens)
        else:
            _, memory = self.register_bundle(document_tokens)
        inputs, attention, labels = prepare_conditioned_batch(
            self.backbone,
            tokenizer,
            examples,
            tools,
            physical_ids,
            memory,
            condition="registered_memory",
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            device=device,
        )
        correct_loss, correct_example_nll = self.readback_nll(
            inputs, attention, labels
        )
        wrong_loss = correct_loss.detach()
        ranking_loss = correct_loss.new_zeros(())
        wrong_indices = wrong_memory_indices(
            [example.tool_identity_hash for example in examples]
        )
        if wrong_memory_weight > 0.0 and wrong_indices is not None:
            wrong_memory = memory[
                torch.tensor(wrong_indices, dtype=torch.long, device=memory.device)
            ]
            wrong_inputs, wrong_attention, wrong_labels = prepare_conditioned_batch(
                self.backbone,
                tokenizer,
                examples,
                tools,
                physical_ids,
                wrong_memory,
                condition="wrong_memory",
                max_prompt_length=max_prompt_length,
                max_target_length=max_target_length,
                device=device,
            )
            wrong_loss, wrong_example_nll = self.readback_nll(
                wrong_inputs, wrong_attention, wrong_labels
            )
            ranking_loss = F.relu(
                wrong_memory_margin + correct_example_nll - wrong_example_nll
            ).mean()

        schema_loss = correct_loss.new_zeros(())
        if schema_weight > 0.0:
            schema_inputs, schema_attention, schema_labels = prepare_conditioned_batch(
                self.backbone,
                tokenizer,
                examples,
                tools,
                physical_ids,
                memory,
                condition="registered_memory",
                target_kind="schema",
                max_prompt_length=max_prompt_length,
                max_target_length=max_target_length,
                device=device,
            )
            schema_loss, _ = self.readback_nll(
                schema_inputs, schema_attention, schema_labels
            )
        else:
            schema_attention = attention.new_zeros(())

        selection_loss = correct_loss.new_zeros(())
        if selection_weight > 0.0:
            selection_inputs = (
                selection_query_tokens,
                selection_document_tokens,
                selection_positive_mask,
                selection_physical_ids,
            )
            if any(value is None for value in selection_inputs):
                raise ValueError("Selection retention requires a complete selection batch")
            assert selection_query_tokens is not None
            assert selection_document_tokens is not None
            assert selection_positive_mask is not None
            assert selection_physical_ids is not None
            selection_loss = self.selection_loss(
                selection_query_tokens,
                selection_document_tokens,
                selection_positive_mask,
                selection_physical_ids,
                selection_reserved_token_ids,
                selection_softmax,
            )

        total_loss = (
            correct_loss
            + wrong_memory_weight * ranking_loss
            + schema_weight * schema_loss
            + selection_weight * selection_loss
        )
        return (
            total_loss,
            attention.sum() + schema_attention.sum(),
            correct_loss.detach(),
            wrong_loss.detach(),
            ranking_loss.detach(),
            schema_loss.detach(),
            selection_loss.detach(),
        )


def tokenize_documents(
    tokenizer,
    tools: list[PreparedTool],
    *,
    max_length: int,
    device: torch.device,
    document_instruction: str = DEFAULT_DOCUMENT_INSTRUCTION,
) -> dict[str, torch.Tensor]:
    texts = [f"{document_instruction}\n{tool.document}" for tool in tools]
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)


def prepare_conditioned_batch(
    backbone: nn.Module,
    tokenizer,
    examples: list[PreparedReadbackExample],
    tools: list[PreparedTool],
    physical_ids: list[int],
    memory: torch.Tensor | None,
    *,
    condition: str,
    target_kind: str = "arguments",
    max_prompt_length: int,
    max_target_length: int,
    device: torch.device,
    include_targets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if condition not in {
        "registered_memory",
        "wrong_memory",
        "static_blank",
        "query_only",
        "full_document",
    }:
        raise ValueError(f"Unknown readback condition: {condition}")
    if memory is not None and len(memory) != len(examples):
        raise ValueError("Memory and readback example batches differ")
    embedding = backbone.get_input_embeddings()
    sequences: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for index, (example, tool, physical_id) in enumerate(
        zip(examples, tools, physical_ids)
    ):
        token_string = tokenizer.convert_ids_to_tokens(physical_id)
        prompt = render_readback_prompt(
            tokenizer,
            example.query,
            selected_token=token_string
            if condition in {"registered_memory", "wrong_memory", "static_blank"}
            else None,
            document=tool.document if condition == "full_document" else None,
            target_kind=target_kind,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        prompt_embeddings = embedding(prompt_tensor).detach()
        if condition in {"registered_memory", "wrong_memory"}:
            positions = (prompt_tensor == physical_id).nonzero(as_tuple=False).flatten()
            if positions.numel() != 1 or memory is None:
                raise ValueError("Registered prompt must contain exactly one physical token")
            position = int(positions.item())
            prefix = memory[index].to(prompt_embeddings.dtype)
            prompt_embeddings = torch.cat(
                [
                    prompt_embeddings[:position],
                    prefix,
                    prompt_embeddings[position + 1 :],
                ]
            )
        if include_targets:
            if target_kind == "arguments":
                target = canonical_arguments(example.arguments)
            elif target_kind == "schema":
                target = canonical_schema_keys(tool)
            else:
                raise ValueError(f"Unknown readback target kind: {target_kind}")
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=max_target_length - 1,
            ).input_ids + [tokenizer.eos_token_id]
            target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
            target_embeddings = embedding(target_tensor).detach()
            sequences.append(torch.cat([prompt_embeddings, target_embeddings]))
            label_rows.append(
                torch.cat(
                    [
                        torch.full(
                            (len(prompt_embeddings),),
                            -100,
                            dtype=torch.long,
                            device=device,
                        ),
                        target_tensor,
                    ]
                )
            )
        else:
            sequences.append(prompt_embeddings)
            label_rows.append(
                torch.full(
                    (len(prompt_embeddings),), -100, dtype=torch.long, device=device
                )
            )
    max_length = max(len(sequence) for sequence in sequences)
    hidden_size = sequences[0].shape[-1]
    inputs = torch.zeros(
        len(sequences),
        max_length,
        hidden_size,
        dtype=sequences[0].dtype,
        device=device,
    )
    attention = torch.zeros(
        len(sequences), max_length, dtype=torch.long, device=device
    )
    labels = torch.full(
        (len(sequences), max_length), -100, dtype=torch.long, device=device
    )
    for row, (sequence, row_labels) in enumerate(zip(sequences, label_rows)):
        inputs[row, : len(sequence)] = sequence
        attention[row, : len(sequence)] = 1
        labels[row, : len(row_labels)] = row_labels
    return inputs, attention, labels


@torch.inference_mode()
def generate_one(
    model: MetaReadbackModel,
    tokenizer,
    example: PreparedReadbackExample,
    tool: PreparedTool,
    physical_id: int,
    *,
    condition: str,
    target_kind: str = "arguments",
    memory_tool: PreparedTool | None = None,
    max_document_length: int,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
    document_instruction: str = DEFAULT_DOCUMENT_INSTRUCTION,
) -> str:
    model.eval()
    memory = None
    if condition in {"registered_memory", "wrong_memory"}:
        document_tokens = tokenize_documents(
            tokenizer,
            [memory_tool or tool],
            max_length=max_document_length,
            device=device,
            document_instruction=document_instruction,
        )
        if model.output_compiler is None:
            memory = model.register(document_tokens)
        else:
            _, memory = model.register_bundle(document_tokens)
    inputs, attention, _ = prepare_conditioned_batch(
        model.backbone,
        tokenizer,
        [example],
        [tool],
        [physical_id],
        memory,
        condition=condition,
        target_kind=target_kind,
        max_prompt_length=max_prompt_length,
        max_target_length=2,
        device=device,
        include_targets=False,
    )
    output = model.backbone(
        inputs_embeds=inputs,
        attention_mask=attention,
        use_cache=True,
        return_dict=True,
    )
    next_token = output.logits[:, -1].argmax(dim=-1)
    past = output.past_key_values
    running_attention = attention
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        running_attention = torch.cat(
            [
                running_attention,
                torch.ones((1, 1), dtype=running_attention.dtype, device=device),
            ],
            dim=1,
        )
        output = model.backbone(
            input_ids=next_token[:, None],
            attention_mask=running_attention,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def generate_same_stream_one(
    model: MetaReadbackModel,
    tokenizer,
    query: str,
    registered_output_rows: torch.Tensor,
    registered_memory: torch.Tensor,
    physical_ids: torch.Tensor,
    reserved_token_ids: torch.Tensor,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> SameStreamGeneration:
    """Emit one physical ID, dereference its memory, and continue one KV stream."""
    if registered_output_rows.ndim != 2:
        raise ValueError("Registered output rows must have shape [registry, hidden]")
    if registered_memory.ndim != 3:
        raise ValueError("Registered memory must have shape [registry, slots, hidden]")
    if physical_ids.ndim != 1 or physical_ids.dtype != torch.long:
        raise ValueError("Physical IDs must be a one-dimensional integer tensor")
    registry_size = len(registered_output_rows)
    if len(registered_memory) != registry_size or len(physical_ids) != registry_size:
        raise ValueError("Output rows, memory, and physical IDs must share a registry")

    prompt = render_query(tokenizer, query)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    prompt_tensor = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )
    attention = torch.ones_like(prompt_tensor)
    output = model.backbone(
        input_ids=prompt_tensor,
        attention_mask=attention,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )
    query_state = output.hidden_states[-1][:, -1]
    base_logits = model.backbone.get_output_embeddings()(
        query_state.to(model.backbone.get_output_embeddings().weight.dtype)
    )
    registry_logits = torch.einsum(
        "bh,rh->br", query_state.float(), registered_output_rows.float()
    )
    full_logits = active_registry_full_logits(
        base_logits,
        registry_logits,
        physical_ids.unsqueeze(0),
        reserved_token_ids,
    )
    selected_token_id = int(full_logits.argmax(dim=1).item())
    selected_positions = (physical_ids == selected_token_id).nonzero(
        as_tuple=False
    ).flatten()
    if selected_positions.numel() == 0:
        return SameStreamGeneration(selected_token_id, None, (), "")
    if selected_positions.numel() != 1:
        raise ValueError("A selected physical ID maps to multiple registry entries")
    selected_index = int(selected_positions.item())

    memory = registered_memory[selected_index : selected_index + 1].to(
        model.backbone.get_input_embeddings().weight.dtype
    )
    running_attention = torch.cat(
        [
            attention,
            torch.ones(
                (1, memory.shape[1]),
                dtype=attention.dtype,
                device=device,
            ),
        ],
        dim=1,
    )
    output = model.backbone(
        inputs_embeds=memory,
        attention_mask=running_attention,
        past_key_values=output.past_key_values,
        use_cache=True,
        return_dict=True,
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
                    (1, 1), dtype=running_attention.dtype, device=device
                ),
            ],
            dim=1,
        )
        output = model.backbone(
            input_ids=next_token[:, None],
            attention_mask=running_attention,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1)
    generated_ids = tuple(generated)
    return SameStreamGeneration(
        selected_token_id,
        selected_index,
        generated_ids,
        tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
    )


@torch.inference_mode()
def evaluate_generation(
    model: MetaReadbackModel,
    tokenizer,
    tools: dict[str, PreparedTool],
    examples: list[PreparedReadbackExample],
    physical_pool: SplitReservedTokenPool,
    logical_pools: dict[str, range],
    *,
    target_kind: str = "arguments",
    sample_limit: int,
    seed: int,
    max_document_length: int,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
    rank: int,
    world_size: int,
    document_instruction: str = DEFAULT_DOCUMENT_INSTRUCTION,
) -> dict[str, dict[str, float]]:
    selected = sorted(
        (example for example in examples if example.call_count == 1),
        key=lambda item: (item.query_hash, item.source_id),
    )[:sample_limit]
    local = selected[rank::world_size]
    conditions = (
        "registered_memory",
        "wrong_memory",
        "static_blank",
        "query_only",
        "full_document",
    )
    wrong_tools = wrong_tool_lookup(examples)
    local_rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    for example in local:
        tool = tools[example.tool_identity_hash]
        logical_slot = logical_slot_for_example(
            example, logical_pools[example.split], seed=seed, epoch=0
        )
        physical_id = physical_pool.physical_id(logical_slot)
        for condition in conditions:
            decoded = generate_one(
                model,
                tokenizer,
                example,
                tool,
                physical_id,
                condition=condition,
                target_kind=target_kind,
                memory_tool=(
                    tools[wrong_tools[example.tool_identity_hash]]
                    if condition == "wrong_memory"
                    else None
                ),
                max_document_length=max_document_length,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                device=device,
                document_instruction=document_instruction,
            )
            local_rows[condition].append(
                {
                    "parsed": parse_json_object(decoded),
                    "target": (
                        example.arguments
                        if target_kind == "arguments"
                        else schema_key_object(tool)
                    ),
                }
            )
    gathered: list[dict[str, list[dict[str, Any]]] | None]
    if dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered = [local_rows]
    results: dict[str, dict[str, float]] = {}
    for condition in conditions:
        rows = [
            row
            for rank_rows in gathered
            if rank_rows is not None
            for row in rank_rows[condition]
        ]
        results[condition] = generation_metrics(rows)
        results[condition]["examples"] = float(len(rows))
    return results


@torch.inference_mode()
def evaluate_same_stream_generation(
    model: MetaReadbackModel,
    tokenizer,
    tools: dict[str, PreparedTool],
    examples: list[PreparedReadbackExample],
    sampler: EpisodicRegistrySampler,
    physical_pool: SplitReservedTokenPool,
    *,
    split: str,
    registry_size: int,
    sample_limit: int,
    max_document_length: int,
    max_prompt_length: int,
    max_new_tokens: int,
    document_instruction: str,
    selection_document_instruction: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float | int | str]:
    if model.output_compiler is None:
        raise ValueError("Same-stream generation requires an output compiler")
    model.eval()
    selected = sorted(
        (example for example in examples if example.call_count == 1),
        key=lambda item: (item.query_hash, item.source_id),
    )[:sample_limit]
    local = selected[rank::world_size]
    reserved_ids = torch.tensor(
        physical_pool.token_ids, dtype=torch.long, device=device
    )
    local_rows: list[dict[str, Any]] = []
    shared_document_view = document_instruction == selection_document_instruction
    for example in local:
        episode = PreparedRetrievalEpisode(
            query_hash=example.query_hash,
            query=example.query,
            split=example.split,
            target_identity_hashes=(example.tool_identity_hash,),
        )
        bound = sampler.bind(episode, registry_size=registry_size, epoch=0)
        selection_tokens = tokenize_documents(
            tokenizer,
            list(bound.tools),
            max_length=max_document_length,
            device=device,
            document_instruction=selection_document_instruction,
        )
        output_rows, selection_memory = model.register_bundle(selection_tokens)
        if shared_document_view:
            memory = selection_memory
        else:
            memory_tokens = tokenize_documents(
                tokenizer,
                list(bound.tools),
                max_length=max_document_length,
                device=device,
                document_instruction=document_instruction,
            )
            memory = model.register(memory_tokens)
        physical_ids = torch.tensor(
            [physical_pool.physical_id(slot) for slot in bound.slot_indices],
            dtype=torch.long,
            device=device,
        )
        generated = generate_same_stream_one(
            model,
            tokenizer,
            example.query,
            output_rows,
            memory,
            physical_ids,
            reserved_ids,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        selected_tool_id = (
            None
            if generated.selected_registry_index is None
            else bound.tools[generated.selected_registry_index].identity_hash
        )
        local_rows.append(
            {
                "tool_id": example.tool_identity_hash,
                "selected_tool_id": selected_tool_id,
                "selected_token_id": generated.selected_token_id,
                "selected_registry_index": generated.selected_registry_index,
                "parsed": parse_json_object(generated.text),
                "target": example.arguments,
            }
        )
    if dist.is_initialized():
        gathered: list[list[dict[str, Any]] | None] = [
            None for _ in range(world_size)
        ]
        dist.all_gather_object(gathered, local_rows)
        rows = [
            row
            for rank_rows in gathered
            if rank_rows is not None
            for row in rank_rows
        ]
    else:
        rows = local_rows
    pipeline = pipeline_metrics(rows)
    emitted = sum(row["selected_registry_index"] is not None for row in rows)
    end_to_end_rows = [
        {
            "parsed": (
                row["parsed"]
                if row["selected_tool_id"] == row["tool_id"]
                else None
            ),
            "target": row["target"],
        }
        for row in rows
    ]
    end_to_end = generation_metrics(end_to_end_rows)
    return {
        "split": split,
        "registry_size": registry_size,
        "registration_document_forwards_per_tool": (
            1 if shared_document_view else 2
        ),
        "examples": len(rows),
        "physical_token_emission_rate": emitted / max(1, len(rows)),
        "ordinary_token_win_rate": 1.0 - emitted / max(1, len(rows)),
        **pipeline,
        "end_to_end_key_precision": end_to_end["key_precision"],
        "end_to_end_key_recall": end_to_end["key_recall"],
        "end_to_end_shared_key_value_accuracy": end_to_end[
            "shared_key_value_accuracy"
        ],
    }


@torch.inference_mode()
def evaluate_teacher_forced(
    model: MetaReadbackModel,
    tokenizer,
    tools: dict[str, PreparedTool],
    examples: list[PreparedReadbackExample],
    physical_pool: SplitReservedTokenPool,
    logical_pools: dict[str, range],
    *,
    target_kind: str = "arguments",
    sample_limit: int,
    seed: int,
    max_document_length: int,
    max_prompt_length: int,
    max_target_length: int,
    device: torch.device,
    rank: int,
    world_size: int,
    document_instruction: str = DEFAULT_DOCUMENT_INSTRUCTION,
) -> dict[str, dict[str, float]]:
    model.eval()
    selected = sorted(
        examples, key=lambda item: (item.query_hash, item.source_id, item.call_index)
    )[:sample_limit]
    local = selected[rank::world_size]
    conditions = (
        "registered_memory",
        "wrong_memory",
        "static_blank",
        "query_only",
        "full_document",
    )
    wrong_tools = wrong_tool_lookup(examples)
    totals = {
        condition: torch.zeros(2, dtype=torch.float64, device=device)
        for condition in conditions
    }
    for example in local:
        tool = tools[example.tool_identity_hash]
        logical_slot = logical_slot_for_example(
            example, logical_pools[example.split], seed=seed, epoch=0
        )
        physical_id = physical_pool.physical_id(logical_slot)
        registered_tokens = tokenize_documents(
            tokenizer,
            [tool],
            max_length=max_document_length,
            device=device,
            document_instruction=document_instruction,
        )
        wrong_tokens = tokenize_documents(
            tokenizer,
            [tools[wrong_tools[example.tool_identity_hash]]],
            max_length=max_document_length,
            device=device,
            document_instruction=document_instruction,
        )
        if model.output_compiler is None:
            registered_memory = model.register(registered_tokens)
            wrong_memory = model.register(wrong_tokens)
        else:
            _, registered_memory = model.register_bundle(registered_tokens)
            _, wrong_memory = model.register_bundle(wrong_tokens)
        for condition in conditions:
            memory = (
                registered_memory
                if condition == "registered_memory"
                else wrong_memory
                if condition == "wrong_memory"
                else None
            )
            inputs, attention, labels = prepare_conditioned_batch(
                model.backbone,
                tokenizer,
                [example],
                [tool],
                [physical_id],
                memory,
                condition=condition,
                target_kind=target_kind,
                max_prompt_length=max_prompt_length,
                max_target_length=max_target_length,
                device=device,
            )
            loss = model.readback_loss(inputs, attention, labels)
            totals[condition] += torch.tensor(
                [float(loss), 1.0], dtype=torch.float64, device=device
            )
    results: dict[str, dict[str, float]] = {}
    for condition, values in totals.items():
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
        count = max(1.0, float(values[1].item()))
        results[condition] = {
            "examples": float(values[1].item()),
            "mean_nll": float(values[0].item() / count),
        }
    return results


@torch.inference_mode()
def evaluate_full_vocabulary_selection(
    model: MetaReadbackModel,
    tokenizer,
    sampler: EpisodicRegistrySampler,
    episodes: list[PreparedRetrievalEpisode],
    physical_pool: SplitReservedTokenPool,
    *,
    split: str,
    registry_size: int,
    sample_limit: int,
    max_query_length: int,
    max_document_length: int,
    document_instruction: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float | int | str]:
    model.eval()
    selected = sorted(episodes, key=lambda item: item.query_hash)[:sample_limit]
    local = selected[rank::world_size]
    reserved_ids = torch.tensor(
        physical_pool.token_ids, dtype=torch.long, device=device
    )
    totals = torch.zeros(7, dtype=torch.float64, device=device)
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
        physical_ids = torch.tensor(
            [
                [physical_pool.physical_id(slot) for slot in bound.slot_indices]
            ],
            dtype=torch.long,
            device=device,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            query_states, registry_logits = model.selection_logits(
                query_tokens, document_tokens, positive_mask
            )
            base_logits = model.backbone.get_output_embeddings()(
                query_states.to(model.backbone.get_output_embeddings().weight.dtype)
            )
        full_logits = active_registry_full_logits(
            base_logits,
            registry_logits,
            physical_ids,
            reserved_ids,
            validate_ids=False,
        )
        loss = full_vocabulary_selection_loss(
            base_logits,
            registry_logits,
            physical_ids,
            reserved_ids,
            positive_mask,
            validate_ids=False,
        )
        positive_physical_ids = physical_ids[positive_mask]
        predicted_id = int(full_logits.argmax(dim=1).item())
        full_hit = int((positive_physical_ids == predicted_id).any().item())
        registry_hit = int(
            positive_mask[0, int(registry_logits.argmax(dim=1).item())].item()
        )
        best_positive_logit = registry_logits[positive_mask].max()
        target_rank = 1 + int((full_logits[0] > best_positive_logit).sum().item())
        registered_log_mass = (
            torch.logsumexp(registry_logits.float(), dim=1)
            - torch.logsumexp(full_logits.float(), dim=1)
        )
        totals += torch.tensor(
            [
                float(loss),
                full_hit,
                registry_hit,
                target_rank,
                int(predicted_id < physical_pool.original_vocab_size),
                float(registered_log_mass.exp().item()),
                1.0,
            ],
            dtype=torch.float64,
            device=device,
        )
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(1.0, float(totals[6].item()))
    return {
        "split": split,
        "softmax": "original_vocabulary_plus_active_registry",
        "episodes": int(totals[6].item()),
        "loss": float(totals[0].item() / count),
        "hit_at_1": float(totals[1].item() / count),
        "registry_hit_at_1": float(totals[2].item() / count),
        "mean_target_rank": float(totals[3].item() / count),
        "ordinary_token_win_rate": float(totals[4].item() / count),
        "mean_active_registry_probability": float(totals[5].item() / count),
    }


def main() -> None:
    args = parse_args()
    if args.selection_document_instruction is None:
        args.selection_document_instruction = args.document_instruction
    for name in ("wrong_memory_weight", "schema_weight", "selection_weight"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.selection_weight > 0.0 and not args.retrieval_compiler_path:
        raise ValueError(
            "--retrieval-compiler-path is required when selection retention is enabled"
        )
    if args.selection_registry_size < 2:
        raise ValueError("--selection-registry-size must be at least two")
    if args.selection_eval_samples is not None and args.selection_eval_samples < 1:
        raise ValueError("--selection-eval-samples must be positive")
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
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    prepared_dir = Path(args.prepared_dir)
    tools = load_prepared_tools(prepared_dir / "tools.jsonl")
    examples = load_readback_examples(prepared_dir / "readback.jsonl", tools)
    examples_by_split = {
        split: [example for example in examples if example.split == split]
        for split in ("train", "validation", "test")
    }
    logical_pools = load_token_pools(prepared_dir / "split_manifest.json")
    retrieval_episodes: list[PreparedRetrievalEpisode] = []
    retrieval_by_split: dict[str, list[PreparedRetrievalEpisode]] = {}
    retrieval_sampler: EpisodicRegistrySampler | None = None
    if args.selection_weight > 0.0:
        retrieval_episodes = load_retrieval_episodes(
            prepared_dir / "retrieval.jsonl", tools
        )
        retrieval_by_split = {
            split: [episode for episode in retrieval_episodes if episode.split == split]
            for split in ("train", "validation", "test")
        }
        retrieval_sampler = EpisodicRegistrySampler(
            tools, logical_pools, seed=args.seed
        )

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Meta-readback training requires peft") from exc
    disabled_torchao = _disable_incompatible_optional_torchao()
    provided_tp_compat = _provide_optional_tensor_parallel_compat()
    if is_main:
        print(
            f"runtime disabled_incompatible_optional_torchao={str(disabled_torchao).lower()} "
            f"provided_optional_tensor_parallel_compat={str(provided_tp_compat).lower()}",
            flush=True,
        )

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
    physical_pool = SplitReservedTokenPool.create(tokenizer, backbone, logical_pools)
    reserved_token_ids = torch.tensor(
        physical_pool.token_ids, dtype=torch.long, device=device
    )
    backbone = PeftModel.from_pretrained(
        backbone,
        args.retrieval_adapter_path,
        is_trainable=args.train_lora,
    )
    if args.gradient_checkpointing:
        try:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            backbone.gradient_checkpointing_enable()
        backbone.enable_input_require_grads()
    backbone.config.use_cache = False
    backbone.to(device)

    input_rows = backbone.get_input_embeddings().weight[: physical_pool.original_vocab_size]
    input_norm = float(input_rows.detach().float().norm(dim=-1).mean())
    memory_compiler = TokenResamplerMemory(
        backbone.config.hidden_size,
        args.compiler_rank,
        args.memory_slots,
        input_norm,
    ).to(device)
    if args.memory_compiler_path:
        memory_compiler.load_state_dict(
            torch.load(
                args.memory_compiler_path, map_location=device, weights_only=True
            )
        )
    output_compiler = None
    if args.retrieval_compiler_path:
        output_rows = backbone.get_output_embeddings().weight[
            : physical_pool.original_vocab_size
        ]
        output_norm = float(output_rows.detach().float().norm(dim=-1).mean())
        output_compiler = PhysicalOutputGenerator(
            backbone.config.hidden_size, args.compiler_rank, output_norm
        ).to(device)
        output_compiler.load_state_dict(
            torch.load(
                args.retrieval_compiler_path, map_location=device, weights_only=True
            )
        )
    model: nn.Module = MetaReadbackModel(
        backbone, memory_compiler, output_compiler
    ).to(device)
    if world_size > 1 and not args.eval_only:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    seed_everything(args.seed + rank)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    batches_per_epoch = len(examples_by_split["train"]) // (
        args.batch_size * world_size
    )
    updates_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_steps = (
        0 if args.eval_only else args.max_steps or args.epochs * updates_per_epoch
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        max(1, total_steps),
    )
    if is_main:
        print(
            f"runtime world_size={world_size} global_batch_size="
            f"{args.batch_size * world_size * args.gradient_accumulation_steps} "
            f"trainable_parameters={sum(p.numel() for p in trainable)} "
            f"train_examples={len(examples_by_split['train'])} total_steps={total_steps} "
            f"schema_weight={args.schema_weight} selection_weight={args.selection_weight} "
            f"selection_registry_size={args.selection_registry_size}",
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)
    training_seen_ids: set[int] = set()
    step = 0
    micro_step = 0
    recent_losses: list[float] = []
    recent_correct_losses: list[float] = []
    recent_wrong_losses: list[float] = []
    recent_ranking_losses: list[float] = []
    recent_schema_losses: list[float] = []
    recent_selection_losses: list[float] = []
    interval_tokens = 0
    interval_started = time.monotonic()
    completed_epochs = 0
    training_epochs = (
        0 if args.eval_only else args.epochs if not args.max_steps else 10**9
    )
    for epoch in range(training_epochs):
        model.train()
        epoch_losses: list[float] = []
        selection_batches = None
        if args.selection_weight > 0.0:
            assert retrieval_sampler is not None
            selection_batches = iter(
                _retrieval_epoch_batches(
                    retrieval_by_split["train"],
                    epoch=epoch,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    rank=rank,
                    world_size=world_size,
                )
            )
        for batch in _epoch_batches(
            examples_by_split["train"],
            epoch=epoch,
            seed=args.seed,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        ):
            batch_tools = [tools[example.tool_identity_hash] for example in batch]
            logical_slots = [
                logical_slot_for_example(
                    example, logical_pools["train"], seed=args.seed, epoch=epoch
                )
                for example in batch
            ]
            physical_ids = [physical_pool.physical_id(slot) for slot in logical_slots]
            training_seen_ids.update(physical_ids)
            document_tokens = tokenize_documents(
                tokenizer,
                batch_tools,
                max_length=args.max_document_length,
                device=device,
                document_instruction=args.document_instruction,
            )
            selection_query_tokens = None
            selection_document_tokens = None
            selection_positive_mask = None
            selection_physical_ids = None
            selection_token_count = 0
            if selection_batches is not None:
                try:
                    raw_selection_batch = next(selection_batches)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Retrieval training data ended before the readback epoch"
                    ) from exc
                assert retrieval_sampler is not None
                batch_registry_size = selection_registry_size_for_batch(
                    raw_selection_batch, args.selection_registry_size
                )
                bound_selection_batch = [
                    retrieval_sampler.bind(
                        item,
                        registry_size=batch_registry_size,
                        epoch=epoch,
                    )
                    for item in raw_selection_batch
                ]
                for bound in bound_selection_batch:
                    training_seen_ids.update(
                        physical_pool.physical_id(slot)
                        for slot in bound.slot_indices
                    )
                selection_physical_ids = torch.tensor(
                    [
                        [physical_pool.physical_id(slot) for slot in bound.slot_indices]
                        for bound in bound_selection_batch
                    ],
                    dtype=torch.long,
                    device=device,
                )
                (
                    selection_query_tokens,
                    selection_document_tokens,
                    selection_positive_mask,
                    selection_token_count,
                ) = _tokenize_bound_episodes(
                    tokenizer,
                    bound_selection_batch,
                    max_query_length=args.max_query_length,
                    max_document_length=args.max_document_length,
                    device=device,
                    document_instruction=args.selection_document_instruction,
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
                    (
                        loss,
                        readback_token_count,
                        correct_loss,
                        wrong_loss,
                        ranking_loss,
                        schema_loss,
                        selection_loss,
                    ) = model(
                        document_tokens,
                        tokenizer,
                        batch,
                        batch_tools,
                        physical_ids,
                        selection_query_tokens=selection_query_tokens,
                        selection_document_tokens=selection_document_tokens,
                        selection_positive_mask=selection_positive_mask,
                        selection_physical_ids=selection_physical_ids,
                        selection_reserved_token_ids=reserved_token_ids,
                        selection_softmax=args.selection_softmax,
                        max_prompt_length=args.max_prompt_length,
                        max_target_length=args.max_target_length,
                        wrong_memory_margin=args.wrong_memory_margin,
                        wrong_memory_weight=args.wrong_memory_weight,
                        schema_weight=args.schema_weight,
                        selection_weight=args.selection_weight,
                        device=device,
                    )
                    scaled_loss = loss / args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite readback loss at micro-step {micro_step + 1}")
                scaled_loss.backward()
            loss_value = float(loss.detach())
            epoch_losses.append(loss_value)
            recent_losses.append(loss_value)
            recent_losses = recent_losses[-100:]
            recent_correct_losses.append(float(correct_loss))
            recent_correct_losses = recent_correct_losses[-100:]
            recent_wrong_losses.append(float(wrong_loss))
            recent_wrong_losses = recent_wrong_losses[-100:]
            recent_ranking_losses.append(float(ranking_loss))
            recent_ranking_losses = recent_ranking_losses[-100:]
            recent_schema_losses.append(float(schema_loss))
            recent_schema_losses = recent_schema_losses[-100:]
            recent_selection_losses.append(float(selection_loss))
            recent_selection_losses = recent_selection_losses[-100:]
            interval_tokens += int(readback_token_count.item()) + int(
                document_tokens["attention_mask"].sum().item()
            ) + selection_token_count
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
            if step <= 5 or step % args.log_every == 0:
                elapsed = max(1e-6, time.monotonic() - interval_started)
                if is_main:
                    memory_gib = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    )
                    print(
                        f"step={step} epoch={epoch + 1} batch_loss={loss_value:.6f} "
                        f"recent_loss={sum(recent_losses) / len(recent_losses):.6f} "
                        f"correct_nll={float(correct_loss):.6f} "
                        f"wrong_nll={float(wrong_loss):.6f} "
                        f"ranking_loss={float(ranking_loss):.6f} "
                        f"schema_nll={float(schema_loss):.6f} "
                        f"selection_loss={float(selection_loss):.6f} "
                        f"grad_norm={float(grad_norm):.4f} "
                        f"tokens_per_second={interval_tokens * world_size / elapsed:.1f} "
                        f"max_memory_gib={memory_gib:.2f}",
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

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    if is_main and not args.eval_only:
        trained_checkpoint = output_dir / "trained_checkpoint"
        unwrapped.backbone.save_pretrained(
            trained_checkpoint / "lora", save_embedding_layers=False
        )
        tokenizer.save_pretrained(trained_checkpoint / "tokenizer")
        torch.save(
            unwrapped.memory_compiler.state_dict(),
            trained_checkpoint / "memory_compiler.pt",
        )
        if unwrapped.output_compiler is not None:
            torch.save(
                unwrapped.output_compiler.state_dict(),
                trained_checkpoint / "compiler.pt",
            )
    if dist.is_initialized():
        dist.barrier()
    unwrapped.eval()
    teacher_forced: dict[str, dict[str, dict[str, float]]] = {}
    generation: dict[str, dict[str, dict[str, float]]] = {}
    schema_teacher_forced: dict[str, dict[str, dict[str, float]]] = {}
    schema_generation: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("validation", "test"):
        teacher_forced[split] = evaluate_teacher_forced(
            unwrapped,
            tokenizer,
            tools,
            examples_by_split[split],
            physical_pool,
            logical_pools,
            sample_limit=args.eval_samples,
            seed=args.seed,
            max_document_length=args.max_document_length,
            max_prompt_length=args.max_prompt_length,
            max_target_length=args.max_target_length,
            device=device,
            rank=rank,
            world_size=world_size,
            document_instruction=args.document_instruction,
        )
        generation[split] = evaluate_generation(
            unwrapped,
            tokenizer,
            tools,
            examples_by_split[split],
            physical_pool,
            logical_pools,
            sample_limit=args.generation_samples,
            seed=args.seed,
            max_document_length=args.max_document_length,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
            rank=rank,
            world_size=world_size,
            document_instruction=args.document_instruction,
        )
        schema_teacher_forced[split] = evaluate_teacher_forced(
            unwrapped,
            tokenizer,
            tools,
            examples_by_split[split],
            physical_pool,
            logical_pools,
            target_kind="schema",
            sample_limit=args.eval_samples,
            seed=args.seed,
            max_document_length=args.max_document_length,
            max_prompt_length=args.max_prompt_length,
            max_target_length=args.max_target_length,
            device=device,
            rank=rank,
            world_size=world_size,
            document_instruction=args.document_instruction,
        )
        schema_generation[split] = evaluate_generation(
            unwrapped,
            tokenizer,
            tools,
            examples_by_split[split],
            physical_pool,
            logical_pools,
            target_kind="schema",
            sample_limit=args.generation_samples,
            seed=args.seed,
            max_document_length=args.max_document_length,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
            rank=rank,
            world_size=world_size,
            document_instruction=args.document_instruction,
        )

    selection_retention: dict[str, dict[str, float | int]] = {}
    full_vocabulary_selection: dict[str, dict[str, float | int | str]] = {}
    same_stream_generation: dict[str, dict[str, float | int | str]] = {}
    selection_eval_samples = args.selection_eval_samples or args.eval_samples
    if unwrapped.output_compiler is not None:
        if retrieval_sampler is None:
            retrieval_episodes = load_retrieval_episodes(
                prepared_dir / "retrieval.jsonl", tools
            )
            retrieval_by_split = {
                split: [
                    episode for episode in retrieval_episodes if episode.split == split
                ]
                for split in ("train", "validation", "test")
            }
            retrieval_sampler = EpisodicRegistrySampler(
                tools, logical_pools, seed=args.seed
            )
        selection_model = MetaRegistrationModel(
            unwrapped.backbone, unwrapped.output_compiler
        )
        for split in ("validation", "test"):
            selection_retention[split] = evaluate_selection(
                selection_model,
                tokenizer,
                retrieval_sampler,
                retrieval_by_split[split],
                split=split,
                registry_size=args.selection_registry_size,
                sample_limit=selection_eval_samples,
                max_query_length=args.max_query_length,
                max_document_length=args.max_document_length,
                document_instruction=args.selection_document_instruction,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            if args.selection_softmax == "full_vocabulary":
                full_vocabulary_selection[split] = (
                    evaluate_full_vocabulary_selection(
                        unwrapped,
                        tokenizer,
                        retrieval_sampler,
                        retrieval_by_split[split],
                        physical_pool,
                        split=split,
                        registry_size=args.selection_registry_size,
                        sample_limit=selection_eval_samples,
                        max_query_length=args.max_query_length,
                        max_document_length=args.max_document_length,
                        document_instruction=args.selection_document_instruction,
                        device=device,
                        rank=rank,
                        world_size=world_size,
                    )
                )
                same_stream_generation[split] = evaluate_same_stream_generation(
                    unwrapped,
                    tokenizer,
                    tools,
                    examples_by_split[split],
                    retrieval_sampler,
                    physical_pool,
                    split=split,
                    registry_size=args.selection_registry_size,
                    sample_limit=args.generation_samples,
                    max_document_length=args.max_document_length,
                    max_prompt_length=args.max_prompt_length,
                    max_new_tokens=args.max_new_tokens,
                    document_instruction=args.document_instruction,
                    selection_document_instruction=(
                        args.selection_document_instruction
                    ),
                    device=device,
                    rank=rank,
                    world_size=world_size,
                )
    if dist.is_initialized():
        gathered_seen: list[set[int] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_seen, training_seen_ids)
        training_seen_ids = set().union(*(item or set() for item in gathered_seen))
    audit = physical_pool.audit(unwrapped.backbone, training_seen_ids)
    if audit["evaluation_ids_seen_during_training"] != 0:
        raise AssertionError("Validation or test IDs leaked into readback training")
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
            torch.save(
                unwrapped.memory_compiler.state_dict(),
                output_dir / "memory_compiler.pt",
            )
            if unwrapped.output_compiler is not None:
                torch.save(
                    unwrapped.output_compiler.state_dict(), output_dir / "compiler.pt"
                )
        results = {
            "config": vars(args),
            "world_size": world_size,
            "completed_steps": step,
            "completed_epochs": completed_epochs,
            "recent_train_loss": sum(recent_losses) / max(1, len(recent_losses)),
            "recent_correct_nll": sum(recent_correct_losses)
            / max(1, len(recent_correct_losses)),
            "recent_wrong_memory_nll": sum(recent_wrong_losses)
            / max(1, len(recent_wrong_losses)),
            "recent_ranking_loss": sum(recent_ranking_losses)
            / max(1, len(recent_ranking_losses)),
            "recent_schema_nll": sum(recent_schema_losses)
            / max(1, len(recent_schema_losses)),
            "recent_selection_loss": sum(recent_selection_losses)
            / max(1, len(recent_selection_losses)),
            "teacher_forced": teacher_forced,
            "generation": generation,
            "schema_teacher_forced": schema_teacher_forced,
            "schema_generation": schema_generation,
            "selection_retention": selection_retention,
            "full_vocabulary_selection": full_vocabulary_selection,
            "same_stream_generation": same_stream_generation,
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
