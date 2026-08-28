from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import benchmark_metrics as benchmark_metrics_module
from . import model as model_module
from . import physical_tokens as physical_tokens_module
from .benchmark_metrics import aggregate_records
from .controlled_registry import ControlledRegistry, build_controlled_registries
from .evaluate_fixed_tokens import (
    argument_prompt,
    generate_arguments,
    load_common_document_agent_audit,
)
from .episodic_data import PreparedTool, load_prepared_tools, load_token_pools
from .model import PhysicalOutputGenerator, TokenResamplerMemory
from .physical_tokens import (
    SplitReservedTokenPool,
    validate_token_identity_audit,
)
from .train_meta_readback import (
    DEFAULT_DOCUMENT_INSTRUCTION,
    MetaReadbackModel,
    _decoder,
    active_registry_full_logits,
    generate_same_stream_one,
    tokenize_documents,
)
from .train_meta_registration import (
    _disable_incompatible_optional_torchao,
    _last_state,
    _provide_optional_tensor_parallel_compat,
    render_query,
)


@dataclass(frozen=True)
class RegisteredBank:
    identities: tuple[str, ...]
    output_rows: torch.Tensor
    memory: torch.Tensor
    wall_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class NearestReferenceControl:
    bank: RegisteredBank
    reference_identities: tuple[str, ...]
    nearest_reference_indices: tuple[int, ...]
    cosine_similarities: tuple[float, ...]


@torch.inference_mode()
def apply_nearest_reference_control(
    bank: RegisteredBank,
    reference_bank: RegisteredBank,
    *,
    query_chunk_size: int = 128,
    reference_chunk_size: int = 4096,
) -> NearestReferenceControl:
    """Replace each late-bound bundle with its nearest registered train bundle."""
    if not reference_bank.identities:
        raise ValueError("Nearest-reference control requires a nonempty reference bank")
    if query_chunk_size < 1 or reference_chunk_size < 1:
        raise ValueError("Nearest-reference chunk sizes must be positive")
    reference_rows = reference_bank.output_rows.float()
    reference_norms = reference_rows.norm(dim=-1).clamp_min(1e-12)
    nearest_indices: list[int] = []
    nearest_similarities: list[float] = []
    for query_start in range(0, len(bank.identities), query_chunk_size):
        query_rows = bank.output_rows[
            query_start : query_start + query_chunk_size
        ].float()
        query_rows = query_rows / query_rows.norm(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        best_scores = torch.full(
            (len(query_rows),),
            float("-inf"),
            dtype=torch.float32,
            device=query_rows.device,
        )
        best_indices = torch.zeros(
            len(query_rows), dtype=torch.long, device=query_rows.device
        )
        for reference_start in range(
            0, len(reference_bank.identities), reference_chunk_size
        ):
            reference_end = min(
                reference_start + reference_chunk_size,
                len(reference_bank.identities),
            )
            normalized_reference = reference_rows[
                reference_start:reference_end
            ] / reference_norms[reference_start:reference_end, None]
            scores = query_rows @ normalized_reference.T
            chunk_scores, chunk_positions = scores.max(dim=1)
            improved = chunk_scores > best_scores
            best_scores = torch.where(improved, chunk_scores, best_scores)
            best_indices = torch.where(
                improved, chunk_positions + reference_start, best_indices
            )
        nearest_indices.extend(int(value) for value in best_indices.cpu().tolist())
        nearest_similarities.extend(
            float(value) for value in best_scores.cpu().tolist()
        )
    indices = torch.tensor(
        nearest_indices, dtype=torch.long, device=bank.output_rows.device
    )
    controlled = RegisteredBank(
        identities=bank.identities,
        output_rows=reference_bank.output_rows.index_select(0, indices),
        memory=reference_bank.memory.index_select(0, indices),
        wall_seconds=bank.wall_seconds,
        peak_memory_bytes=max(bank.peak_memory_bytes, reference_bank.peak_memory_bytes),
    )
    return NearestReferenceControl(
        bank=controlled,
        reference_identities=reference_bank.identities,
        nearest_reference_indices=tuple(nearest_indices),
        cosine_similarities=tuple(nearest_similarities),
    )


def apply_registration_control(
    bank: RegisteredBank,
    control: str,
    *,
    seed: int,
) -> RegisteredBank:
    """Apply a zero-update control while retaining the registration workload."""
    if control == "registered":
        return bank
    output = bank.output_rows
    memory = bank.memory
    if control in {"blank", "query_only"}:
        controlled_output = torch.zeros_like(output)
        controlled_memory = torch.zeros_like(memory)
    elif control == "shared_vector":
        controlled_output = output.mean(dim=0, keepdim=True).expand_as(output)
        controlled_memory = memory.mean(dim=0, keepdim=True).expand_as(memory)
    elif control in {"permuted", "wrong_memory"}:
        if len(bank.identities) < 2:
            raise ValueError(f"Control {control} requires at least two tools")
        shift = 1 + seed % (len(bank.identities) - 1)
        indices = torch.roll(
            torch.arange(len(bank.identities), device=output.device), shifts=shift
        )
        controlled_output = (
            output.index_select(0, indices) if control == "permuted" else output
        )
        controlled_memory = memory.index_select(0, indices)
    elif control == "random":
        generator = torch.Generator(device=output.device)
        generator.manual_seed(seed)
        controlled_output = torch.randn(
            output.shape,
            dtype=torch.float32,
            device=output.device,
            generator=generator,
        )
        controlled_memory = torch.randn(
            memory.shape,
            dtype=torch.float32,
            device=memory.device,
            generator=generator,
        )
        output_norm = output.float().norm(dim=-1).mean()
        memory_norm = memory.float().flatten(1).norm(dim=-1).mean()
        controlled_output = controlled_output / controlled_output.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        controlled_memory = controlled_memory / controlled_memory.flatten(1).norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12).view(-1, 1, 1)
        controlled_output = controlled_output.mul(output_norm).to(output.dtype)
        controlled_memory = controlled_memory.mul(memory_norm).to(memory.dtype)
    else:
        raise ValueError(f"Unsupported registration control: {control}")
    return RegisteredBank(
        identities=bank.identities,
        output_rows=controlled_output,
        memory=controlled_memory,
        wall_seconds=bank.wall_seconds,
        peak_memory_bytes=bank.peak_memory_bytes,
    )


def load_training_provenance(path: str | Path) -> dict[str, Any]:
    results_path = Path(path)
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training results must contain a JSON object")
    audit = payload.get("physical_token_audit")
    if not isinstance(audit, dict):
        raise ValueError("Training results are missing physical_token_audit")
    required_zero = (
        "evaluation_ids_seen_during_training",
        "evaluation_input_row_max_change",
        "evaluation_output_row_max_change",
    )
    missing = [key for key in required_zero if key not in audit]
    if missing:
        raise ValueError(f"Training physical-token audit is incomplete: {missing}")
    failed = {key: audit[key] for key in required_zero if float(audit[key]) != 0.0}
    if failed:
        raise ValueError(f"Training physical-token isolation failed: {failed}")
    identity_audit = validate_token_identity_audit(audit)
    if not identity_audit["passed"]:
        raise ValueError(
            f"Training physical-token identity audit failed: {identity_audit}"
        )
    return {
        "results_path": str(results_path.resolve()),
        "results_sha256": _sha256_file(results_path),
        "completed_steps": int(payload.get("completed_steps", 0)),
        "world_size": int(payload.get("world_size", 0)),
        "config": payload.get("config", {}),
        "physical_token_audit": audit,
        "physical_token_identity": identity_audit,
        "same_stream_generation": payload.get("same_stream_generation", {}),
    }


def _parameter_versions(model: torch.nn.Module) -> dict[str, int]:
    return {name: int(parameter._version) for name, parameter in model.named_parameters()}


def _changed_parameter_versions(
    before: Mapping[str, int], model: torch.nn.Module
) -> list[str]:
    after = _parameter_versions(model)
    if set(before) != set(after):
        raise RuntimeError("Model parameter set changed during evaluation")
    return sorted(name for name, version in before.items() if after[name] != version)


def _binding_fields(
    registry: ControlledRegistry,
    physical_ids: torch.Tensor,
    *,
    include_registry_arrays: bool = True,
    binding_sha256: str | None = None,
) -> dict[str, Any]:
    ids = [int(value) for value in physical_ids.detach().cpu().tolist()]
    if binding_sha256 is None:
        payload = "\n".join(
            f"{identity}\t{slot}\t{token_id}"
            for identity, slot, token_id in zip(
                registry.identities, registry.address_slots, ids
            )
        ).encode("utf-8")
        binding_sha256 = hashlib.sha256(payload).hexdigest()
    fields = {
        "reference_physical_token_ids": [
            ids[position] for position in registry.positive_positions
        ],
        "registry_physical_binding_sha256": binding_sha256,
    }
    if include_registry_arrays:
        fields.update(
            {
                "registry_address_slots": list(registry.address_slots),
                "registry_physical_token_ids": ids,
            }
        )
    return fields


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def latent_condition(source_condition: str, address_status: str) -> str:
    if source_condition == "seen_tool_seen_token":
        tool_status = "seen"
    elif source_condition == "unseen_tool_unseen_token":
        tool_status = "unseen"
    else:
        raise ValueError(f"Unsupported source condition: {source_condition}")
    if address_status not in {"seen", "unseen"}:
        raise ValueError(f"Unsupported address status: {address_status}")
    return f"{tool_status}_tool_{address_status}_address"


def address_pool_for(
    pools: Mapping[str, range], source_condition: str, address_status: str
) -> range:
    if address_status == "seen":
        return pools["train"]
    if source_condition == "seen_tool_seen_token":
        return pools["validation"]
    if source_condition == "unseen_tool_unseen_token":
        return pools["test"]
    raise ValueError(f"Unsupported source condition: {source_condition}")


def load_evaluation_address_pools(
    prepared_manifest_path: str | Path,
    evaluation_manifest_path: str | Path | None,
) -> tuple[dict[str, range], Path]:
    training_manifest = Path(prepared_manifest_path)
    training_pools = load_token_pools(training_manifest)
    selected_manifest = (
        Path(evaluation_manifest_path)
        if evaluation_manifest_path is not None
        else training_manifest
    )
    selected_pools = load_token_pools(selected_manifest)
    if selected_pools["train"] != training_pools["train"]:
        raise ValueError(
            "Evaluation address manifest changed the training address pool"
        )
    return selected_pools, selected_manifest


def candidate_splits_for(
    requested: Sequence[str], source_condition: str
) -> tuple[str, ...]:
    if not requested:
        return (
            ("train",)
            if source_condition == "seen_tool_seen_token"
            else ("test",)
        )
    values = set(requested)
    if "all" in values:
        if len(values) != 1:
            raise ValueError("Candidate split 'all' cannot be combined with other splits")
        return ("train", "validation", "test")
    return tuple(sorted(values))


def load_shared_registry_binding(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    candidate_identities: Sequence[str],
    address_pool: range,
) -> list[ControlledRegistry]:
    binding_path = Path(path)
    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Registry binding must contain a JSON object")
    identities = tuple(str(value) for value in payload["identities"])
    address_slots = tuple(int(value) for value in payload["address_slots"])
    if not identities or len(identities) != len(address_slots):
        raise ValueError("Registry binding identities and addresses must be nonempty")
    if len(set(identities)) != len(identities):
        raise ValueError("Registry binding reused an identity")
    if len(set(address_slots)) != len(address_slots):
        raise ValueError("Registry binding reused an address")
    candidate_set = set(candidate_identities)
    missing_candidates = sorted(set(identities) - candidate_set)
    if missing_candidates:
        raise ValueError(
            f"Registry binding identities are absent from candidates: {missing_candidates[:3]}"
        )
    if not set(address_slots).issubset(set(address_pool)):
        raise ValueError("Registry binding uses an address outside the selected pool")
    identity_sha256 = hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
    if str(payload.get("identity_sha256")) != identity_sha256:
        raise ValueError("Registry binding identity hash does not match its contents")
    positions = {identity: index for index, identity in enumerate(identities)}
    registries: list[ControlledRegistry] = []
    for row in rows:
        targets = tuple(dict.fromkeys(str(value) for value in row["reference_tools"]))
        missing_targets = [identity for identity in targets if identity not in positions]
        if missing_targets:
            raise ValueError(
                f"Registry binding omits evaluation targets: {missing_targets[:3]}"
            )
        registries.append(
            ControlledRegistry(
                identities=identities,
                positive_positions=tuple(positions[identity] for identity in targets),
                address_slots=address_slots,
                identity_sha256=identity_sha256,
            )
        )
    return registries


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_condition: str,
    families: set[str],
    max_examples_per_family: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts = {family: 0 for family in families}
    for row in sorted(
        rows, key=lambda item: (str(item["family"]), str(item["example_id"]))
    ):
        family = str(row["family"])
        if str(row["condition"]) != source_condition or family not in families:
            continue
        if max_examples_per_family and counts[family] >= max_examples_per_family:
            continue
        selected.append(dict(row))
        counts[family] += 1
    return selected


def _build_registries(
    rows: Sequence[Mapping[str, Any]],
    candidate_identities: Sequence[str],
    *,
    registry_size: int,
    registry_seed: int,
    address_pool: range,
    registry_scope: str = "per_example",
) -> list[ControlledRegistry]:
    return build_controlled_registries(
        candidate_identities,
        [[str(value) for value in row["reference_tools"]] for row in rows],
        registry_size=registry_size,
        seed=registry_seed,
        keys=[
            f"{row['family']}:{row['condition']}:{row['example_id']}" for row in rows
        ],
        address_pool=address_pool,
        scope=registry_scope,
    )


@torch.inference_mode()
def register_tool_bank(
    model: MetaReadbackModel,
    tokenizer,
    tools: Mapping[str, PreparedTool],
    identities: Sequence[str],
    *,
    batch_size: int,
    max_document_length: int,
    document_instruction: str,
    device: torch.device,
) -> RegisteredBank:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output_chunks: list[torch.Tensor] = []
    memory_chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(identities), batch_size):
        batch_identities = identities[start : start + batch_size]
        document_tokens = tokenize_documents(
            tokenizer,
            [tools[identity] for identity in batch_identities],
            max_length=max_document_length,
            device=device,
            document_instruction=document_instruction,
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output_rows, memory = model.register_bundle(document_tokens)
        output_chunks.append(output_rows.detach())
        memory_chunks.append(memory.detach())
    wall_seconds = time.perf_counter() - started
    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return RegisteredBank(
        identities=tuple(identities),
        output_rows=torch.cat(output_chunks),
        memory=torch.cat(memory_chunks),
        wall_seconds=wall_seconds,
        peak_memory_bytes=peak,
    )


def _registry_tensors(
    registry: ControlledRegistry,
    bank_index: Mapping[str, int],
    bank: RegisteredBank,
    physical_pool: SplitReservedTokenPool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.tensor(
        [bank_index[identity] for identity in registry.identities],
        dtype=torch.long,
        device=device,
    )
    physical_ids = torch.tensor(
        [physical_pool.physical_id(slot) for slot in registry.address_slots],
        dtype=torch.long,
        device=device,
    )
    return (
        bank.output_rows.index_select(0, indices),
        bank.memory.index_select(0, indices),
        physical_ids,
    )


def _registry_output_tensors(
    registry: ControlledRegistry,
    bank_index: Mapping[str, int],
    bank: RegisteredBank,
    physical_pool: SplitReservedTokenPool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.tensor(
        [bank_index[identity] for identity in registry.identities],
        dtype=torch.long,
        device=device,
    )
    physical_ids = torch.tensor(
        [physical_pool.physical_id(slot) for slot in registry.address_slots],
        dtype=torch.long,
        device=device,
    )
    return bank.output_rows.index_select(0, indices), physical_ids


@torch.inference_mode()
def _retrieval_predictions(
    model: MetaReadbackModel,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    registries: Sequence[ControlledRegistry],
    bank: RegisteredBank,
    physical_pool: SplitReservedTokenPool,
    *,
    batch_size: int,
    max_query_length: int,
    device: torch.device,
    registry_scope: str = "per_example",
    registration_control: str = "registered",
) -> list[dict[str, Any]]:
    bank_index = {identity: index for index, identity in enumerate(bank.identities)}
    reserved_ids = torch.tensor(
        physical_pool.token_ids, dtype=torch.long, device=device
    )
    shared_output: torch.Tensor | None = None
    shared_physical: torch.Tensor | None = None
    shared_binding_sha256: str | None = None
    if registry_scope == "shared" and registries:
        shared_output, shared_physical = _registry_output_tensors(
            registries[0], bank_index, bank, physical_pool, device
        )
        shared_binding_sha256 = str(
            _binding_fields(
                registries[0], shared_physical, include_registry_arrays=False
            )["registry_physical_binding_sha256"]
        )
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_registries = registries[start : start + batch_size]
        prompts = [render_query(tokenizer, str(row["query"])) for row in batch_rows]
        query_tokens = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_query_length,
            return_tensors="pt",
        ).to(device)
        hidden = _decoder(model.backbone)(
            **query_tokens, use_cache=False, return_dict=True
        ).last_hidden_state
        query_states = _last_state(hidden, query_tokens["attention_mask"])
        base_logits = model.backbone.get_output_embeddings()(
            query_states.to(model.backbone.get_output_embeddings().weight.dtype)
        )
        if shared_output is not None and shared_physical is not None:
            stacked_output = shared_output.unsqueeze(0).expand(
                len(batch_registries), -1, -1
            )
            stacked_physical = shared_physical.unsqueeze(0).expand(
                len(batch_registries), -1
            )
        else:
            output_rows: list[torch.Tensor] = []
            physical_rows: list[torch.Tensor] = []
            for registry in batch_registries:
                registered_output, physical_ids = _registry_output_tensors(
                    registry, bank_index, bank, physical_pool, device
                )
                output_rows.append(registered_output)
                physical_rows.append(physical_ids)
            stacked_output = torch.stack(output_rows)
            stacked_physical = torch.stack(physical_rows)
        registry_logits = torch.einsum(
            "bh,brh->br", query_states.float(), stacked_output.float()
        )
        if registration_control == "query_only":
            full_logits = active_registry_full_logits(
                base_logits,
                registry_logits[:, :0],
                stacked_physical[:, :0],
                reserved_ids,
            )
        else:
            full_logits = active_registry_full_logits(
                base_logits,
                registry_logits,
                stacked_physical,
                reserved_ids,
            )
        registry_values, registry_positions = registry_logits.topk(
            min(5, registry_logits.shape[1]), dim=1
        )
        full_values, full_ids = full_logits.topk(5, dim=1)
        for index, registry in enumerate(batch_registries):
            constrained = [
                registry.identities[int(position)]
                for position in registry_positions[index].cpu().tolist()
            ]
            physical_to_identity = {
                int(token_id): identity
                for token_id, identity in zip(
                    stacked_physical[index].cpu().tolist(), registry.identities
                )
            }
            full_ranked = [
                physical_to_identity.get(
                    int(token_id), f"__ordinary_token_id_{int(token_id)}__"
                )
                for token_id in full_ids[index].cpu().tolist()
            ]
            target_positions = registry.positive_positions
            best_target_logit = max(
                float(registry_logits[index, position]) for position in target_positions
            )
            target_rank = (
                None
                if registration_control == "query_only"
                else 1
                + int(
                    (full_logits[index].float() > best_target_logit).sum().item()
                )
            )
            winner_id = int(full_ids[index, 0].item())
            selected_identity = physical_to_identity.get(winner_id)
            selected_binding_verified = (
                selected_identity is not None and full_ranked[0] == selected_identity
            )
            predictions.append(
                {
                    "predicted_tools": full_ranked,
                    "prediction_scores": [
                        float(value) for value in full_values[index].cpu().tolist()
                    ],
                    "registry_predicted_tools": constrained,
                    "registry_prediction_scores": [
                        float(value)
                        for value in registry_values[index].cpu().tolist()
                    ],
                    "selected_physical_token_id": (
                        winner_id if winner_id in physical_to_identity else None
                    ),
                    "selected_physical_binding_verified": (
                        selected_binding_verified if selected_identity is not None else None
                    ),
                    "ordinary_token_win": winner_id not in physical_to_identity,
                    "registry_active": registration_control != "query_only",
                    "target_rank_full_vocabulary": target_rank,
                    "registry_identity_sha256": registry.identity_sha256,
                    **_binding_fields(
                        registry,
                        stacked_physical[index],
                        include_registry_arrays=registry_scope != "shared",
                        binding_sha256=shared_binding_sha256,
                    ),
                }
            )
    return predictions


@torch.inference_mode()
def _argument_predictions(
    model: MetaReadbackModel,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    registries: Sequence[ControlledRegistry],
    bank: RegisteredBank,
    physical_pool: SplitReservedTokenPool,
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
    registry_scope: str = "per_example",
    registration_control: str = "registered",
) -> list[dict[str, Any]]:
    bank_index = {identity: index for index, identity in enumerate(bank.identities)}
    reserved_ids = torch.tensor(
        physical_pool.token_ids, dtype=torch.long, device=device
    )
    shared_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    shared_binding_sha256: str | None = None
    if registry_scope == "shared" and registries:
        shared_tensors = _registry_tensors(
            registries[0], bank_index, bank, physical_pool, device
        )
        shared_binding_sha256 = str(
            _binding_fields(
                registries[0],
                shared_tensors[2],
                include_registry_arrays=False,
            )["registry_physical_binding_sha256"]
        )
    predictions: list[dict[str, Any]] = []
    for row, registry in zip(rows, registries):
        if shared_tensors is None:
            output_rows, memory, physical_ids = _registry_tensors(
                registry, bank_index, bank, physical_pool, device
            )
        else:
            output_rows, memory, physical_ids = shared_tensors
        generation_output = output_rows
        generation_memory = memory
        generation_physical = physical_ids
        if registration_control == "query_only":
            generation_output = output_rows[:0]
            generation_memory = memory[:0]
            generation_physical = physical_ids[:0]
        generated = generate_same_stream_one(
            model,
            tokenizer,
            str(row["query"]),
            generation_output,
            generation_memory,
            generation_physical,
            reserved_ids,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        selected_identity = (
            registry.identities[generated.selected_registry_index]
            if generated.selected_registry_index is not None
            else f"__ordinary_token_id_{generated.selected_token_id}__"
        )
        selected_binding_verified = (
            generated.selected_registry_index is not None
            and generated.selected_token_id
            == int(physical_ids[generated.selected_registry_index].item())
        )
        predictions.append(
            {
                "predicted_tools": [selected_identity],
                "predicted_arguments": generated.text,
                "selected_physical_token_id": (
                    generated.selected_token_id
                    if generated.selected_registry_index is not None
                    else None
                ),
                "selected_physical_binding_verified": (
                    selected_binding_verified
                    if generated.selected_registry_index is not None
                    else None
                ),
                "selected_memory_dereference_verified": (
                    selected_binding_verified
                    if generated.selected_registry_index is not None
                    else None
                ),
                "ordinary_token_win": generated.selected_registry_index is None,
                "registry_active": registration_control != "query_only",
                "generated_token_ids": list(generated.generated_token_ids),
                "registry_identity_sha256": registry.identity_sha256,
                **_binding_fields(
                    registry,
                    physical_ids,
                    include_registry_arrays=registry_scope != "shared",
                    binding_sha256=shared_binding_sha256,
                ),
            }
        )
    return predictions


@torch.inference_mode()
def _common_document_argument_predictions(
    model: MetaReadbackModel,
    tokenizer,
    argument_model,
    argument_tokenizer,
    tools: Mapping[str, PreparedTool],
    rows: Sequence[Mapping[str, Any]],
    registries: Sequence[ControlledRegistry],
    bank: RegisteredBank,
    physical_pool: SplitReservedTokenPool,
    *,
    selection_batch_size: int,
    generation_batch_size: int,
    max_query_length: int,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
    registry_scope: str = "per_example",
    registration_control: str = "registered",
) -> list[dict[str, Any]]:
    """Select one registered address, then generate from its full document."""
    selections = _retrieval_predictions(
        model,
        tokenizer,
        rows,
        registries,
        bank,
        physical_pool,
        batch_size=selection_batch_size,
        max_query_length=max_query_length,
        device=device,
        registry_scope=registry_scope,
        registration_control=registration_control,
    )
    prompts: list[str] = []
    valid_positions: list[int] = []
    for index, (row, selection) in enumerate(zip(rows, selections)):
        identity = str(selection["predicted_tools"][0])
        tool = tools.get(identity)
        if tool is None:
            continue
        prompts.append(
            argument_prompt(
                str(row["query"]),
                selected_token=str(selection.get("selected_physical_token_id")),
                selected_document=tool.document,
                information_condition="common_document",
            )
        )
        valid_positions.append(index)
    generations = (
        generate_arguments(
            argument_model,
            argument_tokenizer,
            prompts,
            batch_size=generation_batch_size,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        if prompts
        else []
    )
    generated_by_position = dict(zip(valid_positions, generations))

    predictions: list[dict[str, Any]] = []
    for index, selection in enumerate(selections):
        identity = str(selection["predicted_tools"][0])
        tool = tools.get(identity)
        document_verified = (
            selection.get("selected_physical_token_id") is not None
            and selection.get("selected_physical_binding_verified") is True
            and tool is not None
        )
        predictions.append(
            {
                **selection,
                "predicted_tools": [identity],
                "predicted_arguments": generated_by_position.get(index, "").strip(),
                "selected_document_identity": identity if tool is not None else None,
                "selected_document_sha256": (
                    hashlib.sha256(tool.document.encode("utf-8")).hexdigest()
                    if tool is not None
                    else None
                ),
                "selected_document_dereference_verified": (
                    document_verified
                    if selection.get("selected_physical_token_id") is not None
                    else None
                ),
            }
        )
    return predictions


def _selected_payload_audit(
    retrieval_values: Sequence[Mapping[str, Any]],
    argument_values: Sequence[Mapping[str, Any]],
    *,
    information_condition: str,
) -> dict[str, Any]:
    selected = [
        value
        for value in (*retrieval_values, *argument_values)
        if value.get("selected_physical_token_id") is not None
    ]
    binding_verified = sum(
        value.get("selected_physical_binding_verified") is True for value in selected
    )
    argument_selected = [
        value
        for value in argument_values
        if value.get("selected_physical_token_id") is not None
    ]
    if information_condition == "common_document":
        payload_verified = sum(
            value.get("selected_document_dereference_verified") is True
            and value.get("selected_document_identity")
            == str(value.get("predicted_tools", [""])[0])
            for value in argument_selected
        )
    elif information_condition == "registered_memory":
        payload_verified = sum(
            value.get("selected_memory_dereference_verified") is True
            for value in argument_selected
        )
    else:
        raise ValueError(f"Unsupported information condition: {information_condition}")
    return {
        "selected_physical_binding_count": len(selected),
        "selected_physical_binding_verified_count": binding_verified,
        "selected_physical_bindings_verified": (
            bool(selected) and binding_verified == len(selected)
        ),
        "selected_argument_payload_count": len(argument_selected),
        "selected_argument_payload_verified_count": payload_verified,
        "selected_argument_payloads_verified": (
            bool(argument_selected) and payload_verified == len(argument_selected)
        ),
    }


def _group_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['family']}:{row['condition']}", []).append(row)
    return {key: aggregate_records(value) for key, value in sorted(grouped.items())}


def evaluate_late_bound_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    benchmark = Path(args.benchmark_dir)
    prepared = Path(args.prepared_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    common_document_agent: dict[str, Any] | None = None
    common_document_agent_path: Path | None = None
    if args.information_condition == "common_document":
        if not args.common_document_agent_model_path:
            raise ValueError(
                "common_document requires --common-document-agent-model-path"
            )
        if not args.common_document_agent_audit_path:
            raise ValueError(
                "common_document requires --common-document-agent-audit-path"
            )
        common_document_agent_path = Path(args.common_document_agent_model_path)
        common_document_agent = load_common_document_agent_audit(
            common_document_agent_path, args.common_document_agent_audit_path
        )
    if args.nearest_reference_limit < 0:
        raise ValueError("Nearest-reference limit cannot be negative")
    training_provenance = load_training_provenance(args.training_results_path)
    tools = load_prepared_tools(prepared / "tools.jsonl")
    logical_pools, address_manifest_path = load_evaluation_address_pools(
        prepared / "split_manifest.json", args.address_manifest_path
    )
    evaluation_rows_path = (
        Path(args.evaluation_rows_path)
        if args.evaluation_rows_path
        else benchmark / "benchmark_eval.jsonl"
    )
    rows = _select_rows(
        _read_jsonl(evaluation_rows_path),
        source_condition=args.source_condition,
        families=set(args.family),
        max_examples_per_family=args.max_examples_per_family,
    )
    if not rows:
        raise ValueError("No benchmark rows matched the requested latent condition")
    candidate_splits = candidate_splits_for(
        args.candidate_split, args.source_condition
    )
    candidate_identities = sorted(
        identity for identity, tool in tools.items() if tool.split in candidate_splits
    )
    address_pool = address_pool_for(
        logical_pools, args.source_condition, args.address_status
    )
    if args.registry_binding_path:
        if args.registry_scope != "shared":
            raise ValueError("A precomputed registry binding requires shared scope")
        registries = load_shared_registry_binding(
            args.registry_binding_path, rows, candidate_identities, address_pool
        )
        if len(registries[0].identities) != args.registry_size:
            raise ValueError(
                "Requested registry size does not match precomputed binding"
            )
    else:
        registries = _build_registries(
            rows,
            candidate_identities,
            registry_size=args.registry_size,
            registry_seed=args.registry_seed,
            address_pool=address_pool,
            registry_scope=args.registry_scope,
        )
    bank_identities = sorted(
        {identity for registry in registries for identity in registry.identities}
    )

    device = torch.device(args.device)
    _disable_incompatible_optional_torchao()
    _provide_optional_tensor_parallel_compat()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    physical_pool = SplitReservedTokenPool.create(tokenizer, base, logical_pools)
    input_rows = base.get_input_embeddings().weight[: physical_pool.original_vocab_size]
    output_rows = base.get_output_embeddings().weight[: physical_pool.original_vocab_size]
    input_norm = float(input_rows.detach().float().norm(dim=-1).mean())
    output_norm = float(output_rows.detach().float().norm(dim=-1).mean())
    backbone = PeftModel.from_pretrained(
        base, args.retrieval_adapter_path, is_trainable=False
    ).to(device)
    memory_compiler = TokenResamplerMemory(
        backbone.config.hidden_size,
        args.compiler_rank,
        args.memory_slots,
        input_norm,
    ).to(device)
    memory_compiler.load_state_dict(
        torch.load(args.memory_compiler_path, map_location=device, weights_only=True)
    )
    output_compiler = PhysicalOutputGenerator(
        backbone.config.hidden_size, args.compiler_rank, output_norm
    ).to(device)
    output_compiler.load_state_dict(
        torch.load(args.retrieval_compiler_path, map_location=device, weights_only=True)
    )
    model = MetaReadbackModel(backbone, memory_compiler, output_compiler).to(device)
    model.eval()

    parameter_versions_before = _parameter_versions(model)
    reserved_input_before = (
        model.backbone.get_input_embeddings()
        .weight[physical_pool.token_ids]
        .detach()
        .float()
        .cpu()
        .clone()
    )
    reserved_output_before = (
        model.backbone.get_output_embeddings()
        .weight[physical_pool.token_ids]
        .detach()
        .float()
        .cpu()
        .clone()
    )

    registered_bank = register_tool_bank(
        model,
        tokenizer,
        tools,
        bank_identities,
        batch_size=args.registration_batch_size,
        max_document_length=args.max_document_length,
        document_instruction=args.document_instruction,
        device=device,
    )
    reference_bank: RegisteredBank | None = None
    nearest_control: NearestReferenceControl | None = None
    if args.registration_control == "nearest_trained":
        reference_identities = sorted(
            identity for identity, tool in tools.items() if tool.split == "train"
        )
        if args.nearest_reference_limit:
            reference_identities = reference_identities[
                : args.nearest_reference_limit
            ]
        reference_bank = register_tool_bank(
            model,
            tokenizer,
            tools,
            reference_identities,
            batch_size=args.registration_batch_size,
            max_document_length=args.max_document_length,
            document_instruction=args.document_instruction,
            device=device,
        )
        nearest_control = apply_nearest_reference_control(
            registered_bank,
            reference_bank,
            query_chunk_size=args.nearest_query_chunk_size,
            reference_chunk_size=args.nearest_reference_chunk_size,
        )
        bank = nearest_control.bank
    else:
        bank = apply_registration_control(
            registered_bank, args.registration_control, seed=args.control_seed
        )
    control_setup_peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    retrieval_rows = [row for row in rows if row["family"] == "retrieval"]
    argument_rows = [row for row in rows if row["family"] == "arguments"]
    retrieval_registries = [
        registry for row, registry in zip(rows, registries) if row["family"] == "retrieval"
    ]
    argument_registries = [
        registry for row, registry in zip(rows, registries) if row["family"] == "arguments"
    ]
    condition = latent_condition(args.source_condition, args.address_status)
    argument_model = None
    argument_tokenizer = None
    if args.information_condition == "common_document":
        argument_tokenizer = AutoTokenizer.from_pretrained(
            common_document_agent_path, local_files_only=True
        )
        if argument_tokenizer.pad_token_id is None:
            argument_tokenizer.pad_token = argument_tokenizer.eos_token
        argument_tokenizer.padding_side = "left"
        argument_model = AutoModelForCausalLM.from_pretrained(
            common_document_agent_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        argument_model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    retrieval_started = time.perf_counter()
    retrieval_values = _retrieval_predictions(
        model,
        tokenizer,
        retrieval_rows,
        retrieval_registries,
        bank,
        physical_pool,
        batch_size=args.query_batch_size,
        max_query_length=args.max_query_length,
        device=device,
        registry_scope=args.registry_scope,
        registration_control=args.registration_control,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    retrieval_seconds = time.perf_counter() - retrieval_started
    argument_started = time.perf_counter()
    if args.information_condition == "common_document":
        assert argument_model is not None and argument_tokenizer is not None
        argument_values = _common_document_argument_predictions(
            model,
            tokenizer,
            argument_model,
            argument_tokenizer,
            tools,
            argument_rows,
            argument_registries,
            bank,
            physical_pool,
            selection_batch_size=args.query_batch_size,
            generation_batch_size=args.generation_batch_size,
            max_query_length=args.max_query_length,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
            registry_scope=args.registry_scope,
            registration_control=args.registration_control,
        )
    else:
        argument_values = _argument_predictions(
            model,
            tokenizer,
            argument_rows,
            argument_registries,
            bank,
            physical_pool,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
            registry_scope=args.registry_scope,
            registration_control=args.registration_control,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    argument_seconds = time.perf_counter() - argument_started
    inference_seconds = retrieval_seconds + argument_seconds
    selected_payload_audit = _selected_payload_audit(
        retrieval_values,
        argument_values,
        information_condition=args.information_condition,
    )
    predictions: list[dict[str, Any]] = []
    for row, value, registry in zip(
        retrieval_rows, retrieval_values, retrieval_registries
    ):
        predictions.append(
            {
                **row,
                **value,
                "source_condition": row["condition"],
                "condition": condition,
                "information_condition": "selection",
                "registry_size": len(registry.identities),
            }
        )
    for row, value, registry in zip(
        argument_rows, argument_values, argument_registries
    ):
        predictions.append(
            {
                **row,
                **value,
                "source_condition": row["condition"],
                "condition": condition,
                "information_condition": args.information_condition,
                "common_document_agent_model_path": (
                    common_document_agent["model_path"]
                    if common_document_agent is not None
                    else None
                ),
                "common_document_agent_audit_sha256": (
                    common_document_agent["audit_sha256"]
                    if common_document_agent is not None
                    else None
                ),
                "registry_size": len(registry.identities),
            }
        )
    predictions.sort(
        key=lambda row: (str(row["family"]), str(row["condition"]), str(row["example_id"]))
    )
    predictions_path = output / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    binding_catalog_path: Path | None = None
    binding_catalog_sha256: str | None = None
    if args.registry_scope == "shared":
        registry = registries[0]
        physical_ids = torch.tensor(
            [physical_pool.physical_id(slot) for slot in registry.address_slots],
            dtype=torch.long,
        )
        binding_catalog_path = output / "registry_binding.json"
        binding_catalog_path.write_text(
            json.dumps(
                {
                    "registry_identity_sha256": registry.identity_sha256,
                    "registry_identities": list(registry.identities),
                    **_binding_fields(registry, physical_ids),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        binding_catalog_sha256 = _sha256_file(binding_catalog_path)

    nearest_mapping_path: Path | None = None
    nearest_mapping_sha256: str | None = None
    nearest_similarity_mean: float | None = None
    if nearest_control is not None:
        nearest_mapping_path = output / "nearest_trained_mapping.jsonl"
        with nearest_mapping_path.open("w", encoding="utf-8") as handle:
            for identity, reference_index, similarity in zip(
                bank.identities,
                nearest_control.nearest_reference_indices,
                nearest_control.cosine_similarities,
            ):
                handle.write(
                    json.dumps(
                        {
                            "identity": identity,
                            "nearest_train_identity": nearest_control.reference_identities[
                                reference_index
                            ],
                            "cosine_similarity": similarity,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        nearest_mapping_sha256 = _sha256_file(nearest_mapping_path)
        nearest_similarity_mean = sum(nearest_control.cosine_similarities) / len(
            nearest_control.cosine_similarities
        )

    constrained_rows = []
    for row in predictions:
        if row["family"] != "retrieval":
            continue
        constrained_rows.append(
            {**row, "predicted_tools": row["registry_predicted_tools"]}
        )
    bytes_per_tool = (
        bank.output_rows[0].numel() * bank.output_rows.element_size()
        + bank.memory[0].numel() * bank.memory.element_size()
    )
    evaluation_mutation_audit = physical_pool.audit(model.backbone, set())
    evaluation_mutation_audit.pop("evaluation_ids_seen_during_training", None)
    reserved_input_after = (
        model.backbone.get_input_embeddings()
        .weight[physical_pool.token_ids]
        .detach()
        .float()
        .cpu()
    )
    reserved_output_after = (
        model.backbone.get_output_embeddings()
        .weight[physical_pool.token_ids]
        .detach()
        .float()
        .cpu()
    )
    evaluation_mutation_audit["all_reserved_input_row_max_change"] = float(
        (reserved_input_after - reserved_input_before).abs().max()
    )
    evaluation_mutation_audit["all_reserved_output_row_max_change"] = float(
        (reserved_output_after - reserved_output_before).abs().max()
    )
    changed_parameters = _changed_parameter_versions(parameter_versions_before, model)
    if changed_parameters:
        raise AssertionError(
            f"Model parameters changed during registration/evaluation: {changed_parameters[:5]}"
        )
    if evaluation_mutation_audit["all_reserved_input_row_max_change"] != 0.0:
        raise AssertionError("Reserved input embedding rows changed during evaluation")
    if evaluation_mutation_audit["all_reserved_output_row_max_change"] != 0.0:
        raise AssertionError("Reserved output vocabulary rows changed during evaluation")
    adapter_path = Path(args.retrieval_adapter_path)
    adapter_weights = adapter_path / "adapter_model.safetensors"
    result = {
        "kind": "qwen_late_bound",
        "code_hashes": {
            "evaluate_late_bound": _sha256_file(Path(__file__)),
            "benchmark_metrics": _sha256_file(
                Path(str(benchmark_metrics_module.__file__))
            ),
            "model": _sha256_file(Path(str(model_module.__file__))),
            "physical_tokens": _sha256_file(
                Path(str(physical_tokens_module.__file__))
            ),
        },
        "condition": condition,
        "source_condition": args.source_condition,
        "address_status": args.address_status,
        "registry_size": len(registries[0].identities),
        "requested_registry_size": args.registry_size,
        "registry_seed": args.registry_seed,
        "registration_control": args.registration_control,
        "information_condition": args.information_condition,
        "common_document_agent": common_document_agent,
        "common_document_agent_config_sha256": (
            _sha256_file(common_document_agent_path / "config.json")
            if common_document_agent_path is not None
            else None
        ),
        "control_seed": args.control_seed,
        "registry_scope": args.registry_scope,
        "candidate_splits": list(candidate_splits),
        "candidate_count": len(candidate_identities),
        "prediction_count": len(predictions),
        "evaluation_rows_path": str(evaluation_rows_path.resolve()),
        "evaluation_rows_sha256": _sha256_file(evaluation_rows_path),
        "input_registry_binding": (
            str(Path(args.registry_binding_path).resolve())
            if args.registry_binding_path
            else None
        ),
        "input_registry_binding_sha256": (
            _sha256_file(Path(args.registry_binding_path))
            if args.registry_binding_path
            else None
        ),
        "benchmark_manifest_sha256": _sha256_file(
            benchmark / "benchmark_manifest.json"
        ),
        "evaluation_address_manifest": str(address_manifest_path.resolve()),
        "evaluation_address_manifest_sha256": _sha256_file(address_manifest_path),
        "registry_binding_catalog": (
            str(binding_catalog_path.resolve()) if binding_catalog_path else None
        ),
        "registry_binding_catalog_sha256": binding_catalog_sha256,
        "nearest_trained_mapping": (
            str(nearest_mapping_path.resolve()) if nearest_mapping_path else None
        ),
        "nearest_trained_mapping_sha256": nearest_mapping_sha256,
        "nearest_trained_mean_cosine_similarity": nearest_similarity_mean,
        "checkpoint_hashes": {
            "retrieval_adapter_config": _sha256_file(
                adapter_path / "adapter_config.json"
            ),
            "retrieval_adapter_weights": _sha256_file(adapter_weights),
            "retrieval_compiler": _sha256_file(Path(args.retrieval_compiler_path)),
            "memory_compiler": _sha256_file(Path(args.memory_compiler_path)),
            "training_results": training_provenance["results_sha256"],
        },
        "training_provenance": training_provenance,
        "registration_audit": {
            "document_encoder_batch_forwards": (
                len(registered_bank.identities) + args.registration_batch_size - 1
            )
            // args.registration_batch_size,
            "registration_forwards_per_tool": 1,
            "distinct_registered_tools": len(registered_bank.identities),
            "wall_seconds": registered_bank.wall_seconds,
            "peak_memory_bytes": max(
                registered_bank.peak_memory_bytes,
                reference_bank.peak_memory_bytes if reference_bank else 0,
                control_setup_peak_memory_bytes,
            ),
            "reference_document_encoder_batch_forwards": (
                (len(reference_bank.identities) + args.registration_batch_size - 1)
                // args.registration_batch_size
                if reference_bank
                else 0
            ),
            "reference_registered_tools": (
                len(reference_bank.identities) if reference_bank else 0
            ),
            "reference_registration_wall_seconds": (
                reference_bank.wall_seconds if reference_bank else 0.0
            ),
            "optimizer_steps": 0,
            "model_or_table_parameters_changed": 0,
            "backbone_parameters_changed": 0,
            "embedding_table_parameters_changed": 0,
            "lm_head_parameters_changed": 0,
            "parameter_version_changes": 0,
            "bytes_stored_per_tool": bytes_per_tool,
            "reference_bytes_stored_total": (
                len(reference_bank.identities) * bytes_per_tool
                if reference_bank
                else 0
            ),
            "shared_document_forward": True,
            "selection_emits_single_physical_id": True,
            **selected_payload_audit,
            "selected_id_dereferences_registered_memory": (
                args.information_condition == "registered_memory"
                and selected_payload_audit["selected_argument_payloads_verified"]
            ),
            "selected_id_dereferences_full_document": (
                args.information_condition == "common_document"
                and selected_payload_audit["selected_argument_payloads_verified"]
            ),
            "selected_id_uses_static_embedding": False,
            "ordinary_vocabulary_competes_at_selection": True,
            "inactive_reserved_ids_masked": True,
        },
        "physical_token_audit": training_provenance["physical_token_audit"],
        "evaluation_mutation_audit": evaluation_mutation_audit,
        "inference_seconds": inference_seconds,
        "latency": {
            "model_loading_excluded": True,
            "retrieval_selection": {
                "count": len(retrieval_rows),
                "total_seconds": retrieval_seconds,
                "mean_seconds_per_example": (
                    retrieval_seconds / len(retrieval_rows)
                    if retrieval_rows
                    else 0.0
                ),
            },
            "argument_path": {
                "count": len(argument_rows),
                "total_seconds": argument_seconds,
                "mean_seconds_per_example": (
                    argument_seconds / len(argument_rows) if argument_rows else 0.0
                ),
                "includes_tool_selection": True,
                "includes_document_dereference_and_generation": True,
            },
        },
        "aggregates": _group_aggregates(predictions),
        "registry_constrained_retrieval": aggregate_records(constrained_rows),
        "ordinary_token_win_rate": (
            sum(bool(row.get("ordinary_token_win")) for row in predictions)
            / len(predictions)
        ),
        "ordinary_token_win_rate_by_family": {
            family: (
                sum(
                    bool(row.get("ordinary_token_win"))
                    for row in predictions
                    if row["family"] == family
                )
                / sum(row["family"] == family for row in predictions)
            )
            for family in sorted({str(row["family"]) for row in predictions})
        },
        "predictions_sha256": _sha256_file(predictions_path),
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one-forward late-bound registration on the controlled manifest"
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--common-document-agent-model-path",
        help=(
            "Shared downstream document-conditioned Agent required by the "
            "common_document information condition."
        ),
    )
    parser.add_argument(
        "--common-document-agent-audit-path",
        help="Checkpoint audit belonging to the shared common-document Agent.",
    )
    parser.add_argument("--retrieval-adapter-path", required=True)
    parser.add_argument("--retrieval-compiler-path", required=True)
    parser.add_argument("--memory-compiler-path", required=True)
    parser.add_argument("--training-results-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--evaluation-rows-path",
        help="Optional controlled subset JSONL; defaults to benchmark_eval.jsonl.",
    )
    parser.add_argument(
        "--registry-binding-path",
        help="Optional immutable shared identity/logical-address mapping.",
    )
    parser.add_argument(
        "--address-manifest-path",
        help=(
            "Optional evaluation-only expanded address manifest; its training "
            "pool must exactly match prepared-dir/split_manifest.json."
        ),
    )
    parser.add_argument(
        "--source-condition",
        choices=("seen_tool_seen_token", "unseen_tool_unseen_token"),
        required=True,
    )
    parser.add_argument("--address-status", choices=("seen", "unseen"), required=True)
    parser.add_argument(
        "--candidate-split",
        action="append",
        default=[],
        choices=("train", "validation", "test", "all"),
        help="Candidate tool split; defaults to train for seen tools and test for unseen tools.",
    )
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--registry-size", type=int, required=True)
    parser.add_argument("--registry-seed", type=int, default=17)
    parser.add_argument(
        "--registration-control",
        choices=(
            "registered",
            "blank",
            "random",
            "wrong_memory",
            "permuted",
            "shared_vector",
            "query_only",
            "nearest_trained",
        ),
        default="registered",
    )
    parser.add_argument("--control-seed", type=int, default=17)
    parser.add_argument(
        "--nearest-reference-limit",
        type=int,
        default=0,
        help="Use zero for the complete train split; nonzero is smoke-only.",
    )
    parser.add_argument("--nearest-query-chunk-size", type=int, default=128)
    parser.add_argument("--nearest-reference-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--registry-scope",
        choices=("per_example", "shared"),
        default="per_example",
        help="Use one candidate/address mapping across all rows for large registries.",
    )
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--compiler-rank", type=int, default=128)
    parser.add_argument("--registration-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-document-length", type=int, default=384)
    parser.add_argument("--max-query-length", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--information-condition",
        choices=("registered_memory", "common_document"),
        default="registered_memory",
        help=(
            "Generate arguments from registered latent memory or fetch the selected "
            "tool's complete document under the common-document protocol."
        ),
    )
    parser.add_argument("--max-examples-per-family", type=int, default=0)
    parser.add_argument(
        "--document-instruction", default=DEFAULT_DOCUMENT_INSTRUCTION
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.family = args.family or ["retrieval", "arguments"]
    result = evaluate_late_bound_checkpoint(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
