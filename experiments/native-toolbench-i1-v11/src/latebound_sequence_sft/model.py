from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import torch
from torch import nn
from torch.nn import functional as F


class BundleCompiler(Protocol):
    def output_rows(self, document_states: torch.Tensor) -> torch.Tensor: ...

    def registered_memory(self, document_states: torch.Tensor) -> torch.Tensor: ...


try:  # The frozen compiler is imported, never edited.
    from latent_register.model import GeneratedMemory, PhysicalOutputGenerator
except ImportError:  # pragma: no cover - tests use ToyBundleCompiler
    GeneratedMemory = None
    PhysicalOutputGenerator = None


class NativeBundleCompiler(nn.Module):
    """Shared output-row and memory compiler with no per-document parameters."""

    def __init__(self, hidden_size: int, rank: int, output_norm: float, memory_slots: int):
        super().__init__()
        if PhysicalOutputGenerator is None or GeneratedMemory is None:
            raise RuntimeError("latent-register/src must be on PYTHONPATH")
        self.output = PhysicalOutputGenerator(hidden_size, rank, output_norm)
        self.memory = GeneratedMemory(hidden_size, rank, memory_slots, output_norm)

    def output_rows(self, document_states: torch.Tensor) -> torch.Tensor:
        # DeepSpeed's BF16 mode casts child parameters, while the frozen
        # compiler intentionally converts its LayerNorm input to FP32.  Keep
        # the frozen architecture and state dict, but run these small
        # projections with FP32 weights so the dtypes agree under ZeRO-3.
        generator = self.output.generator
        latent = document_states.float()
        normalized = F.layer_norm(
            latent,
            (latent.shape[-1],),
            generator.norm.weight.float(),
            generator.norm.bias.float(),
            generator.norm.eps,
        )
        residual = F.linear(
            F.silu(F.linear(normalized, generator.down.weight.float())),
            generator.up.weight.float(),
        )
        direction = F.normalize(latent + residual, dim=-1)
        return direction * generator.log_output_norm.float().exp()

    def registered_memory(self, document_states: torch.Tensor) -> torch.Tensor:
        memory = self.memory
        latent = document_states.float()
        normalized = F.layer_norm(
            latent,
            (latent.shape[-1],),
            memory.norm.weight.float(),
            memory.norm.bias.float(),
            memory.norm.eps,
        )
        residual = F.linear(
            F.silu(F.linear(normalized, memory.down.weight.float())),
            memory.up.weight.float(),
        )
        residual = residual.reshape(len(latent), memory.num_slots, memory.hidden_size)
        direction = F.normalize(latent.unsqueeze(1) + residual, dim=-1)
        return direction * memory.log_output_norm.float().exp()[None, :, None]


@dataclass(frozen=True)
class SequenceDiagnostics:
    document_forward_count: int
    distinct_document_count: int
    dynamic_row_count: int
    reserved_rows_masked: bool
    denominator: str


def _decoder(backbone: nn.Module) -> nn.Module:
    base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    return getattr(base, "model", base)


def _masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(states.dtype).unsqueeze(-1)
    return (states * weights).sum(1) / weights.sum(1).clamp_min(1)


class NativeSequenceSFT(nn.Module):
    """Causal-LM SFT with one positive document forward per distinct document.

    The loss is ordinary token cross-entropy. Reserved physical rows are
    masked, and each example's own dynamic row is scattered into the same
    vocabulary logits before the standard softmax. No retrieval loss or
    explicitly sampled negative document is present.
    """

    def __init__(
        self,
        backbone: nn.Module,
        compiler: BundleCompiler,
        reserved_token_ids: Iterable[int],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.compiler = compiler
        self.register_buffer(
            "reserved_token_ids",
            torch.tensor(sorted({int(value) for value in reserved_token_ids}), dtype=torch.long),
            persistent=False,
        )

    def _document_bundles(
        self,
        document_tokens: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = _decoder(self.backbone)(
            **document_tokens, use_cache=False, return_dict=True
        ).last_hidden_state
        pooled = _masked_mean(states, document_tokens["attention_mask"])
        return self.compiler.output_rows(pooled), self.compiler.registered_memory(pooled)

    def _inject_memory(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        physical_ids: torch.Tensor,
        memories: torch.Tensor,
    ) -> torch.Tensor:
        result = input_embeds.clone()
        for row, physical_id in enumerate(physical_ids.tolist()):
            locations = input_ids[row].eq(int(physical_id))
            if bool(locations.any()):
                result[row, locations] = memories[row].mean(0).to(result.dtype)
        return result

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        document_tokens: dict[str, torch.Tensor],
        physical_ids: torch.Tensor,
        document_index: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SequenceDiagnostics]:
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError("input_ids and labels must be aligned [batch, length]")
        document_count = int(document_tokens["input_ids"].shape[0])
        if document_index is None:
            document_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        if document_index.shape != (input_ids.shape[0],):
            raise ValueError("document_index must contain one entry per sequence row")
        if bool((document_index < 0).any()) or bool((document_index >= document_count).any()):
            raise ValueError("document_index points outside the deduplicated document batch")
        if physical_ids.shape != (input_ids.shape[0],):
            raise ValueError("One physical target address is required per row")
        if loss_weights is None:
            loss_weights = torch.ones(
                input_ids.shape[0], dtype=torch.float32, device=input_ids.device
            )
        if loss_weights.shape != (input_ids.shape[0],):
            raise ValueError("loss_weights must contain one value per sequence row")
        if bool((loss_weights < 0).any()) or not bool(torch.isfinite(loss_weights).all()):
            raise ValueError("loss_weights must be finite and non-negative")
        if self.reserved_token_ids.numel() == 0:
            raise ValueError("The reserved address pool cannot be empty")
        allowed = self.reserved_token_ids.to(physical_ids.device)
        if not bool(torch.isin(physical_ids, allowed).all()):
            raise ValueError("physical_ids must belong to the reserved address pool")
        dynamic_rows, memories = self._document_bundles(document_tokens)
        dynamic_rows = dynamic_rows.index_select(0, document_index.to(dynamic_rows.device))
        memories = memories.index_select(0, document_index.to(memories.device))
        # Reusing an address is legal only when it refers to the same exact
        # document; two different documents sharing an address fail closed.
        seen_addresses: dict[int, int] = {}
        for row, address in enumerate(physical_ids.tolist()):
            source = int(document_index[row])
            previous = seen_addresses.setdefault(int(address), source)
            if previous != source:
                raise ValueError("a physical address is bound to multiple documents")
        input_embeddings = self.backbone.get_input_embeddings()(input_ids)
        input_embeddings = self._inject_memory(
            input_ids, input_embeddings, physical_ids, memories
        )
        hidden = _decoder(self.backbone)(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        logits = self.backbone.get_output_embeddings()(hidden[:, :-1])
        shifted_labels = labels[:, 1:]
        shifted_hidden = hidden[:, :-1].float()
        dynamic_logits = torch.einsum("bth,bh->bt", shifted_hidden, dynamic_rows.float())
        logits = logits.float()
        valid_reserved = self.reserved_token_ids.to(logits.device)
        logits.index_fill_(2, valid_reserved, float("-inf"))
        row_indices = torch.arange(input_ids.shape[0], device=logits.device)
        logits[row_indices, :, physical_ids] = dynamic_logits
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        token_loss = token_loss.reshape(input_ids.shape[0], -1)
        valid = shifted_labels.ne(-100)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every sequence must contain at least one target token")
        per_sequence = (token_loss * valid).sum(1) / valid.sum(1)
        weights = loss_weights.to(per_sequence.device)
        # Keep the complete CE graph on every rank, including ranks whose
        # final distributed batch contains only zero-weight padding. This
        # preserves identical ZeRO-3 collective order while contributing no
        # gradient for those padded rows.
        loss = (per_sequence * weights).sum() / weights.sum().clamp_min(1.0)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Sequence CE is non-finite")
        return loss, SequenceDiagnostics(
            document_forward_count=int(document_tokens["input_ids"].shape[0]),
            distinct_document_count=document_count,
            dynamic_row_count=int(len(dynamic_rows)),
            reserved_rows_masked=True,
            denominator="ordinary_vocabulary_plus_active_dynamic_rows",
        )


def registration_zero_step(
    compiler: BundleCompiler,
    document_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runtime registration helper; intentionally has no optimizer/update path."""
    with torch.no_grad():
        return compiler.output_rows(document_states), compiler.registered_memory(document_states)
