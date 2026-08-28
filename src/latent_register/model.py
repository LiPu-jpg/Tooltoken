from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LowRankResidual(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(F.silu(self.down(value)))


class RegisteredToolSelector(nn.Module):
    """Generate dynamic output rows and align frozen query action states to them."""

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.query_adapter = LowRankResidual(hidden_size, rank)
        self.output_generator = LowRankResidual(hidden_size, rank)
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # ln(1 / 0.07)

    def query_vectors(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_adapter(features.float()), dim=-1)

    def output_rows(self, tool_latents: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.output_generator(tool_latents.float()), dim=-1)

    def forward(self, query_features: torch.Tensor, tool_latents: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * self.query_vectors(query_features) @ self.output_rows(tool_latents).T


class GeneratedVector(nn.Module):
    """Generate a runtime vocabulary vector without any per-token parameters."""

    def __init__(self, hidden_size: int, rank: int, initial_norm: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.log_output_norm = nn.Parameter(torch.tensor(float(initial_norm)).log())
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.silu(self.down(self.norm(latent.float()))))
        direction = F.normalize(latent.float() + residual, dim=-1)
        return direction * self.log_output_norm.exp()


class GeneratedMemory(nn.Module):
    """Expand one registered tool latent into several runtime memory vectors."""

    def __init__(self, hidden_size: int, rank: int, num_slots: int, initial_norm: float):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, num_slots * hidden_size, bias=False)
        self.log_output_norm = nn.Parameter(
            torch.full((num_slots,), float(initial_norm)).log()
        )
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("GeneratedMemory expects [batch, hidden] latents")
        residual = self.up(F.silu(self.down(self.norm(latent.float()))))
        residual = residual.reshape(len(latent), self.num_slots, self.hidden_size)
        direction = F.normalize(latent.float().unsqueeze(1) + residual, dim=-1)
        return direction * self.log_output_norm.exp()[None, :, None]


class DualViewGeneratedMemory(nn.Module):
    """Generate fixed slots from complementary mean and final document states."""

    def __init__(self, hidden_size: int, rank: int, num_slots: int, initial_norm: float):
        super().__init__()
        if num_slots < 2:
            raise ValueError("Dual-view memory requires at least two slots")
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.mean_slots = num_slots // 2
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, num_slots * hidden_size, bias=False)
        self.log_output_norm = nn.Parameter(
            torch.full((num_slots,), float(initial_norm)).log()
        )
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1:] != (2, self.hidden_size):
            raise ValueError(
                "DualViewGeneratedMemory expects [batch, 2, hidden] latents"
            )
        normalized = self.norm(latent.float()).flatten(1)
        residual = self.up(F.silu(self.down(normalized)))
        residual = residual.reshape(len(latent), self.num_slots, self.hidden_size)
        base = torch.cat(
            [
                latent[:, :1].expand(-1, self.mean_slots, -1),
                latent[:, 1:2].expand(-1, self.num_slots - self.mean_slots, -1),
            ],
            dim=1,
        ).float()
        direction = F.normalize(base + residual, dim=-1)
        return direction * self.log_output_norm.exp()[None, :, None]


class SchemaKeyGeneratedMemory(nn.Module):
    """Compile exact schema-key anchors plus one global document view."""

    def __init__(self, hidden_size: int, rank: int, num_slots: int, initial_norm: float):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.document_norm = nn.LayerNorm(hidden_size)
        self.anchor_norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.log_output_norm = nn.Parameter(
            torch.full((num_slots,), float(initial_norm)).log()
        )
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        expected_views = self.num_slots + 1
        if latent.ndim != 3 or latent.shape[1:] != (
            expected_views,
            self.hidden_size,
        ):
            raise ValueError(
                "SchemaKeyGeneratedMemory expects "
                f"[batch, {expected_views}, hidden] latents"
            )
        document = latent[:, :1].expand(-1, self.num_slots, -1).float()
        anchors = latent[:, 1:].float()
        compiler_input = torch.cat(
            [self.document_norm(document), self.anchor_norm(anchors)], dim=-1
        )
        residual = self.up(F.silu(self.down(compiler_input)))
        direction = F.normalize(anchors + residual, dim=-1)
        return direction * self.log_output_norm.exp()[None, :, None]


class ReadoutCrossAttentionMemory(nn.Module):
    """Condition frozen decoder readout states on persistent registered memory."""

    def __init__(self, hidden_size: int, rank: int, num_slots: int, initial_norm: float):
        super().__init__()
        self.num_slots = num_slots
        self.rank = rank
        self.memory = GeneratedMemory(hidden_size, rank, num_slots, initial_norm)
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.02)
        nn.init.zeros_(self.output.weight)

    def registered_memory(self, latent: torch.Tensor) -> torch.Tensor:
        return self.memory(latent)

    def adapt(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or memory.ndim != 3:
            raise ValueError("Expected hidden states and memory shaped [batch, length, hidden]")
        if hidden_states.shape[0] != memory.shape[0]:
            raise ValueError("Hidden-state and registered-memory batches differ")
        queries = self.query(self.hidden_norm(hidden_states.float()))
        normalized_memory = self.memory_norm(memory.float())
        keys = self.key(normalized_memory)
        values = self.value(normalized_memory)
        scores = torch.einsum("btr,bsr->bts", queries, keys) / self.rank**0.5
        context = torch.einsum("bts,bsr->btr", scores.softmax(dim=-1), values)
        return hidden_states.float() + self.output(F.silu(context))

    def forward(self, latent: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.adapt(hidden_states, self.registered_memory(latent))


class GatedLayerwiseCrossAttentionBlock(nn.Module):
    """Read registered memory through an initially identity residual branch."""

    def __init__(self, hidden_size: int, rank: int, max_gate: float):
        super().__init__()
        if max_gate <= 0:
            raise ValueError("max_gate must be positive")
        self.rank = rank
        self.max_gate = float(max_gate)
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.zeros(()))
        for projection in (self.query, self.key, self.value, self.output):
            nn.init.normal_(projection.weight, std=0.02)

    def gate(self) -> torch.Tensor:
        return self.gate_logit.tanh() * self.max_gate

    def forward(self, hidden_states: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3 or memory.ndim != 3:
            raise ValueError("Expected hidden states and memory shaped [batch, length, hidden]")
        if hidden_states.shape[0] != memory.shape[0]:
            raise ValueError("Hidden-state and registered-memory batches differ")
        queries = self.query(self.hidden_norm(hidden_states.float()))
        normalized_memory = self.memory_norm(memory.float())
        keys = self.key(normalized_memory)
        values = self.value(normalized_memory)
        scores = torch.einsum("btr,bsr->bts", queries, keys) / self.rank**0.5
        context = torch.einsum(
            "bts,bsr->btr", scores.softmax(dim=-1), values
        )
        residual = self.gate() * self.output(F.silu(context))
        return hidden_states + residual.to(hidden_states.dtype)


class GatedLayerwiseCrossAttentionMemory(nn.Module):
    """Inject persistent registered memory inside selected frozen decoder layers."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        num_slots: int,
        initial_norm: float,
        layer_indices: Sequence[int],
        max_gate: float = 0.25,
        num_views: int = 1,
    ):
        super().__init__()
        indices = tuple(int(index) for index in layer_indices)
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError("layer_indices must contain unique non-negative indices")
        self.num_slots = num_slots
        self.layer_indices = indices
        if num_views not in {1, 2, num_slots + 1}:
            raise ValueError(
                "num_views must be one, two, or one document view plus one "
                "view per memory slot"
            )
        self.memory = (
            GeneratedMemory(hidden_size, rank, num_slots, initial_norm)
            if num_views == 1
            else (
                DualViewGeneratedMemory(hidden_size, rank, num_slots, initial_norm)
                if num_views == 2
                else SchemaKeyGeneratedMemory(
                    hidden_size, rank, num_slots, initial_norm
                )
            )
        )
        self.blocks = nn.ModuleDict(
            {
                str(index): GatedLayerwiseCrossAttentionBlock(
                    hidden_size, rank, max_gate
                )
                for index in indices
            }
        )
        self._active_memory: torch.Tensor | None = None
        self._suspended = False
        self._hook_handles: list[Any] = []

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.memory(latent)

    def registered_memory(self, latent: torch.Tensor) -> torch.Tensor:
        return self.memory(latent)

    def bind(self, memory: torch.Tensor) -> None:
        if memory.ndim != 3 or memory.shape[1] != self.num_slots:
            raise ValueError("Registered memory has an incompatible shape")
        self._active_memory = memory

    def clear(self) -> None:
        self._active_memory = None

    @contextmanager
    def activate(self, memory: torch.Tensor):
        previous = self._active_memory
        self.bind(memory)
        try:
            yield
        finally:
            self._active_memory = previous

    @contextmanager
    def suspended(self):
        previous = self._suspended
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = previous

    def install(self, decoder_layers: Sequence[nn.Module]) -> None:
        if self._hook_handles:
            raise RuntimeError("Layerwise memory hooks are already installed")
        if max(self.layer_indices) >= len(decoder_layers):
            raise ValueError(
                f"Layer index {max(self.layer_indices)} exceeds decoder depth "
                f"{len(decoder_layers)}"
            )
        for index in self.layer_indices:
            self._hook_handles.append(
                decoder_layers[index].register_forward_hook(self._make_hook(index))
            )

    def remove(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def gate_values(self) -> dict[str, float]:
        return {
            key: float(block.gate().detach()) for key, block in self.blocks.items()
        }

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if self._active_memory is None or self._suspended:
                return output
            block = self.blocks[str(layer_index)]
            if isinstance(output, tuple):
                return (block(output[0], self._active_memory), *output[1:])
            return block(output, self._active_memory)

        return hook


class TokenResamplerMemory(nn.Module):
    """Resample contextual tool-document tokens into runtime memory vectors."""

    def __init__(self, hidden_size: int, rank: int, num_slots: int, initial_norm: float):
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        self.rank = rank
        self.norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.queries = nn.Parameter(torch.empty(num_slots, rank))
        self.value_norm = nn.LayerNorm(hidden_size)
        self.value_down = nn.Linear(hidden_size, rank, bias=False)
        self.value_up = nn.Linear(rank, hidden_size, bias=False)
        self.log_output_norm = nn.Parameter(
            torch.full((num_slots,), float(initial_norm)).log()
        )
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.value_down.weight, std=0.02)
        nn.init.zeros_(self.value_up.weight)

    def forward(
        self,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor,
        value_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_states.ndim != 3 or attention_mask.shape != token_states.shape[:2]:
            raise ValueError("Expected token states [batch, length, hidden] and matching mask")
        if value_states is None:
            value_states = token_states
        if value_states.shape != token_states.shape:
            raise ValueError("Value states must match token states")
        keys = self.key(self.norm(token_states.float()))
        scores = torch.einsum("sr,btr->bst", self.queries, keys) / self.rank**0.5
        scores = scores.masked_fill(~attention_mask.bool().unsqueeze(1), float("-inf"))
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("bst,bth->bsh", weights, value_states.float())
        residual = self.value_up(
            F.silu(self.value_down(self.value_norm(pooled)))
        )
        direction = F.normalize(pooled + residual, dim=-1)
        return direction * self.log_output_norm.exp()[None, :, None]


class PhysicalOutputGenerator(nn.Module):
    """Generate rows scored directly as h^T w, matching an LM-head overwrite."""

    def __init__(self, hidden_size: int, rank: int, initial_row_norm: float):
        super().__init__()
        self.generator = GeneratedVector(hidden_size, rank, initial_row_norm)

    def output_rows(self, tool_latents: torch.Tensor) -> torch.Tensor:
        return self.generator(tool_latents)

    def forward(self, query_states: torch.Tensor, tool_latents: torch.Tensor) -> torch.Tensor:
        return query_states.float() @ self.output_rows(tool_latents).T


def multi_positive_contrastive_loss(
    logits: torch.Tensor,
    query_tool_names: Sequence[str],
    candidate_tool_names: Sequence[str],
) -> torch.Tensor:
    positive = torch.tensor(
        [[left == right for right in candidate_tool_names] for left in query_tool_names],
        device=logits.device,
        dtype=torch.bool,
    )
    if not positive.any(dim=1).all():
        raise ValueError("Every query must have at least one positive tool")
    query_loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(
        logits.masked_fill(~positive, float("-inf")), dim=1
    )
    candidate_has_positive = positive.any(dim=0)
    candidate_loss = torch.logsumexp(logits[:, candidate_has_positive], dim=0) - torch.logsumexp(
        logits[:, candidate_has_positive].masked_fill(
            ~positive[:, candidate_has_positive], float("-inf")
        ),
        dim=0,
    )
    return 0.5 * (query_loss.mean() + candidate_loss.mean())
