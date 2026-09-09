"""Iterative document/field resampling; all tools share the same parameters."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FloatLinear(nn.Linear):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value.float(), self.weight.float(), None if self.bias is None else self.bias.float())


class FloatNorm(nn.LayerNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps)


class SlotAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.query = FloatLinear(width, width, bias=False)
        self.key = FloatLinear(width, width, bias=False)
        self.value = FloatLinear(width, width, bias=False)
        self.output = FloatLinear(width, width, bias=False)

    def forward(self, query: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = query.shape
        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        scores = heads(self.query(query)) @ heads(self.key(source)).transpose(-1, -2) / self.head_dim ** 0.5
        scores = scores.masked_fill(~mask[:, None, None, :].bool(), -torch.inf)
        values = scores.softmax(-1) @ heads(self.value(source))
        return self.output(values.transpose(1, 2).reshape(batch, length, width))


class ReadRefineBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.cross_norm, self.self_norm, self.ffn_norm = [FloatNorm(width) for _ in range(3)]
        self.cross_attention = SlotAttention(width, heads)
        self.self_attention = SlotAttention(width, heads)
        self.ffn = nn.Sequential(FloatLinear(width, width * 4), nn.GELU(), FloatLinear(width * 4, width))

    def forward(self, slots: torch.Tensor, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        slots = slots + self.cross_attention(self.cross_norm(slots), source, source_mask)
        normalized = self.self_norm(slots)
        slots = slots + self.self_attention(normalized, normalized,
                    torch.ones(slots.shape[:2], dtype=torch.bool, device=slots.device))
        return slots + self.ffn(self.ffn_norm(slots))


class StructuredResamplerMemory(nn.Module):
    """Two rounds of cross-attention, slot self-attention and FFN by default.

    Schema field views are pooled from the same contextual document states.
    Final slots also read full-hidden-size values, avoiding a width-only value
    bottleneck. No API-specific weights, field-specific backbone passes or query
    dependence are introduced.
    """
    def __init__(self, hidden_size: int, width: int = 512, num_slots: int = 8,
                 depth: int = 2, heads: int = 8, initial_norm: float = 1.0):
        super().__init__()
        if min(width, num_slots, depth, heads) < 1 or width % heads:
            raise ValueError("Positive dimensions and width divisible by heads are required")
        self.num_slots, self.width, self.depth, self.heads = num_slots, width, depth, heads
        self.input_norm = FloatNorm(hidden_size)
        self.input_projection = FloatLinear(hidden_size, width, bias=False)
        self.source_type = nn.Parameter(torch.randn(2, width) * 0.02)
        self.queries = nn.Parameter(torch.randn(num_slots, width) * 0.02)
        self.blocks = nn.ModuleList([ReadRefineBlock(width, heads) for _ in range(depth)])
        self.final_norm = FloatNorm(width)
        self.read_query = FloatLinear(width, width, bias=False)
        self.read_key = FloatLinear(width, width, bias=False)
        self.output_projection = FloatLinear(width, hidden_size, bias=False)
        self.log_output_norm = nn.Parameter(torch.full((num_slots,), float(initial_norm)).log())

    def forward(self, token_states: torch.Tensor, attention_mask: torch.Tensor,
                field_mask: torch.Tensor | None = None) -> torch.Tensor:
        if token_states.ndim != 3 or attention_mask.shape != token_states.shape[:2]:
            raise ValueError("Expected document states [B,L,H] and mask [B,L]")
        mask = attention_mask.bool()
        if not mask.any(-1).all():
            raise ValueError("Every document must contain an unmasked token")
        if field_mask is None:
            field_mask = torch.zeros((len(mask), 0, mask.shape[1]), device=mask.device, dtype=torch.bool)
        if field_mask.ndim != 3 or field_mask.shape[0] != len(mask) or field_mask.shape[-1] != mask.shape[-1]:
            raise ValueError("Expected field membership mask [B,F,L]")
        if (field_mask.bool() & ~mask[:, None, :]).any():
            raise ValueError("Schema field masks must not include padding")
        with torch.autocast(device_type=token_states.device.type, enabled=False):
            states = token_states.float()
            weights = field_mask.float()
            counts = weights.sum(-1, keepdim=True)
            fields = (weights @ states) / counts.clamp_min(1)
            raw_values = torch.cat([states, fields], dim=1)
            source = self.input_projection(self.input_norm(raw_values))
            type_vectors = torch.cat([self.source_type[0].float().expand(states.shape[1], -1),
                                      self.source_type[1].float().expand(fields.shape[1], -1)])
            source = source + type_vectors[None]
            source_mask = torch.cat([mask, counts.squeeze(-1).gt(0)], dim=1)
            slots = self.queries.float()[None].expand(len(states), -1, -1)
            for block in self.blocks:
                slots = block(slots, source, source_mask)
            slots = self.final_norm(slots)
            scores = self.read_query(slots) @ self.read_key(source).transpose(-1, -2) / self.width ** 0.5
            scores = scores.masked_fill(~source_mask[:, None, :], -torch.inf)
            full_width_values = scores.softmax(-1) @ raw_values
            directions = F.normalize(full_width_values + self.output_projection(slots), dim=-1)
            return directions * self.log_output_norm.float().exp()[None, :, None]
