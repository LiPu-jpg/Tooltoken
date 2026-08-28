from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegistryEntry:
    slot_id: int
    tool_name: str
    output_vector: torch.Tensor
    input_memory: torch.Tensor | None = None


class DynamicRegistry:
    """Runtime mapping from otherwise blank token addresses to generated rows."""

    def __init__(self, entries: list[RegistryEntry]):
        if not entries:
            raise ValueError("registry must contain at least one entry")
        slots = [entry.slot_id for entry in entries]
        tools = [entry.tool_name for entry in entries]
        if len(slots) != len(set(slots)):
            raise ValueError("slot IDs must be unique")
        if len(tools) != len(set(tools)):
            raise ValueError("tool names must be unique")
        memory_presence = [entry.input_memory is not None for entry in entries]
        if any(memory_presence) and not all(memory_presence):
            raise ValueError("input memory must be registered for every entry or none")
        self.entries = entries

    @classmethod
    def from_vectors(
        cls,
        tool_names: list[str],
        vectors: torch.Tensor,
        slot_ids: list[int],
        input_memories: torch.Tensor | None = None,
    ) -> "DynamicRegistry":
        if len(tool_names) != len(vectors) or len(tool_names) != len(slot_ids):
            raise ValueError("tools, vectors, and slots must have equal lengths")
        if input_memories is not None and len(tool_names) != len(input_memories):
            raise ValueError("tools and input memories must have equal lengths")
        return cls(
            [
                RegistryEntry(
                    slot_id=slot,
                    tool_name=name,
                    output_vector=vector,
                    input_memory=(
                        None if input_memories is None else input_memories[index]
                    ),
                )
                for index, (name, vector, slot) in enumerate(
                    zip(tool_names, vectors, slot_ids)
                )
            ]
        )

    def remap(self, slot_ids: list[int]) -> "DynamicRegistry":
        return self.from_vectors(
            [entry.tool_name for entry in self.entries],
            torch.stack([entry.output_vector for entry in self.entries]),
            slot_ids,
            (
                None
                if self.entries[0].input_memory is None
                else torch.stack(
                    [
                        entry.input_memory
                        for entry in self.entries
                        if entry.input_memory is not None
                    ]
                )
            ),
        )

    def score_by_tool(self, query_vector: torch.Tensor) -> dict[str, float]:
        matrix = torch.stack([entry.output_vector for entry in self.entries])
        scores = matrix @ query_vector
        return {
            entry.tool_name: float(score)
            for entry, score in zip(self.entries, scores)
        }

    def select(self, query_vector: torch.Tensor) -> RegistryEntry:
        matrix = torch.stack([entry.output_vector for entry in self.entries])
        selected_index = int((matrix @ query_vector).argmax().item())
        return self.entries[selected_index]

    def expand(self, slot_id: int) -> torch.Tensor:
        entry = next(
            (entry for entry in self.entries if entry.slot_id == slot_id),
            None,
        )
        if entry is None:
            raise KeyError(f"Unregistered slot ID: {slot_id}")
        if entry.input_memory is None:
            raise ValueError(f"Slot {slot_id} has no registered input memory")
        return entry.input_memory
