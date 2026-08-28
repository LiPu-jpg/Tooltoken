from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


def token_identity_audit(
    token_strings: Sequence[str], token_ids: Sequence[int]
) -> dict[str, Any]:
    """Record the runtime proof that every reserved address is one unique ID."""
    if not token_strings or len(token_strings) != len(token_ids):
        raise ValueError("Reserved token strings and IDs must be nonempty and aligned")
    mapping = "\n".join(
        f"{token_string}\t{int(token_id)}"
        for token_string, token_id in zip(token_strings, token_ids)
    ).encode("utf-8")
    sorted_ids = sorted(int(token_id) for token_id in token_ids)
    return {
        "atomic_tokenization_verified": True,
        "distinct_reserved_token_ids": len(set(sorted_ids)),
        "reserved_token_ids_contiguous": sorted_ids
        == list(range(sorted_ids[0], sorted_ids[-1] + 1)),
        "reserved_token_id_mapping_sha256": hashlib.sha256(mapping).hexdigest(),
    }


def validate_token_identity_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the physical-address identity evidence stored in an artifact."""
    total = int(audit.get("total_reserved_tokens", -1))
    distinct = int(audit.get("distinct_reserved_token_ids", -1))
    digest = audit.get("reserved_token_id_mapping_sha256")
    evidence = {
        "total_reserved_tokens": total,
        "atomic_tokenization_verified": audit.get(
            "atomic_tokenization_verified"
        ),
        "distinct_reserved_token_ids": distinct,
        "reserved_token_ids_contiguous": audit.get(
            "reserved_token_ids_contiguous"
        ),
        "reserved_token_id_mapping_sha256": digest,
    }
    evidence["passed"] = (
        total > 0
        and evidence["atomic_tokenization_verified"] is True
        and distinct == total
        and evidence["reserved_token_ids_contiguous"] is True
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    return evidence


@dataclass
class ReservedTokenPool:
    token_strings: list[str]
    token_ids: list[int]
    train_token_ids: list[int]
    heldout_token_ids: list[int]
    original_vocab_size: int
    initial_heldout_input_rows: torch.Tensor
    initial_heldout_output_rows: torch.Tensor

    @classmethod
    def create(
        cls,
        tokenizer,
        model,
        total_tokens: int,
        train_tokens: int,
        prefix: str = "<|latent_tool_slot_",
    ) -> "ReservedTokenPool":
        if not 0 < train_tokens < total_tokens:
            raise ValueError("train_tokens must be between zero and total_tokens")
        original_vocab_size = len(tokenizer)
        token_strings = [f"{prefix}{index:05d}|>" for index in range(total_tokens)]
        added = tokenizer.add_special_tokens({"additional_special_tokens": token_strings})
        if added != total_tokens:
            raise ValueError(f"Expected to add {total_tokens} reserved tokens, added {added}")
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        token_ids = tokenizer.convert_tokens_to_ids(token_strings)
        if len(set(token_ids)) != total_tokens or min(token_ids) < original_vocab_size:
            raise ValueError("Reserved tokens did not receive distinct new vocabulary IDs")
        encoded_rows = tokenizer(
            token_strings, add_special_tokens=False
        ).input_ids
        for token_string, token_id, encoded in zip(
            token_strings, token_ids, encoded_rows
        ):
            if encoded != [token_id]:
                raise ValueError(
                    f"Reserved token {token_string} is not an atomic tokenizer ID: {encoded}"
                )

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        heldout_ids = token_ids[train_tokens:]
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        return cls(
            token_strings=token_strings,
            token_ids=token_ids,
            train_token_ids=token_ids[:train_tokens],
            heldout_token_ids=heldout_ids,
            original_vocab_size=original_vocab_size,
            initial_heldout_input_rows=input_weight[heldout_ids].detach().float().cpu().clone(),
            initial_heldout_output_rows=output_weight[heldout_ids].detach().float().cpu().clone(),
        )

    def token_string(self, token_id: int) -> str:
        return self.token_strings[self.token_ids.index(token_id)]

    def audit(self, model, training_seen_ids: set[int]) -> dict[str, Any]:
        heldout = set(self.heldout_token_ids)
        input_rows = model.get_input_embeddings().weight[self.heldout_token_ids].detach().float().cpu()
        output_rows = model.get_output_embeddings().weight[self.heldout_token_ids].detach().float().cpu()
        return {
            "original_vocab_size": self.original_vocab_size,
            "total_reserved_tokens": len(self.token_ids),
            **token_identity_audit(self.token_strings, self.token_ids),
            "train_reserved_tokens": len(self.train_token_ids),
            "strictly_heldout_tokens": len(self.heldout_token_ids),
            "heldout_ids_seen_during_training": len(heldout & training_seen_ids),
            "heldout_input_row_max_change": float(
                (input_rows - self.initial_heldout_input_rows).abs().max()
            ),
            "heldout_output_row_max_change": float(
                (output_rows - self.initial_heldout_output_rows).abs().max()
            ),
            "all_backbone_parameters_frozen": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
        }


@dataclass
class SplitReservedTokenPool:
    """Materialize manifest slot offsets as real, disjoint tokenizer IDs."""

    token_strings: list[str]
    token_ids: list[int]
    token_ids_by_split: dict[str, tuple[int, ...]]
    original_vocab_size: int
    evaluation_token_ids: tuple[int, ...]
    initial_evaluation_input_rows: torch.Tensor
    initial_evaluation_output_rows: torch.Tensor

    @classmethod
    def create(
        cls,
        tokenizer,
        model,
        pools: Mapping[str, range],
        prefix: str = "<|latent_tool_slot_",
    ) -> "SplitReservedTokenPool":
        required = {"train", "validation", "test"}
        if set(pools) != required:
            raise ValueError(f"Expected token pools {sorted(required)}, got {sorted(pools)}")
        occupied: dict[int, str] = {}
        for split, values in pools.items():
            if not values:
                raise ValueError(f"Token pool {split} is empty")
            for logical_slot in values:
                previous = occupied.setdefault(logical_slot, split)
                if previous != split:
                    raise ValueError(f"Logical token pools overlap at slot {logical_slot}")
        if min(occupied) != 0 or set(occupied) != set(range(max(occupied) + 1)):
            raise ValueError("Logical token pools must densely cover slots from zero")

        original_vocab_size = len(tokenizer)
        token_strings = [
            f"{prefix}{index:05d}|>" for index in range(max(occupied) + 1)
        ]
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": token_strings}
        )
        if added != len(token_strings):
            raise ValueError(
                f"Expected to add {len(token_strings)} reserved tokens, added {added}"
            )
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(tokenizer))
        token_ids = list(tokenizer.convert_tokens_to_ids(token_strings))
        if len(set(token_ids)) != len(token_ids) or min(token_ids) < original_vocab_size:
            raise ValueError("Reserved tokens did not receive distinct new vocabulary IDs")
        encoded_rows = tokenizer(
            token_strings, add_special_tokens=False
        ).input_ids
        for token_string, token_id, encoded in zip(
            token_strings, token_ids, encoded_rows
        ):
            if encoded != [token_id]:
                raise ValueError(
                    f"Reserved token {token_string} is not an atomic tokenizer ID: {encoded}"
                )

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        token_ids_by_split = {
            split: tuple(token_ids[index] for index in values)
            for split, values in pools.items()
        }
        evaluation_ids = (
            token_ids_by_split["validation"] + token_ids_by_split["test"]
        )
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        return cls(
            token_strings=token_strings,
            token_ids=token_ids,
            token_ids_by_split=token_ids_by_split,
            original_vocab_size=original_vocab_size,
            evaluation_token_ids=evaluation_ids,
            initial_evaluation_input_rows=input_weight[list(evaluation_ids)]
            .detach()
            .float()
            .cpu()
            .clone(),
            initial_evaluation_output_rows=output_weight[list(evaluation_ids)]
            .detach()
            .float()
            .cpu()
            .clone(),
        )

    def physical_id(self, logical_slot: int) -> int:
        return self.token_ids[logical_slot]

    def audit(self, model, training_seen_ids: set[int]) -> dict[str, Any]:
        evaluation_set = set(self.evaluation_token_ids)
        input_rows = (
            model.get_input_embeddings()
            .weight[list(self.evaluation_token_ids)]
            .detach()
            .float()
            .cpu()
        )
        output_rows = (
            model.get_output_embeddings()
            .weight[list(self.evaluation_token_ids)]
            .detach()
            .float()
            .cpu()
        )
        return {
            "original_vocab_size": self.original_vocab_size,
            "total_reserved_tokens": len(self.token_ids),
            **token_identity_audit(self.token_strings, self.token_ids),
            "train_reserved_tokens": len(self.token_ids_by_split["train"]),
            "validation_reserved_tokens": len(
                self.token_ids_by_split["validation"]
            ),
            "test_reserved_tokens": len(self.token_ids_by_split["test"]),
            "evaluation_ids_seen_during_training": len(
                evaluation_set.intersection(training_seen_ids)
            ),
            "evaluation_input_row_max_change": float(
                (input_rows - self.initial_evaluation_input_rows).abs().max()
            ),
            "evaluation_output_row_max_change": float(
                (output_rows - self.initial_evaluation_output_rows).abs().max()
            ),
        }
