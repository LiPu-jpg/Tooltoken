"""Explicit weight continuation and recoverable training-state metadata."""
from __future__ import annotations
import json
from pathlib import Path

from .toolbench_checkpoint import load_agent, sha256


def inspect_continuation(path: Path, args, audit: dict) -> dict:
    config = json.loads((path / "agent.json").read_text())
    parent = config["metadata"]
    required = {
        "source_sha256": audit["source_sha256"],
        "train_api_identities": audit["train_api_identities"],
        "train_mode": args.train_mode,
        "schema_weight": args.schema_weight,
    }
    for field, expected in required.items():
        if parent.get(field) != expected:
            raise ValueError(f"Continuation changes parent training contract: {field}")
    expected_config = {
        "condition": args.condition, "rank": args.compiler_rank, "slots": args.memory_slots,
        "memory_compiler": {"kind": args.memory_kind, "width": args.memory_width,
                            "depth": args.memory_depth, "heads": args.memory_heads},
        "limits": {"context": args.max_context_length, "document": args.max_document_length,
                   "target": args.max_target_length},
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            raise ValueError(f"Continuation changes parent model interface: {field}")
    updates = parent.get("updates")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise ValueError("Missing positive parent update count")
    if args.max_updates is None or args.max_updates <= updates:
        raise ValueError("Continuation requires an explicit cumulative target above parent updates")
    return {
        "parent_checkpoint": str(path.resolve()), "parent_manifest_sha256": sha256(path / "SHA256.json"),
        "parent_updates": updates, "target_cumulative_updates": args.max_updates,
        "parent_initialization_repair": parent.get("initialization_repair"),
        "parent_training_stage": parent.get("training_stage", "legacy_teacher"),
        "parent_continuation": parent.get("continuation"),
        "optimizer_restored": False, "scheduler_restored": False, "rng_restored": False,
        "data_order": "new seeded pass over the same training split; previous examples may repeat",
        "parent_world_size": parent.get("world_size"),
        "parent_effective_full_batch": parent.get("effective_full_batch"),
    }


def load_trainable_agent(path, *, train_mode, gradient_checkpointing):
    model = load_agent(path, device="cpu")
    # load_agent is deliberately frozen for serving. Restore compiler gradients
    # as well as the explicitly selected backbone training mode.
    model.requires_grad_(True)
    model.backbone.requires_grad_(train_mode == "full")
    if gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.train()


class RuntimeTrainingState:
    """Accelerator custom checkpoint object: cursor and nonpersistent buffers.

    The normal Accelerator checkpoint saves model, optimizer, scheduler and RNG.
    This object preserves RoPE values omitted by model.state_dict, plus the
    exact batch position needed by a future resume runner. It does not claim
    that the older serving-only checkpoint contained optimizer state.
    """
    def __init__(self, model, contract):
        self.model = model
        self.contract = contract
        self.cursor = {}

    def state_dict(self):
        return {"contract": self.contract, "cursor": self.cursor,
                "backbone_buffers": {k: v.detach().cpu().clone()
                                     for k, v in self.model.backbone.named_buffers()}}

    def load_state_dict(self, state):
        saved = dict(state["contract"])
        current = dict(self.contract)
        # Fields that are not optimization semantics and may legitimately differ
        # across resume: train_api_identities is serving lineage metadata (added
        # after older states were saved); world_size/accumulation are pure
        # parallelization knobs — global batch and the per-sample schedule are
        # fixed, so any world size replays the same samples in the same order.
        for key in ("train_api_identities", "world_size", "accumulation"):
            saved.pop(key, None)
            current.pop(key, None)
        if saved != current:
            raise ValueError("Training state contract mismatch")
        self.cursor = state["cursor"]
        for name, value in state["backbone_buffers"].items():
            parent, _, leaf = name.rpartition(".")
            module = self.model.backbone.get_submodule(parent) if parent else self.model.backbone
            if leaf not in module._buffers:
                raise ValueError(f"Unknown saved runtime buffer: {name}")
            module._buffers[leaf] = value.to(module._buffers[leaf].device)
