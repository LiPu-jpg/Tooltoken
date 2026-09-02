#!/usr/bin/env python3
"""Reload either staged checkpoint and verify its optimizer boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("lora", "full"), required=True)
    parser.add_argument("--reserved-pool-size", type=int, default=999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.checkpoint_root
    metadata_path = root / "training_metadata.json"
    mapping_path = root / "reserved_ids.json"
    if not metadata_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError("checkpoint metadata or reserved mapping is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if metadata.get("phase") != args.phase:
        raise ValueError("checkpoint phase does not match reload request")
    ids = [int(value) for value in mapping["ids"]]
    mapping_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if mapping.get("sha256") != mapping_hash or metadata.get("reserved_ids_sha256") != mapping_hash:
        raise ValueError("reserved physical-ID mapping hash mismatch")
    if len(ids) != args.reserved_pool_size:
        raise ValueError("reserved pool size changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint reload")
    tokenizer = train._tokenizer(args.model_path)
    backbone = train._load_backbone(args.model_path, device, gradient_checkpointing=False)
    if args.phase == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(metadata["lora"]["rank"]),
                lora_alpha=int(metadata["lora"]["alpha"]),
                lora_dropout=float(metadata["lora"]["dropout"]),
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
    compiler = train.NativeBundleCompiler(
        int(backbone.config.hidden_size),
        int(metadata.get("compiler_rank", 64)),
        float(backbone.get_output_embeddings().weight.detach().float().norm(dim=-1).mean()),
        int(metadata.get("memory_slots", 8)),
    ).to(device)
    model = train.NativeSequenceSFT(backbone, compiler, ids).to(device)
    state = train._load_zero3_state(root)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    lora_before_merge = sum("lora_" in name for name, _ in model.named_parameters())
    if args.phase == "full":
        lora_after_merge = sum("lora_" in name for name, _ in model.named_parameters())
        full_trainable = all(parameter.requires_grad for parameter in model.backbone.parameters())
    else:
        lora_after_merge = lora_before_merge
        full_trainable = False
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError(f"non-finite parameter after reload: {name}")
    result = {
        "kind": "native_latebound_staged_reload_audit",
        "passed": True,
        "phase": args.phase,
        "checkpoint_root": str(root),
        "reserved_ids_sha256": mapping_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "lora_parameter_names_before_merge": lora_before_merge,
        "lora_parameter_names_after_merge": lora_after_merge,
        "full_backbone_trainable": full_trainable,
        "registration_optimizer_steps": 0,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.with_name("COMPLETE").write_text(
        hashlib.sha256(args.output.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
