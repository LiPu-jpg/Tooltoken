#!/usr/bin/env python3
"""Reload a smoke checkpoint and verify the full model is finite and usable."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_train_module(path: Path):
    spec = importlib.util.spec_from_file_location("latebound_train_reload", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training entry: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.checkpoint_root
    metadata = json.loads((root / "training_metadata.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "reserved_ids.json").read_text(encoding="utf-8"))
    ids = [int(value) for value in mapping["ids"]]
    expected_mapping_hash = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    if mapping.get("sha256") != expected_mapping_hash:
        raise ValueError("reserved physical-ID mapping hash mismatch")
    if metadata.get("reserved_ids_sha256") != expected_mapping_hash:
        raise ValueError("training metadata does not bind the reserved mapping")

    train = _load_train_module(args.native_root / "scripts" / "train.py")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint reload")
    tokenizer = train._tokenizer(args.model_path)
    backbone = train._load_backbone(args.model_path, device, gradient_checkpointing=False)
    output_norm = float(
        backbone.get_output_embeddings().weight.detach().float().norm(dim=-1).mean()
    )
    compiler_rank = int(metadata.get("compiler_rank", 64))
    memory_slots = int(metadata.get("memory_slots", 8))
    compiler = train.NativeBundleCompiler(
        int(backbone.config.hidden_size), compiler_rank, output_norm, memory_slots
    ).to(device)
    model = train.NativeSequenceSFT(backbone, compiler, ids).to(device)

    checkpoint_file = root / "checkpoint.pt"
    if checkpoint_file.is_file():
        payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        state = payload["model"]
        checkpoint_kind = "single_process"
    else:
        import deepspeed

        state = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
            str(root), tag="final"
        )
        checkpoint_kind = "deepspeed_zero3"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError(f"non-finite parameter after reload: {name}")

    result = {
        "kind": "native_latebound_checkpoint_reload_audit",
        "passed": True,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_root": str(root),
        "model_path": str(args.model_path),
        "reserved_ids_sha256": expected_mapping_hash,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "metadata_sha256": hashlib.sha256(
            (root / "training_metadata.json").read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_name("COMPLETE").write_text(digest + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
