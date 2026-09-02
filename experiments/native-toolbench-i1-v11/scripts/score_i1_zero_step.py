#!/usr/bin/env python3
"""Blindly score official G1/I1 queries against zero-step registered APIs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_train(path: Path):
    spec = importlib.util.spec_from_file_location("native_i1_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training entry: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(train: Any, root: Path, model_path: Path, native_root: Path, device: torch.device):
    metadata = json.loads((root / "training_metadata.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "reserved_ids.json").read_text(encoding="utf-8"))
    train_ids = [int(value) for value in mapping["ids"]]
    expected = hashlib.sha256(json.dumps(train_ids, separators=(",", ":")).encode()).hexdigest()
    if mapping.get("sha256") != expected or metadata.get("reserved_ids_sha256") != expected:
        raise ValueError("training reserved-ID mapping audit failed")
    tokenizer = train._tokenizer(model_path)
    backbone = train._load_backbone(model_path, device, gradient_checkpointing=False)
    output_norm = float(backbone.get_output_embeddings().weight.detach().float().norm(dim=-1).mean())
    compiler = train.NativeBundleCompiler(
        int(backbone.config.hidden_size),
        int(metadata.get("compiler_rank", 64)),
        output_norm,
        int(metadata.get("memory_slots", 8)),
    ).to(device)
    model = train.NativeSequenceSFT(backbone, compiler, train_ids).to(device)
    checkpoint_file = root / "checkpoint.pt"
    if checkpoint_file.is_file():
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)["model"]
        kind = "single_process"
    else:
        import deepspeed

        state = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
            str(root), tag="final"
        )
        kind = "deepspeed_zero3"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise FloatingPointError("non-finite parameter after reload")
    model.eval()
    return model, tokenizer, train_ids, kind


def available_eval_ids(tokenizer, train_ids: list[int], queries: list[str], count: int) -> list[int]:
    observed: set[int] = set(train_ids)
    for query in queries:
        observed.update(tokenizer(query, add_special_tokens=False)["input_ids"])
    observed.update(int(value) for value in tokenizer.all_special_ids if value is not None)
    values = [value for value in range(int(tokenizer.vocab_size) - 1, -1, -1) if value not in observed]
    if len(values) < count:
        raise ValueError("not enough held-out physical IDs")
    return sorted(values[:count])


@torch.inference_mode()
def encode_documents(model, tokenizer, rows: list[dict[str, Any]], batch_size: int, max_length: int, device):
    decoder = model.backbone.model
    result = []
    for start in range(0, len(rows), batch_size):
        texts = [json.dumps(row["document"], ensure_ascii=False, sort_keys=True) for row in rows[start : start + batch_size]]
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        hidden = decoder(**tokens, use_cache=False, return_dict=True).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        result.append(model.compiler.output_rows(pooled).float().cpu())
    return torch.cat(result, dim=0)


@torch.inference_mode()
def encode_queries(model, tokenizer, rows: list[dict[str, str]], batch_size: int, max_length: int, device):
    decoder = model.backbone.model
    result = []
    for start in range(0, len(rows), batch_size):
        texts = [f"<|start_header_id|>user<|end_header_id|>\n\n{row['query']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for row in rows[start : start + batch_size]]
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        hidden = decoder(**tokens, use_cache=False, return_dict=True).last_hidden_state
        lengths = tokens["attention_mask"].sum(1).clamp_min(1) - 1
        result.append(hidden[torch.arange(len(hidden), device=device), lengths].float().cpu())
    return torch.cat(result, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--document-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--max-document-length", type=int, default=512)
    parser.add_argument("--max-query-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    public = args.evaluation_root
    manifest = json.loads((public / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "native_latebound_official_g1_zero_step_evaluation":
        raise ValueError("unexpected evaluation manifest")
    registry = [json.loads(line) for line in (public / "registry.jsonl").read_text(encoding="utf-8").splitlines() if line]
    queries = [json.loads(line) for line in (public / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train = load_train(args.native_root / "scripts" / "train.py")
    model, tokenizer, train_ids, checkpoint_kind = load_checkpoint(train, args.checkpoint_root, args.model_path, args.native_root, device)
    eval_ids = available_eval_ids(tokenizer, train_ids, [row["query"] for row in queries], len(registry))
    binding_path = args.output / "registry_binding.json"
    args.output.mkdir(parents=True)
    binding = {
        "kind": "native_i1_zero_step_registry_binding",
        "train_reserved_ids_sha256": hashlib.sha256(json.dumps(train_ids, separators=(",", ":")).encode()).hexdigest(),
        "evaluation_ids_sha256": hashlib.sha256(json.dumps(eval_ids, separators=(",", ":")).encode()).hexdigest(),
        "evaluation_optimizer_steps": 0,
        "changed_parameter_count": 0,
        "bindings": [{"docid": row["docid"], "identity_hash": row["identity_hash"], "physical_id": physical} for row, physical in zip(registry, eval_ids)],
    }
    binding_path.write_text(json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    document_rows = encode_documents(model, tokenizer, registry, args.document_batch_size, args.max_document_length, device)
    query_rows = encode_queries(model, tokenizer, queries, args.query_batch_size, args.max_query_length, device)
    scores_path = args.output / "scores.npy"
    scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=np.float32, shape=(len(queries), len(registry)))
    for start in range(0, len(queries), args.query_batch_size):
        scores[start : start + args.query_batch_size] = (query_rows[start : start + args.query_batch_size] @ document_rows.T).numpy()
    scores.flush()
    index_path = args.output / "query_index.jsonl"
    index_path.write_text("".join(json.dumps({"row": index, "qid": row["qid"]}, sort_keys=True) + "\n" for index, row in enumerate(queries)), encoding="utf-8")
    score_manifest = {
        "kind": "native_i1_zero_step_blind_scores",
        "passed": True,
        "candidate_documents": len(registry),
        "query_rows": len(queries),
        "score_shape": [len(queries), len(registry)],
        "checkpoint_kind": checkpoint_kind,
        "registry_sha256": hashlib.sha256((public / "registry.jsonl").read_bytes()).hexdigest(),
        "query_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "registry_binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        "qrels_read": False,
        "registration_optimizer_steps": 0,
        "changed_parameter_count": 0,
    }
    manifest_path = args.output / "score_manifest.json"
    manifest_path.write_text(json.dumps(score_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "COMPLETE").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    print(f"i1-blind-score-ready queries={len(queries)} candidates={len(registry)}")


if __name__ == "__main__":
    main()
