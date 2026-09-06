#!/usr/bin/env python3
"""Evaluate the native meta-registration LoRA/compiler on frozen ToolBench-10K."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_register.model import PhysicalOutputGenerator
from latent_register.train_meta_registration import (
    _disable_incompatible_optional_torchao,
    _provide_optional_tensor_parallel_compat,
    render_query,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metrics(scores: np.ndarray, positives: list[int]) -> tuple[dict[str, float], np.ndarray]:
    positive_set = set(positives)
    best = max(float(scores[index]) for index in positives)
    first_rank = 1 + int(np.sum(scores > best))
    indices = np.argpartition(-scores, 9)[:10]
    indices = indices[np.argsort(-scores[indices], kind="stable")]
    result = {
        "hit_at_1": float(any(index in positive_set for index in indices[:1])),
        "hit_at_3": float(any(index in positive_set for index in indices[:3])),
        "hit_at_5": float(any(index in positive_set for index in indices[:5])),
        "recall_at_5": sum(index in positive_set for index in indices[:5]) / len(positive_set),
        "mrr": 1.0 / first_rank,
    }
    for k in (1, 3, 5):
        dcg = sum(
            (1.0 / math.log2(rank + 2)) if index in positive_set else 0.0
            for rank, index in enumerate(indices[:k])
        )
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(positives))))
        result[f"ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
    return result, indices


def finish(accumulator: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / count for key, value in sorted(accumulator.items())} | {"count": count}


def decoder(backbone):
    causal_lm = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    return causal_lm.model


@torch.inference_mode()
def encode_documents(backbone, compiler, tokenizer, rows, batch_size, max_length, device):
    output = []
    for start in range(0, len(rows), batch_size):
        texts = [f"Represent this tool for registration.\n{row['text']}" for row in rows[start : start + batch_size]]
        tokens = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        hidden = decoder(backbone)(**tokens, use_cache=False, return_dict=True).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        output.append(compiler.output_rows(pooled).float().cpu())
    return torch.cat(output)


@torch.inference_mode()
def encode_queries(backbone, tokenizer, rows, batch_size, max_length, device):
    output = []
    for start in range(0, len(rows), batch_size):
        texts = [render_query(tokenizer, row["query"]) for row in rows[start : start + batch_size]]
        tokens = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        hidden = decoder(backbone)(**tokens, use_cache=False, return_dict=True).last_hidden_state
        positions = tokens["attention_mask"].sum(1).clamp_min(1) - 1
        output.append(hidden[torch.arange(len(hidden), device=device), positions].float().cpu())
    return torch.cat(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--max-document-length", type=int, default=256)
    parser.add_argument("--max-query-length", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    _disable_incompatible_optional_torchao()
    _provide_optional_tensor_parallel_compat()
    initial_row_norm = float(
        base.get_output_embeddings().weight.detach().float().norm(dim=-1).mean()
    )
    backbone = PeftModel.from_pretrained(base, args.training_dir / "lora").to(device).eval()
    compiler_state = torch.load(args.training_dir / "compiler.pt", map_location=device, weights_only=True)
    down = compiler_state["generator.down.weight"]
    compiler = PhysicalOutputGenerator(
        int(backbone.config.hidden_size), int(down.shape[0]), initial_row_norm
    ).to(device)
    compiler.load_state_dict(compiler_state)
    compiler.eval()

    documents = read_jsonl(args.prepared_dir / "documents.jsonl")
    queries = read_jsonl(args.prepared_dir / "queries.jsonl")
    if len(documents) != 10_000 or len(queries) != 1_100:
        raise ValueError("frozen ToolBench-10K cardinality mismatch")
    doc_ids = [row["doc_id"] for row in documents]
    doc_index = {value: index for index, value in enumerate(doc_ids)}
    positives = [[doc_index[value] for value in row["positive_doc_ids"]] for row in queries]

    started = time.perf_counter()
    document_rows = encode_documents(
        backbone, compiler, tokenizer, documents, args.document_batch_size,
        args.max_document_length, device,
    )
    query_rows = encode_queries(
        backbone, tokenizer, queries, args.query_batch_size,
        args.max_query_length, device,
    )
    document_gpu = document_rows.to(device)
    score_rows = []
    for start in range(0, len(query_rows), 128):
        score_rows.extend((query_rows[start : start + 128].to(device) @ document_gpu.T).cpu().numpy())

    overall = defaultdict(float)
    per_subset = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "rankings.jsonl").open("w", encoding="utf-8") as stream:
        for query, scores, positive_indices in zip(queries, score_rows, positives):
            values, top = metrics(scores, positive_indices)
            for key, value in values.items():
                overall[key] += value
                per_subset[query["subset"]][key] += value
            counts[query["subset"]] += 1
            stream.write(
                json.dumps(
                    {
                        "query_id": query["query_id"],
                        "subset": query["subset"],
                        "positive_doc_ids": query["positive_doc_ids"],
                        "top10": [
                            {"doc_id": doc_ids[index], "score": float(scores[index])}
                            for index in top
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    summary = {
        "metadata": {
            "method": "Ours-meta-registration-seen",
            "documents": len(documents),
            "queries": len(queries),
            "registration_optimizer_steps": 0,
            "changed_parameter_count_at_registration": 0,
            "memory_used_for_selection": False,
            "elapsed_seconds": time.perf_counter() - started,
            "protocol_sha256": hashlib.sha256(
                (args.prepared_dir / "manifest.json").read_bytes()
            ).hexdigest(),
        },
        "overall": finish(overall, len(queries)),
        "subsets": {
            name: finish(per_subset[name], count) for name, count in sorted(counts.items())
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETE").write_text("complete\n", encoding="ascii")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
