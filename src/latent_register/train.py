from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .data import ToolExample, load_function_call_jsonl, stable_tool_split
from .model import RegisteredToolSelector, multi_positive_contrastive_loss
from .registry import DynamicRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="DATA_JSONL::ANSWER_JSONL",
        help="May be supplied more than once.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-file")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(specifications: list[str], max_examples: int) -> list[ToolExample]:
    examples: list[ToolExample] = []
    for specification in specifications:
        try:
            data_path, answer_path = specification.split("::", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid dataset specification: {specification}") from exc
        examples.extend(load_function_call_jsonl(data_path, answer_path))
    examples = list({item.example_id: item for item in examples}.values())
    if max_examples:
        examples = examples[:max_examples]
    if not examples:
        raise ValueError("No single-tool examples were loaded")
    return examples


class FrozenBackboneEncoder:
    def __init__(self, model_path: str, device: str, max_length: int):
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int, pooling: str) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            if pooling == "mean":
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            elif pooling == "last":
                indices = tokens["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(len(hidden), device=self.device), indices]
            else:
                raise ValueError(f"Unknown pooling mode: {pooling}")
            outputs.append(pooled.float().cpu())
        return torch.cat(outputs)


def build_features(
    examples: list[ToolExample],
    encoder: FrozenBackboneEncoder,
    batch_size: int,
) -> dict[str, Any]:
    query_prompts = [
        f"Select the best tool for this request.\nRequest: {item.query}\nTool:" for item in examples
    ]
    unique_documents = {item.tool_name: item.tool_document for item in examples}
    tool_names = sorted(unique_documents)
    tool_prompts = [
        f"Represent this tool for semantic selection.\n{unique_documents[name]}" for name in tool_names
    ]
    return {
        "examples": [asdict(item) for item in examples],
        "query_features": encoder.encode(query_prompts, batch_size, pooling="last"),
        "tool_names": tool_names,
        "tool_features": encoder.encode(tool_prompts, batch_size, pooling="mean"),
        "hidden_size": encoder.hidden_size,
    }


def metrics_from_scores(
    scores: torch.Tensor,
    query_names: list[str],
    candidate_names: list[str],
) -> dict[str, float]:
    order = scores.argsort(dim=1, descending=True)
    ranks = []
    if len(order) != len(query_names):
        raise ValueError("scores and query names must have equal lengths")
    for row, target in zip(order, query_names):
        ranked_names = [candidate_names[index] for index in row.tolist()]
        ranks.append(ranked_names.index(target) + 1)
    return {
        "top1": sum(rank == 1 for rank in ranks) / len(ranks),
        "top5": sum(rank <= 5 for rank in ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
    }


def select_split_tensors(
    split: list[ToolExample],
    all_examples: list[ToolExample],
    query_features: torch.Tensor,
    all_tool_names: list[str],
    tool_features: torch.Tensor,
) -> tuple[torch.Tensor, list[str], torch.Tensor, list[str]]:
    example_index = {item.example_id: index for index, item in enumerate(all_examples)}
    tool_index = {name: index for index, name in enumerate(all_tool_names)}
    split_tools = sorted({item.tool_name for item in split})
    return (
        query_features[[example_index[item.example_id] for item in split]],
        [item.tool_name for item in split],
        tool_features[[tool_index[name] for name in split_tools]],
        split_tools,
    )


def slot_invariance_audit(
    query_vectors: torch.Tensor,
    output_rows: torch.Tensor,
    tool_names: list[str],
) -> dict[str, float | int]:
    first_slots = list(range(0, len(tool_names)))
    unseen_slots = list(range(3072, 3072 + len(tool_names)))
    first = DynamicRegistry.from_vectors(tool_names, output_rows.cpu(), first_slots)
    second = first.remap(unseen_slots)
    max_delta = 0.0
    disagreements = 0
    for query in query_vectors.cpu():
        left = first.score_by_tool(query)
        right = second.score_by_tool(query)
        max_delta = max(max_delta, max(abs(left[name] - right[name]) for name in tool_names))
        disagreements += max(left, key=left.__getitem__) != max(right, key=right.__getitem__)
    return {
        "max_score_delta": max_delta,
        "prediction_disagreements": disagreements,
        "queries": len(query_vectors),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_file) if args.cache_file else output_dir / "features.pt"

    examples = load_examples(args.dataset, args.max_examples)
    train_examples, eval_examples = stable_tool_split(examples, args.eval_ratio, args.seed)

    if cache_path.exists():
        features = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_ids = [item["example_id"] for item in features["examples"]]
        if cached_ids != [item.example_id for item in examples]:
            raise ValueError("Feature cache does not match the loaded examples")
    else:
        encoder = FrozenBackboneEncoder(args.model_path, args.device, args.max_length)
        features = build_features(examples, encoder, args.encode_batch_size)
        torch.save(features, cache_path)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_q, train_q_names, train_t, train_t_names = select_split_tensors(
        train_examples,
        examples,
        features["query_features"],
        features["tool_names"],
        features["tool_features"],
    )
    eval_q, eval_q_names, eval_t, eval_t_names = select_split_tensors(
        eval_examples,
        examples,
        features["query_features"],
        features["tool_names"],
        features["tool_features"],
    )

    device = torch.device(args.device)
    selector = RegisteredToolSelector(features["hidden_size"], args.rank).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.learning_rate, weight_decay=0.01)
    tool_lookup = {name: index for index, name in enumerate(train_t_names)}

    for epoch in range(args.epochs):
        order = torch.randperm(len(train_q))
        losses = []
        selector.train()
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            names = [train_q_names[index] for index in indices.tolist()]
            candidate_names = sorted(set(names))
            candidate_indices = [tool_lookup[name] for name in candidate_names]
            logits = selector(
                train_q[indices].to(device),
                train_t[candidate_indices].to(device),
            )
            loss = multi_positive_contrastive_loss(logits, names, candidate_names)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch={epoch + 1} loss={sum(losses) / len(losses):.6f}", flush=True)

    selector.eval()
    with torch.inference_mode():
        raw_scores = F.normalize(eval_q.float(), dim=-1) @ F.normalize(eval_t.float(), dim=-1).T
        registered_scores = selector(eval_q.to(device), eval_t.to(device)).cpu()
        query_vectors = selector.query_vectors(eval_q.to(device)).cpu()
        output_rows = selector.output_rows(eval_t.to(device)).cpu()

    results = {
        "hypothesis": "Generated output rows select tool identities never present in training.",
        "split": {
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "train_tools": len(train_t_names),
            "eval_tools": len(eval_t_names),
            "tool_overlap": len(set(train_t_names) & set(eval_t_names)),
        },
        "raw_frozen_cosine": metrics_from_scores(raw_scores, eval_q_names, eval_t_names),
        "registered_output_rows": metrics_from_scores(
            registered_scores, eval_q_names, eval_t_names
        ),
        "unseen_slot_audit": slot_invariance_audit(query_vectors, output_rows, eval_t_names),
        "config": vars(args),
    }
    torch.save(selector.state_dict(), output_dir / "selector.pt")
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
