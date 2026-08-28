from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import benchmark_metrics as benchmark_metrics_module
from .benchmark_metrics import aggregate_records
from .episodic_data import PreparedTool, load_prepared_tools
from .evaluate_fixed_tokens import (
    argument_prompt,
    generate_arguments,
    load_common_document_agent_audit,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def select_argument_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    conditions: set[str],
    max_examples_per_condition: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in sorted(
        rows, key=lambda item: (str(item.get("condition")), str(item.get("example_id")))
    ):
        condition = str(row.get("condition"))
        if row.get("family") != "arguments" or condition not in conditions:
            continue
        if max_examples_per_condition and counts[condition] >= max_examples_per_condition:
            continue
        selected.append(dict(row))
        counts[condition] += 1
    return selected


def _group_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    return {key: aggregate_records(values) for key, values in sorted(grouped.items())}


def evaluate_full_document_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    benchmark = Path(args.benchmark_dir)
    prepared = Path(args.prepared_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    common_document_agent = load_common_document_agent_audit(
        model_path, model_path.parent / "checkpoint_audit.json"
    )
    tools: dict[str, PreparedTool] = load_prepared_tools(prepared / "tools.jsonl")
    rows = select_argument_rows(
        _read_jsonl(benchmark / "benchmark_eval.jsonl"),
        conditions=set(args.condition),
        max_examples_per_condition=args.max_examples_per_condition,
    )
    if not rows:
        raise ValueError("No argument rows matched the requested conditions")

    prompts: list[str] = []
    for row in rows:
        identities = [str(value) for value in row["reference_tools"]]
        if len(identities) != 1:
            raise ValueError("Full-document argument evaluation requires one reference tool")
        identity = identities[0]
        if identity not in tools:
            raise ValueError(f"Unknown reference tool identity: {identity}")
        prompts.append(
            argument_prompt(
                str(row["query"]),
                selected_token="",
                selected_document=tools[identity].document,
                information_condition="full_document_oracle",
            )
        )

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    started = time.perf_counter()
    generations = generate_arguments(
        model,
        tokenizer,
        prompts,
        batch_size=args.generation_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    elapsed = time.perf_counter() - started

    predictions: list[dict[str, Any]] = []
    for row, generated in zip(rows, generations):
        predictions.append(
            {
                **row,
                "predicted_tools": list(row["reference_tools"]),
                "predicted_arguments": generated.strip(),
                "information_condition": "full_document_oracle",
                "common_document_agent_model_path": common_document_agent[
                    "model_path"
                ],
                "common_document_agent_audit_sha256": common_document_agent[
                    "audit_sha256"
                ],
            }
        )
    predictions_path = output / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    result = {
        "kind": "qwen_full_document",
        "code_hashes": {
            "evaluate_full_document": _sha256_file(Path(__file__)),
            "benchmark_metrics": _sha256_file(
                Path(str(benchmark_metrics_module.__file__))
            ),
        },
        "model_path": str(Path(args.model_path).resolve()),
        "common_document_agent": common_document_agent,
        "common_document_agent_config_sha256": _sha256_file(
            model_path / "config.json"
        ),
        "benchmark_manifest_sha256": _sha256_file(
            benchmark / "benchmark_manifest.json"
        ),
        "conditions": sorted(set(args.condition)),
        "information_condition": "full_document_oracle",
        "prediction_count": len(predictions),
        "elapsed_seconds": elapsed,
        "examples_per_second": len(predictions) / max(elapsed, 1e-9),
        "aggregates": _group_aggregates(predictions),
        "predictions_sha256": _sha256_file(predictions_path),
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen full-document argument-generation checkpoint"
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-examples-per-condition", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.condition = args.condition or [
        "seen_tool_seen_token",
        "unseen_tool_unseen_token",
    ]
    result = evaluate_full_document_checkpoint(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
