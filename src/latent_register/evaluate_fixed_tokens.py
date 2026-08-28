from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import benchmark_metrics as benchmark_metrics_module
from .benchmark_metrics import aggregate_records
from .controlled_registry import build_controlled_registries
from .episodic_data import PreparedTool, load_prepared_tools


DEFAULT_SYSTEM = "You are a helpful assistant."
ARGUMENT_SYSTEM = (
    "Select the required tool. After receiving its definition, return only one "
    "compact JSON object containing the tool arguments."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_common_document_agent_audit(
    model_path: str | Path, audit_path: str | Path
) -> dict[str, Any]:
    model = Path(model_path).resolve()
    audit = Path(audit_path)
    if not (audit.parent / "COMPLETE").is_file():
        raise FileNotFoundError(
            f"Common-document Agent checkpoint is incomplete: {audit.parent}"
        )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    if payload.get("kind") != "qwen_full_document_checkpoint_audit":
        raise ValueError("Unexpected common-document Agent audit kind")
    if Path(str(payload.get("model_path", ""))).resolve() != model:
        raise ValueError("Common-document Agent audit refers to a different model")
    if payload.get("full_model_reloaded") is not True:
        raise ValueError("Common-document Agent audit did not reload the full model")
    if payload.get("embedding_tables_all_finite") is not True:
        raise ValueError("Common-document Agent audit found non-finite embeddings")
    return {
        "model_path": str(model),
        "audit_path": str(audit.resolve()),
        "audit_sha256": _sha256_file(audit),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_fixed_token_mapping(path: str | Path) -> dict[str, dict[str, str | None]]:
    mapping: dict[str, dict[str, str | None]] = {}
    for row in _read_jsonl(Path(path)):
        identity = str(row["identity_hash"])
        if identity in mapping:
            raise ValueError(f"Duplicate fixed-token identity: {identity}")
        mapping[identity] = {
            "split": str(row["split"]),
            "fixed_token": str(row["fixed_token"]),
            "original_token": (
                str(row["original_token"])
                if row.get("original_token") is not None
                else None
            ),
        }
    return mapping


def render_qwen_chatml(
    messages: Sequence[Mapping[str, str]], *, add_generation_prompt: bool
) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"])
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def retrieval_prompt(query: str) -> str:
    return render_qwen_chatml(
        [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": query},
        ],
        add_generation_prompt=True,
    )


def argument_prompt(
    query: str,
    *,
    selected_token: str,
    selected_document: str | None,
    information_condition: str,
) -> str:
    if information_condition == "token_memory_only":
        return render_qwen_chatml(
            [
                {"role": "system", "content": ARGUMENT_SYSTEM},
                {"role": "user", "content": query},
            ],
            add_generation_prompt=True,
        ) + selected_token
    if information_condition in {"full_document_oracle", "common_document"}:
        if selected_document is None:
            raise ValueError("Document-conditioned generation requires a tool definition")
        return render_qwen_chatml(
            [
                {"role": "system", "content": ARGUMENT_SYSTEM},
                {
                    "role": "user",
                    "content": f"Request: {query}\nTool definition:\n{selected_document}",
                },
            ],
            add_generation_prompt=True,
        )
    if information_condition != "native":
        raise ValueError(f"Unsupported information condition: {information_condition}")
    if selected_document is None:
        raise ValueError("ToolGen native evaluation requires the selected document")
    return render_qwen_chatml(
        [
            {"role": "system", "content": ARGUMENT_SYSTEM},
            {"role": "user", "content": query},
            {"role": "assistant", "content": selected_token},
            {
                "role": "user",
                "content": f"Tool definition:\n{selected_document}",
            },
        ],
        add_generation_prompt=True,
    )


def atomic_candidate_ids(
    tokenizer,
    mapping: Mapping[str, Mapping[str, str | None]],
    *,
    splits: set[str],
) -> tuple[list[int], list[str], list[str]]:
    token_ids: list[int] = []
    identities: list[str] = []
    token_strings: list[str] = []
    seen_ids: set[int] = set()
    for identity in sorted(mapping):
        row = mapping[identity]
        if row["split"] not in splits:
            continue
        token = str(row["fixed_token"])
        ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Fixed token is not atomic in checkpoint tokenizer: {token}")
        token_id = int(ids[0])
        if token_id in seen_ids:
            raise ValueError(f"Checkpoint tokenizer aliases fixed token ID {token_id}")
        seen_ids.add(token_id)
        token_ids.append(token_id)
        identities.append(identity)
        token_strings.append(token)
    if not token_ids:
        raise ValueError("No fixed-token candidates matched the requested splits")
    return token_ids, identities, token_strings


def rank_candidate_logits(
    logits: torch.Tensor,
    candidate_ids: Sequence[int],
    identities: Sequence[str],
    token_strings: Sequence[str],
    *,
    k: int = 5,
) -> list[dict[str, list[Any]]]:
    if logits.ndim != 2:
        raise ValueError("Expected [batch, vocabulary] logits")
    if not (len(candidate_ids) == len(identities) == len(token_strings)):
        raise ValueError("Candidate IDs, identities, and strings differ in length")
    if max(candidate_ids) >= logits.shape[1] or min(candidate_ids) < 0:
        raise ValueError("A candidate token ID is outside the model vocabulary")
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=logits.device)
    candidate_logits = logits.float().index_select(1, candidate_tensor)
    top_count = min(k, len(candidate_ids))
    values, positions = candidate_logits.topk(top_count, dim=1)
    results: list[dict[str, list[Any]]] = []
    for row_values, row_positions in zip(values.cpu(), positions.cpu()):
        indices = [int(value) for value in row_positions.tolist()]
        results.append(
            {
                "identities": [identities[index] for index in indices],
                "tokens": [token_strings[index] for index in indices],
                "scores": [float(value) for value in row_values.tolist()],
            }
        )
    return results


@torch.inference_mode()
def predict_fixed_tokens(
    model,
    tokenizer,
    queries: Sequence[str],
    *,
    candidate_ids: Sequence[int],
    identities: Sequence[str],
    token_strings: Sequence[str],
    batch_size: int,
    max_query_length: int,
    device: torch.device,
) -> list[dict[str, list[Any]]]:
    predictions: list[dict[str, list[Any]]] = []
    for start in range(0, len(queries), batch_size):
        prompts = [retrieval_prompt(query) for query in queries[start : start + batch_size]]
        tokens = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_query_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**tokens, use_cache=False, return_dict=True).logits[:, -1]
        predictions.extend(
            rank_candidate_logits(
                logits,
                candidate_ids,
                identities,
                token_strings,
                k=5,
            )
        )
    return predictions


@torch.inference_mode()
def predict_fixed_tokens_in_controlled_registries(
    model,
    tokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_to_token_id: Mapping[str, int],
    identity_to_token_string: Mapping[str, str],
    candidate_identities: Sequence[str],
    registry_size: int,
    registry_seed: int,
    registry_scope: str,
    batch_size: int,
    max_query_length: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    registries = build_controlled_registries(
        candidate_identities,
        [[str(value) for value in row["reference_tools"]] for row in rows],
        registry_size=registry_size,
        seed=registry_seed,
        keys=[
            f"{row['family']}:{row['condition']}:{row['example_id']}" for row in rows
        ],
        scope=registry_scope,
    )
    shared_registry_ids: torch.Tensor | None = None
    if registry_scope == "shared" and registries:
        shared_registry_ids = torch.tensor(
            [identity_to_token_id[identity] for identity in registries[0].identities],
            dtype=torch.long,
            device=device,
        )
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_registries = registries[start : start + batch_size]
        prompts = [retrieval_prompt(str(row["query"])) for row in batch]
        tokens = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_query_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**tokens, use_cache=False, return_dict=True).logits[:, -1]
        for row_index, registry in enumerate(batch_registries):
            registry_ids = (
                shared_registry_ids
                if shared_registry_ids is not None
                else torch.tensor(
                    [
                        identity_to_token_id[identity]
                        for identity in registry.identities
                    ],
                    dtype=torch.long,
                    device=logits.device,
                )
            )
            registry_logits = logits[row_index].float().index_select(
                0, registry_ids
            )
            top_count = min(5, registry_ids.numel())
            values, positions = registry_logits.topk(top_count)
            indices = [int(value) for value in positions.cpu().tolist()]
            identities = [registry.identities[index] for index in indices]
            predictions.append(
                {
                    "identities": identities,
                    "tokens": [identity_to_token_string[value] for value in identities],
                    "scores": [float(value) for value in values.cpu().tolist()],
                    "registry_size": len(registry.identities),
                    "registry_identity_sha256": registry.identity_sha256,
                }
            )
    return predictions


@torch.inference_mode()
def generate_arguments(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    generated: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        tokens = tokenizer(
            list(batch),
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        ).to(device)
        input_length = tokens["input_ids"].shape[1]
        outputs = model.generate(
            **tokens,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        generated.extend(
            tokenizer.batch_decode(
                outputs[:, input_length:], skip_special_tokens=True
            )
        )
    return generated


def _group_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['family']}:{row['condition']}"].append(row)
    return {key: aggregate_records(values) for key, values in sorted(grouped.items())}


def evaluate_fixed_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    benchmark = Path(args.benchmark_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    common_document_agent: dict[str, Any] | None = None
    common_document_agent_path: Path | None = None
    if args.information_condition == "common_document":
        if not args.common_document_agent_model_path:
            raise ValueError(
                "common_document requires --common-document-agent-model-path"
            )
        if not args.common_document_agent_audit_path:
            raise ValueError(
                "common_document requires --common-document-agent-audit-path"
            )
        common_document_agent_path = Path(args.common_document_agent_model_path)
        common_document_agent = load_common_document_agent_audit(
            common_document_agent_path, args.common_document_agent_audit_path
        )
    mapping_path = benchmark / "fixed_tokens.jsonl"
    eval_path = benchmark / "benchmark_eval.jsonl"
    source_tools_path = Path(args.prepared_dir) / "tools.jsonl"
    mapping = load_fixed_token_mapping(mapping_path)
    tools: dict[str, PreparedTool] = load_prepared_tools(source_tools_path)
    rows = _read_jsonl(eval_path)
    rows = [
        row
        for row in rows
        if str(row.get("condition")) in set(args.condition)
        and str(row.get("family")) in set(args.family)
    ]
    if args.max_examples_per_family:
        limited: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in sorted(rows, key=lambda item: (str(item["family"]), str(item["example_id"]))):
            family = str(row["family"])
            if counts[family] >= args.max_examples_per_family:
                continue
            limited.append(row)
            counts[family] += 1
        rows = limited
    if not rows:
        raise ValueError("No benchmark rows matched the requested families and conditions")

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
    argument_model = model
    argument_tokenizer = tokenizer
    if args.information_condition == "common_document":
        argument_tokenizer = AutoTokenizer.from_pretrained(
            common_document_agent_path, local_files_only=True
        )
        if argument_tokenizer.pad_token_id is None:
            argument_tokenizer.pad_token = argument_tokenizer.eos_token
        argument_tokenizer.padding_side = "left"
        argument_model = AutoModelForCausalLM.from_pretrained(
            common_document_agent_path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        argument_model.eval()
    candidate_ids, candidate_identities, candidate_tokens = atomic_candidate_ids(
        tokenizer, mapping, splits=set(args.token_split)
    )
    identity_to_token_id = dict(zip(candidate_identities, candidate_ids))
    identity_to_token_string = dict(zip(candidate_identities, candidate_tokens))

    def predict(rows_to_score: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if args.registry_size:
            return predict_fixed_tokens_in_controlled_registries(
                model,
                tokenizer,
                rows_to_score,
                identity_to_token_id=identity_to_token_id,
                identity_to_token_string=identity_to_token_string,
                candidate_identities=candidate_identities,
                registry_size=args.registry_size,
                registry_seed=args.registry_seed,
                registry_scope=args.registry_scope,
                batch_size=args.retrieval_batch_size,
                max_query_length=args.max_query_length,
                device=device,
            )
        return predict_fixed_tokens(
            model,
            tokenizer,
            [str(row["query"]) for row in rows_to_score],
            candidate_ids=candidate_ids,
            identities=candidate_identities,
            token_strings=candidate_tokens,
            batch_size=args.retrieval_batch_size,
            max_query_length=args.max_query_length,
            device=device,
        )

    predictions: list[dict[str, Any]] = []
    retrieval_rows = [row for row in rows if row["family"] == "retrieval"]
    argument_rows = [row for row in rows if row["family"] == "arguments"]
    started = time.perf_counter()
    retrieval_predictions = predict(retrieval_rows)
    for row, prediction in zip(retrieval_rows, retrieval_predictions):
        predictions.append(
            {
                **row,
                "predicted_tools": prediction["identities"],
                "predicted_fixed_tokens": prediction["tokens"],
                "prediction_scores": prediction["scores"],
                "information_condition": "selection",
                **(
                    {
                        "registry_size": prediction["registry_size"],
                        "registry_identity_sha256": prediction[
                            "registry_identity_sha256"
                        ],
                    }
                    if args.registry_size
                    else {}
                ),
            }
        )

    argument_selections = predict(argument_rows)
    prompts: list[str] = []
    selected_identities: list[str] = []
    selected_tokens: list[str] = []
    for row, selection in zip(argument_rows, argument_selections):
        if args.information_condition == "full_document_oracle":
            identity = str(row["reference_tools"][0])
            token = str(mapping[identity]["fixed_token"])
        else:
            identity = str(selection["identities"][0])
            token = str(selection["tokens"][0])
        selected_identities.append(identity)
        selected_tokens.append(token)
        document = tools[identity].document if identity in tools else None
        prompts.append(
            argument_prompt(
                str(row["query"]),
                selected_token=token,
                selected_document=document,
                information_condition=args.information_condition,
            )
        )
    argument_generations = generate_arguments(
        argument_model,
        argument_tokenizer,
        prompts,
        batch_size=args.generation_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    for row, selection, identity, token, generated in zip(
        argument_rows,
        argument_selections,
        selected_identities,
        selected_tokens,
        argument_generations,
    ):
        predictions.append(
            {
                **row,
                "predicted_tools": [identity],
                "predicted_fixed_tokens": [token],
                "predicted_arguments": generated.strip(),
                "information_condition": args.information_condition,
                "common_document_agent_model_path": (
                    common_document_agent["model_path"]
                    if common_document_agent is not None
                    else None
                ),
                "common_document_agent_audit_sha256": (
                    common_document_agent["audit_sha256"]
                    if common_document_agent is not None
                    else None
                ),
                **(
                    {
                        "registry_size": selection["registry_size"],
                        "registry_identity_sha256": selection[
                            "registry_identity_sha256"
                        ],
                    }
                    if args.registry_size
                    else {}
                ),
            }
        )
    elapsed = time.perf_counter() - started

    predictions.sort(key=lambda row: (str(row["family"]), str(row["condition"]), str(row["example_id"])))
    predictions_path = output / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    result = {
        "kind": "qwen_toolgen_fixed",
        "code_hashes": {
            "evaluate_fixed_tokens": _sha256_file(Path(__file__)),
            "benchmark_metrics": _sha256_file(
                Path(str(benchmark_metrics_module.__file__))
            ),
        },
        "model_path": str(Path(args.model_path).resolve()),
        "common_document_agent": common_document_agent,
        "common_document_agent_config_sha256": (
            _sha256_file(common_document_agent_path / "config.json")
            if common_document_agent_path is not None
            else None
        ),
        "benchmark_manifest_sha256": _sha256_file(benchmark / "benchmark_manifest.json"),
        "fixed_token_mapping_sha256": _sha256_file(mapping_path),
        "candidate_splits": sorted(set(args.token_split)),
        "candidate_count": len(candidate_ids),
        "registry_size": args.registry_size or len(candidate_ids),
        "registry_seed": args.registry_seed,
        "registry_scope": args.registry_scope if args.registry_size else None,
        "information_condition": args.information_condition,
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
        description="Evaluate a Qwen ToolGen-style fixed-token checkpoint"
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--common-document-agent-model-path",
        help=(
            "Shared downstream document-conditioned Agent required by the "
            "common_document information condition."
        ),
    )
    parser.add_argument(
        "--common-document-agent-audit-path",
        help="Checkpoint audit belonging to the shared common-document Agent.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-split", action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument(
        "--information-condition",
        choices=(
            "native",
            "common_document",
            "token_memory_only",
            "full_document_oracle",
        ),
        default="native",
    )
    parser.add_argument("--retrieval-batch-size", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-query-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-examples-per-family", type=int, default=0)
    parser.add_argument(
        "--registry-size",
        type=int,
        default=0,
        help="Deterministic per-example candidate registry; zero uses the full split.",
    )
    parser.add_argument("--registry-seed", type=int, default=17)
    parser.add_argument(
        "--registry-scope",
        choices=("per_example", "shared"),
        default="per_example",
        help="Use one candidate registry across all selected rows.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.token_split = args.token_split or ["train"]
    args.condition = args.condition or ["seen_tool_seen_token"]
    args.family = args.family or ["retrieval", "arguments"]
    result = evaluate_fixed_checkpoint(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
