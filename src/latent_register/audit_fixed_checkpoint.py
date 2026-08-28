from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .validate_toolgen_tokenizer import (
    audit_existing_token_mapping,
    sha256_file,
    write_json_atomic,
)


def validate_reference_mapping(
    audit: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    for key in (
        "normalized_token_count",
        "normalized_tokens_sha256",
        "token_mapping_sha256",
        "tokenizer_size",
        "first_tool_token_id",
        "last_tool_token_id",
    ):
        if audit.get(key) != reference.get(key):
            raise ValueError(
                f"Checkpoint token mapping differs from reference for {key}: "
                f"{audit.get(key)!r} != {reference.get(key)!r}"
            )


def validate_reference_subset_mapping(
    audit: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    """Require an old token subset to retain exactly the same physical IDs."""
    for key in (
        "normalized_token_count",
        "normalized_tokens_sha256",
        "token_mapping_sha256",
        "first_tool_token_id",
        "last_tool_token_id",
    ):
        if audit.get(key) != reference.get(key):
            raise ValueError(
                f"Checkpoint token subset differs from reference for {key}: "
                f"{audit.get(key)!r} != {reference.get(key)!r}"
            )


def _all_finite_in_chunks(weight: torch.Tensor, *, chunk_size: int = 4096) -> bool:
    for start in range(0, weight.shape[0], chunk_size):
        if not torch.isfinite(weight[start : start + chunk_size]).all().item():
            return False
    return True


def audit_fixed_checkpoint(
    model_path: str | Path,
    tokens_path: str | Path,
    *,
    expected_count: int | None = None,
    reference_audit: Mapping[str, Any] | None = None,
    reference_subset_audit: Mapping[str, Any] | None = None,
    load_model: bool = False,
    require_suffix_range: bool = True,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from unidecode import unidecode

    model_path = Path(model_path)
    tokens_path = Path(tokens_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokens = tokens_path.read_text(encoding="utf-8").splitlines()
    audit = audit_existing_token_mapping(
        tokenizer,
        tokens,
        transliterate=unidecode,
        expected_count=expected_count,
        require_suffix_range=require_suffix_range,
    )
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config_vocab_size = int(config.vocab_size)
    if config_vocab_size != len(tokenizer):
        raise ValueError(
            f"Model config vocabulary {config_vocab_size} differs from tokenizer "
            f"size {len(tokenizer)}"
        )
    audit.update(
        {
            "kind": "qwen_toolgen_fixed_checkpoint_audit",
            "model_path": str(model_path.resolve()),
            "model_type": config.model_type,
            "config_vocab_size": config_vocab_size,
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "token_file_sha256": sha256_file(tokens_path),
            "full_model_reloaded": False,
        }
    )
    if reference_audit is not None:
        validate_reference_mapping(audit, reference_audit)
        audit["reference_mapping_match"] = True
    if reference_subset_audit is not None:
        validate_reference_subset_mapping(audit, reference_subset_audit)
        audit["reference_subset_mapping_match"] = True

    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        input_weight = model.get_input_embeddings().weight.detach()
        output_module = model.get_output_embeddings()
        if output_module is None:
            raise ValueError("Checkpoint has no output embedding module")
        output_weight = output_module.weight.detach()
        if input_weight.shape[0] != len(tokenizer):
            raise ValueError("Input embedding row count differs from tokenizer size")
        if output_weight.shape[0] != len(tokenizer):
            raise ValueError("Output embedding row count differs from tokenizer size")
        input_finite = _all_finite_in_chunks(input_weight)
        output_finite = _all_finite_in_chunks(output_weight)
        if not input_finite or not output_finite:
            raise ValueError("Checkpoint embedding tables contain non-finite values")
        audit.update(
            {
                "full_model_reloaded": True,
                "input_embedding_shape": list(input_weight.shape),
                "output_embedding_shape": list(output_weight.shape),
                "input_embeddings_all_finite": input_finite,
                "output_embeddings_all_finite": output_finite,
            }
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a trained Qwen ToolGen-style fixed-token checkpoint"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--reference-audit")
    parser.add_argument("--reference-subset-audit")
    parser.add_argument("--allow-non-suffix-range", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference = (
        json.loads(Path(args.reference_audit).read_text(encoding="utf-8"))
        if args.reference_audit
        else None
    )
    reference_subset = (
        json.loads(Path(args.reference_subset_audit).read_text(encoding="utf-8"))
        if args.reference_subset_audit
        else None
    )
    audit = audit_fixed_checkpoint(
        args.model,
        args.tokens,
        expected_count=args.expected_count,
        reference_audit=reference,
        reference_subset_audit=reference_subset,
        load_model=args.load_model,
        require_suffix_range=not args.allow_non_suffix_range,
    )
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
