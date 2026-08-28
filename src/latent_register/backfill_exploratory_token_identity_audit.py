from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .physical_tokens import token_identity_audit, validate_token_identity_audit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def backfill_exploratory_identity_audit(
    *, training_results_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Backfill reproducible token identity fields without altering source evidence."""
    from transformers import AutoTokenizer

    source = Path(training_results_path)
    output = Path(output_path)
    if source.resolve() == output.resolve():
        raise ValueError("The compatibility audit must not overwrite training results")
    payload = json.loads(source.read_text(encoding="utf-8"))
    audit = payload.get("physical_token_audit")
    if not isinstance(audit, dict):
        raise ValueError("Training results are missing physical_token_audit")
    for key in (
        "evaluation_ids_seen_during_training",
        "evaluation_input_row_max_change",
        "evaluation_output_row_max_change",
    ):
        if key not in audit or float(audit[key]) != 0.0:
            raise ValueError(f"Cannot backfill a failed or incomplete isolation audit: {key}")
    total = int(audit.get("total_reserved_tokens", 0))
    original_vocab_size = int(audit.get("original_vocab_size", -1))
    if total < 1 or original_vocab_size < 1:
        raise ValueError("Training audit has invalid reserved-token counts")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if len(tokenizer) != original_vocab_size:
        raise ValueError(
            f"Base tokenizer size {len(tokenizer)} does not match audit "
            f"original_vocab_size {original_vocab_size}"
        )
    token_strings = [f"<|latent_tool_slot_{index:05d}|>" for index in range(total)]
    added = tokenizer.add_special_tokens({"additional_special_tokens": token_strings})
    if added != total:
        raise ValueError(f"Expected {total} new reserved tokens, added {added}")
    token_ids = list(tokenizer.convert_tokens_to_ids(token_strings))
    encodings = tokenizer(token_strings, add_special_tokens=False).input_ids
    if any(encoded != [token_id] for encoded, token_id in zip(encodings, token_ids)):
        raise ValueError("At least one reconstructed reserved token is not atomic")

    reconstructed = token_identity_audit(token_strings, token_ids)
    overlap = set(reconstructed).intersection(audit)
    mismatched = {
        key: (audit[key], reconstructed[key])
        for key in overlap
        if audit[key] != reconstructed[key]
    }
    if mismatched:
        raise ValueError(f"Existing identity evidence disagrees with reconstruction: {mismatched}")
    audit.update(reconstructed)
    validated = validate_token_identity_audit(audit)
    if not validated["passed"]:
        raise AssertionError(f"Reconstructed token identity audit failed: {validated}")

    payload["physical_token_audit"] = audit
    payload["exploratory_compatibility_audit"] = {
        "kind": "retrospective_reserved_token_identity_backfill",
        "source_training_results": str(source.resolve()),
        "source_training_results_sha256": _sha256_file(source),
        "model_path": str(Path(model_path).resolve()),
        "formal_evidence_eligible": False,
        "reason": (
            "The historical run predates mandatory token identity hash logging; "
            "the mapping was reconstructed from its pinned base tokenizer and counts."
        ),
        "validated_identity": validated,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a non-formal compatibility copy with token identity evidence"
    )
    parser.add_argument("--training-results", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = backfill_exploratory_identity_audit(
        training_results_path=args.training_results,
        model_path=args.model_path,
        output_path=args.output,
    )
    print(json.dumps(payload["exploratory_compatibility_audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
