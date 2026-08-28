from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def duplicate_values(values: Sequence[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def normalize_tokens(
    raw_tokens: Sequence[str], transliterate: Callable[[str], str]
) -> list[str]:
    if not raw_tokens:
        raise ValueError("ToolGen token file is empty")
    if any(not token for token in raw_tokens):
        raise ValueError("ToolGen token file contains an empty token")
    raw_duplicates = duplicate_values(raw_tokens)
    if raw_duplicates:
        raise ValueError(f"Duplicate raw ToolGen tokens: {raw_duplicates[:5]}")

    normalized = [transliterate(token).strip() for token in raw_tokens]
    if any(not token for token in normalized):
        raise ValueError("ToolGen normalization produced an empty token")
    normalized_duplicates = duplicate_values(normalized)
    if normalized_duplicates:
        raise ValueError(
            "ToolGen tokens collide after unidecode normalization: "
            f"{normalized_duplicates[:5]}"
        )
    return normalized


def compound_parts(token: str) -> list[str]:
    if not token.startswith("<<") or not token.endswith(">>"):
        raise ValueError(f"Invalid ToolGen atomic token syntax: {token!r}")
    parts = token[2:-2].split("&&")
    if len(parts) < 2 or any(not part.strip() for part in parts):
        raise ValueError(f"Invalid ToolGen compound token: {token!r}")
    return parts


def audit_loaded_tokenizer(
    tokenizer: Any,
    raw_tokens: Sequence[str],
    *,
    transliterate: Callable[[str], str],
    expected_count: int | None = None,
    allowed_control_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    normalized = normalize_tokens(raw_tokens, transliterate)
    normalized_controls = (
        set(normalize_tokens(allowed_control_tokens, transliterate))
        if allowed_control_tokens
        else set()
    )
    unknown_controls = normalized_controls - set(normalized)
    if unknown_controls:
        raise ValueError(
            "Allowed control tokens are absent from the allocation: "
            f"{sorted(unknown_controls)}"
        )
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"Expected {expected_count} ToolGen tokens, found {len(normalized)}"
        )

    base_size = len(tokenizer)
    added_count = tokenizer.add_tokens(new_tokens=normalized, special_tokens=False)
    if added_count != len(normalized):
        raise ValueError(
            f"Expected {len(normalized)} newly allocated rows, tokenizer added "
            f"only {added_count}"
        )

    mapping: dict[str, int] = {}
    component_lengths: list[int] = []
    for token in normalized:
        token_ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Non-atomic ToolGen token after registration: {token!r}")
        mapping[token] = int(token_ids[0])

        if token in normalized_controls:
            if not token.startswith("<<") or not token.endswith(">>") or not token[2:-2]:
                raise ValueError(f"Invalid ToolGen control token syntax: {token!r}")
            parts = [token[2:-2]]
        else:
            parts = compound_parts(token)
        component_ids = tokenizer(
            " ".join(parts), add_special_tokens=False
        ).input_ids
        if not component_ids:
            raise ValueError(f"Empty initialization decomposition: {token!r}")
        component_lengths.append(len(component_ids))

    ids = list(mapping.values())
    if len(set(ids)) != len(normalized):
        raise ValueError("ToolGen tokenizer mapping is not one-to-one")
    expected_ids = list(range(base_size, base_size + len(normalized)))
    if sorted(ids) != expected_ids:
        raise ValueError("ToolGen rows are not a contiguous newly allocated range")
    if len(tokenizer) != base_size + len(normalized):
        raise ValueError("Tokenizer length does not match the allocated ToolGen rows")

    return {
        "version": 1,
        "raw_token_count": len(raw_tokens),
        "normalized_token_count": len(normalized),
        "base_tokenizer_size": base_size,
        "final_tokenizer_size": len(tokenizer),
        "new_rows_added": added_count,
        "first_tool_token_id": min(ids),
        "last_tool_token_id": max(ids),
        "mapping_is_bijective": True,
        "mapping_is_contiguous_new_range": True,
        "allowed_control_token_count": len(normalized_controls),
        "component_token_length_min": min(component_lengths),
        "component_token_length_max": max(component_lengths),
        "normalized_tokens_sha256": canonical_sha256(normalized),
        "token_mapping_sha256": canonical_sha256(mapping),
    }


def audit_existing_token_mapping(
    tokenizer: Any,
    raw_tokens: Sequence[str],
    *,
    transliterate: Callable[[str], str],
    expected_count: int | None = None,
    require_suffix_range: bool = True,
) -> dict[str, Any]:
    normalized = normalize_tokens(raw_tokens, transliterate)
    if expected_count is not None and len(normalized) != expected_count:
        raise ValueError(
            f"Expected {expected_count} ToolGen tokens, found {len(normalized)}"
        )

    mapping: dict[str, int] = {}
    for token in normalized:
        token_ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Non-atomic existing ToolGen token: {token!r}")
        mapping[token] = int(token_ids[0])

    ids = list(mapping.values())
    if len(set(ids)) != len(normalized):
        raise ValueError("Existing ToolGen tokenizer mapping is not one-to-one")
    expected_ids = list(range(len(tokenizer) - len(normalized), len(tokenizer)))
    suffix_range = sorted(ids) == expected_ids
    if require_suffix_range and not suffix_range:
        raise ValueError("Existing ToolGen rows are not the tokenizer suffix range")

    return {
        "version": 1,
        "raw_token_count": len(raw_tokens),
        "normalized_token_count": len(normalized),
        "tokenizer_size": len(tokenizer),
        "first_tool_token_id": min(ids),
        "last_tool_token_id": max(ids),
        "mapping_is_bijective": True,
        "mapping_is_contiguous_suffix_range": suffix_range,
        "normalized_tokens_sha256": canonical_sha256(normalized),
        "token_mapping_sha256": canonical_sha256(mapping),
    }


def audit_mapping_coverage(
    full_tokens: Sequence[str],
    mapped_tokens: Sequence[str],
    *,
    transliterate: Callable[[str], str],
) -> dict[str, Any]:
    full = normalize_tokens(full_tokens, transliterate)
    mapped = normalize_tokens(mapped_tokens, transliterate)
    full_set = set(full)
    mapped_set = set(mapped)
    extras = sorted(mapped_set - full_set)
    if extras:
        raise ValueError(
            f"Retrieval mapping contains tokens outside the allocation: {extras[:5]}"
        )
    missing = sorted(full_set - mapped_set)
    return {
        "allocated_token_count": len(full),
        "mapped_token_count": len(mapped),
        "unmapped_token_count": len(missing),
        "mapping_coverage": len(mapped) / len(full),
        "mapped_tokens_are_all_allocated": True,
        "finish_control_token_is_unmapped": "<<Finish>>" in missing,
        "unmapped_tokens_sha256": canonical_sha256(missing),
        "unmapped_token_examples": missing[:20],
    }


def audit_tokenizer(
    model_path: str | Path,
    tokens_path: str | Path,
    *,
    expected_count: int | None = None,
    allowed_control_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer
    from unidecode import unidecode

    model_path = Path(model_path)
    tokens_path = Path(tokens_path)
    raw_tokens = tokens_path.read_text(encoding="utf-8").splitlines()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    audit = audit_loaded_tokenizer(
        tokenizer,
        raw_tokens,
        transliterate=unidecode,
        expected_count=expected_count,
        allowed_control_tokens=allowed_control_tokens,
    )
    config = AutoConfig.from_pretrained(model_path)
    audit.update(
        {
            "model_path": str(model_path.resolve()),
            "model_type": config.model_type,
            "architectures": list(getattr(config, "architectures", []) or []),
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "token_file_sha256": sha256_file(tokens_path),
        }
    )
    return audit


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ToolGen token allocation before model training"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--allow-control-token", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = audit_tokenizer(
        args.model,
        args.tokens,
        expected_count=args.expected_count,
        allowed_control_tokens=args.allow_control_token,
    )
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
