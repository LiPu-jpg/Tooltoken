"""Apply the audited Transformers compatibility patch required by ToolGen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ORIGINAL = "model.resize_token_embeddings(len(tokenizer))"
PATCHED = "model.resize_token_embeddings(len(tokenizer), mean_resizing=False)"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def patch_toolgen_resize(path: Path) -> dict[str, object]:
    before = path.read_bytes()
    text = before.decode("utf-8")
    original_count = text.count(ORIGINAL)
    patched_count = text.count(PATCHED)

    if original_count != 1 or patched_count != 0:
        raise ValueError(
            "Expected exactly one unpatched ToolGen resize call; "
            f"found original={original_count}, patched={patched_count}"
        )

    updated = text.replace(ORIGINAL, PATCHED, 1).encode("utf-8")
    path.write_bytes(updated)
    return {
        "target": str(path),
        "reason": (
            "Restore the pre-mean_resizing Transformers behavior used by ToolGen; "
            "ToolGen overwrites input rows with semantic means after resizing."
        ),
        "replacement": {"before": ORIGINAL, "after": PATCHED},
        "sha256_before": _sha256(before),
        "sha256_after": _sha256(updated),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = patch_toolgen_resize(args.input)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
