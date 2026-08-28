"""Reload and audit a full-document Agent checkpoint after transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import sha256_file, write_json_atomic


AUDIT_KIND = "qwen_full_document_checkpoint_audit"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _provenance_digests(path: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"Malformed checksum line in {path}: {line!r}")
        name = Path(fields[1].lstrip("* ")).name
        if name in digests:
            raise ValueError(f"Duplicate provenance entry for {name}")
        digests[name] = fields[0]
    return digests


def validate_transfer_evidence(
    *,
    model_path: Path,
    source_audit_path: Path,
    manifest_path: Path,
    source_verification_path: Path,
    destination_verification_path: Path,
    provenance_path: Path,
    local_verification_path: Path | None = None,
) -> dict[str, Any]:
    source_audit = _load_json(source_audit_path)
    manifest = _load_json(manifest_path)
    source_verification = _load_json(source_verification_path)
    destination_verification = _load_json(destination_verification_path)
    local_verification = (
        _load_json(local_verification_path)
        if local_verification_path is not None
        else None
    )
    provenance = _provenance_digests(provenance_path)

    expected_files = {
        "checkpoint_audit.json": source_audit_path,
        "model_transfer_manifest.json": manifest_path,
        "source_transfer_verification.json": source_verification_path,
    }
    for name, path in expected_files.items():
        if provenance.get(name) != sha256_file(path):
            raise ValueError(f"Transfer provenance does not bind {name}")

    if source_audit.get("kind") != AUDIT_KIND:
        raise ValueError("Unexpected source checkpoint audit kind")
    if source_audit.get("full_model_reloaded") is not True:
        raise ValueError("Source checkpoint was not fully reloaded")
    if source_audit.get("embedding_tables_all_finite") is not True:
        raise ValueError("Source checkpoint audit found non-finite embeddings")

    tree_digest = manifest.get("tree_sha256")
    if not isinstance(tree_digest, str) or len(tree_digest) != 64:
        raise ValueError("Transfer manifest has an invalid tree digest")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise ValueError("Transfer manifest has no file catalog")
    manifest_by_path = {
        str(entry.get("path")): entry
        for entry in manifest_files
        if isinstance(entry, dict)
    }
    for filename, audit_field in (
        ("config.json", "config_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
    ):
        entry = manifest_by_path.get(filename)
        if entry is None or source_audit.get(audit_field) != entry.get("sha256"):
            raise ValueError(
                f"Source checkpoint audit does not bind manifest {filename}"
            )
    verifications = [
        ("source", source_verification),
        ("destination", destination_verification),
    ]
    if local_verification is not None:
        verifications.insert(1, ("local", local_verification))
    for label, verification in verifications:
        if verification.get("passed") is not True:
            raise ValueError(f"{label} transfer verification did not pass")
        if verification.get("tree_sha256") != tree_digest:
            raise ValueError(f"{label} transfer verification has another tree digest")
        if int(verification.get("file_count", -1)) != int(
            manifest.get("file_count", -2)
        ):
            raise ValueError(f"{label} transfer verification file count differs")
        if int(verification.get("total_bytes", -1)) != int(
            manifest.get("total_bytes", -2)
        ):
            raise ValueError(f"{label} transfer verification byte count differs")

    if Path(str(destination_verification.get("root", ""))).resolve() != model_path.resolve():
        raise ValueError("Destination verification refers to another model path")

    result = {
        "manifest_tree_sha256": tree_digest,
        "manifest_sha256": sha256_file(manifest_path),
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_verification_sha256": sha256_file(source_verification_path),
        "destination_verification_sha256": sha256_file(
            destination_verification_path
        ),
        "provenance_sha256": sha256_file(provenance_path),
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        "source_checkpoint": {
            key: source_audit.get(key)
            for key in (
                "tokenizer_size",
                "config_vocab_size",
                "input_embedding_shape",
                "output_embedding_shape",
                "training_hardware",
                "world_size",
                "effective_global_batch_size",
                "deepspeed_zero_stage",
                "optimizer_offload",
            )
        },
    }
    if local_verification_path is not None:
        result["local_verification_sha256"] = sha256_file(
            local_verification_path
        )
    return result


def audit_checkpoint(
    model_path: Path,
    *,
    transfer: Mapping[str, Any] | None = None,
    expected_tokenizer_size: int | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer_size = len(tokenizer)
    if expected_tokenizer_size is not None and tokenizer_size != expected_tokenizer_size:
        raise ValueError(
            f"Tokenizer size {tokenizer_size} differs from expected "
            f"{expected_tokenizer_size}"
        )
    if int(config.vocab_size) != tokenizer_size:
        raise ValueError("Saved model and tokenizer vocabulary sizes differ")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    input_weight = model.get_input_embeddings().weight.detach()
    output_weight = model.get_output_embeddings().weight.detach()
    if input_weight.shape[0] != tokenizer_size or output_weight.shape[0] != tokenizer_size:
        raise ValueError("Reloaded embedding shape differs from tokenizer size")
    for name, weight in (("input", input_weight), ("output", output_weight)):
        for start in range(0, weight.shape[0], 4096):
            if not torch.isfinite(weight[start : start + 4096]).all().item():
                raise ValueError(f"Non-finite {name} embedding weights")

    audit: dict[str, Any] = {
        "kind": AUDIT_KIND,
        "model_path": str(model_path),
        "full_model_reloaded": True,
        "tokenizer_size": tokenizer_size,
        "config_vocab_size": int(config.vocab_size),
        "input_embedding_shape": list(input_weight.shape),
        "output_embedding_shape": list(output_weight.shape),
        "embedding_tables_all_finite": True,
        "config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
        "audit_hardware": "A100 receiving cluster",
    }
    if transfer is not None:
        audit["transfer"] = dict(transfer)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-size", type=int)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-verification", type=Path)
    parser.add_argument("--destination-verification", type=Path)
    parser.add_argument("--local-verification", type=Path)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()

    evidence_paths = (
        args.source_audit,
        args.manifest,
        args.source_verification,
        args.destination_verification,
        args.provenance,
    )
    if any(evidence_paths) and not all(evidence_paths):
        parser.error("All transfer-evidence arguments must be supplied together")
    transfer = None
    if all(evidence_paths):
        transfer = validate_transfer_evidence(
            model_path=args.model,
            source_audit_path=args.source_audit,
            manifest_path=args.manifest,
            source_verification_path=args.source_verification,
            destination_verification_path=args.destination_verification,
            provenance_path=args.provenance,
            local_verification_path=args.local_verification,
        )
    audit = audit_checkpoint(
        args.model,
        transfer=transfer,
        expected_tokenizer_size=args.expected_tokenizer_size,
    )
    write_json_atomic(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
