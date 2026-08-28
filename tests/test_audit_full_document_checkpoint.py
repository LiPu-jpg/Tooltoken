import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from latent_register.audit_full_document_checkpoint import validate_transfer_evidence
from latent_register.checkpoint_transfer_manifest import create_manifest, verify_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuditFullDocumentCheckpointTests(unittest.TestCase):
    def _evidence(self, root: Path) -> dict[str, Path]:
        model = root / "Qwen3-8B-Full-Document"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (model / "weights.safetensors").write_bytes(b"weights")

        manifest_path = root / "model_transfer_manifest.json"
        manifest_path.write_text(
            json.dumps(create_manifest(model), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_verification = root / "source_transfer_verification.json"
        source_verification.write_text(
            json.dumps(verify_manifest(model, manifest), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination_verification = root / "destination_transfer_verification.json"
        destination_verification.write_text(
            json.dumps(verify_manifest(model, manifest), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        local_verification = root / "local_transfer_verification.json"
        local_verification.write_text(
            json.dumps(verify_manifest(model, manifest), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_audit = root / "checkpoint_audit.json"
        source_audit.write_text(
            json.dumps(
                {
                    "kind": "qwen_full_document_checkpoint_audit",
                    "full_model_reloaded": True,
                    "embedding_tables_all_finite": True,
                    "config_sha256": _sha256(model / "config.json"),
                    "tokenizer_sha256": _sha256(model / "tokenizer.json"),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = root / "transfer_provenance.sha256"
        provenance.write_text(
            "".join(
                f"{_sha256(path)}  {path}\n"
                for path in (source_audit, manifest_path, source_verification)
            ),
            encoding="utf-8",
        )
        return {
            "model_path": model,
            "source_audit_path": source_audit,
            "manifest_path": manifest_path,
            "source_verification_path": source_verification,
            "destination_verification_path": destination_verification,
            "local_verification_path": local_verification,
            "provenance_path": provenance,
        }

    def test_accepts_consistent_source_and_destination_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._evidence(Path(directory))
            result = validate_transfer_evidence(**paths)
        self.assertEqual(result["file_count"], 3)
        self.assertEqual(len(result["manifest_tree_sha256"]), 64)
        self.assertEqual(len(result["local_verification_sha256"]), 64)

    def test_rejects_local_staging_for_another_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._evidence(Path(directory))
            local = paths["local_verification_path"]
            payload = json.loads(local.read_text(encoding="utf-8"))
            payload["tree_sha256"] = "0" * 64
            local.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "local.*another tree digest"):
                validate_transfer_evidence(**paths)

    def test_rejects_modified_source_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._evidence(Path(directory))
            paths["source_audit_path"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                validate_transfer_evidence(**paths)

    def test_rejects_destination_for_another_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._evidence(Path(directory))
            destination = paths["destination_verification_path"]
            payload = json.loads(destination.read_text(encoding="utf-8"))
            payload["root"] = str(Path(directory) / "different-model")
            destination.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another model path"):
                validate_transfer_evidence(**paths)

    def test_rejects_source_audit_for_another_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._evidence(Path(directory))
            source_audit = paths["source_audit_path"]
            payload = json.loads(source_audit.read_text(encoding="utf-8"))
            payload["config_sha256"] = "0" * 64
            source_audit.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            provenance = paths["provenance_path"]
            manifest = paths["manifest_path"]
            source_verification = paths["source_verification_path"]
            provenance.write_text(
                "".join(
                    f"{_sha256(path)}  {path}\n"
                    for path in (source_audit, manifest, source_verification)
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not bind manifest"):
                validate_transfer_evidence(**paths)


if __name__ == "__main__":
    unittest.main()
