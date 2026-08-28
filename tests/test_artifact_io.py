import json
import tempfile
import unittest
from pathlib import Path

from latent_register.artifact_io import sha256_file, write_json_atomic


class ArtifactIoTests(unittest.TestCase):
    def test_atomic_write_creates_parent_and_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "artifact.json"
            write_json_atomic(output, {"passed": True, "value": 3})
            payload = json.loads(output.read_text(encoding="utf-8"))
            temporary_files = list(output.parent.glob(f".{output.name}.*.tmp"))
        self.assertEqual(payload, {"passed": True, "value": 3})
        self.assertEqual(temporary_files, [])

    def test_sha256_file_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.bin"
            path.write_bytes(b"abc")
            digest = sha256_file(path)
        self.assertEqual(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )


if __name__ == "__main__":
    unittest.main()
