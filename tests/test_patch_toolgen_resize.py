import tempfile
import unittest
from pathlib import Path

from latent_register.patch_toolgen_resize import ORIGINAL, PATCHED, patch_toolgen_resize


class PatchToolGenResizeTests(unittest.TestCase):
    def test_patches_exactly_one_resize_call_and_records_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loading.py"
            path.write_text(f"before\n    {ORIGINAL}\nafter\n", encoding="utf-8")

            audit = patch_toolgen_resize(path)

            self.assertIn(PATCHED, path.read_text(encoding="utf-8"))
            self.assertNotEqual(audit["sha256_before"], audit["sha256_after"])

    def test_rejects_already_patched_or_unexpected_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loading.py"
            path.write_text(PATCHED, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one unpatched"):
                patch_toolgen_resize(path)


if __name__ == "__main__":
    unittest.main()
