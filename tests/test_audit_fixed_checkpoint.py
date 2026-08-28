import unittest

from latent_register.audit_fixed_checkpoint import (
    validate_reference_mapping,
    validate_reference_subset_mapping,
)


class AuditFixedCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = {
            "normalized_token_count": 3,
            "normalized_tokens_sha256": "tokens",
            "token_mapping_sha256": "mapping",
            "tokenizer_size": 10,
            "first_tool_token_id": 7,
            "last_tool_token_id": 9,
        }

    def test_accepts_identical_mapping(self) -> None:
        validate_reference_mapping(self.audit, dict(self.audit))

    def test_rejects_changed_physical_token_ids(self) -> None:
        reference = dict(self.audit)
        reference["token_mapping_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "token_mapping_sha256"):
            validate_reference_mapping(self.audit, reference)

    def test_rejects_changed_tokenizer_size(self) -> None:
        reference = dict(self.audit)
        reference["tokenizer_size"] = 11
        with self.assertRaisesRegex(ValueError, "tokenizer_size"):
            validate_reference_mapping(self.audit, reference)

    def test_subset_mapping_allows_larger_tokenizer(self) -> None:
        expanded = dict(self.audit)
        expanded["tokenizer_size"] = 20
        expanded["mapping_is_contiguous_suffix_range"] = False
        validate_reference_subset_mapping(expanded, self.audit)

    def test_subset_mapping_rejects_changed_old_ids(self) -> None:
        expanded = dict(self.audit)
        expanded["token_mapping_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "token_mapping_sha256"):
            validate_reference_subset_mapping(expanded, self.audit)


if __name__ == "__main__":
    unittest.main()
