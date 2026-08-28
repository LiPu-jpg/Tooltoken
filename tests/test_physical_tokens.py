import unittest

from latent_register.physical_tokens import (
    token_identity_audit,
    validate_token_identity_audit,
)


class PhysicalTokenIdentityAuditTests(unittest.TestCase):
    def test_records_atomic_unique_contiguous_mapping(self) -> None:
        audit = {
            "total_reserved_tokens": 3,
            **token_identity_audit(["<slot-0>", "<slot-1>", "<slot-2>"], [8, 9, 10]),
        }
        evidence = validate_token_identity_audit(audit)
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["distinct_reserved_token_ids"], 3)
        self.assertEqual(len(evidence["reserved_token_id_mapping_sha256"]), 64)

    def test_rejects_duplicate_or_noncontiguous_ids(self) -> None:
        duplicate = {
            "total_reserved_tokens": 3,
            **token_identity_audit(["a", "b", "c"], [8, 8, 9]),
        }
        noncontiguous = {
            "total_reserved_tokens": 3,
            **token_identity_audit(["a", "b", "c"], [8, 9, 11]),
        }
        self.assertFalse(validate_token_identity_audit(duplicate)["passed"])
        self.assertFalse(validate_token_identity_audit(noncontiguous)["passed"])


if __name__ == "__main__":
    unittest.main()
