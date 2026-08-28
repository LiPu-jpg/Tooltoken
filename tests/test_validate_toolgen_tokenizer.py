import unittest

from latent_register.validate_toolgen_tokenizer import (
    audit_existing_token_mapping,
    audit_loaded_tokenizer,
    audit_mapping_coverage,
    normalize_tokens,
)


class Encoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    def __init__(self, existing=None):
        self.vocab = {"base-a": 0, "base-b": 1}
        for token in existing or []:
            self.vocab[token] = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def add_tokens(self, new_tokens, special_tokens=False):
        del special_tokens
        added = 0
        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.vocab:
            return Encoding([self.vocab[text]])
        return Encoding([0 for _ in text.split()])


class ValidateToolGenTokenizerTest(unittest.TestCase):
    def test_allows_explicit_finish_control_token(self):
        audit = audit_loaded_tokenizer(
            FakeTokenizer(),
            ["<<Weather&&forecast>>", "<<Finish>>"],
            transliterate=lambda value: value,
            expected_count=2,
            allowed_control_tokens=["<<Finish>>"],
        )
        self.assertEqual(audit["allowed_control_token_count"], 1)

    def test_rejects_undeclared_finish_control_token(self):
        with self.assertRaisesRegex(ValueError, "compound token"):
            audit_loaded_tokenizer(
                FakeTokenizer(),
                ["<<Weather&&forecast>>", "<<Finish>>"],
                transliterate=lambda value: value,
            )

    def test_allocates_a_contiguous_bijection(self):
        audit = audit_loaded_tokenizer(
            FakeTokenizer(),
            ["<<Weather&&forecast>>", "<<Maps&&route>>"],
            transliterate=lambda value: value,
            expected_count=2,
        )
        self.assertEqual(audit["new_rows_added"], 2)
        self.assertEqual(audit["first_tool_token_id"], 2)
        self.assertEqual(audit["last_tool_token_id"], 3)
        self.assertTrue(audit["mapping_is_bijective"])

    def test_rejects_normalization_collisions(self):
        with self.assertRaisesRegex(ValueError, "collide after unidecode"):
            normalize_tokens(
                ["<<A&&x>>", "<<B&&x>>"], lambda value: "<<same&&x>>"
            )

    def test_rejects_token_already_in_base_vocabulary(self):
        token = "<<Weather&&forecast>>"
        with self.assertRaisesRegex(ValueError, "newly allocated rows"):
            audit_loaded_tokenizer(
                FakeTokenizer(existing=[token]),
                [token],
                transliterate=lambda value: value,
                expected_count=1,
            )

    def test_rejects_malformed_compound_token(self):
        with self.assertRaisesRegex(ValueError, "compound token"):
            audit_loaded_tokenizer(
                FakeTokenizer(),
                ["<<missing-separator>>"],
                transliterate=lambda value: value,
                expected_count=1,
            )

    def test_audits_existing_contiguous_suffix_mapping(self):
        tokens = ["<<Weather&&forecast>>", "<<Maps&&route>>"]
        audit = audit_existing_token_mapping(
            FakeTokenizer(existing=tokens),
            tokens,
            transliterate=lambda value: value,
            expected_count=2,
        )
        self.assertEqual(audit["first_tool_token_id"], 2)
        self.assertEqual(audit["last_tool_token_id"], 3)
        self.assertTrue(audit["mapping_is_contiguous_suffix_range"])

    def test_rejects_existing_mapping_outside_suffix(self):
        tokens = ["<<Weather&&forecast>>", "<<Maps&&route>>"]
        tokenizer = FakeTokenizer(existing=[tokens[0], "ordinary", tokens[1]])
        with self.assertRaisesRegex(ValueError, "suffix range"):
            audit_existing_token_mapping(
                tokenizer,
                tokens,
                transliterate=lambda value: value,
                expected_count=2,
            )

    def test_audits_retrieval_mapping_coverage(self):
        audit = audit_mapping_coverage(
            ["<<A&&one>>", "<<B&&two>>", "<<Finish>>"],
            ["<<A&&one>>", "<<B&&two>>"],
            transliterate=lambda value: value,
        )
        self.assertEqual(audit["allocated_token_count"], 3)
        self.assertEqual(audit["mapped_token_count"], 2)
        self.assertEqual(audit["unmapped_token_count"], 1)
        self.assertTrue(audit["finish_control_token_is_unmapped"])

    def test_rejects_retrieval_token_outside_allocation(self):
        with self.assertRaisesRegex(ValueError, "outside the allocation"):
            audit_mapping_coverage(
                ["<<A&&one>>"],
                ["<<B&&two>>"],
                transliterate=lambda value: value,
            )


if __name__ == "__main__":
    unittest.main()
