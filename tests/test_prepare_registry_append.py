import unittest

from latent_register.prepare_registry_append import build_append_registries


class PrepareRegistryAppendTests(unittest.TestCase):
    def test_builds_a_true_prefix_preserving_append(self) -> None:
        rows = [
            {
                "family": "retrieval",
                "condition": "unseen_tool_unseen_token",
                "example_id": "one",
                "reference_tools": ["a"],
            },
            {
                "family": "arguments",
                "condition": "unseen_tool_unseen_token",
                "example_id": "two",
                "reference_tools": ["b"],
            },
        ]
        initial, extended, audits = build_append_registries(
            rows,
            ["a", "b", "c", "d", "e", "f"],
            range(10, 30),
            initial_size=4,
            extended_size=6,
            seed=17,
        )
        self.assertEqual(extended.identities[:4], initial.identities)
        self.assertEqual(extended.address_slots[:4], initial.address_slots)
        self.assertTrue(audits["old_identity_prefix_unchanged"])
        self.assertTrue(audits["old_address_prefix_unchanged"])
        self.assertEqual(audits["target_count"], 2)


if __name__ == "__main__":
    unittest.main()
