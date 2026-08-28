import unittest

from latent_register.controlled_registry import (
    build_controlled_registries,
    build_controlled_registry,
    extend_controlled_registry,
)


class ControlledRegistryTests(unittest.TestCase):
    def test_is_deterministic_and_contains_every_target(self) -> None:
        first = build_controlled_registry(
            ["a", "b", "c", "d", "e"],
            ["b", "d"],
            registry_size=4,
            seed=17,
            key="example",
            address_pool=range(10, 20),
        )
        second = build_controlled_registry(
            ["e", "d", "c", "b", "a"],
            ["b", "d"],
            registry_size=4,
            seed=17,
            key="example",
            address_pool=range(10, 20),
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first.identities), 4)
        self.assertEqual({first.identities[index] for index in first.positive_positions}, {"b", "d"})
        self.assertEqual(len(set(first.address_slots)), 4)

    def test_key_rebinds_registry(self) -> None:
        first = build_controlled_registry(
            [str(index) for index in range(100)],
            ["0"],
            registry_size=10,
            seed=17,
            key="first",
        )
        second = build_controlled_registry(
            [str(index) for index in range(100)],
            ["0"],
            registry_size=10,
            seed=17,
            key="second",
        )
        self.assertNotEqual(first.identities, second.identities)

    def test_rejects_insufficient_address_pool(self) -> None:
        with self.assertRaisesRegex(ValueError, "addresses"):
            build_controlled_registry(
                ["a", "b", "c"],
                ["a"],
                registry_size=3,
                seed=17,
                key="example",
                address_pool=range(2),
            )

    def test_shared_registry_reuses_identity_and_address_mapping(self) -> None:
        registries = build_controlled_registries(
            ["a", "b", "c", "d", "e"],
            [["a"], ["c", "d"]],
            registry_size=4,
            seed=17,
            keys=["first", "second"],
            address_pool=range(10, 20),
            scope="shared",
        )
        self.assertIs(registries[0].identities, registries[1].identities)
        self.assertIs(registries[0].address_slots, registries[1].address_slots)
        self.assertEqual(registries[0].identity_sha256, registries[1].identity_sha256)
        self.assertEqual(
            {registries[0].identities[index] for index in registries[0].positive_positions},
            {"a"},
        )
        self.assertEqual(
            {registries[1].identities[index] for index in registries[1].positive_positions},
            {"c", "d"},
        )

    def test_shared_registry_requires_room_for_all_evaluation_targets(self) -> None:
        registries = build_controlled_registries(
            ["a", "b", "c"],
            [["a"], ["b"]],
            registry_size=1,
            seed=17,
            keys=["first", "second"],
            scope="shared",
        )
        self.assertEqual(len(registries[0].identities), 2)

    def test_shared_registry_is_independent_of_example_order(self) -> None:
        first = build_controlled_registries(
            ["a", "b", "c", "d"],
            [["a"], ["b"]],
            registry_size=4,
            seed=17,
            keys=["first", "second"],
            address_pool=range(10, 20),
            scope="shared",
        )
        second = build_controlled_registries(
            ["a", "b", "c", "d"],
            [["b"], ["a"]],
            registry_size=4,
            seed=17,
            keys=["second", "first"],
            address_pool=range(10, 20),
            scope="shared",
        )
        self.assertEqual(first[0].identities, second[0].identities)
        self.assertEqual(first[0].address_slots, second[0].address_slots)

    def test_append_preserves_every_old_identity_address_binding(self) -> None:
        initial = build_controlled_registry(
            ["a", "b", "c", "d", "e", "f"],
            ["a"],
            registry_size=3,
            seed=17,
            key="initial",
            address_pool=range(10, 20),
        )
        extended = extend_controlled_registry(
            initial,
            ["a", "b", "c", "d", "e", "f"],
            ["a", "f"],
            registry_size=6,
            seed=17,
            key="append",
            address_pool=range(10, 20),
        )
        self.assertEqual(
            extended.identities[: len(initial.identities)], initial.identities
        )
        self.assertEqual(
            extended.address_slots[: len(initial.address_slots)],
            initial.address_slots,
        )
        self.assertEqual(len(set(extended.identities)), 6)
        self.assertEqual(len(set(extended.address_slots)), 6)
        self.assertEqual(
            {extended.identities[index] for index in extended.positive_positions},
            {"a", "f"},
        )

    def test_append_is_deterministic_and_cannot_shrink(self) -> None:
        initial = build_controlled_registry(
            ["a", "b", "c", "d"],
            ["a"],
            registry_size=3,
            seed=17,
            key="initial",
            address_pool=range(10, 20),
        )
        first = extend_controlled_registry(
            initial,
            ["a", "b", "c", "d"],
            ["a"],
            registry_size=4,
            seed=17,
            key="append",
            address_pool=range(10, 20),
        )
        second = extend_controlled_registry(
            initial,
            ["d", "c", "b", "a"],
            ["a"],
            registry_size=4,
            seed=17,
            key="append",
            address_pool=range(10, 20),
        )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            extend_controlled_registry(
                initial,
                ["a", "b", "c", "d"],
                ["a"],
                registry_size=2,
                seed=17,
                key="append",
                address_pool=range(10, 20),
            )


if __name__ == "__main__":
    unittest.main()
