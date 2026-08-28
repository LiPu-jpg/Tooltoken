from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class ControlledRegistry:
    identities: tuple[str, ...]
    positive_positions: tuple[int, ...]
    address_slots: tuple[int, ...]
    identity_sha256: str


def _seed_value(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_controlled_registry(
    candidate_identities: Sequence[str],
    target_identities: Sequence[str],
    *,
    registry_size: int,
    seed: int,
    key: str,
    address_pool: range | None = None,
) -> ControlledRegistry:
    if registry_size < 1:
        raise ValueError("Registry size must be positive")
    candidates = tuple(sorted(set(candidate_identities)))
    targets = tuple(dict.fromkeys(target_identities))
    if not targets:
        raise ValueError("A controlled registry requires at least one target")
    candidate_set = set(candidates)
    missing = sorted(set(targets) - candidate_set)
    if missing:
        raise ValueError(f"Registry targets are absent from candidates: {missing[:3]}")
    final_size = max(registry_size, len(targets))
    if final_size > len(candidates):
        raise ValueError(
            f"Registry size {final_size} exceeds {len(candidates)} candidates"
        )
    if address_pool is not None and final_size > len(address_pool):
        raise ValueError(
            f"Registry size {final_size} exceeds {len(address_pool)} addresses"
        )

    generator = random.Random(_seed_value(seed, key))
    selected = set(targets)
    identities = list(targets)
    needed = final_size - len(identities)
    if final_size * 2 < len(candidates):
        while len(identities) < final_size:
            identity = candidates[generator.randrange(len(candidates))]
            if identity in selected:
                continue
            selected.add(identity)
            identities.append(identity)
    else:
        distractors = [identity for identity in candidates if identity not in selected]
        identities.extend(generator.sample(distractors, needed))
    generator.shuffle(identities)
    target_set = set(targets)
    positive_positions = tuple(
        index for index, identity in enumerate(identities) if identity in target_set
    )
    slots = (
        tuple(generator.sample(address_pool, final_size))
        if address_pool is not None
        else tuple()
    )
    payload = "\n".join(identities).encode("utf-8")
    return ControlledRegistry(
        identities=tuple(identities),
        positive_positions=positive_positions,
        address_slots=slots,
        identity_sha256=hashlib.sha256(payload).hexdigest(),
    )


def build_controlled_registries(
    candidate_identities: Sequence[str],
    target_identities_by_example: Sequence[Sequence[str]],
    *,
    registry_size: int,
    seed: int,
    keys: Sequence[str],
    address_pool: range | None = None,
    scope: Literal["per_example", "shared"] = "per_example",
) -> list[ControlledRegistry]:
    """Build per-example registries or one shared candidate universe.

    A shared registry retains one immutable identity/address mapping while each
    example keeps its own positive positions. This is the practical protocol
    for 10K/47K evaluation and avoids materializing a distinct 47K-entry
    registry for every query.
    """
    if len(target_identities_by_example) != len(keys):
        raise ValueError("Registry targets and keys must have equal lengths")
    if scope == "per_example":
        return [
            build_controlled_registry(
                candidate_identities,
                targets,
                registry_size=registry_size,
                seed=seed,
                key=key,
                address_pool=address_pool,
            )
            for targets, key in zip(target_identities_by_example, keys)
        ]
    if scope != "shared":
        raise ValueError(f"Unsupported registry scope: {scope}")
    if not target_identities_by_example:
        return []

    all_targets = tuple(
        sorted(
            {
                identity
                for targets in target_identities_by_example
                for identity in targets
            }
        )
    )
    shared = build_controlled_registry(
        candidate_identities,
        all_targets,
        registry_size=registry_size,
        seed=seed,
        key="shared",
        address_pool=address_pool,
    )
    positions = {identity: index for index, identity in enumerate(shared.identities)}
    return [
        ControlledRegistry(
            identities=shared.identities,
            positive_positions=tuple(
                positions[identity] for identity in dict.fromkeys(targets)
            ),
            address_slots=shared.address_slots,
            identity_sha256=shared.identity_sha256,
        )
        for targets in target_identities_by_example
    ]


def extend_controlled_registry(
    registry: ControlledRegistry,
    candidate_identities: Sequence[str],
    target_identities: Sequence[str],
    *,
    registry_size: int,
    seed: int,
    key: str,
    address_pool: range,
) -> ControlledRegistry:
    """Append identities and addresses without changing any existing binding."""
    candidates = tuple(sorted(set(candidate_identities)))
    candidate_set = set(candidates)
    targets = tuple(dict.fromkeys(target_identities))
    required = set(registry.identities) | set(targets)
    missing = sorted(required - candidate_set)
    if missing:
        raise ValueError(f"Registry identities are absent from candidates: {missing[:3]}")
    if registry_size < len(registry.identities):
        raise ValueError("An appended registry cannot shrink")
    final_size = max(registry_size, len(required))
    if final_size > len(candidates):
        raise ValueError(
            f"Registry size {final_size} exceeds {len(candidates)} candidates"
        )
    allowed_slots = set(address_pool)
    if not set(registry.address_slots).issubset(allowed_slots):
        raise ValueError("Existing registry uses an address outside the append pool")
    if final_size > len(address_pool):
        raise ValueError(
            f"Registry size {final_size} exceeds {len(address_pool)} addresses"
        )

    identities = list(registry.identities)
    selected = set(identities)
    identities.extend(identity for identity in targets if identity not in selected)
    selected.update(targets)
    generator = random.Random(_seed_value(seed, key))
    available_identities = [
        identity for identity in candidates if identity not in selected
    ]
    generator.shuffle(available_identities)
    identities.extend(available_identities[: final_size - len(identities)])

    used_slots = set(registry.address_slots)
    available_slots = [slot for slot in address_pool if slot not in used_slots]
    generator.shuffle(available_slots)
    slots = list(registry.address_slots)
    slots.extend(available_slots[: final_size - len(slots)])
    target_set = set(targets)
    positive_positions = tuple(
        index for index, identity in enumerate(identities) if identity in target_set
    )
    payload = "\n".join(identities).encode("utf-8")
    return ControlledRegistry(
        identities=tuple(identities),
        positive_positions=positive_positions,
        address_slots=tuple(slots),
        identity_sha256=hashlib.sha256(payload).hexdigest(),
    )
