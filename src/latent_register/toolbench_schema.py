"""Document-derived schema targets and exact field spans, without call answers."""
from __future__ import annotations

from typing import Any, Iterator

from .toolbench_data import ToolSpec, compact

SCHEMA_TASKS = {
    "schema_names": "List the JSON Schema pointers of every named parameter (each properties entry), including nested parameters. Return a JSON array.",
    "schema_types": "Return a JSON object mapping each schema-node pointer with an explicitly declared type to its exact type value. Do not infer missing types.",
    "schema_required": "Return a JSON object mapping each schema-node pointer with a required list to that exact list. Preserve scope; do not assume nested fields are globally required.",
    "schema_enums": "Return a JSON object mapping each schema-node pointer with enum or const to its explicitly declared enum/const constraints. Preserve JSON types.",
    "schema_defaults": "Return a JSON object mapping each schema-node pointer with an explicit default to that default. Missing defaults must not be invented. Preserve null, false and zero.",
}


def pointer(parts: tuple[str, ...]) -> str:
    return "".join("/" + part.replace("~", "~0").replace("/", "~1") for part in parts)


def schema_nodes(schema: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], dict]]:
    if not isinstance(schema, dict):
        return
    yield path, schema
    for keyword in ("properties", "patternProperties", "definitions", "$defs", "dependentSchemas"):
        for name, value in schema.get(keyword, {}).items():
            yield from schema_nodes(value, path + (keyword, name))
    for keyword in ("additionalProperties", "additionalItems", "contains", "not", "if", "then", "else", "propertyNames"):
        yield from schema_nodes(schema.get(keyword), path + (keyword,))
    items = schema.get("items")
    if isinstance(items, dict):
        yield from schema_nodes(items, path + ("items",))
    elif isinstance(items, list):
        for index, value in enumerate(items):
            yield from schema_nodes(value, path + ("items", str(index)))
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for index, value in enumerate(schema.get(keyword, [])):
            yield from schema_nodes(value, path + (keyword, str(index)))
    for name, value in schema.get("dependencies", {}).items():
        if isinstance(value, dict):
            yield from schema_nodes(value, path + ("dependencies", name))


def schema_targets(schema: dict) -> dict[str, Any]:
    targets: dict[str, Any] = {name: {} for name in SCHEMA_TASKS}
    targets["schema_names"] = []
    for path, node in schema_nodes(schema):
        here = pointer(path)
        for name in node.get("properties", {}):
            targets["schema_names"].append(pointer(path + ("properties", name)))
        for keyword, task in (("type", "schema_types"), ("required", "schema_required"), ("default", "schema_defaults")):
            if keyword in node:
                targets[task][here] = node[keyword]
        constraints = {key: node[key] for key in ("enum", "const") if key in node}
        if constraints:
            targets["schema_enums"][here] = constraints
    targets["schema_names"].sort()
    return targets


def registration_layout(tool: ToolSpec) -> tuple[str, list[tuple[str, int, int]]]:
    """Serialize identically to ToolSpec, recording each property key+subtree.

    Pointers are schema locations, not instance paths. References remain literal:
    definitions are traversed once, with no recursive reference expansion.
    """
    property_containers = {path + ("properties",) for path, node in schema_nodes(tool.parameters) if "properties" in node}
    chunks: list[str] = []
    spans: list[tuple[str, int, int]] = []
    position = 0
    def emit(text: str) -> None:
        nonlocal position
        chunks.append(text)
        position += len(text)
    def write(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            emit("{")
            for index, (key, item) in enumerate(value.items()):
                if index:
                    emit(",")
                start = position
                emit(compact(key) + ":")
                write(item, path + (key,))
                if path in property_containers:
                    spans.append((pointer(path + (key,)), start, position))
            emit("}")
        elif isinstance(value, list):
            emit("[")
            for index, item in enumerate(value):
                if index:
                    emit(",")
                write(item, path + (str(index),))
            emit("]")
        else:
            emit(compact(value))
    emit(tool.document + "\nParameters JSON Schema:\n")
    write(tool.parameters, ())
    document = "".join(chunks)
    if document != tool.registration_document:
        raise AssertionError("Schema span serialization changed the registered document")
    return document, sorted(spans)


def canonical(value: Any) -> str:
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def target_facts(task: str, value: Any) -> set[str]:
    if task == "schema_names":
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("Expected an array of schema pointers")
        return {canonical(item) for item in value}
    if not isinstance(value, dict):
        raise ValueError("Expected a schema fact object")
    if task == "schema_required":
        if any(not isinstance(items, list) or any(not isinstance(item, str) for item in items) for items in value.values()):
            raise ValueError("Expected scoped required-name arrays")
        return {canonical([path, item]) for path, items in value.items() for item in items}
    return {canonical([path, item]) for path, item in value.items()}
