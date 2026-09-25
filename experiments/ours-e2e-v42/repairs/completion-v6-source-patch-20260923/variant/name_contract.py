"""Offline wire-name ablation. Not imported by any training/evaluation runtime.

This makes requests agree with the simulator's *declared* parameter names.
It does not establish the real API wire contract or repair argument values.
"""
import copy
import re


def canonical_name(raw):
    name = re.sub(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]", "_", raw)
    name = re.sub(r"_+", "_", name).lower().strip("_")
    if name and name[0].isdigit():
        name = "get_" + name
    return "is_" + name if name in {"from", "class", "return", "false", "true", "id", "and"} else name


def build_mapping(tool, mirror):
    apis = mirror.get("api_info", [])
    if len(apis) != 1:
        raise ValueError("Expected one bound API document")
    api = apis[0]
    if api["name"] != tool["source_binding"]["api_name"]:
        raise ValueError("API identity binding mismatch")
    required = [p["name"] for p in api.get("required_parameters", [])]
    optional = [p["name"] for p in api.get("optional_parameters", [])]
    raw_names = required + optional
    normalized = [canonical_name(n) for n in raw_names]
    if len(set(raw_names)) != len(raw_names) or len(set(normalized)) != len(normalized):
        raise ValueError("Ambiguous parameter-name contract")
    properties = tool["parameters"]["properties"]
    if set(properties) != set(normalized):
        raise ValueError("Full schema property mismatch")
    mapping = dict(zip(normalized, raw_names))
    if {canonical_name(n) for n in required} != set(tool["parameters"].get("required", [])):
        raise ValueError("Required-property mismatch")
    for name, prop in properties.items():
        if prop.get("x-toolbench-original-name") != mapping[name]:
            raise ValueError("Original-name metadata mismatch")
    return mapping


def translate(arguments, mapping):
    if not isinstance(arguments, dict) or set(arguments) - set(mapping):
        raise ValueError("Unknown or non-object arguments")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Non-injective mapping")
    # Only rename top-level keys; preserve nested values, missing fields and types.
    return {mapping[k]: copy.deepcopy(v) for k, v in arguments.items()}
