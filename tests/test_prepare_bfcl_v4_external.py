from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from latent_register.prepare_bfcl_v4_external import (
    _normalize_schema,
    prepare_bfcl_v4_external,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_normalize_bfcl_types_to_json_schema() -> None:
    schema = _normalize_schema(
        {
            "type": "dict",
            "properties": {
                "distance": {"type": "float"},
                "labels": {"type": "ArrayList", "items": {"type": "String"}},
                "payload": {"type": "any"},
            },
            "required": ["distance"],
        }
    )
    Draft202012Validator.check_schema(schema)
    assert schema["type"] == "object"
    assert schema["properties"]["distance"]["type"] == "number"
    assert schema["properties"]["labels"]["type"] == "array"
    assert "type" not in schema["properties"]["payload"]


def test_prepare_single_call_bfcl_v4_manifest(tmp_path: Path) -> None:
    data_root = tmp_path / "bfcl"
    category = "simple_python"
    row = {
        "id": "simple_python_0",
        "question": [[{"role": "user", "content": "Area, please"}]],
        "function": [
            {
                "name": "triangle_area",
                "description": "Compute an area.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
    }
    answer = {
        "id": "simple_python_0",
        "ground_truth": [
            {"triangle_area": {"base": [10], "height": [5], "unit": [""]}}
        ],
    }
    _write_jsonl(data_root / f"BFCL_v4_{category}.json", [row])
    _write_jsonl(
        data_root / "possible_answer" / f"BFCL_v4_{category}.json", [answer]
    )
    split_manifest = tmp_path / "source_split_manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "token_pools": {
                    "train": {"start_inclusive": 0, "end_exclusive": 4},
                    "validation": {"start_inclusive": 4, "end_exclusive": 6},
                    "test": {"start_inclusive": 6, "end_exclusive": 10},
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    manifest = prepare_bfcl_v4_external(
        data_root=data_root,
        split_manifest_template=split_manifest,
        output_dir=output,
        categories=[category],
        upstream_commit="abc123",
    )

    assert manifest["counts"]["distinct_tools"] == 1
    assert manifest["counts"]["single_call_examples"] == 1
    assert manifest["counts"]["evaluation_rows"] == 2
    assert manifest["audits"]["official_bfcl_score"] is False
    tools = [
        json.loads(line)
        for line in (output / "prepared" / "tools.jsonl").read_text().splitlines()
    ]
    rows = [
        json.loads(line)
        for line in (output / "benchmark" / "benchmark_eval.jsonl")
        .read_text()
        .splitlines()
    ]
    assert tools[0]["split"] == "test"
    assert tools[0]["parameters"]["type"] == "object"
    assert {row["family"] for row in rows} == {"retrieval", "arguments"}
    argument = next(row for row in rows if row["family"] == "arguments")
    assert argument["reference_arguments"] == {"base": 10, "height": 5}
    assert argument["reference_tools"] == [tools[0]["identity_hash"]]
