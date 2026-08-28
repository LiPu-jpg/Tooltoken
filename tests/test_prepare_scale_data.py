import json
import tempfile
import unittest
from pathlib import Path

from latent_register.prepare_scale_data import (
    CanonicalTool,
    TokenPools,
    iter_toolgen_registrations,
    iter_toolgen_trajectories,
    iter_toolace_calls,
    normalized_query_hash,
    parse_tool_token,
    parse_toolace_calls,
    parse_toolace_system_tools,
    prepare_scaled_data,
    split_for_group,
)


def _write_array(path: Path, values: list[dict]) -> None:
    path.write_text(json.dumps(values), encoding="utf-8")


def _toolgen_record(query: str, token: str) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": query, "loss": False},
            {"role": "assistant", "content": token, "loss": True},
        ]
    }


class PrepareScaleDataTests(unittest.TestCase):
    def test_parses_tool_token(self) -> None:
        self.assertEqual(parse_tool_token("<<Weather&&forecast>>"), ("Weather", "forecast"))
        with self.assertRaises(ValueError):
            parse_tool_token("Weather")

    def test_streams_toolgen_registration_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registration.json"
            _write_array(path, [_toolgen_record("Tool Name: Weather", "<<Weather&&forecast>>")])
            tools = list(iter_toolgen_registrations(path))
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].tool_name, "Weather")
            self.assertEqual(tools[0].endpoint_name, "forecast")

    def test_toolace_prompt_uses_structured_json_parser(self) -> None:
        system = (
            "prefix Here is a list of functions in JSON format that you can invoke:\n"
            '[{"name":"weather","description":"forecast","parameters":{"type":"object"}}]. '
            "Should you decide"
        )
        self.assertEqual(parse_toolace_system_tools(system)[0]["name"], "weather")

    def test_toolace_call_parser_handles_declared_names_and_structured_values(self) -> None:
        calls = parse_toolace_calls(
            '[Market Trends API(from="2025-01-01", enabled=True, '
            'filters={"tags": ["a,b", "c"]}), /status()]',
            ["Market Trends API", "/status"],
        )
        self.assertEqual(calls[0][0], "Market Trends API")
        self.assertEqual(calls[0][1]["from"], "2025-01-01")
        self.assertEqual(calls[0][1]["filters"]["tags"][0], "a,b")
        self.assertEqual(calls[1], ("/status", {}))
        with self.assertRaises(ValueError):
            parse_toolace_calls("[undeclared()]", ["weather"])

    def test_toolace_call_iterator_requires_immediate_user_query(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toolace.json"
            system = (
                "Here is a list of functions in JSON format that you can invoke:\n"
                '[{"name":"weather api","description":"forecast",'
                '"parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]. '
                "Should you decide"
            )
            _write_array(
                path,
                [
                    {
                        "system": system,
                        "conversations": [
                            {"from": "user", "value": "Weather in Suzhou?"},
                            {
                                "from": "assistant",
                                "value": '[weather api(city="Suzhou")]',
                            },
                        ],
                    }
                ],
            )
            records = list(iter_toolace_calls(path))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].query, "Weather in Suzhou?")
            self.assertEqual(records[0].arguments, {"city": "Suzhou"})

    def test_prepare_pipeline_emits_toolace_readback_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "registration.json"
            retrieval = root / "retrieval.json"
            trajectories = root / "trajectories.json"
            toolace = root / "toolace.json"
            output = root / "prepared"
            _write_array(registration, [])
            _write_array(retrieval, [])
            _write_array(trajectories, [])
            system = (
                "Here is a list of functions in JSON format that you can invoke:\n"
                '[{"name":"weather","description":"forecast",'
                '"parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]. '
                "Should you decide"
            )
            _write_array(
                toolace,
                [
                    {
                        "system": system,
                        "conversations": [
                            {"from": "user", "value": "Weather in Suzhou?"},
                            {
                                "from": "assistant",
                                "value": '[weather(city="Suzhou")]',
                            },
                        ],
                    }
                ],
            )
            manifest = prepare_scaled_data(
                toolgen_registration=registration,
                toolgen_retrieval=retrieval,
                toolgen_trajectories=trajectories,
                toolace=toolace,
                output_dir=output,
            )
            rows = [
                json.loads(line)
                for line in (output / "readback.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["arguments"], {"city": "Suzhou"})
            self.assertEqual(sum(manifest["counts"]["readback_calls_by_split"].values()), 1)

    def test_trajectory_ignores_tokens_only_mentioned_in_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.json"
            _write_array(
                path,
                [
                    {
                        "id": "x",
                        "conversations": [
                            {"from": "user", "value": "query"},
                            {
                                "from": "assistant",
                                "value": "The previous <<Wrong&&call>> failed.",
                            },
                            {"from": "assistant", "value": "<<Right&&call>>"},
                        ],
                    }
                ],
            )
            records = list(iter_toolgen_trajectories(path))
            self.assertEqual(records[0].target_tokens, ("<<Right&&call>>",))

    def test_group_hash_is_source_independent_and_identity_preserves_schema(self) -> None:
        first = CanonicalTool.create(
            source="a",
            source_id="1",
            tool_name="Weather",
            endpoint_name="Forecast",
            document="first",
            parameters={"type": "object", "description": "one"},
        )
        second = CanonicalTool.create(
            source="b",
            source_id="2",
            tool_name=" weather ",
            endpoint_name="forecast",
            document="second",
            parameters={"type": "object", "description": "two"},
        )
        self.assertEqual(first.group_hash, second.group_hash)
        self.assertEqual(first.schema_hash, second.schema_hash)
        self.assertEqual(first.identity_hash, second.identity_hash)

    def test_split_is_deterministic_for_cross_source_group(self) -> None:
        group = "a" * 64
        self.assertEqual(
            split_for_group(group, seed=17),
            split_for_group(group, seed=17),
        )

    def test_token_pools_reject_overlap(self) -> None:
        with self.assertRaises(ValueError):
            TokenPools(train=(0, 10), validation=(9, 12), test=(12, 15)).validate()

    def test_query_hash_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(
            normalized_query_hash("Find   WEATHER in Suzhou!"),
            normalized_query_hash("find weather in suzhou"),
        )

    def test_prepare_pipeline_emits_disjoint_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "registration.json"
            retrieval = root / "retrieval.json"
            trajectories = root / "trajectories.json"
            output = root / "prepared"
            tokens = ["<<Weather&&forecast>>", "<<Math&&sum>>", "<<Search&&web>>"]
            _write_array(
                registration,
                [_toolgen_record(f"document {index}", token) for index, token in enumerate(tokens)],
            )
            _write_array(
                retrieval,
                [_toolgen_record(f"query {index}", token) for index, token in enumerate(tokens)],
            )
            _write_array(
                trajectories,
                [
                    {
                        "id": str(index),
                        "conversations": [
                            {"from": "user", "value": f"query {index}"},
                            {"from": "assistant", "value": token},
                        ],
                    }
                    for index, token in enumerate(tokens)
                ],
            )

            manifest = prepare_scaled_data(
                toolgen_registration=registration,
                toolgen_retrieval=retrieval,
                toolgen_trajectories=trajectories,
                output_dir=output,
            )

            self.assertTrue(manifest["audits"]["tool_group_overlap_zero"])
            self.assertTrue(manifest["audits"]["token_pool_overlap_zero"])
            self.assertEqual(manifest["counts"]["unique_tools"], 3)
            self.assertTrue((output / "tools.jsonl").is_file())
            self.assertTrue((output / "retrieval.jsonl").is_file())
            self.assertTrue((output / "trajectories.jsonl").is_file())
            self.assertTrue((output / "split_manifest.json").is_file())

    def test_prepare_pipeline_quarantines_token_document_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "registration.json"
            retrieval = root / "retrieval.json"
            trajectories = root / "trajectories.json"
            output = root / "prepared"
            token = "<<Demo&&call>>"
            _write_array(
                registration,
                [
                    _toolgen_record("unrelated document one", token),
                    _toolgen_record("unrelated document two", token),
                ],
            )
            _write_array(retrieval, [_toolgen_record("query", token)])
            _write_array(
                trajectories,
                [
                    {
                        "id": "x",
                        "conversations": [
                            {"from": "user", "value": "query"},
                            {"from": "assistant", "value": token},
                        ],
                    }
                ],
            )

            manifest = prepare_scaled_data(
                toolgen_registration=registration,
                toolgen_retrieval=retrieval,
                toolgen_trajectories=trajectories,
                output_dir=output,
            )

            self.assertEqual(manifest["counts"]["toolgen_ambiguous_tokens"], 1)
            self.assertEqual(manifest["counts"]["toolgen_ambiguous_identity_variants"], 2)
            self.assertEqual(
                manifest["counts"]["retrieval_skipped"]["queries_with_ambiguous_token"],
                1,
            )
            self.assertEqual(manifest["counts"]["trajectories_skipped"]["ambiguous_token"], 1)

    def test_prepare_pipeline_aggregates_multi_positive_retrieval_queries(self) -> None:
        candidate_tokens = [
            f"<<Tool {index}&&call {index}>>" for index in range(20)
        ]
        by_split: dict[str, list[str]] = {}
        for token in candidate_tokens:
            tool_name, endpoint_name = parse_tool_token(token)
            group_hash = CanonicalTool.create(
                source="toolgen",
                source_id=token,
                tool_name=tool_name,
                endpoint_name=endpoint_name,
                document=token,
                token=token,
            ).group_hash
            by_split.setdefault(split_for_group(group_hash, seed=17), []).append(token)
        selected = next(tokens[:2] for tokens in by_split.values() if len(tokens) >= 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "registration.json"
            retrieval = root / "retrieval.json"
            trajectories = root / "trajectories.json"
            output = root / "prepared"
            _write_array(
                registration,
                [_toolgen_record(f"document {index}", token) for index, token in enumerate(selected)],
            )
            _write_array(
                retrieval,
                [_toolgen_record("one query", token) for token in selected],
            )
            _write_array(trajectories, [])

            manifest = prepare_scaled_data(
                toolgen_registration=registration,
                toolgen_retrieval=retrieval,
                toolgen_trajectories=trajectories,
                output_dir=output,
            )

            records = [json.loads(line) for line in (output / "retrieval.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["target_tokens"], sorted(selected))
            self.assertEqual(sum(manifest["counts"]["retrieval_queries_by_split"].values()), 1)
            self.assertEqual(
                sum(manifest["counts"]["retrieval_target_pairs_by_split"].values()),
                2,
            )


if __name__ == "__main__":
    unittest.main()
