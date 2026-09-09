"""V2-specific architecture, schema contracts and content-diagnostic tests."""
import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from test_toolbench_agent import agent, tools, row
from latent_register.structured_memory import StructuredResamplerMemory
from latent_register.toolbench_agent import ToolBenchAgent, ExecutionResult, Decision, prompt_parts, run_serial_agent
from latent_register.toolbench_checkpoint import load_agent, save_agent, sha256
from latent_register.toolbench_data import FINISH, ToolSpec, compact, trajectory_steps
from latent_register.toolbench_schema import SCHEMA_TASKS, canonical, registration_layout, schema_targets
from latent_register.evaluate_toolbench_memory import (argument_metrics, evaluate, schema_metrics, summarize, wrong_mapping)
from latent_register.train_toolbench_agent import audit_token_lengths, parse_args


def test_schema_targets_preserve_scope_types_null_and_references():
    schema = {"type": "object", "properties": {
        "a/b~c": {"type": ["number", "null"], "default": 0},
        "nested": {"type": "object", "properties": {"enabled": {"type": "boolean", "default": False}}, "required": ["enabled"]},
        "items": {"type": "array", "items": {"type": "object", "properties": {"unit": {"enum": ["C", None], "default": None}}, "required": ["unit"]}},
        "other": {"$ref": "#/definitions/rule"}},
        "definitions": {"rule": {"type": "string", "const": "fixed"}}, "required": ["a/b~c"],
        "examples": [{"properties": {"NOT_A_FIELD": {"default": 999}}}]}
    targets = schema_targets(schema)
    assert targets["schema_types"]["/properties/a~1b~0c"] == ["number", "null"]
    assert targets["schema_defaults"] == {"/properties/a~1b~0c": 0,
        "/properties/nested/properties/enabled": False, "/properties/items/items/properties/unit": None}
    assert targets["schema_required"][""] == ["a/b~c"]
    assert targets["schema_required"]["/properties/nested"] == ["enabled"]
    assert targets["schema_enums"]["/definitions/rule"] == {"const": "fixed"}
    assert "NOT_A_FIELD" not in compact(targets)
    assert "/properties/other" not in targets["schema_types"]  # No invented resolved type.


def test_layout_byte_identity_nested_and_unicode_field_spans():
    tool = ToolSpec("synthetic", "中文说明", {"type": "object", "properties": {
        "城市/名": {"type": "string"}, "options": {"type": "object", "properties": {"unit": {"const": "celsius"}}}}})
    text, spans = registration_layout(tool)
    assert text == tool.registration_document
    assert {name for name, _, _ in spans} == {"/properties/城市~1名", "/properties/options", "/properties/options/properties/unit"}
    for name, begin, end in spans:
        assert text[begin:end].startswith('"') and ":" in text[begin:end]
        assert 0 <= begin < end <= len(text)


def test_field_views_use_one_backbone_pass_and_exact_offsets(agent, tools):
    with patch.object(agent, "_hidden", wraps=agent._hidden) as hidden, patch.object(
            agent.memory_compiler, "forward", wraps=agent.memory_compiler.forward) as compiler:
        rows, memory = agent.compile([tools["weather/forecast"]], profile=True)
        assert hidden.call_count == 1
        states, mask, fields = compiler.call_args.args
        assert fields.shape[1] == 5
        assert fields.any(-1).all()
        assert not (fields & ~mask[:, None, :].bool()).any()
        assert memory.shape == (1, 8, 32) and rows.shape == (1, 32)
        assert agent.last_compile_profile["schema_fields"] == 5
        assert agent.last_compile_profile["memory_compiler_seconds"] >= 0


def test_schema_only_prefix_has_no_query_identity_field_name_or_answer(agent, tools):
    tool = tools["weather/forecast"]
    for task in SCHEMA_TASKS:
        with patch.object(agent.tokenizer, "apply_chat_template", wraps=agent.tokenizer.apply_chat_template) as render:
            prompt_parts(agent.tokenizer, [], task, condition="memory", document=tool.registration_document)
            body = compact(render.call_args.args[0])
            assert "weather/forecast" not in body and "Paris" not in body
            assert "alerts" not in body and "Weather forecast" not in body
        with pytest.raises(ValueError): prompt_parts(agent.tokenizer, [{"query": "secret"}], task, condition="memory")


def test_schema_supervision_alone_updates_both_refinement_layers_and_shared_qwen(agent, tools):
    states = []
    def capture(_module, _args, output):
        output.last_hidden_state.retain_grad()
        states.append(output.last_hidden_state)
    handle = agent.backbone.model.register_forward_hook(capture)
    _, memory = agent.compile([tools["weather/forecast"]])
    agent.schema_loss(tools["weather/forecast"], memory[0], tasks_per_step=5).backward()
    handle.remove()
    assert states[0].grad.abs().sum() > 0  # Document-side backbone path, not only caller.
    for block in agent.memory_compiler.blocks:
        for parameter in (block.cross_attention.query.weight, block.self_attention.query.weight, block.ffn[0].weight):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert agent.backbone.get_input_embeddings().weight.grad.abs().sum() > 0
    assert agent.backbone.lm_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in agent.output_compiler.parameters())


@pytest.mark.parametrize("slots", [8, 16])
def test_compiler_all_slots_gradients_padding_and_field_contribution(slots):
    torch.manual_seed(9)
    compiler = StructuredResamplerMemory(32, 32, slots, 2, 4)
    states = torch.randn(2, 11, 32, requires_grad=True)
    mask = torch.ones(2, 11, dtype=torch.bool)
    mask[:, -2:] = False
    fields = torch.zeros(2, 2, 11, dtype=torch.bool)
    fields[:, 0, :4] = True
    fields[:, 1, 4:9] = True
    output = compiler(states, mask, fields)
    (output * torch.randn_like(output)).sum().backward()
    assert (compiler.queries.grad.abs().sum(-1) > 0).all()
    padded = states.detach().clone()
    padded[:, -2:] = 10000
    assert torch.allclose(output.detach(), compiler(padded, mask, fields), atol=1e-6)
    assert not torch.allclose(output.detach(), compiler(states.detach(), mask, torch.zeros_like(fields)))
    assert output.shape == (2, slots, 32)


def test_structured_compiler_bf16_and_empty_fields():
    compiler = StructuredResamplerMemory(32, 32, 8, 2, 4).bfloat16()
    states = torch.randn(1, 8, 32, dtype=torch.bfloat16, requires_grad=True)
    output = compiler(states, torch.ones(1, 8))
    (output * torch.randn_like(output)).sum().backward()
    assert output.dtype == torch.float32 and torch.isfinite(states.grad).all()
    with pytest.raises(ValueError): compiler(states, torch.zeros(1, 8))


@pytest.mark.parametrize("kind,slots", [("structured", 8), ("structured", 16), ("legacy", 8)])
def test_checkpoint_architecture_and_zero_step_registration_roundtrip(agent, tools, tmp_path, kind, slots):
    model = ToolBenchAgent(copy.deepcopy(agent.backbone), agent.tokenizer, rank=8, slots=slots,
        memory_kind=kind, memory_width=32, memory_heads=4, limits=agent.limits).eval().requires_grad_(False)
    model.register_tools([tools["city/search"]], profile=True)
    parameters_before = {name: parameter.clone() for name, parameter in model.named_parameters()}
    before = model._registry["city/search"][2].clone()
    output = tmp_path / "model"
    save_agent(model, output, metadata={"train_api_identities": ["unrelated-train-api"]})
    restored = load_agent(output)
    restored.register_tools([tools["city/search"], tools["weather/forecast"]], profile=True)
    assert restored.memory_config["kind"] == kind and restored.slots == slots
    assert torch.allclose(before, restored._registry["city/search"][2], atol=1e-6)
    for name, parameter in restored.named_parameters():
        assert torch.equal(parameter, parameters_before[name])
    assert sum(item["backbone_forwards"] for item in restored.registration_profiles.values()) == 2


def test_v1_checkpoint_load_keeps_legacy_compiler(agent, tools, tmp_path):
    legacy = ToolBenchAgent(agent.backbone, agent.tokenizer, rank=8, memory_kind="legacy", limits=agent.limits).eval()
    path = tmp_path / "v1"
    save_agent(legacy, path, metadata={})
    config = json.loads((path / "agent.json").read_text())
    config["version"] = 1
    del config["memory_compiler"]
    (path / "agent.json").write_text(json.dumps(config))
    manifest = json.loads((path / "SHA256.json").read_text())
    manifest["agent.json"] = sha256(path / "agent.json")
    (path / "SHA256.json").write_text(json.dumps(manifest))
    restored = load_agent(path)
    assert restored.memory_config["kind"] == "legacy"
    assert torch.equal(legacy.compile([tools["city/search"]])[1], restored.compile([tools["city/search"]])[1])


def test_required_type_enum_metrics_and_strict_json(tools):
    tool = tools["weather/forecast"]
    target = {"city": "Paris", "days": 2, "options": {"units": "C", "alerts": True}}
    missing = argument_metrics(tool, target, '{"city":"Paris","days":2,"options":{}}', False)
    assert missing["missing_required_count"] == 2
    wrong = argument_metrics(tool, target, '{"city":"Paris","days":"2","options":{"units":"K","alerts":true}}', False)
    assert wrong["type_error_count"] == 1 and wrong["enum_const_error_count"] == 1
    assert argument_metrics(tool, target, compact(target), True)["exact"] is False
    assert schema_metrics("schema_defaults", {"/x": False}, '{"/x":0}', False)["exact"] is False


def test_different_contract_negative_control_and_empty_fact_guard(tools):
    mapping = wrong_mapping(tools)
    assert mapping["city/search"] == "weather/forecast"
    same = replace(tools["city/search"], api_identity="duplicate", aliases=())
    with pytest.raises(ValueError): wrong_mapping({"city/search": tools["city/search"], "duplicate": same})
    records = []
    for identity in ("a", "b", "c"):
        for condition in ("correct", "blank", "wrong"):
            text = '["/properties/a"]' if condition == "correct" else '[]'
            records.append(dict(kind="schema", api_identity=identity, task="schema_names", condition=condition,
                truncated=False, wrong_changes_target=True,
                metrics=schema_metrics("schema_names", ["/properties/a"], text, False)))
    report = summarize(records)
    assert report["content_signal"] == "positive_on_this_development_checkpoint"
    for record in records:
        record["metrics"] = schema_metrics("schema_names", [], "[]", False)
    assert summarize(records)["content_signal"] == "no_positive_signal_or_insufficient_evidence"


def test_real_model_diagnostic_same_length_conditions_and_no_updates(agent, tools, row):
    agent.eval().requires_grad_(False)
    snapshots = {name: value.clone() for name, value in agent.named_parameters()}
    records, report = evaluate(agent, tools, trajectory_steps(row, tools, source_format="toolbench"),
                               max_new_tokens=1, max_thought_tokens=1)
    assert len([record for record in records if record["kind"] == "schema"]) == 30
    assert len([record for record in records if record["kind"] == "arguments"]) == 3
    for task in SCHEMA_TASKS:
        panel = [record for record in records if record["kind"] == "schema" and record["task"] == task]
        assert len({record["cost"]["input_positions"] for record in panel}) == 1
        assert {record["cost"]["memory_positions"] for record in panel} == {8}
    assert report["content_signal"] == "no_positive_signal_or_insufficient_evidence"
    assert all(torch.equal(value, snapshots[name]) for name, value in agent.named_parameters())


def test_argument_plan_is_generated_once_and_gold_thought_never_used(agent, tools, row):
    from latent_register.toolbench_agent import Generation
    steps = trajectory_steps(row, tools, source_format="toolbench")[:1]
    agent.eval().requires_grad_(False)
    plan = "GENERATED_PLAN_WITHOUT_GOLD"
    seen = []
    def generate(prefix, **kwargs):
        seen.append(prefix.shape[1])
        return Generation(plan if kwargs["max_new_tokens"] == 7 else "{}", (), True)
    with patch.object(agent, "generate", side_effect=generate):
        records, _ = evaluate(agent, tools, steps, max_new_tokens=9, max_thought_tokens=7)
    arguments = [record for record in records if record["kind"] == "arguments"]
    assert len(seen) == 34  # 30 schema + one shared plan + 3 argument conditions.
    assert {record["fixed_generated_plan"] for record in arguments} == {plan}


def test_execution_receipts_do_not_claim_answer_correctness(tools):
    class Policy:
        def decide(self, history, registry):
            if len(history) == 1:
                return Decision("city/search", {"text": "Paris"}, "", ())
            return Decision(FINISH, {"return_type": "give_answer", "final_answer": "possibly wrong"}, "", ())
    result = run_serial_agent(Policy(), "query", tools, lambda *_: ExecutionResult({"id": 1}, True))
    assert result["costs"]["executions_succeeded"] == 1
    assert result["costs"]["model_decisions"] == 2
    assert result["task_success_judged"] is False


def test_development_cli_audits_split_and_exports_real_costs(agent, tools, tmp_path):
    checkpoint = tmp_path / "model"
    save_agent(agent, checkpoint, metadata={"train_api_identities": ["separate_training_tool"]})
    path = tmp_path / "dev.jsonl"
    path.write_text("".join(compact({"api_identity": identity, "document": tool.document,
        "parameters": tool.parameters, "split": "dev"}) + "\n" for identity, tool in tools.items() if identity != FINISH))
    output = tmp_path / "evaluation"
    command = [sys.executable, "-m", "latent_register.evaluate_toolbench_memory", "--checkpoint", str(checkpoint),
        "--tools", str(path), "--split", "dev", "--max-new-tokens", "1", "--output-dir", str(output)]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((output / "REPORT.json").read_text())
    assert report["optimizer_updates"] == 0
    assert report["schema"]["correct"]["presentations"] == 10
    assert report["registration"]["city/search"]["memory_compiler_seconds"] >= 0
    path.write_text(path.read_text().replace('"dev"', '"test"'))
    failed = subprocess.run(command[:-1] + [str(tmp_path / "bad")], capture_output=True, text=True, env=environment, timeout=60)
    assert failed.returncode != 0
    assert not (tmp_path / "bad").exists()


def test_recipes_differ_only_in_capacity_and_allow_explicit_override():
    root = Path(__file__).resolve().parents[1]
    first = json.loads((root / "configs/toolbench-memory-v2-8.json").read_text())
    second = json.loads((root / "configs/toolbench-memory-v2-16.json").read_text())
    assert {key for key in first if first[key] != second[key]} == {"memory_slots"}
    args = parse_args(["--recipe", str(root / "configs/toolbench-memory-v2-8.json"),
        "--tools", "a", "--trajectories", "b", "--source-format", "toolbench", "--output-dir", "c", "--memory-slots", "16"])
    assert args.memory_slots == 16 and args.memory_depth == 2 and args.schema_weight == 0.5


def test_preflight_measures_expanded_slots_and_all_schema_targets(agent, tools, row):
    from latent_register.toolbench_agent import Limits
    steps = trajectory_steps(row, tools, source_format="toolbench")
    eight = audit_token_lengths(agent.tokenizer, tools, steps, limits=agent.limits, slots=8, condition="memory", schema_enabled=True)
    sixteen = audit_token_lengths(agent.tokenizer, tools, steps, limits=agent.limits, slots=16, condition="memory", schema_enabled=True)
    assert sixteen["arguments_prefix"] == eight["arguments_prefix"] + 8
    assert all(task + "_target" in eight for task in SCHEMA_TASKS)
    with pytest.raises(ValueError, match="overflow before model loading"):
        audit_token_lengths(agent.tokenizer, tools, steps, limits=Limits(2048, 512, 2), slots=8, condition="memory", schema_enabled=True)


def test_same_query_can_require_a_document_only_constant(agent):
    def tool(unit):
        return ToolSpec(unit, "Return the weather. The unit is fixed by this endpoint.",
            {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"const": unit}}, "required": ["city", "unit"]})
    c, f = tool("celsius"), tool("fahrenheit")
    assert schema_targets(c.parameters)["schema_enums"] != schema_targets(f.parameters)["schema_enums"]
    assert argument_metrics(c, {"city": "Paris", "unit": "celsius"}, '{"city":"Paris"}', False)["missing_required_count"] == 1
    assert argument_metrics(f, {"city": "Paris", "unit": "fahrenheit"}, '{"city":"Paris","unit":"celsius"}', False)["enum_const_error_count"] == 1
    assert prompt_parts(agent.tokenizer, [], "schema_enums", condition="memory", document=c.registration_document) == \
           prompt_parts(agent.tokenizer, [], "schema_enums", condition="memory", document=f.registration_document)


def test_generation_failure_still_counts_elapsed_model_cost(tools):
    class Policy:
        calls = 0
        def _clock(self, _sync):
            self.calls += 1
            return float(self.calls)
        def decide(self, *_args):
            raise ValueError("invalid generated JSON")
    result = run_serial_agent(Policy(), "query", tools, lambda *_: pytest.fail("must not execute"))
    assert result["status"] == "generation_error"
    assert result["costs"]["model_seconds"] == 1
    assert result["costs"]["tool_calls_attempted"] == 0


def test_full_document_baseline_does_not_compute_or_cache_unused_memory(agent, tools, row):
    agent.condition = "full_document"
    agent.memory_compiler.requires_grad_(False)
    with patch.object(agent.memory_compiler, "forward", side_effect=AssertionError("unused compiler called")):
        rows, memory = agent.compile([tools["city/search"]], profile=True)
        assert memory.shape == (1, 0, 32)
        assert agent.last_compile_profile["cache_tensor_bytes"] == rows.numel() * rows.element_size()
        agent(trajectory_steps(row, tools, source_format="toolbench"), tools)["loss"].backward()
    assert agent.backbone.lm_head.weight.grad.abs().sum() > 0
