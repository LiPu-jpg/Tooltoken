"""Synthetic CPU contracts. These tests do not measure ToolBench performance."""
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
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from latent_register.model import PhysicalOutputGenerator, TokenResamplerMemory
from latent_register.toolbench_agent import Decision, Limits, ToolBenchAgent, run_serial_agent
from latent_register.toolbench_checkpoint import load_agent, save_agent
from latent_register.toolbench_data import (FINISH, AgentStep, ToolSpec, alias_map, compact,
    finish_tool, json_object, load_training_steps, load_training_tools, strict_json, trajectory_steps)
from latent_register.train_toolbench_agent import accumulation_window_size

torch.set_num_threads(1)


@pytest.fixture
def tools():
    schema = {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer", "minimum": 1},
        "options": {"type": "object", "properties": {"units": {"enum": ["C", "F"]},
                    "alerts": {"type": "boolean"}}, "required": ["units", "alerts"]}},
        "required": ["city", "days", "options"], "additionalProperties": False}
    return {FINISH: finish_tool(), "weather/forecast": ToolSpec("weather/forecast", "Weather forecast.", schema,
            ("forecast", "<<Weather&&forecast>>")),
            "city/search": ToolSpec("city/search", "Search cities.",
                {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                ("search", "<<City&&search>>"))}


@pytest.fixture
def row():
    return {"id": "synthetic-001", "split": "train", "messages": [
        {"role": "user", "content": "Forecast Paris for 2 days in C, with alerts."},
        {"role": "assistant", "content": "Find the forecast.", "tool_calls": [{"id": "c1", "function": {
            "name": "forecast", "arguments": '{"city":"Paris","days":2,"options":{"units":"C","alerts":true}}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": {"temperatures": [20, 21], "error": None}},
        {"role": "assistant", "content": "I have the result.", "function_call": {
            "name": "Finish", "arguments": {"return_type": "give_answer", "final_answer": "20 and 21 C."}}}]}


@pytest.fixture
def agent():
    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
    for item in "user assistant system History Task forecast Weather Paris city days options C F true 2 Finish thought selected tool input complete JSON object arguments temperatures return_type give_answer final_answer".split():
        if item not in vocab:
            vocab[item] = len(vocab)
    engine = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    engine.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    torch.manual_seed(17)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=4096, tie_word_embeddings=False, attention_dropout=0.0,
        eos_token_id=1, pad_token_id=2))
    return ToolBenchAgent(model, tokenizer, rank=8, slots=8, limits=Limits(2048, 512, 512),
                          memory_width=32, memory_heads=4)


def test_causal_trajectory_preserves_typed_arguments_and_observations(row, tools):
    first, last = trajectory_steps(row, tools, source_format="toolbench")
    assert first.history == ({"type": "user", "content": row["messages"][0]["content"]},)
    assert "temperatures" not in compact(first.history)
    assert isinstance(first.arguments["days"], int)
    assert first.arguments["options"]["alerts"] is True
    assert last.history[-1]["content"]["temperatures"] == [20, 21]
    assert last.api_identity == FINISH
    assert last.arguments["final_answer"] not in compact(last.history)


@pytest.mark.parametrize("change", ["heldout", "observation", "parallel", "unfinished", "unknown", "system"])
def test_bad_trajectories_fail_closed(change, row, tools):
    if change == "heldout": row["split"] = "test"
    if change == "observation": row["messages"][2]["tool_call_id"] = "bad"
    if change == "parallel": row["messages"][1]["tool_calls"] *= 2
    if change == "unfinished": row["messages"].pop()
    if change == "unknown": row["messages"][1]["tool_calls"][0]["function"]["name"] = "unknown"
    if change == "system": row["messages"].insert(0, {"role": "system", "content": "gold tools"})
    with pytest.raises(ValueError): trajectory_steps(row, tools, source_format="toolbench")


def test_toolgen_adapter_consumes_document_wrapper_without_answer_leak(tools):
    row = {"id": "synthetic-toolgen", "split": "train", "conversations": [
        {"role": "user", "content": "Find Paris."},
        {"role": "assistant", "content": "Search."},
        {"role": "user", "content": "Please generate the action."},
        {"role": "assistant", "content": "<<City&&search>>"},
        {"role": "user", "content": "Please give the input. Here is the documentation: source wrapper"},
        {"role": "assistant", "content": '{"text":"Paris"}'},
        {"role": "function", "content": "Found Paris."},
        {"role": "assistant", "content": "<<Finish>>"},
        {"role": "user", "content": "Please give the input. Here is the documentation: Finish"},
        {"role": "assistant", "content": '{"return_type":"give_answer","final_answer":"Paris"}'}]}
    steps = trajectory_steps(row, tools, source_format="toolgen")
    assert len(steps) == 2
    assert steps[0].thought == "Search."
    assert "source wrapper" not in compact(steps[1].history)
    assert steps[1].history[-1]["content"] == "Found Paris."


@pytest.mark.parametrize("bad", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1} junk', '["Paris"]', '```json\n{}\n```'])
def test_arguments_strict_json_not_stq_normalization(bad):
    with pytest.raises(ValueError): json_object(bad)


def test_schema_requires_actual_nested_types_and_enum(tools):
    tool = tools["weather/forecast"]
    for invalid in [{"city": "Paris"}, {"city": "Paris", "days": "2", "options": {"units": "C", "alerts": True}},
                    {"city": "Paris", "days": 2, "options": {"units": "K", "alerts": True}}]:
        with pytest.raises(ValueError): tool.validate_arguments(invalid)
    with pytest.raises(ValueError): ToolSpec("x", "x", {"function_name": "x"})
    with pytest.raises(ValueError): ToolSpec("x", "x", {"type": "object", "$ref": "https://example.com/schema"})
    assert '"required"' in tool.registration_document


def test_ambiguous_binding_and_heldout_registry_rejected(tmp_path, tools):
    duplicate = replace(tools["city/search"], aliases=("forecast",))
    with pytest.raises(ValueError): alias_map({**tools, "city/search": duplicate})
    path = tmp_path / "tools.jsonl"
    path.write_text(json.dumps({"split": "test"}) + "\n")
    with pytest.raises(ValueError): load_training_tools(path)


def test_duplicate_trajectory_ids_rejected(tmp_path, row, tools):
    path = tmp_path / "train.jsonl"
    path.write_text((compact(row) + "\n") * 2)
    with pytest.raises(ValueError): load_training_steps(path, tools, source_format="toolbench")


def test_real_qwen_joint_gradients_reach_backbone_and_both_compilers(agent, tools, row):
    steps = trajectory_steps(row, tools, source_format="toolbench")
    output = agent(steps, tools, candidate_count=3)
    output["loss"].backward()
    for parameter in [agent.backbone.get_input_embeddings().weight, agent.backbone.lm_head.weight,
                      agent.backbone.model.layers[0].self_attn.q_proj.weight,
                      agent.output_compiler.generator.up.weight, agent.memory_compiler.queries]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_memory_slots_all_receive_argument_gradients(agent, tools, row):
    step = trajectory_steps(row, tools, source_format="toolbench")[0]
    _, memories = agent.compile([tools[step.api_identity]])
    memories.retain_grad()
    prefix = agent.prefix(step.history, "arguments", step.thought, memory=memories[0])
    agent.nll(prefix, compact(step.arguments)).backward()
    assert memories.grad.shape == (1, 8, 32)
    assert (memories.grad.abs().sum(-1) > 0).all()


def test_frozen_backbone_updates_only_compilers(agent, tools, row):
    agent.backbone.requires_grad_(False)
    agent(trajectory_steps(row, tools, source_format="toolbench"), tools)["loss"].backward()
    assert all(parameter.grad is None for parameter in agent.backbone.parameters())
    assert agent.memory_compiler.queries.grad.abs().sum() > 0


def test_limits_never_truncate_document_schema_or_target(agent, tools):
    agent.limits = Limits(2048, 2, 2)
    with pytest.raises(ValueError, match="Document/schema"): agent.compile([tools["weather/forecast"]])
    prefix = agent.prefix([], "thought")
    with pytest.raises(ValueError, match="target"): agent.nll(prefix, "too many words here")
    agent.limits = Limits(prefix.shape[1], 512, 512)
    with pytest.raises(ValueError): agent.nll(prefix, "x")


def test_train_and_generate_first_logits_use_identical_expanded_prefix(agent, tools, row):
    agent.eval()
    step = trajectory_steps(row, tools, source_format="toolbench")[0]
    _, memory = agent.compile([tools[step.api_identity]])
    prefix = agent.prefix(step.history, "arguments", step.thought, memory=memory[0])
    full = agent._hidden(inputs_embeds=prefix, use_cache=False).last_hidden_state[:, -1]
    cached = agent._hidden(inputs_embeds=prefix, use_cache=True)
    assert torch.allclose(full, cached.last_hidden_state[:, -1], atol=1e-6)
    next_id = torch.tensor([[3]])
    extended = torch.cat([prefix, agent.backbone.get_input_embeddings()(next_id)], 1)
    continued = agent._hidden(input_ids=next_id, past_key_values=cached.past_key_values, use_cache=True).last_hidden_state
    replay = agent._hidden(inputs_embeds=extended, use_cache=False).last_hidden_state[:, -1:]
    assert torch.allclose(continued, replay, atol=1e-5)


def test_eos_and_truncation_distinguished(agent):
    agent.eval()
    prefix = agent.prefix([], "thought")
    with torch.no_grad(): agent.backbone.lm_head.weight.zero_()
    result = agent.generate(prefix, max_new_tokens=2)
    assert result.truncated and len(result.token_ids) == 2
    original = agent.tokenizer.eos_token_id
    agent.tokenizer.eos_token_id = 0
    result = agent.generate(prefix, max_new_tokens=2)
    assert result.stopped_on_eos and result.token_ids == ()
    agent.tokenizer.eos_token_id = original


def test_zero_step_registration_one_forward_and_incremental_cache(agent, tools):
    with pytest.raises(ValueError): agent.register_tools(list(tools.values()))
    agent.eval().requires_grad_(False)
    with patch.object(agent, "_hidden", wraps=agent._hidden) as forward:
        agent.register_tools([tools["city/search"]])
        old = agent._registry["city/search"][1].clone()
        agent.register_tools([tools["city/search"], tools["weather/forecast"]])
        assert forward.call_count == 2
        assert torch.equal(old, agent._registry["city/search"][1])
    with pytest.raises(ValueError): agent.register_tools([replace(tools["city/search"], document="new")])
    agent.train()
    assert not agent._registry


def test_checkpoint_reload_retains_outputs_and_nonpersistent_buffers(agent, tools, tmp_path):
    agent.eval()
    rope = agent.backbone.model.rotary_emb
    rope.inv_freq = rope.inv_freq.bfloat16().float()
    before_rows, before_memory = agent.compile([tools["city/search"]])
    output = tmp_path / "checkpoint"
    save_agent(agent, output, metadata={"synthetic": True})
    restored = load_agent(output)
    rows, memory = restored.compile([tools["city/search"]])
    assert torch.equal(rope.inv_freq, restored.backbone.model.rotary_emb.inv_freq)
    assert torch.allclose(before_rows, rows, atol=1e-6)
    assert torch.allclose(before_memory, memory, atol=1e-6)
    with pytest.raises(FileExistsError): save_agent(agent, output, metadata={})
    (output / "agent.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"): load_agent(output)


def test_bf16_compilers_keep_finite_values_and_gradients():
    memory = TokenResamplerMemory(16, 8, 8, 1.0).bfloat16()
    output = PhysicalOutputGenerator(16, 8, 1.0).bfloat16()
    states = torch.randn(2, 5, 16, dtype=torch.bfloat16, requires_grad=True)
    slots = memory(states, torch.ones(2, 5))
    rows = output.output_rows(states.mean(1))
    (slots.square().mean() + rows.square().mean()).backward()
    assert torch.isfinite(slots).all() and torch.isfinite(states.grad).all()
    with pytest.raises(ValueError): memory(states, torch.zeros(2, 5))


def test_runtime_history_feedback_finish_and_exact_identity(tools):
    seen = []
    class ScriptedPolicy:
        def decide(self, history, registry):
            seen.append(copy.deepcopy(history))
            if len(seen) == 1:
                return Decision("city/search", {"text": "Paris"}, "search", tuple(registry))
            return Decision(FINISH, {"return_type": "give_answer", "final_answer": "Found"}, "done", tuple(registry))
    calls = []
    def executor(identity, arguments):
        calls.append((identity, arguments))
        return {"city_id": 101}
    result = run_serial_agent(ScriptedPolicy(), "Find Paris", tools, executor)
    assert result["status"] == "give_answer"
    assert calls == [("city/search", {"text": "Paris"})]
    assert seen[1][-1]["content"] == {"city_id": 101}
    assert "city_id" not in compact(seen[0])


def test_runtime_call_budget_and_errors(tools):
    class ScriptedPolicy:
        def decide(self, history, registry):
            return Decision("city/search", {"text": "Paris"}, "", ())
    result = run_serial_agent(ScriptedPolicy(), "x", tools, lambda *_: (_ for _ in ()).throw(RuntimeError("failed")), max_calls=1)
    assert result["status"] == "call_budget_exhausted"
    assert result["history"][-1]["content"]["error"] == "RuntimeError"


def test_accumulation_tail_uses_real_window_size():
    assert [accumulation_window_size(i, 5, 4) for i in range(5)] == [4, 4, 4, 4, 1]


@pytest.mark.parametrize("condition", ["full_document", "query_only"])
def test_matched_baselines_train_independent_caller(agent, row, tools, condition):
    agent.condition = condition
    agent.memory_compiler.requires_grad_(False)
    agent(trajectory_steps(row, tools, source_format="toolbench"), tools)["loss"].backward()
    assert agent.backbone.lm_head.weight.grad.abs().sum() > 0
    assert agent.output_compiler.generator.up.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in agent.memory_compiler.parameters())


def test_real_training_cli_flushes_partial_windows_and_exports_reloadable_model(agent, row, tools, tmp_path):
    base = tmp_path / "tiny-base"
    agent.backbone.save_pretrained(base)
    agent.tokenizer.save_pretrained(base)
    registry = tmp_path / "tools.jsonl"
    registry.write_text("".join(compact({"api_identity": key, "document": tool.document,
        "parameters": tool.parameters, "aliases": tool.aliases, "split": "train"}) + "\n"
        for key, tool in tools.items() if key != FINISH))
    trajectories = tmp_path / "train.jsonl"
    trajectories.write_text(compact(row) + "\n")
    output = tmp_path / "train-run"
    command = [sys.executable, "-m", "latent_register.train_toolbench_agent", "--tools", str(registry),
        "--trajectories", str(trajectories), "--source-format", "toolbench", "--output-dir", str(output),
        "--model-path", str(base), "--model-role", "base", "--epochs", "2",
        "--gradient-accumulation-steps", "3", "--compiler-rank", "8", "--candidate-count", "3",
        "--memory-width", "32", "--memory-heads", "4", "--schema-tasks-per-step", "5",
        "--max-context-length", "2048", "--max-document-length", "512", "--max-target-length", "512"]
    environment = {**os.environ, "ACCELERATE_USE_CPU": "true", "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads((output / "TRAINING_COMPLETE.json").read_text())["updates"] == 2
    reloaded = load_agent(output / "checkpoint")
    assert not torch.equal(agent.backbone.get_input_embeddings().weight, reloaded.backbone.get_input_embeddings().weight)
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    metadata = json.loads((output / "checkpoint" / "agent.json").read_text())
    assert metadata["memory_compiler"]["depth"] == 2
    assert set(metadata["metadata"]["schema_task_exposures"].values()) == {4}


def test_training_export_converter_round_trip_and_failure_evidence(row, tools, tmp_path):
    registry = tmp_path / "tools.jsonl"
    registry.write_text("".join(compact({"api_identity": key, "document": tool.document,
        "parameters": tool.parameters, "aliases": tool.aliases, "split": "train"}) + "\n"
        for key, tool in tools.items() if key != FINISH))
    source = tmp_path / "source.json"
    original = {key: value for key, value in row.items() if key != "split"}
    source.write_text(compact([original]))
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    common = [sys.executable, "-m", "latent_register.prepare_toolbench_agent", "--tools", str(registry),
        "--training-source", str(source), "--input-format", "json-array", "--source-format", "toolbench"]
    output = tmp_path / "prepared"
    done = subprocess.run(common + ["--output-dir", str(output)], capture_output=True, text=True, env=environment, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (output / "train_tools.jsonl").read_bytes() == registry.read_bytes()
    loaded = load_training_steps(output / "train_trajectories.jsonl", tools, source_format="toolbench")
    assert loaded == trajectory_steps(row, tools, source_format="toolbench")
    source.write_text(compact([{**original, "split": "test"}]))
    failed_dir = tmp_path / "failed"
    failed = subprocess.run(common + ["--output-dir", str(failed_dir)], capture_output=True, text=True, env=environment, timeout=60)
    assert failed.returncode != 0
    assert json.loads((failed_dir / "FAILED.json").read_text())["record_id"] == row["id"]
    assert not (failed_dir / "PREPARED.json").exists()
