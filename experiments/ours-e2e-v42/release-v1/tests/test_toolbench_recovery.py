"""Runtime regressions only; synthetic policies are not model quality evidence."""
import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from test_toolbench_agent import agent, tools
from test_toolbench_http_eval import bindings
from latent_register.toolbench_agent import Decision, Generation, GenerationFailure, run_serial_agent
from latent_register.toolbench_data import FINISH
from latent_register.toolbench_fac_diagnostics import summarize_fac
from latent_register.toolbench_eval_format import convert_trace


def test_short_answer_can_use_remaining_context_without_cutting_prompt(agent):
    agent.eval()
    prefix = agent.prefix([], "thought")
    original = prefix.clone()
    agent.limits = replace(agent.limits, context=prefix.shape[1] + 2)
    with torch.no_grad():
        agent.backbone.lm_head.weight.zero_()
    agent.tokenizer.eos_token_id = 0
    result = agent.generate(prefix, max_new_tokens=100)
    assert result.stopped_on_eos and not result.token_ids
    assert result.budget() == {"requested_max_new_tokens": 100, "effective_max_new_tokens": 2}
    assert torch.equal(prefix, original)


def test_reduced_budget_without_eos_remains_truncated(agent):
    agent.eval()
    prefix = agent.prefix([], "thought")
    agent.limits = replace(agent.limits, context=prefix.shape[1] + 2)
    with torch.no_grad():
        agent.backbone.lm_head.weight.zero_()
    result = agent.generate(prefix, max_new_tokens=100)
    assert result.truncated and len(result.token_ids) == 2
    agent.limits = replace(agent.limits, context=prefix.shape[1])
    with pytest.raises(ValueError, match="No generation positions"):
        agent.generate(prefix, max_new_tokens=100)


def invalid():
    return GenerationFailure("'text' is required", stage="arguments", text="{}",
                             identity="city/search", thought="Search", reason="invalid_arguments")


def test_finish_shaped_json_is_not_rebound_to_finish_behind_the_selector(agent, tools):
    agent.eval().requires_grad_(False)
    raw = '{"return_type":"give_answer","final_answer":"Found"}'
    with patch.object(agent, "selection_scores", return_value=torch.tensor([[0., 10., 0.]])), \
         patch.object(agent, "generate", side_effect=[Generation("Search", (), True), Generation(raw, (), True)]):
        with pytest.raises(GenerationFailure) as failure:
            agent.decide([{"type": "user", "content": "Find Paris"}], tools)
    assert failure.value.details["reason"] == "invalid_arguments"
    assert failure.value.details["api_identity"] == "city/search"
    assert failure.value.details["generated_text"] == raw


def test_feedback_redecides_and_preserves_call_budget_and_actual_observation(tools):
    histories, calls = [], []
    class Policy:
        def decide(self, history, registry):
            histories.append(copy.deepcopy(history))
            if len(histories) == 1:
                raise invalid()
            if len(histories) == 2:
                assert history[-1]["type"] == "validation_error"
                assert history[-1]["generated_arguments"] == "{}"
                return Decision("city/search", {"text": "Paris"}, "Correct the input", tuple(registry))
            assert history[-1]["type"] == "observation"
            assert history[-1]["content"] == {"city_id": 101}
            return Decision(FINISH, {"return_type": "give_answer", "final_answer": "Found"}, "Done", tuple(registry))
    def execute(identity, arguments):
        calls.append((identity, arguments)); return {"city_id": 101}
    result = run_serial_agent(Policy(), "Find Paris", tools, execute, max_calls=1, max_validation_retries=1)
    assert result["status"] == "give_answer"
    assert calls == [("city/search", {"text": "Paris"})]
    assert result["costs"]["model_decisions"] == 3
    assert result["costs"]["tool_calls_attempted"] == 1
    assert result["costs"]["validation_retries"] == result["costs"]["validation_failures"] == 1
    assert result["decisions"][0]["generation_failure"]["generated_text"] == "{}"


@pytest.mark.parametrize("budget", [0, 2])
def test_invalid_generation_never_executes_or_retries_forever(tools, budget):
    class Policy:
        def decide(self, history, registry): raise invalid()
    def execute(*args): raise AssertionError("Invalid arguments must not reach executor")
    result = run_serial_agent(Policy(), "x", tools, execute, max_validation_retries=budget)
    assert result["status"] == "generation_error"
    assert result["costs"]["model_decisions"] == budget + 1
    assert result["costs"]["validation_retries"] == budget
    assert result["costs"]["tool_calls_attempted"] == 0


@pytest.mark.parametrize("failure", [RuntimeError("CUDA failure"), GenerationFailure(
    "truncated", stage="arguments", text="{", identity="city/search", reason="truncated")])
def test_infrastructure_and_truncation_errors_are_not_schema_retries(tools, failure):
    class Policy:
        def decide(self, history, registry): raise failure
    result = run_serial_agent(Policy(), "x", tools, lambda *_: None, max_validation_retries=2)
    assert result["costs"]["model_decisions"] == 1
    assert result["costs"]["validation_retries"] == 0


def test_validation_feedback_exports_without_fake_execution_or_finish(tools, bindings):
    feedback = {"type": "validation_error", "retry": 1, "api_identity": "city/search",
                "generated_arguments": "{}", "thought": "Search", "error": "text is required"}
    trace = {"status": "generation_error", "history": [{"type": "user", "content": "Paris?"}, feedback]}
    exported = convert_trace("Paris?", trace, tools, bindings)
    assert exported["answer"]["final_answer"] == ""
    node = exported["answer"]["answer_details"][0]
    nodes = [node]
    while node["next"]:
        node = node["next"][0]; nodes.append(node)
    assert not any(n["role"] == "tool" for n in nodes)
    assert nodes[-1]["role"] == "user" and '"generated_arguments":"{}"' in nodes[-1]["message"]
    feedback["api_identity"] = "unregistered/tool"
    with pytest.raises(ValueError, match="unbound validation"):
        convert_trace("Paris?", trace, tools, bindings)


def fac_data(answer=""):
    inputs = {"native": {"query": "native question", "answer": {"final_answer": answer}}}
    rows = [{"query": "native question", "final_answer": answer or "final_answer was not properly loaded",
             "evaluation": "Answer Status: Unsolved\nReason: Parsing error",
             "prompt": "real judge prompt" if answer else "an error has occured"}]
    for name, label in [("positive", "Solved"), ("negative", "Unsolved")]:
        query = "__adapter_control_" + name + "__ question"
        inputs[name] = {"query": query, "answer": {"final_answer": name}}
        rows.append({"query": query, "final_answer": name, "evaluation": "Answer Status: " + label,
                     "prompt": "real judge prompt"})
    return rows, inputs


def test_fac_missing_answer_is_failure_but_not_semantic_judgment():
    result = summarize_fac(*fac_data())
    assert result["judge_controls_valid"]
    assert result["native_missing_final_answers"] == 1
    assert result["native_semantic_judgments"] == 0
    assert result["native_semantic_score_mean"] is None
    assert result["task_outcome_mean"] == 0


def test_fac_does_not_confuse_reason_text_with_upstream_fallback():
    result = summarize_fac(*fac_data("An actual answer"))
    assert result["native_semantic_judgments"] == 1
    assert result["native_judge_or_export_errors"] == 0


def test_fac_missing_rows_or_judge_failure_cannot_count_as_valid_judgments():
    rows, inputs = fac_data("An actual answer")
    with pytest.raises(ValueError, match="multiset"):
        summarize_fac(rows[:-1], inputs)
    rows[0]["prompt"] = "an error has occured"
    result = summarize_fac(rows, inputs)
    assert result["native_judge_or_export_errors"] == 1
    assert result["native_semantic_judgments"] == 0
    assert result["task_outcome_mean"] is None
