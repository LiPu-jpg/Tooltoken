"""Run the memory_fp32 reader with the bounded completion controller."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve()
ADAPTER = HERE.parents[1]
sys.path.insert(0, str(ADAPTER / "variant" / "task_completion"))
sys.path.insert(0, str(ADAPTER / "variant"))

from completion import Budget
from native_adapter import run_native
from reader_variant import install as install_reader
from latent_register.toolbench_agent import ExecutionResult
from latent_register.toolbench_checkpoint import load_agent, sha256
from latent_register.toolbench_data import FINISH, load_tools, read_jsonl
from latent_register.run_toolbench_e2e import load_queries
from latent_register.toolbench_eval_format import evaluator_names
from latent_register.toolbench_shared_export import SharedToolExport


TRANSIENT_HTTP = {429, 502, 503, 504}


def transport_status(result: ExecutionResult) -> str:
    metadata = result.metadata or {}
    status = metadata.get("http_status")
    if status in TRANSIENT_HTTP:
        return "http_" + str(status)
    raw = str(metadata.get("transport_status", "")).casefold()
    if "timeout" in raw:
        return "timeout"
    if "reset" in raw:
        return "connection_reset"
    return "ok" if result.success is True else "permanent_or_unknown"


def normalize_trace(result, receipts, seconds, executions):
    status = result["status"]
    answer = result.get("answer", "")
    if status == "give_answer":
        arguments = {"return_type": "give_answer", "final_answer": answer}
        thought = "<final>"
    else:
        status = "give_up_and_restart"
        arguments = {"return_type": "give_up_and_restart"}
        thought = "<give_up>"
    history = list(result["history"]) + [{
        "type": "call",
        "thought": thought,
        "api_identity": FINISH,
        "arguments": arguments,
    }]
    return {
        "status": status,
        "result": arguments,
        "history": history,
        "costs": {
            "model_seconds": seconds,
            "executor_seconds": executions["seconds"],
            "model_decisions": result["budget"]["generations"],
            "tool_calls_attempted": result["budget"]["calls"],
            "executions_succeeded": executions["succeeded"],
            "executions_failed": executions["failed"],
            "executions_without_receipt": 0,
            "validation_failures": sum(event.get("action") == "rejected_schema" for event in result["events"]),
            "validation_retries": sum(event.get("action") == "rejected_schema" for event in result["events"]),
            "generation_capacity_tokens": result["budget"]["tokens"],
        },
        "execution_receipts": receipts,
        "decisions": result["events"],
        "completion_controller": {
            "version": "bounded-requirement-coverage-routing-v4",
            "reason": result["reason"],
            "ledger": result["ledger"],
            "requirement_extraction_fallback": result["requirement_extraction_fallback"],
            "requirement_extraction_mode": result["requirement_extraction_mode"],
            "programmatic_finish_adapter": True,
            "semantic_success_certified": False,
        },
        "task_success_judged": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "tools", "queries", "executor-config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--executor-factory", required=True)
    parser.add_argument("--split", choices=["dev"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-calls", type=int, default=8)
    parser.add_argument("--max-thought-tokens", type=int, default=32)
    parser.add_argument("--max-argument-tokens", type=int, default=256)
    parser.add_argument("--max-answer-tokens", type=int, default=1024)
    parser.add_argument("--max-validation-retries", type=int, default=2)
    parser.add_argument("--train-api-identities", type=Path, required=True)
    parser.add_argument("--allow-seen-api", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.allow_seen_api or args.max_calls != 8 or args.max_validation_retries != 2:
        raise ValueError("Require the frozen seen-API I1 call/retry protocol")

    queries = load_queries(args.queries, args.split)
    tools = load_tools(args.tools, split=args.split)
    bindings = {row["api_identity"]: row.get("source_binding", {}) for row in read_jsonl(args.tools)}
    evaluator_names(tools, bindings)
    lineage = set(json.loads(args.train_api_identities.read_text()))
    if not ((set(tools) - {FINISH}) & lineage):
        raise ValueError("Expected explicit seen-API overlap")
    module_name, function_name = args.executor_factory.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    config = json.loads(args.executor_config.read_text())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    agent = load_agent(args.checkpoint, device=args.device)
    install_reader(agent, tools, "memory_fp32")
    if agent.reader_variant != "memory_fp32":
        raise ValueError("Memory reader was not installed")
    agent.register_tools(list(tools.values()), profile=True)
    policy = Counter()
    started = time.monotonic()
    with ExitStack() as stack:
        handle = stack.enter_context((args.output_dir / "episodes.jsonl").open("x"))
        export = stack.enter_context(SharedToolExport(args.output_dir, tools, bindings))
        for query in queries:
            base = factory(query_id=query["id"], config=config, tool_bindings=bindings)
            receipts = []
            executions = {"seconds": 0.0, "succeeded": 0, "failed": 0}

            def execute(identity, arguments):
                before = time.monotonic()
                result = base(identity, arguments)
                executions["seconds"] += time.monotonic() - before
                if not isinstance(result, ExecutionResult):
                    raise TypeError("Executor must provide a receipt")
                receipts.append(result.metadata)
                executions["succeeded" if result.success is True else "failed"] += 1
                return {"content": result.content, "transport": transport_status(result)}

            before = time.monotonic()
            result = run_native(agent, query["query"], tools, execute, budget=Budget(8, 55, 20064))
            trace = normalize_trace(result, receipts, time.monotonic() - before, executions)
            handle.write(json.dumps({"id": query["id"], "trace": trace}, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            export.append(query["id"], query["query"], trace)
            policy[trace["status"]] += 1
            policy["requirement_extraction_fallback"] += int(trace["completion_controller"]["requirement_extraction_fallback"])
            policy[trace["completion_controller"]["requirement_extraction_mode"]] += 1
            policy["answer_repair"] += int(any(event.get("reason") == "coverage_answer_repaired" for event in trace["decisions"]))
            policy["replan"] += int(any(event.get("action") == "replan" for event in trace["decisions"]))
            policy["evidence_answer"] += int(any(event.get("action") == "partial_answer" for event in trace["decisions"]))
            policy["selection_recovery"] += int(any(event.get("action") == "selection_recovery" for event in trace["decisions"]))
            policy["coverage_selection_recovery"] += int(any(event.get("action") == "coverage_selection_recovery" for event in trace["decisions"]))
            (args.output_dir / "progress.json").write_text(json.dumps({
                "completed_episodes": sum(policy[s] for s in ("give_answer", "give_up_and_restart")),
                "total_episodes": len(queries),
                "policy": dict(policy),
            }) + "\n")

    report = {
        "episodes": len(queries),
        "statuses": {key: policy[key] for key in ("give_answer", "give_up_and_restart")},
        "completion_policy": dict(policy),
        "seconds": time.monotonic() - started,
        "reader_mode": "memory_fp32",
        "controller": "bounded-requirement-coverage-routing-v4",
        "programmatic_finish_adapter": True,
        "task_success_judged": False,
        "official_sopr": False,
        "optimizer_updates": 0,
        "budgets": {"max_calls": 8, "max_decisions": 11, "max_generations": 55, "generation_capacity_tokens": 20064},
    }
    (args.output_dir / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance = {
        "checkpoint_manifest_sha256": sha256(args.checkpoint / "SHA256.json"),
        "tools_sha256": sha256(args.tools),
        "queries_sha256": sha256(args.queries),
        "executor_factory": args.executor_factory,
        "executor_source_sha256": sha256(Path(module.__file__)),
        "executor_config_sha256": sha256(args.executor_config),
        "runner_source_sha256": sha256(HERE),
        "completion_source_sha256": {
            name: sha256(ADAPTER / "variant" / "task_completion" / name)
            for name in ("completion.py", "runner.py", "native_adapter.py", "prompts.py")
        },
        "agent_source_sha256": sha256(Path(inspect.getfile(type(agent)))),
        "reader_mode": "memory_fp32",
        "split": "dev",
        "allow_seen_api": True,
        "optimizer_updates": 0,
    }
    (args.output_dir / "READER_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
