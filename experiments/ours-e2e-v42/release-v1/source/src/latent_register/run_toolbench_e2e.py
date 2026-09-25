"""Run serial ToolBench episodes against an explicitly supplied executor.

Queries contain only id/query/split. The executor must use actual environment
feedback, never gold trajectory replay. This produces traces, not official SoPR.
"""
from __future__ import annotations
import argparse
import importlib
import inspect
from contextlib import ExitStack
import json
from collections import Counter
from pathlib import Path

from .toolbench_agent import run_serial_agent
from .toolbench_checkpoint import load_agent, sha256
from .toolbench_data import FINISH, compact, load_tools, read_jsonl
from .toolbench_eval_format import convert_trace, evaluator_names
from .toolbench_shared_export import SharedToolExport


def load_queries(path: Path, split: str) -> list[dict]:
    rows, seen = [], set()
    for row in read_jsonl(path):
        if set(row) != {"id", "query", "split"} or row["split"] != split:
            raise ValueError("Supply only explicit development id/query/split, without gold answers or tools")
        if not isinstance(row["id"], str) or row["id"] in seen or not isinstance(row["query"], str) or not row["query"].strip():
            raise ValueError("Invalid or duplicate episode")
        seen.add(row["id"]); rows.append(row)
    if not rows: raise ValueError("Empty episode panel")
    return rows


class DecisionPolicy:
    def __init__(self, agent, thought_tokens: int, argument_tokens: int, answer_tokens: int = 1024):
        self.agent, self.thought_tokens, self.argument_tokens = agent, thought_tokens, argument_tokens
        self.answer_tokens = answer_tokens

    def _clock(self, synchronized): return self.agent._clock(synchronized)

    def decide(self, history, tools):
        extra = {"max_answer_tokens": self.answer_tokens} if getattr(self.agent, "interface_version", "") in {"native-toolbench-intent-read-recover-v27", "native-toolbench-top5-memory-reader-v40", "native-toolbench-top5-selected-document-v41", "native-toolbench-task-state-hybrid-v42"} else {}
        return self.agent.decide(history, tools, max_thought_tokens=self.thought_tokens,
                                 max_argument_tokens=self.argument_tokens, **extra)


def run_panel(agent, tools, queries, factory, config, bindings, output: Path, *, max_calls=8,
              thought_tokens=32, argument_tokens=256, answer_tokens=1024, export_tooleval=False,
              max_validation_retries=0):
    if type(max_validation_retries) is not int or max_validation_retries < 0:
        raise ValueError("max_validation_retries must be a nonnegative integer")
    output.mkdir(parents=True, exist_ok=False)
    if export_tooleval:
        names = evaluator_names(tools, bindings)
        (output / "evaluation_identity_map.json").write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n")
    agent.register_tools(list(tools.values()), profile=True)
    policy = DecisionPolicy(agent, thought_tokens, argument_tokens, answer_tokens)
    counts = Counter()
    with ExitStack() as stack:
        handle = stack.enter_context((output / "episodes.jsonl").open("x"))
        export = stack.enter_context(SharedToolExport(output, tools, bindings)) if export_tooleval else None
        for query in queries:
            # Reset the executor's episode state. No gold API, argument, result
            # or answer is provided to either policy or factory.
            execute = factory(query_id=query["id"], config=config, tool_bindings=bindings)
            if not callable(execute): raise TypeError("Executor factory must return (exact_identity, arguments) -> result")
            trace = run_serial_agent(policy, query["query"], tools, execute, max_calls=max_calls,
                                     max_validation_retries=max_validation_retries)
            handle.write(compact({"id": query["id"], "trace": trace}) + "\n"); handle.flush()
            if export_tooleval:
                export.append(query["id"], query["query"], trace)
            counts[trace["status"]] += 1
            (output / "progress.json").write_text(compact({"completed_episodes": sum(counts.values()),
                                                          "total_episodes": len(queries)}) + "\n")
    report = dict(episodes=len(queries), statuses=dict(counts), registration=agent.registration_profiles,
                  budgets=dict(max_calls=max_calls, thought_tokens=thought_tokens, argument_tokens=argument_tokens, answer_tokens=answer_tokens,
                               max_validation_retries=max_validation_retries),
                  task_success_judged=False, official_sopr=False, oracle_tool_or_observation=False,
                  backend_kind=config.get("backend_kind", "explicit_external_factory"),
                  backend_revision=config.get("backend_revision"),
                  adaptation_gate_passed=None, adaptation_gate_status="not_assessed_by_episode_runner")
    (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tools", "queries", "executor-config", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--executor-factory", required=True, help="Reviewed Python module:function; returns an executor for each episode")
    p.add_argument("--split", choices=["dev", "validation"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-thought-tokens", type=int, default=1024)
    p.add_argument("--max-argument-tokens", type=int, default=256)
    p.add_argument("--max-answer-tokens", type=int, default=1024)
    p.add_argument("--max-validation-retries", type=int, default=0,
                   help="Total bounded local validation feedback retries per episode; disabled by default")
    p.add_argument("--train-api-identities", type=Path, default=None,
                   help="Sidecar JSON list of training-supervised API identities; required when "
                        "the checkpoint metadata predates the train_api_identities lineage field")
    p.add_argument("--allow-seen-api", action="store_true",
                   help="Explicitly permit development APIs that overlap the training registry "
                        "(seen-API protocol, e.g. official full-library ToolBench). Default keeps "
                        "the strict train/dev API-disjoint assertion.")
    args=p.parse_args()
    if args.max_validation_retries < 0:
        p.error("--max-validation-retries must be nonnegative")
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    queries=load_queries(args.queries,args.split)
    tools=load_tools(args.tools,split=args.split)
    metadata=json.loads((args.checkpoint/"agent.json").read_text())["metadata"]
    lineage=metadata.get("train_api_identities")
    if not isinstance(lineage,list):
        if args.train_api_identities is None:
            raise ValueError("Missing training lineage")
        lineage=json.loads(args.train_api_identities.read_text())
        if not isinstance(lineage,list) or not all(isinstance(x,str) for x in lineage):
            raise ValueError("Training lineage sidecar must be a JSON list of identity strings")
    overlap = (set(tools)-{FINISH}) & set(lineage)
    if overlap and not args.allow_seen_api:
        raise ValueError("Development APIs overlap training; rerun with --allow-seen-api only for an "
                         "explicitly authorized seen-API protocol ({} identities)".format(len(overlap)))
    if overlap and args.split != "dev":
        raise ValueError("Seen-API override is restricted to the development split")
    bindings={row["api_identity"]: row.get("source_binding",{}) for row in read_jsonl(args.tools)}
    module_name, function_name=args.executor_factory.split(":",1)
    module=importlib.import_module(module_name); factory=getattr(module,function_name)
    config=json.loads(args.executor_config.read_text())
    # Fail configuration/binding checks before allocating the 8B model.
    evaluator_names(tools, bindings)
    if not callable(factory(query_id=queries[0]["id"], config=config, tool_bindings=bindings)):
        raise TypeError("Executor factory must return a callable")
    agent=load_agent(args.checkpoint,device=args.device)
    report=run_panel(agent,tools,queries,factory,config,bindings,args.output_dir,max_calls=args.max_calls,
        thought_tokens=args.max_thought_tokens,argument_tokens=args.max_argument_tokens,answer_tokens=args.max_answer_tokens,export_tooleval=True,
        max_validation_retries=args.max_validation_retries)
    provenance=dict(checkpoint_manifest_sha256=sha256(args.checkpoint/"SHA256.json"),
        tools_sha256=sha256(args.tools),queries_sha256=sha256(args.queries),
        executor_factory=args.executor_factory,executor_source_sha256=sha256(Path(module.__file__)),
        executor_config_sha256=sha256(args.executor_config),split=args.split,optimizer_updates=0,
        allow_seen_api=bool(args.allow_seen_api),seen_api_overlap_count=len(overlap),
        runner_source_sha256=sha256(Path(__file__)),
        agent_source_sha256=sha256(Path(inspect.getfile(type(agent)))),
        base_agent_source_sha256=sha256(Path(__file__).with_name("toolbench_agent.py")),
        model_interface=getattr(agent, "interface_version", "legacy_thought_select_args"),
        format_source_sha256=sha256(Path(__file__).with_name("toolbench_eval_format.py")))
    (args.output_dir/"PROVENANCE.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"episodes":report["episodes"],"task_success_judged":False}))

if __name__=="__main__": main()
