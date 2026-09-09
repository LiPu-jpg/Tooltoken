"""Run serial ToolBench episodes against an explicitly supplied executor.

Queries contain only id/query/split. The executor must use actual environment
feedback, never gold trajectory replay. This produces traces, not official SoPR.
"""
from __future__ import annotations
import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

from .toolbench_agent import run_serial_agent
from .toolbench_checkpoint import load_agent, sha256
from .toolbench_data import FINISH, compact, load_tools, read_jsonl


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
    def __init__(self, agent, thought_tokens: int, argument_tokens: int):
        self.agent, self.thought_tokens, self.argument_tokens = agent, thought_tokens, argument_tokens

    def _clock(self, synchronized): return self.agent._clock(synchronized)

    def decide(self, history, tools):
        return self.agent.decide(history, tools, max_thought_tokens=self.thought_tokens,
                                 max_argument_tokens=self.argument_tokens)


def run_panel(agent, tools, queries, factory, config, bindings, output: Path, *, max_calls=8,
              thought_tokens=1024, argument_tokens=1024):
    output.mkdir(parents=True, exist_ok=False)
    agent.register_tools(list(tools.values()), profile=True)
    policy = DecisionPolicy(agent, thought_tokens, argument_tokens)
    counts = Counter()
    with (output / "episodes.jsonl").open("x") as handle:
        for query in queries:
            # Reset the executor's episode state. No gold API, argument, result
            # or answer is provided to either policy or factory.
            execute = factory(query_id=query["id"], config=config, tool_bindings=bindings)
            if not callable(execute): raise TypeError("Executor factory must return (exact_identity, arguments) -> result")
            trace = run_serial_agent(policy, query["query"], tools, execute, max_calls=max_calls)
            handle.write(compact({"id": query["id"], "trace": trace}) + "\n"); handle.flush()
            counts[trace["status"]] += 1
    report = dict(episodes=len(queries), statuses=dict(counts), registration=agent.registration_profiles,
                  budgets=dict(max_calls=max_calls, thought_tokens=thought_tokens, argument_tokens=argument_tokens),
                  task_success_judged=False, official_sopr=False, oracle_tool_or_observation=False)
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
    p.add_argument("--max-argument-tokens", type=int, default=1024)
    args=p.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    queries=load_queries(args.queries,args.split)
    tools=load_tools(args.tools,split=args.split)
    metadata=json.loads((args.checkpoint/"agent.json").read_text())["metadata"]
    if not isinstance(metadata.get("train_api_identities"),list): raise ValueError("Missing training lineage")
    if (set(tools)-{FINISH}) & set(metadata["train_api_identities"]): raise ValueError("Development APIs overlap training")
    bindings={row["api_identity"]: row.get("source_binding",{}) for row in read_jsonl(args.tools)}
    module_name, function_name=args.executor_factory.split(":",1)
    module=importlib.import_module(module_name); factory=getattr(module,function_name)
    config=json.loads(args.executor_config.read_text())
    agent=load_agent(args.checkpoint,device=args.device)
    report=run_panel(agent,tools,queries,factory,config,bindings,args.output_dir,max_calls=args.max_calls,
        thought_tokens=args.max_thought_tokens,argument_tokens=args.max_argument_tokens)
    provenance=dict(checkpoint_manifest_sha256=sha256(args.checkpoint/"SHA256.json"),
        tools_sha256=sha256(args.tools),queries_sha256=sha256(args.queries),
        executor_factory=args.executor_factory,executor_source_sha256=sha256(Path(module.__file__)),
        executor_config_sha256=sha256(args.executor_config),split=args.split,optimizer_updates=0)
    (args.output_dir/"PROVENANCE.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"episodes":report["episodes"],"task_success_judged":False}))

if __name__=="__main__": main()
