"""Lossless serial trace export to the upstream ToolEval answer format.

This module does not implement or substitute an automatic judge.
"""
from __future__ import annotations

import copy
import hashlib

from .toolbench_data import FINISH, compact
from .toolbench_http import wire_bindings


def evaluator_names(tools, bindings):
    wire = wire_bindings(bindings)
    names, used = {FINISH: "Finish"}, {"Finish"}
    for identity in sorted(set(tools) - {FINISH}):
        if identity not in wire:
            raise ValueError("Missing evaluator/executor identity binding")
        target = wire[identity]
        name = target["api_name"] + "_for_" + target["tool_name"]
        # Cross-category endpoints may share a display name. Preserve exact
        # identities with a deterministic display suffix, never merge them.
        if name in used:
            name += "_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        if name in used:
            raise ValueError("Evaluator name collision")
        names[identity] = name
        used.add(name)
    return names


def convert_trace(query, trace, tools, bindings, *, method="NativeMemory.Serial", names=None, include_available=True):
    names = evaluator_names(tools, bindings) if names is None else names
    history = trace["history"]
    if not history or history[0] != {"type": "user", "content": query}:
        raise ValueError("Trace/query mismatch")
    available = [{"name": names[identity], "description": tool.document,
                  "parameters": copy.deepcopy(tool.parameters)} for identity, tool in sorted(tools.items())] if include_available else None
    # Upstream ExecutionGraph.convert_to_dict leaves system/user messages empty.
    nodes = [{"role": "system", "message": "", "next": []},
             {"role": "user", "message": "", "next": []}]
    index, final_answer, finished = 1, "", False
    while index < len(history):
        call = history[index]
        if not finished and call.get("type") == "validation_error":
            if (call.get("api_identity") not in names or type(call.get("retry")) is not int
                    or call["retry"] < 1 or not isinstance(call.get("error"), str)
                    or not isinstance(call.get("generated_arguments"), str)):
                raise ValueError("Invalid or unbound validation feedback")
            # Preserve runtime feedback in the exported graph without inventing
            # an executed tool, its return, or a model-generated final answer.
            nodes.append({"role": "user", "message": "Runtime validation feedback: " + compact(call), "next": []})
            index += 1
            continue
        if finished or call.get("type") != "call" or call.get("api_identity") not in names:
            raise ValueError("Nonserial or unbound call in trace")
        identity = call["api_identity"]
        if call.get("thought"):
            nodes.append({"role": "assistant", "message": call["thought"], "next": []})
        response = ""
        if identity == FINISH:
            finished = True
            tools[FINISH].validate_arguments(call["arguments"])
            if call["arguments"]["return_type"] == "give_answer":
                final_answer = call["arguments"]["final_answer"]
            index += 1
        else:
            if index + 1 >= len(history):
                raise ValueError("Call is missing its actual observation")
            obs = history[index + 1]
            if obs.get("type") != "observation" or obs.get("api_identity") != identity:
                raise ValueError("Observation is bound to a different API")
            response = obs["content"] if isinstance(obs["content"], str) else compact(obs["content"])
            index += 2
        nodes.append({"role": "tool", "message": {"name": names[identity],
            "arguments": compact(call["arguments"]), "response": response}, "next": []})
    if finished:
        expected = history[-1]["arguments"]["return_type"]
        if trace["status"] != expected:
            raise ValueError("Finish status disagrees with trace")
    elif trace["status"] in {"give_answer", "give_up_and_restart"}:
        raise ValueError("Never fabricate Finish for an interrupted episode")
    for left, right in zip(nodes, nodes[1:]):
        left["next"] = [right]
    return {"query": query, **({"available_tools": available} if include_available else {}),
            "answer": {"method": method, "total_steps": len(nodes),
                       "final_answer": final_answer, "answer_details": [nodes[0]]}}
