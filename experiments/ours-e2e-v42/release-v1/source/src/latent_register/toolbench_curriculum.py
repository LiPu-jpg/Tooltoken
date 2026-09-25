"""Shared-Qwen request -> dynamic selection -> bound memory -> args/repair.

Fixed control strings use the existing tokenizer. No per-tool token parameters.
All ranks run the same module traversal, including zero-weight masked labels.
"""
from __future__ import annotations

import json
import torch
from torch.nn import functional as F

from .toolbench_agent import (ToolBenchAgent, Decision, GenerationFailure, MEMORY_MARKER, SYSTEM)
from .toolbench_data import FINISH, compact, strict_json, alias_map
from .toolbench_curriculum_data import CONTROL, clean_history, candidate_ids, information_targets

INTERFACE = "native-toolbench-intent-read-recover-v27"
TASK = {
    "intent": "Write a short plan for the next action needed to satisfy the user, based on the causal history. You may identify the intended tool and inputs when supported by that history. A plan is not an observation: do not claim an unexecuted tool has returned a result or supply a final answer.",
    "control": "Choose exactly one control string and nothing else: <tool_request> to obtain more information using a tool; <final> only when the available evidence addresses every requested part; <give_up> when you cannot complete the request.",
    "arguments": "Return only the complete JSON object of inputs for the selected tool. Read its field rules from its information; obtain values from the history. Do not add a plan.",
    "repair": "Inspect the failed input and selected tool. If this is the appropriate tool and its inputs can be obtained from history, return the complete corrected JSON object. If the selected tool is inappropriate or necessary information must first be obtained elsewhere, output exactly <reselect>. Do not invent missing values.",
    "information": "Read the supplied tool information. Return the requested fact as JSON, without guessing absent facts.",
    "answer": "Answer every requested part using the observations already obtained. Distinguish supported results from missing or unavailable information. Do not claim missing information was obtained, invent tool results, or call another tool.",
}


def render_parts(tokenizer, history, task, *, condition="memory", document="", previous="", error="", fact="capability", history_format="legacy", next_intent=""):
    if task not in TASK:
        raise ValueError("Unknown curriculum task")
    if task == "information" and history:
        raise ValueError("Document readback cannot receive a query or answer")
    body = "" if task == "information" else "Causal history JSON:\n" + compact(clean_history(history, history_format))
    if task == "information":
        body += "\nRequested fact: " + fact
    if next_intent:
        if task != "arguments":
            raise ValueError("Current intent is only a condition for argument generation")
        body += "\nCurrent causal subtask (not evidence for parameter values):\n" + next_intent
    if task == "repair":
        body += "\nPrevious complete or malformed JSON:\n" + previous + "\nActual validation error:\n" + error
    if MEMORY_MARKER in body or MEMORY_MARKER in document:
        raise ValueError("Reserved injection marker in source text")
    uses_memory = task in {"arguments", "repair", "information"}
    if uses_memory:
        if condition == "memory":
            body += "\nSelected tool information:\n" + MEMORY_MARKER
        elif condition == "full_document":
            body += "\nSelected tool information:\n" + document
        elif condition != "query_only":
            raise ValueError("Unsupported condition")
    body += "\nTask: " + TASK[task]
    rendered = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
        {"role": "user", "content": body}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    parts = rendered.split(MEMORY_MARKER)
    if len(parts) != (2 if uses_memory and condition == "memory" else 1):
        raise ValueError("Memory marker must occur exactly once for the reader")
    return [tokenizer.encode(part, add_special_tokens=False) for part in parts]


def json_character_groups(text):
    """Key/value/structural character assignment, preserving serialized targets."""
    strict_json(text)
    groups = [2] * len(text)
    i = 0
    while i < len(text):
        if text[i] == '"':
            start = i
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            end = i
            while i < len(text) and text[i].isspace():
                i += 1
            kind = 0 if i < len(text) and text[i] == ":" else 1
            groups[start:end] = [kind] * (end - start)
        elif text[i] in "{}[],:" or text[i].isspace():
            i += 1
        else:
            start = i
            while i < len(text) and text[i] not in "{}[],:" and not text[i].isspace():
                i += 1
            groups[start:i] = [1] * (i - start)
    return groups


class CurriculumAgent(ToolBenchAgent):
    interface_version = INTERFACE
    history_format = "legacy"

    @torch.no_grad()
    def register_tools(self, tools, *, profile=False):
        return super().register_tools([tool for tool in tools if tool.api_identity != FINISH], profile=profile)

    def prefix(self, history, task, thought="", *, memory=None, document="", condition=None,
               previous="", error="", fact="capability", next_intent=""):
        if thought:
            raise ValueError("This interface never accepts a gold plan")
        parts = render_parts(self.tokenizer, history, task, condition=condition or self.condition,
                             document=document, previous=previous, error=error, fact=fact, history_format=self.history_format, next_intent=next_intent)
        embedding = self.backbone.get_input_embeddings()
        chunks = [embedding(self._ids(parts[0]))]
        if len(parts) == 2:
            if memory is None or memory.shape != (self.slots, embedding.weight.shape[1]):
                raise ValueError("Inject the full bound eight-slot memory")
            chunks += [memory.to(chunks[0].dtype), embedding(self._ids(parts[1]))]
        result = torch.cat(chunks).unsqueeze(0)
        if not 0 < result.shape[1] <= self.limits.context:
            raise ValueError(f"Complete curriculum prefix exceeds context: {result.shape[1]}")
        return result

    def request_prefix(self, history, intent=""):
        if intent:
            prefix = self.prefix(history, "intent")
            tokens = self.tokenizer.encode(intent, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            result = torch.cat([prefix, self.backbone.get_input_embeddings()(self._ids(tokens)).unsqueeze(0)], 1)
            if result.shape[1] > self.limits.context:
                raise ValueError("Complete intent exceeds context")
            return result
        prefix = self.prefix(history, "control")
        request = self.backbone.get_input_embeddings()(self._ids(self.tokenizer.encode(CONTROL["tool"], add_special_tokens=False))).unsqueeze(0)
        result = torch.cat([prefix, request], 1)
        if result.shape[1] > self.limits.context:
            raise ValueError("Complete request exceeds context")
        return result  # Last complete request token, before any candidate/target.

    def selection_scores(self, history, thought, rows):
        query = self._hidden(inputs_embeds=self.request_prefix(history, thought), use_cache=False).last_hidden_state[:, -1].float()
        return query @ rows.float().T

    def argument_loss(self, prefix, arguments):
        text = compact(arguments)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"] + [self.tokenizer.eos_token_id]
        if len(ids) > self.limits.target or prefix.shape[1] + len(ids) > self.limits.context:
            raise ValueError("Complete JSON target exceeds limits; refusing truncation")
        char_groups = json_character_groups(text)
        memberships = []
        for left, right in encoded["offset_mapping"]:
            memberships.append([sum(g == kind for g in char_groups[left:right]) / max(1, right-left) for kind in range(3)])
        memberships.append([0., 0., 1.])  # EOS is structural supervision.
        target = self._ids(ids)
        inputs = torch.cat([prefix, self.backbone.get_input_embeddings()(target).unsqueeze(0)], 1)
        hidden = self._hidden(inputs_embeds=inputs, use_cache=False).last_hidden_state
        from .training_memory import target_cross_entropy
        ce = target_cross_entropy(self.backbone.get_output_embeddings(), hidden[:, prefix.shape[1]-1:prefix.shape[1]-1+len(ids)], target)
        member = torch.tensor(memberships, device=ce.device, dtype=ce.dtype)
        means = (ce[:, None] * member).sum(0) / member.sum(0).clamp_min(1.)
        # Fixed denominator: empty objects do not receive the same aggregate
        # weight as objects containing actual key/value facts.
        return (means * ce.new_tensor([1., 1., .2])).sum() / 2.2

    def recovery_loss(self, record, tools):
        recovery = record.get("recovery")
        if not recovery:
            return next(self.parameters()).new_zeros(())
        tool = tools[recovery["selected"]]
        _, memories = self.compile([tool], memory_indices=[0])
        prefix = self.prefix(record["history"], "repair", memory=memories[0],
            document=tool.registration_document, previous=recovery["previous"], error=recovery["error"])
        return self.nll(prefix, "<reselect>")

    def forward(self, records, tools, *, hard=None, stage=1, candidate_count=16, seed=17, information_index=None, **unused):
        if stage not in (1, 2) or not records:
            raise ValueError("Choose a nonempty, synchronized curriculum stage")
        losses = {key: [] for key in ("selection", "arguments", "information", "control", "repair", "answer", "intent")}
        for record in records:
            ids = candidate_ids(record, tools, hard or {}, candidate_count, seed)
            gold = ids.index(record["selected"])
            tool = tools[record["selected"]]
            rows, memories = self.compile([tools[key] for key in ids])
            memory = memories[gold]
            mask = record["masks"]
            # Fixed traversal even for terminal/masked records. No wrong API is
            # ever paired with another API's positive parameter target.
            intent = record.get("next_intent", "") if record.get("intent_supervision") else ""
            scores = self.selection_scores(record["history"], intent, rows)
            losses["intent"].append(self.nll(self.prefix(record["history"], "intent"), intent) if intent else rows.new_zeros(()))
            losses["selection"].append(F.cross_entropy(scores, self._ids([gold])) * mask["selection"])
            args = record["arguments"] if mask["arguments"] else {}
            args_prefix = self.prefix(record["history"], "arguments", memory=memory, document=tool.registration_document, next_intent=intent)
            losses["arguments"].append(self.argument_loss(args_prefix, args) * mask["arguments"])
            from .schema_curriculum import targets as schema_targets, request as schema_request
            targets = schema_targets(tool)
            offset = (record["id"][1] if isinstance(record["id"][1], int) else 0) + seed
            fact = targets[(offset if information_index is None else information_index) % len(targets)]
            request = schema_request(fact)
            info_prefix = self.prefix([], "information", memory=memory, document=tool.registration_document, fact=request)
            losses["information"].append(self.nll(info_prefix, compact(fact)))
            if stage == 2:
                losses["control"].append(self.nll(self.prefix(record["history"], "control"), CONTROL[record["mode"]]) * mask["control"])
                repair = record.get("repair")
                repaired = repair["target"] if repair else {}
                repair_prefix = self.prefix(record["history"], "repair", memory=memory, document=tool.registration_document,
                    previous=repair["previous"] if repair else "{}", error=repair["error"] if repair else "No repair supervision for this record.")
                losses["repair"].append(self.argument_loss(repair_prefix, repaired) * float(repair is not None) + self.recovery_loss(record, tools))
                losses["answer"].append(self.nll(self.prefix(record["history"], "answer"), record["answer"]) * mask["answer"])
            else:
                for name in ("control", "repair", "answer"):
                    losses[name].append(rows.new_zeros(()))
        result = {name: torch.stack(values).mean() for name, values in losses.items()}
        result["loss"] = (result["selection"] + result["arguments"] + .6 * result["information"]
                          + result["control"] + 1.0 * result["repair"] + result["answer"] + result["intent"])
        return result

    def select_from_ranking(self, history, intent, ranking):
        return ranking[0], {}

    @torch.no_grad()
    def decide(self, history, tools, *, max_thought_tokens=32, max_argument_tokens=256, max_answer_tokens=1024):
        alias_map(tools)
        ordinary = {key: value for key, value in tools.items() if key != FINISH}
        if not ordinary or FINISH not in tools:
            raise ValueError("Need ordinary tools plus the protocol Finish adapter")
        self.register_tools(list(ordinary.values()))
        repair = history[-1] if history and history[-1].get("type") == "validation_error" else None
        causal = clean_history(history)
        budgets = {"interface": INTERFACE, "repair_keeps_identity": repair is not None}
        if repair:
            identity = repair["api_identity"]
            if identity not in ordinary:
                raise ValueError("Cannot repair an unbound or terminal action")
            ranking = (identity,)
            mode = CONTROL["tool"]
            prefix = self.prefix(causal, "repair", memory=self._registry[identity][2],
                document=tools[identity].registration_document, previous=repair["generated_arguments"], error=repair["error"])
        else:
            control_prefix = self.prefix(causal, "control")
            generated = self.generate(control_prefix, max_new_tokens=max_thought_tokens)
            mode = generated.text.strip()
            budgets["control"] = generated.budget()
            if generated.truncated or mode not in CONTROL.values():
                raise GenerationFailure("Incomplete or invalid shared control action", stage="control", text=generated.text,
                                        reason="invalid_control", generation=generated)
            if mode == CONTROL["give_up"]:
                return Decision(FINISH, {"return_type": "give_up_and_restart"}, mode, (), budgets)
            if mode == CONTROL["final"]:
                final = self.generate(self.prefix(causal, "answer"), max_new_tokens=max_answer_tokens)
                if final.truncated or not final.text.strip():
                    raise GenerationFailure("Final answer incomplete", stage="answer", text=final.text, reason="truncated", generation=final)
                budgets["answer"] = final.budget()
                return Decision(FINISH, {"return_type": "give_answer", "final_answer": final.text}, mode, (), budgets)
            identities = sorted(ordinary)
            rows = torch.stack([self._registry[key][1] for key in identities])
            intent = self.generate(self.prefix(causal, "intent"), max_new_tokens=96)
            if intent.truncated or not intent.text.strip():
                raise GenerationFailure("Incomplete next intent", stage="intent", text=intent.text, reason="invalid_intent", generation=intent)
            budgets["intent"] = intent.budget()
            budgets["next_intent"] = intent.text
            scores = self.selection_scores(causal, intent.text, rows)[0]
            ranking = tuple(identities[i] for i in scores.argsort(descending=True).tolist())
            identity, reader_budget = self.select_from_ranking(causal, intent.text, ranking)
            budgets.update(reader_budget)
            prefix = self.prefix(causal, "arguments", memory=self._registry[identity][2], document=tools[identity].registration_document, next_intent=intent.text)
        generated = self.generate(prefix, max_new_tokens=max_argument_tokens)
        if repair and not generated.truncated and generated.text.strip() == "<reselect>":
            recovery = {"type": "recovery", "api_identity": identity,
                        "previous_arguments": repair["generated_arguments"], "error": repair["error"],
                        "action": "<reselect>"}
            decision = self.decide(causal + [recovery], tools, max_thought_tokens=max_thought_tokens,
                                   max_argument_tokens=max_argument_tokens, max_answer_tokens=max_answer_tokens)
            decision.generation_budgets["recovery"] = {"action": "<reselect>", "generation": generated.budget()}
            return decision
        budgets["arguments"] = generated.budget()
        budgets["argument_prefix_tokens"] = prefix.shape[1]
        if generated.truncated:
            raise GenerationFailure("Arguments truncated", stage="arguments", text=generated.text, identity=identity,
                                    thought=mode, reason="truncated", generation=generated, ranking=ranking)
        try:
            arguments = tools[identity].validate_arguments(strict_json(generated.text))
        except ValueError as exc:
            raise GenerationFailure(str(exc), stage="arguments", text=generated.text, identity=identity,
                                    thought=mode, reason="invalid_arguments", generation=generated, ranking=ranking) from exc
        return Decision(identity, arguments, mode, ranking, budgets)
