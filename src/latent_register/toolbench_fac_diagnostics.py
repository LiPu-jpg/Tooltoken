"""Separate upstream FAC fallbacks from actual semantic judgments.

Missing model answers are failed episodes. They are not judgments produced by
the FAC model, even though upstream exports an Unsolved fallback for them.
"""
from collections import Counter
import re


def summarize_fac(rows, answers):
    expected = Counter(item["query"] for item in answers.values())
    if Counter(row["query"] for row in rows) != expected:
        raise ValueError("FAC rows do not match the exported query multiset")
    source_answers = {}
    for item in answers.values():
        source_answers.setdefault(item["query"], []).append(item["answer"]["final_answer"])
    if any(len(set(values)) > 1 for values in source_answers.values()):
        raise ValueError("Ambiguous duplicate FAC query with different answers")
    records = []
    for row in rows:
        query = row["query"]
        original = source_answers[query][0]
        control = query.startswith("__adapter_control_")
        fallback = row.get("prompt", "").strip() == "an error has occured"
        match = re.search(r"answer\s+status\s*:?\s*(solved|unsolved)\b",
                          row.get("evaluation", ""), re.IGNORECASE)
        if not isinstance(original, str) or not original.strip():
            kind, outcome = "missing_final_answer", 0
        elif fallback or not match or row.get("final_answer") != original:
            kind, outcome = "judge_or_export_error", None
        else:
            kind, outcome = "semantic_judgment", int(match.group(1).lower() == "solved")
        records.append({"query": query, "control": control, "kind": kind,
                        "task_outcome": outcome, "upstream_fallback": fallback})
    native = [r for r in records if not r["control"]]
    controls = [r for r in records if r["control"]]
    control_scores = {r["query"].split()[0]: r["task_outcome"] for r in controls}
    labels = [r["task_outcome"] for r in native if r["kind"] == "semantic_judgment"]
    outcomes = [r["task_outcome"] for r in native]
    return {"records": records, "native_rows": len(native),
            "native_semantic_judgments": len(labels),
            "native_missing_final_answers": sum(r["kind"] == "missing_final_answer" for r in native),
            "native_judge_or_export_errors": sum(r["kind"] == "judge_or_export_error" for r in native),
            "native_semantic_score_mean": sum(labels) / len(labels) if labels else None,
            "task_outcome_mean": sum(outcomes) / len(outcomes)
            if outcomes and all(x is not None for x in outcomes) else None,
            "judge_controls_valid": len(controls) == 2
            and all(r["kind"] == "semantic_judgment" for r in controls)
            and control_scores == {"__adapter_control_positive__": 1, "__adapter_control_negative__": 0},
            "official_sopr": False}
