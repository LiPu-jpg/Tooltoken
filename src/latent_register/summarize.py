from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(run_dir: Path) -> dict[str, Any]:
    result_paths = sorted(run_dir.glob("seed-*/results.json"))
    if not result_paths:
        raise ValueError(f"No seed results found under {run_dir}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]

    summary: dict[str, Any] = {
        "num_seeds": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "eval_tools_per_seed": [run["split"]["eval_tools"] for run in runs],
        "tool_overlap": [run["split"]["tool_overlap"] for run in runs],
        "metrics": {},
    }
    for metric in ("top1", "top5", "mrr", "mean_rank"):
        raw = [run["raw_frozen_cosine"][metric] for run in runs]
        registered = [run["registered_output_rows"][metric] for run in runs]
        deltas = [right - left for left, right in zip(raw, registered)]
        summary["metrics"][metric] = {
            "raw_mean": statistics.mean(raw),
            "raw_population_std": statistics.pstdev(raw),
            "registered_mean": statistics.mean(registered),
            "registered_population_std": statistics.pstdev(registered),
            "paired_delta_mean": statistics.mean(deltas),
            "paired_deltas": deltas,
        }
    summary["slot_audit"] = {
        "max_score_delta": max(run["unseen_slot_audit"]["max_score_delta"] for run in runs),
        "prediction_disagreements": sum(
            run["unseen_slot_audit"]["prediction_disagreements"] for run in runs
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    destination = args.run_dir / "summary.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
