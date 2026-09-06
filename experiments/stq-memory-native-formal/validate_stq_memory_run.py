"""Validate saved artifacts, not whether a model scored sufficiently high."""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--phase", choices=["stage1", "stage2"], required=True)
    ap.add_argument("--evaluation", type=Path)
    args = ap.parse_args()
    p = args.checkpoint
    results = json.loads((p / "results.json").read_text())
    assert results["completed_epochs"] == 5
    assert results["completed_steps"] == 6550
    assert math.isfinite(results["recent_train_loss"])
    with safe_open(p / "lora/adapter_model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        assert keys
        for k in keys:
            assert torch.isfinite(f.get_tensor(k)).all(), k
    for name in ["compiler.pt"] + (["memory_compiler.pt"] if args.phase == "stage2" else []):
        state = torch.load(p / name, map_location="cpu", weights_only=True)
        assert state and all(torch.isfinite(t).all() for t in state.values())
        if name == "memory_compiler.pt":
            assert torch.count_nonzero(state["value_up.weight"]) > 0, "Memory residual remained at zero initialization"
    audit = results["physical_token_audit"]
    assert audit["evaluation_ids_seen_during_training"] == 0
    assert audit["evaluation_input_row_max_change"] == 0
    assert audit["evaluation_output_row_max_change"] == 0
    assert not list(p.rglob('*optim_states*'))
    if args.evaluation:
        d = json.loads((args.evaluation / "summary.json").read_text())
        assert d["metadata"]["queries"] == 1066 and d["metadata"]["documents"] == 837
        assert d["metadata"]["memory_slots"] == 8 and not d["metadata"]["smoke_only"]
        assert all(math.isfinite(v) for v in d["retrieval"].values())
        assert all(math.isfinite(v) for v in d["call"].values())
    print(json.dumps({"validated": str(p), "phase": args.phase, "evaluation": str(args.evaluation)}))


if __name__ == "__main__":
    main()
