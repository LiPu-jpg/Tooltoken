"""Measure compiler-only overhead on synthetic states, without loading Qwen."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .model import TokenResamplerMemory
from .structured_memory import StructuredResamplerMemory


def measure(*, kind, slots, hidden_size, width, heads, rank, tokens, fields, device, repetitions):
    torch.manual_seed(17)
    states = torch.randn(1, tokens, hidden_size, device=device)
    mask = torch.ones(1, tokens, dtype=torch.bool, device=device)
    field_mask = torch.zeros(1, fields, tokens, dtype=torch.bool, device=device)
    for index in range(fields):
        field_mask[:, index, index * tokens // fields:(index + 1) * tokens // fields] = True
    compiler = (TokenResamplerMemory(hidden_size, rank, slots, 1.0) if kind == "legacy" else
                StructuredResamplerMemory(hidden_size, width, slots, 2, heads, 1.0)).to(device).eval()
    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
    def forward():
        return compiler(states, mask) if kind == "legacy" else compiler(states, mask, field_mask)
    elapsed = []
    with torch.no_grad():
        for _ in range(2):
            forward()
        sync()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repetitions):
            sync()
            start = time.perf_counter()
            output = forward()
            sync()
            elapsed.append(time.perf_counter() - start)
    return dict(kind=kind, slots=slots, parameters=sum(p.numel() for p in compiler.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for p in compiler.parameters()),
        output_bytes=output.numel() * output.element_size(), output_dtype=str(output.dtype),
        median_seconds=statistics.median(elapsed), mean_seconds=statistics.mean(elapsed), all_seconds=elapsed,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--fields", type=int, default=16)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--cpu-threads", type=int, default=1)
    args = p.parse_args()
    if min(args.tokens, args.fields, args.repetitions, args.cpu_threads) < 1 or args.fields > args.tokens:
        raise ValueError("Need positive counts and fields <= tokens")
    if args.output_dir.exists():
        raise FileExistsError("Profile output must be new")
    torch.set_num_threads(args.cpu_threads)
    records = [measure(kind=kind, slots=slots, hidden_size=args.hidden_size, width=args.width, heads=args.heads,
                       rank=args.rank, tokens=args.tokens, fields=args.fields, device=args.device,
                       repetitions=args.repetitions) for kind, slots in (("legacy", 8), ("structured", 8), ("structured", 16))]
    report = dict(scope="compiler-only synthetic hidden states; excludes Qwen, tokenizer, schema parsing and caller",
        config=vars(args), python=platform.python_version(), torch=torch.__version__, machine=platform.machine(),
        optimizer_updates=0, backbone_forwards=0, records=records)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "PROFILE.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
