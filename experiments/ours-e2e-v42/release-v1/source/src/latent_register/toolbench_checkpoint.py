"""Self-contained shared-backbone checkpoints, including nonpersistent buffers."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .toolbench_agent import Limits, ToolBenchAgent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_base_reference(reference):
    root = Path(reference['path'])
    if not root.is_absolute() or not reference['files']:
        raise ValueError('Missing pinned adapter base')
    for relative, expected in reference['files'].items():
        path = root / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Invalid base file path')
        algorithm = expected['algorithm']
        if algorithm == 'sha256':
            actual = sha256(path)
        elif algorithm == 'git-blob-sha1':
            digest = hashlib.sha1((f'blob {path.stat().st_size}\0').encode())
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
            actual = digest.hexdigest()
        else:
            raise ValueError('Unknown base digest')
        if actual != expected['digest']:
            raise ValueError('Adapter base bytes changed: ' + relative)
    return root


def save_agent(agent: ToolBenchAgent, path: str | Path, *, metadata: dict,
               state_dict: dict[str, torch.Tensor] | None = None) -> None:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    adapter = hasattr(agent.backbone, 'peft_config')
    if adapter and not hasattr(agent, '_native_base_reference'):
        raise ValueError('Adapter export requires authenticated base provenance')
    state = agent.state_dict() if state_dict is None else state_dict
    backbone_state = {key[len("backbone."):]: value for key, value in state.items() if key.startswith("backbone.")}
    agent.backbone.save_pretrained(output / "backbone", state_dict=backbone_state, safe_serialization=True)
    agent.tokenizer.save_pretrained(output / "backbone")
    extra = {key: value.detach().cpu() for key, value in state.items() if not key.startswith("backbone.")}
    # RoPE inv_freq may not be in state_dict, and a BF16 -> FP32 round trip is
    # not reversible. Preserve the actual runtime values and their dtype.
    buffers = {name: value.detach().cpu() for name, value in agent.backbone.named_buffers()}
    torch.save({"compilers": extra, "backbone_buffers": buffers}, output / "runtime.pt")
    config = {"version": 4 if adapter else (3 if hasattr(agent, "interface_version") else 2),
              "interface": getattr(agent, "interface_version", None),
              "history_format": getattr(agent, "history_format", "legacy"),
              "rank": agent.rank, "slots": agent.slots, "memory_compiler": agent.memory_config,
              "condition": agent.condition, "limits": asdict(agent.limits), "metadata": metadata,
              "scale_initialization": agent.scale_initialization}
    if adapter:
        config['base_reference'] = agent._native_base_reference
        config['storage'] = 'shared_adapter_plus_compilers_with_pinned_external_base'
    (output / "agent.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    manifest = {str(item.relative_to(output)): sha256(item) for item in sorted(output.rglob("*")) if item.is_file()}
    (output / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_agent(path: str | Path, *, device: str = "cpu", torch_dtype: Any = "auto") -> ToolBenchAgent:
    root = Path(path)
    manifest = json.loads((root / "SHA256.json").read_text())
    for relative, expected in manifest.items():
        item = root / relative
        if not item.resolve().is_relative_to(root.resolve()) or sha256(item) != expected:
            raise ValueError(f"Checkpoint checksum failed: {relative}")
    config = json.loads((root / "agent.json").read_text())
    if config["version"] not in {1, 2, 3, 4}:
        raise ValueError("Unsupported checkpoint interface")
    tokenizer = AutoTokenizer.from_pretrained(root / "backbone", local_files_only=True)
    backbone_root = verify_base_reference(config['base_reference']) if config['version'] == 4 else root / 'backbone'
    backbone = AutoModelForCausalLM.from_pretrained(backbone_root, local_files_only=True, torch_dtype=torch_dtype)
    if config['version'] == 4:
        from peft import PeftModel
        backbone = PeftModel.from_pretrained(backbone, root/'backbone', local_files_only=True, is_trainable=False)
    memory_config = (config["memory_compiler"] if config["version"] >= 2 else
                     {"kind": "legacy", "width": 512, "depth": 2, "heads": 8})
    cls = ToolBenchAgent
    if config["version"] in {3, 4}:
        from .toolbench_curriculum import CurriculumAgent, INTERFACE
        from .toolbench_topk import TopKCurriculumAgent, INTERFACE as TOPK_INTERFACE
        from .toolbench_hybrid import HybridDocumentAgent, INTERFACE as HYBRID_INTERFACE
        from .toolbench_task_state import TaskStateAgent, INTERFACE as STATE_INTERFACE
        if config["interface"] == STATE_INTERFACE:
            cls = TaskStateAgent
        elif config["interface"] == HYBRID_INTERFACE:
            cls = HybridDocumentAgent
        elif config["interface"] == TOPK_INTERFACE:
            cls = TopKCurriculumAgent
        elif config["interface"] == INTERFACE:
            cls = CurriculumAgent
        else:
            raise ValueError("Unknown curriculum checkpoint interface")
    agent = cls(backbone, tokenizer, rank=config["rank"], slots=config["slots"],
                           condition=config["condition"], limits=Limits(**config["limits"]),
                           **{"memory_" + key: value for key, value in memory_config.items()})
    agent.history_format = config.get("history_format", "legacy")
    if agent.history_format not in {"legacy", "observation_envelope_v1"}:
        raise ValueError("Unknown checkpoint history serialization")
    if config['version'] == 4:
        agent._native_base_reference = config['base_reference']
    runtime = torch.load(root / "runtime.pt", map_location="cpu", weights_only=True)
    expected = {key for key in agent.state_dict() if not key.startswith("backbone.")}
    if set(runtime["compilers"]) != expected:
        raise ValueError("Incomplete or incompatible compiler checkpoint")
    agent.load_state_dict(runtime["compilers"], strict=False)
    for name, value in runtime["backbone_buffers"].items():
        parent, _, leaf = name.rpartition(".")
        module = backbone.get_submodule(parent) if parent else backbone
        if leaf not in module._buffers:
            raise ValueError(f"Backbone buffer not found: {name}")
        module._buffers[leaf] = value
    agent.to(device).eval().requires_grad_(False)
    return agent
