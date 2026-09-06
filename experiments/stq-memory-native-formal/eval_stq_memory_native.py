"""STQ full-bank retrieval + native same-stream memory call evaluation.

The model's core selection/decoding is imported unchanged. Gold labels are used
only after predictions. Tool IDs and entity arguments are scored using the old
STQ agent table's whitespace/case normalization, not an LLM judge.
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("```json", "").replace("```", "")
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def selection_metrics(ranking, target):
    rank = ranking.index(target) + 1
    m = {"mrr": 1.0 / rank}
    for k in [1, 3, 5]:
        m[f"hit_at_{k}"] = float(rank <= k)
        m[f"ndcg_at_{k}"] = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    return m


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    local = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    from latent_register.train_meta_registration import (
        _disable_incompatible_optional_torchao,
        _provide_optional_tensor_parallel_compat,
        render_query,
    )
    _disable_incompatible_optional_torchao()
    _provide_optional_tensor_parallel_compat()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from latent_register.train import seed_everything
    from latent_register.episodic_data import load_prepared_tools, load_token_pools
    from latent_register.physical_tokens import SplitReservedTokenPool
    from latent_register.model import PhysicalOutputGenerator, TokenResamplerMemory
    from latent_register.train_meta_readback import MetaReadbackModel, generate_same_stream_one, tokenize_documents

    config = json.loads((args.checkpoint / "results.json").read_text())["config"]
    seed_everything(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    pools = load_token_pools(args.prepared_dir / "split_manifest.json")
    pool = SplitReservedTokenPool.create(tokenizer, base, pools)
    backbone = PeftModel.from_pretrained(base, args.checkpoint / "lora", is_trainable=False)
    backbone.to(device).eval()
    compiler = PhysicalOutputGenerator(4096, config["compiler_rank"], 1.0).to(device)
    compiler.load_state_dict(torch.load(args.checkpoint / "compiler.pt", map_location=device, weights_only=True))
    memory = TokenResamplerMemory(4096, config["compiler_rank"], config["memory_slots"], 1.0).to(device)
    memory.load_state_dict(torch.load(args.checkpoint / "memory_compiler.pt", map_location=device, weights_only=True))
    model = MetaReadbackModel(backbone, memory, compiler).to(device).eval()
    tools_by_hash = load_prepared_tools(args.prepared_dir / "tools.jsonl")
    test_tools = sorted((t for t in tools_by_hash.values() if t.split == "test"), key=lambda t: t.endpoint_name)
    assert len(test_tools) == 837
    assert len({t.endpoint_name for t in test_tools}) == 837
    examples = [r for r in read_rows(args.prepared_dir / "readback.jsonl") if r["split"] == "test"]
    assert len(examples) == 1066
    if args.limit:
        examples = examples[:args.limit]
    identities = [t.identity_hash for t in test_tools]
    positions = {v: i for i, v in enumerate(identities)}
    physical = torch.tensor([pool.physical_id(s) for s in list(pools["test"])[:837]], device=device)
    reserved = torch.tensor(pool.token_ids, dtype=torch.long, device=device)
    bank_o, bank_m = [], []
    for start in range(0, len(test_tools), 16):
        tokens = tokenize_documents(tokenizer, test_tools[start:start+16],
            max_length=config["max_document_length"], device=device,
            document_instruction=config["document_instruction"])
        o, m = model.register_bundle(tokens)
        bank_o.append(o)
        bank_m.append(m)
    bank_o = torch.cat(bank_o)
    bank_m = torch.cat(bank_m)
    assert tuple(bank_m.shape) == (837, 8, 4096)
    assert torch.isfinite(bank_o).all() and torch.isfinite(bank_m).all()
    signature = {
        "checkpoint": str(args.checkpoint),
        "lora_sha256": sha(args.checkpoint / "lora/adapter_model.safetensors"),
        "output_compiler_sha256": sha(args.checkpoint / "compiler.pt"),
        "memory_compiler_sha256": sha(args.checkpoint / "memory_compiler.pt"),
        "data_sha256": sha(args.prepared_dir / "manifest.json"),
        "world_size": world, "limit": args.limit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sf = args.output_dir / "run_signature.json"
    if rank == 0:
        if sf.exists():
            assert json.loads(sf.read_text()) == signature, "Cannot mix prediction versions"
        else:
            sf.write_text(json.dumps(signature, indent=2) + "\n")
    if world > 1:
        dist.barrier()
    pred_path = args.output_dir / f"predictions.rank{rank}.jsonl"
    old = read_rows(pred_path)
    done = {r["query_id"] for r in old}
    assert len(done) == len(old)
    start_time = time.monotonic()
    with pred_path.open("a", encoding="utf-8", buffering=1) as stream:
        for n, row in enumerate(examples[rank::world]):
            if row["query_id"] in done:
                continue
            prompt = render_query(tokenizer, row["query"])
            token_ids = tokenizer(prompt, add_special_tokens=False).input_ids[-config["max_query_length"]:]
            ids = torch.tensor([token_ids], device=device)
            decoder = backbone.get_base_model().model
            q = decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False, return_dict=True).last_hidden_state[:, -1].float()
            scores = (q @ bank_o.float().T)[0]
            ranking = torch.argsort(scores, descending=True, stable=True).tolist()
            generated = generate_same_stream_one(
                model, tokenizer, row["query"], bank_o, bank_m, physical, reserved,
                max_prompt_length=config["max_prompt_length"],
                max_new_tokens=config["max_new_tokens"], device=device,
            )
            # Gold labels are first used below, never in the inference call.
            selected = generated.selected_registry_index
            predicted_tool = test_tools[selected].endpoint_name if selected is not None else ""
            gold_tool = tools_by_hash[row["tool_identity_hash"]].endpoint_name
            parsed = extract_json(generated.text)
            predicted_param = parsed.get("entity", "") if parsed is not None else ""
            gold_param = row["arguments"]["entity"]
            tool_ok = normalize(predicted_tool) == normalize(gold_tool)
            param_ok = normalize(predicted_param) == normalize(gold_param)
            metrics = selection_metrics(ranking, positions[row["tool_identity_hash"]])
            value = {
                "query_id": row["query_id"], "query": row["query"],
                "gold_tool": gold_tool, "gold_param": gold_param,
                "predicted_tool": predicted_tool, "predicted_param": predicted_param,
                "selected_physical_id": generated.selected_token_id,
                "selected_registry_index": selected,
                "memory_slots_consumed": 8 if selected is not None else 0,
                "raw_response": generated.text, "retrieval_metrics": metrics,
                "top5": [{"endpoint": test_tools[i].endpoint_name, "score": float(scores[i])} for i in ranking[:5]],
                "tool_exact": tool_ok, "param_exact": param_ok,
                "joint_exact": tool_ok and param_ok, "parse_success": parsed is not None,
                "ordinary_token_win": selected is None,
            }
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            if (n+1) % 20 == 0 or n == 0:
                print(f"rank={rank} processed={n+1}/{len(examples[rank::world])} elapsed={time.monotonic()-start_time:.1f}s", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        results = [r for i in range(world) for r in read_rows(args.output_dir / f"predictions.rank{i}.jsonl")]
        assert len(results) == len(examples)
        assert {r["query_id"] for r in results} == {r["query_id"] for r in examples}
        summary = {
            "metadata": {
                **signature, "method": "Ours STQ native two-stage memory",
                "queries": len(results), "documents": 837,
                "registration_optimizer_steps": 0, "memory_slots": 8,
                "native_runtime": "generate_same_stream_one (unchanged)",
                "call_protocol": "full-vocabulary physical ID -> its memory -> same KV stream arguments",
                "scoring": "legacy STQ tool/param normalized exact; entity field maps to STQ param",
                "stage1_only": False, "tool_execution": False,
                "smoke_only": bool(args.limit),
            },
            "retrieval": {k: sum(r["retrieval_metrics"][k] for r in results)/len(results)
                          for k in results[0]["retrieval_metrics"]},
            "call": {k: sum(r[k] for r in results)/len(results)
                     for k in ["tool_exact", "param_exact", "joint_exact", "parse_success", "ordinary_token_win"]},
            "physical_token_audit": pool.audit(backbone, set()),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output_dir / "COMPLETE").write_text("complete\n")
        print(json.dumps(summary, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
