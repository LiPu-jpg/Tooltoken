#!/usr/bin/env bash
set -euo pipefail

root=/data/L202500289/hujinchao/projects/Tooltoken/formal_20260903
python="$root/runtime/formal_env/bin/python"
source_root="$root/upstream/native-late-bound-tool-registration"
model=/data/L202500289/hujinchao/projects/models/Qwen3-8B
prepared="$root/data/ours_meta_toolbench10k_seen"
protocol="$root/data/toolbench_10k_protocol"
output="$root/artifacts/table3_toolbench10k/ours_seen"
training="$output/meta_training"
evaluation="$output/eval"
log_dir="$root/logs/table3_toolbench10k/ours_seen"
smoke=/tmp/hujinchao_tooltoken_stage/ours_meta_seen_smoke

mkdir -p "$output" "$log_dir"
rm -f "$output/FAILED" "$output/COMPLETE"
trap 'code=$?; printf "%s\n" "$code" > "$output/FAILED"; exit "$code"' ERR

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test -s "$prepared/manifest.json"
test -s "$prepared/tools.jsonl"
test -s "$prepared/retrieval.jsonl"
test -s "$prepared/split_manifest.json"

common=(
  --prepared-dir "$prepared"
  --model-path "$model"
  --seed 42
  --registry-size 128
  --batch-size 1
  --gradient-accumulation-steps 1
  --epochs 5
  --max-query-length 128
  --max-document-length 64
  --document-instruction "Represent this tool for registration."
  --compiler-rank 128
  --lora-rank 16
  --lora-alpha 32
  --lora-dropout 0.05
  --learning-rate 2e-4
  --weight-decay 0.01
  --warmup-ratio 0.03
  --max-grad-norm 1.0
  --log-every 10
)

rm -rf -- "$smoke"
"$python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m latent_register.train_meta_registration \
  "${common[@]}" --output-dir "$smoke" --max-steps 2 --eval-samples 8 \
  > "$log_dir/smoke.log" 2>&1
test -s "$smoke/lora/adapter_model.safetensors"
test -s "$smoke/compiler.pt"
test -s "$smoke/results.json"
"$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["completed_steps"]==2; assert d["recent_train_loss"] > 0' "$smoke/results.json"
rm -rf -- "$smoke"

rm -rf -- "$training" "$evaluation"
"$python" -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m latent_register.train_meta_registration \
  "${common[@]}" --output-dir "$training" --eval-samples 64 \
  > "$log_dir/train.log" 2>&1

test -s "$training/lora/adapter_model.safetensors"
test -s "$training/compiler.pt"
test -s "$training/results.json"
"$python" -c 'import json,math,sys; d=json.load(open(sys.argv[1])); assert d["completed_steps"]>0; assert math.isfinite(d["recent_train_loss"]); assert d["recent_train_loss"]>0' "$training/results.json"

CUDA_VISIBLE_DEVICES=0 "$python" "$root/runner/eval_ours_meta_toolbench10k.py" \
  --model-path "$model" --training-dir "$training" --prepared-dir "$protocol" \
  --output-dir "$evaluation" --device cuda:0 \
  --max-query-length 128 --max-document-length 64 \
  > "$log_dir/eval.log" 2>&1

test "$("$python" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["metadata"]["documents"],d["metadata"]["queries"])' "$evaluation/summary.json")" = "10000 1100"
sha256sum "$prepared/manifest.json" "$protocol/manifest.json" \
  "$training/results.json" "$training/compiler.pt" \
  "$training/lora/adapter_model.safetensors" "$evaluation/summary.json" \
  > "$output/SHA256SUMS"
printf '%s\n' complete > "$output/COMPLETE"
rm -f "$output/FAILED"
printf '%s\n' "$(date -Is) complete; LoRA/compiler retained; no optimizer checkpoint saved" \
  > "$output/OPTIMIZER_POLICY"
