#!/usr/bin/env bash
set -Eeuo pipefail
root=/data/L202500289/hujinchao/projects/Tooltoken/formal_20260903
python="$root/runtime/formal_env/bin/python"
source_root="$root/upstream/native-late-bound-tool-registration"
prepared="$root/data/stq_memory_native_20260905"
out="$root/artifacts/stq_memory_native_20260905"
logs="$root/logs/stq_memory_native_20260905"
model=/data/L202500289/hujinchao/projects/models/Qwen3-8B
mkdir -p "$out" "$logs"
exec 9>"$out/RUN.lock"
flock -n 9 || { echo 'An STQ memory queue already owns this run'; exit 2; }
trap 'rc=$?; printf "%s exit=%s line=%s\n" "$(date -Is)" "$rc" "$LINENO" > "$out/FAILED"; exit "$rc"' ERR
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -s "$prepared/readback.jsonl"
test -s "$prepared/manifest.json"
printf '%s\n' 'No optimizer checkpoint is saved by these native trainers; retain LoRA and both compilers.' > "$out/OPTIMIZER_POLICY"
sha256sum "$source_root/src/latent_register/train_meta_registration.py" \
  "$source_root/src/latent_register/train_meta_readback.py" \
  "$source_root/src/latent_register/model.py" \
  "$prepared/manifest.json" > "$out/SOURCE_AND_DATA_SHA256"

registration=(
  --prepared-dir "$prepared" --model-path "$model" --seed 17
  --registry-size 128 --batch-size 1 --gradient-accumulation-steps 2
  --epochs 5 --max-query-length 256 --max-document-length 128
  --document-instruction 'Represent this tool for registration.'
  --compiler-rank 128 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
  --learning-rate 2e-4 --weight-decay 0.01 --warmup-ratio 0.03
  --max-grad-norm 1 --gradient-checkpointing --log-every 10
)
readback=(
  --prepared-dir "$prepared" --model-path "$model" --seed 17
  --memory-slots 8 --compiler-rank 128 --batch-size 2
  --gradient-accumulation-steps 1 --epochs 5 --learning-rate 1e-4
  --weight-decay 0.01 --warmup-ratio 0.03 --max-grad-norm 1
  --max-document-length 128 --max-query-length 256 --max-prompt-length 256
  --max-target-length 128 --max-new-tokens 128
  --document-instruction 'Represent this tool for registration.'
  --selection-document-instruction 'Represent this tool for registration.'
  --train-lora --gradient-checkpointing --selection-registry-size 128
  --selection-softmax full_vocabulary --selection-weight 1.0
  --schema-weight 0.1 --wrong-memory-weight 0.1 --wrong-memory-margin 0.1
  --log-every 10
)
train_registration() {
  "$python" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m latent_register.train_meta_registration "${registration[@]}" "$@"
}
train_readback() {
  "$python" -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m latent_register.train_meta_readback "${readback[@]}" "$@"
}

if [ ! -f "$out/SMOKE_COMPLETE" ]; then
  printf '%s smoke_registration\n' "$(date -Is)" > "$out/STATUS"
  test ! -e "$out/smoke_registration/results.json"
  train_registration --output-dir "$out/smoke_registration" --max-steps 2 --eval-samples 4 \
    > "$logs/smoke_registration.log" 2>&1
  printf '%s smoke_readback\n' "$(date -Is)" > "$out/STATUS"
  train_readback --retrieval-adapter-path "$out/smoke_registration/lora" \
    --retrieval-compiler-path "$out/smoke_registration/compiler.pt" \
    --output-dir "$out/smoke_readback" --max-steps 2 --eval-samples 4 \
    --selection-eval-samples 4 --generation-samples 4 > "$logs/smoke_readback.log" 2>&1
  "$python" "$root/runner/eval_stq_memory_native.py" --prepared-dir "$prepared" \
    --model-path "$model" --checkpoint "$out/smoke_readback" \
    --output-dir "$out/smoke_eval" --limit 2 > "$logs/smoke_eval.log" 2>&1
  date -Is > "$out/SMOKE_COMPLETE"
fi
if [ ! -f "$out/STAGE1_COMPLETE" ]; then
  printf '%s stage1_registration\n' "$(date -Is)" > "$out/STATUS"
  test ! -e "$out/stage1/results.json"
  train_registration --output-dir "$out/stage1" --eval-samples 32 > "$logs/stage1.log" 2>&1
  "$python" "$root/runner/validate_stq_memory_run.py" --checkpoint "$out/stage1" --phase stage1
  date -Is > "$out/STAGE1_COMPLETE"
fi
if [ ! -f "$out/STAGE2_COMPLETE" ]; then
  printf '%s stage2_memory_readback\n' "$(date -Is)" > "$out/STATUS"
  test ! -e "$out/stage2/results.json"
  train_readback --retrieval-adapter-path "$out/stage1/lora" \
    --retrieval-compiler-path "$out/stage1/compiler.pt" \
    --output-dir "$out/stage2" --eval-samples 32 --selection-eval-samples 32 \
    --generation-samples 8 > "$logs/stage2.log" 2>&1
  "$python" "$root/runner/validate_stq_memory_run.py" --checkpoint "$out/stage2" --phase stage2
  date -Is > "$out/STAGE2_COMPLETE"
fi
if [ ! -f "$out/EVAL_COMPLETE" ]; then
  printf '%s full_837_tool_eval\n' "$(date -Is)" > "$out/STATUS"
  "$python" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$root/runner/eval_stq_memory_native.py" --prepared-dir "$prepared" \
    --model-path "$model" --checkpoint "$out/stage2" --output-dir "$out/eval" \
    > "$logs/eval.log" 2>&1
  "$python" "$root/runner/validate_stq_memory_run.py" --checkpoint "$out/stage2" \
    --phase stage2 --evaluation "$out/eval"
  date -Is > "$out/EVAL_COMPLETE"
fi
sha256sum "$out/stage1/results.json" "$out/stage2/results.json" \
  "$out/stage2/lora/adapter_model.safetensors" "$out/stage2/compiler.pt" \
  "$out/stage2/memory_compiler.pt" "$out/eval/summary.json" > "$out/RESULT_SHA256"
date -Is > "$out/COMPLETE"
printf '%s complete\n' "$(date -Is)" > "$out/STATUS"
