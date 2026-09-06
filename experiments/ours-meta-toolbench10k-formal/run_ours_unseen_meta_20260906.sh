#!/usr/bin/env bash
set -euo pipefail
r=/data/L202500289/hujinchao/projects/Tooltoken/formal_20260903
p="$r/runtime/formal_env/bin/python"
prepared="$r/data/ours_meta_toolbench10k_unseen_20260906"
out="$r/artifacts/table3_toolbench10k/ours_unseen_meta_20260906"
log="$r/logs/table3_toolbench10k/ours_unseen_meta_20260906"
mkdir -p "$out" "$log"
exec 9>"$out/RUN.lock"
flock -n 9 || exit 3
test ! -f "$out/COMPLETE"
test ! -d "$out/meta_training" # Never silently overwrite/restart a partial run.
trap 'c=$?; echo "$c" > "$out/FAILED"; exit "$c"' ERR
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$r/upstream/native-late-bound-tool-registration/src"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
common=(--prepared-dir "$prepared" --model-path /data/L202500289/hujinchao/projects/models/Qwen3-8B
 --seed 42 --registry-size 128 --batch-size 1 --gradient-accumulation-steps 1 --epochs 5
 --max-query-length 128 --max-document-length 64 --document-instruction "Represent this tool for registration."
 --compiler-rank 128 --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05
 --learning-rate 2e-4 --weight-decay 0.01 --warmup-ratio 0.03 --max-grad-norm 1.0 --log-every 10 --eval-samples 0)
echo smoke > "$out/STATUS"
"$p" -m torch.distributed.run --standalone --nproc_per_node=4 -m latent_register.train_meta_registration \
 "${common[@]}" --output-dir "$out/smoke" --max-steps 2 > "$log/smoke.log" 2>&1
"$p" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["completed_steps"]==2' "$out/smoke/results.json"
echo training > "$out/STATUS"
"$p" -m torch.distributed.run --standalone --nproc_per_node=4 -m latent_register.train_meta_registration \
 "${common[@]}" --output-dir "$out/meta_training" > "$log/train.log" 2>&1
"$p" -c 'import json,math,sys; d=json.load(open(sys.argv[1])); assert d["completed_steps"]>0 and math.isfinite(d["recent_train_loss"])' "$out/meta_training/results.json"
echo evaluating > "$out/STATUS"
CUDA_VISIBLE_DEVICES=0 "$p" "$r/runner/eval_ours_meta_toolbench10k.py" \
 --model-path /data/L202500289/hujinchao/projects/models/Qwen3-8B --training-dir "$out/meta_training" \
 --prepared-dir "$r/data/toolbench_10k_protocol" --output-dir "$out/eval" --device cuda:0 \
 --max-query-length 128 --max-document-length 64 > "$log/eval.log" 2>&1
"$p" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["metadata"]["documents"]==10000 and d["metadata"]["queries"]==1100' "$out/eval/summary.json"
sha256sum "$prepared/manifest.json" "$out/meta_training/results.json" "$out/meta_training/compiler.pt" \
 "$out/meta_training/lora/adapter_model.safetensors" "$out/eval/summary.json" > "$out/SHA256SUMS"
date -Is > "$out/COMPLETE"
echo complete > "$out/STATUS"
