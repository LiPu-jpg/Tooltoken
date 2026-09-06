# STQ memory-native formal run (best STQ result)

Formal drivers for the best STQ strict-unseen result, produced by the two-stage
native memory pipeline on Qwen3-8B (stage 1 `latent_register.train_meta_registration`
→ stage 2 `latent_register.train_meta_readback`), evaluated over the full 837-tool
registry on 1,066 queries.

## Result (`artifacts/stq_memory_native_20260905/eval/summary.json`)

Retrieval:

| H@1 | H@3 | H@5 | MRR | NDCG@1 | NDCG@3 | NDCG@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 93.43 | 97.75 | 98.59 | 95.73 | 93.43 | 96.03 | 96.38 |

Call (tool/param normalized exact):

| tool_exact | param_exact | joint_exact | parse_success |
|---:|---:|---:|---:|
| 93.43 | 79.17 | 74.77 | 100.0 |

Reference points on the same split (see paper Table 1): best baseline
BGE-large-en-v1.5 reaches 82.18 H@1; oracle upper bound joint_exact 76.9.

## Files

- `run_stq_memory_native_20260905.sh` — master launcher: smoke → stage 1
  registration → stage 2 memory readback → full 837-tool eval. flock-guarded,
  per-stage resumable via `*_COMPLETE` markers, records `SOURCE_AND_DATA_SHA256`
  and `RESULT_SHA256`.
- `prepare_stq_memory_run_20260905.py` — builds the prepared dataset
  (`manifest.json` + registration/readback queries).
- `eval_stq_memory_native.py` — full-registry evaluation (legacy STQ tool/param
  normalized exact scoring).
- `validate_stq_memory_run.py` — per-stage checkpoint validation (finite LoRA /
  compiler tensors, result sanity).

## Reproduction

Requires `src/latent_register` from this repository on `PYTHONPATH`, Qwen3-8B,
`peft`/`safetensors`, and 4 GPUs (torchrun `--nproc_per_node=4`).

Stage 1 (registration): registry 128, LoRA r=16 α=32 dropout 0.05, compiler rank
128, lr 2e-4, 5 epochs, seed 17, max query/doc length 256/128.

Stage 2 (readback): memory slots 8, lr 1e-4, selection softmax `full_vocabulary`,
selection weight 1.0, schema weight 0.1, wrong-memory weight/margin 0.1, 5 epochs,
seed 17. No optimizer state is checkpointed (native trainers); LoRA + both
compilers are the artifacts.

## Provenance (sha256, from the formal eval summary)

- LoRA: `e190ef805bbef85ef96b54aaa4798aa38340639892eb4b7732837f8cc77e1d92`
- output compiler: `04c2b8a1f1ee3d006106b64c384e27b5ce1f239134de7cfd4cd60bd46bea5ae0`
- memory compiler: `dfb94f2ddf71a71d46b219ee337e1be3b478458d80b564d937688a34a683748c`
- data: `481865c7924bbe15716e9d56ca2bc8df4de2a8eaa86aadc7c95c046e87a01f9a`
