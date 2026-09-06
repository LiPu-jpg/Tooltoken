# Ours meta-registration on ToolBench-10K (best ToolBench result)

Formal drivers for our best ToolBench-10K tool-selection results, produced by
`latent_register.train_meta_registration` (this repository's `src/latent_register`)
on Qwen3-8B, evaluated on the 10K-candidate registry / 1,100-query protocol.

## Results

Ours-seen (`artifacts/table3_toolbench10k/ours_seen/eval/summary.json`):

| H@1 | H@3 | H@5 | MRR | NDCG@3 | NDCG@5 | R@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 75.55 | 88.00 | 91.45 | 82.62 | 69.36 | 73.17 | 76.16 |

Ours-unseen, meta-registration — official formal run
(`artifacts/table3_toolbench10k/ours_unseen_meta_20260906/eval/summary.json`;
completion/audit note in that directory):

| H@1 | H@3 | H@5 | MRR | NDCG@3 | NDCG@5 | R@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 75.18 | 86.18 | 89.18 | 81.62 | 66.44 | 70.26 | 72.32 |

Unseen run audit: 5 epochs / 12,050 steps; all 288 LoRA tensors and compiler
parameters independently checked finite; training tools (9,065) have zero
overlap with the 10,000 test candidates; full coverage of the 1,100 protocol
query IDs with no duplicates or missing; registration performs no optimizer
steps. Note: that directory's `metadata.method` string is hardcoded to
`Ours-meta-registration-seen` by the shared eval script; it is an unseen run
(verified via launch script, independent prepared data, training records, and
tool-set disjointness).

Exposure note for comparisons: `ToolRetriever` (82.27 H@1) is the official
full-seen checkpoint and `ToolWeaver` (88.18 H@1) is our reproduction under its
own exposure; both see a different amount/kind of supervision than the strict
1K-registry / 259-supervised setting of the rows above.

## Files

- `build_ours_meta_toolbench10k.py` — builds the prepared meta-registration
  datasets (seen / unseen variants).
- `run_ours_meta_toolbench10k_seen.sh` — smoke → train → eval for the seen run.
- `run_ours_unseen_meta_20260906.sh` — smoke → train → eval for the unseen run
  (flock-guarded, refuses to overwrite a partial run, records per-stage status).
- `eval_ours_meta_toolbench10k.py` — evaluation over the prepared
  `toolbench_10k_protocol` registry.

## Reproduction

Requires `src/latent_register` from this repository on `PYTHONPATH`, Qwen3-8B,
`peft`/`safetensors`, and 4 GPUs (torchrun `--nproc_per_node=4`); evaluation
runs on a single GPU.

Common hyperparameters: registry 128, LoRA r=16 α=32 dropout 0.05, compiler
rank 128, lr 2e-4, 5 epochs, seed 42, max query/document length 128/64,
document instruction "Represent this tool for registration." Data preparation
and protocol: `data/toolbench_10k_protocol` (10,000 documents, 1,100 queries,
subsets G1/G2/G3).
