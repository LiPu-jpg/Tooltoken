# STQ Native staged protocol (v6 reload repair)

The scientific variable is optimizer schedule only. Both conditions use the
same fixed 999-tool/10,483-row seen split, Qwen3-8B revision, compiler,
reserved physical IDs, tokenizer, sequence renderer, batch and evaluator.

Stage L trains shared compiler parameters and attention LoRA on the seen split
with ordinary causal sequence cross-entropy. Stage F starts from the exact
Stage-L checkpoint, merges/removes LoRA, then trains the complete backbone and
shared compiler with the same causal sequence loss. A held-out API is never a
training label or optimizer target during registration: one document forward,
zero deployment optimizer steps, zero static-row mutation.

Every phase must produce checkpoint, parameter counts, step times, peak memory,
loss/token records, phase audit, reload audit, source/data/model hashes, and a
machine-readable terminal event. Formal scoring is blind to qrels until both
phase checkpoints and final reload pass; report Hit@1/3/5, MRR, NDCG@1/3/5,
coverage, invalid predictions, identity collisions, and registration steps.

The v4 failure mode was an all-zero padding rank returning early from the model
forward, which made ZeRO-3 ranks issue different all-gather sequences on the
final incomplete global batch. v5 fixed this and completed Stage-L. v5's
reload gate then failed because its one-GPU job did not set CUDA_HOME before
importing DeepSpeed. v6 recreates the job-local CUDA shim in both reload gates;
the already audited v5 Stage-L checkpoint is the only warmup input.
The deployment uses four L20 GPUs only when the scheduler actually grants four;
the smoke records `world_size` and refuses a distributed run with a mismatched
rank count. A failed phase creates a new deployment version rather than
mutating or resubmitting the failed checkpoint.
