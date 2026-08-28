# ToolGen and Late-Binding Benchmark Protocol

Status: corrected protocol draft 2
Date: 2026-08-09

## Objective

Measure closed-set tool use and post-training tool registration without mixing
retrieval, argument generation, and end-to-end task completion into one number.
The primary controlled comparison uses Qwen3-8B for every learned method. A
separate Llama-3-8B track reproduces the published ToolGen recipe and is not
used as the main comparison against the proposed method.

## Systems

| ID | System | Deployment information available after selection |
| --- | --- | --- |
| `llama_toolgen_official` | Official ToolGen commit `6839374a255810efe69deea4056eec5c55e25802` on Llama-3-8B | Full fetched tool documentation |
| `qwen_full_document` | Qwen3-8B full-document caller | Full tool documentation in context |
| `qwen_toolgen_fixed` | ToolGen-style fixed atomic tokens on Qwen3-8B | Full fetched tool documentation |
| `qwen_toolgen_incremental` | Fixed-token ToolGen continued on newly added tools | Full fetched tool documentation |
| `qwen_late_bound` | Shared compiler plus strictly held-out physical addresses | Selected full document in the primary comparison; registered memory in the auxiliary condition |
| `qwen_late_bound_oracle` | Same late-bound checkpoint with the correct tool supplied | Full tool documentation without a selection decision |

The following controls use the same Qwen3-8B checkpoint and test manifest:
query-only, static blank memory, random row/memory, wrong-tool memory,
nearest-trained-token, and one shared vector reused on both input and output
sides. Formal common-document controls also use the exact same audited Agent;
only the registration treatment may differ. Registered-memory control runs are
auxiliary diagnostics and cannot be substituted into the common-document
control cells.

`query-only` masks every reserved address rather than treating a zero vector as
an active tool. `nearest-trained-token` compiles the complete train split once,
finds the maximum-cosine train output row for each test registration, and then
uses that train tool's complete output/memory bundle. It must save the exact
test-to-train mapping, reference-bank size/storage, and similarity distribution;
a sampled reference bank is a smoke test only and cannot enter the main table.

## Information Conditions

Every system is evaluated under all applicable conditions. Results from
different conditions must not be placed in the same comparison column.

1. `common_document`: both systems receive the selected tool's full
   documentation after selection and hand it to the exact same independently
   trained full-document Agent checkpoint. The checkpoint path and audit hash
   are embedded in every argument prediction, and paired comparison rejects a
   mismatch. This is the primary fair end-to-end comparison: token registration
   is tested without also requiring lossy schema compression or confounding
   downstream Agent weights.
2. `registered_memory`: late binding generates from registered memory without a
   document lookup. This is an additional capability and mechanism diagnostic,
   not a requirement for the core post-training registration claim.
3. `token_memory_only`: no documentation lookup. ToolGen receives only its
   trained token; late binding receives its registered memory. This is a
   mechanism diagnostic, not ToolGen's intended setting.
4. `full_document_oracle`: the correct document is supplied without a
   retrieval decision. This estimates the same-model execution ceiling.

## Data and Splits

### ToolBench and ToolGen

- Use the three public ToolGen files without silently merging ToolWeaver as a
  second corpus: 49,936 memorization records, 489,702 retrieval records, and
  183,336 trajectories.
- Preserve the official ToolBench I1/G1, I2/G2, and I3/G3 evaluation groups for
  the paper-compatible track.
- Use the existing source-aware, tool-group-disjoint manifest for the
  controlled Qwen3-8B track.
- Record SHA-256 hashes for every source, generated subset, tokenizer, model
  revision, and split manifest.
- BFCL definitions used for reporting remain absent from all training stages.

### Five-Thousand-Tool Pilot

The pilot contains exactly 5,000 unique ToolGen atomic tool identities chosen
by seeded hash order (`seed=17`), not by source file order. All memorization,
retrieval, and trajectory records are filtered by those identities. Records
that reference both included and excluded tools are excluded rather than
partially relabeled. The pilot is a pipeline and throughput gate, not a result
for the paper.

### Full Run

After the pilot passes, use all 46,985 virtual API tokens for the official
track and the complete leakage-safe manifest for the controlled track. Do not
change data, model, or evaluator code between the successful pilot and full
submission except for declared scale parameters and output paths.

## Benchmarks

### 1. ToolBench Retrieval

Report I1, I2, and I3 separately under in-domain and multi-domain candidate
spaces:

- NDCG@1, NDCG@3, and NDCG@5;
- hit@1, recall@5, and MRR;
- valid registered-token rate and invalid-token rate;
- allocated-token count, reachable candidate count, mapping coverage, and
  normalized/truncated identifier collision count;
- target rank against the complete ordinary vocabulary where applicable.

The official ToolGen evaluator constructs the NDCG relevance vector only from
relevant tools that appear in its predicted beams. On queries with multiple
relevant tools this changes the ideal DCG and can inflate the score. Preserve
that implementation as `ndcg_paper_compatible` for reproduction. Also report
`ndcg_corrected`, whose relevance vector marks every labeled relevant tool.

### 2. StableToolBench End-to-End

Report SoPR and SoWR for I1, I2, I3, I1-Tool, I1-Category, and I2-Category.
Pin the evaluator model, prompt, API simulator, reference answer set, number of
judge repetitions, and code revision. ToolGen's published GPT-3.5 reference
and ToolWeaver's GPT-4o-mini reference are separate historical tracks and must
not be directly compared. The controlled main table uses one shared evaluator
configuration for all systems.

### 3. BFCL Deterministic Calling

Report each available category and the micro/macro average:

- tool-name exact match and tool-call-set exact match;
- canonical JSON/AST argument exact match;
- argument key precision, recall, and F1;
- argument value exact match and scalar value F1;
- JSON parse rate;
- type, required-field, enum, nested-path, and schema validation rates;
- ordered multi-call exact match;
- executable success where the benchmark supplies an executor.

Do not train on the reported BFCL tool definitions or answers.

### 4. Late-Binding Mechanism Bench

Report the four tool/address quadrants separately:

| Tool status | Address/path status | Purpose |
| --- | --- | --- |
| Seen | Seen | Closed-set retention |
| Seen | Unseen | Address rebinding |
| Unseen | Seen | Tool semantic generalization |
| Unseen | Unseen | Core post-training registration |

Also measure registries of 10, 100, 1K, 10K, and 47K tools; address
permutation; sequential registry append; old-tool regression; same-name and
same-schema hard negatives; unseen field names; nested objects; arrays; enums;
and required/optional dependencies. Two-token and three-token paths remain
blocked until the atomic same-stream gate passes.

Sequential append must preserve the exact identity -> logical slot -> physical
ID prefix of the pre-append registry. Rebuilding both registries from the same
seed is not sufficient because it may silently rebind old tools. The append
artifact must hash both mappings, assert every old binding is unchanged, and
score the same old-tool examples before and after adding distractors/new tools.

The current prepared manifest has 6,144 training addresses and 1,024 addresses
in each evaluation split, so the active matched array covers 10, 100, and 1K
without address reuse. The 10K and 47K rows use a separate evaluation-only
manifest with 47,000 validation and 47,000 test addresses. That manifest must
preserve the original training range exactly, keep all three ranges disjoint,
and create one distinct physical token ID per active registry entry.

The 10K/47K rows are reported for unseen-address conditions. A 10K or 47K
"seen-address" row would be invalid because only 6,144 physical addresses were
available during meta-training. For unseen tools, the large candidate universe
mixes train, validation, and test tools because the strict test split contains
only 6,335 tools; every evaluation target nevertheless remains a group-disjoint
test tool. Large runs use one deterministic shared registry mapping across
queries and store its exact identity -> logical slot -> physical token binding
once in `registry_binding.json`; each prediction references that mapping by
SHA-256 and records its target physical token IDs.

## Metrics and Statistical Reporting

Selection, argument generation, and task completion are separate metric
families. Teacher-forced NLL is diagnostic only.

- Canonicalize JSON structurally; never score JSON with raw string equality.
- Report sample counts and exclusions for every metric.
- Use training seeds `17`, `29`, and `43` for formal learned-system
  comparisons. Keep the source-aware data split, held-out address ranges, test
  examples, and evaluation registry seed fixed at `17`; vary model
  initialization, training order, and episodic binding through `TRAIN_SEED`.
- A late-bound seed is independent only if both the retrieval LoRA/output
  compiler stage and the full-vocabulary joint LoRA/output/memory stage start
  from Qwen3-8B under that seed. Changing only the continuation seed of a
  shared seed-17 checkpoint does not count as an independent run.
- Report paired bootstrap 95% confidence intervals on shared test examples.
- Save per-example predictions, canonical references, error categories, and
  registry membership so aggregate values can be independently recomputed.
- Predeclare missing-output and invalid-JSON handling as failures.
- Embed the SHA-256 of the evaluator and structural-metric modules in every
  evaluation result, and embed comparison/metric module hashes in every paired
  comparison. Three-seed aggregation must reject code-hash differences across
  seeds rather than averaging results produced by different scoring code.

## Registration and Runtime Audit

For every late-bound run report:

- document-encoder/compiler forward count;
- registration wall time and peak memory;
- optimizer steps at registration;
- changed backbone, embedding-table, and LM-head parameter counts;
- exact maximum change of held-out static input/output rows;
- bytes stored per tool;
- selection and argument latency at each registry size;
- full-vocabulary physical-address hit@1 and ordinary-token win rate;
- same-KV-stream address emission, dereference, and argument continuation.

The deployment contract requires one registration forward, zero deployment
optimizer steps, zero model/table mutation, and zero evaluation-address
training exposure.

### Incremental Fixed-Token Cost Baseline

The `qwen_toolgen_incremental` cost track starts from the completed controlled
fixed-token Agent checkpoint, appends every group-disjoint test-tool token, and
continues only the ToolGen document-to-token memorization stage. It receives no
test retrieval query, argument answer, or trajectory. This is the leakage-free
comparison for registration cost: the fixed-token method may update parameters
and token tables, while late binding receives the same tool documents through
one compiler forward with zero updates. Report total and per-tool wall time,
optimizer steps, changed-parameter status, unseen-tool retrieval and argument
metrics at registries 10/100/1K, and old-tool regression. A separately labeled
supervised-query upper bound may be added later but cannot replace this row.

The incremental checkpoint must prove that every old train-token physical ID
is unchanged after the tokenizer grows. New-tool and old-tool evaluations use
the unchanged controlled benchmark rows; evaluation queries and answers are
never reused as incremental training labels.

Every latent training and evaluation artifact must also carry a physical-token
identity audit. The audit records that all reserved strings tokenize to one
atomic ID, the number of distinct IDs equals the complete reserved pool, the
IDs occupy one contiguous preallocated range, and the exact string-to-ID
mapping has a valid SHA-256 digest. Readiness and the final paper gate reject
the run if any of these conditions is absent or false; a logical address or a
multi-token spelling cannot substitute for a dedicated physical token.
The same registration contract must state that selection emits exactly one
physical ID, ordinary vocabulary rows remain in the selection denominator,
inactive reserved IDs are masked, and the winning ID dereferences its own
registered payload rather than consuming a static embedding row. The payload
is the selected full document in the primary `common_document` condition and
the registered memory in the auxiliary `registered_memory` condition; the two
conditions must record distinct, mutually consistent dereference flags. Every
registered winner also records a per-example check that its emitted physical
ID maps to the same registry identity used for payload lookup; result-level
audits report the checked and passed counts and reject any mismatch.

## Gates

### Pilot Gate

The 5K pilot advances only if:

1. all source and subset hashes are recorded;
2. the tool-token mapping is one-to-one after normalization;
3. the first five optimizer steps are finite and use all requested GPUs;
4. a checkpoint reload produces byte-identical tokenizer mappings;
5. paper-compatible and corrected metrics recompute from saved predictions;
6. no BFCL reporting definition appears in training data.

### Claim-Level Gate

- unseen-tool plus unseen-address full-vocabulary selection is nonzero and
  significantly beats blank, random, permuted, query-only,
  nearest-trained-token, and shared-vector controls;
- after selection, the `common_document` path produces effective calls and is
  evaluated against fixed-token ToolGen, incremental ToolGen, and the
  correct-document oracle on identical examples; all absolute gaps and ratios
  are reported without an arbitrary pass threshold;
- wrong-memory and registered-memory argument recovery are reported as
  auxiliary diagnostics and do not gate the core registration claim;
- closed-set retention and old-tool behavior after append are reported with
  confidence intervals rather than a preselected two-point cutoff;
- registration satisfies the one-forward, zero-update, zero-mutation contract.
- final artifacts revalidate tool/address quadrants, shared-Agent and
  late-bound-checkpoint identity, control treatment labels, runtime/storage and
  latency reporting, and the exact old identity/address prefix after append;
  directory names and submission exports alone are not identity evidence.

Before spending three full independent seeds, the exploratory checkpoint must
also pass the machine-readable atomic readiness gate produced by
`latent_register.assess_formal_readiness`: nonzero full-vocabulary Hit@1,
paired-bootstrap lower bounds above zero against every declared selection
control, and the complete one-forward/zero-update/isolation/mutation audit. The
wrong-memory artifact is hash-checked but remains a memory-only diagnostic. The
gate uses at least 2,000 retrieval examples and rejects smoke subsets. It has no
self-defined absolute accuracy, key-recall, or oracle-retention threshold.

For argument-family rows, `end_to_end_argument_exact`,
`end_to_end_key_exact`, and `end_to_end_key_recall` are zero unless the
predicted tool-call set is also exact. Standalone argument metrics remain
available as diagnostics but cannot satisfy an end-to-end gate.

## Execution Order

1. Reproduce official ToolGen retrieval on its released checkpoint before
   training, validating both NDCG implementations.
2. Run the 5K three-stage Llama-3-8B ToolGen pilot and record measured
   throughput for each sequence-length stage.
3. Run the full official Llama-3-8B reproduction.
4. Train `qwen_full_document` and `qwen_toolgen_fixed` on the frozen controlled
   manifest.
5. Evaluate the late-bound checkpoint on the identical manifest and
   information conditions.
6. Run BFCL and the late-binding mechanism bench.
7. Add ToolWeaver only after the atomic ToolGen comparison is complete.

Formal late-bound seeds use
`slurm/a100_train_late_bound_formal.sbatch`: 1,000 retrieval-registration
steps followed by 3,000 full-vocabulary joint steps on six A100s. Both stages
always log optimizer steps 1--5, require exact held-out-row isolation, reject
non-finite result values, and write seed/step/world-size audits before exposing
a `COMPLETE` checkpoint to evaluators. The earlier seed-17 continuation chain
remains an exploratory readiness run and is not one of the three formal seed
replicates.

Build and inspect a seed DAG without submitting jobs:

```bash
PYTHONPATH=src python -m latent_register.submit_formal_pipeline \
  --seed 29 \
  --project-root "$PWD" \
  --manifest /path/to/seed-29-submission.json
```

Add `--submit` only after reviewing the manifest/printed commands. A real
submission refuses to overwrite an existing manifest and atomically rewrites
it after every successful `sbatch`, so a VPN disconnect or later submission
failure cannot erase already allocated job IDs. Each seed DAG contains 25 jobs
covering all three learned systems, matched and large-registry evaluations,
paired fixed/oracle comparisons, semantic controls, query-only,
nearest-trained, and sequential append.

A real submission additionally requires
`--readiness /path/to/passed/readiness.json`. The file must have its sibling
`COMPLETE` marker and must report `formal_latebound_readiness_gate` with no
failed checks. Dry-run planning may omit it; `--submit` may not. The readiness
path and SHA-256 are copied into every seed submission manifest.

After all three seed DAGs complete, aggregate all 27 comparison cells in one
audited pass:

```bash
PYTHONPATH=src python -m latent_register.aggregate_formal_pipeline \
  --manifest 17=/path/to/seed-17-submission.json \
  --manifest 29=/path/to/seed-29-submission.json \
  --manifest 43=/path/to/seed-43-submission.json \
  --output-dir /path/to/formal-three-seed-aggregate
```

The cluster wrapper is `slurm/a100_aggregate_formal_pipeline.sbatch`. Generate
and inspect its cross-seed dependency submission without changing Slurm state:

```bash
PYTHONPATH=src python -m latent_register.submit_formal_aggregation \
  --manifest 17=/path/to/seed-17-submission.json \
  --manifest 29=/path/to/seed-29-submission.json \
  --manifest 43=/path/to/seed-43-submission.json \
  --project-root "$PWD" \
  --submission-manifest /path/to/aggregation-submission.json
```

Only add `--submit` after inspecting the generated manifest and command. The
submitter builds one `afterok` dependency over the nine terminal comparison
jobs from each seed DAG, rejects duplicate job IDs, and refuses to overwrite a
real submission record. The all-cell aggregator rejects dry-run or errored
submission manifests, missing jobs, incomplete comparison cells, and any cell
count other than 27. It writes one artifact per cell, an atomic
`aggregate.json`, and a digest-bearing `COMPLETE` marker.

Paired-comparison and cross-seed aggregation code itself is CPU-only, but the
account's visible Slurm partitions reject jobs that do not request a GPU.
These wrappers therefore request the mandatory minimum of one GPU and keep
short time limits; model inference evaluations retain their own explicit GPU
requests.

Each cell delegates to `latent_register.aggregate_formal_seeds`, which recomputes a
prediction-independent fingerprint over example IDs, queries, references, and
schemas and requires it to match within and across seeds. The fingerprint
normalizes the original source condition and deliberately excludes the
system-specific evaluated condition (for example, fixed-token versus
late-bound unseen-address), which is a treatment label rather than reference
data. It preserves each
seed's paired-bootstrap interval over shared examples, then separately reports
the mean, sample standard deviation, and Student-t 95% interval of the three
seed-level deltas. Predictions from different training seeds must not be pooled
as though they were additional independent benchmark examples.

The same aggregation job then runs
`latent_register.assess_formal_paper_gate` and writes `paper_gate.json` plus
`PAPER_GATE_COMPLETE`. Its claim gates are significant unseen-token selection
gains over every selection control and exact training/runtime isolation audits
for all three seeds. Common-document end-to-end results, fixed/incremental
ToolGen gaps, correct-document oracle ratios, closed-set retention,
registered-memory readback, and append regression are report-only measurements,
not arbitrary pass thresholds. A scoring-integrity check requires identical comparison/metric code
hashes across all 27 cells and identical metric code between evaluator and
comparator. A completed report may legitimately contain `passed=false`; the
completion marker proves the decision was computed, not that the claim passed.

## Access Amendment 1

Meta-Llama-3-8B currently returns HTTP 403 for the cluster account. Therefore
execution step 2 uses Qwen3-8B for the deterministic 5K pipeline and throughput
gate. This pilot does not count as an official-paper result. The released
Llama-3-8B ToolGen checkpoints remain the source for official-fidelity
retrieval and end-to-end evaluation, while from-base Llama training in step 3
is deferred until authorization is available. The access limitation does not
block the controlled same-model Qwen3 comparison in steps 4--6.
