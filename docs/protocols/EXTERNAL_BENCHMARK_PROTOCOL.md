# External Benchmark Protocol: BFCL v4 and tau2-bench

Status: additive external evaluation protocol. It does not modify the frozen
ToolGen versus late-bound controlled DAG in `BENCHMARK_PROTOCOL.md`.

## 1. Question

Measure whether one-forward post-training registration improves tool use on
public, evaluation-only tools while holding the model, downstream Agent,
prompts, examples, decoding, and hardware constant.

Registration means:

1. read only the benchmark-supplied tool name, description, and schema;
2. run one frozen document forward per distinct tool, batchable across tools;
3. produce a dynamic output row and latent memory bundle;
4. bind that bundle to a never-trained physical token ID;
5. perform zero optimizer steps and zero parameter, embedding, LM-head, or
   tokenizer mutation.

No benchmark query, reference call, answer, simulator state, or task reward may
enter registration or training.

## 2. Common Paired Treatments

Every example or task is evaluated under the following paired conditions. A
pair uses the same checkpoint, Agent, candidate tools, example, seed, decoding
settings, and downstream document.

| Condition | Selection path | Document available to downstream Agent |
| --- | --- | --- |
| `full_document_standard` | Standard benchmark function interface | Exactly the benchmark-provided function documents |
| `late_bound_registered` | One-forward bundle, full-vocabulary physical-token selection | Only the selected token's own document |
| `query_only` | Reserved token IDs masked; no active registry | None through the registry path |
| `blank` | Active physical IDs with zero bundles | Selected document only if a physical ID wins |
| `random` | Active physical IDs with norm-matched random bundles | Selected document only if a physical ID wins |
| `permuted` | Correct bundles assigned to the wrong identities | Selected token's mapped document |
| `nearest_trained` | Nearest meta-training bundle substitutes for the new bundle | Selected token's mapped document |

`query_only` is a strict mechanism sanity control, not a substitute for the
normal full-document baseline. The primary comparison is
`late_bound_registered` versus `full_document_standard`; the remaining controls
test whether any gain is specifically caused by the registered bundle.

## 3. BFCL v4

Pin the official Gorilla repository by commit hash and archive the exact data,
possible answers, evaluator revision, model adapter, and category list. The
local file currently named `data/bfcl4.tar.gz` contains BFCL v3 files and must
not be treated as BFCL v4 evidence.

### 3.1 Official track

Use the function set attached to each official BFCL example and the unmodified
official evaluator. Run in this order:

1. `simple_python`;
2. `multiple`, `parallel`, and `parallel_multiple`;
3. `simple_java` and `simple_javascript`;
4. live and irrelevance categories;
5. multi-turn categories;
6. agentic memory and web-search categories only after their external backends
   and credentials are pinned.

Report official per-category scores and the official aggregate. Also retain
per-example tool names, canonical calls, parse failures, invalid tools, and
latencies so registered and standard conditions can be paired.

### 3.2 Registry stress track

This is a new mechanism benchmark and must not be labeled as an official BFCL
leaderboard score. Deduplicate BFCL tool documents by canonical
`(name, description, schema)` identity, register all tools in a category-wide
shared registry, and query against that registry.

The first confirmatory subset is `simple_python`, because each example has a
single reference call and isolates physical-token selection from multi-call
planning. Later categories are diagnostic until the model supports emitting a
complete ordered physical-token call sequence.

For every selected physical token, audit:

```text
generated vocabulary ID
-> active registry address
-> canonical BFCL tool identity
-> that identity's own document
-> shared downstream Agent
-> canonical tool call
```

Primary outputs are full-vocabulary Hit@1, ordinary-token win rate, selected-ID
binding validity, own-document dereference validity, end-to-end tool-name exact,
canonical argument exact, JSON parse rate, schema validity, registration
latency, inference latency, and bytes stored per tool.

Multiple and parallel categories additionally report target-set recall and
ordered call exactness, but retrieval top-k must not be relabeled as official
multi-call exactness.

## 4. tau2-bench

Pin `sierra-research/tau2-bench` release `v1.0.1` or a later explicitly recorded
revision. Use the `base` task split for compatibility with the original text
benchmark. Start with text-only `airline`, `retail`, and `telecom`. Exclude
voice and `banking_knowledge` from the first run because they add audio or RAG
systems unrelated to token registration.

At each Agent tool decision:

```text
current dialogue and tool observations
-> registered physical-token selection
-> own tool schema dereference
-> the same Agent generates arguments
-> tau2 environment executes the call
-> next turn
```

The standard condition gives the same Agent the normal tau2 tool interface.
The registered and control conditions vary only the selection interface.

Block and pair runs by `(domain, task_id, trial, user-simulator seed)`. Fix the
user simulator model, Agent model, policy text, tool schemas, task split,
maximum turns, temperature, and environment revision. Randomize treatment run
order within task blocks so time and node effects do not align with one
condition.

Report task reward/pass rate, action correctness, invalid-tool rate,
tool-selection accuracy where a reference action exists, argument/schema
validity, turns, tool calls, latency, and registration cost. Task reward is the
official end-to-end outcome; selection diagnostics explain failures but do not
replace it.

## 5. Analysis

The independent unit is the BFCL example or tau2 task-trial pair, not each token
or dialogue turn. Compute paired per-unit differences and paired bootstrap 95%
confidence intervals. Report all runs and seeds; do not choose a seed after
seeing accuracy. No self-defined absolute accuracy threshold determines
feasibility.

Evidence for the registration claim requires all of the following:

1. a nonzero full-vocabulary physical-token selection signal on evaluation-only
   tools and never-trained evaluation token IDs;
2. superiority over blank, random, permuted, query-only, and nearest-trained
   controls under paired tests;
3. zero-update and zero-mutation audits;
4. verified selected-ID-to-own-document dereference;
5. an end-to-end comparison against the standard full-document condition under
   the same downstream Agent.

BFCL and tau2 results supplement the controlled ToolGen comparison. They cannot
replace the frozen controlled `summary.json`, and no feasibility claim is made
until that summary and these external run audits are complete.

## 6. Execution Plan on L20

1. Install and pin BFCL v4 and tau2-bench in isolated environments.
2. Materialize hashes and leakage audits without loading an 8B checkpoint.
3. Run CPU/unit smoke tests for data conversion and official evaluators.
4. Run 16-example BFCL `simple_python` paired GPU smoke with an exploratory
   late-bound checkpoint.
5. Run 3 tasks per tau2 text domain and one trial as an integration smoke.
6. After the formal late-bound checkpoint is complete, run the frozen paired
   BFCL matrix, followed by tau2 with the predeclared task/trial schedule.

The external jobs must not consume GPUs allocated to or change artifacts used
by the A100 controlled DAG.
