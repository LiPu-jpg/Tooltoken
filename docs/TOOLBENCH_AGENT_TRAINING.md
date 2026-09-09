# ToolBench: training audit and trajectory-based improvement

This fork starts from `HIT-HJC/Tooltoken` commit
`82510f93d4c1ac13575471e02ab4dd688e1cbe3d`. The new implementation is on
`native/toolbench-call-training-v1`. **ToolBench is the target; STQ's single
string parameter-array task is not its training contract.**

No benchmark scores, original experiment drivers, datasets or historical
checkpoints were changed. No GPU jobs or external simulator/judge requests
were made during this implementation. Synthetic tests are not research results.

## Findings in the upstream code

| Finding | Evidence | Consequence / scope |
|---|---|---|
| The formal ToolBench-10K builder emits tools and query-to-API retrieval records. Its driver trains `train_meta_registration`. | `experiments/ours-meta-toolbench10k-formal/build_ours_meta_toolbench10k.py`; `run_ours_unseen_meta_20260906.sh` | That run supervises selection, not multi-step calling, observations or task completion. High retrieval Hit does not establish high SoPR. This does not certify what code/checkpoint the senior used in a separate run. |
| The generic readback examples come from ToolACE; ToolGen trajectories are stored separately. | `prepare_scale_data.py`, readback/trajectory output sections; `train_meta_readback.py`, loading `readback.jsonl` | The existing retrieval dataset cannot by itself supply ToolBench argument and completion labels. |
| Teacher-forced readback uses a JSON-input instruction; same-stream inference starts with the selection prompt and immediately appends memory. | `train_meta_readback.py`: `render_readback_prompt`, `prepare_conditioned_batch`, `generate_same_stream_one` | Training and inference conditioning differ. A plausible source of degradation, not a proven explanation for a particular SoPR score. |
| Documents and argument targets are silently truncated, including targets with an appended EOS. The formal retrieval driver's document limit is 64. | `tokenize_documents`, `prepare_conditioned_batch`; formal ToolBench shell driver | ToolBench schema fields or valid JSON endings can be lost. Whether/which real examples are affected needs a training-data token-length audit. |
| The readback entry point updates compiler/optional LoRA, not the whole backbone. Its prompt and target embeddings are detached. | `train_meta_readback.py`, model loading and `prepare_conditioned_batch` | Detach is compatible with its frozen-base setup, but cannot simply be carried into a full-backbone recipe. |
| Wrong-memory ranking pairs another API with the same target arguments without establishing semantic incompatibility. Schema auxiliary targets enumerate keys with null values. | `MetaReadbackModel.forward` and schema helpers | These are not sufficient supervision for nested types, required fields, enum values, or successful tool use. Wrong-memory degradation alone is not evidence of useful memory. |
| Some old training paths automatically evaluate validation and test after training. | End of `train_meta_readback.py` | The new entry point must remain train-only and require explicitly selected evaluation inputs. |

The stage-2 objective therefore keeps **selection supervision** while adding
actual argument and continuation supervision. It does not assume that memory
must improve Hit, or that full document inputs must be inferior.

## Implemented architecture

There is **one shared trainable backbone**, not a frozen E0 paired with a
separately updated caller. For the new `full` recipe, all backbone parameters,
including input embeddings and LM head, receive the joint training objective.

For each current document forward:

```text
tool description + complete JSON Schema
                  |
          shared backbone (all final-layer token states)
                  |
        +---------+----------------+
        |                          |
masked mean -> C_sel              C_mem -> 8 vectors
        |                          |
   dynamic output row          selected-tool content
```

For each serial trajectory decision:

```text
user + past calls + past observations
       -> generate a short plan
       -> last selection-prompt hidden state q
       -> scores q @ O.T over the available registry, including Finish
       -> select exact API by runtime binding
       -> read that API's 8 memory vectors and generate a complete JSON object
       -> execute callback -> append observation -> next decision
```

Finish is a shared protocol action with `give_answer` / `give_up_and_restart`
arguments. It ends the trajectory without executing an external API. This
branch implements an explicit ToolBench action/Finish decision; it does not
train the separate `<tool_request>` protocol, an implicit per-token gate, or
claim full-vocabulary competition with the dynamic tool rows.

This is the senior's **select-then-read-one-tool** branch: one selected tool
contributes 8 memory positions. It is not the separate STQ experiment that
injected 5 tools / 40 positions before selecting a slot. Each phase uses a
fresh prefill, followed by normal cached decoding. No mid-cache insertion is
claimed. Planning, selection and arguments use the same prompt functions in
training and inference. The current selected API identity is bound outside
the argument prompt; the prompt receives its content representation.

Registration happens after freezing the backbone and compilers. Each new API
gets one document forward producing both outputs, with zero optimizer steps
and no new API-specific trainable parameter. Old cache entries are reused;
changed documents under the same identity require a fresh registry snapshot.
Entering training clears the cache. A registry belongs to its model instance;
do not mutate frozen weights in place or transplant caches across checkpoints.

## Training objective and boundaries

For each step:

```text
loss = selection CE + argument-token CE + 0.2 * planning-token CE
```

Token losses include EOS and average over each step's complete target. Steps
then receive equal weight within a batch. Tool observations, source documents
and prompts are inputs, not targets. The Finish argument loss provides final
answer / give-up supervision. A later observation or answer never enters an
earlier step's prefix. Empty planning targets learn immediate EOS.

Selection training uses sampled training APIs, always including the gold and
Finish, and shuffles positions. The candidate count is configurable. This is
sampled classification, not an unbiased estimate of the full-registry softmax.
Inference scores the actual available registry and does not add gold APIs.
Maintaining a selection loss does not guarantee retention of earlier retrieval
metrics; that must be measured with a matching development protocol.

The training modes are:

| Mode | Backbone | C_sel | C_mem |
|---|---|---|---|
| `full`, condition `memory` | All parameters trained | Trained | Trained |
| `compiler_only`, condition `memory` | Frozen | Trained | Trained |
| `full`, condition `full_document` | Independently trained caller | Trained | Unused, frozen |
| `full`, condition `query_only` | Independently trained caller | Trained | Unused, frozen |

Full-document and query-only are matched **argument-reader** baselines: all
three conditions retain the same document-based selection branch. Query-only
therefore means no selected-tool document/memory in the argument prompt,
not no tool representation anywhere in the system. Changing an already trained
Native checkpoint's condition is only an input ablation.

The new path does not use the wrong-memory margin objective or the null-key
schema auxiliary. ToolBench's actual typed argument labels provide the main
calling supervision. Schema validation rejects malformed labels, unsupported
schema wrappers, and external `$ref` URLs. It does not prove that the model
uses memory; correct/blank/wrong-memory development comparisons are still needed.

## Required training data

`train_tools.jsonl` contains only the training API set. Every entry requires:

```json
{"api_identity":"provider/tool/exact_endpoint","split":"train","document":"Tool purpose and calling instructions.","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city","days"]},"aliases":["exact_toolbench_function_name","<<Tool&&endpoint>>"]}
```

The aliases must map one-to-one to exact API identities. Do not infer identity
from a display name. The formal retrieval builder's `api_params` metadata
wrapper is **not automatically a JSON Schema**. Supply the actual schema from
the training tool catalog, with its provenance; do not fabricate one from STQ
or silently discard unresolved endpoints. The adapter appends the complete
schema to each registration document.

Trajectories support two explicitly selected formats:

- `toolbench`: serial `messages` or `conversations`, with an initial user query,
  assistant `function_call` or one `tool_calls` entry, corresponding
  `function`/`tool` observations, and an explicit Finish call.
- `toolgen`: assistant action token, the known
  `Please give the input. Here is the documentation:` wrapper, assistant
  JSON arguments, and function/tool observation. Optional preceding planning
  text is supervised. The wrapper is replaced by the registry document/memory.

Nested objects, arrays, numeric/boolean/string types and array order are
preserved. Duplicate JSON keys, trailing junk and incomplete targets are
errors. Parallel calls, unknown actions, missing observations, unfinished
traces, or unrecognized control formats stop conversion with an identified
record. They are not silently flattened or skipped. The adapter supports the
converted serial formats above, not raw DFS tree dumps or every ToolBench
export variant. Actual training-source compatibility remains to be audited.

To adapt an explicitly selected training export (CPU, no model load):

```bash
PYTHONPATH=src python -m latent_register.prepare_toolbench_agent \
  --tools /path/to/train_tools.jsonl \
  --training-source /path/to/toolgen_atomic_G123_dfs.json \
  --input-format json-array --source-format toolgen \
  --replace-source-system-prompt \
  --output-dir /path/to/NEW-prepared-toolbench
```

`--replace-source-system-prompt` explicitly replaces the upstream agent
protocol and any enumerated tools with this branch's common protocol. It does
not copy the source system prompt into model inputs. Check source-specific
task instructions before using the flag. The query and prior observations
remain intact. The input file is explicitly declared to be a training export;
this declaration is not independent proof of its upstream provenance.

Outputs include copied training registry, adapted trajectories, hashes and
counts. A failed conversion keeps partial evidence and a `FAILED.json`; the
trainer refuses that directory. No test files are discovered or opened.

## Running joint training

```bash
PYTHONPATH=src python -m latent_register.train_toolbench_agent \
  --tools /path/to/NEW-prepared-toolbench/train_tools.jsonl \
  --trajectories /path/to/NEW-prepared-toolbench/train_trajectories.jsonl \
  --source-format toolgen --audit-only \
  --output-dir /path/to/NEW-data-audit
```

The real training entry point supports Accelerate and a separately chosen
distributed configuration. For example, after configuring appropriate GPU
resources, run:

```bash
PYTHONPATH=src accelerate launch --config_file /path/to/reviewed-accelerate.yaml \
  -m latent_register.train_toolbench_agent \
  --tools /path/to/NEW-prepared-toolbench/train_tools.jsonl \
  --trajectories /path/to/NEW-prepared-toolbench/train_trajectories.jsonl \
  --source-format toolgen \
  --model-path /path/to/base-Qwen3-8B --model-role base \
  --train-mode full --condition memory --gradient-checkpointing \
  --epochs 3 --batch-size 1 --gradient-accumulation-steps 8 \
  --candidate-count 8 --compiler-rank 128 --memory-slots 8 \
  --max-context-length 6144 --max-document-length 2048 --max-target-length 1024 \
  --learning-rate 1e-5 --compiler-learning-rate 1e-4 \
  --seed 17 --output-dir /path/to/NEW-toolbench-training
```

These values are a configurable starting recipe, not a validated optimum or
approved resource allocation. Document candidates participate in the current
backbone gradient graph; memory use scales with candidate count and document
length. No cached frozen document states are used during joint training.
Measure real 8B memory/throughput before a long run. The code's Accelerate
integration has been tested on CPU only; CUDA, multiple ranks and DeepSpeed
are not yet certified. An appropriate distributed configuration must be
provided; no Slurm launcher or automatic retry is included.

To continue a stage-1 retrieval model, use `--model-role warm-start` and a
`--warm-start-manifest` containing the union of exact `train_api_identities`
from **every** upstream training stage. Optional `--retrieval-adapter` is
merged into the base before full training; `--retrieval-compiler` loads the
upstream `compiler.pt`. The manifest must be contained in this training API
set. This is a lineage consistency check, not independent certification of
the checkpoint's history. A Hub model reference requires an immutable 40-digit
revision. Local base/checkpoint authenticity must be verified separately.

The final partial accumulation window is flushed with its actual denominator.
Output directories must be new. Final export includes the complete backbone,
both compilers, tokenizer, interface settings, source/API metadata, and hashes.
Actual backbone buffers, including nonpersistent RoPE frequencies, are saved
and restored; this is not a claim that the historical 8B ZeRO discrepancy has
been diagnosed or fixed. There is no automatic dev/test evaluation and no
optimizer-resume claim for this serving export.

## Calling and evaluation

```python
from latent_register.toolbench_checkpoint import load_agent
from latent_register.toolbench_agent import run_serial_agent
from latent_register.toolbench_data import finish_tool, FINISH

agent = load_agent("/path/to/checkpoint", device="cuda")
# tools: explicit mapping exact_identity -> ToolSpec from the serving catalog.
tools[FINISH] = finish_tool()
trace = run_serial_agent(agent, user_query, tools, execute=your_tool_executor)
```

`your_tool_executor(exact_identity, arguments)` is the boundary for the senior's
existing injection/calling environment or a StableToolBench adapter. This fork
does not ship credentials, invoke a simulator, or silently replay gold tool
results. It catches execution errors as observations, enforces a call budget,
and refuses invalid/truncated argument generations. Its trace is a local
diagnostic format, **not the official ToolEval/SoPR output**. Connecting the
senior's real backend and official result conversion remains separate work.

For the next authorized ToolBench experiment, compare memory and full-document
callers with the same training exposure, API split, candidate registry,
executor/simulator responses, judge, and budgets. Report retrieval Hit/Recall,
argument validity, execution errors, successful completion and costs separately.
Use a registry-only selection check and an oracle-tool readback diagnostic to
localize errors, but keep both labeled as diagnostics. End-to-end runs must
choose from their actual registry and receive actual environment feedback.
Do not compare a custom simulator's score directly with a published SoPR or
infer that changing simulator/judge necessarily explains a low score.

## Validation

On 2026-09-09 the explicit suite below passed **56 tests** (32 new ToolBench
tests and 24 upstream regression tests) on CPU. After the final checkpoint
loading checks were tightened, the two checkpoint/actual-training-CLI tests
were rerun and passed. Environment: Python 3.9, PyTorch 2.2.2, Transformers
4.57.6, Accelerate 1.10.1. A system LibreSSL/urllib3 warning was emitted; the
tests used local fixtures and no remote model or tool API.

The synthetic CPU suite covers causal trajectory construction, typed/nested
schemas, exact binding, full 8-slot gradients, shared-backbone/LM-head gradients,
compiler-only freezing, matched reader baselines, strict length and EOS rules,
KV continuation, incremental zero-step registration, self-contained checkpoint
reload and nonpersistent buffers, serial observation/Finish handling, and the
actual training CLI including a partial accumulation window.

Run only the explicit tests; no benchmark data or 8B weights are needed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_toolbench_agent.py tests/test_meta_readback.py \
  tests/test_meta_registration.py tests/test_memory_oracle.py tests/test_physical_tokens.py
```

One shared implementation change in `model.py` makes `GeneratedVector` and
`TokenResamplerMemory` arithmetic explicitly FP32 with differentiable parameter
casts, even when an engine casts their stored weights to BF16. All-masked
documents are rejected. Historical snapshots outside this fork remain intact;
new GPU numerical results must be reported under this changed implementation.
