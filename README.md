# Native Late-Bound Tool Registration

This repository contains the public implementation of native late-bound tool
registration. A shared encoder/compiler is trained once; a new executable API
is then registered from one document forward pass, without API-specific
optimizer updates or changes to the frozen model's static vocabulary rows.

## Method

For a query `x` and tool document `D`, the shared modules compute

```text
q_x = Q(x)
h_D = E(D)
(O_D, M_D, X_D) = C(h_D)
```

`O_D` contains dynamic output rows, `M_D` is optional read-back memory, and
`X_D` is the exact immutable API identity and execution payload. At runtime a
registry binds a physical address to this bundle:

```text
address -> (exact identity, document, output rows, memory, payload)
```

The selector writes `q_x dot O_D` into the ordinary language-model logit
column for that address. If it wins, the registry resolves the address back to
the same exact API. Physical IDs are addresses only; they are not learned
semantic representations.

The registration contract is:

```text
optimizer steps for a new API = 0
API-specific learned parameters = 0
document forwards per distinct API = 1
static model/tokenizer rows changed = 0
```

## Repository layout

- `src/latent_register/`: model, registry, data, training, evaluation, and
  audit primitives.
- `tests/`: unit and contract tests for identity, masking, metrics, and
  registration invariants.
- `experiments/`: source-only snapshots for the current A100 ToolBench I1 and
  L20 STQ staged runs, including their training/audit entrypoints and Slurm
  contracts.
- `docs/PROJECT.md`: paper scope, claims, and comparison boundaries.
- `docs/PROTOCOL.md`: reproducibility and evidence rules.
- `docs/protocols/BENCHMARK_PROTOCOL.md`: benchmark definitions and metric
  conventions.
- `docs/EXPERIMENTS.md`: dated experiment ledger, failure analysis, and
  evidence status.
- `PAPER_TABLES_WORKING_2026-08-09.md`: evidence-bound paper table ledger;
  pending or blocked cells are kept explicit.

Large model weights, benchmark dumps, qrels, prediction files, checkpoint
artifacts, run logs, and cluster credentials are intentionally excluded.
Supply those inputs locally and record their hashes when reproducing a run.

## Latest audited diagnostic

The current Native Qwen3-8B 1K closed-set regression used 1,000 candidate APIs,
1,951 queries, and 240 target identities with disjoint train/test query text.
After a blind-score import-path fix, the sealed aggregate passed with Hit@1
86.21%, Hit@3 90.62%, Hit@5 91.95%, exact MRR 88.82%, and NDCG@5 89.34%.
Registration used zero optimizer steps. This is a source-derived closed-set
diagnostic, not an official ToolScalER reproduction; full evidence and large
artifacts remain outside this public repository.

## Installation

Python 3.9 or newer is supported. Install the package and development tools:

```bash
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

The core library uses PyTorch and Hugging Face Transformers. GPU training is
optional for the unit-test suite.

## Reproducibility

Use a pinned model/tokenizer revision and an immutable input manifest. Keep
training and evaluation splits disjoint at the exact API level. Report the
candidate denominator, seed, model revision, source hash, and registration
audit alongside every metric. Do not open sealed qrels or report partial
scores before the complete artifact audit has passed.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
