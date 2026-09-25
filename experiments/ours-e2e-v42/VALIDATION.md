# Source-export validation — 2026-09-25

Command: `python3 experiments/ours-e2e-v42/check.py`.

| Check | Result |
| --- | --- |
| Export source hashes | 62/62 matched |
| v42 core and inherited runtime regressions | 78 passed, 1 skipped |
| FP32 memory/full-document reader | 6 passed |
| completion-v6 source-patch and real-renderer integration | 25 passed |
| native / reader / completion-v6 CLI help | All passed |

Total: **109 passed, 1 skipped**. The skipped test compares with the fixed
upstream StableToolBench converter and requires
`NATIVE_TEST_STABLETOOLBENCH_SOURCE`; that external checkout is not bundled.
Dependencies are recorded in `requirements-cpu.txt` (Python 3.9, PyTorch 2.2.2,
Transformers 4.57.6). The host emitted the existing urllib3/LibreSSL compatibility
warning. HTTP tests use loopback; model fixtures are locally initialized tiny Qwen.

Export corrections discovered during verification: include the training helper
and preparation CLI required by inherited tests; pass the legacy E2E test's
1024/1024 budgets explicitly. No frozen runtime source was changed.

This validates the exported CPU contracts, checkpoint roundtrips, import paths
and CLI parsers. It does not establish 8B generation quality, CUDA behavior,
simulator/judge availability or a new task-success/SoPR score. No GPU jobs or paid
API evaluations were launched for this export.
