# Experiment snapshots

This directory contains source-only snapshots of the two latest Native
Late-Bound experiment lines run on the A100 and L20 clusters. Each snapshot
keeps its model/data preparation scripts, training and audit code, Slurm
entrypoints, tests, protocol, and a generated `SOURCE.sha256` catalog.

The snapshots intentionally omit model weights, checkpoints, benchmark data,
qrels, prediction matrices, run logs, and cluster output directories. Supply
those inputs locally and override the path variables documented by each
snapshot before submitting a job.

## Published snapshots

- `native-toolbench-i1-v11/`: four-A100 Native ToolBench I1 API-disjoint
  full-parameter sequence-SFT line. The formal run is qrels-blind until the
  complete audit and sealed aggregation.
- `native-stq-staged-v6/`: four-L20 Native STQ staged schedule recovery. It
  contains the Stage-L/Stage-F training, reload, and dependency-chain gates.

These are reproducibility snapshots, not claims that every listed run has
produced a paper-eligible metric. See `docs/EXPERIMENTS.md` for current run
status and evidence gates.
