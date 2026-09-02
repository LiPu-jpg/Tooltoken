# Native STQ staged ablation v6

Version 6 is a reload-environment repair after v5. v5 Stage-L completed the
full 10,483-row run and produced a valid checkpoint, but its one-GPU reload
gate failed before importing the ZeRO-3 state materializer because CUDA_HOME
was not set. v6 recreates the job-local CUDA shim in both reload jobs and
reuses only the successful v5 Stage-L checkpoint; all v4/v5 outputs remain
immutable.

This deployment is our Native late-bound method, not an official ToolScalER
reproduction. It compares two schedules on the fixed SimpleToolQuestions
unseen split: `pure_full_sequence_sft` (the v19 control) and
`staged_lora_then_full` (this deployment).

Stage L attaches LoRA only to Qwen3-8B `q_proj/k_proj/v_proj/o_proj` and trains
the shared Native compiler. Stage F loads that checkpoint, verifies and merges
the adapter, removes every LoRA parameter, unfreezes the full backbone, and
continues ordinary causal sequence SFT. No per-API parameters, codebook, E5,
GIST, sampled softmax, or fixed negative-document set is used.

The fixed split contains 999 seen training tools and 10,483 training rows,
plus 837 unseen candidates and 1,066 test queries. Exact identity overlap is
zero. Test registration uses one document forward and zero optimizer steps;
qrels are outside this deployment and are not read during training or smoke.

`slurm/preflight.sbatch`, `stage_l_reload.sbatch`,
`stage_f_formal.sbatch`, and `stage_f_reload.sbatch` form the formal afterok
chain. `submit_l20_formal.sh` submits the chain only after the preflight gate.
Stage-L is reused from the audited v5 checkpoint; Stage-F consumes the same
complete 10,483-row seen split with one epoch. The smoke used five steps only
and is not a result. `SOURCE.sha256` binds all
source, scripts, and configs in this public snapshot; behavior changes require a new
versioned deployment.

The public snapshot keeps the source, Slurm contracts, and tests but omits the
deployment's input manifest, checkpoints, and run artifacts. Set
`PROJECT_ROOT`, `MANIFEST`, `ARTIFACT_ROOT`, `MODEL_ROOT`, `NATIVE_ROOT`,
`PYTHON_BIN`, `RUN_ROOT`, and `SOURCE_STAGE_L` to site-local paths before using
the submission wrappers.
