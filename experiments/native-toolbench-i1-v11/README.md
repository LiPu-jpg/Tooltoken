# Native Late-Bound ToolBench 官方 I1 API-disjoint v11 four-A100 formal training

这是冻结 `latent-register/` 之外的新实验 deployment。目标是在 1,000 个与官方
ToolBench G1/I1 测试 action 完全不相交的训练 API 上训练 Native Late-Bound 的共享
backbone/compiler，然后在官方 G1/I1 测试集上对未见 API 做 zero-step 注册和检索。

## 当前输入

- manifest：`artifacts/manifest-api-disjoint-1k.json`
- manifest SHA-256：`cb82ccf1cfcaf803bff88b627a212f30586ec32b6436253fcf0208d95db7cdfa`
- ToolGen retrieval：`a608f573651d645ed949dbf1359b112735dc91f6e791ac5029bf7541c30e546a`
- G123 corpus：`851ac79ad84704e1430d6c6ba6922283d506cceba364b79b7bc3f75e4c0fb407`
- 官方 atomic map：`6f1d0406396fd893e4633661a84bc0606b93a8876f10d4348a4311831e2ddc9d`

训练 manifest 只包含 1,000 个 singleton exact API。训练 action 与官方 G1 test qrels
中出现的 action 交集为零；训练文本也排除了官方 G1 test query。评测 registry 直接
来自官方 `G1/corpus.tsv`，保留原始 docid 和 qrels，不使用 G123 截断 registry。

## 运行入口

- `scripts/build_api_disjoint_manifest.py`：构建 API-disjoint 1K 训练 manifest。
- `scripts/build_manifest.py`：旧版 qrels-blind manifest 入口，不用于本实验。
- `scripts/train.py`：真实 full-parameter sequence SFT smoke/训练入口。
- `src/latebound_sequence_sft/model.py`：dynamic output row、runtime memory 和 exact
  physical address 接入普通 causal CE。

v13 的 GPU-only ZeRO-3 smoke 在第二个真实 optimizer step 的参数 all-gather 阶段 OOM。
v14 保持标准 causal sequence SFT 和全参数 AdamW，改用 ZeRO-3 参数/优化器 CPU offload，
并将 smoke 限制为 `max_length=128`、`max_document_length=64` 的真实两步训练。CPU offload
只用于降低峰值显存，不改变训练目标或注册契约；v14 通过 checkpoint reload 和速度门禁后，
才能估算正式 ToolBench 训练并创建 DAG。v13 结果不被覆盖或复用。

v14 已通过真实两步、checkpoint 和 reload，但 DeepSpeed 分片后的元数据计数为 0，因而不能
作为完整审计证据。v15 在 ZeRO-3 初始化前记录 full/trainable/frozen 参数量、backbone/compiler
梯度状态和 optimizer 参数组数量；其训练目标与 v14 完全相同。

本版本只用于 4 x A100 五步吞吐 smoke，显式固定 `gpu059`、`gres/gpu:4` 和
`nproc_per_node=4`。它在作业级 include path 中加入 `stdatomic.h`
兼容 shim，修复 CentOS 7/GCC 4.8 上 Triton helper 的编译门禁；不改变模型、loss、
数据或训练目标。正式训练仍须通过速度门禁后另行建立 DAG。

v9 只补齐四卡审计采集：每步在四个 rank 间归约实际 sequence/document token 数、
最大 wall time 和最大显存，并记录 tokens/s、global batch、实际/配置序列长度及
`registration_optimizer_steps=0`。它不改变数据、模型、loss、optimizer 或训练计划。

v19 只测试吞吐配置：沿用 v18 的每卡 micro-batch=2 和 ZeRO-3 通信配置，但关闭
gradient checkpointing，以测量重算开销；ZeRO-3 reduce/prefetch bucket
提升到 50M，启用通信重叠和 pinned CPU offload，并缓存与模型无关的 tokenization。数据、
模型、loss、候选身份、全参数目标和注册契约不变；它不是正式训练。

v11 在 v9 审计通过且正式训练估算低于 24 小时后，提交 1 个 epoch 的完整 10,118
样本全参数 AdamW sequence SFT。训练和 checkpoint/reload 仍使用同一固定 manifest；
本 deployment 不读取 qrels，也不产生评测分数。

v11 修复最后一个不完整 global batch 的 ZeRO-3 通信一致性：所有 rank 保持同一 CE
计算图，zero-weight padding 仅通过权重归一化贡献零梯度，不丢弃真实样本。v10 的
NCCL timeout 失败输出保留为审计证据，不复用其 checkpoint。

The public snapshot keeps the source, Slurm contracts, and tests but omits the
deployment's data manifests and run artifacts. Set `PROJECT_ROOT`, `MANIFEST`,
`EVAL_ROOT`, `MODEL_ROOT`, `NATIVE_ROOT`, `PYTHON_BIN`, and `RUN_ROOT` (as
applicable) to site-local paths before using the submission wrappers.
