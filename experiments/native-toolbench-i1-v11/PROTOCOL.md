# 官方 I1 API-disjoint 标准序列 SFT 协议（v1）

## 实验定义

训练只使用 1,000 个 ToolBench exact API。候选 API 按完整 action identity 分组，
仅选择在 G123 corpus 中恰好只有一个 exact document 的 singleton action；这些 action
必须与官方 G1/I1 `qrels.test.tsv` 中出现的全部 action 不相交。训练构建器不读取 qrels，
但会读取官方 G1 测试 query 文本用于去重审计。官方 G1/I1 的 `test.query.txt`、
`qrels.test.tsv` 和 `corpus.tsv` 只在评分阶段使用。

评测 registry 直接使用官方 G1 `corpus.tsv` 的唯一 `(docid, document)` 行；不能用
G123 截断库替代，也不能按名称、action code 或 family 合并重复文档。每个 I1 文档
在评测时执行一次 document forward 生成动态 output row/memory，optimizer steps=0，
backbone/compiler/静态词表均不得改变。

主结果是 ToolBench 官方兼容的 `NDCG@1/3/5`，使用原始 G1 qrels。`Hit@1/3/5`、
Recall@5、MRR 和 exact identity binding 只作为辅助诊断。所有 qrels 必须在完整盲
分数矩阵、checkpoint reload 和 checksum 审计之后才打开。

## 训练目标

每条样本是 query/context 到一个 physical action token 的标准 causal-LM 序列。只对
assistant action token 和终止 token 计算普通 token cross-entropy；system、query 和
padding label 为 `-100`。dynamic output row 写回对应 physical-ID 的 logits 列，
runtime memory 替换同一 physical-ID 的输入 embedding。普通词表仍在 softmax 分母中；
未激活 reserved rows 在 logits 中屏蔽。

训练采用 Qwen3-8B backbone 与共享 compiler 全参数 AdamW。禁止 LoRA-only、E5、GIST、
codebook、sampled-softmax、multi-positive retrieval loss、32 个负文档和 API 专属参数。

## exact identity

官方 corpus 中每个 `docid` 都是独立 identity。一个 action 对应多个 exact 文档时，
训练样本展开到全部文档，并按文档数逆频率加权；不能选择第一个、最后一个或合并文档。
物理地址按完整 corpus 的 numeric docid 确定性分配，注册时不做 optimizer step。

## 数据隔离

训练 manifest 只读取 ToolGen retrieval JSON、G123 corpus、官方 atomic map 和官方 G1
测试 query 文件；训练 action 与 I1 测试 action 必须交集为零，训练 query 文本与 I1 测试
文本也必须交集为零。构建器不读取 qrels；评分阶段禁止把 qrels 传给 GPU 任务。只有
完整预测、模型 reload、checksum 和独立审计通过后才可计算指标。

## smoke gate

正式训练前必须用真实模型、真实 tokenizer、真实 compiler、真实 backward 和 optimizer
step 跑 2 或 5 步，并记录：每步 wall time（第 2 步以后中位数）、峰值显存、有效 batch、
sequence/document 长度、tokens/s、world size、optimizer step 和 checkpoint reload。
按实际配置估计超过 24 小时就停止，不提交正式 DAG。

## 资源和版本

最多 4 张 L20 或 1 张 A100，以实时调度器为准。v19 的四卡 smoke 使用 ZeRO-3 参数和优化器
CPU offload，每卡 micro-batch=2，并把 reduce/prefetch bucket 提升为 50,000,000 个元素，
启用通信重叠和 pinned host memory；这只改变内存调度，不改变全参数 SFT 的数学目标。v15
还在 ZeRO-3 分片前记录完整参数审计，避免分片张量的 `numel()` 被误记为零。v17 通过 5 个
optimizer step 后，才可据实测中位 step time 估算正式训练；行为修复必须建立新
worktree/deployment，
重新生成 `SOURCE.sha256`；不得修改冻结 `latent-register/` 或已提交 deployment。
