# 实验总账与当前状态

## 2026-09-02 Native STQ staged ablation v6 reload 修复后提交

v5 Stage-L `78564` 已完整完成（4×L20，`L20006`，2:31:39，1,311 optimizer steps），
尾批同步修复有效并生成 checkpoint。其 reload `78565` 在 `L20004` 失败，唯一根因是
reload shell 未建立训练阶段使用的 CUDA_HOME shim，DeepSpeed 导入时触发
`MissingCUDAException: CUDA_HOME does not exist`；checkpoint 本身完整且未读 qrels。

已取消不可达后继 `78566–78567`。v6 deployment
`latebound-stq-native-staged-v6-20260902` 在两个 reload 作业中加入同一 CUDA_HOME shim，
并复用唯一已验证的 v5 Stage-L checkpoint：
`/mnt/home/user46/runs/stq-native-staged-v5-formal-20260902/stage-l-formal/checkpoint`。
新的恢复链为 `78639` preflight -> `78640` Stage-L reload -> `78641` Stage-F 全量 ->
`78642` Stage-F reload，运行根目录为
`/mnt/home/user46/runs/stq-native-staged-v6-recovery-20260902`。v6 已完成远端源码校验并
启动机械 watcher；训练、reload、盲预测和 sealed aggregate 尚未完成，论文表保持不变。

## 2026-09-02 Native STQ staged ablation v5 尾批同步修复后重新提交

v4 正式 Stage-L `78452` 在 1,310 个正常 optimizer steps 后处理最后一个不完整 global
batch 时失败。10,483 条样本按 global batch 8 为 `1,310 x 8 + 3`；padding 后 rank 2/3
只有 zero-weight 样本，旧 `NativeSequenceSFT.forward` 在第二次 backbone forward 前提前
返回，导致不同 rank 进入不同的 ZeRO-3 `_ALLGATHER_BASE` 序列，最终 watchdog 在 600 秒后
报 NCCL timeout。显存峰值约 30 GiB/卡，故不是 OOM；v4 没有可信 checkpoint。

已取消不可达后继 `78453–78455`，没有取消有效祖先。v5 新 deployment
`latebound-stq-native-staged-v5-20260902` 只改这一处：所有 rank 始终执行完整文档和 query
backbone forward，`loss_weights=0` 仅在最终 CE 加权时生效，分母使用 `clamp_min(1.0)`。
本地 13 项测试和远端 `sha256sum -c SOURCE.sha256` 全部通过。新的正式 L20 DAG 为
`78563` preflight -> `78564` Stage-L -> `78565` reload -> `78566` Stage-F -> `78567`
reload，运行根目录为 `/mnt/home/user46/runs/stq-native-staged-v5-formal-20260902`；提交
事件记录在 `ops/events/l20-stq-native-staged-v5-formal-78563-78567-submitted-20260902.json`。
训练、reload、盲预测和 sealed aggregate 尚未完成，论文表保持不变。

## 2026-09-02 Native STQ staged ablation v4 正式链已提交

v3 Stage-F smoke `78424` 在 `L20002` 使用 4xL20 完成，耗时 00:11:26；5 步真实全参数
训练、ZeRO-3 checkpoint 和普通 backbone reload 均通过。终态审计确认 `phase=full`、
`lora_parameter_names_after_merge=0`、`full_backbone_trainable=true`、注册 optimizer
steps=0。训练元数据 SHA-256 为
`0b6cd2304cae3e925513f856e171a7d21f5b456d451b6a42628ab4a511910591`，reload 审计 SHA-256
为 `82798fac403c4e652347967ca92869a5474b1949c09c5ea00bc78cb5f73f1aa9`；qrels 未读取。

依据 smoke 的实际 step time 和 global batch=8，完整 10,483 行预计在 24 小时门槛内完成。
因此建立不可变 v4 deployment，并提交唯一正式 L20 DAG：`78451` preflight -> `78452`
formal Stage-L -> `78453` Stage-L reload -> `78454` formal Stage-F -> `78455` Stage-F
reload。当前提交事件记录在
`ops/events/l20-stq-native-staged-v4-formal-78451-78455-submitted-20260902.json`；训练阶段
各申请 4 张 L20，qrels-blind 预测和评分尚未启动，论文表不变。

## 2026-09-02 Native STQ staged v2/v3 Stage-F smoke 修复

v2 的 Stage-L `78370` 已通过；Stage-F `78399` 完成了 5 步训练并写出 ZeRO-3
checkpoint，但 reload gate 错误地用 LoRA-wrapped 结构加载已经 merge/remove LoRA 的
Stage-F 模型，因 state-dict key 不匹配而失败（无 qrels、无分数）。失败事件记录在
`ops/events/l20-stq-native-staged-v2-78399-failed-reload-20260902.json`。

修复版 v3 `latebound-stq-native-staged-v3-20260902` 已通过本地 13 个测试和远端 source
校验，复用已审计的 v2 Stage-L checkpoint 作为 warmup，仅重新运行 Stage-F。preflight
`78423` 和 Stage-F smoke `78424` 均已完成；该版本的 full reload 直接加载普通 merged
backbone 并通过审计。正式训练已转移到 v4，论文表仍未修改。

## 2026-09-02 Native STQ staged ablation v2（Stage-L smoke 运行中）

这是 Native late-bound 的受控训练 schedule 对比，不是官方 ToolScalER 复现。固定输入为
999 个 seen exact tools、10,483 条训练行、837 个 unseen candidates、1,066 条测试 query，
seen/unseen exact identity overlap=0；manifest SHA-256 为
`caaaf5de6fc6e3b60fb39e782599b217dfd994bb1ca9fa5365d1a6dcd6fbbb59`。

部署 `latebound-stq-native-staged-v2-20260902` 从已验证 Native v19 sequence-SFT 路径
版本化复制而来，补上了与 STQ schema 兼容的 manifest、4 卡 ZeRO-3 入口和阶段 reload 审计。
Stage L 只训练共享 Native compiler 和 Qwen3-8B 的 `q_proj/k_proj/v_proj/o_proj` LoRA；
Stage F 读取 Stage-L checkpoint，先 merge/remove LoRA，再解冻完整 backbone，继续普通
causal sequence SFT。没有 E5、GIST、codebook、sampled-softmax、固定负文档或 per-API
参数；unseen 注册仍是一文档一次 forward、optimizer steps=0。

本地 13 个测试、compileall、Slurm shell 语法和 SOURCE 校验全部通过。远端 source/artifact
校验通过；preflight `78369` 在 `L20001` 完成，Stage-L 5-step 真实 GPU smoke `78370` 在
`L20007` 运行（4 GPUs，依赖 `afterok:78369`）。Stage-F、正式训练、blind prediction 和
sealed aggregate 均未提交；论文表不变。一次性 1,800 秒终态 watcher 为
`ops/watch_stq_native_staged_v2_smoke.sh`，只检查终态/checkpoint/reload，不读取分数。

## 2026-09-01 Native Late-Bound I1 两阶段 v7（当前主链）

旧 v3/v4 单阶段 sequence SFT 被代码审计否定：每个样本只把自己的正确动态行 scatter 回
softmax，其他 999 个训练工具不在分母中，因此不能学习完整 registry 内的工具区分。
v5 只补了 Slurm 链，未修复该 loss，未提交。

v7 分为两个训练边界。Stage 1 冻结原始 Qwen3-8B，四卡并行预计算文档/query hidden
states，只训练共享 `C_out/C_mem`，每个样本均使用完整 1,000-way output/memory CE。
Stage 2 固定 compiler、动态 output rows 和 runtime memories，全参数训练 Qwen；普通词表与
全部 1,000 个动态工具行共同归一化，每个样本有 999 个真实动态负行，不做负例采样。
测试时原始冻结 Qwen 与 Stage-1 compiler 对 10,439 个未见文档执行一次 zero-step 注册，
Stage-2 Qwen 编码 query，最后才由 sealed aggregate 计算官方 NDCG。

本地与远端均为 `15 passed`，Python 编译、shell 语法、源码 checksum 和 sbatch dry-run
通过；远端 Transformers 4.56 tokenizer 实际加载也通过。源码清单 SHA-256 为
`e94a989cf31004a99b65bd110655dedec28469ffd6a491d7ab62a55f66327738`。正式 DAG：
`77595 -> 77596 -> 77597 -> 77598 -> 77599 -> 77600 -> 77601 -> 77602`；preflight
`77595` 已完成；ToolScalER 单卡任务 `77568` 已取消，Stage 1 `77596` 用四张 L20 完成
317 步。原 reload `77597` 漏传冻结架构路径，修复作业 `77607` 已通过；Stage-2 gate
又依次暴露 CUDA_HOME 与 BF16 buffer dtype 两个训练前问题，均建立新不可变修复包。
最终五步 gate `77621` 完成真实训练、ZeRO-3 checkpoint 和全模型 reload，热身后约
9 秒/step、峰值约 38.42 GiB/卡。正式训练 `77622` 已在 `L20004` 四卡运行，后续
`77623 -> 77624 -> 77625` 保持 `afterok`。v6 的 preflight 完成后发现旧版
tokenizer 风险，训练尚未启动，故其 `77588–77594` 已取消；v7 不改变科学定义。完整 sealed
audit 前没有结果，不得更新论文表。

## 2026-08-31 ToolBench I1 API-disjoint 对比审计状态

当前共享输入已固定为 1,000 个训练 exact API、10,118 条训练样本、官方 G1 候选库
10,439 个文档、官方 I1 query 1,335 行和 1,008 个 qrel 相关文档；训练/API 与 I1
action 交集为 0。Native Late-Bound v3 的 preflight `76909` 在 `L20007`、两步 smoke
`76910` 在 `L20002` 均以 `COMPLETED 0:0` 结束，真实全参数训练、checkpoint/reload
和 zero-step 注册审计通过；该制品仍仅为 smoke，不产生 NDCG。

本轮尚无可纳入比较表的 CoTools 或 ToolScalER 官方 I1 结果。现有 ToolScalER Qwen3
制品是闭集、source-derived 诊断（审计明确 `formal_evidence_eligible=false`），现有
CoTools source-only 入口和 checkpoint 仍面向 SimpleToolQuestions，缺少 API-disjoint
ToolBench I1 适配、盲预测矩阵和官方 I1 聚合。详细状态与哈希见
`paper-evidence/diagnostics/toolbench-i1-api-disjoint-comparison-status-20260831.json`；
论文表未修改。

本文档是精简后的叙事实验总账。精确哈希和论文准入资格仍以
[`PAPER_TABLES_WORKING_2026-08-09.md`](../PAPER_TABLES_WORKING_2026-08-09.md)
及各证据包内部声明为准。

## 2026-08-31 Native Late-Bound 官方 I1 API-disjoint v3（当前 smoke）

本实验在 1,000 个与官方 ToolBench G1/I1 测试 action 完全不相交的 singleton exact API
上做全参数标准 causal sequence SFT；评测 registry 直接使用官方 G1 `corpus.tsv` 的
10,439 个唯一文档，测试使用官方 `test.query.txt`（1,335 行）和 `qrels.test.tsv`
（1,008 个相关文档）。测试 API 只执行一次 document forward 生成动态 bundle，注册时
optimizer steps=0、参数改变数=0。主指标为官方兼容 NDCG@1/3/5，exact-API Hit 仅作
辅助诊断。

训练清单已通过本地和远端 preflight：训练 API=1,000、训练 exact 文档=1,000、训练样本
=10,118，训练/I1 action 交集=0；训练文本还排除了 600 个官方 I1 测试 query 文本。
版本化 deployment 为 `latebound-toolbench-i1-unseen-api-v3-20260831`，源码清单哈希
为 `7f6e6e8ee25798f085186b73d678aeab74207dacdb256430dedbb0a1c6e606d5`。

v1（`76902`）因 manifest 缺少 `exact_identity_count` 在模型加载前失败；v2（`76908`）
因 smoke 脚本未设置远端 `CUDA_HOME`，在 DeepSpeed 导入前失败。两次均没有 optimizer
step、checkpoint 或分数，保留作工程失败审计。修复版 v3 依赖链为 `76909 -> 76910`
（`afterok`），两项均已以 `COMPLETED 0:0` 结束；终态前没有读取 qrels 或任何部分分数，
也没有更新论文表。v3 仍仅是 2-step smoke，不是正式 NDCG 运行。

## 2026-08-31 Native Late-Bound 标准序列 SFT 显存门禁

旧的 GPU-only ZeRO-3 smoke `76457` 在四张 L20 的第二个真实 backward 参数
all-gather 阶段 OOM；它只完成 1/2 个 optimizer step，没有 checkpoint、reload 或分数。
该失败保存在 `ops/events/l20-latebound-sequence-sft-v13-76457-failed.json`。

修复版 `v14`（作业 `76569`）启用 ZeRO-3 参数/优化器 CPU offload 和 2,000,000 元素
通信 bucket，真实两步均完成，峰值显存约 17.92 GiB，单步约 48.7/41.3 秒，并成功
reload；但 DeepSpeed 分片后训练元数据把 full/trainable 参数量记成 0，因此只作为
审计缺陷记录，事件为 `l20-latebound-sequence-sft-v14-76569-audit-invalid.json`。

修复版 `v15`（作业 `76664`，部署源码哈希
`39915df78f5d7c625c3d710d351287df9d23fc059cdf4ea20a785bbf35f9425e`）在 ZeRO-3 初始化
前记录完整参数审计。四张 L20 节点 `L20007` 上两步真实全参数 SFT 均完成：
`8,193,635,337` 个参数全部可训练、冻结参数为 0、backbone/compiler 均保持梯度、
optimizer 参数组为 1；峰值显存 `17.9232/17.9222 GiB`，rank-0 step 用时
`50.6735/43.7698 s`，完整 checkpoint 和独立 reload 均通过。训练 manifest 为
`d17c3314f6c4c1e29b37657fb2a9836534a9a9d2f8ddb370446a90356ed75d68`，证据事件为
`ops/events/l20-latebound-sequence-sft-v15-76664-completed.json`。

该结果只证明当前全参数标准 sequence-SFT 路径在 4x L20 上可行，尚未产生 ToolBench
分数，也不更新论文表。按 2-step 后续规则，下一门禁应是同一源码和输入的 5-step
速度/稳定性 smoke；只有 5-step 中位速度能让正式训练低于 24 小时，才建立正式 DAG。

`v16` 的 5-step smoke（作业 `76706`）已完成并通过连续训练、checkpoint、reload 和
完整参数审计。rank-0 步时为 `52.7543, 41.1267, 44.0824, 41.0575, 41.9836 s`，
排除首步后的中位数为 `41.5552 s`，峰值显存约 `17.92 GiB`。按
`ceil(513,895 / 4) * 41.5552 / 3600` 计算，在当前 global batch=4 下一个 epoch
约需 **1,483.0 小时**（约 61.8 天）；原记录的 710.5 小时是估算公式错误，已更正。
因此正式速度门禁失败；不能提交正式
ToolBench DAG。下一项只允许测试较大 ZeRO-3 offload bucket/max-live 参数的 throughput
版本，确认通信开销是否为主要瓶颈，仍须保持真实全参数 SFT 和同一输入。

`v17` 吞吐 smoke 已提交为 L20 作业 `76783`。它保持 v16 的数据、模型、loss、全参数
目标和 5 步审计，只将每卡 micro-batch 提升到 2，ZeRO-3 reduce/prefetch bucket 提升到
50M，启用通信重叠、pinned CPU offload，并缓存 tokenization。该作业只用于测吞吐和显存，
不产生论文结果；终态前不读取任何 score/qrels。作业虽然通过了 checkpoint/reload，
但提交时误用了 ToolScalER 初始化模型（词表 153,717、参数 8,208,225,289），而 v16
基准使用 `/mnt/home/user46/windy/models/Qwen3-8B`（词表 151,936、参数 8,193,635,337），
故 v17 吞吐结果不具备与 v16 的严格比较资格，已记录为审计无效，必须用正确模型重跑。

修正版 `v18` 已提交为 L20 作业 `76804`，使用与 v16 完全相同的
`/mnt/home/user46/windy/models/Qwen3-8B`，保留 v17 的吞吐配置和 tokenization cache。
它是唯一可用于和 v16 配对比较的速度 smoke；终态后通过 checkpoint/reload 和参数审计，
但热身后中位 step 为 `49.7001 s`，global batch=8 下估算一个 epoch 约 `886.83 小时`
（36.95 天），仍未通过 24 小时门禁。证据收集目录为
`paper-evidence/l20/latebound-sequence-sft-v18-throughput-76804/`，不更新论文表。

## 最近主链：Native Late-Bound ToolBench 全参数 v8 xformers5-r2（已失败）

这是最近一次门禁链，已完成终态核对但没有产生可用模型。部署目录为
`/mnt/home/user46/external/latebound-toolbench-fullsft-v8-xformers5-r2-20260831`，
源码清单 SHA-256 为
`2aa8f3c646fa51397e934b8167b72a48b3598130c47eb177db289cd4f853d9fe`。
链条为 `75990 -> 75991 -> 75992 -> 75993 -> 75994 -> 75995 -> 75996 -> 75997_[0-5%4]`，
全部使用 `afterok`；`75990–75992` 为 `COMPLETED 0:0`，`75993` 为 `FAILED 1:0`，
`75994–75997` 因依赖失败取消。`75993` 的根因是 ZeRO-3/NCCL `_ALLGATHER_BASE`
超时，不是 OOM；没有 optimizer step、checkpoint、`COMPLETE` 或分数。
该链不具备论文证据资格，且其旧训练目标不符合当前标准序列 SFT 决定。

上一条 tokenizerfix1 链的 75605 在模型加载时失败：远端没有 `flash_attn` Python 包，
但脚本强制选择 `flash_attention_2`，因此没有产生 compiler snapshot 或 checkpoint；
75606--75610 因 `DependencyNeverSatisfied` 保持阻塞并已停止监督。新链不覆盖该部署或输出。
当前链在检测不到可选 `flash_attn` 时使用远端已安装的 xFormers memory-efficient attention，
并在训练审计中写入 `attention_backend=xformers_memory_efficient`；xFormers 不可用时
 warm-up 会直接失败，不会降级为慢速 eager attention。xformers1 的 warm-up 虽然完成了
前向，但在 ZeRO-3 保存阶段触发 L20 NCCL all-gather watchdog；xformers2 已将该阶段改为
单卡非分布式，避免这条与科学目标无关的通信路径。

更早的 gatefix3 链的 `75510` 在 tokenizer 初始化阶段失败：远端 Transformers 4.56.1
将模型 `tokenizer_config.json` 中的列表型 `extra_special_tokens` 当作字典访问，导致
`AttributeError`；没有产生 checkpoint。其不可达后继已取消，失败日志和调度器终态保留。
修复版只在加载时把同一列表显式传为 `additional_special_tokens`，并用空字典覆盖不兼容
字段；远端独立 tokenizer smoke 已通过，token ID 未改变。

新链复用同一份已核验 manifest：489,570 条训练记录、49,936 个 exact corpus identities、
10,969 个多正例行，理论全局步数为 638。训练配置为 4x L20、ZeRO-3、BF16、
fused Flash-SDP、gradient checkpointing、每卡 batch 8、累积 24（global batch 768），
Qwen backbone 和 Native compiler 均为全参数 AdamW，无 LoRA、无 member bank。
该链原计划用 500-step throughput gate 外推 638 步，但在正式 gate 前即失败；后继没有执行。
评测器虽设计为先完成 qrels-blind 排名 staging，再读取 qrels，但本链没有预测制品。
当前尚未产生正式分数，论文表保持不变。旧的 xformers1--xformers5、v9、v10、v11
部署快照已不再作为当前入口；提交事件仍保留在 `ops/events/` 供审计追溯。

## 当前实验：ToolScalER Qwen3-8B 千 API 全参数训练（预测恢复已完成）

## 并行实验：Native Late-Bound 官方 I1/I2/I3

目的：在 ToolGen retrieval SFT、官方 G1/G2/G3 六个 split 和完整 G123 候选库下，
重跑 Native Late-Bound 的检索兼容性。该轨道与严格未见 API 证据分开，不读取运行中的
部分分数，也不在终态审计前写入论文表。

依赖链：`73768 -> 73769 -> 73770 -> 73771 -> 73772 -> 73773`（全部 `afterok`）。
`73768` 预检已 `COMPLETED 0:0`，`73769` 数据准备运行中；其余任务等待依赖。
输入和 deployment checksum、4 个本地测试及远端源码清单均已通过。训练默认 seed=17、
4 张 L20、1 epoch；评分为 1,099 个 query 对 49,936 个 G123 文档的盲分数矩阵，聚合
同时计算官方兼容/修正 NDCG 和 exact-API Hit@1/3/5/MRR。

同一 deployment 已迁移到 A100 的独立路径。账户 `GrpTRES` 实际只有 1 张 GPU；原 4 卡
后继和首个缺少 `jq` 的 smoke 链均已保留失败痕迹。修复后的单卡替代链为
`1706581 -> 1706582 -> 1706583 -> 1706584 -> 1706585 -> 1706586`，训练有效梯度
累积为 32，后续仍严格使用 `afterok`，且不与 L20 链共享输出。

目的：在固定的 1,000 个 ToolBench API 上，诊断基于 ToolScalER 官方源码适配的
Qwen3-8B 全参数训练和 exact-API 检索。它不是官方 checkpoint 复现，不是严格未见工具
证据，也不能填写论文表格。

| 项目 | 数值/说明 |
| --- | --- |
| 模型 | Qwen3-8B，8,205,325,312 个参数全部可训练 |
| Registry | 1,000 个 API；996 个唯一 code，另有 4 个双 API 碰撞桶 |
| 数据 | 7,963 行：7,166 训练 / 797 验证 |
| 测试 | 1,951 个 query-text-disjoint 查询，覆盖 240 个目标 API |
| 训练 | 5 个 epoch，最大长度 1,024，全局 batch 32，4 张 L20 |
| 预测 | 4 个不接触 qrels 的分片 |
| 指标 | code Hit@k，以及对碰撞失败关闭的 exact-API Hit@k/MRR |
| qrels 规则 | 只有所有预测通过后，终态审计才能打开 |

当前不可变 v18 依赖链：

```text
73515 -> 73516 -> 73517 -> 73518 -> 73519_[0-3%4] -> 73520
```

完整训练 `73518` 已完成 `1,120/1,120`；其原预测链因环境依赖问题失败，未被复用。
修复后的不可变恢复链为 `74187 -> 74188_[0-3] -> 74189`，全部
`COMPLETED 0:0`。`74187` 校验了训练模型、Transformers 5.2.0、scikit-learn
1.7.2、FastChat 和官方 evaluator；`74188_[0-3]` 四个 qrel-blind shard
全部通过 checksum，`74189` 在四个 shard 完整后才打开 qrels 并通过终态审计。
远端输出已收集到
`paper-evidence/diagnostics/toolscaler-qwen3-1k-retrieval-v18-predict-recovery-v4/`，
本地收集哈希为 `fb223d109bdc0e39aff5915b3ca835e5422be91bba869659ed4889b364764021`。
该运行仍是 source-derived 1K 闭集诊断，`formal_evidence_eligible=false`，不更新论文表。

### 精简恢复历史

| 版本 | 终态事实 | 结论/修复 |
| --- | --- | --- |
| v1-v6 | 数据路径、环境和模型审计工程失败 | 修复上游工作目录、Python 边界和 tensor 输入审计，未打开 qrels |
| v7-v9 | 缺少 launcher，随后分布式数据/cache 阻塞 | 加入 Python module `torchrun` shim，将相同 tokenization 移到单 CPU 阶段 |
| v10-v11 | ZeRO-3 NCCL watchdog，随后确认 `zero.init()` 真死锁 | 关闭 heartbeat 后仍无进展，证明不是误报 |
| v12 | FSDP 进入训练，但 68 分钟无一步 | 默认 collective 路径阻塞 |
| v13-v14 | ZeRO-2 到达 optimizer 构造；随后补齐 Ninja PATH | 仅修环境，不改方法 |
| v15-v16 | 629M 元素 broadcast 超时；禁用 P2P 后越过，但首个 GEMM 失败 | 定位 L20 collective/topology 不兼容 |
| v17 | FSDP/P2P smoke 通过；完整训练第 1 步后在第 2 次 backward OOM | 每卡约占 42.92/44.53 GiB，仍需 2.35 GiB |
| v18 | CPU 参数/梯度 offload 与可扩展 allocator | 训练完成；原预测环境依赖失败后由 v4 恢复链完成四 shard 和终态审计 |

原 `73519/73520` 失败链和旧 watcher 已不再承担监督职责。`74189` 的完整摘要如下：

| 指标 | 数值 |
| --- | ---: |
| 候选 API | 1,000 |
| 查询 | 1,951（四个 shard，query-text-disjoint） |
| code Hit@1 / @3 / @5 | 91.645% / 95.079% / 95.797% |
| fail-closed exact-API Hit@1 / @3 / @5 | 91.235% / 94.669% / 95.387% |
| MRR@5（fail-closed） | 92.997% |
| 目标身份 / 唯一 code | 240 / 995 |
| 完整审计 | `passed=true`，所有 checksum 和 `COMPLETE` 通过 |

该结果是已训练候选上的闭集检索分数，不能解释为未见工具零步注册能力，也不能替代官方
ToolBench 或论文 Table 1/2 证据。

## 可用于论文的已验证结果

### 注册成本与规模：可用

| 方法 | 注册耗时 | 优化器步 | 改变标量数 | 每工具存储 | @1K / 10K / 47K 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ToolGen-Fixed | 51,895 工具共 88,113.136 s | 892 | 7,203,140,330 | 332,759 B | 37.323 / 18.650 / 18.584 ms |
| Incremental ToolGen | 1,315 工具共 1,282.000 s | 16 | 5,563,348,715 | 13,148,928 B | 40.119 / 19.240 / 19.301 ms |
| Native Late-Bound | 47K 工具共 2,823.849 s | 0 | 0 | 147,456 B | 34.964 / 36.744 / 53.762 ms |

100 到 1K 顺序追加审计证明所有旧 identity、logical slot 和 physical address 不变。
以上是已审计的 Table 5 数值，不是准确率。

### 受控 BFCL 基线：仅限声明的子集

Qwen3-8B-FC 在受控 8 类 AST 子集上为 1,237/1,800 = 68.72%：Simple 95.50、
Multiple 96.00、Parallel 92.00、Parallel Multiple 88.00、Multi-turn 37.88。
该 Overall 不是公共 BFCL leaderboard 的总体样本。

### ToolGen 官方检索复现

官方发布 ToolGen checkpoint 的聚合已经完整且校验和通过，同时保留论文兼容 NDCG 和
修正指标。它属于独立的 Llama/官方轨道，不是 Native 同 backbone 证明。

## 主要选择诊断

2026-08-23 的 source-only 重置后，下列 clean-room 数值仍可用于方法开发，但不能作为
官方对照行。

### STQ 匹配 837-way 面板

数据：999 个已见 API、10,483 个训练查询、1,707 个 dev 查询、837 个 exact 未见 API、
1,066 个测试查询。

| 方法 | Hit@1 | Hit@3 | Hit@5 | MRR | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| Native Late-Bound | 28.99 | 53.03 | 63.45 | 44.06 | 三 seed clean-room 诊断 |
| CoTools Qwen 适配 | 22.64 | 42.59 | 52.47 | 36.83 | clean-room，非官方 checkpoint |
| ToolScalER 严格适配 | 6.88 | 14.60 | 19.36 | 13.50 | clean-room，非官方 checkpoint |

### Native 容量和规模诊断

| 实验 | 候选/查询规模 | Hit@3 | 解释 |
| --- | --- | ---: | --- |
| Native-confuser / sibling-live 严格面板 | 每 panel 448-487 API，共 13,443 个保留查询 | 82.24 / 82.26 | 受控规模正面诊断；不是 STQ |
| H00 full-tap | 200 个目标 API，340 个查询 | 92.06 | 强的小规模同域控制 |
| Native 与 H10 匹配 | 500 候选，2,438 查询 | 74.90 与 76.87 | H10 高 1.97 pp；Native fold SD 为 2.18 |
| Native / H10 @1K | 1,000 候选，2,438 查询 | 66.69 / 66.82 | 随规模下降 |
| Native / H10 @2K | 2,000 候选，2,438 查询 | 56.74 / 56.56 | 规模下降继续 |
| 500-group hub 探针 | 500 组 | 2.75 | Top-1 只有 3 个唯一值，严重全局 hub collapse |

82% 面板不能直接与 STQ 53% 比较：候选规模、confuser/sibling 训练、查询过滤和 panel
边界都不同。

### Backbone 与目标函数实验

| 分支 | 结果 | 决定 |
| --- | --- | --- |
| 从已审计 Native checkpoint warm-start 全参数训练 | 1,066 x 837 上 Hit@3 52.06 -> 52.81（+0.75 pp） | 小幅诊断提升，非正式 |
| 冷启动/欠训练全参数训练 | Hit@3 保持 1.18%；Top-1 hub 升至 94.71% | 不能据此判断容量；匹配训练不足 |
| Residual-anchor | Hit@3 52.06 -> 34.15（-17.92 pp） | 已审计负面结果 |
| MFLI facet 聚合 | +0.014 pp；facet cosine 0.999994 | 表示坍缩，停止 |
| IMOB moments | 约 +0.55 pp，CI 跨零 | 不确定 |
| RDPO/DPO replay | 约 +0.216 pp，CI 跨零 | 不晋级 |
| EFCG | 更新后分数下降 | 否决 |
| CCAG | Hit@3 影响严格为零 | 空结果 |
| FRDI-G0 | base/full 均为 89.7135 | 该 panel 下为空结果 |
| CoTools/DPO/GRPO/ToolScalER 风格 Native preference arms | compiler-only preference/listwise 目标 | 仅诊断，不进入论文 |

## 下游与外部诊断

| 轨道 | 结果 | 边界 |
| --- | --- | --- |
| BFCL 外部 400-tool registry stress | registered Hit@1 22.50%、Recall@5 47.50%；控制近零 | 旧 checkpoint 的强机制诊断；不是官方 BFCL |
| BFCL Late-Bound v17 | 908/1,800 = 50.44%；多轮结构性失败 | top-k/单步 selector 没学会有状态下一工具策略 |
| BFCL Incremental 旧运行 | 117/1,800 = 6.50% | caller/envelope 缺陷，不能作为性能证据 |
| Tau2 v3/v4 | retail 失败导致 array 不完整 | 禁止聚合，仅探索 |
| MCPEvol/ContDa | 已完成资产/schema 和动态 registry 探针 | 探索性，不能填写 STQ/ToolBench |
| StableToolBench | 未运行 | 用户暂停；禁止调用付费 simulator/judge |

## 对照源码状态

- CoTools 公共源码固定在提交 `bd67fd8743082639a8b81dff724504bdcb3d0527`。
  仓库含 STQ 训练代码，但未找到匹配的公共 STQ evaluator 或官方 checkpoint；匹配分数
  必须标为迁移。
- ToolScalER 公共源码固定在提交
  `a72d46d3352358f39a9274aeb8bf8217b164bb01`。当前千 API Qwen 运行未修改
  上游 codebook/E2E 核心，但属于 source-derived port，不是论文的 Llama-3 条件。
- Native 正式 launcher 从冻结 `latent-register/` 外部调用源码，禁止修改冻结文件。

## 论文表状态与下一步

| 表格 | 状态 | 下一项有效工作 |
| --- | --- | --- |
| 1A STQ | 正式对照行需要 source-only 重置 | 完成固定源码 CoTools/ToolScalER/Native，或明确报告不可用 |
| 1B ToolBench | 需要重置 | 分开源码原生条件与匹配迁移，统一 exact-API 指标 |
| 2 严格身份 | 阻塞 | 构建无碰撞 exact identity 清单；禁止合并或删 qrels |
| 3 BFCL | 仅基线，可选 | 只在 selector/caller 对齐和官方评测器通过时运行 |
| 4 StableToolBench | 暂停 | 需要用户明确恢复并授权付费 API |
| 5 成本/规模 | 已审计可用 | 保留证据，不推断准确率 |
| 6 消融 | 不完整，可选 | 主要 gate 通过后只做预注册严格条件 |

当前首要任务是完成并审计 v18，再依据其 source-derived 千 API exact-API 结果决定是否
值得扩展 ToolScalER 源码实验。终态审计、checksum、`COMPLETE`、完整重载和逐样本重聚合
全部通过前，任何分数都不可信。

## 2026-08-29 Native Late-Bound ToolBench SFT 改进候选

用户澄清本轮目标是改进我们自己的 Late-Bound，不是复现 ToolScalER。新工作树为
`worktrees/active/latebound-toolbench-sft-v1/`，不改动冻结 `latent-register/`，也未提交
集群任务。保留四视图 compiler、dynamic output row、optional memory 和 zero-step 注册合同。

新代码参考 ToolGen 的 atomic retrieval 数据组织：读取 `toolgen_atomic_retrieval_G123.json`，
过滤全部 1,099 条官方 G1/G2/G3 测试 query，再以 corpus `docid` 保留 exact API 身份。
这不是 ToolBench 原生的 `G1/G2/G3/{train,test}.query.txt` + `qrels.{train,test}.tsv`
训练划分，而是对 G123 派生记录的 query-disjoint 重组；因此只能称为
ToolGen-derived 训练集。
同名 action 只产生多正例标签，不合并候选或物理地址。训练目标改为完整 exact-identity
registry 分母的 multi-positive SFT，以检验旧的局部 live-set 分母是否造成规模 hub collapse。

本地真实数据审计：corpus 49,936 文档；无泄漏训练行 489,570；多正例行 10,969；5K pilot
解析通过。分支测试为 `43 passed, 1 skipped`，`SOURCE.sha256` 为
`3a985c4a09b216f77f14fb2f5fdae275ac22361c6be755e1ab69c18e0f375fdb`。尚未生成 Qwen 特征
缓存、GPU checkpoint 或 Hit@3，不能填写论文表。

## 2026-08-29 Native ToolBench Full-SFT v1（待提交）

本版本是对上面 `latebound-toolbench-sft-v1` 的独立修订，不覆盖旧输出，也不修改冻结
`latent-register/`。目标是把 Native Late-Bound 的训练制度改成 ToolScalER retrieval-SFT
所采用的全参数制度，同时保留 Native 的连续四视图 compiler、dynamic output row 和
zero-step 新工具注册能力。

| 项目 | 新版本约定 |
| --- | --- |
| 训练数据 | ToolGen-derived `toolgen_atomic_retrieval_G123.json`；过滤全部 1,099 个官方测试 query；不是 ToolBench 原生 qrels train split |
| 候选身份 | `corpus_G123.tsv` 的 49,936 个独立 `docid`；同 action 只做多正例，不合并文档 |
| 可训练参数 | Qwen3-8B 全部 backbone + 共享 compiler；无 LoRA、无 GIST token、无 per-tool 参数 |
| 训练目标 | query/document 前向后，对 batch 正例和确定性负例做 multi-positive retrieval SFT |
| 官方测评 | 完整 corpus 打分；G1/G2/G3 的 6 个可用 query/qrels split |
| 指标 | released top-100 兼容 NDCG@1/3/5、corrected NDCG@1/3/5、Hit@1/5、MRR |
| 注册合同 | 每个新文档一次 compiler forward，0 optimizer step，0 参数/词表/表格 mutation |

历史工作树 v1（本地已清理；提交事件仍保留）已通过：官方 manifest/index
小样本构建（49,936 候选、1,099 排除 query）、qrels/query ID 审计、`46 passed, 1 skipped`
单测、Python 编译、Slurm 语法和 `SOURCE.sha256` 自校验。独立 `fullsft_audit` 会在训练
结束后重新加载全部 Qwen tensor 和 compiler；reload、`COMPLETE`、逐样本预测和独立 NDCG
重算通过前，结果不得写入论文表。

该版本的训练循环已接入单节点四卡 DeepSpeed ZeRO-3，并记录 world-size 与配置哈希；单卡
路径只保留作 smoke，不能替代四卡正式结果。`slurm/submit_fullsft.sh` 保留 manifest →
full-SFT → reload audit → 6-way official evaluation 的 `afterok` 链，远端 Python、模型
路径和输入均已确认。

原单卡试提交 `74602 -> 74603 -> 74604 -> 74605` 在 AdamW 首步 OOM，已隔离且未产生论文
结果；失败日志显示单卡仅剩约 94 MiB。四卡 ZeRO-3 替代链为
`74619 -> 74620 -> 74621 -> 74622_[0-5%4]`，其中 `74619` manifest 正在运行，后续
训练、reload 和六个官方 split 全部使用 `afterok`；任何失败都会阻止后继阶段，不会自动
填论文表。

`74620` 后因计算节点未设置 `CUDA_HOME`，DeepSpeed 在第一步前导入失败；该环境错误已
隔离。随后四卡链 `74630 -> 74631 -> 74632 -> 74633 -> 74634_[0-5%4]` 在训练初始化时
再次失败：`ds_z3_config.json` 的三个 batch 字段写成字符串 `"auto"`，DeepSpeed 直接比较
字符串与整数并以 `TypeError` 退出（`74632 FAILED 1:0`），没有 checkpoint 或 optimizer
step。该失败链及依赖保持原样，不产生论文结果。

已将 batch 配置固定为四卡/每卡 batch=1/梯度累积=1 的整数 `4/1/1`，launcher 拒绝未同步
的 batch 覆盖，并新增配置回归测试（本地 `47 passed, 1 skipped`）。修复后的替代链
`74645 -> 74646 -> 74647 -> 74648 -> 74649_[0-5%4]` 中，`74647` 又在未设置 P2P
禁用时发生 622,329,856 元素 NCCL broadcast 超时；进程挂起后已明确取消
（`CANCELLED+`），没有 checkpoint 或 optimizer step。该链及其依赖保持原样，不产生论文
结果。

既有 L20 collective 探针已证明 `NCCL_P2P_DISABLE=1` 能通过同规模 BF16 broadcast，
因此替代链 `74667 -> 74668 -> 74669 -> 74670 -> 74671_[0-5%4]` 强制使用该环境，
但 `74669` 在首个 query 前向暴露了第二个错误：ZeRO-3 对通过普通方法直接访问的
compiler learned-query 参数没有自动 gather，部分 rank 看到长度为 0 的参数并触发
einsum shape 错误；无 checkpoint 或 optimizer step。该链及其依赖保持原样，不产生论文
结果。

修复在 `RoleAnchoredRetriever.compile_queries/compile_members` 中使用
`GatheredParameters(..., fwd_module=self)`，并继续记录 P2P 环境；本地测试为
`49 passed, 1 skipped`。新的替代链为 `74699 -> 74700 -> 74701 -> 74702 ->
74703_[0-5%4]`，仍是 manifest → preflight → 四卡 ZeRO-3 full-SFT → reload → 六个官方
NDCG split，全部 `afterok`。watcher 只记录调度状态、退出码和 `COMPLETE`，不读取运行中
的预测分数。

### 2026-08-30 Native ToolBench Full-SFT v2/v3（L20 四卡恢复链）

四卡链 `74934 -> 74935 -> 74936 -> 74937 -> 74938 -> 74939_[0-5%4]` 在 `74936` 的
AdamW 状态初始化阶段 OOM：每张 L20 约剩 5 GiB，但 GPU optimizer state 需约 7.05 GiB，
没有 checkpoint、`COMPLETE` 或可用分数。v2 将 ZeRO-2 optimizer state 改为 CPU offload，
随后因 DeepSpeed 默认拒绝外部 AdamW 而在初始化阶段失败；两条失败链及其依赖均保留。

v3 只增加 `zero_force_ds_cpu_optimizer=false`，继续使用普通 AdamW 与 CPU offload，不改
Qwen/共享 compiler 的全参数训练、数据、损失、候选身份或 zero-step 注册合同。远端源码
checksum 和测试均通过（`50 passed, 1 skipped`）。最新不可变链为
`75149 -> 75150 -> 75151 -> 75152 -> 75153 -> 75154_[0-5%4]`，其中 `75151/75152`
明确申请 `gres:gpu:4`。watcher PID `721008` 只记录调度器、退出码和完成标记，终态审计前
不读取分数，也不更新论文表。

### 2026-08-30 Native ToolBench Full-SFT v2（L20 四卡恢复链）

上一条四卡 ZeRO-2 链 `74934 -> 74935 -> 74936 -> 74937 -> 74938 -> 74939_[0-5%4]`
在 `74936` smoke 的 AdamW 状态初始化阶段失败。四张 L20 均已分配，但每卡只剩约 5 GiB，
GPU optimizer state 还需约 7.05 GiB；没有 checkpoint、`COMPLETE` 或可用分数。失败日志和
依赖保持不变。

历史修复版本 v2（本地已清理）仅在 ZeRO-2 中加入
`offload_optimizer.device=cpu`（不使用 pin memory）；Qwen backbone 与共享 compiler
仍为全参数 AdamW，四卡、数据、损失、评测和 zero-step 注册合同不变。远端源码校验和与
测试均通过（`50 passed, 1 skipped`）。

新的不可变链为 `75106 -> 75107 -> 75108 -> 75109 -> 75110 -> 75111_[0-5%4]`：
manifest、预检、四卡 smoke、四卡正式训练、reload 审计和六路官方 NDCG，全部使用
`afterok`；`75108/75109` 明确申请 `gres:gpu:4`。watcher PID `364942` 只记录调度器、
退出码和完成标记，完整终态审计前不读取分数，也不更新论文表。

v4 将训练作业内存提高到 `256G` 后，`75157` 已通过 optimizer 初始化，但因
`engine.get_global_grad_norm()` 在 CPU offload 模式返回 `None`，记录校验异常退出；
没有执行 optimizer step。v5 仅对该返回值做显式兼容（保留 loss 有限性检查，并让
`engine.step()` 执行），远端测试为 `50 passed, 1 skipped`。新链为
`75161 -> 75162 -> 75163 -> 75164 -> 75165 -> 75166_[0-5%4]`，四卡训练内存为
`256G`，watcher PID `1488791` 已启动。

### 2026-08-30 上游训练速度审计与 Native v6（历史）

本节保留因果结论和远端审计摘要；其中 v1--v7 工作树已从本地移除，不能再从这些路径
派生代码。对应提交事件、失败日志和哈希仍在 `ops/events/` 及证据目录中。

对照源码已固定在 `baselines/ToolGen/` 和 `external/Toolscaler-a72d46d/`。ToolGen
retrieval 官方脚本使用 8 GPU、每卡 batch 2、梯度累积 64、global batch 1024、
FlashAttention、关闭 gradient checkpointing；489,702 条 retrieval 记录约为 479 个
optimizer step。ToolScalER atomic retrieval 使用全参数 ZeRO-3、每卡 batch 24、梯度累积
8、5 epochs；其 E2E 脚本使用 4 GPU、global batch 256、FlashAttention。

当前 L20 `75164` 虽然分配了四张 GPU，但使用每卡 batch 1、global batch 4、每条 query
重新编码 1 个正例和 32 个负例、`document_micro_batch=1`、CPU optimizer offload 和
gradient checkpointing；因此不能与上游标准 causal SFT 的 step 数或吞吐直接比较。该链
仍保持原依赖且不读取运行中分数。

Native v6（本地已清理，保留 deployment/event 记录）不修改冻结树；其改动仅针对执行效率
和分布式安全性：负例拒绝采样、一次性文档 tokenization、
可调文档 micro-batch、8 步梯度累积、显式 FlashAttention 开关、尾部样本等长处理和
source-aligned 默认配置。v6 本地 `51 passed, 1 skipped`，已有 `SOURCE.sha256`；它仍
需要远端显存 smoke、500--1,000 step 吞吐门禁、完整 reload/checkpoint 审计后才能提交。
v6 训练器现已增加默认每 500 个 optimizer step 的 DeepSpeed 中间 checkpoint，并在
`audit.json` 中记录已完成的 checkpoint tag；这解决旧 v5 只在末尾保存、受 48 小时墙
影响时没有可恢复模型的问题。

### 2026-08-30 Native v7 banked-document 候选

为解决 v5/v6 的核心计算瓶颈，曾新增独立的 v7 bank 工作树（本地已清理）。它先对完整
exact-`docid`
语料建立一次 compiled member bundle bank，再在训练 step 中只编码 query，使用完全相同
的 H00 scorer 和 multi-positive loss。bank 同时绑定 compiler 初始权重、identity 顺序、
语料 hash，并要求 `artifact.sha256`/`COMPLETE` 校验；训练审计记录
`document_forward_count=0`、`document_side_frozen=true` 和 bank metadata hash。

这是明确的非对称科学变体：文档侧 compiler/bundle 在 SFT 期间冻结，query 侧 Qwen 仍为
全参数训练。它可能把训练从“每样本 33 个文档 forward”降到“bank 一次构建 + query
forward”，但不等价于 v6 的共享文档全反向路径，不能替代或合并 v6 结果，也尚未提交
集群。v7 的 L20 默认改为每卡 query batch=8、gradient accumulation=1（global batch=32），
使用独立的整数 DeepSpeed 配置；这是与上游吞吐方向一致的保守设置，仍须通过显存 smoke
验证后才能扩大。待 L20 资源释放后，先运行 bank 构建和单步显存 smoke，再做 500--1,000
step 吞吐门禁；任何完整结果仍需 reload、checksum、逐样本预测和独立 NDCG 审计。

#### batch8 + tap-hook 远端门禁

为避免在已部署包上原位改写，batch8/tap-hook 版本以独立不可变目录
`/mnt/home/user46/external/latebound-toolbench-fullsft-v7-bank-batch8` 部署，源码清单
哈希为 `8c225868d6d22007a2e0295215907cff0e4c9be4e07150d33d54afc21fa678b1`，远端
`sha256sum -c SOURCE.sha256` 与全部 `sbatch --test-only` 均通过。已提交但尚未开始正式
计算的新链为：

```text
75265 manifest -> 75266 preflight -> 75267 member bank -> 75268 smoke
       -> 75269 training -> 75270 reload -> 75271_[0-5%4] official NDCG
```

其中 `75269` 默认四卡 query batch=8、累积=1（global batch=32），文档 bank 只构建一次，
query forward 使用 layer hook 仅保留 3 个所需 hidden-state tap。该链仍与旧
`75164` 完全分离；在 `75268` smoke 和 500--1,000 step 吞吐门禁前，不得把它当作正式
结果或替代 v6/v5。

本地后续修订又把 49,936-entry exact identity 索引从每 batch 重建改为 bank 加载时一次
构建；该修订尚未部署或提交新作业，因此不影响正在等待的 752xx 链，也不能把它的源码
变化归入远端包哈希 `8c2258...`。

另一个待提交的 bank 构建修订将文档 bank 分成四个 contiguous identity shard，由四个
L20 rank 并行编码后按原顺序合并；当前 75267 仍使用已提交的一卡 bank 包，故不在其上
原位替换。该修订通过本地编译、shell 检查和 exact-id 合并逻辑测试，并已预部署为
`/mnt/home/user46/external/latebound-toolbench-fullsft-v7-bank-distbank`，其
`SOURCE.sha256` 为 `f2001e5e0d0ba125ee0a7532e180b1a58a7a316838854ce06553193a59db76f6`。
尚未提交新作业，只有在 752xx 链终止或明确重开新链后才使用。

#### 训练时长边界（源码对照）

这里的“上游速度”不能直接套到 Native：ToolGen retrieval 每个样本只有一次 causal
SFT 前向，8 卡 global batch 为 1024，489,702 条记录约 479 个 optimizer step；
ToolScalER atomic retrieval 使用 LLaMA-Factory 的全参数 ZeRO-3，每卡 batch 24、
梯度累积 8，文档已经由 GIST/codebook 表示。Native v5 则在每个 query 上重新编码
1 个正例和 32 个负例，并对这些文档保留反向图；四卡每卡 batch=1、global batch=4，
所以 489,570 条记录需要 122,393 个 optimizer step。`75164` 实测约 40 秒/step，
推算约 56 天，超过 48 小时限时。

v6 的 O(negative-count) 采样、一次性分词、文档 micro-batch 和梯度累积只减少 CPU
开销、padding 浪费和 optimizer/通信频率；梯度累积本身**不会**减少总样本数或文档
模型前向数，也不会把 33 个文档前向变成一次。因此在默认每卡 batch=1 时，v6 的总
GPU 计算量仍与 v5 同阶，不能声称已经达到 ToolGen/ToolScalER 的小时级训练时间。若要达到上游量级，需另立
预注册版本：先建立文档 feature bank，在训练 step 中只反传 query 侧并按固定间隔刷新
文档 bank。这会改变共享文档/查询编码器的梯度路径，必须作为新的科学变体、单独审计和
 单独结果，不能替换 v6 或把两者的分数混在同一行。

### 2026-08-30 Native v8 吞吐门禁候选

在不改写已部署 v7 包的前提下，建立了 v8 bank 工作树（当前保留，因为最新 L20 deployment
由该源码血缘派生）。v8 保留 v7 的 bank
非对称定义，只增加两项工程约束：

1. 训练器启动前强制检查 DeepSpeed 的 `train_batch_size`、每卡 micro-batch 和
   `gradient_accumulation_steps` 与 launcher 参数一致，拒绝字符串 `auto` 或不一致配置。
2. 准备一个未提交的 500-step L20 门禁：每卡 query batch=8、累积 24、4 卡 global
   batch=768，配置哈希由 `SOURCE.sha256` 固定。

该设置预计把 489,570 条训练记录的 optimizer step 从约 15,300 降到 638，但不会减少
总 query forward 数；是否真正加速必须以门禁 wall time 和 tokens/s 为准。v8 bank 模式
同时修正了 v7 文档与代码不一致的问题：训练 loss 现在对完整 49,936-entry registry
计算，只有显式不带 bank 的非正式 pilot 才使用 32 个 sampled negatives。v8 源码清单
`SOURCE.sha256` SHA-256 为 `a87e179ded5324630a3f5333e6a4b5956f9e663bcf8a31d36f8fcb7395b97bda`，本地
compileall、全部 Slurm shell 语法和 global-batch 算术检查通过。该候选尚未部署或提交，
不得与已提交的 752xx v7 bank 链混报；完整训练仍需 bank、smoke、checkpoint、reload、
逐样本预测和官方 NDCG 审计全部通过后才有论文资格。

#### 2026-08-30 上游源码对照后的执行修订

75164 的远端日志给出可复核的速度：step 10--570 每 10 步约耗时 6 分 40 秒，约
40 秒/step；`scontrol` 确认其命令来自 v5，配置为每卡 batch=1、累积=1、ZeRO-2
CPU optimizer offload，且未传 `--flash-attention`。因此它的 48 小时墙限内不可能完成
489,570 条样本的 122,393 个更新，这不是 GPU 未分配而是负载和启动参数错误。

本地已对照固定源码：ToolGen retrieval 脚本（commit
`6839374a255810efe69deea4056eec5c55e25802`）为 8 卡、每卡 batch=2、累积=64、
global batch=1024、长度 1024、FlashAttention、关闭 checkpoint，489,702 条样本约
479 个 optimizer step；ToolScalER atomic retrieval（commit
`a72d46d3352358f39a9274aeb8bf8217b164bb01`）为全参数 ZeRO-3、4 卡、每卡 batch=24、
累积=8、5 epochs、BF16、16 个预处理 worker，文档在数据阶段已转为普通 SFT 输入，
训练步不为每个 query 重算候选文档。

v8 未提交候选已修订为：四卡每卡 batch=8、累积=24（global batch=768），固定整数
DeepSpeed ZeRO-3、无 CPU optimizer/parameter offload，开启 FlashAttention，训练步只
编码 query 并对 immutable exact-docid bank 做 Native H00 全库打分。所有 Slurm 默认
路径现指向 v8，避免误执行 v7；新增配置为
`worktrees/active/latebound-toolbench-fullsft-v8-bank/slurm/ds_z3_bank_batch8_accum24_config.json`。
这仍是“文档侧冻结、查询侧全参数”的非对称 Native 变体，不冒充共享文档全反向训练。
v8 默认 `EPOCHS=1`，与 ToolGen retrieval 的一轮设置对齐；ToolScalER atomic retrieval
官方为 5 epochs。若做训练时长的严格 ToolScalER 对照，必须显式设置 `EPOCHS=5`，并
单独记录约五倍的 optimizer steps，不能把默认 v8 的 638 步当作五轮复现。

本地验证：66 个 pytest（1 个环境相关 skip）、`compileall` 和全部 Slurm shell 语法
通过；配置算术验证为 `8 * 24 * 4 = 768`，且无 offload 字段。gatefix2 的
`SOURCE.sha256` 精确 SHA-256 为
`cf66540fa08959b6bf0fa5469a4272e030b46416c7dab8efd4e40eb80674d06f`，已部署并由
75491--75498 链使用。

本次源码复核还修正了 `fullsft.py` 非 bank 预分词分支的 tap 选择错误：该分支已经返回
`(24,30,36)` 三个 Native tap，旧代码却再次按六 tap 布局切片并得到空 tuple。修复后
分支显式统一两种输入布局，并在 tap 数量异常时立即失败。该修复已包含在 gatefix2
deployment；已提交的旧 752xx 链不受影响。

#### v8 bank 的科学有效性门禁（新增）

进一步逐行核对 `scripts/build_member_bank.py` 后确认：正式 bank builder 现在强制读取
前置 compiler warm-up 的 `COMPLETE` 完成态快照，并在 bank metadata 中记录该快照的精确
hash。随机初始化 compiler 生成的 bank 仍只能作为纯吞吐/通信 smoke；在 warm-up、bank
和完整审计均通过前，v8 的 `formal_evidence_eligible` 必须保持 `false`，不得提交全量
训练或填表。

2026-08-30 实时清理后，远端 `75164` 及其 `75165/75166` 后继、v7 `75266--75271`
均为用户明确取消；未读取运行中 qrels 或分数，保留的只是日志和取消审计。没有提交
新的 v8 作业，因此 v8 目前只有源码和本地
门禁证据，没有远端吞吐数字或论文结果。

此前版本 v8 曾以只读目录部署到远端
`/mnt/home/user46/external/latebound-toolbench-fullsft-v8-bank`；当前 warm-up/NCCL
修订尚未重新部署。当前本地 `SOURCE.sha256` SHA-256 为
`e9e2cfb8d48c49adb89546d3d21f97c1a167f4c8b7569036a12ea21423b53ba9`，也尚未提交
任何 Slurm 作业，未改变 75164 或 752xx 的依赖。待 L20 资源门禁可调度后，仍按
manifest -> preflight -> member bank -> 1-step smoke -> 500-step throughput gate 的
顺序执行，门禁通过前不提交全量训练。

远端已对该目录的七个 Slurm wrapper 执行 `sbatch --test-only`；解析全部通过，输出的
`75300--75306` 仅为测试模式预测编号，并未创建调度作业。当前 v8 仍无实际运行作业。

#### v8 远端一致性修订（本地待重新部署）

源码复核发现，已部署的 v8 member-bank 构建脚本没有继承训练脚本的
`NCCL_P2P_DISABLE=1`；在 L20 既有拓扑上，这可能使 bank gate 在训练前因 P2P
broadcast 超时失败。该环境变量已补入本地 v8 工作树的
`slurm/l20_member_bank.sbatch`，并加入协议回归断言。修订后两个变更文件的精确哈希为：

```text
slurm/l20_member_bank.sbatch       450e2ea54fdb59c470453456ac588db44711cba4c24b86dc1d2c381315604c56
tests/test_fullsft_protocol.py     5138057a86969581c3f7890388a13992e463aa25a0e7e2b2a550af35ba6f44e2
SOURCE.sha256                       cdce9882a7ddb56a0dddb1c5ea4fe448a2c75cd7f09cca04fe83645d63454964
```

本地 `sha256sum -c SOURCE.sha256`、`compileall`、全部 Slurm shell 语法和 65 个
pytest（1 个环境相关 skip）检查通过。已部署的
`/mnt/home/user46/external/latebound-toolbench-fullsft-v8-bank` 仍保持原始
旧 `40e2b4...` 清单且未被原位修改。提交新门禁前必须建立新的不可变远端目录并重新做
逐文件 checksum；该修订不改变 v8 的科学定义，只修复 L20 集体通信环境契约。

#### 2026-08-30 v8 non-bank 全参数主链已提交

为满足全参数定义，主链不加载 immutable member bank，而是使用 warm-up 生成的
compiler 快照初始化；正式训练中 Qwen3-8B backbone 和四视图 compiler 都保持可训练并
交给 AdamW。每个 micro-batch 保留 exact positives，并共享 32 个确定性负例，以接近
ToolGen/ToolScalER 的大 batch 训练制度，同时避免旧版逐 query 重复 33 次文档前向。
bank 路径仍保留，但只作为单独的 asymmetric 速度变体。

新不可变部署目录：
`/mnt/home/user46/external/latebound-toolbench-fullsft-v8-nonbank-20260830`。
远端 `SOURCE.sha256` 哈希为
`5cfbaccf02c5eefce8ea28298493eae4a2628b702ced4bb7a74dcd7043bbcf4f`。

提交链为：`75483 -> 75484 -> 75485 -> 75486 -> 75487 -> 75488 -> 75489 ->
75490_[0-5%4]`。`75483` manifest 已启动，`75484` preflight 已运行，其余任务严格使用
`afterok` 等待。完整训练 `75488` 只有在 500-step throughput gate `75487` 成功后才会
启动；当前尚无正式分数，论文表保持不变。
# 当前运行更新（2026-09-01）

Native Late-Bound ToolBench I1 两阶段主实验的旧 Stage 2 训练 `77622` 在
`1264/1265` 后因尾批 rank 不均衡触发 NCCL 600 秒超时，未生成最终 checkpoint；
这不是 OOM。新 deployment `latebound-toolbench-i1-unseen-api-v8-tailbatchfix-20260901`
仅修复分布式尾批调度，复用已审计的 Stage 1 制品，四卡 L20 链为
`77731 -> 77732 -> 77733 -> 77734 -> 77735`。该链已完成 Stage 2 gate、1,265 步
正式全参数训练、reload 和盲评分；`77735` 聚合阶段因 `aggregate_i1.py` 在写入
`audit/summary.json` 前未创建 `audit/` 目录而失败。训练和盲评分制品均保留且通过各自
的 `COMPLETE`/checksum 门禁，盲评分清单明确 `qrels_read=false`。新建不可变修复包
`latebound-toolbench-i1-unseen-api-v8-aggregatefix-20260901`，仅将输出目录初始化
修正为 `args.output.mkdir(...)`，远端清单 SHA-256 为
`90759f88533334c0a33f6a123c793e231b51ab26b3c445d0af6290084d46bf50`。由于 Slurm 已不
接受对已结束 `77734` 的新依赖，修复聚合由脚本内的盲评分 `COMPLETE`、manifest 和
score checksum 检查门控，作业为 `77926`，当前已完成。聚合摘要和 COMPLETE 的 SHA-256
均为 `0818ef4ad2901e43d45709bbd13893c4a601aebfe04d7c2b2c7478d03bbc4336`。

#### I1 分数低于闭集结果的原因

E044 的 `17.08/22.57/26.11`（NDCG@1/3/5）和 `28.99%` Hit@3 不能与此前闭集
1K 结果的 `95.08%` Hit@3 或 STQ 的 `53.03%` Hit@3 直接比较：

- E044 只用 1,000 个训练 API、10,118 条训练样本，测试是 10,439 个与训练 action
  完全不相交的 I1 文档；注册对每个测试文档均为 zero-step。
- 95.08% 来自 1,000 API 闭集诊断，训练和测试共享同一动态 API registry；53.03% 来自
  STQ 的 837-way/1,066-query 面板，数据域和候选规模也不同。
- ToolScalER 论文的约 93% NDCG 是其完整 ToolBench 训练制度和 action-level 评测，不能
  当作当前 1K 训练、完全未见 API 的同条件基线。

因此 E044 的下降首先反映候选规模、API 分布和 zero-step 跨 API 泛化难度；当前
`score_manifest`、sealed qrel 顺序、NDCG 聚合和 checksum 均通过，尚无证据表明分数被
评测器压低。若要与上游论文公平比较，下一条实验必须扩大到官方完整训练 split，并同时
保留 API-disjoint 结果作为更严格的泛化面板，不能混写成同一行。

需要特别区分：此前计划中提到的“约 1,000 个候选的小子集选择”并未在 E044 中执行。
E044 使用的是完整 I1 候选目录（10,439 个文档），因此只能作为完整 I1 严格诊断；小子集
实验仍是待运行项目，不能把 E044 当成小规模结果。

#### 2026-09-01 ToolScalER 1K 闭集 Native 对齐实验

为直接测量“240 个目标 API 的那套集合”，新建 deployment
`latebound-toolscaler-1k-closedset-v1-20260901`。它复用已审计的 ToolScalER 1K 输入：
1,000 个候选 identity、7,963 条训练行、1,951 个 query-text-disjoint 测试 query，测试
qrels 覆盖 240 个目标 identity。这是与此前约 95% Hit@3 结果同候选集的 Native 对照诊断，
不是官方 ToolBench 主表。

ToolScalER 四-token code 的 5 个双 identity collision bucket 造成 20 条训练行无法唯一
确定 exact API；这些行全部 fail-closed 排除。原始 corpus 中无文档的 8 个候选只保留
metadata payload，且不在 240 个 qrel 目标中。Native 采用冻结 backbone 的完整 1,000-way
compiler Stage 1，再进行全参数 causal Stage 2；测试注册 optimizer steps=0。qrels 只在
最终聚合打开。

L20 DAG 已提交：`77968 -> 77969 -> 77970 -> 77971 -> 77972 -> 77973 -> 77974 -> 77975`。
训练、评分和聚合均与旧 I1 输出隔离，源码清单 SHA-256 为
`a6ec869c0b508bc86f9697fa5b5227a159566d8820c8a7bb30ccd4f7c4ef0f32`。终态前不读取部分
分数，结果完成后只作为闭集诊断记录，不自动更新论文表。

原评分 `77974` 因漏设冻结 `latent-register/src` 的 `PYTHONPATH` 在模型加载后退出；训练和
reload 制品未受影响。不可变修复部署 `latebound-toolscaler-scorefix-v2-20260901` 仅修复
评分环境，复用同一 Stage 1/Stage 2 制品，重新提交 `78205 -> 78206`，两者均完成
`COMPLETED 0:0`。完整证据已复制到
`paper-evidence/l20/native-toolscaler-1k-closedset-scorefix-78205-78206/`，远端与本地
哈希一致。最终结果：exact/code Hit@1/3/5 为 `86.21% / 90.62% / 91.95%`，MRR 为
`88.82%`，单目标 NDCG@1/3/5 为 `86.21% / 88.79% / 89.34%`。评分 manifest 明确
`qrels_read=false`，仅最终聚合打开 sealed qrels；注册 optimizer steps=0。该结果仍是
1K 闭集诊断（与测试共享候选 registry），`formal_evidence_eligible=false`，不写入官方
ToolBench 主表。
