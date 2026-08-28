# 论文故事与方法

## 研究问题

模型能否学会一种可复用的注册协议：全部共享训练结束后，只根据一个此前未见过的
可执行 API 文档，就把它挂载为原生生成动作，而且不为该 API 优化或修改模型？

这是一篇关于注册与选择的论文，不是普通检索论文；它也不宣称选择器正确就会自动解决
参数生成、多轮规划、BFCL、Tau2 或最终任务奖励。

## 方法

对于查询 `x` 和 API 文档 `D_i`，共享模块生成：

```text
q_x = Q(x)
h_i = E(D_i)
(O_i, M_i, X_i) = C(h_i)
```

- `O_i`：一个或多个用于选择的动态输出行。
- `M_i`：可选的输入/读回记忆。
- `X_i`：不可变的 exact identity 与执行胶囊。

对于一一对应的注册表绑定：

```text
r_i -> (exact_identity_i, D_i, O_i, M_i, X_i)
```

原生选择器计算：

```text
s_i(x) = aggregate_j(q_x^T O_ij)
logits[r_i] <- s_i(x)
```

普通词表 logits 全部保留在同一个归一化分母中，未激活的预留地址会被屏蔽。如果
`r_i` 获胜，运行时将该地址解引用到该 API 自己的文档、记忆和执行胶囊。`r_i` 对应的
静态输入/输出词表行不是语义表示。

### 训练边界与部署边界

共享训练阶段可以优化查询/文档编码器、compiler、检索头和明确声明的共享 backbone
适配参数；部署阶段要求更严格：

```text
optimizer_steps(new API) = 0
API_specific_learned_parameters = 0
document_forwards_per_distinct_API = 1
delta(backbone, compiler, tokenizer, static rows) = 0
identity -> logical slot -> physical ID 一一对应
```

当前 STQ 主诊断冻结了 Qwen3-8B，只训练约 1,103 万个 Native retriever/compiler
参数；它不是 backbone LoRA 结果。历史分支另行测试过 LoRA 和全参数监督微调的容量。

### 为什么删除地址轮换

物理 ID 只是数组下标。第一训练阶段中，它根本不进入前向计算；读回阶段中，只要标签和
注册表同步，对激活 ID 进行置换只会置换 logit 的列和注册表查找。因此，按 epoch 随机
轮换地址不会产生语义学习信号。

正式方法改为使用确定性训练占位地址，并在部署时绑定任意新地址。地址置换和从未暴露过的
地址测试仍保留，用于证明语义跟随动态 bundle，而不是跟随整数 ID；它们不是方法组件，
也不是论文消融项。

## 创新点

我们不声称任一单独组件首次出现。已有工作已经包含文档检索器、工具 token、动态词表、
latent memory、codebook、摊销适配器和零步文档编码器。可辩护的研究目标是以下组合：

```text
训练结束后注册未见可执行 API
+ 有界文档 -> 输出/读回/身份 bundle
+ 在普通词表最终 softmax 中作为原生动作
+ 精确解引用到自己的身份和载荷
+ 每个新 API 零优化器步、零专属学习参数
+ 静态模型零改动
+ 显式地址置换与顺序追加不变量
```

建议表述为：

> 将可执行工具接口摊销式延迟绑定到冻结语言模型的生成动作空间。

只有创新性还不够。方法还必须在匹配的完整候选空间中保持有竞争力的 exact API 选择
能力。如果明显弱于忠实复现的相邻方法，论文必须如实报告“协议新颖但效果不足”，不能
宣称性能先进。

## 相近方法

### CoTools / Chain-of-Tools

CoTools 学习共享的查询投影、工具投影和维度权重，再使用归一化相似度进行检索：

```text
q = normalize(w * (A_q(h_q) + h_q))
d = normalize(w * (A_d(h_d) + h_d))
score(q,d) = q^T d
```

它可以在不进行新工具专属优化的情况下编码新工具，因此是相近的选择对照。它使用外部
retriever，而不是物理 token 动作和精确运行时解引用。

### ToolScalER

ToolScalER 将工具文档压缩成 gist 向量，拟合 PCA 和残差聚类，分配 semantic code，
再训练模型生成合法 code token。新文档可通过冻结的 PCA/centroid 零步分配 code。
它的源码 E2E 阶段采用全参数监督微调，而且 semantic code 可能对应多个 exact API。
严格比较必须保留公共源码，同时报告原生 NDCG 与 exact-API Hit@k，并对碰撞按失败处理。

### ToolGen-Fixed 与 Incremental ToolGen

固定 ToolGen 学习静态工具身份；Incremental ToolGen 为新工具继续执行文档到 token
的优化。它们是适配成本基线，不是零步竞争者。必须明确报告优化器步数、改变的标量数、
checkpoint 字节数、耗时和旧 token 映射保持情况。

### 其他边界

只有 ToolWeaver 的公共源码、数据、codebook、checkpoint 和严格冻结后 holdout 审计
全部通过后，才能加入比较。ToolkenGPT/TInR/ParaTool 一类 API 专属 embedding 或
模块不满足本项目的零步合同。BM25 和冻结 E5 仅是可达性控制，不是同类方法行。

## Benchmark 故事

1. **STQ exact unseen-API selection** 是主要的同类匹配面板：999 个已见 API、
   837 个未见 API，1,066 个测试查询在完整 837-way registry 中排序。
2. **ToolBench** 用于源码兼容性、原生方法复现、迁移和规模证据；不得用更友好的分组身份
   替换 STQ 主结论。
3. **严格物理 token 身份实验** 证明 exact identity、未见地址、普通词表归一化和自己的
   文档解引用。官方 ToolGen G1 映射仍被已审计碰撞阻塞。
4. **注册成本与可扩展性** 报告 10/100/1K/10K/47K registry 的成本、存储、延迟和
   顺序追加稳定性。
5. **BFCL/Tau2/MCPEvol/StableToolBench** 是下游或探索轨道，不能修补失败的选择结果。

## 论文主张纪律

- 选择、参数生成和端到端任务完成是三类独立指标。
- 源码/制品完整性审计可以在负面结果上通过；`passed=true` 不表示方法性能好。
- 正式对照行必须来自 source-only 执行；旧 clean-room 数值可用于诊断，但不得称为官方复现。
- 运行中作业、部分分数、smoke、单 seed 或没有完整制品的调度器退出码都不能支持可行性主张。
- 当前论文可以依赖注册成本/规模结果；主要效果和严格身份仍受论文表格 gate 约束。

## 主要科学风险

- 使用局部或采样负例训练时，在全候选空间形成全局 hub。
- 文档归一化或 semantic code 碰撞造成 exact-API 歧义。
- backbone 表示容量与目标函数/候选几何之间相互混淆。
- 单步检索训练的选择器即使单步正确，也可能无法进行有状态的多轮下一工具决策。
- 源码适配在不知不觉中变成 clean-room 重实现。

对应控制包括：完整分母评测、碰撞失败关闭、匹配训练边界消融、官方源码运行和独立的
下游评测。
