# ToolBench memory v2：结构化压缩与调用规则读取

本分支在 `2233676df1c7246a5a0738a476041905879f18fa` 的 ToolBench v1 上实现。
主线为 **两层 structured resampler、8 slots、schema 内容监督、共享 Qwen 全参数联合训练**。
16 slots 使用相同配方，仅改变输出容量。这里记录工程实现和 CPU 验证，不宣称已经改善
ToolBench 参数正确率、SoPR 或 unseen 泛化。

旧 v1 文档、验证记录和上游实验脚本保留。STQ 只提供问题定位的历史依据，不作为
ToolBench 多参数 schema 能力的证据；本轮没有读取真实测试答案或运行远程作业。

## 注册与训练的数据流

```text
工具说明 + 完整 JSON Schema
             |
     同一个 Qwen，一次文档 forward
             |
  最后一层全部有效 token states H_D
      |                        |
 masked mean                 原 token states + 参数字段聚合视图
      |                        |
   原 C_sel               两层 read/refine：
      |                    cross-attention -> slot self-attention -> FFN
  动态检索行 O_D                  |
      |                    全 hidden-size value readout + 投影残差
      |                        |
      +------------------ 8 × H memory M_D
```

默认内部宽度 512、8 heads、2 层。每层都有文档 cross-attention、槽位之间的 self-attention
和 FFN。最终读取完整 H 维的值，并加入 learned projection，输出仍是每工具 `8 × H`；
16-slot 版本输出 `16 × H`。内部 512 维不等同于最终 memory 维度。

字段视图来自已有文档 states：序列化完整 schema 时记录每个 `properties` 字段的
“名字＋子树”的字符区间，经 fast tokenizer 的 offsets 映射到 token，做 masked mean。
嵌套字段同时保留粗粒度和细粒度视图。这些字段视图与原 token states 一起供 slots 读取，
不需要按字段重新 forward Qwen，也不使用调用参数标签、问题或测试答案。

序列化必须与原 `ToolSpec.registration_document` 完全一致。字段包含 `/`、`~` 和中文时
保留精确字符与 JSON Pointer escaping。字段没有有效 token、文档过长、mask 包含 padding
都会报错。没有 `properties` 的 schema 仍可通过原文 states 注册。

文档侧、选择侧、caller 都使用同一个 backbone。训练期间重新编码当前参数下的文档，
没有冻结 E0 的旧 states bank。完成训练后冻结所有参数：每个新 API 一次文档 forward
同时生成 key 和 memory；不增加 API 专属可训练参数、不更新 optimizer、不修改静态词表。

## Schema 监督：内容只来自 memory

新辅助任务不恢复整篇名称和介绍。它要求 caller 输出调用所需的结构化事实：

| 任务 | 精确目标 | 范围 |
|---|---|---|
| `schema_names` | 参数字段的 schema pointer 列表 | 包含嵌套 `properties` 字段 |
| `schema_types` | schema pointer → 显式 type | 保留 type 数组，不推断未声明类型 |
| `schema_required` | schema pointer → 原始 required 列表 | 保留嵌套/条件位置，不能当作全局必填 |
| `schema_enums` | schema pointer → enum/const | 保留值、顺序和 JSON 类型 |
| `schema_defaults` | schema pointer → 显式 default | 不漏掉 null、false、0；不补造默认值 |

pointer 指向 schema 节点，例如 `/properties/options/properties/unit`，不等于调用实例路径。
`definitions`、数组 items 和组合/条件 schema 的显式事实会被记录；本地 `$ref` 保持引用，
不会递归展开，definitions 只遍历一次。examples 中看起来像 schema 的内容不会成为标签。
这些目标不等于完整 JSON Schema 逻辑求解器。

**Native 的 schema 提示只包含通用任务指令和 memory**：没有 query、历史调用、当前 plan、
API identity、字段名称提示或原文。目标由训练文档的真实 schema 得到，不从调用答案反推。
默认值监督只采用结构化 `default`；不会自动猜测自然语言里的隐含默认值。自然语言约束
仍由完整文档编码和真实调用目标训练。JSON Schema 的 default 是注解，不意味着在线必须
自动填入所有默认值。

例如，问题只问巴黎天气，但某个 endpoint 的 schema 要求 `unit`，且 `const="celsius"`。
漏掉 unit 会被必填项指标记录，输出 fahrenheit 会被 const 指标记录。换成另一个固定
fahrenheit 的文档，schema 目标也必须改变；通用 schema 问题本身保持一致。

每步训练损失：

```text
L = L_selection + L_arguments + 0.2 × L_thought + 0.5 × L_schema
```

保留检索选择损失，默认训练整个 Qwen，包括输入 embedding 和 LM head。Schema 梯度同时
进入 memory compiler、文档侧 backbone 和 caller。它复用本步已经编译的 memory，不增加
文档 forward；辅助 caller forward 的成本需要另计。

默认每次曝光轮换监督五类中的一类，以控制训练开销。`--schema-tasks-per-step 5` 会在
每次曝光监督全部五类并取平均，开销相应增加。全局轮换不保证每个 API 都获得五类监督；
checkpoint 的 `schema_task_exposures` 记录实际全局曝光数，不能据此声称每 API 完整覆盖。
数据审计记录各类事实数，token 预检检查每个被监督 API 的全部五类完整目标。

空事实目标保留，用于学习“未声明”；诊断中单独报告非空目标数量，空目标高正确率
不能触发内容利用认证。没有加入任意 wrong memory 下压低原答案概率的训练损失。

## 配方和运行入口

两份可直接读取的配方只有 `memory_slots` 不同：

- `configs/toolbench-memory-v2-8.json`：主线。
- `configs/toolbench-memory-v2-16.json`：容量对照。

先按 [v1 的数据合同](TOOLBENCH_AGENT_TRAINING.md#required-training-data) 准备训练 API
完整 schema 和 ToolBench/ToolGen 串行调用轨迹。STQ 参数数组不适用。数据目录和输出目录
必须明确指定；不会自动发现或读取测试文件。

```bash
PYTHONPATH=src accelerate launch --config_file /path/to/reviewed-accelerate.yaml \
  -m latent_register.train_toolbench_agent \
  --recipe configs/toolbench-memory-v2-8.json \
  --tools /path/to/train_tools.jsonl \
  --trajectories /path/to/train_trajectories.jsonl --source-format toolbench \
  --model-path /path/to/base-Qwen3-8B --model-role base \
  --epochs 3 --batch-size 1 --gradient-accumulation-steps 8 \
  --gradient-checkpointing --seed 17 --output-dir /path/to/NEW-memory-v2-8
```

16-slot 对照换配方文件和新输出目录。显式命令行参数覆盖 recipe 默认值，最终配置和
recipe hash 会保存。配方是待验证的起点，不是最优设置或资源授权。

`--memory-kind legacy` 保留旧 resampler 以便比较；如要隔离架构影响，应保持同样的 schema
监督、数据、更新次数与 caller 训练范围。`--schema-weight 0` 是明确关闭辅助项的消融，
不能再称作新主线。`--condition full_document` / `query_only` 仍支持独立训练的 reader
基线；它们共享选择机制，并以各自实际输入进行相同的辅助任务。把已有 Native checkpoint
临时换输入只能算输入消融。
全文/query-only reader 不执行未使用的 C_mem，也不缓存其输出，避免把额外压缩成本错误地
计入对照方法。

训练前先用 tokenizer 检查全部完整调用目标、实际展开的 memory 位置和五类 schema 目标，
过长会在加载 backbone 权重前报错，不静默截断。`--audit-only` 只做数据审计，不加载模型；
它不包含需要 tokenizer 的长度预检。Warm start 仍要求各历史训练阶段的 API lineage 合集。

Checkpoint v2 保存 compiler 种类、宽度、层数、head 数、slots、整个 backbone、两条
compiler、tokenizer 和实际非持久 buffer。v1 checkpoint 会明确恢复为 legacy compiler，
不会被自动装入新结构。保存和评估输出都拒绝覆盖已有目录。

## 开发诊断：规则读回和参数错误分开

```bash
PYTHONPATH=src python -m latent_register.evaluate_toolbench_memory \
  --checkpoint /path/to/checkpoint \
  --tools /path/to/dev_tools.jsonl --split dev \
  --trajectories /path/to/dev_trajectories.jsonl --source-format toolbench \
  --device cuda --max-new-tokens 512 --output-dir /path/to/NEW-v2-dev-diagnostic
```

省略 trajectories 可以只做 schema 读回。只接受 `dev` 或 `validation` 标签，拒绝 test，
并检查开发 API 与 checkpoint 声明的训练 API 不重合；Finish 不作为 API 泛化样本。
Lineage 的真实性仍依赖已有来源认证，不能通过改标签把已训练 API 变成 unseen。

每个目标执行 correct、blank、wrong 三条件。Blank 仍是同样数量的有效 attention 位置；
wrong 使用固定的不同 schema 工具内容。三条件的控制文本和 memory 位置数一致。
错配按确定性规则选择，不保证一一映射；完整映射保存。没有不同 schema 时拒绝伪造负对照。

Schema 诊断不带问题或 API 标签，五类都自由生成，报告结构化 exact、事实 precision/recall、
非空目标数及截断数。正向内容 signal 要求 correct 对 blank 和 wrong 的事实 recall 差值，
在 API 聚类配对 bootstrap 下 95% 区间下界均大于零。仅使用目标非空、且错配会改变该项
目标的事实；这个资格由文档决定，不按生成结果筛选。只有一个合格 API 时不认证区间。
这是“这批开发 API、这个 checkpoint 存在部分事实读取信号”，不等于完整 schema 保真、
参数质量提升或跨 seed 结论。额外字段仍体现在 precision 和 exact 中。

参数诊断是 **oracle-tool + 已记录的因果历史**。当前 plan 由模型生成一次，在三条件中
固定；不输入 gold thought。生成 plan 失败时保留失败分母。参数侧报告：

- 根 JSON object exact、解析成功率、schema 合法率；
- 已解析输出中的必填字段遗漏、类型错误、enum/const 错误率；
- 未解析/截断单列，不当成“零个缺失字段”；
- 每条生成文本、原始目标、计划失败状态、输入位置和生成开销。

条件 schema 的错误来自 validator 的失败分支，不等于一个人工整理过的最小修复集合。
这些诊断不补成真实检索端到端结果，也不产出 SoPR。正式实验仍要在相同 executor、simulator、
judge 和预算下比较实际任务成功率；不能因为 memory 参数指标改善就自动宣称端到端改善。

## 实际成本记录

`register_tools(..., profile=True)` 分开记录 tokenization/字段准备、backbone forward、
C_sel、C_mem 时间、文档 token/字段数和缓存 tensor bytes。GPU 计时时同步设备。
新 API 只登记一次，重复注册复用缓存。诊断为每个条件记录真实输入位置、生成 token 和时间；
一份 argument plan 被三个条件复用，其成本只能计一次。

在线 `run_serial_agent` 记录模型决策时间、executor 时间、实际调用次数和执行回执：
executor 可返回 `ExecutionResult(content, success=True/False/None)`。
普通工具结果不会被猜测为执行成功；异常记为失败。Finish 给出答案也不代表答案正确，
输出始终标明 `task_success_judged=False`，最终质量仍需独立评估。

本轮已实测 compiler-only CPU 开销（arm64、单 CPU 线程、FP32，合成
`[1,256,4096]` states、16 个字段视图、2 次 warmup 后 5 次测量）：

| Compiler | 共享参数量 | 中位耗时 | 每工具 memory 输出字节 |
|---|---:|---:|---:|
| Legacy 8 | 1,590,280 | 0.751 ms | 131,072 |
| Structured 8 | 13,132,808 | 3.759 ms | 131,072 |
| Structured 16 | 13,136,912 | 3.962 ms | 262,144 |

该测量没有加载 Qwen，不包含 backbone、tokenizer、字段解析或 caller，不能当成 A100
注册延迟。8-slot 新结构增加约 11.54M 共享参数，但输出位置数和 memory 缓存容量保持不变。
缓存表中的检索行和元数据开销另计。原始测量见
`reports/toolbench-memory-v2-20260909/compiler-profile-cpu/PROFILE.json`。

```bash
PYTHONPATH=src python -m latent_register.profile_toolbench_compiler \
  --output-dir /path/to/NEW-compiler-profile --device cpu
```

## 验证与未完成的研究验证

本轮 **79 项 CPU 测试全部通过**，结果与源码哈希见
[验证记录](../reports/toolbench-memory-v2-20260909/VALIDATION.json)。
CPU 验证包括原有回归、两层 attention/FFN 梯度、schema-only 梯度回到文档侧
Qwen、字段区间、空/错 memory 的等长接口、8/16 slots、v1/v2 重载、真实 tiny Qwen 训练
CLI、五类监督曝光记录、开发诊断 CLI、配对统计和失败成本。它们验证实现，不验证 8B 精度。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_toolbench_memory_v2.py tests/test_toolbench_agent.py \
  tests/test_meta_readback.py tests/test_meta_registration.py \
  tests/test_memory_oracle.py tests/test_physical_tokens.py
```

尚未运行真实 ToolBench 8B 训练、多卡/DeepSpeed、正式 executor 或 SoPR。新结构是否减少
参数缺失、8 与 16 的质量/成本取舍，必须用匹配的开发实验验证；不能由 CPU 测试或旧 STQ
结果代替。本分支的工程实现已具备这些训练和诊断入口。
