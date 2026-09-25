# Ours best5400：实际训练数据与 PORTS 对齐入口

本包绑定当前 765 题评测使用的 **v42、累计 5400 更新** checkpoint，
其 manifest SHA-256 为
`8ea5d5741e8cfb47aa4ffb66d4fcdd1ab4d34c6b084a4ea08d24a8073abb066d`。
评测 reader 为 `documents_fp32`。这不是旧 5000 更新 checkpoint，也不是 completion-v6。

本次从训练集群读取了实际样本、工具文档和 hard negatives，并与训练时的文件哈希核验。
Git 中提供完整样本重建合同；未经修改的原始轨迹另行保留，因为少量训练轨迹包含
来源未确认的 API-key 字面量。**本目录不公开原始 query/history/argument/answer 文本。**
拿到下面指定哈希的两份原始输入，或已取回的样本归档，即可逐条重建实际训练样本。
工具文档、参数 schema、原始 API identity/source binding 和 hard negatives 已随包提供。

## 绑定到哪个训练集合

| 项目 | 数量 | 文件/含义 |
| --- | ---: | --- |
| 实际贡献到 checkpoint 的样本呈现 | 43,200 | `sample-order.jsonl.gz`；每次呈现都有 update、rank、源索引、mask 和记录指纹 |
| 不同 prepared decision ID | 42,832 | 不按 ID 合并；同 ID 的 task-state/mask 可以不同 |
| 不同 query ID | 31,289 | 来自真实消费记录 |
| tool / final / give_up 呈现 | 26,360 / 11,193 / 5,647 | 三类均参与原训练；不是 43,200 条完整轨迹 |
| 实际正例选择目标 API | 7,171 | `positive-seen-api-ids.json`；mode=tool 且 selection mask>0 |
| 所有 mode 的 selected 文档关联 API | 19,254 | `selected-document-api-ids.json`；包含终止记录关联的文档 |
| 原训练合同声明的正例 API 名单 | 9,900 | `ancestor-declared-train-api-ids.json`；整个 prepared inventory 的合同名单 |
| 实际训练工具候选库 | 48,318 | `training-registry-api-ids.json` 和 `train-tools.jsonl.gz` |
| 最后 200 更新的 task-state 监督 | 865 / 1,600 | 其余 735 条保留原接口回放；具体 masks 保留在原始记录 |

7,171 不能作为所有文档、负例、schema、历史暴露的并集，不能简单用候选库减去它
就宣称其余 API 严格 unseen。48,318 是训练工具文件中的普通 API 数，运行时另加 Finish；
也不要与 765 题评测的 47,323 个普通候选 API 混用。

前 5,200 更新的 41,600 次呈现由九段已提交 checkpoint 的各 rank 记录、恢复游标和
optimizer 边界共同确定。最后 200 更新由有序 1,600 条输入和全部进度记录确定，原 trainer
没有独立的逐 rank 消费日志。最后一段有 368 个 ID 在祖先中出现过，保留其改变后的监督。

**按 `sample-order.jsonl.gz` 重建，不按计划 shuffle 取前缀。** 从两卡切换到四卡时，
实际训练跳过了 shuffle 的 `[8000,16000)` 区间。原有 5000 步理想前缀清单已被消费记录纠正。
本包亦支持导出真实 5000/5200 更新前缀；5000 更新应有 40,000 次呈现、7,004 个正例 API。

## 校验和重建

仅验证公开清单、样本顺序、工具名单及文件哈希（不需 GPU 或第三方库）：

```sh
python3 experiments/ours-e2e-v42/training-data/rebuild.py --verify-only
```

重建需要两份原输入，`SOURCE_INPUTS.json` 给出训练路径及精确 SHA-256：

- base：`prepared-v6-plan/records.jsonl`，共 173,746 条，SHA-256
  `340cedb8969462a6c4142c0d5c9578dce014d2c5685995ec2318873b98a5c445`。
- task-state：v42 的 `prepared-v1/records.jsonl`，共 1,600 条，SHA-256
  `a747b872ecb3f611934b4e3edd60edd19d4af6b95ed85d02d10a96c416d6e037`。

```sh
python3 experiments/ours-e2e-v42/training-data/rebuild.py \
  --base-records /path/to/prepared-v6-plan/records.jsonl \
  --task-state-records /path/to/prepared-v1/records.jsonl \
  --output-dir /path/to/new-ours5400-data
```

也可将 `--base-records …` 替换为 `--base-selection /path/to/records-base.jsonl.gz`，
使用本次已取回的 41,600 条原始样本包装归档。task-state 输入也接受 gzip。
脚本先校验输入全文哈希，再验证每条实际样本的 canonical JSON SHA-256、ID、query、
selected、mode 和 masks，最后输出按实际消费顺序排列的 `records.jsonl.gz` 与
`RECONSTRUCTION.json`。输出包含完整因果历史、参数/回答目标、repair、intent 和
task-state（适用时），不会重新生成教师标签、删掉 final/give_up 或合并重复 ID。

添加 `--through-update 5000` 或 `5200` 可导出祖先前缀，此时不需要 task-state 输入。
输出目录必须是新目录；本脚本不会覆盖旧数据。

## 如何对齐 PORTS

先选定比较层级。若比较 selector，使用上述样本中 `mode=tool` 且
`masks.selection>0` 的正确目标，并明确 PORTS 输入使用原始 query、因果历史还是 intent；
这些输入形式不应被悄悄互换。若比较端到端系统，还需对齐 final/give_up、填参、修复、
task-state 监督以及各自 caller 的训练来源。

对照时至少核对 source decision/query ID、精确 API/source binding、样本重复次数、
候选库、负例规则、监督 masks 和数据版本。`hard-negatives.json.gz` 是冻结负例候选表，
不代表其所有条目在每次更新都被使用。模型相关的 Top-5、实际 token 数和梯度不能仅凭
候选表恢复；本包不声称两种方法已有相同 token 或计算预算。

`lineage.json` 记录各段的训练配置与原合同哈希；`source/train_task_state.py`
保留最终段的原始采样与训练代码。这里只重建数据，不启动训练。
**当前尚未获得并比对 PORTS 的实际消费清单，因此 `PORTS_training_alignment_verified`
保持 false。** 这份包提供 Ours 侧可核验的基准，不代替 PORTS 侧的证据。
