# Ours ToolBench 端到端评测源码

本目录将本地 v42 task-state release、memory／全文 reader 对照和
completion-v6 来源约束补答代码整理为可独立使用的源码快照。
模型权重、训练/评测数据、逐题结果、凭据和集群作业脚本由运行环境单独提供。
本次发布验证 CPU 回归与 CLI 接线，没有重新运行真实 8B、GPU 或 SoPR 评测。

## 入口与版本

| 入口 | 行为 | 来源 |
| --- | --- | --- |
| `--protocol native` | 原生 v42 task-state agent，使用 checkpoint 的 reader | `release-v1` |
| `--protocol reader --reader-mode memory_fp32` | FP32 投影，Top-5 memory reader | `eval-reader-ab-v1` |
| `--protocol reader --reader-mode documents_fp32` | FP32 投影，Top-5 完整文档 reader | `eval-reader-ab-v1` |
| `--protocol completion-v6 --reader-mode …` | 完整 query 回退、来源片段补答、共享调用/生成预算 | `repairs/completion-v6-source-patch-20260923` |

后两类入口必须显式选择 reader。全文 reader 是推理输入消融，不是训练匹配的全文基线。
completion-v6 是独立推理协议变化，不能将其效果归因为 memory 或训练的单变量贡献。
v6 claim-guard 的后续探针尚未完成原生重规划/工具执行接线，因此未纳入上述运行入口。

`release-v1/source/src/latent_register/` 保存运行及测试所需依赖闭包。
`release-v1/tests/` 包含真实 tiny Qwen 前向/梯度/checkpoint roundtrip、串行工具调用、
HTTP 回执、数据防泄漏和有界恢复测试。reader 与 completion 测试在各自目录运行，
避免不同版本的同名 Python 模块相互污染。

`analysis-v1/.../deployment/` 和 `repairs/completion-e2e-50-v5-coverage-routing-20260923/`
仅保存 completion-v6 的八个父源码文件，用于原有不可变父哈希回归，不是推荐运行入口。
`SOURCE_MANIFEST.json` 记录每个导出文件在本地工作区中的相对来源及 SHA-256；
这份清单不是原集群 release 的完整部署清单。唯一历史测试适配是显式传入原来的
1024/1024 生成预算，避免依赖 v42 已改为 32/256 的默认值。核心源码逐字节保留。

## 安装与 CPU 检查

在仓库根目录执行（Python 3.9 为本次验证环境）：

```sh
python3 -m venv .venv-ours-e2e
. .venv-ours-e2e/bin/activate
python -m pip install -r experiments/ours-e2e-v42/requirements-cpu.txt
python experiments/ours-e2e-v42/check.py
```

`check.py` 先校验导出哈希，再分别运行三组 pytest，最后检查三个 CLI 的 `--help`。
设置离线 Hugging Face 模式；tiny 模型在本地随机初始化，不下载权重，不调用 simulator/judge。
HTTP 合同测试只启动本机 loopback 服务。可选的上游转换器集成测试在未设置
`NATIVE_TEST_STABLETOOLBENCH_SOURCE` 时明确 skip；需要额外提供固定版本的上游源码。
GPU 环境需另选与 CUDA 匹配的 PyTorch；CPU 依赖文件不是 CUDA 环境锁文件。

## 真实评测输入

需要准备：

- 本版本支持的 checkpoint：包含 `agent.json`、`SHA256.json`、`runtime.pt` 和 `backbone/`。
  LoRA checkpoint 还要求 `base_reference` 中的绝对路径和文件哈希可核验。
- tools JSONL：精确 API identity、完整文档、参数 schema、split 与 source binding。
- queries JSONL：每行只允许 `id`、`query`、`split`，不含答案或 gold 工具。
- 训练 API identity JSON 列表，供 seen/unseen 边界检查。
- 单独的 executor JSON：服务地址、backend kind/revision、超时及凭据环境变量名。
  可参考 `executor.example.json`，替换占位服务与版本后再使用。

以下命令中的 `/path/to/...` 均需替换；输出目录必须尚不存在：

```sh
python experiments/ours-e2e-v42/run.py \
  --protocol native \
  --checkpoint /path/to/checkpoint \
  --tools /path/to/tools.jsonl \
  --queries /path/to/queries.jsonl \
  --train-api-identities /path/to/train-api-identities.json \
  --executor-factory latent_register.toolbench_http:create_executor \
  --executor-config /path/to/executor.json \
  --split dev --allow-seen-api --device cuda:0 \
  --max-calls 8 --max-validation-retries 2 \
  --max-thought-tokens 32 --max-argument-tokens 256 --max-answer-tokens 1024 \
  --output-dir /path/to/new-output
```

reader 对照将 `--protocol native` 替换为
`--protocol reader --reader-mode memory_fp32` 或 `documents_fp32`，其他输入相同，
每臂使用独立输出目录。补答修复将其替换为
`--protocol completion-v6 --reader-mode memory_fp32`（或 `documents_fp32`）。
`run.py` 自动选择本快照的包路径，不需要安装或修改仓库根部的旧版 `latent_register`。
`--allow-seen-api` 是这里明确声明的开发协议；不能将该结果称为未见 API 测试。

可选字段名映射 executor 为 `declared_name_executor:create_executor`：
reader/completion 入口已加入对应 variant 路径，还需在配置中提供
`enable_declared_parameter_names`、`parameter_mapping_path`、`parameter_mapping_sha256`、
`parameter_mapping_tools_sha256`、`parameter_mapping_documents_sha256`。
映射必须由同版本工具及 simulator 文档认证，不能猜测或填补参数值。

## 输出和验证边界

原生/reader 入口保存 episodes、REPORT、PROVENANCE 和 ToolEval 导出；reader 另存
`READER_PROVENANCE.json`。completion-v6 保存实际运行预算/哈希的
`RUN_CONFIGURATION.json`、逐题事件/执行回执及共享工具导出。
失败后保留不完整目录，使用新目录重跑；没有完整导出标记的输出不能当作完成面板。

程序的 Finish、非空答案、HTTP 成功和来源引用通过均不等于任务成功。
这里的 runner 不执行 judge，也不产生官方 SoPR；完整任务率需要固定评判器及全量覆盖审计。
completion-v6 的同模型语义复核仍可能误收/误拒，其 CPU 反例测试不证明真实任务改善。
