# ToolBench 端到端训练与执行入口

2026-09-09 的第一轮实际训练使用共享 Qwen3-8B、两层 8-slot compiler 和 schema 监督。
训练源码固定于 `ed1c5a6`；后续增加的执行入口不会混入已经提交的训练源码。

## 数据与预算

从明确的 ToolGen 端到端 **training** 导出中整理 2,048 条完整串行轨迹，包含 7,279 个
决策步骤与 2,521 个 API。训练覆盖计划、选择、参数、工具观察后的 continuation 和 Finish。
保留 128 条开发轨迹、110 个开发 API；训练与开发 API、规范化初始问题交集均为零。
这属于经过过滤的内部 pilot，不是官方 ToolBench I1/I2/I3 或最终测试面板。

每个 normalized query 只保留一条完整支持的因果路径。DFS 重试中间状态、未结束轨迹、
身份歧义、非法 JSON/参数和超长记录逐条登记，原始文件不修改。对保留轨迹的全部决策重新
施加监督，不复用上游按 DFS 状态拆分的 prefix loss 标记。

完整参数与目标不静默截断。工具参数名称遵循 ToolBench 的 `standardize/change_name`；
原名称和类型保留。Catalog 的 default 按 ToolBench 上游作为 example，不冒充 JSON
Schema default，也不凭自然语言猜 enum。原始工具返回中已有的上游截断继续标明。

单 seed 17，4 L20、microbatch 1、累积 2、global batch 8；最多 500 次更新、最多 1 epoch。
训练计算 3 小时后在同步边界导出，调度总时限 4 小时；这是适量训练，不是完整收敛主表。
独立全文 caller 按 Native 的实际更新次数比较；两个条件合计最多 32 L20 GPU-hours，
启动失败时间也计入。若触及时限导致更新次数不同，必须注明不匹配。

## 真正的端到端输入

推理问题文件只含：

```json
{"id":"episode-001","query":"用户的实际任务","split":"dev"}
```

不能带 gold API、调用参数、工具返回或答案。每个任务从同一完整开发 registry 选择工具，
执行器收到生成的 exact identity 和 JSON 参数，返回真实执行反馈；caller 再做下一次决定。
不能在预测错误时替换为 gold API，也不能以 gold 返回重放代替实际环境。

```bash
PYTHONPATH=src:/path/to/reviewed-executor python -m latent_register.run_toolbench_e2e \
  --checkpoint /path/to/trained-checkpoint \
  --tools /path/to/dev_tools.jsonl --queries /path/to/dev_queries.jsonl --split dev \
  --executor-factory senior_adapter:create_executor \
  --executor-config /path/to/private-backend-config.json \
  --max-calls 8 --max-thought-tokens 1024 --max-argument-tokens 1024 \
  --output-dir /path/to/NEW-e2e-episodes
```

生成预算显式覆盖训练面板中较长的计划和参数，不沿用旧接口的 128/512 默认值。
遇到上下文溢出或截断记录失败，不补造答案。

## 学长执行器的接入合同

```python
def create_executor(*, query_id, config, tool_bindings):
    # 每个 episode 初始化独立环境。config 由私有文件提供，不写入结果。
    # tool_bindings[exact_identity] 包含 catalog 的 category/product/tool/api 信息。
    def execute(exact_identity, arguments):
        # 调用已选定的实际执行器或 simulator，返回其真实结果。
        return ExecutionResult(content=actual_result, success=execution_success)
    return execute
```

没有内置默认服务，也不会自动调用外部 API。执行器适配代码需要固定版本；同一比较的
simulator、judge、registry、调用预算和失败处理必须一致。工厂不会收到 gold 工具或答案。

输出是完整 `episodes.jsonl`、资源与状态报告及来源哈希。执行成功回执只代表工具执行状态；
Finish/give_answer 只代表模型结束回答。最终任务成功需要独立评估，因此本入口不把它们
写成 SoPR，也不声称可直接比较不同 simulator 下的论文分数。

当前有两个合成 CPU 测试验证：执行返回确实进入下一步上下文，以及问题输入拒绝 gold 字段。
这些测试不是 8B 端到端结果。真实后台地址/配置、正式 evaluator 仍需接入。
