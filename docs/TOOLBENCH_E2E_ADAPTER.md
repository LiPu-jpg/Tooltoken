# ToolBench HTTP / ToolEval 适配

此路径使用已训练的 Native 模型，自主选择完整开发 registry 中的 API，并逐次接收执行器返回。
内部 Finish 身份在输出给评测器时映射为 `Finish`；其余名称一一对应实际工具，保留独立 identity 表。
没有给未结束或生成失败的轨迹补造 Finish，没有把 give_answer 当作成功分数。

HTTP 请求采用 ToolBench 的 category/tool_name/api_name/tool_input/strip/toolbench_key 合同。
输入类型不强制转换；工具名称遵循上游标准化，标准化后的端点碰撞会拒绝执行。
没有默认服务地址或密钥；没有自动重试、真实 API 回退或 gold 返回回放。
执行错误进入 caller 的 observation；HTTP 错误、无效响应与模型生成错误分开记录。

实际后台配置示例（占位值必须替换）：

```json
{
  "service_url": "http://YOUR-REVIEWED-SERVER:8080/virtual",
  "backend_kind": "stabletoolbench_virtual",
  "backend_revision": "PIN-THE-SERVER-AND-SIMULATOR-VERSION",
  "toolbench_key_env": "TOOLBENCH_KEY",
  "timeout_seconds": 120,
  "max_response_bytes": 1048576
}
```

密钥仅从指定环境变量读取，不写入请求证据或报告。无凭据本地服务需要显式设置
`allow_empty_toolbench_key: true`。loopback_contract_only 仅用于运输层诊断，不能通过端到端门禁。

`run_toolbench_e2e` 产生原始 episodes、实际请求回执、`evaluation_identity_map.json` 及
`tooleval_answers.json`。后者对接上游 ToolEval/StableToolEval 的 query、available_tools、
answer 格式。官方转换函数与 get_steps 的独立兼容验证使用 StableToolBench
`aa4ed9f4737ad98bd706663f01d63623c3427812`；没有修改上游源码，也没有启动 LLM judge。

本地验证包含 typed HTTP 往返、实际 observation 接续、HTTP/工具/响应协议错误、名字碰撞、
缺失凭据、原始生成失败保留、错误 observation 拒绝、Finish 规范化、与上游转换结果逐字段一致。
loopback 路由独立于外部 HTTP 代理；首轮代理导致的失败测试报告保留。

## 继续训练前的适配门禁

1. 固定可用的实际 executor/simulator、版本、工具目录、judge 及任务调用预算。
2. 在少量内部开发任务上，用真实 172-step checkpoint 跑自由生成与实际执行反馈。
3. 确认名字/参数/返回/Finish/最终答案运输正确，基础设施错误可辨认，评测器完成评分并留原始回执。
4. 低任务成功率本身不算适配失败；mock/loopback 通过、模型返回 Finish 或 CPU 测试通过均不能替代这一步。
5. 通过后方可启动 Native continuation。当前 export 不含旧优化器状态，必须明确登记为重置优化器的 continuation。
   后续保存完整训练状态并按目标更新数完成，时间限额只触发分段保存，不作为训练完成标准。

本开发面板来自原训练源中的 API 留出，不是官方固定可解测试集。未作可解性认证前不能命名为官方 SoPR；
评分器兼容不等于官方结果复现。固定单路径推理，最多 8 次工具调用；严格 JSON/Schema 失败终止当前路径，
不使用 DFS 搜索、自动参数修复或重复尝试。与其他系统比较时必须匹配这份推理预算。
