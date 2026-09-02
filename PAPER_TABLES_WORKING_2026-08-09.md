# ToolGen 与 Late-Bound 论文表格（工作草稿）

## 仅源码重置规则（2026-08-23）

本历史台账中此前所有 CoTools/ToolScalER clean-room 或自定义移植行均标为
`ARCHIVED_DIAGNOSTIC`，没有论文准入资格。新行只能来自 `PROTOCOL.md` 仅源码章节所述的
固定上游源码链和冻结 Native launcher。本规则不删除或改写下方历史审计证据。

状态：受证据约束的工作草稿，尚不能投稿。数值单元格只能从已打开制品填写。`PENDING`
表示声明的制品尚不存在或未通过审计；`BLOCKED` 表示当前输入或实现违反已声明的协议要求；
`N/A` 表示该方法未定义此操作，不能解释为零。

每张表仍需人工科学审核签字。

执行资源边界（2026-08-21）：新任务限一张 A100 或最多四张 L20。历史六张 A100 制品保留
原始资源标签，不能暗中改称四张 L20 或单张 A100 结果。详见 `PROTOCOL.md` 的算力边界。

## 当前表格目录（六张论文表）

Table 1 分为两个面板，使 STQ 主对比和修订后的 ToolBench Native-vs-CoTools 选择/迁移对比
同时保留，而不新增第七张表。本目录是唯一执行清单；下方详细表格只包含已审计数值或明确状态。

| 表格 | Benchmark / 面板 | 必需实验 | 当前状态 |
| --- | --- | --- | --- |
| 1A | STQ 未见 exact API 选择（面板 A） | 上游 CoTools、上游 ToolScalER、冻结 Native；使用相同声明划分和 seeds | `RESET_REQUIRED`；此前 clean-room/移植行已归档为诊断，不能使用 |
| 1B | ToolBench exact-API 选择和跨域迁移 | 上游 ToolScalER、上游对照入口和冻结 Native；只能使用源码兼容划分 | `RESET_REQUIRED`；旧 B1/B2 和 clean-room 行仅是历史来源，不是当前证据 |
| 2 | ToolBench 严格物理 token 身份 | 无碰撞 G1 身份清单和物理地址控制 | 被已审计的一对多碰撞 `BLOCKED`；禁止删除 qrel 或合并 |
| 3 | BFCL v4 下游调用 | Standard-Qwen 基线；学习方法行仅在 selector/caller 对齐并通过审计后填写 | `OPTIONAL`；当前仅有基线 |
| 4 | StableToolBench 端到端 | 付费 simulator/judge 和轨迹 adapter | `PAUSED`；禁止提交或调用 API |
| 5 | 注册成本与可扩展性 | Fixed、Incremental、Late-Bound 在 10/100/1K/10K/47K 规模及追加审计 | `AUDITED`，可使用 |
| 6 | 严格机制消融 | Table 1A 后仅做预注册消融；学习方法行使用三个 seeds | 可选后续；尚无数值行 |

因此，当前 benchmark 范围为 **STQ + ToolBench + 注册成本/可扩展性**。Tau2、
ContDa/MCPEvol、BM25/E5 和历史 BFCL 探索仍是诊断，不能填写 Table 1A 或 1B。

## 证据台账

| ID | 来源 | SHA-256 / 状态 | 范围 |
| --- | --- | --- | --- |
| E001 | `paper-evidence/sources/TOOLGEN_LATEBOUND_HANDOFF_2026-08-09.md` | `4289d1b8df51b4da0d14b2f5d068b03f9f5d5a5c2d92f60b63bd7eb664335053` | 原始六表设计与主张规则；文档整理后仍作为哈希完全一致的来源证据保留 |
| E002 | `paper-evidence/a100/toolgen-official-retrieval-aggregate-v2.json` | `9f4de3d70e57addfeafa0a9e984c0268f9050dadf7225e9b8d65c43635aff912` | 完整的官方 ToolGen 发布 checkpoint 检索聚合 |
| E003 | A100 已审计受控 Qwen 替代摘要 | `PENDING`；历史六 A100 链：不可变 v2 DAG `1662469` 至审计尾任务 `1662489`；提交清单 `697638faec14cffdf27f807c4fc6f0688bdab90afc55e218b070bbf38db65365`；原 `1659232` 保留但不可用 | 次要共同 Agent caller 控制。当前单 A100/四 L20 边界无法执行六 A100；任何替代都必须是有独立预检的新资源受限分支，且不能称为原 E003 结果。它不阻塞 STQ 选择证据。 |
| E004 | `paper-evidence/a100/toolgen-g1-collision-audit-1661134/collision_audit.json` | `b70b860f7d5f843c5f6ad0d8295c8ed68f472d5840a6d19d5cee28bce20f5185`；审计完成且 `passed=false` | 严格官方 G1 物理身份兼容性阻塞：49 个碰撞 key、157 个文档身份，13 个查询中的 20 行正 qrel 受影响 |
| E005 | L20 失败 v3 `tau2-domain-matrix-60941`；基于官方 Tau2 `v1.0.1` 的不可变 wrapper 快照 v4 array `60968` | v3 永久不完整且无聚合资格；v4 airline `60968_0` 完成 `0:0` 并独立通过，制品树 SHA-256 `d4d4341845596cd9c660253a7c9f6752437b9b1bc3e00b13459d9a84207d8808`；telecom `60968_2` 完成 `0:0`，保存审计与独立重算完全一致，domain audit SHA-256 `6b9a4e8e1cf3568c9be0e3b4d28ccf7731638f71087688644884e147962ca0fc`，源码目录 SHA-256 `506c218beeb7d1225c77c93dfb417e6fbf427edc8f6e5085b21df1f170cdd089`，22 文件制品树 SHA-256 `7c60b1ce7e35c63588dbc3a1a28831f523f9fa4c1ac3dbb2142fb077e425b2c6`；retail `60968_1` 失败 `1:0`，未打开其部分制品；wrapper SHA-256 `eb3a5f2254459bff1f2a85e7f529266a42a4ec3bdbff45fa63219905bb30fe14`；v4 永久不完整且禁止聚合；仅探索 | Tau2 补充集成，绝非主表证据 |
| E006 | `paper-evidence/l20/bfcl-qwen-standard-60959/table3_row.json` | 行 `794b2e510408d3f3d48815997fa8eb59e4f656f20667690cd318688affde762c`；本地目录 `3d4f36fd3b0148f59be2edcb23fce2fab92eef15e8cf8848381dd1a6a3473ea3`；聚合审计 `0e8b15e06d025df7d0acfd44e942cdf44fe7e14db13a5105c36c4dd742cac2d3`；聚合目录 `0836456149cd0ea67481bf478c78648fb88967877fa1dde7f7959d8b780a2f17`；`COMPLETE` `3fc9b0b84f81086c7397fd37600cafdefdad1456d90b69323f70180c658e8767` | 声明的 1,800 样本八类别 AST 子集上的完整受控 `Qwen/Qwen3-8B-FC` 结果，基础设施错误为零；不是 BFCL 官方排行榜 Overall 数据群体 |
| E007 | `paper-evidence/a100/no-rebinding-seed-17-1661221/` | 调度器 `COMPLETED 0:0`；收集目录 `d4d88c5a1775aed83e70250c714c1dc13bad81d46cfcd5e6f06e905927de1c57`；远端验证输入目录 `0c3777f9414fad16325d437bd9c58730784b17f42e09a88a90125dd0918c9aa1`；输出目录 `f0e5f4192f168c12e8a812760dfbc000541ca163024dcf330c291d944301357b`；无轮换审计 `a36410dc9334e0b15d7a3db7a8c6d76bd70ccaab01e1e2a7dc5de5d4a89a7ce4` | 仅审计完整方法 seed 17 训练：准确 1,000/3,000 步、world size 6、结果有限、8,192 个原子/不同/连续预留 ID、评测暴露和行改动为零，并满足无地址/固定占位合同。不满足三 seed 聚合要求，不填写数值单元格。 |
| E008 | `paper-evidence/l20/table5-latebound-61001/` | array `61001` 与依赖聚合 `61006` 均为 `COMPLETED 0:0`；收集目录 `794f19ac28b8c553a6f9db7b6d28c967bd5a46f3a750d91e59fb027644eadd90`；摘要 `daac60675df0daeca2136515366ab59f12311fda8915567ed32a4ccfdac52792`；源码目录 `ac2a2a690849862a650e1ae637e6fff121f2ac9dc0e110f837d06cfb8e604183` | 已审计共同 L20 Table 5 Late-Bound 运行时行，覆盖 10/100/1K/10K/47K registry；五个任务目录、任务审计、`COMPLETE`、共同运行时身份和聚合哈希均独立验证 |
| E009 | `paper-evidence/l20/table5-fixed-61022/` | array `61022` 与依赖聚合 `61023` 均为 `COMPLETED 0:0`；收集目录 `8c729de666d310da363e6fcce2c6286c3633a6706fc90e6db9ebe747db5a8999`；摘要 `a96489dcd95ba7f5804245888c0d4ce0fc3e55eeebdc1cb1611a3e7ef3aed711`；源码目录 `35da3f8854842028f5a7264741f0fd4e688ad7cb9e5c1a8487481f24dc73950e` | 已审计共同 L20 Table 5 ToolGen-Fixed 运行时行，覆盖 10/100/1K/10K/47K registry；五个任务目录、任务审计、`COMPLETE`、共同运行时身份和聚合哈希均独立验证 |
| E010 | `paper-evidence/a100/toolgen-table1-latebound-compatibility/compatibility_audit.json` | `f8e74edcd39a2eebc456f5050bb9e516152dc4a0decba1006e595449ce3c3f0c`；制品目录 `88ea681bf0b556929d72bccca5a25a492d0f52dc8e050cc83c3c588d49d99102` | 从 E004 精确输入派生的已审计 Table 1 G1 指令兼容性阻塞：9 个发布候选 token 映射到多个不同完整文档，影响 6 个查询的 12 行正 qrel |
| E011 | `paper-evidence/l20/table6-bfcl-full-seed-17-61158/` | 收集目录 `f33d80f184426bb6467272822cb428c8506fd980e6e1219288462c64775acf1c`；收集审计 `f80ac57b793a8ab4e86a2827be51e334816ab717aeb2624eb35e50748462177c`；条件审计 `fb8d7475a78118756a005414d26b0dc77b2d2b438d3c6240ec079c8d9ccc0b59`；1,800 样本指标 `9383583901bc65a731679a18c6e918f4a22e5ba785693932a824700b201d55c3` | 仅已审计单 seed 诊断，明确为 `formal_evidence_eligible=false`：v10 官方聚合基础设施错误为零，但暴露 result adapter 缺陷；已生成的 arguments-only JSON 因所选 registry 工具名未放入官方调用 envelope 而被丢弃。不填写 Table 6。 |
| E012 | `paper-evidence/l20/table6-call-binding-smoke-v11-61173/` | 收集目录 `a65de4e6fafede617517df8ad4d0481ae1ab277f46cbb08cde71776eef944277`；绑定审计 `17660b7bd5d30e2cbb40eae553fd79e0f709d2b33e8012d0dc1a41de735c0889`；远端制品目录 `a01a4254e91a9bb9fc76ffe87564bcc726224bbd94f67af389373bcae8b2f50d` | 仅诊断的官方 `simple_python` 回归：400 个样本，其中 301 个 arguments-only 响应使用已选 registry 名称绑定，官方准确率 `0.7075`，未使用参考名称/参数/数量。明确不是论文证据，不填写数值。 |
| E013 | `paper-evidence/l20/table6-bfcl-registration-full-seed29-v11-61183/` | 收集目录 `39ff86318c09fa0401d2ce627caba8e456918f4c4288c931323aaa0b1e7e695f`；注册审计 `a1a8fccee79b9eeef5bb648b273668ab07b734dd4826a83b516ffbbed48cdb32`；远端制品目录 `ab47fb9d0e7c6c3d2b4de3ebcfee73546664362c3e511ea13cab6a70429762f5` | 已审计 seed-29 v11 注册 cache：1,315 个工具/前向，优化器/参数/静态行改动为零，八个 memory slot。完整生成/聚合证据为 E014；三 seed 联合聚合仍待完成。 |
| E014 | `paper-evidence/l20/table6-bfcl-full-seed-29-61217/` | 调度器 `COMPLETED 0:0`；收集目录 `f62e0b8d3a5c474f1e5a188fbc9b954bd7afe10041fdb986bd61bf0b089dde9a`；收集审计 `7fd8846fbc21ae9fedf31ec714adfd268e9dbbb844dee3b5409d1af7f8bda5d1`；条件审计 `f098c2552f3c749be6b19005fdb8c183060a5f295a8c031fb3c2217ada508e3a`；恢复 gate 目录 `4073c350852c83fdc8b61bec4c07dc5b059bab607428ce25cf5de3da6de173b4` | 已审计 v11/recovery-v3 完整方法 seed-29 BFCL 条件，覆盖 1,800 个样本，基础设施错误为零。明确为 `formal_evidence_eligible=false`；单 seed 不能填写 Table 6。 |
| E015 | `paper-evidence/l20/table6-bfcl-registration-full-seed17-v11-61220/` | 调度器 `COMPLETED 0:0`；收集目录 `f73dc80fdf9132883db3649b879941629aa8f14a7771eba32ce860866a7a2916`；注册审计 `f3b74d07b4c4496a998c33baba6a747b443462b4a2c93094bf0bdd2333fc2edb` | 已审计 seed-17 v11 注册 cache：1,315 个单次前向工具、互不相同的未暴露地址、八个 memory slot、优化器/参数/静态行改动为零，且未读取 BFCL 评测字段。完整生成/聚合证据为 E019；三 seed 联合仍待完成。 |
| E016 | `paper-evidence/a100/toolgen-table1-latebound-grouped-recovery-v2-1665749/` | gate/评测/聚合 `1665746 -> 1665747/1665748 -> 1665749` 均为 `COMPLETED 0:0`；收集目录 `099a5d98bb1aa6a609bba96c9c337d26eae0168dd4b6a5756f437b17d07721d5`；摘要 `a741f37ea572faa9ba48015f277ab1416f017ba977d01382c3ea3841a057848d`；收集审计 `36fb6573a294e8cf6b4af3ea6feecc2fd3755c3b3104b6e859df6dc41fe2b3f0`；提交事件 `0c8fbe1f5165b604274dad13338a7b7f7a29ec11b5aa9c6ff8d3a2330f3d81ac` | 已审计三 seed、仅 Table 1 的工具受限检索，准确覆盖 46,732 个发布 token 组并保留全部 qrel。独立重聚合、调度器和校验和审计通过。无 Table 2 资格，并明确禁止整体可行性主张。 |
| E017 | `paper-evidence/a100/bfcl-incremental-training-1661227/` | 调度器 `COMPLETED 0:0`；收集目录 `a839d585807a668fe60c15cfde628e5bc7e8d71047373b44fdfc93a5c3c36e0e`；运行时 `3d03ac5b06e22242efb18b9b12cd883425a0455abbc1a5a792629e183b1729d6`；训练审计 `a09e615e06ee7cb3da11c9625d4aa127133c489aefcb2ad18a5af0d56029130a`；checkpoint 审计 `6707460659647eb8218f437bec426e201e7d3b1d528823afad925c4c8aa15b01` | 已审计对 1,315 份 BFCL 工具文档的 Incremental ToolGen 纯文档适配：1,282 s、16 个优化器步、六张 A100、精确保留 51,895-token 基础前缀、完整重载 53,210-token，查询/答案/调用/参数/奖励暴露为零。 |
| E018 | `paper-evidence/l20/table5-incremental-61296/` | array `61296` 与依赖聚合 `61297` 完成 `0:0`；收集目录 `3a65ba3421665ea7821ee8b18cfe2e4f35f84dc5d2fcc3f831abd4eae8f93b54`；摘要 `b787655f8c30dda821e22894050ac62241433f31c020c2985c18d37dee1b1487`；部署源码目录 `2d6049e908acbbae9f13f13d196a7c53cd5b6c07c18253a534b246aa49a45c1f` | 已审计共同 L20 Incremental ToolGen v3 运行时，覆盖 10/100/1K/10K/47K registry；仅使用 base-train 与 BFCL-incremental 候选，并排除不可用的 controlled-test token。 |
| E019 | `paper-evidence/l20/table6-bfcl-full-seed-17-61222/` | 调度链 `61220 -> 61221_[0-7%4] -> 61222` 完成 `0:0`；收集目录 `90323e102754e8629a6c826509ba4c4d1e8ea2327470747444d89ed82ecbdc46`；收集审计 `1ed2ed34855761cb936683489a2c05ff3342557bfeee180ef816b826e1829c19`；条件审计 `f46a945d91f0e21a2415622d53ae12e074871e195f92b2d0150ef4a071b2a06d` | 已审计 v11/recovery-v3 完整方法 seed-17 BFCL 条件，覆盖 1,800 个样本。明确为 `formal_evidence_eligible=false`；单 seed 不能填写 Table 6。 |
| E020 | `paper-evidence/l20/table5-incremental-delta-v4-61474/` | gate `61473` 和审计 `61474` 均为 `COMPLETED 0:0`；收集目录 `374747fc5f656f4c1a9e0073854001ca5aa64e88e7f4f3e9850f0e55a55fb38c`；delta 审计 `0ec21f8488312c1df305bc5022dbc2d352e8eab48e7b940033e069979bd1cc2e` | 已审计 Fixed 到 Incremental 的精确 checkpoint 对比：5,563,348,715 个改变或新增的存储标量；完整 checkpoint 17,290,840,884 B；checkpoint delta 22,310,265 B；CPU/CUDA 精确比较一致；仅为 Table 5 训练成本证据 |
| E026 | `paper-evidence/a100/table5-fixed-training-1659160-1659267/` 和 `paper-evidence/l20/table5-fixed-delta-v4-61482/` | 训练目录 `b590b824000a0ac2fc81faa76fc746c78bd8061ea98d3b61ee2c91be5cae282f`；训练审计 `252f4911c7d20b989edfaf6b45a02b9ed6b2a484e17245adde602374e992f9cb`；新 gate `61481` 与依赖审计 `61482` 完成 `0:0`；delta 目录 `923f65e295f6246b9c39a7bc9047da9298c4394c8be37a3d1352bf2ebe56da15`；delta 审计 `0df642549e575b19f54bafb6631c89a6621fd85d9725718ae7f7635f14848a0e` | 已审计成功的三阶段 Fixed 训练路径：88,113.1357 s、892 个优化器步、51,895 个工具；7,203,140,330 个改变或新增的存储标量；完整 checkpoint 17,268,530,619 B。已取消主 Agent 尝试的 7 条优化器记录属于恢复开销，不计入成功路径时间和步数。 |
| E027 | `paper-evidence/l20/table5-sequential-append-61494/` | 评测 `61493` 与依赖审计 `61494` 完成 `0:0`；收集目录 `2d8b0fa76c193ee2b7353608b8100de2d1f8d523fb67157655c2fb95c3e120c4`；追加审计 `cfe9d55cba2ec72f55039497028f05ad59d117c29fb788aa8b89c38d20145fb0`；对比 `f7fb4b7afb7613c5a5d3a80390047a61fd5395d9e5ad8d7d295fdaf324516fc7` | 同一 64 个样本上的完整 100 到 1K 顺序追加审计；旧 identity、logical slot 和 physical address 前缀不变 |
| E028 | `paper-evidence/a100/table1-qwen-fixed-1663464/` | 原链 `1661314 -> 1661315 -> 1661316_[0-2] -> 1661317`，聚合 `1661317` 失败；不可变恢复 gate/聚合 `1663463 -> 1663464` 完成 `0:0`；收集目录 `d2fbdf1beecabbaa9fac7ae583aaf0d430adafb09399aa7ce358e4d4a9bb0ad6`；Table 1 行 `70dc182a1294267835b8b622f1d524707f5f55acc6853c0d3e01b2584765969b`；调度器 `9ff4e95684180b9c1060800d142635c643811767d44519bc646c912a3448acd0`；恢复提交事件 `e60079f667ff5a325656d9bbf92dcd499de74ba9ce48b8ace5b6a14246ccc540` | 已审计 Qwen3-8B ToolGen-Fixed 检索行，覆盖完整官方 G1/G2/G3 指令群体和全部 46,732 个可达发布 token 候选。保留失败的原聚合；恢复仅复用已完成预测 array，并独立验证全部 checkpoint/输入/评测器绑定。 |
| E029 | `paper-evidence/diagnostics/simpletoolquestions-native-stq-matched-v2-recovery-v10-a100/` | 精简 `COLLECTED.sha256` `195f38c0a8e6dd4ae8848eb846f8285b9aa6da75e51adcb387506a360e69cef5`；seed-17 审计 `49448855d3ab61e48ecdeb1563da23e7080dc96e0dde38baa86773790b80189e`；权威文件 `94646c112a8a22781166f4c750563956c21cc342cbe5c5e5fb2523d1a7101c0e` | 通过 A100 v10 STQ 决策 gate：Hit@1 `28.61%`、Hit@3 `52.16%`、Hit@5 `62.48%`、MRR `43.68%`、最大 Top-1 hub `0.844%`、837 个候选、1,066 个查询、未见工具优化器步为零并通过冻结 Qwen 审计。仅在验证远端目录后从精简收集中排除大型 cache tensor；尚不填写论文单元格。 |
| E030 | v4 L20 STQ v10 提交 `66704 -> 66709` 和 `66710 -> 66715` | v4 源码目录 `ea9f2bfb7765432eadc23a1cc456d9eceb2d691ba0e56ddd58ff51ee3649413f` 与 `21097ab5e73c2ac415e3e450a382de92a9b03278ee06a5331e6a3abcba565a10`；两个训练任务均在优化器步前因标量摘要序列化失败 `1:0`；未打开分数/checkpoint | 保留不可变失败；全部后继仍被原 `afterok` 阻塞。 |
| E031 | v5 L20 STQ v10 恢复提交 `66736 -> 66741` 和 `66742 -> 66747` | 源码目录 `bffc380a0bbd645fd7f068df5ff08df6d80d3f688b2cbf414f2e21525be813de` 与 `dac05f56904808ddc1136015d8cf6585cbe5bf723aff8af306af3202a0c5f3a6`；不可变归档 `e2d9d3580de0bf8d326b1e62d0fff6881e9465ef6431f4023ff27784695233ba` 与 `efe91488c29d229dc58a4b4812d00fe58b1e5465db9ee5c0c05e22fa063523f9`；提交事件 `l20-stq-matched-v2-v10-seed29-v5-submitted.json` 和 `l20-stq-matched-v2-v10-seed43-v5-submitted.json` | 标量安全的纯摘要恢复；本地测试、远端源码校验和、compileall 和 Slurm `--test-only` 通过。等待终态链；尚无 Table 1A 单元格。 |
| E032 | v5 精简收集 `seed29/43` | 收集 SHA-256 `f5273fece4d0843d4d2df0f0ae52f04ddfd2599c54de0ae23c2e00c288494d18` 与 `af86cfa5c84d5201edd55536f6c9951a2aecc0e116e714386af204bc7e0a97a5`；审计通过，两者 Hit@3 均为 `0.281%`，最大 Top-1 hub `1.000` / `0.462` | 仅运行时不匹配诊断：v5 使用旧 replay 路径，而非 A100 v10 完整前向 tap。不是 Table 1 证据。 |
| E033 | v6 L20 STQ v10 提交 `66786 -> 66791` 和 `66792 -> 66797` | 源码目录 `8b054f269fe1196758d1b9ffccbc0a3a3a1147006ea4e2876864a63b341953c3` 与 `a5658714801f3e394798966232de6cf9eb0584ba4dfac12a9a576f91c46f3796`；不可变归档 `041402ef781b03bc31cb79597c4ba250acf104a908145fa56d01d4e2e9ff26d4` 与 `508359dc8432aee1df62a8a98a8c172c9db20d6cc49cb8ce89e41660084b80e9` | 在 L20 上精确运行 A100 v10 完整前向 `H24/H30/H36`；本地四测试合同、compileall、源码校验和及 Slurm `--test-only` 通过。等待终态审计；尚无 Table 1A 单元格。 |
| E034 | v6 精简收集与 Native 三 seed 聚合 | seed-29 收集 `a91d6b48389d51db9d1db16a938fdd4723e59fd54bbd6e0777bf7c873c032485`；seed-43 收集 `d49e2f2f546bfb5749899282b8ce677bcbd0e9221648745ab3429ff3a30af42b`；聚合 `paper-evidence/diagnostics/simpletoolquestions-native-stq-matched-v2-v10-three-seed-aggregate.json` | 独立聚合通过：合并 Hit@1/3/5 `28.99/53.03/63.41%`；seed-29/43 Hit@3 均为 `53.47%`；零步和冻结 Qwen 审计通过。同类面板经 E035/E036 完成；clean-room 同类制品仍非官方。 |
| E035 | `paper-evidence/diagnostics/stq-cotools-cleanroom-qwen-v1-66832/` 和 `paper-evidence/diagnostics/stq-same-class-native-vs-cotools-three-seed-20260821.json` | 终态任务 `66832` `COMPLETED 0:0`；精简收集 `0cdf571b347df59d389ff1b43e487c44c435f2a442386361790714801b713919`；CoTools 聚合 `3010aa85fa8cbf64e619e302a46523da31a8a2e810a749cdf923dfd3ee70336a`；共同配对聚合 `ba493b4773a140146526f664581244c2b31ca68f5ea1989eb31dc9ce3f1a81d1` | CoTools clean-room Qwen 三 seed 对照在 1,066 个查询和完整 837-way registry 上通过：宏平均 Hit@1/3/5 `22.64/42.59/52.47%`，MRR `36.83%`。Native 减 CoTools Hit@3 为 `+10.44` 个百分点；分层配对 bootstrap 95% CI `[+6.20,+14.76]`。未见工具优化器步为零。已审计 clean-room 证据，不是官方上游结果。 |
| E036 | `paper-evidence/diagnostics/stq-toolscaler-cleanroom-qwen-v1-66887/` 和 `ops/events/l20-stq-toolscaler-cleanroom-v1-submitted.json` | 所有 `66877–66887` 根任务完成 `0:0`；源码目录 `a14f0602fea5f1b10f31f2076d3c9a3b3b1f6be947895937f8b7ba0e0f133116`；收集目录 `39c16ba92ab4e6073d2eea6eca6cfd07935007bbfd44a155c4c966b79d6df50e`；共同聚合 `29d1ee645d67415dc163cea58f76b4c400ce8b2563a0d082383599cddac29463` | 三 seed 严格 ToolScalER Algorithm-1 clean-room 对照已完整审计：宏平均 Hit@1/3/5 `6.88/14.60/19.36%`、MRR `13.50%`、最大 Top-1 hub `9.66%`、未见工具优化器步为零。负面证据；制品标记仍为 `formal_evidence_eligible=false`。 |
| E037 | `paper-evidence/diagnostics/stq-registration-timing-v2-66991/` 和 `ops/events/l20-stq-registration-timing-v2-submitted.json` | gate/计时 `66990 -> 66991` 完成 `0:0`；源码目录 `3466eefbc61ebe2e12f6cb9147eb5a8acf2ca01d9cb9f19a61c3e8774d12ce1c`；部署归档 `9b57d09f82b7418ecb2778c1db8f4689b96e5a2619c7ff2574b323a773e30273`；收集目录 `3548519850f6401d17307797df3287c771dc6f8436e179a9252571026f6d04e2` | 共同 837 个未见 API 上的纯计时注册审计：Native seeds 29/43 平均 `12.945 s`（`15.466 ms/tool`），CoTools seeds 17/29/43 平均 `12.650 s`（`15.113 ms/tool`）；每次运行 837 次文档前向、零优化器步、不接触 qrels/分数、无参数改动。仅披露成本；`formal_evidence_eligible=false`。 |
| E038 | `paper-evidence/diagnostics/table1a-reaudit-20260821.json` | 制品 SHA-256 `6043228f8adf5d897ec4aabb2bfa55e2271be9f061eb9bca3dfc6ce2646f9cff`；`resource_profile=cpu-local-audit`；清单目录 `7af73147532dde8693d05a0b6e06ac1862fe13364e2d6804cac20e3b7d1ce16b`；三 seed、10,000 次分层 bootstrap | 对完整 STQ 同类面板独立失败关闭式重聚合。Native/CoTools/ToolScalER 宏平均 Hit@3 `53.03/42.59/14.60%`；九个分数根全部通过校验和/形状/顺序审计后才打开 qrels；clean-room 同类结果仍为 `formal_evidence_eligible=false`。 |
| E039 | `paper-evidence/diagnostics/table1b-toolbench-exactapi-prep-20260821-v3/` | 根目录 `2bd49d3701b7096fabc1e1a47baa59e541478ab1c268bf4857198d74e12aaf0d`；训练根 `8b57c8e8a7fe277f837695d93c19b1b30afbc797704ea41d002f0094e5b6f8bd`；评测根 `8b321775a48c7f7435addf83c049045d16cc5c493ac41c5635f45fe9755bd662` | 纯 CPU exact-API 准备通过：保留 46,840 个候选、12,752 个有 qrel 覆盖的身份、互斥 1,024/1,024 目标划分、4,096/1,024 个查询、ToolBench/STQ 名称对重合为零、144 个碰撞标签及明确记录的 4,712 个排除项。仅输入制品；B1/B2 分数仍 `PENDING`。 |
| E040 | `paper-evidence/diagnostics/table1b-toolbench-native-b1-67755/` | 聚合目录 `de06a795f09accbd4051d96ac4ef4d997c0a9bde52d7d34033fb72efad51dd04`；聚合 `aggregate.json` SHA-256 `e3b6f519a622b36480c52ae3901e602c79377f43d147fdbc92bab9f5e4138d37`；Native 审计目录 `b7a9c2b2661f645e54f95fbb119fccd36e8e5b5049bcdb2b64c1530d5b61e601`、`badd48f2b70320f1e7562ead51388a2a8ce2b1c7d73b64aa7a68cb44629ed3e0`、`cc3ad946a8684c24bf4f3f62f214d7a2f5fc937e4d3794ac2dc9fb37a30c0e01`；CoTools 审计目录 `4e8448d393bca69c0305bb7f9776eb9553f2a74a59a6b22b00adcde867ff5b28`、`62f6eb690d185637481a2a92bdf69b7291001076f439a96d7a22746c6d909efd`、`928e581c368da19b708639bee58f4ae6a5330fcca28363db5a15e61fee5787a8` | 已审计 B1 Native-vs-CoTools exact-API ToolBench 面板：46,840 个候选、1,024 个查询、三个 seeds，仅在依赖审计中打开 qrels，`passed=true`、`formal_evidence_eligible=true`。合并 Native Hit@1/3/5/MRR `1.2044/3.1250/4.2969/3.4361%`；CoTools `0.0977/0.3255/0.7161/0.6719%`。 |
| E041 | `paper-evidence/diagnostics/table1b-toolbench-native-b1-tiered-67755/` | 规模聚合 SHA-256：10 `33f8fc5ceff1b958061f2dbde85e956177a51874f447c7a5f146d65a721728f6`，100 `9c87f012a4d95845bbf6ea98e6bf68ac9bf2e3f0d5c4d81057cce9a1585cd2c1`，1K `b9687efda571513fb090439e6295fb4f15387b19cba518d848bd8a98b74c625a`，10K `d0704c4a792d98d4043ecd8e1b47dc7c0b874671df023503871233ea7e2bbad1`，完整 `e3b6f519a622b36480c52ae3901e602c79377f43d147fdbc92bab9f5e4138d37` | 从六个 E040 审计根派生的分级聚合；使用相同 1,024 个查询、三个 seeds、exact qrels 和嵌套候选前缀。五个规模目录均独立通过。不是新训练；作为规模分析有正式证据资格。 |
| E042 | `paper-evidence/diagnostics/table1b-toolbench-native-b1-1688051/` | 聚合目录 `a1db99a8631b0920d2eefc30c3299d966289209e110652b33878bd5540b9f69e`；聚合 `artifact.sha256` SHA-256 `2a47931c908e40406f36488517c3ac267809e89eab151d32e8abd199887b3bc6`；A100 v2 归档 `e867215fc1232c409fb25dbb8f9d1fd4379b32dba2fd31f7469b480ca5453dfe` | 已审计 B1 exact-API 面板的独立三 seed 重复：Native Hit@1/3/5/MRR `1.3020/3.2227/4.2969/3.5027%`；CoTools `0.1302/0.3255/0.6836/0.6671%`。Registry 46,840，1,024 个评测查询，仅在依赖审计中打开 qrels，七个根均通过校验和。具正式资格的来源证据；Table 1B-A 主行仍采用 E040。 |
| E043 | `paper-evidence/diagnostics/toolscaler-qwen3-1k-retrieval-v18-predict-recovery-v4/` | 本地收集 `fb223d109bdc0e39aff5915b3ca835e5422be91bba869659ed4889b364764021`；终态摘要 `a202ce4798cc3619f407f515bf93cda50391d8f64e9e16aeb4710b87e192a214`；作业 `74187 -> 74188_[0-3] -> 74189` 全部 `COMPLETED 0:0` | 已修复并审计 ToolScalER Qwen3-8B 1K 闭集预测：1,951 query、1,000 API，code/失碰撞关闭 exact-API Hit@3 `95.079/94.669%`，MRR@5 `92.997%`；四 shard、模型完整重载和终态 checksum 均通过。`formal_evidence_eligible=false`，不填写 Table 1/2。 |
| E044 | `paper-evidence/l20/native-i1-v8-tailbatchfix-77926/` | audit summary `0818ef4ad2901e43d45709bbd13893c4a601aebfe04d7c2b2c7478d03bbc4336`；score manifest `d780c5d34eef1a1fc1dc027cab446b4629545de9bc3edacf68f5a99990829132`；stage2 metadata `afed4d25d0d77687043cea1bd64b3725b01d81a4a098a9b23db78d127ece8cc9`；reload audit `57839dba4eee083b6e5d763adb96dd78900727e0808ae03114164743899e497c`；deployment SOURCE `90759f88533334c0a33f6a123c793e231b51ab26b3c445d0af6290084d46bf50` | Native Late-Bound 官方 ToolBench G1/I1 单 seed 正式链：1,000 训练 API、10,439 未见候选文档、1,335 query rows（600 qrel queries），Stage 2 1,265 步全参数训练，注册 optimizer steps=0；官方 NDCG@1/3/5 `17.0787/22.5669/26.1148%`，exact/action Hit@1/3/5 `17.0787/28.9888/35.2809%`，MRR `23.4494%`。仅填 I1 单 seed 检查行，不代表完整 G1/G2/G3 三 seed。 |

## Table 1A：STQ 未见 exact API 选择（%）

数据群体：感知来源的 SimpleToolQuestions 匹配划分，包含 999 个已见训练 API、1,707 个
已见开发查询、837 个未见测试 API 和 1,066 个测试查询。每种方法对完整 837-way 候选
registry 排序。主面板使用 exact API identity、Hit@1/3/5 和 MRR；全部学习方法行需要
独立 seeds 17、29、43、相同评测器及配对不确定性。BM25/E5 和 ContDa/MCPEvol 不进入
本面板。只有不接触分数的制品、qrel 边界、校验和目录和独立重聚合全部通过后才能填分。

| 方法 | 面板 | Hit@1 | Hit@3 | Hit@5 | MRR | 零步注册 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| CoTools clean-room Qwen | 零步同类 | 22.64 | 42.59 | 52.47 | 36.83 | 是 | 完整三 seed 共同审计；clean-room 对照，制品标记 `formal_evidence_eligible=false` |
| ToolScalER 严格 Algorithm 1 | 零步同类 | 6.88 | 14.60 | 19.36 | 13.50 | 是 | 完整三 seed 分数/qrel/身份审计；clean-room 负面对照，制品标记 `formal_evidence_eligible=false` |
| 我们：Native Late-Bound | 零步主方法 | 28.99 | 53.03 | 63.45 | 44.06 | 是 | 完整三 seed 共同审计；Native 减 ToolScalER Hit@3 CI 为 `[+34.71,+42.08]` 个百分点 |
| ToolGen-Fixed | 可选面板 B 成本基线 | N/A | N/A | N/A | N/A | 否 | 零步面板 A 主张不需要；当前未授权忠实未见工具适配协议 |
| Incremental ToolGen | 可选面板 B 成本基线 | N/A | N/A | N/A | N/A | 否 | 零步面板 A 主张不需要；需单独预注册新工具优化器预算 |

面板 A 三种方法均具有完整且通过校验和的 1,066 x 837 分数根、独立 qrel 审计及两次共同
重聚合，其中包括独立 E038 制品。Native 宏平均 Hit@3 为 `53.03%`，CoTools 为 `42.59%`，
ToolScalER 为 `14.60%`。分层配对 bootstrap 95% 区间：Native 减 CoTools 为
`[+6.20,+14.76]`，Native 减 ToolScalER 为 `[+34.71,+42.08]` 个百分点。两个 clean-room
同类制品仍明确不是官方结果（`formal_evidence_eligible=false`），禁止称为上游 benchmark 复现。

协议中保留 ToolGen-Fixed 和 Incremental ToolGen 作为可选面板 B 成本基线，但它们不是
面板 A 的缺失行。其 N/A 不能解释为零或实验失败。运行它们需要另行声明完全未见 API 如何
接受优化器更新，这与训练后零步注册是不同问题。

### Table 1A 算力披露

以下为已审计忠实运行成本，不是等 wall time 匹配成本。训练时间按 seed 报告，平均值仅作描述。
注册计时使用选择面板相同的 837 个未见 API registry。`not recorded` 表示制品证明零步注册
和参数不变，但不包含 wall-clock 测量；不表示零。

| 方法 | 每 seed 训练 wall time | 优化器步 | 可训练参数 | 未见工具注册前向次数 | 每 seed 注册 wall time | 每工具注册时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 我们：Native Late-Bound | 111.177 / 121.383 / 105.924 s（平均 112.828 s） | 656 | 11,032,582 | 837 | 13.748 / 12.141 s（平均 12.945 s；seeds 29/43） | 15.466 ms |
| CoTools clean-room Qwen | 225.314 / 254.893 / 222.643 s（平均 234.284 s） | 560 | 301,993,984 | 837 | 12.709 / 12.636 / 12.604 s（平均 12.650 s；seeds 17/29/43） | 15.113 ms |
| ToolScalER clean-room Qwen | 2,824.718 / 2,672.702 / 2,750.719 s（平均 2,749.380 s；compressor + codebook + retriever） | 1,068（84 + 984） | 15,355,904 compressor + 32,112,640 retriever | 837 | 11.762 / 11.748 / 11.681 s（平均 11.730 s） | 14.0 ms |

ToolScalER 计时包含单独审计的 CPU codebook 阶段，不含模型加载。Table 5 现有 47K registry
计时属于不同数据群体，不能替代此处结果。

## Table 1B：ToolBench exact-API 选择与跨域迁移（%）

本表有意不做 ToolGen 复现。它只比较所提 Native selector 与 CoTools 对照，使用与 Table 1A
相同的 exact-API Hit@1/3/5 和 MRR 评测器。单独的 ToolScalER 兼容子面板对每种方法使用
一份不接触 qrel 的排序，同时报告 action/code NDCG 和 exact-API Hit/MRR。本表不包含完整
文档 caller、参数生成或发布 token 碰撞分组。

### Table 1B-A：ToolBench 原生条件

两种方法在相同的无泄漏 ToolBench 训练划分上训练，并在相同留出 ToolBench 查询、exact API
qrels 和完整候选 registry 上评测。训练 seed、候选顺序、负例策略、checkpoint step 和评分
代码必须匹配。Native 与 CoTools 都通过相同的不接触 qrel、校验和绑定审计后，行才可填写。

| 方法 | 训练域 | 评测域 | Hit@1 | Hit@3 | Hit@5 | MRR | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 我们：Native | ToolBench | ToolBench | 1.2044 | 3.1250 | 4.2969 | 3.4361 | 已审计 E040；46,840 候选 registry、1,024 个查询、三个 seeds |
| CoTools clean-room Qwen | ToolBench | ToolBench | 0.0977 | 0.3255 | 0.7161 | 0.6719 | 已审计 E040；clean-room 对照，不是官方 checkpoint |

指标说明：E040 和独立重复 E042 是 exact-API B1 面板，因此只报告 Hit/MRR。较旧的 E041
NDCG 转换按 ToolScalER 兼容公式复用 E040 分数矩阵，但其划分不是官方 ToolScalER
G1/G2/G3，仍仅是诊断。只有 L20 G1/G2/G3 双指标链通过后，Table 1B-C 才报告官方 NDCG。

### Table 1B-C：官方 ToolScalER 划分的双指标视图

本子面板使用固定的 ToolScalER G1/G2/G3 指令划分和完整 47,152 候选目录。同一份不接触
qrel 的分数矩阵同时生成论文兼容 action/code NDCG 和严格 exact-API Hit/MRR。三个 seeds
和三个阶段全部通过独立校验和、身份、官方公式和配对 bootstrap 审计前，数值保持 `PENDING`。

| 方法 | 划分 | NDCG@1 | NDCG@3 | NDCG@5 | Exact Hit@1 | Exact Hit@3 | Exact Hit@5 | Exact MRR | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ToolScalER clean-room Qwen | G1/G2/G3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | Full-I123 v2 等待 G1 恢复事件 gate |
| 我们：Native full-I123 | G1/G2/G3 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | 全数据 Native 对照已准备；`afterok` 提交仍被 gate |

I1 单 seed 已先完成一条独立检查行（不是完整 G1/G2/G3 主行）：

| 方法 | 划分 | NDCG@1 | NDCG@3 | NDCG@5 | Exact Hit@1 | Exact Hit@3 | Exact Hit@5 | Exact MRR | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 我们：Native Late-Bound | G1/I1，seed-17 | 17.08 | 22.57 | 26.11 | 17.08 | 28.99 | 35.28 | 23.45 | 已审计 E044；单 seed 检查行，不替代三 seed G1/G2/G3 主表 |

外部 ToolScalER 论文参考（不复制进这些行）：G1/I1 的 NDCG@1/3/5 为
`93.00/93.87/94.85`，G2/I2 为 `90.50/92.26/93.68`，G3/I3 为
`89.00/88.16/91.98`。

### Table 1B-B：跨域迁移

主迁移方向固定为 **STQ 训练 -> ToolBench 评测**：冻结 Table 1A Native 与 CoTools
checkpoints，在同一 ToolBench query/qrel/registry 清单上评分，ToolBench 优化器步为零。
现有 1,000 查询 Native 和 CoTools 迁移制品因使用分组发布 token 身份，且尚非最终 exact-API
匹配面板，仅保留为探索诊断，不能填写这些单元格。

| 方法 | 训练域 | 评测域 | Hit@1 | Hit@3 | Hit@5 | MRR | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 我们：Native | STQ | ToolBench | PENDING | PENDING | PENDING | PENDING | 现有迁移仅探索；等待 exact-API 匹配重审计 |
| CoTools clean-room Qwen | STQ | ToolBench | PENDING | PENDING | PENDING | PENDING | 现有迁移仅探索；等待 exact-API 匹配重审计 |

历史 ToolGen/ToolGen-Fixed/Grouped-Late-Bound ToolBench NDCG 制品保留在证据台账和目录中
作为可复现来源，但只作附录背景，不是修订后 Table 1B 的行。

### Table 1B-A 规模分析（嵌套 registry 前缀，%）

分数矩阵与审计和 E040 完全相同。每行复用相同留出查询和 exact qrels，只将排序限制到
已审计确定性顺序中的前 `N` 个候选。这些是嵌套 registry 规模，不是独立训练运行。

| Registry 规模 | Native Hit@1 | Native Hit@3 | Native Hit@5 | Native MRR | CoTools Hit@1 | CoTools Hit@3 | CoTools Hit@5 | CoTools MRR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 74.61 | 94.04 | 96.97 | 84.57 | 62.27 | 90.04 | 96.52 | 77.03 |
| 100 | 44.63 | 68.00 | 76.01 | 58.82 | 17.45 | 41.54 | 55.92 | 34.72 |
| 1,000 | 14.65 | 30.70 | 39.68 | 26.79 | 3.84 | 9.24 | 14.10 | 10.36 |
| 10,000 | 3.58 | 8.11 | 11.23 | 8.54 | 0.55 | 1.69 | 2.67 | 2.21 |
| 46,840 | 1.20 | 3.13 | 4.30 | 3.44 | 0.10 | 0.33 | 0.72 | 0.67 |

## Table 2A：严格训练后物理 token Hit@1（%）

数据群体：官方 ToolBench G1 单工具检索，使用确定性匹配的 1K registry，普通 Qwen 词表
保留在分母中。完整表被 E004 阻塞：49 个发布 ToolGen 名称/token 各自映射到多个完整语料
文档，其中 12 个碰撞组影响 20 行正 qrel。审计明确禁止删除 qrels、合并文档、取最后一个
映射值或放宽一一对应物理身份。不得用探索性内部 benchmark 替代官方查询和 qrels。
[evidence:E001,E004]

| 方法 | 已见工具 + 已见地址 | 已见工具 + 从未暴露地址 | 未见工具 + 已见地址 | 未见工具 + 从未暴露地址 |
| --- | ---: | ---: | ---: | ---: |
| Qwen ToolGen-Fixed | BLOCKED | N/A | N/A | N/A |
| Incremental ToolGen | BLOCKED | BLOCKED | 除非明确定义，否则 N/A | BLOCKED |
| 我们：Late-Bound | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

## Table 2B：严格未见工具 + 从未暴露地址控制

除 MRR 外，指标均为查询级百分比。物理身份、零更新、静态行改动和自有文档检查是强制
有效性审计，不是准确率列。[evidence:E001,E004]

| 注册处理 | Hit@1 | MRR | Recall@5 | 普通 token 获胜率 |
| --- | ---: | ---: | ---: | ---: |
| 正确 Late-Bound 注册 | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 仅查询 | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 空白 | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 随机 | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 置换 | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| 最近已训练 token | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

## Table 3：BFCL v4 受控 AST 子集下游函数调用准确率（%）

数据群体：E006 声明的 1,800 样本、八类别 BFCL v4 AST 受控子集。这是下游 caller/执行表，
不是 Late-Bound 主选择表。必须处理全部给定候选文档；BFCL 查询、标签、调用、参数、答案和
奖励不得进入注册或优化。历史 BFCL 选择压力测试不进入本表。[evidence:E001,E006]

| 方法 | Simple | Multiple | Parallel | Parallel Multiple | Multi-turn | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen3-8B-FC 受控 1,800 样本子集 | 95.50 | 96.00 | 92.00 | 88.00 | 37.88 | 68.72 |
| Incremental ToolGen（仅文档） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 我们：Late-Bound 注册（下游 caller） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

Table 3 尚缺：在同一 BFCL 候选文档矩阵上的已审计 Incremental ToolGen 与 Late-Bound 运行，
以及完整评测器聚合。历史 Incremental 聚合 `61292` 被排除，因为它错误地将 selector
checkpoint 复用为 caller，因而没有输出可解析的 BFCL 调用 envelope；新 v12 链强制分离
selector 和冻结 caller checkpoint。`68.72%` 绑定 E006 的受控 1,800 样本八类别子集，
不是 BFCL 官方排行榜 Overall。

用户提供的官方 `Qwen3-8B Prompt` Overall 为 `40.43%`，基于另一套 Web Search / Memory /
Multi-Turn / Live AST / Non-Live AST / Irrelevance 数据群体；它与 E006 不可比，因此不能填入
本表制品派生单元格。本表不能用作 Late-Bound 选择结果；所需完整且身份审计通过的制品存在后，
选择结果应进入严格 registry 选择表。

## Table 4：StableToolBench 多轮端到端性能（%）

SoPR 和 SoWR 需要分别设置未见指令与未见工具面板。原生 ToolGen Agent 与受控共同 Qwen
Agent 必须在视觉上分开，因为其模型和 Agent 权重不同。[evidence:E001]

| 面板 / 方法 | I1 SoPR | I2 SoPR | I3 SoPR | 平均 SoPR | I1 SoWR | I2 SoWR | I3 SoWR | 平均 SoWR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原生：ToolGen 官方 Agent | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 受控：Qwen 标准完整文档 Agent | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 受控：Qwen ToolGen-Fixed | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 受控：Incremental ToolGen（仅文档） | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| 受控：我们，Late-Bound 注册 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

在 StableToolBench `planning -> acting -> calling` adapter 和轨迹 meta-registration 阶段
存在前，Late-Bound 行保持阻塞。未见工具面板必须使用 I1-Tool、I1-Category 和 I2-Category，
最终论文采用相同分组列。用户已暂停 Table 4；明确恢复前不得运行付费 simulator 或 judge API。

## Table 5：注册成本与可扩展性

时间和存储必须写明单位；延迟必须说明测量路径与分析群体。模型加载与注册分开报告。
[evidence:E001,E003,E017,E018,E020,E009,E026,E027]

| 方法 | 注册 wall time | 每工具时间 | 优化器步 | 改变参数数 | 每工具存储 | 延迟 @1K | 延迟 @10K | 延迟 @47K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ToolGen-Fixed 初始训练 | 88,113.136 s @51,895 | 1,697.912 ms/tool | 892 | 7,203,140,330 个标量值 | 332,759.045 B/tool | 37.323 ms/example | 18.650 ms/example | 18.584 ms/example |
| Incremental ToolGen 仅文档适配 | 1,282.000 s @1,315 | 974.905 ms/tool | 16 | 5,563,348,715 个标量值 | 13,148,928.429 B/tool | 40.119 ms/example | 19.240 ms/example | 19.301 ms/example |
| 我们：Late-Bound 注册 | 2,823.849 s @47K | 60.082 ms @47K | 0 | 0 | 147,456 B（144 KiB） | 34.964 ms/example | 36.744 ms/example | 53.762 ms/example |

其他必需审计字段：峰值显存、checkpoint 大小、文档前向次数、静态输入/输出行最大改动，以及
从 100 追加到 1K 后旧 identity-to-slot-to-physical-ID 前缀的精确保留。E008 的 47K 共享
registry 任务峰值分配 GPU 显存为 31,977,021,952 B（29.781 GiB），compiler 使用 735 次
文档 batch 前向并保证每工具一次前向，全部预留输入/输出行最大改动严格为零。报告的注册与
选择路径不含模型加载。E020/E026 提供完整 checkpoint 大小证据。E027 提供 100 到 1K
顺序追加审计：追加 900 个工具后对相同 64 个样本重新评分，每个旧 identity、logical slot
和 physical address 均不变。E008/E027 不填写或解除任何检索准确率表的阻塞。

E009 使用相同 L20 软件族，并报告每个 Fixed 任务峰值分配 GPU 显存为 22,846 MiB。其 1K 行
是逐样本匹配 registry，而 10K 和 47K 使用一个共享 registry；因此延迟保留各自声明范围，
不能解释为单调扩展曲线。E026 从已审计成功三阶段路径和精确 base-to-Fixed tensor 对比填写
五个 Fixed 训练成本单元格。每工具存储为完整可部署的 17,268,530,619-byte checkpoint 除以
51,895 个工具；另行报告的 checkpoint delta 为 871,069,353 B。已取消主 Agent 尝试的
7 条优化器记录仅披露为恢复开销，不得暗中混入成功路径数值。

E017 填写前三个 Incremental 训练成本字段，E018 独立填写 L20 1K/10K/47K 延迟单元格。
E020 比较已审计 Fixed 与 Incremental checkpoint 的每个存储 tensor 值，统计新词表参数并
填写剩余两个单元格。每工具存储使用完整可部署 17,290,840,884-byte Incremental checkpoint
除以准确的 1,315 个工具；另行报告的 checkpoint delta 为 22,310,265 B。Table 5 没有
任何数值从 BFCL 准确率推导。

## Table 6：严格条件消融

数据群体：严格未见工具 + 从未暴露地址样本。学习方法行需要独立 seeds 17、29、43；确定性
控制在相同样本上保持配对，不视为伪重复。[evidence:E001,E007,E014,E019]

| 变体 | 严格 Hit@1 | MRR | BFCL 参数 exact | BFCL 完整调用 exact | 普通 token 获胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整方法 | PENDING | PENDING | PENDING | PENDING | PENDING |
| 无共享 LoRA | PENDING | PENDING | PENDING | PENDING | PENDING |
| 仅 output compiler / 无 memory compiler | PENDING | PENDING | PENDING | PENDING | PENDING |
| 无 wrong-memory loss | PENDING | PENDING | PENDING | PENDING | PENDING |
| 使用仅 registry softmax，而非完整词表训练 | PENDING | PENDING | PENDING | PENDING | PENDING |
| 无轨迹 meta-training | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

完整方法 BFCL seeds 17 和 29 已完成各自官方条件聚合，但均仍为非正式联合输入。Seed 43 和
全部十二个非完整方法条件继续按依赖排在已完成且审计通过的前驱之后。最终十五条件 join 通过前，
Table 6 不得填写任何显示数值。

按 episode 随机物理地址重绑定不再是学习方法组件。选择训练与地址无关；readback 使用由身份
确定的占位符，并由生成 memory 完全替换。评测可以将一一对应 registry 绑定到任意从未暴露
地址。地址置换是有效性测试，不是 Table 6 准确率消融。

## 填表 gate

1. 在记录精确来源制品、哈希、分母、单位和分析群体前，禁止用数值替换 `PENDING`。
2. E003 必须通过自身制品检查，任何受控 Qwen 注册到执行的数值才可被描述为完整 pipeline
   可行的证据。它不是 Table 1A 中单独审计的 STQ 选择行的前提。
3. 除非满足官方评测器和正式 checkpoint 合同，否则 BFCL 和 Tau2 探索输出只能进入补充诊断。
4. 负面、空、失败和不确定结果必须保留并标注；禁止用任意准确率阈值删除行。
