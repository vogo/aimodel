# Anthropic Messages API 变更记录

本文件记录 **aimodel 的 Anthropic wrapper 与官方 Messages API 的同步情况**:官方哪一版发生了什么变更、wrapper 如何跟进。

- **官方协议**:Anthropic Messages API(`POST /v1/messages`)
- **官方文档**:https://platform.claude.com/docs/en/api/messages
- **实现说明**:[anthropic-chat-api.md](./anthropic-chat-api.md)
- **总账**:仓库根 [CHANGES.md](../../CHANGES.md)(含 OpenAI 侧条目)

**维护约定**:官方 API 变更时同步更新四处 —— wrapper 代码、README/CLAUDE.md、CHANGES.md、本文件。每条记录至少包含:日期、官方变更、wrapper 变更摘要。

倒序排列(新在前)。

---

## 2026-07-21 — `output_config`、usage 扩展、`container`/`inference_geo`、工具字段、未知块保留、profile 头

**官方变更**

自 2026-06-02 同步以来的六项新增,本次作为**同一个版本项**一起跟进:

1. 推理深度与结构化输出合并进单个 `output_config` 对象 —— `output_config.effort`(`low`/`medium`/`high`/`xhigh`/`max`)取代原顶层 `effort`,`output_config.format`(`{type:"json_schema", schema:…}`)取代已弃用的 `output_format`。
2. 响应 `usage` 新增 `output_tokens_details.thinking_tokens`(思考 token 成本)、`server_tool_use`(`{web_search_requests, web_fetch_requests}`)、`inference_geo`、`service_tier`。
3. 请求接受 `container`(复用服务端代码执行容器)与 `inference_geo`(数据驻留路由);响应与 `message_start` 返回 `container` 形如 `{id, expires_at}`。
4. `content[]` 中出现服务端工具块(`server_tool_use`、`web_search_tool_result`、`code_execution_tool_result` 等)与 `citations` 引用注解。
5. 工具定义接受 `type`(版本化内置工具,默认 `custom`)、`strict`、`defer_loading`、`allowed_callers`、`eager_input_streaming`、`input_examples`。
6. `anthropic-user-profile-id` 请求头把请求关联到终端用户档案。

**wrapper 变更**

- **`output_config`**:`anthropicRequest` 新增 `OutputConfig *anthropicOutputConfig`(含 `Effort string` 与 `Format *anthropicOutputFormat{Type, Schema any}`)。新增 `toAnthropicOutputConfig` / `toAnthropicOutputFormat` 助手:`ReasoningEffort` → `effort`,JSON Schema 形态的 `ResponseFormat` → `format`,同时接受 OpenAI 的嵌套 `json_schema.schema` 与扁平 `schema`;schema 本身原样透传,不做校验或改写;无法提取 schema 的形态(如 `{type:"json_object"}`)不伪造 `format`。两半皆空时整个字段省略。**行为变更**:顶层 `anthropicRequest.Effort` 标注弃用且不再赋值,线上改发 `output_config.effort`,字段本身保留以维持内部源码兼容。
- **Usage**:`anthropicUsage` 新增 `OutputTokensDetails` / `ServerToolUse` / `InferenceGeo` / `ServiceTier`。规范 `Usage` 新增 `ServerToolUse *ServerToolUse`(零值计数 `omitempty`)、`InferenceGeo`、`ServiceTier`;`ReasoningTokens` 现在也取自 Anthropic 的 `output_tokens_details.thinking_tokens`(显式顶层 `reasoning_tokens` 仍优先)。三者均经 `usageJSON` 往返;`Usage.Add` 累加服务端工具计数,但不动 geo/tier(它们描述单次请求)。流式不再只覆写 `output_tokens`:新增 `mergeAnthropicUsage`,只更新终止 `message_delta` 实际携带的字段,`message_start` 的输入/缓存/geo/tier/服务端工具信息因此得以保留。
- **`container` / `inference_geo`**:`ChatRequest` 新增两个 `omitempty` 字段直通到 `anthropicRequest`。`anthropicResponse.Container` 因字段形状一致,直接反序列化到新的规范 `ResponseContainer{ID, ExpiresAt string}` —— `ExpiresAt` 保持服务端字符串,不解析过期、不自动续期、不重试。非流式挂 `ChatResponse.Container`;流式新增 `StreamChunk.Container`,在解析到 `message_start` 时**立即发出一次**,使仅有工具事件或随即结束的流也不丢失容器 ID。
- **未知内容块**:`Message` 新增 `ExtraBlocks []json.RawMessage`(`json:"-"`,仅运行时)。`anthropicResponse.Content` 由 `[]anthropicContentBlock` 改为新的 `[]anthropicResponseBlock` —— 既解码已知字段,又保留每个块的原始 JSON(先解码再 marshal 会丢失未建模字段)。`fromAnthropicResponse` 新增 `default` 分支追加未识别块;带 `citations` 的 `text` 块照常贡献文本,并额外保留整块原文。`anthropic_stream.go` 逐事件同理:未识别的 `content_block_start` 发出原始 `content_block` 并记录 index,该 index 上后续所有 delta(以及已知块上未知类型的 delta)发出原始 `delta`,按到达顺序。**顺带修复一个解码故障**:`anthropicContentBlock.ResultContent` 是 tag 为 `content` 的 `string`,而响应侧 `content` 是多态的(服务端工具结果为数组、代码执行结果为对象),含此类块的响应此前**整体解码失败**;`anthropicResponseBlock` 用 `json.RawMessage` 遮蔽该键,请求侧字符串语义不变。`Message.AppendDelta` 只追加不解析、不合并。`signature_delta` 仍被忽略,`text`/`thinking`/`tool_use`/`input_json_delta` 行为不变。
- **工具字段**:规范 `Tool` 与 `anthropicTool` 同时新增 `Strict *bool` / `DeferLoading *bool` / `AllowedCallers []string` / `EagerInputStreaming *bool` / `InputExamples []any`(均 `omitempty`),在工具转换循环中原样复制;`anthropicTool` 另有 `Type string`。`ChatRequest.clone()` 现在复制每个工具的 `AllowedCallers` / `InputExamples` slice(元素保持浅拷贝,符合既有 `any` 契约)。**与规范的偏离**:规范原本要求 `Tool.Type` 无条件映射,但规范层该字段在 OpenAI 语义下对每个普通工具都是 `"function"`,发给 Anthropic 会被拒绝 —— 因此 `"function"` 与空串一并视为默认 custom 工具而不发送,其余取值(如 `web_search_20260209`)原样透传,不枚举、不校验。
- **`anthropic-user-profile-id`**:`Client` 新增 `anthropicUserProfileID` 与 `WithAnthropicUserProfileID(string)`(空串忽略,与 `WithAnthropicVersion` 一致);`setAnthropicHeaders` 仅在非空时发送该头。
- **默认行为不变**:不设置任何新字段时,请求 wire JSON 与此前一致,唯一区别是 `ReasoningEffort` 改落在 `output_config.effort` 而非顶层 `effort`;新增的 `json:"-"` 成员不会进入 OpenAI 请求体;响应缺少新成员时新增字段保持零值/nil 并从 JSON 省略;默认不发送新请求头。
- **有意未实现**:不枚举 Anthropic 版本化内置工具类型(类型名作为不透明 `string`);不解析 `citations` 为规范字段(改为保留原始块);`tool_reference` / `tool_search` 元协议延后;顶层 `effort` 本次保留(不再赋值)而非删除。

## 2026-06-02 — 顶层自动缓存 + usage 的 `cache_creation` TTL 明细

**官方变更**

自 2026-02-19,**自动缓存**允许请求携带单个位于请求根的 `cache_control`(`{type:"ephemeral"}`,可选 `ttl:"1h"`),取代逐块标记:服务端缓存最后一个可缓存块,并随对话增长自动前移断点。响应 `usage` 额外返回 `cache_creation` 对象,按 TTL 拆分缓存写入 —— `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`,两者之和等于 `cache_creation_input_tokens`。

wrapper 此前只支持块级 `cache_control`,且完全未暴露缓存写入计数。

**wrapper 变更**

- `ChatRequest`(`schema.go`)新增 `AutoCache bool` / `AutoCacheTTL string`,均 `json:"-"`(结构体局部,不出现在 OpenAI 形状请求体,与 `CacheBreakpoint` 一致)。
- `toAnthropicRequest`(`anthropic.go`)在 `AutoCache` 为真时设置新的顶层 `anthropicRequest.CacheControl`;`anthropicCacheControl` 新增 `TTL string`。空 `ttl` → 省略 → 默认 5 分钟;`"1h"` → 1 小时。与逐块 `CacheBreakpoint` 独立共存。
- `anthropicUsage` 新增 `CacheCreation *anthropicCacheCreation`;规范 `Usage` 新增 `CacheWriteTokens`(← `cache_creation_input_tokens`,总量,此前未暴露)、`CacheWrite5mTokens`、`CacheWrite1hTokens`,均纳入 `Usage.Add` 与 JSON 往返。
- 非流式(`fromAnthropicResponse`)与流式(`anthropic_stream.go`,重构为把 `message_start` 的 usage 携带到终帧)共用新的 `anthropicCanonicalUsage`。
- **默认行为不变**:`AutoCache` 为假时不发请求根 `cache_control`;OpenAI 响应无 `cache_creation`,新字段保持 0/省略。

## 2026-06-02 — 透传 `top_k` 采样参数

**官方变更**:Messages API 原生支持 `top_k`(top-k 截断采样,每步只在概率最高的 K 个 token 中采样)。规范 `ChatRequest` 此前无对应字段。

**wrapper 变更**:`ChatRequest` 新增 `TopK *int`(紧邻 `TopP`),`anthropicRequest` 新增同名字段,`toAnthropicRequest` 直通映射。OpenAI Chat Completions 无 `top_k`,而规范请求就是 OpenAI 形状 —— 未设置时 `omitempty` 省略,设置时原样透传(接受该字段的 OpenAI 兼容后端会生效,其余忽略未知字段)。

## 2026-06-02 — 可配置 `anthropic-beta` / `anthropic-version` 请求头

**官方变更**:许多 Anthropic 能力(compaction、context-editing、structured-outputs(beta 期)、fast-mode、advisor 等)通过 `anthropic-beta` 请求头选择性开启(多值逗号连接);`anthropic-version` 头(默认 `2023-06-01`)选择 API 版本。

**wrapper 变更**:仅新增基础设施,不接入任何具体 beta 能力。`Client` 新增 `anthropicBeta []string` / `anthropicVersion string`,以及 `WithAnthropicBeta(values ...string)`(跨次调用累加、忽略空串、线上逗号连接)与 `WithAnthropicVersion(string)`(空串保留默认)。`setAnthropicHeaders` 发送 `anthropic-version`,仅在非空时发送 `anthropic-beta`。**默认行为不变**。

## 2026-06-02 — 新增 `stop_reason` 常量与 `stop_details`(拒答分类)

**官方变更**

`stop_reason` 新增三个取值:

- `model_context_window_exceeded`(2025-09-29)—— 输入 + 输出超出模型上下文窗口,**与触及请求 `max_tokens` 不同**;
- `pause_turn` —— 长任务/服务端工具的一轮被暂停,客户端可重放继续;
- `refusal`(2026-05-28,Opus 4.8)—— 流式分类器对潜在违规内容介入。

当 `stop_reason` 为 `refusal` 时,响应与终帧 `message_delta` 携带 `stop_details`(`{type, category, explanation}`)给出分类。

**wrapper 变更**

`mapAnthropicStopReason` 此前只映射 `end_turn`/`stop_sequence`/`max_tokens`/`tool_use`,其余原样透传字符串。现在把三个新取值映射为具名常量 `FinishReasonModelContextWindowExceeded` / `FinishReasonRefusal` / `FinishReasonPauseTurn`,常量值即 Anthropic 原字符串(纯增量,行为不变;**不**折叠进 `content_filter`/`length`,保留 Anthropic 语义)。新增规范 `StopDetails{Type, Category, Explanation}`;`Choice` 与 `StreamChunkChoice` 新增 `StopDetails *StopDetails`。`anthropicResponse` 与 `anthropicMessageDeltaData` 因字段形状完全一致,直接反序列化到规范类型。

## 2026-06-02 — 映射 `effort`、支持 `thinking.display`、弃用 `budget_tokens`

**官方变更**:自 2026-02-05,顶层 `effort` 参数 GA,取代 `thinking.budget_tokens` 控制推理深度(同时启用 `thinking.type:"adaptive"`,由模型自行决定思考量)。自 2026-03-16,`thinking.display:"omitted"` 可抑制思考内容以加快流式。

**wrapper 变更**:`toAnthropicRequest` 把规范 `ChatRequest.ReasoningEffort` 映射到新的顶层 `anthropicRequest.Effort`(空值省略;同一字段仍驱动 OpenAI 的 `reasoning_effort`)。`Thinking` 新增 `Display string`;`Type` 保持 `string` 以便 `"adaptive"` 透传;`BudgetTokens` 标注弃用并指向 `effort`/`adaptive`。

## 2026-06-02 — `tool_choice` 的 `"none"` 映射与 `disable_parallel_tool_use`

**官方变更**:自 2024-10-03,`tool_choice` 在 `auto`/`any`/`tool` 之外接受 `disable_parallel_tool_use`(每轮至多一次工具调用)。自 2025-02-27,`tool_choice:{type:"none"}` 显式禁止任何工具调用 —— 与省略该字段(模型自选)语义不同。

**wrapper 变更**:`convertToolChoice` 此前对 `"none"` 返回 `nil`,整个字段被丢弃(变成"模型自选"),现在正确映射为 `{type:"none"}`。`anthropicToolChoice` 新增 `DisableParallelToolUse *bool`。组装逻辑折入规范 `ParallelToolCalls`:显式为 `false` 时置 `disable_parallel_tool_use:true` —— 未指定 choice 但有工具时兜底用 `{type:"auto"}` 承载(无工具时不发 `tool_choice`,否则会被拒绝),且**绝不**附加到 `{type:"none"}`。未设置或为 `true` 时不改动 choice。

## 2026-06-02 — 保留会话中的 `system` 消息(修复上提)

**官方变更**:自 2026-05-28(Opus 4.8),`messages` 允许在非首位出现 `role:"system"` 条目(会话中系统消息),使调用方能在会话中途更改指令并保持 prompt 缓存命中。

**wrapper 变更**:`toAnthropicRequest` 此前把**所有** `RoleSystem` 消息无视位置地上提到顶层 `system`,错误地把会话中系统消息挪到最前并丢失位置语义。现在只提取**前导连续段**(第一个非 system 消息之前的那些);出现在会话中途的 system 消息落到 `toAnthropicMessage`,原位保留为 `role:"system"` 消息。`CacheBreakpoint` 与块/字符串形态的行为不变。

## [基线] 2026-06-02

- **官方协议**:Anthropic Messages API(`/v1/messages`,以官方文档端点为准)
- **变更摘要**:封装非流式与流式 Messages API,与规范 OpenAI 兼容类型双向翻译;system 消息提取到独立 `system` 字段;流式使用 Anthropic SSE 事件类型(`content_block_delta`、`message_delta` 等)。
