# Anthropic Messages API 变更记录

本文件记录 **aimodel 的 Anthropic wrapper 与官方 Messages API 的同步情况**:官方哪一版发生了什么变更、wrapper 如何跟进。

- **官方协议**:Anthropic Messages API(`POST /v1/messages`)
- **官方文档**:https://platform.claude.com/docs/en/api/messages
- **实现说明**:[anthropic-chat-api.md](./anthropic-chat-api.md)
- **总账**:仓库根 [CHANGES.md](../../CHANGES.md)(含 OpenAI 侧条目)

**维护约定**:官方 API 变更时同步更新四处 —— wrapper 代码、README/CLAUDE.md、CHANGES.md、本文件。每条记录至少包含:日期、官方变更、wrapper 变更摘要。

倒序排列(新在前)。

---

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
