# Anthropic Messages API 封装设计与实现

- **官方协议**:Anthropic Messages API(`POST /v1/messages`)
- **官方文档**:https://platform.claude.com/docs/en/api/messages
- **实现文件**:`anthropic.go`(类型与双向翻译)、`anthropic_chat.go`(HTTP 与鉴权)、`anthropic_stream.go`(SSE 解析)
- **变更记录**:[anthropic-api-changes.md](./anthropic-api-changes.md)

核心设计前提见 [../api.md](../api.md):SDK 以 **OpenAI 格式为规范表示**,Anthropic 协议是唯一需要双向翻译的路径。

---

## 1. 整体结构

```
ChatRequest ──toAnthropicRequest()──▶ anthropicRequest ──JSON──▶ POST {base}/v1/messages
                                                                        │
ChatResponse ◀─fromAnthropicResponse()── anthropicResponse ◀────────────┘   (非流式)

Stream.Recv() ◀─anthropicRecvFunc()── SSE events(message_start / content_block_* / message_delta / …)
```

**所有 Anthropic 类型都是包私有的**(`anthropicRequest`、`anthropicResponse`、`anthropicContentBlock`…),不对外暴露 —— 调用方始终只面对规范类型,协议细节不泄漏到公开 API。

## 2. 端点、鉴权与请求头

```go
const (
    anthropicDefaultBaseURL   = "https://api.anthropic.com"
    anthropicAPIVersion       = "2023-06-01"
    anthropicDefaultMaxTokens = 4096
)
```

- **Base URL**:`Client.baseURL` 非空则用它,否则回退默认值 —— 这也是构造期允许 Anthropic 协议不传 Base URL 的原因。
- **请求头**(`setAnthropicHeaders`):

| 头 | 值 |
|---|---|
| `Content-Type` | `application/json` |
| `x-api-key` | `Client.apiKey`(注意:不是 `Authorization: Bearer`) |
| `anthropic-version` | `WithAnthropicVersion` 配置值,否则 `2023-06-01` |
| `anthropic-beta` | `WithAnthropicBeta` 累积值逗号连接;**为空时整个头省略** |
| `anthropic-user-profile-id` | `WithAnthropicUserProfileID` 配置值;**为空时整个头省略** |

`anthropic-beta` 是通用的 beta 能力开关基础设施(compaction、context-editing、structured-outputs、fast-mode、advisor 等),SDK 只负责发头,不为任何具体 beta 能力做字段建模。

## 3. 请求翻译(`toAnthropicRequest`)

### 3.1 直通字段

| 规范字段 | Anthropic 字段 |
|---|---|
| `Model` | `model` |
| `Temperature` | `temperature` |
| `TopP` | `top_p` |
| `TopK` | `top_k` |
| `Stop []string` | `stop_sequences` |
| `Stream` | `stream` |
| `Thinking` | `thinking`(结构体直接复用) |
| `ReasoningEffort` | `output_config.effort` |
| `ResponseFormat`(JSON Schema 形态) | `output_config.format` |
| `Container` | `container` |
| `InferenceGeo` | `inference_geo` |

### 3.2 `max_tokens`(必填)

Anthropic 的 `max_tokens` 是**必填字段**,而规范请求里它可以缺省,因此采用三级回退:

```
MaxCompletionTokens(优先) → MaxTokens(已弃用) → 4096(anthropicDefaultMaxTokens)
```

### 3.3 system 消息:只提取前导连续段

这是一处**位置语义敏感**的翻译。Anthropic 把系统提示放在顶层 `system` 字段,但自 Opus 4.8(2026-05-28)起 `messages` 里也允许出现 `role:"system"` 的**会话中系统消息**。

规则:

- 只有**首个非 system 消息之前**的连续 system 消息会被提取到顶层 `system`;
- 之后出现的 system 消息**原位保留**为 `role:"system"` 的 Anthropic 消息。

内部用 `seenNonSystem` 标志实现。这样做保留了两件事:指令的**位置语义**,以及**prompt 缓存命中**(把中途指令上提到最前会让整段前缀失效)。

`system` 字段有两种线上形态,由 `json.RawMessage` 承载:

| 条件 | 形态 |
|---|---|
| 纯文本、无缓存标记 | 字符串(多条用 `\n` 连接) |
| 含多模态 parts,或任一条设置了 `CacheBreakpoint` | 内容块数组 `[{type:"text",text:...}]` |

设置了缓存标记时,`cache_control` 挂在**最后一个块**上 —— Anthropic 缓存"截至并包含该块"的全部内容。

### 3.4 消息翻译(`toAnthropicMessage`)

按消息类型选择内容形态:

| 输入 | 输出 |
|---|---|
| `RoleTool` | `role:"user"` + `[{type:"tool_result", tool_use_id, content}]`;缺 `ToolCallID` 直接报错 |
| `RoleAssistant` 且含 thinking 或 tool_calls | 块数组:`thinking` 块 → `text` 块 → 每个 `tool_use` 块(`Input` 由 `Function.Arguments` 原样作 `json.RawMessage`) |
| 含多模态 parts | 块数组:`text` → `{type:"text"}`;`image_url` → `{type:"image", source:…}` |
| 纯文本 + `CacheBreakpoint` | 单元素块数组(为了挂 `cache_control`) |
| 纯文本 | 字符串 |

**图片来源判别**:`parseDataURI` 识别 `data:<mediaType>;base64,<data>` 形态 → `source{type:"base64", media_type, data}`;否则视为远程 URL → `source{type:"url", url}`。

**未映射的部分**:`input_audio` / `file` 内容块在 Anthropic 侧无对应,`switch` 不匹配即安全跳过(不报错、不产生空块)。

所有块数组形态下,`CacheBreakpoint` 都挂到**最后一个块**。

### 3.5 工具与 `tool_choice`

工具映射直白:`Function.Name/Description/Parameters` → `name/description/input_schema`;`Tool.CacheBreakpoint` 为真则挂 `cache_control`(Anthropic 缓存截至并包含该工具的全部工具定义)。

Anthropic 工具定义扩展字段原样复制:`Strict` → `strict`、`DeferLoading` → `defer_loading`、`AllowedCallers` → `allowed_callers`、`EagerInputStreaming` → `eager_input_streaming`、`InputExamples` → `input_examples`(均 `omitempty`)。

`Tool.Type` 兼作 Anthropic 的工具类型,但规范层该字段在 OpenAI 语义下对每个普通工具都是 `"function"`,原样发给 Anthropic 会被拒绝 —— 因此 `"function"` 与空串一并视为 Anthropic 的默认 custom 工具而**不发送**;其余取值(版本化内置工具,如 `web_search_20260209`)原样透传,不枚举、不校验版本名。

`tool_choice` 映射(`convertToolChoice`):

| 规范值 | Anthropic |
|---|---|
| `"auto"` | `{type:"auto"}` |
| `"required"` | `{type:"any"}` |
| `"none"` | `{type:"none"}` —— **禁止任何工具调用**,与"省略字段"(模型自选)语义不同 |
| `{"function":{"name":"x"}}` | `{type:"tool", name:"x"}` |
| 其它 | `nil`(省略) |

`ParallelToolCalls` 的折叠规则(因为 Anthropic 把该开关放在 `tool_choice` 内部):

- 仅当显式为 `false` 时才生效,置 `disable_parallel_tool_use:true`;
- 未指定 `tool_choice` 但**有工具**时,兜底构造 `{type:"auto"}` 来承载这个标志 —— 无工具时不能凭空发 `tool_choice`(会被拒绝);
- **绝不**挂到 `{type:"none"}` 上(本就不允许调用,该标志无意义);
- 未设置或为 `true` 时完全不动 `tool_choice`。

### 3.6 推理与思考

- `ReasoningEffort` → `output_config.effort`,空值省略。它**取代** `thinking.budget_tokens` 成为新模型的推理深度控制。顶层 `anthropicRequest.Effort` 已弃用且不再赋值,两者不会同时发出。
- `ResponseFormat` → `output_config.format`(`{type:"json_schema", schema:…}`):同时接受 OpenAI 的嵌套 `json_schema.schema` 与扁平 `schema`;schema 原样透传,不校验、不改写;无法提取 schema 的形态(如 `{type:"json_object"}`)不伪造 `format`。`effort` 与 `format` 皆空时整个 `output_config` 省略。
- `Thinking.Type`:`"enabled"` / `"disabled"` / `"adaptive"`(由模型自行决定思考量);保持 `string` 以便透传。
- `Thinking.BudgetTokens`:**已弃用**,仅为仍需固定预算的模型/调用方保留。
- `Thinking.Display`:设为 `"omitted"`(2026-03-16 起)可抑制 thinking 内容下发,加快流式响应。

### 3.7 Prompt 缓存:两种并存模式

| 模式 | 触发 | 线上表现 |
|---|---|---|
| **逐块断点** | `Message.CacheBreakpoint` / `Tool.CacheBreakpoint`(均 `json:"-"`) | 在对应块/工具上挂 `cache_control:{type:"ephemeral"}` |
| **自动缓存** | `ChatRequest.AutoCache` + `AutoCacheTTL`(均 `json:"-"`) | 请求根挂单个 `cache_control:{type:"ephemeral", ttl:…}` |

自动缓存(2026-02-19 起)由**服务端**把断点放在最后一个可缓存块上,并随对话增长自动前移,省去调用方手工维护断点。`AutoCacheTTL` 为空 → 省略 `ttl` → 默认 5 分钟;`"1h"` → 1 小时缓存。

两者是**独立字段,可以共存**。两个开关都是 `json:"-"` 的结构体局部字段,永远不会出现在 OpenAI 形状的请求体里。

## 4. 响应翻译(`fromAnthropicResponse`)

内容块聚合成单个 assistant 消息:

| 块类型 | 归属 |
|---|---|
| `thinking` | 累积后 `\n` 连接 → `Message.Thinking` |
| `text` | 累积后 `\n` 连接 → `Message.Content` |
| `tool_use` | 追加 `ToolCall{Index:序号, ID, Type:"function", Function{Name, Arguments:string(Input)}}` |
| 其它(`server_tool_use`、`web_search_tool_result`、`code_execution_tool_result`、未来新增类型) | 原始 JSON 追加到 `Message.ExtraBlocks` |

带 `citations` 的 `text` 块照常贡献文本,并**额外**把整块原文追加到 `ExtraBlocks` —— 引用注解不提升为规范字段,但也不丢失。

保真要求决定了实现方式:`anthropicResponse.Content` 的元素类型是 `anthropicResponseBlock`,它在解码已知字段的同时保留每个块的原始字节(先解码到已知结构再 marshal 会丢掉未建模字段)。该类型还用 `json.RawMessage` 遮蔽了请求侧的 `ResultContent`(tag 同为 `content`)—— 响应侧 `content` 是多态的(服务端工具结果为数组、代码执行结果为对象),按 `string` 解码会让整个响应解码失败。

其余字段:`ID` → `ID`;`Object` 固定填 `"chat.completion"`;`Model` 直通;`container` → `ChatResponse.Container`(`*ResponseContainer{ID, ExpiresAt}`,字段形状一致直接反序列化;`ExpiresAt` 保持服务端字符串,不解析过期、不自动续期);固定产出**单个 `Choice`**(Anthropic 无 `n` 概念)。

### 4.1 停止原因映射(`mapAnthropicStopReason`)

| Anthropic `stop_reason` | 规范 `FinishReason` |
|---|---|
| `end_turn`、`stop_sequence` | `stop` |
| `max_tokens` | `length` |
| `tool_use` | `tool_calls` |
| `model_context_window_exceeded` | 同名常量(**不**折叠成 `length` —— 是上下文窗口溢出,而非请求的 `max_tokens` 触顶) |
| `refusal` | 同名常量(流式分类器判定潜在违规而中止) |
| `pause_turn` | 同名常量(长任务/服务端工具被暂停,客户端可重放继续) |
| 其它 | 原样透传为 `FinishReason(reason)` |

后三者保留 Anthropic 语义而非归一到 `content_filter`/`length`,是有意为之:归一会丢失调用方据以决策(重放?换模型?换提示?)的关键信息。

### 4.2 `stop_details`

`anthropicResponse.StopDetails` 直接声明为规范的 `*StopDetails` —— 两侧字段形状完全一致(`type` / `category` / `explanation`),因此**直接反序列化,无需转换**,再挂到 `Choice.StopDetails`。

### 4.3 用量映射(`anthropicCanonicalUsage`)

非流式与流式**共用**这个辅助函数:

```
PromptTokens       = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
CompletionTokens   = output_tokens
TotalTokens        = PromptTokens + CompletionTokens
CacheReadTokens    = cache_read_input_tokens
CacheWriteTokens   = cache_creation_input_tokens
CacheWrite5mTokens = cache_creation.ephemeral_5m_input_tokens   (存在时)
CacheWrite1hTokens = cache_creation.ephemeral_1h_input_tokens   (存在时)
ReasoningTokens    = output_tokens_details.thinking_tokens      (存在时)
ServerToolUse      = server_tool_use{web_search_requests, web_fetch_requests}  (存在时)
InferenceGeo       = inference_geo
ServiceTier        = service_tier
```

注意 `PromptTokens` 是**含缓存部分的总输入**(`totalInputTokens()`);缓存读/写计数是它的子集,单独暴露仅供可观测。

`Usage.Add` 累加所有计数(含 `ServerToolUse`),但**不动** `InferenceGeo` / `ServiceTier` —— 它们描述单次请求,拼接或累加都没有意义。

## 5. 流式实现(`anthropic_stream.go`)

Anthropic SSE 与 OpenAI 有两点结构性差异,都由 `anthropicRecvFunc` 吸收:

1. **事件成对出现**:`event: <type>` 行后跟 `data: <json>` 行;解析器读到 `event:` 后继续向下扫描,跳过空行与 `:` 注释,直到取到 `data:`。
2. **有状态**:`message_start` 提供 `id`/`model`/输入侧 usage,后续 chunk 需要携带这些信息。

闭包持有的状态:`msgID`、`model`、`startUsage`、`blockToTool map[int]int`、`nextToolIdx`、`unknownBlocks map[int]bool`。

### 5.1 事件处理表

| 事件 | 行为 |
|---|---|
| `message_start` | 记录 `msgID` / `model` / `startUsage`;**带 `container` 时立即产出一个只含 `Container` 的 chunk**,否则不产出 chunk |
| `content_block_start`(`tool_use`) | 分配工具序号,建立 `blockToTool[块索引]=工具索引`,产出携带 `ID`/`Name` 的工具调用 chunk |
| `content_block_start`(`text` / `thinking`) | 跳过 |
| `content_block_start`(未知类型) | 记 `unknownBlocks[块索引]=true`,产出携带原始 `content_block` 的 `ExtraBlocks` chunk |
| `content_block_delta`(所属块在 `unknownBlocks` 中) | 无论 delta 自身类型,产出携带原始 `delta` 的 `ExtraBlocks` chunk |
| `content_block_delta` / `text_delta` | 产出 `Delta.Content` |
| `content_block_delta` / `thinking_delta` | 产出 `Delta.Thinking` |
| `content_block_delta` / `input_json_delta` | 按 `blockToTool` 查出工具序号,产出 `Function.Arguments` 分片;查不到则跳过 |
| `content_block_delta` / `signature_delta` | 跳过 |
| `content_block_delta`(已知块上的未知 delta 类型) | 产出携带原始 `delta` 的 `ExtraBlocks` chunk |
| `message_delta` | 产出终帧:`FinishReason`(经 `mapAnthropicStopReason`)+ `StopDetails`;若带 `usage`,经 `mergeAnthropicUsage` 合入 `startUsage` 后由 `anthropicCanonicalUsage` 产出完整 `Usage` |
| `message_stop` | 返回 `io.EOF` |
| `error` | 返回 `*APIError{Type, Message}` |
| `ping` / `content_block_stop` | 跳过 |

### 5.2 索引重映射

Anthropic 对**所有**内容块(text、thinking、tool_use)使用统一递增的块索引,而规范侧 `Message.AppendDelta` 期望的是**仅在工具调用范围内**编号的索引。`blockToTool` 就是这个重映射表:遇到 `tool_use` 的 `content_block_start` 时建立映射,后续 `input_json_delta` 据此把参数分片投递到正确的工具调用上。

### 5.3 用量的两段拼装

Anthropic 把输入侧计数(含缓存读写、`inference_geo`、`service_tier`、`server_tool_use`)放在 `message_start`,把最终 `output_tokens` 放在 `message_delta`。因此解析器先在 `startUsage` 里暂存前者,终帧到达时经 `mergeAnthropicUsage` 合并后再统一转换 —— 这保证流式与非流式拿到**结构完全一致**的 `Usage`。

合并而非替换是关键:终帧通常只带 `output_tokens`,`mergeAnthropicUsage` 只更新该事件**实际携带**的字段,否则零值会抹掉 `message_start` 已建立的输入/缓存/地域/服务等级/服务端工具信息。

### 5.4 容器 ID 的早发

`container` 在 `message_start` 里就已知,却要等到有文本 delta 才随 chunk 返回的话,仅有工具事件或随即结束的流会彻底丢失它 —— 而流式路径不产出 `ChatResponse`,调用方没有别的途径拿到下一轮要复用的 ID。因此 `StreamChunk` 新增 `Container` 字段,并在读到 `message_start` 时**立即**产出一个只携带它的 chunk;后续普通 delta 不重复携带,整个流中只出现一次。

## 6. 错误处理

`parseAnthropicErrorResponse`:读取响应体(上限 1 MB)→ 解析 `{"type":"error","error":{"type":..,"message":..}}` → 填充 `APIError{StatusCode, Type, Message}`;JSON 解析失败或 `message` 为空时,把**原始 body 原样**放入 `Message`,不丢失诊断信息。

流式路径下的 `error` 事件同样产出 `*APIError`,但没有 HTTP 状态码(`StatusCode` 为 0)。

## 7. 已知不映射项

以下规范字段在 Anthropic 侧无对应,翻译时静默忽略:`N`、`FrequencyPenalty`、`PresencePenalty`、`Seed`、`User`、`Verbosity`、`Logprobs` / `TopLogprobs` / `LogitBias`、`ServiceTier`(请求侧;响应 `usage.service_tier` 有映射)、`Store`、`Metadata`、`PromptCacheKey`、`Modalities` / `Audio`、`StreamOptions`(Anthropic 流式总是返回 usage)。

`ResponseFormat` 仅在 JSON Schema 形态下映射到 `output_config.format`;其余形态(如 `{type:"json_object"}`)按上表忽略,不伪造配置。

反向上,`LogProbs` 与 `Message.Audio` 在 Anthropic 响应里永不出现,保持 `nil`。
