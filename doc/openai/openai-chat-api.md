# OpenAI Chat Completions API 封装设计与实现

- **官方协议**:OpenAI Chat Completions API(`POST {baseURL}/chat/completions`)
- **官方文档**:https://platform.openai.com/docs/api-reference/chat
- **实现文件**:`openai_chat.go`(HTTP 与错误解析)、`openai_stream.go`(SSE 解析)、`schema.go`(规范类型即 OpenAI 形状)
- **变更记录**:[openai-api-changes.md](./openai-api-changes.md)

---

## 1. 零翻译路径

SDK 以 **OpenAI Chat Completions 格式作为规范表示**(见 [../api.md](../api.md) §2),因此 OpenAI 路径**没有翻译层**:

```
ChatRequest ──json.Marshal──▶ POST {baseURL}/chat/completions
ChatResponse ◀──json.Decode── 响应体
```

`schema.go` 里的 `ChatRequest` / `ChatResponse` / `Message` / `Choice` / `Usage` 就是 OpenAI 的线上结构。这也意味着:**新增一个 OpenAI 请求参数,只需在 `ChatRequest` 上加一个 `omitempty` 字段** —— 无需改动任何协议代码。

同样的性质使它天然适配所有 OpenAI 兼容后端(DeepSeek、Kimi/Moonshot、GLM、Qwen、Doubao、MiniMax、Gemini 的 OpenAI 兼容端点等):后端的私有扩展参数只要形状相符即可直接透传,未知字段由后端自行忽略。

## 2. 请求发送(`doRequest`)

```go
POST {baseURL}/chat/completions
Content-Type: application/json
Authorization: Bearer {apiKey}
```

- `baseURL` 为空直接返回 `ErrNoBaseURL` —— OpenAI 兼容后端太多,无法给默认值(与 Anthropic 不同)。
- `baseURL` 在 `WithBaseURL` / 环境变量读取时已去掉尾部 `/`,拼接时不会产生 `//`。
- 端点路径固定为 `/chat/completions`,即调用方传入的 `baseURL` 应含版本段(如 `https://api.openai.com/v1`)。

## 3. 非流式(`openaiChatCompletion`)

流程:

1. `req.clone()` 深拷贝,强制 `Stream = false`;
2. 填充默认模型;
3. 发请求;
4. **状态码非 200** → `parseErrorResponse`(见 §6);
5. 解码为 `ChatResponse`;
6. **200 但响应体含 `error`** → 仍构造 `APIError`(部分兼容后端会这么返回);
7. `Choices` 为空 → `ErrEmptyResponse`。

第 6 步是必要的防御:OpenAI 兼容实现的错误返回方式并不统一。

## 4. 流式(`openaiChatCompletionStream`)

流程与非流式一致,差异有二:

1. 强制 `Stream = true`;
2. **若 `StreamOptions` 为 `nil`,自动补 `{IncludeUsage: true}`** —— 否则末帧不会带 usage,`Stream.Usage()` 将拿不到数据。调用方显式设置过则尊重其选择。

成功后返回 `newStream(resp.Body)`,失败时先关闭 body 再返回错误。

### 4.1 SSE 解析(`openaiRecvFunc`)

OpenAI 的 SSE 是**无状态的逐行 `data:` 流**,解析规则:

| 行 | 处理 |
|---|---|
| 空行 | 跳过 |
| 以 `:` 开头(SSE 注释/心跳) | 跳过 |
| 非 `data: ` 前缀 | 跳过 |
| `data: [DONE]` | 返回 `io.EOF` |
| `data: {json}` | 解析并产出 chunk |

每个 chunk 用 `streamChunkOrError` 做**单次解析**(内嵌 `StreamChunk` + 可选 `Error`):若含 `error` 字段,直接返回 `*APIError`(无 HTTP 状态码);否则返回 chunk。JSON 解析失败返回 `decode stream chunk` 包装错误。

扫描结束后:`Scanner` 有错误则返回该错误,否则返回 `io.EOF`(容忍缺失 `[DONE]` 的兼容后端)。

单行上限 `maxStreamLineSize = 1 MB`,缓冲初始 64 KB。

### 4.2 增量合并

流式 delta 通过规范侧的 `Message.AppendDelta` / `ToolCall.Merge` 累积(见 [../api.md](../api.md) §5.1):文本与 thinking 追加,工具调用按 `Index` 就地合并,`Function.Arguments` 字符串拼接。

## 5. 字段说明(与官方对齐要点)

### 5.1 token 上限

- `MaxTokens *int` —— **已弃用**。OpenAI 已在 Chat Completions 上弃用 `max_tokens`,推理模型(o 系列、GPT-5.x)会**直接拒绝**含该字段的请求。
- `MaxCompletionTokens *int` —— 取代前者。其上限**同时覆盖可见输出 token 与内部推理 token**,是推理模型唯一接受的 token 上限字段。

新代码一律使用 `MaxCompletionTokens`;`MaxTokens` 仅为仍接受它的老模型/非推理模型保留。

### 5.2 推理与冗长度

- `ReasoningEffort string` —— 取值 `none` / `minimal` / `low` / `medium` / `high` / `xhigh`(GPT-5.1 默认 `none`),对应常量 `ReasoningEffort*`。
- `Verbosity string` —— 取值 `low` / `medium` / `high`,对应常量 `Verbosity*`。

两者都**保持 `string` 而非枚举类型**,以便向定义了私有取值的兼容后端透传。

### 5.3 多模态

**输入**(`ContentPart.Type`):

| Type | 载荷 |
|---|---|
| `text` | `Text string` |
| `image_url` | `ImageURL{URL, Detail}` |
| `input_audio` | `InputAudio{Data(base64), Format("wav"/"mp3")}` |
| `file` | `FilePart{FileID}` 或 `{Filename, FileData(base64)}` |

**输出**:`ChatRequest.Modalities []string`(如 `["text","audio"]`)+ `Audio *AudioConfig{Voice, Format}`;生成的音频返回在 `Message.Audio`(`MessageAudio{ID, Data, Transcript, ExpiresAt}`)。请求音频输出时 `Modalities` 必须包含 `"audio"`。

### 5.4 可观测与路由参数

| 字段 | 说明 |
|---|---|
| `Logprobs *bool` / `TopLogprobs *int` | 请求逐 token 对数概率;`TopLogprobs`(0–20)需 `Logprobs=true` |
| `LogitBias map[string]int` | 键为 token ID 字符串,值域 `[-100, 100]` |
| `ParallelToolCalls *bool` | 是否允许一轮多次工具调用,服务端默认 true |
| `ServiceTier string` | `auto` / `default` / `flex` / `priority`,保持 `string` 透传 |
| `Store *bool` | 是否持久化本次补全(供 dashboard / evals 取用) |
| `Metadata map[string]string` | 至多 16 组键值,随存储的补全一并返回 |
| `PromptCacheKey string` | 把共享可缓存前缀的请求路由到同一缓存,提升命中率 |

响应侧:`Choice.LogProbs *LogProbs{Content, Refusal []TokenLogprob}`,`TokenLogprob{Token, Logprob, Bytes, TopLogprobs}`,`TopLogprob{Token, Logprob, Bytes}`。

### 5.5 用量的嵌套字段解析

OpenAI 把两项细分计数放在嵌套对象里,`Usage.UnmarshalJSON` 会**提升到顶层规范字段**:

| OpenAI 线上路径 | 规范字段 |
|---|---|
| `prompt_tokens_details.cached_tokens` | `CacheReadTokens` |
| `completion_tokens_details.reasoning_tokens` | `ReasoningTokens` |

规则:**显式顶层字段优先** —— 只有顶层为 0 时才回填嵌套值。`CacheReadTokens` 是 `PromptTokens` 的子集,`ReasoningTokens` 是 `CompletionTokens` 的子集,单独暴露仅供可观测,不要重复计费。

OpenAI 无缓存写入计费概念,`CacheWriteTokens` 系列字段在该路径下恒为 0(省略)。

### 5.6 结束原因

`FinishReason` 常量:`stop` / `length` / `tool_calls` / `content_filter`,以及遗留的 `function_call`(旧 functions API,保留兼容)。

### 5.7 Prompt 缓存

OpenAI **自动缓存** 1024 token 以上的前缀,请求侧无需任何标记。因此 `Message.CacheBreakpoint` / `Tool.CacheBreakpoint` / `ChatRequest.AutoCache` 这些 Anthropic 专属开关全部标注 `json:"-"`,永不出现在 OpenAI 请求体中。想提升命中率时用 `PromptCacheKey`。

## 6. 错误处理(`parseErrorResponse`)

1. 读取响应体,上限 `maxErrorBodySize = 1 MB`(`io.LimitReader`);
2. 读取失败 → `APIError{StatusCode, Message:"failed to read error response", Err}`;
3. 按 `{"error":{code,message,param,type}}` 解析;
4. **解析失败或 `error` 缺失 → 把原始 body 原样放进 `Message`**,保证诊断信息不丢;
5. 成功 → `APIError{StatusCode, Code, Message, Type}`。

## 7. Anthropic 侧无对应的字段

以下规范字段在切换到 `ProtocolAnthropic` 时会被静默忽略:`N`、`FrequencyPenalty`、`PresencePenalty`、`Seed`、`User`、`ResponseFormat`、`Verbosity`、`Logprobs` / `TopLogprobs` / `LogitBias`、`ServiceTier`、`Store`、`Metadata`、`PromptCacheKey`、`Modalities` / `Audio`、`StreamOptions`。

跨协议编写代码时应把这些视为"尽力而为"的参数。
