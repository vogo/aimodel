# OpenAI Chat Completions API 变更记录

本文件记录 **aimodel 的 OpenAI wrapper 与官方 Chat Completions API 的同步情况**:官方发生了什么变更、wrapper 如何跟进。

- **官方协议**:OpenAI Chat Completions API(`POST /chat/completions`;无独立版本号,以端点为准)
- **官方文档**:https://platform.openai.com/docs/api-reference/chat
- **实现说明**:[openai-chat-api.md](./openai-chat-api.md)
- **总账**:仓库根 [CHANGES.md](../../CHANGES.md)(含 Anthropic 侧条目)

**维护约定**:官方 API 变更时同步更新四处 —— wrapper 代码、README/CLAUDE.md、CHANGES.md、本文件。每条记录至少包含:日期、官方变更、wrapper 变更摘要。

由于 OpenAI 格式即 SDK 的规范表示,绝大多数 OpenAI 侧变更表现为 **`schema.go` 上的字段增补**,不涉及协议代码。

倒序排列(新在前)。

---

## 2026-06-02 — 多模态输入/输出(`input_audio`/`file` 内容块,`modalities`/`audio`)

**官方变更**:Chat Completions 支持通过内容块 `input_audio` 与 `file` 输入音频/文件,通过 `modalities` + `audio`(voice/format)请求音频输出,并在 `choices[].message.audio` 返回生成的音频。

**wrapper 变更**

- `ContentPart` 新增 `InputAudio *InputAudio`(`{data, format}`)与 `File *FilePart`(`{file_id | filename + file_data}`),均 `omitempty`;字符串/数组的多态仍由 `Content` 的 `MarshalJSON`/`UnmarshalJSON` 承担,未改动。
- `ChatRequest` 新增 `Modalities []string` 与 `Audio *AudioConfig{Voice, Format}`;`clone()` 增加对 `Modalities` 切片的深拷贝。
- `Message` 新增 `Audio *MessageAudio`(`{id, data, transcript, expires_at}`),解析助手生成的音频。
- Anthropic 翻译未动:无对应能力,新内容块类型在其 `text`/`image_url` 分支的 `switch` 中安全跳过。

## 2026-06-02 — 扩充 `ChatRequest` 常用请求字段(+ 响应 `logprobs`)

**官方变更**:Chat Completions 暴露请求参数 `logprobs` / `top_logprobs`、`logit_bias`、`parallel_tool_calls`、`service_tier`、`store`、`metadata`、`prompt_cache_key`,并在设置 `logprobs` 时返回 `choices[].logprobs`。

**wrapper 变更**

- `ChatRequest` 新增八个 `omitempty` 字段:`Logprobs *bool`、`TopLogprobs *int`、`LogitBias map[string]int`、`ParallelToolCalls *bool`、`ServiceTier string`、`Store *bool`、`Metadata map[string]string`、`PromptCacheKey string`。
- `clone()` 通过 `maps.Clone` 深拷贝 `LogitBias` / `Metadata`,避免副本的修改影响原请求。
- 响应侧新增 `Choice.LogProbs *LogProbs`,配套 `LogProbs{Content, Refusal []TokenLogprob}`、`TokenLogprob{Token, Logprob, Bytes, TopLogprobs}`、`TopLogprob{Token, Logprob, Bytes}`。

其中 `ParallelToolCalls` 后续被 Anthropic 翻译复用为 `disable_parallel_tool_use`(见 [Anthropic 变更记录](../anthropic/anthropic-api-changes.md))。

## 2026-06-02 — 同步 `reasoning_effort` 取值,新增 `verbosity`

**官方变更**:`reasoning_effort` 的取值扩展为 `none` / `minimal` / `low` / `medium` / `high` / `xhigh`(GPT-5.1 默认 `none`);新增 `verbosity` 参数(`low` / `medium` / `high`)控制输出详略。

**wrapper 变更**:新增 `ReasoningEffort*` 与 `Verbosity*` 常量;两个字段保持 `string` 类型以便向非 OpenAI 后端透传。新增 `ChatRequest.Verbosity string`;`clone()` 无需改动(标量字段)。

`ReasoningEffort` 后续被 Anthropic 翻译复用为顶层 `effort`。

## 2026-06-02 — 响应类型对齐(`reasoning_tokens`、`finish_reason` 常量)

**官方变更**:响应 `usage` 新增 `completion_tokens_details.reasoning_tokens`(推理模型内部思考的开销);`finish_reason` 正式包含 `content_filter`(以及遗留的 `function_call`)。

**wrapper 变更**:新增 `Usage.ReasoningTokens int`,从嵌套的 `completion_tokens_details.reasoning_tokens` 解析(**显式顶层字段优先**,与 `cached_tokens` 的处理一致);`Usage.Add` 一并累加。新增 `FinishReasonContentFilter` 与 `FinishReasonFunctionCall`(遗留兼容)常量。

## 2026-06-02 — 支持 `max_completion_tokens`,弃用 `max_tokens`

**官方变更**:OpenAI 在 Chat Completions 上弃用 `max_tokens`;推理模型(o 系列、GPT-5.x)拒绝 `max_tokens`,要求改用 `max_completion_tokens` —— 其上限同时覆盖可见输出 token 与内部推理 token。

**wrapper 变更**:新增 `ChatRequest.MaxCompletionTokens *int`;保留的 `MaxTokens` 标注为已弃用、与推理模型不兼容。Anthropic 翻译器发送 `max_tokens` 时**优先取 `MaxCompletionTokens`**,其次 `MaxTokens`,都缺省时回退 4096。

## [基线] 2026-06-02

- **官方协议**:OpenAI Chat Completions API(`/chat/completions`,无独立版本号,以端点为准)
- **变更摘要**:封装非流式 `ChatCompletion` 与流式 `ChatCompletionStream`(SSE),并以 OpenAI 兼容格式作为规范表示。
