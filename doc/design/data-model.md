# Data Model

The canonical request/response types in `ais/schema.go` model only semantics with verified mappings in at least two providers (see [../architecture.md](../architecture.md) §2).

- **Canonical types**: `ais/schema.go`
- **Per-protocol mapping**: [../openai/openai-chat-api.md](../openai/openai-chat-api.md) · [../anthropic/anthropic-message-api.md](../anthropic/anthropic-message-api.md)

Pointer types (`*float64` / `*int` / `*bool`) exist to distinguish "unset" from "explicitly zero": `Temperature=0` differs from omitting temperature, and only an explicit `ParallelToolCalls=false` triggers Anthropic's `disable_parallel_tool_use`.

---

## 1. `ChatRequest`

### 1.1 Basics

`Model`, `Messages`, `Temperature`, `TopP`, `TopK`, `Stop`, `ResponseFormat`.

`TopK *int` → `top_k` is top-k truncation sampling (restrict sampling to the K most-likely tokens). It is native to Anthropic, where the translator maps it straight through. OpenAI's Chat Completions has no `top_k`, so it is simply omitted when unset and passed through verbatim when set — compatible backends that accept it honour it, the rest ignore the unknown field.

### 1.2 Token limits

- `MaxCompletionTokens *int` → `max_completion_tokens` — the current OpenAI cap. Its limit covers **both** visible output tokens and internal reasoning tokens, and it is the only token-cap field reasoning models (o-series, GPT-5.x, …) accept. Prefer it.
- `MaxTokens *int` → `max_tokens` — **deprecated** by OpenAI and rejected outright by reasoning models. Keep it only for older / non-reasoning models that still accept it.

Both are `omitempty`. Anthropic always requires `max_tokens`, so its translator prefers `MaxCompletionTokens` over `MaxTokens` and falls back to 4096 when neither is set.

### 1.3 Reasoning & thinking

- `ReasoningEffort string` → OpenAI `reasoning_effort` / Anthropic `output_config.effort`: how many reasoning tokens the model spends. Constants: `ReasoningEffortNone` / `Minimal` / `Low` / `Medium` / `High` / `XHigh` (GPT-5.1 defaults to `none`).
- `Thinking *Thinking` → `{Type, BudgetTokens, Display}`. `Type` is `enabled` / `disabled` / `adaptive` (the model sizes its own thinking); `BudgetTokens` is **deprecated** in favour of `ReasoningEffort` or `adaptive`; `Display: "omitted"` suppresses thinking content to speed up streaming.

`ReasoningEffort` stays a plain `string` so providers can map newly introduced shared values without a canonical enum migration.

### 1.4 Tools

`Tools []Tool`, `ToolChoice any` (`"auto"` / `"required"` / `"none"`, or `{"type":"function","function":{"name":…}}`), `ParallelToolCalls *bool`. See [tool-use.md](./tool-use.md).

### 1.5 Streaming

`Stream bool`. The OpenAI provider adds its wire-only `stream_options.include_usage=true` for streaming requests. See [streaming.md](./streaming.md).

### 1.8 Provider extensions (`Extensions`)

`Extensions ais.Extensions` (`json:"-"`) is the unified provider extension channel: a map keyed by registered provider name holding one provider-defined value per namespace. It exists on `ChatRequest`, `Message`, `Tool`, `ChatResponse`, `Choice`, `Usage`, `StreamChunk` and `StreamChunkChoice`, and never appears in canonical JSON. See [../architecture.md](../architecture.md) §2 for the contract, and `provider/anthropic/extension.go` for the built-in Anthropic surface (`RequestExtension` with `AutoCache` / `AutoCacheTTL` / `Container` / `InferenceGeo`, plus the message / tool / response extensions).

### 1.9 `Clone()`

Every dispatch deep-copies the request first, so the SDK's own rewrites (`Stream`, default model) never mutate the caller's object. `Clone()` duplicates the retained `Messages`, `Stop` and `Tools` slices and the `Extensions` map at every node (request, each message, each tool). Dynamic `any` values and extension values remain shared read-only configuration.

---

## 2. Messages & content

```go
type Message struct {
    Role       Role       // system / user / assistant / tool
    Content    Content
    Thinking   string     `json:"reasoning_content,omitempty"`
    ToolCallID string
    ToolCalls  []ToolCall
    Extensions ais.Extensions `json:"-"`  // provider extension channel
}
```

`Content` is a **polymorphic wrapper**: it privately holds `text string` and `parts []ContentPart`, and its custom `MarshalJSON` / `UnmarshalJSON` switch between the two wire shapes — a bare string, or an array of content blocks.

```go
ais.NewTextContent("hello")                          // → "hello"
ais.NewPartsContent(                                 // → [{...},{...}]
    ais.ContentPart{Type: "text", Text: "Describe this image"},
    ais.ContentPart{Type: "image_url", ImageURL: &ais.ImageURL{URL: dataURI}},
)
```

- `Content.Text()` — returns the text directly; for multimodal content it concatenates every `text` part.
- `Content.Parts()` — returns the content blocks for multimodal content, `nil` for plain text.
- Unmarshalling discriminates on the first byte: `"` → string, `[` → array, `null` → both cleared.

`ContentPart` selects exactly one payload by `Type`:

| `Type` | Payload |
|---|---|
| `text` | `Text string` |
| `image_url` | `ImageURL{URL, Detail}` |

On the Anthropic path, native content blocks the canonical layer does not model are preserved verbatim on the message's extension (`anthropic.MessageExtensionOf(&msg).ExtraBlocks`) — see [streaming.md](./streaming.md) §4.

---

## 3. `ChatResponse`

```go
type ChatResponse struct {
    ID, Object, Model string
    Created int64
    Choices []Choice
    Usage   Usage
    Error   *Error
    Extensions ais.Extensions `json:"-"`  // provider response metadata
}

type Choice struct {
    Index        int
    Message      Message
    FinishReason FinishReason
    Extensions ais.Extensions `json:"-"`  // provider per-choice metadata
}
```

### 3.1 `FinishReason`

Mirrors OpenAI's `finish_reason`: `stop` / `length` / `tool_calls` / `content_filter`, plus the legacy `function_call`. `FinishReason` stays an open string: providers pass values with no canonical equivalent through **verbatim** rather than folding them into an existing value, and name their own convenience constants — e.g. `anthropic.FinishReasonRefusal` / `FinishReasonPauseTurn` / `FinishReasonModelContextWindowExceeded` for Anthropic's pass-through stop reasons. Folding those into `content_filter` / `length` would destroy exactly the information a caller decides on (replay? switch model? change the prompt?). Treat any non-canonical `FinishReason` as an opaque string.

### 3.2 Provider response metadata

Response information with no cross-provider consensus rides the `Extensions` channel and is read through the owning provider's typed accessors. For the built-in Anthropic provider:

| Information | Accessor |
|---|---|
| Structured stop classification (`stop_details`, e.g. the refusal category) | `anthropic.ChoiceExtensionOf(&resp.Choices[0]).StopDetails` (unary) / `anthropic.ChunkChoiceExtensionOf(&chunk.Choices[0])` (terminal stream chunk) |
| Server-side execution container | `anthropic.ResponseExtensionOf(resp).Container` (unary) / `anthropic.ChunkExtensionOf(chunk)` (the chunk carrying `message_start`); feed the ID back via `anthropic.RequestExtension.Container` |
| Cache writes, server-tool counts, inference geography | `anthropic.UsageExtensionOf(&resp.Usage)` — see §4 |

Each accessor returns `nil` when the response carries no such metadata.

---

## 4. `Usage`

```go
type Usage struct {
    PromptTokens, CompletionTokens, TotalTokens int
    CacheReadTokens int     // cache-hit prompt tokens (OpenAI + Anthropic)
    ReasoningTokens int     // reasoning model's internal thinking (OpenAI + Anthropic)
    ServiceTier     string  // tier that served the request (OpenAI + Anthropic)
    Extensions ais.Extensions `json:"-"`  // provider usage accounting
}
```

The canonical fields are exactly the counts with a cross-provider mapping. Anthropic-only accounting comes back on `anthropic.UsageExtensionOf(&u)`:

```go
type UsageExtension struct {  // package anthropic
    CacheWriteTokens   int  // total prompt-cache writes
    CacheWrite5mTokens int  // split by TTL
    CacheWrite1hTokens int
    ServerToolUse *ServerToolUse  // server-side tool invocations
    InferenceGeo  string          // where inference ran
}
```

Key points:

- **Cache read/write tokens are subsets of `PromptTokens`**, and `ReasoningTokens` a subset of `CompletionTokens`. They are surfaced separately for observability — do not add them again when computing cost.
- `UnmarshalJSON` promotes nested protocol details to the top-level canonical fields: `prompt_tokens_details.cached_tokens` → `CacheReadTokens`; `completion_tokens_details.reasoning_tokens` (OpenAI) / `output_tokens_details.thinking_tokens` (Anthropic) → `ReasoningTokens`. **An explicit top-level field wins** — the nested value is only used when the top-level one is 0.
- `CacheWrite5mTokens + CacheWrite1hTokens == CacheWriteTokens` whenever Anthropic returns the breakdown; the extension is absent when the response carries no Anthropic-only accounting (OpenAI path always).
- `Add(other)` accumulates the canonical counts, which makes multi-turn / multi-model aggregation straightforward. `ServiceTier` and `Extensions` describe **one** request, so `Add` leaves them untouched — aggregate provider-specific accounting by reading each response's extension.
