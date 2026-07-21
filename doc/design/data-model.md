# Data Model

The canonical request/response types in `schema.go`. They **are** the OpenAI Chat Completions wire shape (see [../api.md](../api.md) §2), so the OpenAI path serializes them directly and only the Anthropic path translates.

- **Canonical types**: `schema.go`
- **Per-protocol mapping**: [../openai/openai-chat-api.md](../openai/openai-chat-api.md) · [../anthropic/anthropic-chat-api.md](../anthropic/anthropic-chat-api.md)

Pointer types (`*float64` / `*int` / `*bool`) exist to distinguish "unset" from "explicitly zero": `Temperature=0` differs from omitting temperature, and only an explicit `ParallelToolCalls=false` triggers Anthropic's `disable_parallel_tool_use`.

---

## 1. `ChatRequest`

### 1.1 Basics

`Model`, `Messages`, `Temperature`, `TopP`, `TopK`, `N`, `Stop`, `FrequencyPenalty`, `PresencePenalty`, `Seed`, `User`, `ResponseFormat`.

`TopK *int` → `top_k` is top-k truncation sampling (restrict sampling to the K most-likely tokens). It is native to Anthropic, where the translator maps it straight through. OpenAI's Chat Completions has no `top_k`, so it is simply omitted when unset and passed through verbatim when set — compatible backends that accept it honour it, the rest ignore the unknown field.

### 1.2 Token limits

- `MaxCompletionTokens *int` → `max_completion_tokens` — the current OpenAI cap. Its limit covers **both** visible output tokens and internal reasoning tokens, and it is the only token-cap field reasoning models (o-series, GPT-5.x, …) accept. Prefer it.
- `MaxTokens *int` → `max_tokens` — **deprecated** by OpenAI and rejected outright by reasoning models. Keep it only for older / non-reasoning models that still accept it.

Both are `omitempty`. Anthropic always requires `max_tokens`, so its translator prefers `MaxCompletionTokens` over `MaxTokens` and falls back to 4096 when neither is set.

### 1.3 Reasoning & thinking

- `ReasoningEffort string` → OpenAI `reasoning_effort` / Anthropic `output_config.effort`: how many reasoning tokens the model spends. Constants: `ReasoningEffortNone` / `Minimal` / `Low` / `Medium` / `High` / `XHigh` (GPT-5.1 defaults to `none`).
- `Verbosity string` → OpenAI `verbosity`: how detailed the output is. Constants: `VerbosityLow` / `Medium` / `High`.
- `Thinking *Thinking` → `{Type, BudgetTokens, Display}`. `Type` is `enabled` / `disabled` / `adaptive` (the model sizes its own thinking); `BudgetTokens` is **deprecated** in favour of `ReasoningEffort` or `adaptive`; `Display: "omitted"` suppresses thinking content to speed up streaming.

`ReasoningEffort` and `Verbosity` stay plain `string` rather than enums, so any value a custom OpenAI-compatible backend accepts passes through.

### 1.4 Tools

`Tools []Tool`, `ToolChoice any` (`"auto"` / `"required"` / `"none"`, or `{"type":"function","function":{"name":…}}`), `ParallelToolCalls *bool`. See [tool-use.md](./tool-use.md).

### 1.5 Streaming

`Stream bool`, `StreamOptions *StreamOptions{IncludeUsage}`. See [streaming.md](./streaming.md).

### 1.6 Observability & routing (OpenAI)

| Field | Wire | Meaning |
|---|---|---|
| `Logprobs *bool` / `TopLogprobs *int` | `logprobs` / `top_logprobs` | Per-token log probabilities and the N most-likely alternatives per position. `TopLogprobs` (0–20) requires `Logprobs=true`. |
| `LogitBias map[string]int` | `logit_bias` | Per-token-ID bias in `[-100, 100]`; keys are token IDs as strings. |
| `ParallelToolCalls *bool` | `parallel_tool_calls` | Whether the model may emit multiple tool calls in one turn; server-side default is true. |
| `ServiceTier string` | `service_tier` | Latency/throughput tier (`auto` / `default` / `flex` / `priority`); plain `string` for pass-through. |
| `Store *bool` | `store` | Persist the completion for dashboards / evals. |
| `Metadata map[string]string` | `metadata` | Up to 16 key/value pairs returned alongside a stored completion. |
| `PromptCacheKey string` | `prompt_cache_key` | Route requests sharing a cacheable prefix to the same cache. |

### 1.7 Multimodal

`Modalities []string` (e.g. `["text","audio"]`) and `Audio *AudioConfig{Voice, Format}`. Requesting audio output requires `"audio"` in `Modalities`; the generated audio comes back on `Message.Audio`.

### 1.8 Anthropic pass-through

Optional and `omitempty`, so a zero-value request's wire JSON is unchanged:

- `Container string` → `container`: reuse a server-side execution container across requests, keeping its code-execution state alive. Feed back the ID from a previous `ChatResponse.Container` / `StreamChunk.Container`.
- `InferenceGeo string` → `inference_geo`: pin the inference geography for data residency (e.g. `"us"` / `"eu"`); plain `string` for pass-through.

### 1.9 Anthropic struct-local switches

Marked `json:"-"`, so they **never** appear on the canonical (OpenAI-shape) body — only the Anthropic translator reads them:

- `AutoCache bool` + `AutoCacheTTL string` — see [prompt-caching.md](./prompt-caching.md).

### 1.10 `clone()`

Every dispatch deep-copies the request first, so the SDK's own rewrites (`Stream`, default model) never mutate the caller's object. `clone()` duplicates the `Messages` / `Stop` / `Modalities` / `Tools` slices, the `LogitBias` / `Metadata` maps, and each tool's `AllowedCallers` / `InputExamples` slices. Elements themselves stay shallow — dynamic `any` values (`Function.Parameters`, `InputExamples` entries) are shared by contract.

---

## 2. Messages & content

```go
type Message struct {
    Role       Role       // system / user / assistant / tool
    Content    Content
    Thinking   string     `json:"reasoning_content,omitempty"`
    ToolCallID string
    ToolCalls  []ToolCall
    Audio      *MessageAudio
    CacheBreakpoint bool          `json:"-"`
    ExtraBlocks []json.RawMessage `json:"-"`
}
```

`Content` is a **polymorphic wrapper**: it privately holds `text string` and `parts []ContentPart`, and its custom `MarshalJSON` / `UnmarshalJSON` switch between the two wire shapes — a bare string, or an array of content blocks.

```go
aimodel.NewTextContent("hello")                          // → "hello"
aimodel.NewPartsContent(                                 // → [{...},{...}]
    aimodel.ContentPart{Type: "text", Text: "Describe this image"},
    aimodel.ContentPart{Type: "image_url", ImageURL: &aimodel.ImageURL{URL: dataURI}},
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
| `input_audio` | `InputAudio{Data (base64), Format ("wav"/"mp3")}` |
| `file` | `FilePart{FileID}`, or `{Filename, FileData}` for inline base64 contents |

`ExtraBlocks` is the verbatim escape hatch for protocol content blocks this wrapper does not model — see [streaming.md](./streaming.md) §4.

---

## 3. `ChatResponse`

```go
type ChatResponse struct {
    ID, Object, Model string
    Created int64
    Choices []Choice
    Usage   Usage
    Error   *Error
    Container *ResponseContainer  // Anthropic server-side execution container
}

// ExpiresAt stays the server-supplied string: no expiry parsing,
// no auto-renewal, no retry.
type ResponseContainer struct {
    ID        string
    ExpiresAt string
}

type Choice struct {
    Index        int
    Message      Message
    FinishReason FinishReason
    LogProbs     *LogProbs     // present when the request set Logprobs
    StopDetails  *StopDetails  // Anthropic structured stop classification
}
```

### 3.1 `FinishReason`

Mirrors OpenAI's `finish_reason`: `stop` / `length` / `tool_calls` / `content_filter`, plus the legacy `function_call`. Anthropic stop reasons with no OpenAI equivalent **pass through verbatim** as named constants rather than being folded into an existing value:

| Constant | Meaning |
|---|---|
| `FinishReasonModelContextWindowExceeded` | Input + output exceeded the model's context window — distinct from hitting the requested `max_tokens` (`length`). |
| `FinishReasonRefusal` | Streaming classifiers intervened on a potential policy violation. |
| `FinishReasonPauseTurn` | A long-running / server-tool turn was paused; the client may replay it to continue. |

Folding these into `content_filter` / `length` would destroy exactly the information a caller decides on (replay? switch model? change the prompt?). Treat any non-canonical `FinishReason` as an opaque string.

### 3.2 `StopDetails`

`{Type, Category, Explanation}`, all `omitempty`, returned alongside `stop_reason:"refusal"` — e.g. `{type:"refusal", category:"cyber", explanation:…}`. It surfaces on `Choice.StopDetails` (non-streaming) and `StreamChunkChoice.StopDetails` (the terminal `message_delta`), and is `nil` when absent. `Explanation` is best-effort and not guaranteed stable across model versions.

### 3.3 `LogProbs`

When `Logprobs` is true, `Choice.LogProbs` carries the parsed `content` / `refusal` token log probabilities: `LogProbs{Content, Refusal []TokenLogprob}`, `TokenLogprob{Token, Logprob, Bytes, TopLogprobs}`, `TopLogprob{Token, Logprob, Bytes}`.

### 3.4 `MessageAudio`

`Message.Audio` (`{ID, Data, Transcript, ExpiresAt}`) carries assistant-generated audio when audio output was requested. Never present on the Anthropic path.

---

## 4. `Usage`

```go
type Usage struct {
    PromptTokens, CompletionTokens, TotalTokens int
    CacheReadTokens    int  // cache-hit prompt tokens
    CacheWriteTokens   int  // total prompt-cache writes (Anthropic)
    CacheWrite5mTokens int  // split by TTL
    CacheWrite1hTokens int
    ReasoningTokens    int  // reasoning model's internal thinking
    ServerToolUse *ServerToolUse  // server-side tool invocations (Anthropic)
    InferenceGeo  string          // where inference ran (Anthropic)
    ServiceTier   string          // tier that served the request (Anthropic)
}

type ServerToolUse struct {
    WebSearchRequests int
    WebFetchRequests  int
}
```

Key points:

- **Cache read/write tokens are subsets of `PromptTokens`**, and `ReasoningTokens` a subset of `CompletionTokens`. They are surfaced separately for observability — do not add them again when computing cost.
- `UnmarshalJSON` promotes nested protocol details to the top-level canonical fields: `prompt_tokens_details.cached_tokens` → `CacheReadTokens`; `completion_tokens_details.reasoning_tokens` (OpenAI) / `output_tokens_details.thinking_tokens` (Anthropic) → `ReasoningTokens`. **An explicit top-level field wins** — the nested value is only used when the top-level one is 0.
- `CacheWrite5mTokens + CacheWrite1hTokens == CacheWriteTokens` whenever Anthropic returns the breakdown.
- Both `ServerToolUse` counts are individually `omitempty` (omitted from JSON at 0); the whole object is `nil` when no server tool ran.
- `Add(other)` accumulates every count, including `ServerToolUse`, which makes multi-turn / multi-model aggregation straightforward. `InferenceGeo` and `ServiceTier` describe **one** request, so `Add` leaves them untouched.
- OpenAI has no cache-write accounting, so the `CacheWrite*` fields stay 0 on that path.
