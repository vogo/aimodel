# aimodel

[![Build](https://github.com/vogo/aimodel/actions/workflows/build.yml/badge.svg)](https://github.com/vogo/aimodel/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/vogo/aimodel/branch/main/graph/badge.svg)](https://codecov.io/gh/vogo/aimodel)

A Go SDK for AI model APIs with multi-protocol support (OpenAI, Anthropic). Zero external dependencies.

## Design Scope

This SDK is a **thin API wrapper** — it translates requests, manages connections, and normalizes responses across protocols. 
It intentionally does **not** include retry, rate limiting, request validation, caching / persistence, logging / metrics.
This keeps the SDK minimal and composable. Control mechanisms belong in the layer above, where you have full context over your application's requirements.

## Official API References

aimodel is a thin wrapper over the following official protocols. Each wrapper maps one-to-one to its official documentation:

| Protocol | Official docs | Wrapper code |
|------|------|------|
| OpenAI (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | `openai_chat.go` / `openai_stream.go` |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | `anthropic.go` / `anthropic_chat.go` / `anthropic_stream.go` |

The sync status against the official APIs (target version, change summary) is recorded in [CHANGES.md](./CHANGES.md).

**Maintenance convention**: when an official API changes, update all three in sync — the wrapper code, this document, and CHANGES.md — keeping them consistent and continuously up to date.


## Usage

```go
import "github.com/vogo/aimodel"
```

Set env vars `AI_API_KEY` and `AI_BASE_URL` (or `OPENAI_API_KEY` / `OPENAI_BASE_URL`).

### Chat Completion

```go
client, _ := aimodel.NewClient()

resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model: aimodel.ModelOpenaiGPT4o,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})

fmt.Println(resp.Choices[0].Message.Content.Text())
```

#### Token limits

- `MaxCompletionTokens` → `max_completion_tokens`: the current OpenAI cap. Its limit covers both visible output tokens and internal reasoning tokens, and it is the **only** token-cap field accepted by reasoning models (o-series, GPT-5.x, …). Prefer it.
- `MaxTokens` → `max_tokens`: **deprecated** by OpenAI and rejected by reasoning models. Keep it only for older / non-reasoning models that still accept it.

Both fields are `*int` with `omitempty`. For the Anthropic protocol (which always uses `max_tokens`), the translator prefers `MaxCompletionTokens` over `MaxTokens`, defaulting to 4096 when neither is set.

#### Reasoning effort & verbosity

- `ReasoningEffort` → OpenAI `reasoning_effort` / Anthropic top-level `effort`: how many reasoning tokens the model spends. Use the `ReasoningEffort*` constants — `none` / `minimal` / `low` / `medium` / `high` / `xhigh` (GPT-5.1 defaults to `none`). On the Anthropic protocol this `effort` field (GA 2026-02-05) supersedes `thinking.budget_tokens`.
- `Verbosity` → `verbosity`: how detailed the output is. Use the `Verbosity*` constants — `low` / `medium` / `high`.

Both fields stay plain `string` with `omitempty`, so you can also pass any value a custom OpenAI-compatible backend accepts.

For Anthropic extended thinking, `Thinking` carries `Type` (`enabled` / `disabled` / `adaptive`), the now-deprecated `BudgetTokens` (prefer `ReasoningEffort` or `Type:"adaptive"`), and `Display` — set `Display: "omitted"` to suppress thinking content and speed up streaming.

```go
resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model:           aimodel.ModelOpenaiGPT4o,
    ReasoningEffort: aimodel.ReasoningEffortHigh,
    Verbosity:       aimodel.VerbosityLow,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
```

#### Common request fields

All optional and `omitempty`, mapping one-to-one to OpenAI's Chat Completions parameters:

- `TopK *int` → `top_k`: top-k truncation sampling (restrict sampling to the K most-likely tokens). Native to Anthropic, where `toAnthropicRequest` maps it straight through. OpenAI's Chat Completions has no `top_k`, so it is simply omitted when unset and passed through verbatim when set.
- `Logprobs *bool` → `logprobs`, `TopLogprobs *int` → `top_logprobs`: per-token log probabilities (and the N most-likely alternatives per position) for observability.
- `LogitBias map[string]int` → `logit_bias`: per-token-ID bias in `[-100, 100]`.
- `ParallelToolCalls *bool` → `parallel_tool_calls`: whether the model may emit multiple tool calls in one turn. On Anthropic an explicit `false` maps to `tool_choice.disable_parallel_tool_use:true` (defaulting the choice to `{type:"auto"}` when none is named and tools are present; never attached to `{type:"none"}`).
- `ServiceTier string` → `service_tier`: latency/throughput tier (e.g. `auto` / `default` / `flex` / `priority`); plain `string` for pass-through.
- `Store *bool` → `store`, `Metadata map[string]string` → `metadata`: persist the completion and attach up to 16 key/value pairs.
- `PromptCacheKey string` → `prompt_cache_key`: route requests sharing a prefix to the same cache to improve hit rates.
- `AutoCache bool` + `AutoCacheTTL string` (Anthropic only; both struct-local, never on the OpenAI wire): enable Anthropic's *automatic caching* — a single request-root `cache_control` (`{type:"ephemeral"}`, or with `ttl` `"1h"`; empty TTL keeps the default 5-minute cache). The server caches the last cacheable block and advances the breakpoint forward as the conversation grows, no per-block markers needed. It coexists with the explicit per-block `Message.CacheBreakpoint` / `Tool.CacheBreakpoint` flags. OpenAI-compatible backends ignore it.

`clone()` deep-copies the `LogitBias` and `Metadata` maps, so mutating a cloned request never affects the original.

When `Logprobs` is `true`, each `ChatResponse.Choices[i].LogProbs` (a `*LogProbs`) carries the parsed `content` / `refusal` token log probabilities, each token exposing `Logprob`, `Bytes`, and `TopLogprobs`.

```go
logprobs := true
topLogprobs := 5
resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model:          aimodel.ModelOpenaiGPT4o,
    Logprobs:       &logprobs,
    TopLogprobs:    &topLogprobs,
    ServiceTier:    "priority",
    PromptCacheKey: "tenant-42",
    Metadata:       map[string]string{"env": "prod"},
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
```

#### Response usage

`ChatResponse.Usage` normalizes per-request token counts:

- `PromptTokens` / `CompletionTokens` / `TotalTokens` → the OpenAI-compatible top-level counts.
- `CacheReadTokens` → cache-hit prompt tokens, parsed from OpenAI's nested `prompt_tokens_details.cached_tokens` (an explicit top-level `cache_read_tokens` takes precedence).
- `CacheWriteTokens` → prompt-cache write tokens (Anthropic's `cache_creation_input_tokens`, total). `CacheWrite5mTokens` / `CacheWrite1hTokens` break it down by TTL (Anthropic's `usage.cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}`, summing to `CacheWriteTokens`). Cache read/write tokens are subsets of `PromptTokens`, surfaced separately for observability; OpenAI leaves them at 0.
- `ReasoningTokens` → tokens spent on a reasoning model's internal thinking, parsed from OpenAI's nested `completion_tokens_details.reasoning_tokens` (an explicit top-level `reasoning_tokens` takes precedence).

`Usage.Add` accumulates all of the above, which is handy when aggregating multi-turn or multi-call usage.

`ChatResponse.Choices[i].FinishReason` mirrors OpenAI's `finish_reason`: `stop`, `length`, `tool_calls`, `content_filter`, and the legacy `function_call`. Anthropic also emits stop reasons with no OpenAI canonical equivalent; these pass through verbatim and are named for readability: `model_context_window_exceeded` (input + output exceeded the model's context window — distinct from `length`), `refusal` (streaming classifiers intervened on a policy violation), and `pause_turn` (a long-running/server-tool turn was paused and may be replayed to continue).

When a refusal carries a classification, both `ChatResponse.Choices[i].StopDetails` (non-streaming) and `StreamChunk.Choices[i].StopDetails` (the terminal `message_delta`) expose it as `StopDetails{Type, Category, Explanation}` (e.g. `{type:"refusal", category:"cyber", explanation:"…"}`); the field is `nil` when absent. `Explanation` is best-effort and not guaranteed stable across model versions.

```go
maxCompletionTokens := 1024
resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model:               aimodel.ModelOpenaiO3, // reasoning model
    MaxCompletionTokens: &maxCompletionTokens,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
```

### Multimodal input & audio output

`Content` is polymorphic — `NewTextContent` for plain text, `NewPartsContent` for a multimodal array. Each `ContentPart` carries one payload selected by `Type`:

- `text` → `Text`
- `image_url` → `ImageURL{URL, Detail}`
- `input_audio` → `InputAudio{Data, Format}` — base64 audio plus encoding (`wav` / `mp3`)
- `file` → `FilePart{FileID}` (reference an uploaded file) or `FilePart{Filename, FileData}` (inline base64 contents)

To request audio output, set `Modalities` (e.g. `["text", "audio"]`) and `Audio` (`AudioConfig{Voice, Format}`); the generated audio comes back on `resp.Choices[i].Message.Audio` (`MessageAudio{ID, Data, Transcript, ExpiresAt}`).

```go
resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model:      aimodel.ModelOpenaiGPT4o,
    Modalities: []string{"text", "audio"},
    Audio:      &aimodel.AudioConfig{Voice: "alloy", Format: "wav"},
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewPartsContent(
            aimodel.ContentPart{Type: "text", Text: "What is said in this clip?"},
            aimodel.ContentPart{Type: "input_audio", InputAudio: &aimodel.InputAudio{
                Data: base64Audio, Format: "wav",
            }},
        )},
    },
})

if a := resp.Choices[0].Message.Audio; a != nil {
    fmt.Println(a.Transcript) // generated audio's text transcript
}
```

### Streaming

```go
stream, _ := client.ChatCompletionStream(context.Background(), &aimodel.ChatRequest{
    Model: aimodel.ModelOpenaiGPT4o,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
defer stream.Close()

for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    fmt.Print(chunk.Choices[0].Delta.Content.Text())
}
```

### Anthropic Protocol

Use `WithProtocol` to select the Anthropic Messages API:

```go
client, _ := aimodel.NewClient(
    aimodel.WithAPIKey("sk-ant-xxx"),
    aimodel.WithProtocol(aimodel.ProtocolAnthropic),
)

resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model: aimodel.ModelAnthropicClaude4Sonnet,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
```

The same `ChatCompletion` / `ChatCompletionStream` methods work for all protocols — routing is handled internally.

**System message translation**: only the *leading* run of `RoleSystem` messages (those before the first user/assistant turn) is hoisted into Anthropic's top-level `system` field. A `RoleSystem` message that appears mid-conversation is kept inline as a `role:"system"` message in its original position (supported since Opus 4.8), so you can switch instructions mid-session without losing prompt-cache hits.

**Tool choice translation**: `"auto"` → `{type:"auto"}`, `"required"` → `{type:"any"}`, `"none"` → `{type:"none"}` (explicitly forbid any call — distinct from omitting the field, which lets the model choose), and a specific function → `{type:"tool", name:...}`. `ParallelToolCalls: false` adds `disable_parallel_tool_use:true` to the resulting choice (see the request-field note above).

### Client Options

```go
client, _ := aimodel.NewClient(
    aimodel.WithAPIKey("your-key"),
    aimodel.WithBaseURL("https://api.example.com/v1"),
    aimodel.WithProtocol(aimodel.ProtocolAnthropic),
    aimodel.WithTimeout(30 * time.Second),
)
```

**Anthropic header options** (no effect under the OpenAI protocol):

```go
client, _ := aimodel.NewClient(
    aimodel.WithAPIKey("sk-ant-xxx"),
    aimodel.WithProtocol(aimodel.ProtocolAnthropic),
    // Opt into beta capabilities via the "anthropic-beta" header.
    // Values append across calls and are comma-joined on the wire.
    aimodel.WithAnthropicBeta("context-1m-2025-08-07"),
    // Override the "anthropic-version" header (default "2023-06-01").
    aimodel.WithAnthropicVersion("2023-06-01"),
)
```

`WithAnthropicBeta` ignores empty strings and, with no value configured, the `anthropic-beta` header is omitted entirely (default behavior). `WithAnthropicVersion("")` keeps the default version. These are the infrastructure for enabling beta features (compaction, context-editing, structured-outputs, fast-mode, advisor, …).

### Multi-Model Compose

The `composes` package dispatches requests across multiple backends with failover, random, or weighted strategies:

```go
import "github.com/vogo/aimodel/composes"

openai, _ := aimodel.NewClient(aimodel.WithAPIKey("sk-openai"), aimodel.WithBaseURL("https://api.openai.com/v1"))
anthropic, _ := aimodel.NewClient(
    aimodel.WithAPIKey("sk-ant"),
    aimodel.WithProtocol(aimodel.ProtocolAnthropic),
)

cc, _ := composes.NewComposeClient(composes.StrategyFailover, []composes.ModelEntry{
    {Name: "gpt-4o", Client: openai},
    {Name: "claude-sonnet", Client: anthropic},
})

resp, _ := cc.ChatCompletion(ctx, req)
```
