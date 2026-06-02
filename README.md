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

The sync status against the official APIs (target version, change summary, affected files) is recorded in [CHANGES.md](./CHANGES.md).

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

- `ReasoningEffort` → `reasoning_effort`: how many reasoning tokens the model spends. Use the `ReasoningEffort*` constants — `none` / `minimal` / `low` / `medium` / `high` / `xhigh` (GPT-5.1 defaults to `none`).
- `Verbosity` → `verbosity`: how detailed the output is. Use the `Verbosity*` constants — `low` / `medium` / `high`.

Both fields stay plain `string` with `omitempty`, so you can also pass any value a custom OpenAI-compatible backend accepts.

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

#### Response usage

`ChatResponse.Usage` normalizes per-request token counts:

- `PromptTokens` / `CompletionTokens` / `TotalTokens` → the OpenAI-compatible top-level counts.
- `CacheReadTokens` → cache-hit prompt tokens, parsed from OpenAI's nested `prompt_tokens_details.cached_tokens` (an explicit top-level `cache_read_tokens` takes precedence).
- `ReasoningTokens` → tokens spent on a reasoning model's internal thinking, parsed from OpenAI's nested `completion_tokens_details.reasoning_tokens` (an explicit top-level `reasoning_tokens` takes precedence).

`Usage.Add` accumulates all of the above, which is handy when aggregating multi-turn or multi-call usage.

`ChatResponse.Choices[i].FinishReason` mirrors OpenAI's `finish_reason`: `stop`, `length`, `tool_calls`, `content_filter`, and the legacy `function_call`.

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

### Client Options

```go
client, _ := aimodel.NewClient(
    aimodel.WithAPIKey("your-key"),
    aimodel.WithBaseURL("https://api.example.com/v1"),
    aimodel.WithProtocol(aimodel.ProtocolAnthropic),
    aimodel.WithTimeout(30 * time.Second),
)
```

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
