# provider/anthropic

The Anthropic Messages API provider: the registered `anthropic` provider used by the unified
`aimodel.Client`, plus a public **native** client over Anthropic's own wire types.

Importing this package registers the provider under `Name` (`"anthropic"`); select it from the
root package with `aimodel.WithProvider(anthropic.Name)`.

| Endpoint | Native client | Canonical translation |
|---|---|---|
| `POST {baseURL}/v1/messages` | `Client.Messages` / `MessagesStream` | yes — `ais.ChatRequest` ⇄ `MessagesRequest` |

The base URL defaults to `https://api.anthropic.com`, the API version header to `2023-06-01`.
Wire mapping details: [doc/anthropic/anthropic-message-api.md](../../doc/anthropic/anthropic-message-api.md).

## What this package exposes

### Provider registration

| Symbol | Purpose |
|---|---|
| `Name` | Registered provider name |
| `New(ais.Config) (ais.ChatProvider, error)` | Provider constructor called by the registry. Base URL is optional; `cfg.Options`, when set, must be `Options` |
| `Options` | Client-level vendor options — `Beta` (`anthropic-beta` header), `Version` (`anthropic-version`), `UserProfileID` (`anthropic-user-profile-id`). Pass via `aimodel.WithProviderOptions` |

### Native client

| Symbol | Purpose |
|---|---|
| `NewClient(apiKey string, options ...ClientOption) *Client` | Native Messages client |
| `WithBaseURL`, `WithHTTPClient`, `WithVersion`, `WithBeta`, `WithUserProfileID` | Client options |
| `(*Client).Messages(ctx, *MessagesRequest) (*MessagesResponse, error)` | Non-streaming Messages call |
| `(*Client).MessagesStream(ctx, *MessagesRequest) (*MessageStream, error)` | Streaming Messages call; `Recv()` returns `*StreamEvent`, ends with `io.EOF` |
| `HTTPError` | Non-2xx error, retaining the bounded raw body |

### Wire types

`MessagesRequest` / `MessagesResponse`, `MessagesMessage`, `ContentBlock` / `ResponseContentBlock`,
`ContentSource`, `MessagesTool`, `ToolChoice`, `MessagesThinking`, `OutputConfig` / `OutputFormat`,
`CacheControl`, `MessagesUsage` (with `CacheCreation`, `OutputTokensDetails`, `ServerToolUse`),
`StopDetails`, `ResponseContainer`, and the SSE event types
(`StreamEvent`, `MessageStartEvent`, `ContentBlockStartEvent`, `ContentBlockDeltaEvent`,
`MessageDeltaEvent`).

Fidelity guarantee: every response block and stream delta keeps its verbatim JSON in `Raw`, so
server-tool results, citations and future block types survive without a lossy round trip.

### Extension channel (canonical requests / responses)

Anthropic-only semantics never enter `ais`; they ride the `ais.Extensions` channel through
typed helpers owned by this package.

Request side — attach with `Extend*`, read back with `*Of`:

| Type | Attach / read | Carries |
|---|---|---|
| `RequestExtension` | `ExtendRequest` / `RequestExtensionOf` | `AutoCache`, `AutoCacheTTL`, `Container`, `InferenceGeo` |
| `MessageExtension` | `ExtendMessage` / `MessageExtensionOf` | `CacheBreakpoint`; on the response side, unmodelled content blocks in `ExtraBlocks` |
| `ToolExtension` | `ExtendTool` / `ToolExtensionOf` | `CacheBreakpoint`, `DeferLoading`, `AllowedCallers`, `EagerInputStreaming`, `InputExamples` |

Response side — written by this provider, read with:

| Reader | Returns |
|---|---|
| `ResponseExtensionOf(*ais.ChatResponse)` / `ChunkExtensionOf(*ais.StreamChunk)` | `*ResponseExtension` — the server-side execution `Container` |
| `ChoiceExtensionOf(*ais.Choice)` / `ChunkChoiceExtensionOf(*ais.StreamChunkChoice)` | `*ChoiceExtension` — the structured `StopDetails` |
| `UsageExtensionOf(*ais.Usage)` | `*UsageExtension` — cache-write tokens (with 5m/1h split), `ServerToolUse`, `InferenceGeo` |

### Pass-through finish reasons

`FinishReasonModelContextWindowExceeded`, `FinishReasonRefusal`, `FinishReasonPauseTurn` — Anthropic
stop reasons with no canonical equivalent, surfaced verbatim instead of folded into
`stop` / `length` / `content_filter`. Treat any non-canonical `ais.FinishReason` as opaque.

## Usage

### Unified client (canonical types)

```go
import (
    "github.com/vogo/aimodel"
    "github.com/vogo/aimodel/ais"
    "github.com/vogo/aimodel/provider/anthropic"
)

client, _ := aimodel.NewClient(
    aimodel.WithProvider(anthropic.Name),
    aimodel.WithAPIKey(apiKey),
)

resp, _ := client.ChatCompletion(ctx, &ais.ChatRequest{
    Model: ais.ModelAnthropicClaudeSonnet5,
    Messages: []ais.Message{
        {Role: ais.RoleUser, Content: ais.NewTextContent("Hello!")},
    },
})

fmt.Println(resp.Choices[0].Message.Content.Text())
```

Client-level vendor headers:

```go
client, _ := aimodel.NewClient(
    aimodel.WithProvider(anthropic.Name),
    aimodel.WithAPIKey(apiKey),
    aimodel.WithProviderOptions(anthropic.Options{Beta: []string{"context-1m-2025-08-07"}}),
)
```

### Prompt caching and other vendor-only parameters

```go
req := &ais.ChatRequest{Model: model, Messages: messages}

// Automatic caching: one cache_control at the request root; the server advances the breakpoint.
anthropic.ExtendRequest(req, &anthropic.RequestExtension{AutoCache: true, AutoCacheTTL: "1h"})

// Or an explicit per-message breakpoint.
anthropic.ExtendMessage(&req.Messages[0], &anthropic.MessageExtension{CacheBreakpoint: true})

resp, _ := client.ChatCompletion(ctx, req)

if usage := anthropic.UsageExtensionOf(&resp.Usage); usage != nil {
    fmt.Println(usage.CacheWriteTokens, usage.CacheWrite1hTokens)
}
```

See [doc/design/prompt-caching.md](../../doc/design/prompt-caching.md) for the full caching API.

### Reading unmodelled response blocks

Server-tool blocks and text blocks carrying citations are preserved verbatim:

```go
if ext := anthropic.MessageExtensionOf(&resp.Choices[0].Message); ext != nil {
    for _, block := range ext.ExtraBlocks {
        fmt.Printf("extra block: %s\n", block)
    }
}
```

### Native Messages call

```go
client := anthropic.NewClient(apiKey) // add anthropic.WithBaseURL(...) for a proxy

resp, err := client.Messages(ctx, &anthropic.MessagesRequest{
    Model:     model,
    MaxTokens: 1024,
    Messages: []anthropic.MessagesMessage{{
        Role:    "user",
        Content: json.RawMessage(`"Hello!"`),
    }},
})
if err != nil {
    return err
}

for _, block := range resp.Content {
    if block.Type == "text" {
        fmt.Println(block.Text)
    }
}
```

`MessagesMessage.Content` is raw JSON, so it accepts both wire forms: a bare string, or a content
block array (`json.Marshal([]anthropic.ContentBlock{...})`).

### Native streaming

```go
stream, err := client.MessagesStream(ctx, &anthropic.MessagesRequest{
    Model:     model,
    MaxTokens: 64,
    Messages:  []anthropic.MessagesMessage{{Role: "user", Content: json.RawMessage(`"Count to three."`)}},
})
if err != nil {
    return err
}
defer func() { _ = stream.Close() }()

for {
    event, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        return err
    }

    if event.Type == anthropic.StreamEventTypeContentBlockDelta {
        fmt.Print(event.ContentBlockDelta.Delta.Text)
    }
}
```

Every event also carries its verbatim payload in `event.Raw`.

`const.go` names the Messages API discriminator values — SSE event types
(`StreamEventType*`), content block types (`ContentBlockType*`), delta types
(`DeltaType*`), native stop reasons (`StopReason*`), `tool_choice` types
(`ToolChoiceType*`). Every one of those fields stays an open string on the
wire, so a value this SDK does not list still decodes and reaches you through
`Raw` — the constants are for readable comparisons, not a closed enum.

### Errors

```go
var httpErr *anthropic.HTTPError
if errors.As(err, &httpErr) {
    fmt.Println(httpErr.StatusCode, httpErr.Type, httpErr.Message)
}
```

The unified client maps the same failures to `*ais.APIError`; see
[doc/design/errors.md](../../doc/design/errors.md).

## Notes

- `MaxTokens` is required by the Messages API; the canonical path supplies a default of 4096 when
  the request sets none.
- Reasoning depth and structured output travel in `output_config` (`OutputConfig.Effort` /
  `OutputConfig.Format`), mapped from canonical `ReasoningEffort` / `ResponseFormat`. The
  top-level `Effort` field is deprecated and no longer sent.
- Extension values of the wrong type are rejected with `*ais.ExtensionTypeError` before any
  network I/O — they can never silently take effect.
- Runnable examples live in [`integrations/anthropic_tests`](../../integrations/anthropic_tests) and
  run against a real endpoint when `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` (and optionally
  `ANTHROPIC_BASE_URL`) are set.
