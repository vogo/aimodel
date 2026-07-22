# aimodel

[![Build](https://github.com/vogo/aimodel/actions/workflows/build.yml/badge.svg)](https://github.com/vogo/aimodel/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/vogo/aimodel/branch/main/graph/badge.svg)](https://codecov.io/gh/vogo/aimodel)

A Go SDK for AI model APIs with multi-protocol support (OpenAI, Anthropic). Zero external dependencies.

This SDK is a **thin API wrapper** — it translates requests, manages connections, and normalizes responses across protocols. It intentionally does **not** include retry, rate limiting, request validation, caching / persistence, or logging / metrics. Control mechanisms belong in the layer above, where you have full context over your application's requirements.

The SDK is layered. The **canonical layer** exposes only semantics with verified mappings in at least two providers, so portable code switches backends without changes. It is built on a per-vendor **native layer** whose job is complete fidelity to each official API. The **compose layer** ([`composes`](./composes/)) dispatches across multiple models above both. Architecture details: [doc/architecture.md](./doc/architecture.md).

## Documentation

This README covers usage. The design lives under [`doc/`](./doc/):

| Topic | Document |
|---|---|
| Architecture, canonical representation, client & dispatch | [doc/architecture.md](./doc/architecture.md) |
| Request/response types, `Usage` | [doc/design/data-model.md](./doc/design/data-model.md) |
| Streaming, delta merging, unmodelled blocks | [doc/design/streaming.md](./doc/design/streaming.md) |
| Tool definitions, `tool_choice`, parallel tool results | [doc/design/tool-use.md](./doc/design/tool-use.md) |
| Prompt caching | [doc/design/prompt-caching.md](./doc/design/prompt-caching.md) |
| Error model | [doc/design/errors.md](./doc/design/errors.md) |
| Multi-model composition | [doc/design/compose.md](./doc/design/compose.md) |
| Anthropic wire mapping | [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md) |
| OpenAI wire mapping | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) |

Sync status against the official APIs: [CHANGES.md](./CHANGES.md).

| Protocol | Official docs | Provider package |
|---|---|---|
| OpenAI (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | `provider/openai/` |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | `provider/anthropic/` |

## Usage

```go
import (
    "github.com/vogo/aimodel"
    "github.com/vogo/aimodel/ais"
)
```

Set env vars `AI_API_KEY` and `AI_BASE_URL` (or `OPENAI_API_KEY` / `OPENAI_BASE_URL`).

### Chat Completion

```go
client, _ := aimodel.NewClient()

resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model: ais.ModelOpenaiGPT41,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})

fmt.Println(resp.Choices[0].Message.Content.Text())
```

Use `MaxCompletionTokens` rather than the deprecated `MaxTokens` — it is the only token cap reasoning models accept:

```go
maxCompletionTokens := 1024
resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model:               ais.ModelOpenaiO3, // reasoning model
    MaxCompletionTokens: &maxCompletionTokens,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
```

### Reasoning effort

```go
resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model:           ais.ModelOpenaiGPT41,
    ReasoningEffort: aimodel.ReasoningEffortHigh, // none/minimal/low/medium/high/xhigh
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
```

The value stays a plain `string`. For Anthropic extended thinking, set `Thinking` (`Type` is `enabled` / `disabled` / `adaptive`; `Display: "omitted"` suppresses thinking content). See [doc/design/data-model.md](./doc/design/data-model.md) §1.3.

Provider-only parameters use the provider's native API or an established extension surface. For example, Anthropic prompt-cache breakpoints and automatic caching use helpers:

```go
import "github.com/vogo/aimodel/provider/anthropic"

anthropic.ExtendRequest(req, &anthropic.RequestExtension{AutoCache: true})
anthropic.ExtendMessage(&req.Messages[0], &anthropic.MessageExtension{CacheBreakpoint: true})
```

See [doc/architecture.md](./doc/architecture.md) §2 for the extension-channel contract and [doc/design/prompt-caching.md](./doc/design/prompt-caching.md) for the caching API.

`resp.Usage` normalizes token counts across protocols (cache read/write, reasoning tokens, server-tool counts, inference geography, service tier); see [doc/design/data-model.md](./doc/design/data-model.md) §4.

### Multimodal input

`Content` is polymorphic — `NewTextContent` for plain text, `NewPartsContent` for a shared multimodal array (`text` / `image_url`).

```go
resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model: ais.ModelOpenaiGPT41,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewPartsContent(
            aimodel.ContentPart{Type: "text", Text: "Describe this image"},
            aimodel.ContentPart{Type: "image_url", ImageURL: &aimodel.ImageURL{URL: imageURL}},
        )},
    },
})
```

### Streaming

```go
stream, _ := client.ChatCompletionStream(context.Background(), &aimodel.ChatRequest{
    Model: ais.ModelOpenaiGPT41,
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

Accumulate a full message with `Message.AppendDelta`, and read the final token counts from `stream.Usage()` after the stream ends. See [doc/design/streaming.md](./doc/design/streaming.md).

### Anthropic Protocol

Select the Anthropic provider by name with `WithProvider(anthropic.Name)`:

```go
import "github.com/vogo/aimodel/provider/anthropic"

client, _ := aimodel.NewClient(
    aimodel.WithAPIKey("sk-ant-xxx"),
    aimodel.WithProvider(anthropic.Name),
)

resp, _ := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
    Model: ais.ModelAnthropicClaudeSonnet5,
    Messages: []aimodel.Message{
        {Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hello!")},
    },
})
```

The same `ChatCompletion` / `ChatCompletionStream` methods work for every provider — the client delegates to the resolved provider while the canonical types remain provider-neutral.

Translation behavior worth knowing about when you switch protocols — system-message positioning, `tool_choice` mapping, parallel tool results, `output_config`, and how unrecognized content blocks are preserved — is documented in [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md).

### Client Options

```go
client, _ := aimodel.NewClient(
    aimodel.WithAPIKey("your-key"),
    aimodel.WithBaseURL("https://api.example.com/v1"),
    aimodel.WithProvider(anthropic.Name),
    aimodel.WithTimeout(30 * time.Second),
)
```

**Anthropic header options** travel through the unified `WithProviderOptions` channel as an `anthropic.Options` value (only the Anthropic provider reads it):

```go
import "github.com/vogo/aimodel/provider/anthropic"

client, _ := aimodel.NewClient(
    aimodel.WithAPIKey("sk-ant-xxx"),
    aimodel.WithProvider(anthropic.Name),
    aimodel.WithProviderOptions(anthropic.Options{
        // Opt into beta capabilities via the "anthropic-beta" header.
        // Empty strings are ignored; values are comma-joined on the wire.
        Beta: []string{"context-1m-2025-08-07"},
        // Override the "anthropic-version" header (default "2023-06-01").
        Version: "2023-06-01",
        // Associate requests with an end-user profile via
        // the "anthropic-user-profile-id" header.
        UserProfileID: "user_abc123",
    }),
)
```

Each field ignores an empty value and omits its header entirely when unset. The full option table is in [doc/architecture.md](./doc/architecture.md) §3.1.

### Multi-Model Compose

The `composes` package dispatches requests across multiple backends with failover, random, or weighted strategies:

```go
import "github.com/vogo/aimodel/composes"

openaiClient, _ := aimodel.NewClient(aimodel.WithAPIKey("sk-openai"), aimodel.WithBaseURL("https://api.openai.com/v1"))
anthropicClient, _ := aimodel.NewClient(
    aimodel.WithAPIKey("sk-ant"),
    aimodel.WithProvider(anthropic.Name),
)

cc, _ := composes.NewComposeClient(composes.StrategyFailover, []composes.ModelEntry{
    {Name: "gpt-4o", Client: openaiClient},
    {Name: "claude-sonnet", Client: anthropicClient},
})

resp, _ := cc.ChatCompletion(ctx, req)
```

Health tracking, exponential-backoff recovery probes, and cancellation semantics are documented in [doc/design/compose.md](./doc/design/compose.md).
