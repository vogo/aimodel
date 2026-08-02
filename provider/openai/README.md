# provider/openai

The OpenAI-compatible provider: the registered `openai` provider used by the unified
`aimodel.Client`, plus a public **native** client over OpenAI's own wire types.

Importing this package registers the provider under `Name` (`"openai"`); the root
`aimodel` package imports it by default, so `aimodel.NewClient()` selects it when no
provider is named.

Two API surfaces live here:

| Endpoint | Native client | Canonical translation |
|---|---|---|
| `POST {baseURL}/chat/completions` | `Client.ChatCompletions` / `ChatCompletionsStream` | yes — `ais.ChatRequest` ⇄ `ChatCompletionRequest` |
| `POST {baseURL}/responses` | `Client.Responses` / `ResponsesStream` | **no** — native types end to end ([ADR 0006](../../doc/adr/0006-responses-capability-on-provider-native-types.md)) |

Design background: [doc/openai/openai-chat-api.md](../../doc/openai/openai-chat-api.md) and
[doc/openai/openai-response-api.md](../../doc/openai/openai-response-api.md).

## What this package exposes

### Provider registration

| Symbol | Purpose |
|---|---|
| `Name` | Registered provider name, and the default when a client names none |
| `New(ais.Config) (ais.ChatProvider, error)` | Provider constructor called by the registry. Requires a non-empty base URL (OpenAI-compatible endpoints have no universal default) and accepts no vendor options |

### Native client

| Symbol | Purpose |
|---|---|
| `NewClient(apiKey string, options ...ClientOption) *Client` | Native client; base URL defaults to `https://api.openai.com/v1` |
| `WithBaseURL(url)`, `WithHTTPClient(hc)` | Client options |
| `(*Client).ChatCompletions(ctx, *ChatCompletionRequest) (*ChatCompletionResponse, error)` | Non-streaming chat completion |
| `(*Client).ChatCompletionsStream(ctx, *ChatCompletionRequest) (*ChatCompletionStream, error)` | Streaming chat completion; `Recv()` returns `*ChatCompletionChunk`, ends with `io.EOF` |
| `(*Client).Responses(ctx, *ResponsesRequest) (*Response, error)` | Non-streaming Responses call |
| `(*Client).ResponsesStream(ctx, *ResponsesRequest) (*ResponseStream, error)` | Streaming Responses call; `Recv()` returns `*ResponseStreamEvent`, ends with `io.EOF` |
| `HTTPError` | Non-2xx / stream-level error, retaining the bounded raw body |

### Wire types

- **Chat Completions** — `ChatCompletionRequest` / `ChatCompletionResponse` / `ChatCompletionChunk`,
  `ChatCompletionMessage`, `ChatCompletionTool`, `ChatCompletionToolCall`, `ChatCompletionUsage`,
  and the polymorphic `ChatCompletionContent` (build with `NewTextContent` / `NewPartsContent`,
  read with `Text()` / `Parts()`).
- **Responses** — `ResponsesRequest` / `Response` / `ResponseStreamEvent`, the polymorphic
  `ResponseInput` (`NewResponseTextInput` / `NewResponseItemsInput`) and `ResponseMessageContent`
  (`NewResponseTextContent` / `NewResponseContentParts`), input/output items
  (`ResponseInputItem`, `ResponseOutputItem`, `NewResponseInputMessage`), hosted-tool items
  (`ResponseWebSearchCall`, `ResponseFileSearchCall`, `ResponseCodeInterpreterCall`),
  `ResponseTool`, `ResponseReasoningItem`, `ResponseUsage`.
- **Constants** — item/content/tool/annotation discriminators, response statuses, `include`
  values, truncation strategies, and every documented SSE event type (`ResponseEvent*`).
  `ResponseStreamEventTypes()` returns the documented baseline.

Fidelity guarantee: an item, tool or event type this SDK does not model still reaches the caller
with its discriminator and verbatim payload in `Raw`, and is re-encoded as-is on a round trip.

## Usage

### Unified client (canonical types, chat)

```go
import (
    "github.com/vogo/aimodel"
    "github.com/vogo/aimodel/ais"
)

client, _ := aimodel.NewClient(
    aimodel.WithAPIKey(apiKey),
    aimodel.WithBaseURL("https://api.openai.com/v1"),
)

resp, _ := client.ChatCompletion(ctx, &ais.ChatRequest{
    Model: ais.ModelOpenaiGPT41,
    Messages: []ais.Message{
        {Role: ais.RoleUser, Content: ais.NewTextContent("Hello!")},
    },
})

fmt.Println(resp.Choices[0].Message.Content.Text())
```

### Native chat completion

```go
import "github.com/vogo/aimodel/provider/openai"

client := openai.NewClient(apiKey, openai.WithBaseURL(baseURL))

resp, err := client.ChatCompletions(ctx, &openai.ChatCompletionRequest{
    Model: model,
    Messages: []openai.ChatCompletionMessage{
        {Role: "user", Content: openai.NewTextContent("Hello!")},
    },
})
if err != nil {
    return err
}

fmt.Println(resp.Choices[0].Message.Content.Text())
```

### Native streaming

```go
stream, err := client.ChatCompletionsStream(ctx, &openai.ChatCompletionRequest{
    Model:    model,
    Messages: []openai.ChatCompletionMessage{{Role: "user", Content: openai.NewTextContent("Count to three.")}},
})
if err != nil {
    return err
}
defer func() { _ = stream.Close() }()

for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        return err
    }

    fmt.Print(chunk.Choices[0].Delta.Content.Text())
}
```

### Responses (native types, with a hosted tool)

```go
resp, err := client.Responses(ctx, &openai.ResponsesRequest{
    Model:        model,
    Instructions: "Answer in one sentence.",
    Input:        openai.NewResponseTextInput("What is the Responses API?"),
    Tools:        []openai.ResponseTool{{Type: openai.ResponseToolTypeWebSearch}},
    Include:      []string{openai.ResponseIncludeWebSearchResults},
})
if err != nil {
    return err
}

fmt.Println(resp.Status, resp.OutputText) // OutputText concatenates every output_text part
```

A failed or incomplete response is **not** a Go error — inspect `resp.Status`, `resp.Error` and
`resp.IncompleteDetails`. Only transport and non-2xx failures return `*HTTPError`.

### Responses streaming

```go
stream, err := client.ResponsesStream(ctx, &openai.ResponsesRequest{
    Model: model,
    Input: openai.NewResponseTextInput("Count from one to three."),
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

    switch event.Type {
    case openai.ResponseEventOutputTextDelta:
        fmt.Print(event.Delta)
    case openai.ResponseEventCompleted:
        fmt.Println(event.Response.OutputText)
    }
}
```

### Responses through the unified client

`aimodel.Client` implements `aimodel.Responder`, which speaks these native types — the one
documented exception to "canonical in, canonical out". It reuses the client's API key, base URL,
HTTP client and timeout; the client's default model is **not** applied, and chat interception and
compose failover do not run.

```go
var responder aimodel.Responder = client // *aimodel.Client

resp, err := responder.Responses(ctx, &openai.ResponsesRequest{
    Model: model,
    Input: openai.NewResponseTextInput("Hello!"),
})
```

A provider without the capability fails with `*ais.CapabilityError` before any network I/O.

### Errors

```go
var httpErr *openai.HTTPError
if errors.As(err, &httpErr) {
    fmt.Println(httpErr.StatusCode, httpErr.Type, httpErr.Code, httpErr.Message)
}
```

The unified client maps the same failures to `*ais.APIError`; see
[doc/design/errors.md](../../doc/design/errors.md).

## Notes

- Prefer `MaxCompletionTokens` over the deprecated `MaxTokens` — it is the only token cap
  reasoning models accept.
- Vendor-only chat parameters belong on the native request or the `ais.Extensions` channel,
  never on canonical types — see [doc/architecture.md](../../doc/architecture.md) §2.
- Runnable examples live in [`integrations/openai_tests`](../../integrations/openai_tests) and run
  against a real endpoint when `OPENAI_API_KEY` / `OPENAI_MODEL` (and optionally
  `OPENAI_BASE_URL`) are set.
