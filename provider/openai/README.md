# provider/openai

A complete client for OpenAI's HTTP API over OpenAI's own wire types, serving OpenAI and every
OpenAI-compatible backend. It imports no other package in this module.

Two interaction forms live here, as two method sets on one client:

| Endpoint | Methods |
|---|---|
| `POST {baseURL}/chat/completions` | `ChatCompletions` / `ChatCompletionsStream` |
| `POST {baseURL}/responses` | `Responses` / `ResponsesStream` |

Design background: [doc/openai/openai-chat-api.md](../../doc/openai/openai-chat-api.md) and
[doc/openai/openai-response-api.md](../../doc/openai/openai-response-api.md).

## What this package exposes

### Client

| Symbol | Purpose |
|---|---|
| `NewClient(apiKey string, options ...ClientOption) *Client` | Base URL defaults to `https://api.openai.com/v1`. Returns no error — there is nothing left to validate at construction |
| `WithBaseURL(url)` | Point at any compatible backend; trailing `/` stripped |
| `WithHTTPClient(hc)` | Full transport control; `nil` panics |
| `WithTimeout(d)` | Bounds a whole call. Copies the client configured so far, so the caller's `*http.Client` is untouched and an earlier transport survives. Apply after `WithHTTPClient` |
| `(*Client).ChatCompletions(ctx, *ChatCompletionRequest) (*ChatCompletionResponse, error)` | Non-streaming chat completion |
| `(*Client).ChatCompletionsStream(ctx, *ChatCompletionRequest) (*ChatCompletionStream, error)` | Streaming chat completion; `Recv()` returns `*ChatCompletionChunk`, ends with `io.EOF` |
| `(*Client).Responses(ctx, *ResponsesRequest) (*Response, error)` | Non-streaming Responses call |
| `(*Client).ResponsesStream(ctx, *ResponsesRequest) (*ResponseStream, error)` | Streaming Responses call; `Recv()` returns `*ResponseStreamEvent`, ends with `io.EOF` |
| `HTTPError` | Non-2xx / stream-level error, retaining the bounded raw body. `StatusCode() int` is the accessor |

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
- **Constants** — model names, roles, finish reasons, reasoning-effort values, content-part and
  tool discriminators (`model.go`); item/content/tool/annotation discriminators, response statuses,
  `include` values, truncation strategies and every documented SSE event type
  (`responses_const.go`). `ResponseStreamEventTypes()` returns the documented baseline.

Fidelity guarantee: on the Responses path, an item, tool or event type this SDK does not model
still reaches the caller with its discriminator and verbatim payload in `Raw`, and is re-encoded
as-is on a round trip. On the Chat Completions path, unmodelled *request* parameters travel through
`ExtraBody` and survive a decode → encode round trip. `TestWireTypesRoundTripLosslessly` checks
every exported wire type.

## Usage

### Chat Completions

```go
client := openai.NewClient(apiKey, openai.WithBaseURL("https://api.openai.com/v1"))

response, err := client.ChatCompletions(ctx, &openai.ChatCompletionRequest{
    Model: openai.ModelGPT41,
    Messages: []openai.ChatCompletionMessage{
        {Role: openai.RoleUser, Content: openai.NewTextContent("Hello!")},
    },
})

fmt.Println(response.Choices[0].Message.Content.Text())
```

### Streaming

The stream accumulates while you read, so the assembled message and the token counts are available
without merging deltas yourself:

```go
stream, err := client.ChatCompletionsStream(ctx, request)
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

response := stream.Response()   // text, reasoning content and tool-call arguments, assembled
usage := stream.Usage()         // requires StreamOptions{IncludeUsage: new(true)}
```

### Backend-private parameters

```go
request.ExtraBody = map[string]json.RawMessage{
    "enable_thinking":      json.RawMessage(`true`),
    "chat_template_kwargs": json.RawMessage(`{"thinking":false}`),
}
```

Additive only: a key colliding with a modelled field, an empty key, or invalid JSON fails at
marshal time — before any network I/O.

### Responses

```go
response, err := client.Responses(ctx, &openai.ResponsesRequest{
    Model:        "gpt-5",
    Instructions: "Answer in one sentence.",
    Input:        openai.NewResponseTextInput("What changed in the Responses API?"),
    Tools:        []openai.ResponseTool{{Type: openai.ResponseToolTypeWebSearch}},
})

fmt.Println(response.OutputText)
```

### Errors

```go
var httpErr *openai.HTTPError
if errors.As(err, &httpErr) {
    log.Printf("HTTP %d %s: %s", httpErr.StatusCode(), httpErr.Type, httpErr.Message)
}
```

To stay protocol-agnostic, match the structural interface instead:

```go
type statusCoder interface{ StatusCode() int }
```

## Boundaries

- This package imports nothing else from this module, and nothing in this module sits between it
  and `provider/anthropic`. Duplication between the two is expected — see
  [ADR 0002](../../doc/adr/0002-provider-native-as-the-only-public-interface.md).
- `Model` is required on every request; there is no client-level default and no environment
  fallback.
- Requests are never mutated: the stream flag is set on a copy.
