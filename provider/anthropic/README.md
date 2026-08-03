# provider/anthropic

A complete client for the Anthropic Messages API over Anthropic's own wire types. It imports no
other package in this module.

| Endpoint | Methods |
|---|---|
| `POST {baseURL}/v1/messages` | `Messages` / `MessagesStream` |

The base URL defaults to `https://api.anthropic.com`, the API version header to `2023-06-01`.
Protocol details: [doc/anthropic/anthropic-message-api.md](../../doc/anthropic/anthropic-message-api.md).

## What this package exposes

### Client

| Symbol | Purpose |
|---|---|
| `NewClient(apiKey string, options ...ClientOption) *Client` | Returns no error — there is nothing left to validate at construction |
| `WithBaseURL(url)` | Override the default endpoint; trailing `/` stripped |
| `WithHTTPClient(hc)` | Full transport control; `nil` panics |
| `WithTimeout(d)` | Bounds a whole call. Copies the client configured so far, so the caller's `*http.Client` is untouched and an earlier transport survives. Apply after `WithHTTPClient` |
| `WithVersion(v)` | `anthropic-version` header |
| `WithBeta(values...)` | `anthropic-beta` header; empty values dropped, the rest comma-joined, header omitted when empty |
| `WithUserProfileID(id)` | `anthropic-user-profile-id` header; omitted when empty |
| `(*Client).Messages(ctx, *MessagesRequest) (*MessagesResponse, error)` | Non-streaming Messages call |
| `(*Client).MessagesStream(ctx, *MessagesRequest) (*MessageStream, error)` | Streaming Messages call; `Recv()` returns `*StreamEvent`, ends with `io.EOF` |
| `HTTPError` | Non-2xx error, retaining the bounded raw body. `StatusCode() int` is the accessor |

Auth is the `x-api-key` header — this protocol does not use `Authorization: Bearer`.

### Wire types

`MessagesRequest` / `MessagesResponse`, `MessagesMessage`, `ContentBlock` / `ResponseContentBlock`,
`ContentSource`, `MessagesTool`, `ToolChoice`, `MessagesThinking`, `OutputConfig` / `OutputFormat`,
`CacheControl`, `MessagesUsage` (with `CacheCreation`, `OutputTokensDetails`, `ServerToolUse`),
`StopDetails`, `ResponseContainer`, and the SSE event types
(`StreamEvent`, `MessageStartEvent`, `ContentBlockStartEvent`, `ContentBlockDeltaEvent`,
`MessageDeltaEvent`).

Constants for models, roles, effort levels, thinking types, block/delta/event discriminators, stop
reasons, tool-choice types and cache control live in `model.go` and `const.go`. Every one of those
fields stays an open string.

Fidelity guarantee: every response block and stream delta keeps its verbatim JSON in `Raw`, so
server-tool results, citations and future block types survive without a lossy round trip.
`TestWireTypesRoundTripLosslessly` checks every exported wire type.

## Usage

```go
client := anthropic.NewClient(apiKey)

response, err := client.Messages(ctx, &anthropic.MessagesRequest{
    Model:     anthropic.ModelClaudeSonnet5,
    MaxTokens: 1024,
    Messages: []anthropic.MessagesMessage{
        {Role: anthropic.RoleUser, Content: json.RawMessage(`"Hello!"`)},
    },
})

fmt.Println(response.Content[0].Text)
```

`MaxTokens` is required by the API. `MessagesMessage.Content` is `json.RawMessage` because the
protocol accepts both a bare string and a content-block array there — pass a quoted string, or a
marshalled `[]anthropic.ContentBlock`. A system prompt is the top-level `System` field, not a role.

### Streaming

The stream accumulates while you read:

```go
stream, err := client.MessagesStream(ctx, request)
defer func() { _ = stream.Close() }()

for {
    event, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        return err
    }
    if event.ContentBlockDelta != nil {
        fmt.Print(event.ContentBlockDelta.Delta.Text)
    }
}

message := stream.Message()   // content blocks assembled, tool inputs reassembled from partial JSON
usage := stream.Usage()       // message_start baseline merged with the terminal counts
```

`Usage()` merges field-wise: the terminal `message_delta` carries only `output_tokens`, and must
not blank out the input, cache, geography, tier and server-tool numbers established at
`message_start`.

### Prompt caching

Explicit in this protocol. Mark where the cacheable prefix ends:

```go
System: systemBlocks,   // last block carries CacheControl{Type: CacheControlTypeEphemeral}
```

…or let the server maintain the breakpoint for you:

```go
request.CacheControl = &anthropic.CacheControl{
    Type: anthropic.CacheControlTypeEphemeral,
    TTL:  anthropic.CacheControlTTL1h,   // empty = the default 5-minute cache
}
```

Accounting comes back on `MessagesUsage`: `CacheReadInputTokens`, `CacheCreationInputTokens` and
the per-TTL `CacheCreation` split. These are reported **alongside** `InputTokens`, not inside it —
`TotalInputTokens()` returns the billable sum.

### Errors

```go
var httpErr *anthropic.HTTPError
if errors.As(err, &httpErr) {
    log.Printf("HTTP %d %s: %s", httpErr.StatusCode(), httpErr.Type, httpErr.Message)
}
```

To stay protocol-agnostic, match the structural interface instead:

```go
type statusCoder interface{ StatusCode() int }
```

An `error` event inside a stream arrives as `StreamEvent.Error`, not as a Go error — it is part of
the event sequence.

## Boundaries

- This package imports nothing else from this module, and nothing in this module sits between it
  and `provider/openai`. Duplication between the two is expected — see
  [ADR 0002](../../doc/adr/0002-provider-native-as-the-only-public-interface.md).
- Tool results are `user` turns carrying `tool_result` blocks; batch parallel results into one
  message, because consecutive `user` turns are rejected.
- Requests are never mutated: the stream flag is set on a copy.
