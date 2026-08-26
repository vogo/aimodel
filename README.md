# aimodel

[![Build](https://github.com/vogo/aimodel/actions/workflows/build.yml/badge.svg)](https://github.com/vogo/aimodel/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/vogo/aimodel/branch/main/graph/badge.svg)](https://codecov.io/gh/vogo/aimodel)

Go clients for AI model APIs — one complete, independent client per protocol. Zero external dependencies.

This SDK is a **thin API wrapper**: it builds requests, manages connections, and decodes responses. It intentionally does **not** include rate limiting, request validation, caching / persistence, logging / metrics, or multi-backend routing. Control mechanisms belong in the layer above — for agent applications, see [github.com/vogo/vage/largemodel](https://github.com/vogo/vage) for routed callers with retry and failover.

There is no unified client and no shared schema. You pick a protocol by importing its package, and that package expresses its official API completely rather than the part another vendor happens to share. Architecture: [doc/architecture.md](./doc/architecture.md).

## Documentation

This README covers usage. The design lives under [`doc/`](./doc/):

| Topic | Document |
|---|---|
| Architecture, package boundaries, what may be shared | [doc/architecture.md](./doc/architecture.md) |
| OpenAI Chat Completions | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) |
| OpenAI Responses API | [doc/openai/openai-response-api.md](./doc/openai/openai-response-api.md) |
| Anthropic Messages API | [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md) |

| Protocol | Official docs | Package |
|---|---|---|
| OpenAI Chat Completions (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | [`openai/`](./openai/README.md) |
| OpenAI Responses | https://platform.openai.com/docs/api-reference/responses | [`openai/`](./openai/README.md) |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | [`anthropic/`](./anthropic/README.md) |

## Usage

### Chat Completions (OpenAI and OpenAI-compatible)

```go
import "github.com/vogo/aimodel/openai"

client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

response, err := client.ChatCompletions(ctx, &openai.ChatCompletionRequest{
    Model: openai.ModelGPT41,
    Messages: []openai.ChatCompletionMessage{
        {Role: openai.RoleUser, Content: openai.NewTextContent("Hello!")},
    },
})

fmt.Println(response.Choices[0].Message.Content.Text())
```

Point `WithBaseURL` at any compatible backend:

```go
client := openai.NewClient(apiKey,
    openai.WithBaseURL("https://dashscope.aliyuncs.com/compatible-mode/v1"),
    openai.WithTimeout(90*time.Second),
)
```

Use `MaxCompletionTokens` rather than the deprecated `MaxTokens` — it is the only token cap reasoning models accept:

```go
request := &openai.ChatCompletionRequest{
    Model:               openai.ModelO3,
    MaxCompletionTokens: new(1024),
    ReasoningEffort:     openai.ReasoningEffortHigh,   // none/minimal/low/medium/high/xhigh
    Messages:            messages,
}
```

Backend-private parameters go through `ExtraBody`, which can add top-level fields but never override a modelled one:

```go
request.ExtraBody = map[string]json.RawMessage{
    "enable_thinking": json.RawMessage(`true`),
}
```

### Multimodal input

```go
Content: openai.NewPartsContent(
    openai.ChatCompletionContentPart{Type: openai.ContentPartTypeText, Text: "Describe this image"},
    openai.ChatCompletionContentPart{Type: openai.ContentPartTypeImageURL,
        ImageURL: &openai.ImageURL{URL: imageURL, Detail: "high"}},
)
```

### Streaming

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

response := stream.Response()   // assembled message: text, reasoning, tool calls
usage := stream.Usage()         // set stream_options.include_usage to get this
```

The stream accumulates while you read, so you never merge deltas yourself.

### Tools

```go
request.Tools = []openai.ChatCompletionTool{{
    Type: openai.ToolTypeFunction,
    Function: openai.ChatCompletionFunction{
        Name:        "get_weather",
        Description: "Get the current weather in a city",
        Parameters:  schema,          // your JSON Schema, passed through as-is
    },
}}
request.ToolChoice = "auto"
```

### Anthropic Messages

```go
import "github.com/vogo/aimodel/anthropic"

client := anthropic.NewClient(os.Getenv("ANTHROPIC_API_KEY"))

response, err := client.Messages(ctx, &anthropic.MessagesRequest{
    Model:     anthropic.ModelClaudeSonnet5,
    MaxTokens: 1024,
    Messages: []anthropic.MessagesMessage{
        {Role: anthropic.RoleUser, Content: json.RawMessage(`"Hello!"`)},
    },
})

fmt.Println(response.Content[0].Text)
```

`MaxTokens` is required by this API. `Content` is `json.RawMessage` because the protocol accepts both a bare string and a content-block array there — pass a quoted string, or a marshalled `[]anthropic.ContentBlock`. Header options are the package's own:

```go
client := anthropic.NewClient(apiKey,
    anthropic.WithVersion("2023-06-01"),
    anthropic.WithUserProfileID("user_abc123"),
)
```

Streaming works the same way as OpenAI's, over this protocol's events:

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

message := stream.Message()   // assembled content blocks
usage := stream.Usage()       // message_start baseline merged with the terminal counts
```

Prompt caching is explicit in this protocol — mark where the cacheable prefix ends, or let the server maintain the breakpoint:

```go
request.CacheControl = &anthropic.CacheControl{
    Type: anthropic.CacheControlTypeEphemeral,
    TTL:  anthropic.CacheControlTTL1h,
}
```

### OpenAI Responses API

`POST /v1/responses` is OpenAI's forward-looking interface — hosted tools, response chaining and server-side conversations live there. It is a separate method set on the same client:

```go
response, err := client.Responses(ctx, &openai.ResponsesRequest{
    Model:        openai.ModelGPT55,
    Instructions: "Answer in one sentence.",
    Input:        openai.NewResponseTextInput("What changed in the Responses API?"),
    Tools:        []openai.ResponseTool{{Type: openai.ResponseToolTypeWebSearch}},
})

fmt.Println(response.OutputText) // aggregated from the output_text parts
```

Streaming yields one typed event at a time and ends with `io.EOF`:

```go
stream, _ := client.ResponsesStream(ctx, request)
defer func() { _ = stream.Close() }()

for {
    event, err := stream.Recv()
    if errors.Is(err, io.EOF) {
        break
    }
    if err != nil {
        return err
    }
    if event.Type == openai.ResponseEventOutputTextDelta {
        fmt.Print(event.Delta)
    }
}
```

Full wire reference: [doc/openai/openai-response-api.md](./doc/openai/openai-response-api.md).

### Errors

Each package returns its own `*HTTPError`, and both implement the same tiny interface, so status-code handling needs no provider import:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests {
    // back off
}
```
