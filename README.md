# aimodel

[![Build](https://github.com/vogo/aimodel/actions/workflows/build.yml/badge.svg)](https://github.com/vogo/aimodel/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/vogo/aimodel/branch/main/graph/badge.svg)](https://codecov.io/gh/vogo/aimodel)

A Go SDK for AI model APIs with multi-protocol support (OpenAI, Anthropic).

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

openai, _ := aimodel.NewClient(aimodel.WithAPIKey("sk-openai"))
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
