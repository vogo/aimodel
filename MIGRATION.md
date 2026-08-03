# Migration: canonical API → provider-native clients

`aimodel` stops shipping a shared request/response model for OpenAI-compatible and Anthropic
protocols. From **v0.7.0** the public surface is two complete, mutually independent native
clients — [`provider/openai`](./provider/openai/README.md) and
[`provider/anthropic`](./provider/anthropic/README.md) — plus a small set of tools that carry no
protocol semantics.

| Version | What happens |
|---|---|
| **v0.6.1** | No behavior change. Every symbol removed in v0.7.0 carries a Go `Deprecated:` comment pointing here, so `staticcheck` / editors flag the call sites ahead of time. |
| **v0.7.0** | The canonical layer is deleted: `ais`, the root `Client`/`ChatCompleter`/`Stream`/interception/`Responder`, both providers' translation layers, and the provider registry. |

Nothing is renamed in place — every removed symbol has a native counterpart listed below.

## Why

The canonical layer promised one request shape for both protocols. In practice it cost more than
it delivered:

- Translation is lossy by construction. Anything only one vendor models had to travel through a
  side channel (`ais.Extensions`), so the "portable" request was never actually complete.
- Every vendor field had to pass a "≥ 2 providers share this semantic" test before callers could
  reach it, which delayed access to features that were already shipped and documented upstream.
- Cross-vendor dispatch — the one capability the shared model existed for — was never used:
  compose deployments in this repository always dispatch across backends of the *same* wire
  format.

The native layer already targeted full fidelity to each official API. v0.7.0 makes it the only
layer, so a vendor feature is reachable the day it is wired.

## At a glance

```go
// before (v0.5.x)
client, err := aimodel.NewClient(
    aimodel.WithAPIKey(key),
    aimodel.WithBaseURL("https://api.openai.com/v1"),
    aimodel.WithDefaultModel("gpt-4o"),
)
resp, err := client.ChatCompletion(ctx, &ais.ChatRequest{
    Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("hi")}},
})
text := resp.Choices[0].Message.Content.Text()

// after (v0.7.0), OpenAI-compatible
client := openai.NewClient(key, openai.WithBaseURL("https://api.openai.com/v1"))
resp, err := client.ChatCompletions(ctx, &openai.ChatCompletionRequest{
    Model:    openai.ModelGPT4o,
    Messages: []openai.ChatCompletionMessage{{Role: "user", Content: openai.NewTextContent("hi")}},
})
text := resp.Choices[0].Message.Content.Text()

// after (v0.7.0), Anthropic
client := anthropic.NewClient(key)
resp, err := client.Messages(ctx, &anthropic.MessagesRequest{
    Model:     anthropic.ModelClaudeSonnet5,
    MaxTokens: 1024,
    Messages:  []anthropic.MessagesMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
})
text := resp.Content[0].Text
```

`MessagesMessage.Content` is `json.RawMessage` because the Messages API accepts both a bare string
and a content-block array in that position. Pass either — a marshalled
`[]anthropic.ContentBlock` for the block form, a quoted string for the shorthand.

Two rules cover most of the diff:

1. **`Model` is required.** There is no client-level default model and no `AI_MODEL` fallback.
2. **The client no longer reads the environment.** Pass the API key and base URL explicitly.
   `AI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and the `*_BASE_URL` chain are gone;
   read whichever variable your deployment actually uses and hand it to `NewClient`.

## Construction

| ≤ v0.6.0 | v0.7.0 (OpenAI) | v0.7.0 (Anthropic) |
|---|---|---|
| `aimodel.NewClient(...)` | `openai.NewClient(apiKey, opts...)` | `anthropic.NewClient(apiKey, opts...)` |
| `aimodel.WithAPIKey(k)` | first argument of `NewClient` | first argument of `NewClient` |
| `aimodel.WithBaseURL(u)` | `openai.WithBaseURL(u)` | `anthropic.WithBaseURL(u)` |
| `aimodel.WithHTTPClient(hc)` | `openai.WithHTTPClient(hc)` | `anthropic.WithHTTPClient(hc)` |
| `aimodel.WithTimeout(d)` | `openai.WithTimeout(d)` | `anthropic.WithTimeout(d)` |
| `aimodel.WithDefaultModel(m)` | *(removed)* set `ChatCompletionRequest.Model` | *(removed)* set `MessagesRequest.Model` |
| `aimodel.WithProvider(name)` | *(removed)* pick the package | *(removed)* pick the package |
| `aimodel.WithProviderOptions(anthropic.Options{...})` | — | `anthropic.WithVersion` / `WithBeta` / `WithUserProfileID` |
| `ais.Register` / `ais.Lookup` / `ais.Config` / `ais.Factory` | *(removed)* | *(removed)* |

`NewClient` no longer returns an error — misconfiguration that used to fail at construction
(unknown provider name, missing base URL, wrong options type) is no longer representable. Native
clients default to the vendor's public endpoint; call `WithBaseURL` for anything else.

`WithTimeout(d)` is exactly `WithHTTPClient(&http.Client{Timeout: d})`; use `WithHTTPClient` when
you need to control the transport as well. Applying both leaves the last one wins.

## Chat, non-streaming

| ≤ v0.6.0 | v0.7.0 (OpenAI) | v0.7.0 (Anthropic) |
|---|---|---|
| `client.ChatCompletion(ctx, req)` | `client.ChatCompletions(ctx, req)` | `client.Messages(ctx, req)` |
| `ais.ChatRequest` | `openai.ChatCompletionRequest` | `anthropic.MessagesRequest` |
| `ais.ChatResponse` | `openai.ChatCompletionResponse` | `anthropic.MessagesResponse` |
| `ais.Message` | `openai.ChatCompletionMessage` | `anthropic.MessagesMessage` |
| `ais.Role*` constants | plain strings (`"user"`, `"assistant"`, …) | plain strings (`"user"`, `"assistant"`) |
| `ais.NewTextContent` / `NewPartsContent` | `openai.NewTextContent` / `NewPartsContent` | `json.RawMessage` — a quoted string or marshalled `[]ContentBlock` |
| `ais.Tool` / `ais.FunctionDefinition` | `openai.ChatCompletionTool` / `ChatCompletionFunction` | `anthropic.MessagesTool` |
| `ais.ToolCall` | `openai.ChatCompletionToolCall` | `tool_use` content block |
| `ais.FinishReason*` | `finish_reason` string on the choice | `StopReason*` on the response |
| `ais.Thinking` | `openai.Thinking` | `anthropic.MessagesThinking` + `OutputConfig.Effort` |
| `ais.ReasoningEffort*` | `ChatCompletionRequest.ReasoningEffort` (string) | `OutputConfig.Effort` |
| `ais.ErrEmptyResponse` | check `len(resp.Choices)` yourself | check `len(resp.Content)` yourself |

The canonical `Choices[0].Message.Content` shape survives on the OpenAI side (same field names,
same `Text()` / `Parts()` accessors). Anthropic responses are content-block arrays — the shape the
API actually returns — so `resp.Content` is walked by block type instead of being flattened into a
single string. `ResponseContentBlock.Raw` keeps the verbatim JSON of every block, so blocks this
SDK does not model reach you intact instead of landing in a side channel.

`MaxTokens` is required by the Anthropic API and is now enforced by the wire type rather than
defaulted by a translator. Set it explicitly.

## Chat, streaming

| ≤ v0.6.0 | v0.7.0 (OpenAI) | v0.7.0 (Anthropic) |
|---|---|---|
| `client.ChatCompletionStream(ctx, req)` | `client.ChatCompletionsStream(ctx, req)` | `client.MessagesStream(ctx, req)` |
| `*aimodel.Stream` | `*openai.ChatCompletionStream` | `*anthropic.MessageStream` |
| `stream.Recv() (*ais.StreamChunk, error)` | `stream.Recv() (*ChatCompletionChunk, error)` | `stream.Recv() (*StreamEvent, error)` |
| `stream.Close()` | same (idempotent) | same (idempotent) |
| `stream.Usage() *ais.Usage` | `stream.Usage() *ChatCompletionUsage` | `stream.Usage() *MessagesUsage` |
| `msg.AppendDelta(delta)` | `stream.Response() *ChatCompletionResponse` | `stream.Message() *MessagesResponse` |
| `aimodel.WrapStream(s, onClose)` | *(removed)* call `Usage()` after `io.EOF` | *(removed)* call `Usage()` after `io.EOF` |
| `aimodel.InterceptStream(s, onChunk, onDone)` | *(removed)* observe in your own `Recv` loop | *(removed)* observe in your own `Recv` loop |

Both native streams accumulate while you read: every `Recv` folds its event into an in-progress
result, so after the stream ends `Response()` / `Message()` returns the assembled message —
text, thinking/reasoning content and tool-call fragments already merged — and `Usage()` returns
the token accounting. Neither accessor requires you to keep the deltas yourself, which is what
`AppendDelta` used to be for.

`InterceptStream` and `WrapStream` existed because the canonical `Stream` hid the decoder. Native
streams do not: a `for { ev, err := stream.Recv() }` loop is the interception point, and the
"fires exactly once, on the first error or on Close" guarantee becomes an ordinary `defer` in that
loop.

`ais.ErrStreamClosed` is gone. Both native streams return `io.EOF` at end of stream and are safe
to `Close` repeatedly.

## Usage accounting

| ≤ v0.6.0 | v0.7.0 (OpenAI) | v0.7.0 (Anthropic) |
|---|---|---|
| `ais.Usage.PromptTokens` | `ChatCompletionUsage.PromptTokens` | `MessagesUsage.InputTokens` |
| `ais.Usage.CompletionTokens` | `ChatCompletionUsage.CompletionTokens` | `MessagesUsage.OutputTokens` |
| `ais.Usage.TotalTokens` | `ChatCompletionUsage.TotalTokens` | sum the input/output fields |
| `ais.Usage.CacheReadTokens` | `PromptTokensDetails.CachedTokens` | `MessagesUsage.CacheReadInputTokens` |
| `ais.Usage.ReasoningTokens` | `CompletionTokensDetails.ReasoningTokens` | `OutputTokensDetails.ThinkingTokens` |
| `ais.Usage.ServiceTier` | `ChatCompletionResponse.ServiceTier` | `MessagesUsage.ServiceTier` |
| `anthropic.UsageExtensionOf(u).CacheWriteTokens` | — | `MessagesUsage.CacheCreationInputTokens` |
| `…CacheWrite5mTokens` / `…CacheWrite1hTokens` | — | `MessagesUsage.CacheCreation.Ephemeral5mInputTokens` / `…1h…` |
| `…ServerToolUse` | — | `MessagesUsage.ServerToolUse` |
| `…InferenceGeo` | — | `MessagesUsage.InferenceGeo` |
| `ais.Usage.Add(other)` | *(removed)* sum the fields you track | *(removed)* sum the fields you track |

Anthropic reports usage twice in a stream: a baseline on `message_start` and the final counts on
the terminal `message_delta`. `MessageStream.Usage()` merges them — a later event overwrites only
the fields it actually carries, so the terminal event reporting just `output_tokens` does not
blank out the input, cache, geo, tier or server-tool numbers established at the start. This is the
behavior the canonical stream used to provide through a private helper; it is now a documented
method on the native stream.

## Errors

There is no shared error type. Each package returns its own `*HTTPError`, and both implement the
same tiny interface, so status-code handling stays protocol-agnostic without importing either
package:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests {
    // back off
}
```

| ≤ v0.6.0 | v0.7.0 |
|---|---|
| `*ais.APIError` | `*openai.HTTPError` / `*anthropic.HTTPError` |
| `apiErr.StatusCode` (field) | `httpErr.StatusCode()` (method) |
| `apiErr.Message` / `.Type` / `.Code` | same fields on each `HTTPError` (`Code` is OpenAI-only) |
| — | `httpErr.Body` — the bounded raw error body, new |
| `ais.ErrNoAPIKey` / `ais.ErrNoBaseURL` | *(removed)* `NewClient` no longer validates |
| `ais.ErrCapabilityNotSupported` / `*ais.CapabilityError` | *(removed)* a missing capability is now a compile error |
| `ais.ErrStreamClosed` | *(removed)* `io.EOF`, `Close` is idempotent |
| `ais.ErrNoActiveModels` | `composes.ErrNoActiveModels` |
| `*ais.ModelError` / `*ais.MultiError` | `*composes.ModelError` / `*composes.MultiError` |

**Breaking within the native layer**: `HTTPError.StatusCode` was an exported *field* in v0.5.x's
native clients. Go does not allow a field and a method to share a name, so in v0.7.0 the field is
renamed to `Status` and `StatusCode() int` becomes the accessor. Read it through the method.

## Models

`ais/model.go` held model-name constants for every backend anyone had used. Model names are not a
vendor-neutral contract, so the constants move into the package whose protocol serves them:

| ≤ v0.6.0 | v0.7.0 |
|---|---|
| `ais.ModelAnthropicClaude*` | `anthropic.ModelClaude*` |
| `ais.ModelOpenaiGPT*` / `ais.ModelOpenaiO*` | `openai.ModelGPT*` / `openai.ModelO*` |
| `ais.ModelDeepseek*`, `ais.ModelGemini*`, `ais.ModelMinimax*`, `ais.ModelKimi*`, `ais.ModelGLM*`, `ais.ModelDoubao*`, `ais.ModelQwen*` | `openai.Model*` (reached over the OpenAI-compatible protocol) |

No shared constants package replaces them. A plain string literal remains valid everywhere.

## Vendor extensions

`ais.Extensions` and the Anthropic `Extend*` / `*Of` helpers are removed. Everything they carried
is an ordinary field of the native request or response:

| ≤ v0.6.0 | v0.7.0 (Anthropic) |
|---|---|
| `anthropic.ExtendRequest(r, &RequestExtension{AutoCache: true})` | `MessagesRequest.CacheControl` at the request root |
| `RequestExtension.Container` | `MessagesRequest.Container` |
| `RequestExtension.InferenceGeo` | `MessagesRequest.InferenceGeo` |
| `anthropic.ExtendMessage(m, &MessageExtension{CacheBreakpoint: true})` | `cache_control` on the content block |
| `MessageExtension.ExtraBlocks` | the response's own content blocks — nothing is unmodelled |
| `anthropic.ExtendTool(t, &ToolExtension{...})` | the corresponding `MessagesTool` fields |
| `anthropic.ChoiceExtensionOf(c).StopDetails` | `MessagesResponse.StopDetails` |
| `anthropic.ResponseExtensionOf(r).Container` | `MessagesResponse.Container` |
| `anthropic.FinishReason*` constants | `anthropic.StopReason*` constants |
| `*ais.ExtensionTypeError` | *(removed)* mistyped extensions are no longer representable |

For OpenAI-compatible backends that accept private top-level parameters (`enable_thinking`,
`chat_template_kwargs`, …) there is a controlled escape hatch:

```go
req := &openai.ChatCompletionRequest{Model: "qwen3.7-plus", Messages: msgs}
req.ExtraBody = map[string]json.RawMessage{
    "enable_thinking": json.RawMessage(`true`),
}
```

`ExtraBody` keys are merged into the top level of the request body. A key that collides with a
modelled field is rejected at marshal time — it can add parameters, never override or duplicate
one this package already models.

## Responses API (OpenAI)

The root `Responder` capability only forwarded to the native client, so it is removed rather than
replaced:

| ≤ v0.6.0 | v0.7.0 |
|---|---|
| `aimodel.Responder` | *(removed)* use `*openai.Client` directly |
| `client.Responses(ctx, req)` | `openai.NewClient(key).Responses(ctx, req)` |
| `client.ResponsesStream(ctx, req)` | `openai.NewClient(key).ResponsesStream(ctx, req)` |
| `aimodel.CapabilityResponses` | *(removed)* |

The request, response, item and event types are unchanged — they were already
`provider/openai` native types.

## Compose

`composes` keeps failover, random and weighted dispatch, health tracking and recovery probes. It
now dispatches within **one** wire format — OpenAI-compatible — instead of across protocols, which
is how it has always been used in practice:

```go
// before
entries := []composes.ModelEntry{{Name: "gpt-4o", Client: openaiClient}}          // aimodel.ChatCompleter
resp, err := compose.ChatCompletion(ctx, &ais.ChatRequest{...})

// after
entries := []composes.ModelEntry{{Name: "gpt-4o", Client: openai.NewClient(key)}} // composes.ChatCompleter
resp, err := compose.ChatCompletions(ctx, &openai.ChatCompletionRequest{...})
```

| ≤ v0.6.0 | v0.7.0 |
|---|---|
| `ModelEntry.Client aimodel.ChatCompleter` | `ModelEntry.Client composes.ChatCompleter` (satisfied by `*openai.Client`) |
| `compose.ChatCompletion(ctx, *ais.ChatRequest)` | `compose.ChatCompletions(ctx, *openai.ChatCompletionRequest)` |
| `compose.ChatCompletionStream(...)` | `compose.ChatCompletionsStream(...)` |
| `*ais.MultiError` | `*composes.MultiError` |
| `ais.ErrNoActiveModels` | `composes.ErrNoActiveModels` |

`ModelEntry.Name` still overrides the request's `Model` per backend, an empty `Name` still leaves
the request's own model in place, and context cancellation still does not mark a backend
unhealthy.

To dispatch across Anthropic backends, compose over `*anthropic.Client` in your own code — the
loop is small, and keeping it out of this package is what stops a shared message model from
growing back.

## Removed symbols

Everything below is deleted in v0.7.0. Each carries a `Deprecated:` comment in v0.6.1.

**Package `ais` (entire package)** — `ChatRequest`, `ChatResponse`, `Message`, `Choice`, `Content`,
`ContentPart`, `ImageURL`, `Tool`, `FunctionDefinition`, `ToolCall`, `FunctionCall`, `Thinking`,
`StreamChunk`, `StreamChunkChoice`, `Usage`, `Error`, `Role*`, `FinishReason*`, `ReasoningEffort*`,
`NewTextContent`, `NewPartsContent`, `Extensions`, `ExtensionMerger`, `ExtensionTypeError`,
`ChatProvider`, `StreamDecoder`, `Config`, `Factory`, `Register`, `Lookup`, `MaxStreamLineSize`,
`APIError`, `CapabilityError`, `ModelError`, `MultiError`, `Err*` sentinels, and all `Model*`
constants.

**Root package `aimodel`** — `Client`, `NewClient`, `Option`, `WithAPIKey`, `WithBaseURL`,
`WithDefaultModel`, `WithHTTPClient`, `WithProvider`, `WithProviderOptions`, `WithTimeout`,
`ChatCompleter`, `Stream`, `WrapStream`, `InterceptStream`, `Responder`, `CapabilityResponses`,
`GetEnv`.

**Package `provider/openai`** — `Name`, `New` (the `ais.Factory`). The native client, wire types
and Responses surface stay.

**Package `provider/anthropic`** — `Name`, `New`, `Options`, `RequestExtension`,
`MessageExtension`, `ToolExtension`, `ChoiceExtension`, `ResponseExtension`, `UsageExtension`,
`Extend*`, `*Of`, `FinishReason*`. The native client and wire types stay — including
`StopDetails`, `ResponseContainer` and `ServerToolUse`, which the extension layer borrowed but
which decode directly from the wire.

## If you cannot migrate yet

Every release up to and including v0.6.0 still ships the canonical API, and none of them is
retracted — pinning `≤ v0.6.0` keeps it working. Those versions will not receive further protocol
updates, though; new features land in the native layer only.
