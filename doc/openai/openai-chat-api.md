# OpenAI Chat Completions — Wrapper Design & Implementation

How `provider/openai` wraps `POST {baseURL}/chat/completions`, for OpenAI and for every OpenAI-compatible backend.

- **Official reference**: https://platform.openai.com/docs/api-reference/chat
- **Code**: `provider/openai/native.go` (client, SSE), `wire.go` (types), `accumulate.go` (stream assembly), `model.go` (constants)
- **Change log**: [openai-api-changes.md](./openai-api-changes.md)
- **Responses API**: [openai-response-api.md](./openai-response-api.md) — a separate interaction form on the same client

---

## 1. Client

```go
client := openai.NewClient(apiKey,
    openai.WithBaseURL("https://api.openai.com/v1"),
    openai.WithTimeout(90*time.Second),
)
```

| Option | Purpose |
|---|---|
| `WithBaseURL(string)` | API base URL, trailing `/` stripped. Defaults to `https://api.openai.com/v1`; point it at any compatible backend |
| `WithHTTPClient(*http.Client)` | Full transport control. `nil` panics — that is a programming error, not a runtime condition |
| `WithTimeout(time.Duration)` | Bounds a whole call, including reading a streaming body. Copies the client configured so far, so the caller's `*http.Client` is never mutated and an earlier transport survives. Apply it *after* `WithHTTPClient` |

`NewClient` returns no error. Auth is `Authorization: Bearer <key>`; the body is `application/json`.

## 2. Request

`ChatCompletionRequest` models the documented body field for field. `Model` and `Messages` are required; everything else is `omitempty`, and pointer types (`*float64`, `*int`, `*bool`) exist so an explicit zero — `temperature: 0`, `logprobs: false` — is distinguishable from "unset" and reaches the backend.

`Content` is polymorphic in the protocol, so it is a small type rather than a bare string:

```go
openai.NewTextContent("hello")                       // "content": "hello"
openai.NewPartsContent(                              // "content": [ … ]
    openai.ChatCompletionContentPart{Type: openai.ContentPartTypeText, Text: "what is this?"},
    openai.ChatCompletionContentPart{Type: openai.ContentPartTypeImageURL,
        ImageURL: &openai.ImageURL{URL: "https://example.com/cat.png", Detail: "high"}},
)
```

`Text()` returns the string form, or the concatenation of the text parts. Multimodal parts cover `text`, `image_url`, `input_audio` and `file`.

The stream flag is set on a copy of the request, so a call never mutates the caller's value.

### 2.1 `ExtraBody` — backend-private parameters

OpenAI-compatible backends add their own top-level parameters (`enable_thinking`, `chat_template_kwargs`, …). `ExtraBody` carries them:

```go
req.ExtraBody = map[string]json.RawMessage{
    "enable_thinking": json.RawMessage(`true`),
}
```

Entries are merged into the top level of the body verbatim. The channel is **additive only**:

- a key that collides with a modelled field is an error at marshal time, before any network I/O — it can add parameters, never override or duplicate one this package models;
- an empty key, or a value that is not valid JSON, is an error too;
- the modelled key set is derived from the struct tags, so it cannot drift as fields are added.

Decoding a request body fills `ExtraBody` with every key the package does not model, which is what makes a request round-trip losslessly (`TestWireTypesRoundTripLosslessly`).

### 2.2 Tools

```go
req.Tools = []openai.ChatCompletionTool{{
    Type: openai.ToolTypeFunction,
    Function: openai.ChatCompletionFunction{
        Name:        "get_weather",
        Description: "Get the current weather in a city",
        Parameters:  map[string]any{ /* JSON Schema, passed through as-is */ },
        Strict:      new(true),
    },
}}
req.ToolChoice = "auto"                  // or {"type":"function","function":{"name":…}}
req.ParallelToolCalls = new(false)
```

`Parameters` is `any` and is never validated or rewritten — a JSON Schema is the caller's contract with the model, not with this SDK. `ToolChoice` is `any` for the same reason: it is a string in one form and an object in another.

A tool result is a message with `Role: openai.RoleTool` and the originating `ToolCallID`.

## 3. Non-streaming response

`ChatCompletions` decodes into `ChatCompletionResponse`. Two behaviors are worth knowing:

- A **2xx response whose body carries an `error` object** becomes an `*HTTPError` anyway. Compatible backends are inconsistent about the status code they use for a rejected request.
- An empty `Choices` array is **not** an error. The response is returned as-is; check `len(resp.Choices)` if your call site requires one.

`FinishReason` is a `*string`: the protocol distinguishes "not finished yet" (`null`, on a streaming chunk) from a finish reason. The constants in `model.go` name the documented values without closing the set.

## 4. Streaming

`ChatCompletionsStream` returns a `*ChatCompletionStream`. `Recv` returns one `ChatCompletionChunk` per SSE `data:` line; `[DONE]` becomes `io.EOF`. Comment lines (`:`) and non-`data:` lines are skipped, and the scanner's line cap is 1 MB.

An `error` object inside a chunk closes the stream and surfaces as an `*HTTPError` with no status code — it did not come from the HTTP layer.

### 4.1 Accumulation

The stream folds every chunk into the completion it reconstructs while the caller reads:

```go
for {
    chunk, err := stream.Recv()
    if errors.Is(err, io.EOF) { break }
    if err != nil { return err }
    fmt.Print(chunk.Choices[0].Delta.Content.Text())
}

response := stream.Response()   // assembled message, the same shape as the unary one
usage := stream.Usage()         // nil unless stream_options.include_usage was set
```

Merge rules:

| Field | Rule |
|---|---|
| `content`, `reasoning_content`, `refusal` | concatenated in arrival order |
| `tool_calls[].function.arguments` | concatenated — the model streams the JSON in fragments that are only valid once complete |
| `tool_calls[].id` / `.type` / `.function.name` | last non-empty value wins |
| `logprobs` | appended |
| `finish_reason`, `usage`, `service_tier`, `system_fingerprint` | last value wins; the usage-bearing chunk is terminal and reports totals for the whole completion |

Choices grow by index, so a backend that emits them out of order or skips one does not drop the earlier ones. Before `io.EOF`, `Response()` is a live snapshot in which a tool call's `Arguments` may still be an incomplete JSON fragment.

`Close` is idempotent and safe to call concurrently with `Recv`.

## 5. Usage and prompt caching

OpenAI caching is **automatic**: prefixes over roughly 1024 tokens are cached with no request-side marker, and the accounting comes back on the response.

| Field | Meaning |
|---|---|
| `Usage.PromptTokens` / `CompletionTokens` / `TotalTokens` | The totals |
| `Usage.PromptTokensDetails.CachedTokens` | Prompt tokens served from cache — a **subset** of `PromptTokens`, not an addition to it |
| `Usage.PromptTokensDetails.AudioTokens` | Audio input tokens |
| `Usage.CompletionTokensDetails.ReasoningTokens` | Internal reasoning tokens — a subset of `CompletionTokens` |
| `Usage.CompletionTokensDetails.{Accepted,Rejected}PredictionTokens` | Predicted-outputs accounting |

`ServiceTier` is reported at the response root, not inside `usage`. OpenAI has no cache-*write* accounting; that is an Anthropic concept.

`PromptCacheKey` on the request routes requests to the same cache partition. If `CachedTokens` stays 0 across requests that should share a prefix, something is invalidating it — a per-request timestamp or ID early in the prompt, a non-deterministic map serialization, or a changed tool list (tools serialize before messages, so any tool change invalidates everything after).

Usage is only reported on a stream when the request asks for it:

```go
req.StreamOptions = &openai.StreamOptions{IncludeUsage: new(true)}
```

This is deliberately not set for you: it changes the event sequence the backend sends.

## 6. Errors

```go
type HTTPError struct {
    Status              int             // read it through StatusCode()
    Code, Type, Message string
    Body                json.RawMessage // the bounded raw body, always retained
    Err                 error
}

func (e *HTTPError) StatusCode() int
```

Parsing: read the body under a 1 MB cap, try `{"error":{code,message,param,type}}`, and if that fails or carries no message, keep the raw body as `Message`. Diagnostic information is never discarded.

`StatusCode()` is a method rather than a field so a consumer can match any provider's transport error structurally, without importing this package:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests { /* back off */ }
```

## 7. Protocol capability notes

Facts about the protocol that this wrapper passes through rather than resolves:

- **`max_tokens` is deprecated by OpenAI** and rejected outright by reasoning models (the o-series, GPT-5.x, …), which require `max_completion_tokens`. Both fields exist here; pick per model.
- **Programmatic tool calling is not accepted on Chat Completions.** OpenAI's programmatic tool calling is a Responses-API capability; this endpoint rejects it. `tools` / `tool_choice` pass through unchanged, so whether a backend accepts the combination is the server's decision.
- **On GPT-5.4 and later, `reasoning_effort: "none"` disables tool calling.** Pick `low` or higher while tools are active, or use the Responses API where the combination is fully supported. Both fields stay plain strings here; the caller avoids the conflicting combination.
- **`top_k` is not an OpenAI parameter.** It is modelled because several OpenAI-compatible backends accept it. Against OpenAI itself it is an unknown field.
- **`thinking` is not an OpenAI parameter either.** It is the shared reasoning control of several compatible backends (Qwen, GLM, DeepSeek-style). OpenAI's own control is `reasoning_effort`.
- **`functions` / `function_call` are the deprecated pre-tools API**, retained for backends that still serve them.
- **Unmodelled response fields are not preserved on this path.** Chat Completions is a closed object shape. The Responses API, which is item-based and still evolving, keeps the raw payload of an unmodelled item instead ([openai-response-api.md](./openai-response-api.md)).
- **Model names are strings.** `model.go` names the common ones across OpenAI and the compatible backends this SDK is used with; a model released tomorrow works without an SDK update.
