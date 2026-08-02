# Anthropic Messages API — Wrapper Design & Implementation

How `provider/anthropic` wraps `POST {baseURL}/v1/messages`.

- **Official reference**: https://platform.claude.com/docs/en/api/messages
- **Code**: `provider/anthropic/native.go` (client, SSE), `wire.go` (types), `accumulate.go` (stream assembly, usage merge), `model.go` / `const.go` (constants)
- **Change log**: [anthropic-api-changes.md](./anthropic-api-changes.md)

---

## 1. Client

```go
client := anthropic.NewClient(apiKey,
    anthropic.WithBaseURL("https://api.anthropic.com"),   // optional
    anthropic.WithVersion("2023-06-01"),                  // optional
    anthropic.WithBeta("context-1m-2025-08-07"),          // optional
    anthropic.WithUserProfileID("user_abc123"),           // optional
    anthropic.WithTimeout(90*time.Second),                // optional
)
```

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |
| `x-api-key` | the API key — note this protocol does **not** use `Authorization: Bearer` |
| `anthropic-version` | `WithVersion`, else `2023-06-01` |
| `anthropic-beta` | `WithBeta` values comma-joined, empty strings dropped; the header is omitted entirely when empty |
| `anthropic-user-profile-id` | `WithUserProfileID`; omitted when empty |

`anthropic-beta` is generic infrastructure for opting into beta capabilities (compaction, context editing, structured outputs, fast mode, advisor, …). This SDK emits the header; it models no specific beta capability's fields.

`WithTimeout` bounds a whole call, including reading a streaming body. It copies the client configured so far, so the caller's own `*http.Client` is never mutated and a transport installed by an earlier `WithHTTPClient` survives; apply it after `WithHTTPClient`.

## 2. Request

```go
resp, err := client.Messages(ctx, &anthropic.MessagesRequest{
    Model:     anthropic.ModelClaudeSonnet5,
    MaxTokens: 1024,
    Messages: []anthropic.MessagesMessage{
        {Role: anthropic.RoleUser, Content: json.RawMessage(`"Hi"`)},
    },
})
```

Two shapes in this protocol differ from what a Chat Completions user expects:

- **`MaxTokens` is required.** The API rejects a request without it, and the wire type does not default it.
- **`Content` is `json.RawMessage`.** The API accepts both a bare string and a content-block array in that position, so the field carries whichever the caller sends: a quoted string, or a marshalled `[]ContentBlock`. Nothing is reshaped on the way out.

**System prompts are not a role.** They are the top-level `System` field, itself either a string or a block array — which is what makes a cache breakpoint on the system prompt expressible.

The stream flag is set on a copy of the request, so a call never mutates the caller's value.

### 2.1 Content blocks

`ContentBlock` covers the documented request-side kinds; `const.go` names the discriminators without closing the set:

| `Type` | Fields used |
|---|---|
| `text` | `Text` |
| `thinking` | `Thinking` |
| `image` / `document` | `Source` — `base64` (with `MediaType` + `Data`), `url`, `text` or nested `content` |
| `tool_use` | `ID`, `Name`, `Input` (raw JSON) |
| `tool_result` | `ToolUseID`, `ResultContent` |

A **tool result is a `user` turn** in this protocol, not a role of its own:

```go
{Role: anthropic.RoleUser, Content: blocks(
    anthropic.ContentBlock{Type: anthropic.ContentBlockTypeToolResult,
        ToolUseID: "toolu_1", ResultContent: `{"temp_c":18}`},
)}
```

Several parallel tool results belong in **one** user message as several blocks — the API rejects consecutive `user` turns.

### 2.2 Tools

```go
req.Tools = []anthropic.MessagesTool{{
    Name:        "get_weather",
    Description: "Get the current weather in a city",
    InputSchema: map[string]any{ /* JSON Schema, passed through as-is */ },
    Strict:      new(true),
}}
req.ToolChoice = &anthropic.ToolChoice{Type: anthropic.ToolChoiceTypeAuto}
```

`Type` on a tool selects its *kind*: empty means the default custom tool, and a versioned built-in (`web_search_20260209`, `code_execution_20260521`, …) passes through unvalidated, so a newly released tool version works without an SDK update.

| Field | Meaning |
|---|---|
| `Strict` | Guarantee the tool input validates exactly against the declared schema |
| `CacheControl` | Cache every tool definition up to and including this one (§4) |
| `DeferLoading` | Keep this tool's schema out of the initial context for on-demand discovery by tool search. At least one tool must stay loaded |
| `AllowedCallers` | Restrict who may invoke the tool, e.g. `["code_execution_20260120"]` for programmatic tool calling |
| `EagerInputStreaming` | Stream this tool's input as partial JSON instead of buffering it |
| `InputExamples` | Sample inputs demonstrating a complex schema |

`ToolChoice.DisableParallelToolUse` limits the model to one tool call per turn; `Type` is `auto`, `any` (some tool required), `tool` (the one named in `Name`) or `none`.

### 2.3 Reasoning

Two independent controls:

- `Thinking` — `{Type: "enabled"|"disabled"|"adaptive", BudgetTokens, Display}`. `Display: "omitted"` suppresses thinking blocks in the response.
- `OutputConfig.Effort` — `low`/`medium`/`high`/`xhigh`/`max`. This supersedes the former top-level `effort` parameter, which is no longer sent.

`OutputConfig.Format` carries structured outputs; the caller's JSON Schema is passed through unvalidated.

## 3. Response

```go
type MessagesResponse struct {
    ID, Type, Role, Model string
    Content               []ResponseContentBlock
    StopReason            string
    StopSequence          *string
    StopDetails           *StopDetails       // structured stop classification, e.g. a refusal category
    Usage                 MessagesUsage
    Container             *ResponseContainer // server-side execution container, when one was used
}
```

The response is a **content-block array**, not a single string: walk it by block type. `ResponseContentBlock` embeds the known fields and keeps the verbatim JSON of the whole block in `Raw`, so a block kind this SDK does not model — a server-tool result, a future type, a text block carrying citations — reaches the caller intact rather than being dropped.

`Content` on a response block is `json.RawMessage` rather than a string: server-tool results carry an array there, code-execution results an object.

`StopReason` values are named in `const.go` (`end_turn`, `stop_sequence`, `max_tokens`, `tool_use`, `model_context_window_exceeded`, `refusal`, `pause_turn`) and the field stays an open string.

## 4. Prompt caching

Anthropic caching is **explicit**: the request says where the cacheable prefix ends. Two mechanisms, independent and combinable:

**Per-block breakpoint** — `cache_control` on a content block or a tool. Anthropic caches everything up to *and including* the marked block, so the marker belongs on the last block of the prefix you want cached:

```go
System: blocks(anthropic.ContentBlock{
    Type: anthropic.ContentBlockTypeText, Text: longSystemPrompt,
    CacheControl: &anthropic.CacheControl{Type: anthropic.CacheControlTypeEphemeral},
})
```

**Request-root automatic caching** — one `cache_control` at the request root. The server places the breakpoint on the last cacheable block and advances it as the conversation grows, so the caller maintains nothing:

```go
req.CacheControl = &anthropic.CacheControl{
    Type: anthropic.CacheControlTypeEphemeral,
    TTL:  anthropic.CacheControlTTL1h,   // empty = the default 5-minute cache
}
```

Accounting comes back on `MessagesUsage`:

| Field | Meaning |
|---|---|
| `CacheReadInputTokens` | Tokens served from cache |
| `CacheCreationInputTokens` | Tokens written to cache, across TTLs |
| `CacheCreation.Ephemeral5mInputTokens` / `Ephemeral1hInputTokens` | The per-TTL split; the two sum to `CacheCreationInputTokens` |

Unlike OpenAI, these counts are reported **alongside** `InputTokens` rather than inside it: `TotalInputTokens()` returns the billable sum.

If cache reads stay 0 across requests that should share a prefix, something is invalidating it — a per-request timestamp or ID early in the prompt, a non-deterministic map serialization, or a changed tool list (tools serialize before messages, so any tool change invalidates everything after).

## 5. Streaming

`MessagesStream` returns a `*MessageStream`. `Recv` returns one `StreamEvent` per SSE event, in arrival order, with the decoded payload on the matching field and the verbatim JSON always on `Raw`:

| `Type` | Payload field |
|---|---|
| `message_start` | `MessageStart` — the message envelope and the baseline usage |
| `content_block_start` | `ContentBlockStart` |
| `content_block_delta` | `ContentBlockDelta` — `text_delta`, `thinking_delta`, `signature_delta`, `input_json_delta` |
| `message_delta` | `MessageDelta` — the stop reason and the terminal usage |
| `error` | `Error` |
| `ping`, `content_block_stop`, `message_stop`, anything else | none — the payload is on `Raw` |

An event type this SDK does not model is delivered rather than skipped, so a new event kind is visible to the caller the day the API ships it.

### 5.1 Accumulation

The stream folds every event into the message it reconstructs while the caller reads:

```go
for {
    event, err := stream.Recv()
    if errors.Is(err, io.EOF) { break }
    if err != nil { return err }
    if event.ContentBlockDelta != nil { fmt.Print(event.ContentBlockDelta.Delta.Text) }
}

message := stream.Message()   // assembled message, the same shape as the unary one
usage := stream.Usage()       // merged token accounting
```

| Delta | Rule |
|---|---|
| `text_delta` | concatenated onto the block's `Text` |
| `thinking_delta` | concatenated onto the block's `Thinking` |
| `input_json_delta` | concatenated into the block's `Input` — tool input arrives as partial JSON that is only valid once complete |
| `message_delta` | supplies `StopReason`, `StopSequence` and `StopDetails` |

Blocks grow by index, so an out-of-order or skipped index does not drop the earlier ones. `ResponseContentBlock.Raw` holds the block as it first arrived and is not rewritten by later deltas. Before `io.EOF`, `Message()` is a live snapshot in which a tool block's `Input` may still be incomplete.

`Close` is idempotent and safe to call concurrently with `Recv`.

### 5.2 Usage arrives in two parts

Anthropic reports usage twice: a baseline on `message_start` (input, cache reads/writes, geography, service tier) and the final counts on the terminal `message_delta` (output tokens). `Usage()` merges them, and the merge is **field-wise**: a later event overwrites only what it actually carries, so a terminal event reporting just `output_tokens` does not blank out everything established at the start.

```go
usage := stream.Usage()
// InputTokens, CacheReadInputTokens, InferenceGeo, ServiceTier — from message_start
// OutputTokens                                                 — from message_delta
```

`Usage()` returns nil before the first `message_start`, and `Message().Usage` is the same value.

## 6. Errors

```go
type HTTPError struct {
    Status        int             // read it through StatusCode()
    Type, Message string
    Body          json.RawMessage // the bounded raw body, always retained
    Err           error
}

func (e *HTTPError) StatusCode() int
```

Parsing: read the body under a 1 MB cap, try `{"type":"error","error":{type,message}}`, and if that fails or carries no message, keep the raw body as `Message`.

`StatusCode()` is a method rather than a field so a consumer can match any provider's transport error structurally, without importing this package:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests { /* back off */ }
```

An `error` event inside a stream is delivered as `StreamEvent.Error`, not as a Go error — it is part of the event sequence and the caller decides what to do with it.

## 7. Protocol capability notes

Facts about the protocol that this wrapper passes through rather than resolves:

- **No `system` role.** A system prompt is the top-level `System` field. A conversation that tries to send one as a message will be rejected.
- **No standalone tool role.** Tool results are `user` turns carrying `tool_result` blocks, and consecutive `user` turns are rejected — batch parallel results into one message.
- **`top_k` is native here**, unlike on Chat Completions where it depends on the backend.
- **Thinking blocks come back as content**, alongside text, rather than in a separate field.
- **Unmodelled blocks and events are preserved verbatim** (`ResponseContentBlock.Raw`, `ContentBlockDelta.Raw`, `StreamEvent.Raw`), because this protocol adds block and event kinds faster than a wrapper can model them.
- **Model names are strings.** `model.go` names the current Claude models; a model released tomorrow works without an SDK update.
