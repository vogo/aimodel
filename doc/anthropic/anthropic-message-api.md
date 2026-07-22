# Anthropic Messages API — Wrapper Design & Implementation

- **Official protocol**: Anthropic Messages API (`POST /v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Implementation** (all under `provider/anthropic/`): `anthropic.go` (native wire types and bidirectional translation), `provider.go` (request building, auth headers, response/error parsing, `Options`), `stream.go` (SSE parsing)
- **Change log**: [anthropic-api-changes.md](./anthropic-api-changes.md)

The core premise is in [../api.md](../api.md): the SDK's canonical representation is the **OpenAI shape**, so Anthropic is the only path that translates in both directions. Canonical type semantics live in [../design/data-model.md](../design/data-model.md).

---

## 1. Overall structure

```
ChatRequest ──toAnthropicRequest()──▶ anthropicRequest ──JSON──▶ POST {base}/v1/messages
                                                                        │
ChatResponse ◀─fromAnthropicResponse()── anthropicResponse ◀────────────┘   (non-streaming)

Stream.Recv() ◀─anthropicRecvFunc()── SSE events (message_start / content_block_* / message_delta / …)
```

**Every Anthropic type is package-private** (`anthropicRequest`, `anthropicResponse`, `anthropicContentBlock`, …) and never exposed — callers only ever face the canonical types, so protocol details cannot leak into the public API.

## 2. Endpoint, auth & headers

```go
const (
    anthropicDefaultBaseURL   = "https://api.anthropic.com"
    anthropicAPIVersion       = "2023-06-01"
    anthropicDefaultMaxTokens = 4096
)
```

- **Base URL**: the configured base URL when non-empty, otherwise the default — which is why the Anthropic factory allows construction without a base URL.
- **Headers** (`provider.setHeaders`):

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |
| `x-api-key` | the configured API key (note: **not** `Authorization: Bearer`) |
| `anthropic-version` | `Options.Version`, else `2023-06-01` |
| `anthropic-beta` | `Options.Beta` values comma-joined (empty strings dropped); **header omitted entirely when empty** |
| `anthropic-user-profile-id` | `Options.UserProfileID`; **header omitted entirely when empty** |

The Anthropic-specific configuration is passed as an `anthropic.Options` value through `aimodel.WithProviderOptions` at client construction.

`anthropic-beta` is generic infrastructure for opting into beta capabilities (compaction, context-editing, structured-outputs, fast-mode, advisor, …). The SDK only emits the header; it models no specific beta capability's fields.

## 3. Request translation (`toAnthropicRequest`)

### 3.1 Pass-through fields

| Canonical field | Anthropic field |
|---|---|
| `Model` | `model` |
| `Temperature` | `temperature` |
| `TopP` | `top_p` |
| `TopK` | `top_k` |
| `Stop []string` | `stop_sequences` |
| `Stream` | `stream` |
| `Thinking` | `thinking` (struct reused directly) |
| `ReasoningEffort` | `output_config.effort` |
| `ResponseFormat` (JSON-schema shape) | `output_config.format` |
| `Container` | `container` |
| `InferenceGeo` | `inference_geo` |

### 3.2 `max_tokens` (required)

Anthropic's `max_tokens` is **mandatory**, while the canonical request may omit it, so there is a three-level fallback:

```
MaxCompletionTokens (preferred) → MaxTokens (deprecated) → 4096 (anthropicDefaultMaxTokens)
```

### 3.3 System messages: only the leading run is hoisted

This is a **position-sensitive** translation. Anthropic puts the system prompt in a top-level `system` field, but since Opus 4.8 (2026-05-28) `messages` may also contain **mid-conversation** `role:"system"` entries.

The rule:

- only the consecutive system messages **before the first non-system message** are hoisted into the top-level `system`;
- any system message appearing later stays **in place** as a `role:"system"` Anthropic message.

A `seenNonSystem` flag implements this. It preserves two things: the instruction's **position semantics**, and **prompt-cache hits** (lifting a mid-conversation instruction to the front invalidates the entire prefix).

The `system` field has two wire shapes, carried by `json.RawMessage`:

| Condition | Shape |
|---|---|
| Plain text, no cache marker | String (multiple entries joined with `\n`) |
| Contains multimodal parts, or any entry set `CacheBreakpoint` | Block array `[{type:"text",text:…}]` |

When a cache marker is set, `cache_control` attaches to the **last** block — Anthropic caches everything "up to and including" that block.

### 3.4 Message translation (`toAnthropicMessage`)

The content shape is chosen per message type:

| Input | Output |
|---|---|
| `RoleTool` | `role:"user"` + `[{type:"tool_result", tool_use_id, content}]`; a missing `ToolCallID` is an error |
| `RoleAssistant` with thinking or tool calls | Block array: `thinking` block → `text` block → one `tool_use` block each (`Input` is `Function.Arguments` verbatim as `json.RawMessage`) |
| Contains multimodal parts | Block array: `text` → `{type:"text"}`; `image_url` → `{type:"image", source:…}` |
| Plain text + `CacheBreakpoint` | Single-element block array (so there is a block to attach `cache_control` to) |
| Plain text | String |

**Image source discrimination**: `parseDataURI` recognizes the `data:<mediaType>;base64,<data>` form → `source{type:"base64", media_type, data}`; anything else is treated as a remote URL → `source{type:"url", url}`.

**Unmapped parts**: `input_audio` / `file` content blocks have no Anthropic counterpart — the `switch` simply does not match them, so they are skipped safely (no error, no empty block).

In every block-array shape, `CacheBreakpoint` attaches to the **last** block.

### 3.5 Tools & `tool_choice`

The base mapping is direct: `Function.Name/Description/Parameters` → `name/description/input_schema`; a true `Tool.CacheBreakpoint` attaches `cache_control` (Anthropic caches every tool definition up to and including it).

The tool-definition extensions are copied verbatim: `Strict` → `strict`, `DeferLoading` → `defer_loading`, `AllowedCallers` → `allowed_callers`, `EagerInputStreaming` → `eager_input_streaming`, `InputExamples` → `input_examples` (all `omitempty`).

`Tool.Type` doubles as the Anthropic tool type, but in OpenAI semantics that canonical field is `"function"` for every ordinary tool, and sending that verbatim would be rejected — so **`"function"` and empty alike are treated as Anthropic's default custom tool and not sent**. Any other value (a versioned built-in such as `web_search_20260209`) passes through with no enumeration and no version-name validation.

The `tool_choice` mapping and the `ParallelToolCalls` folding rules are documented in [../design/tool-use.md](../design/tool-use.md) §2, and the consecutive-`RoleTool` merge in §3.1 of that document.

### 3.6 Reasoning & thinking

- `ReasoningEffort` → `output_config.effort`; empty is omitted. It **supersedes** `thinking.budget_tokens` as the reasoning-depth control for new models. The top-level `anthropicRequest.Effort` is deprecated and no longer assigned, so the two are never sent together.
- `ResponseFormat` → `output_config.format` (`{type:"json_schema", schema:…}`): both OpenAI's nested `json_schema.schema` and the flat `schema` are accepted; the schema passes through unvalidated and unrewritten; a shape with no extractable schema (e.g. `{type:"json_object"}`) yields **no** `format` rather than a fabricated one. When both `effort` and `format` are empty, the whole `output_config` is omitted.
- `Thinking.Type`: `"enabled"` / `"disabled"` / `"adaptive"` (the model sizes its own thinking); kept a `string` for pass-through.
- `Thinking.BudgetTokens`: **deprecated**, retained only for models / callers that still pin an explicit budget.
- `Thinking.Display`: `"omitted"` (since 2026-03-16) suppresses thinking content to speed up streaming.

### 3.7 Prompt caching

Two coexisting modes — per-block breakpoints and request-root automatic caching. Both switches are `json:"-"` struct-local fields that never appear in an OpenAI-shape body. See [../design/prompt-caching.md](../design/prompt-caching.md) §2.

## 4. Response translation (`fromAnthropicResponse`)

Content blocks are aggregated into a single assistant message:

| Block type | Destination |
|---|---|
| `thinking` | Accumulated, joined with `\n` → `Message.Thinking` |
| `text` | Accumulated, joined with `\n` → `Message.Content` |
| `tool_use` | Appended as `ToolCall{Index, ID, Type:"function", Function{Name, Arguments:string(Input)}}` |
| anything else (`server_tool_use`, `web_search_tool_result`, `code_execution_tool_result`, future types) | Raw JSON appended to `Message.ExtraBlocks` |

A `text` block carrying `citations` contributes its text as usual **and additionally** appends its whole original block to `ExtraBlocks` — the annotations are not promoted to canonical fields, but they are not lost either.

Fidelity dictates the implementation: `anthropicResponse.Content`'s element type is `anthropicResponseBlock`, which decodes the known fields **while retaining each block's original bytes** (decoding into a known struct and re-marshalling would drop unmodelled fields). That type also shadows the request-side `ResultContent` (tagged `content` too) with a `json.RawMessage` — the response-side `content` is polymorphic (an array for server-tool results, an object for code-execution results), and decoding it as a `string` would fail the entire response.

Remaining fields: `ID` → `ID`; `Object` is fixed at `"chat.completion"`; `Model` passes through; `container` → `ChatResponse.Container` (`*ResponseContainer{ID, ExpiresAt}` — identical field shape, so it deserializes directly; `ExpiresAt` stays the server string, with no expiry parsing and no auto-renewal); exactly **one `Choice`** is produced (Anthropic has no `n` concept).

### 4.1 Stop-reason mapping (`mapAnthropicStopReason`)

| Anthropic `stop_reason` | Canonical `FinishReason` |
|---|---|
| `end_turn`, `stop_sequence` | `stop` |
| `max_tokens` | `length` |
| `tool_use` | `tool_calls` |
| `model_context_window_exceeded` | Same-named constant (**not** folded into `length` — it is a context-window overflow, not the requested `max_tokens` being hit) |
| `refusal` | Same-named constant (streaming classifiers aborted on a potential policy violation) |
| `pause_turn` | Same-named constant (a long-running / server-tool turn was paused; the client may replay it) |
| anything else | Passed through verbatim as `FinishReason(reason)` |

The last three keep Anthropic's semantics rather than being normalized into `content_filter` / `length` — deliberately, because normalizing destroys exactly the information a caller decides on (replay? switch model? change the prompt?).

### 4.2 `stop_details`

`anthropicResponse.StopDetails` is declared directly as the canonical `*StopDetails` — the field shapes are identical (`type` / `category` / `explanation`), so it **deserializes with no conversion** and is attached to `Choice.StopDetails`.

### 4.3 Usage mapping (`anthropicCanonicalUsage`)

Non-streaming and streaming **share** this helper:

```
PromptTokens       = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
CompletionTokens   = output_tokens
TotalTokens        = PromptTokens + CompletionTokens
CacheReadTokens    = cache_read_input_tokens
CacheWriteTokens   = cache_creation_input_tokens
CacheWrite5mTokens = cache_creation.ephemeral_5m_input_tokens   (when present)
CacheWrite1hTokens = cache_creation.ephemeral_1h_input_tokens   (when present)
ReasoningTokens    = output_tokens_details.thinking_tokens      (when present)
ServerToolUse      = server_tool_use{web_search_requests, web_fetch_requests}  (when present)
InferenceGeo       = inference_geo
ServiceTier        = service_tier
```

Note that `PromptTokens` is the **total input including the cached portion** (`totalInputTokens()`); the cache read/write counts are subsets of it, surfaced separately purely for observability. `Usage.Add` accumulates every count including `ServerToolUse`, but leaves `InferenceGeo` / `ServiceTier` alone — they describe one request, so neither concatenation nor summation is meaningful.

## 5. Streaming (`provider/anthropic/stream.go`)

Anthropic's SSE differs structurally from OpenAI's in two ways, both absorbed by `streamDecoder.Next`:

1. **Events come in pairs**: an `event: <type>` line followed by a `data: <json>` line. After reading `event:` the parser keeps scanning downward, skipping blank lines and `:` comments, until it finds the `data:`.
2. **It is stateful**: `message_start` provides `id` / `model` / input-side usage that later chunks must carry.

Closure state: `msgID`, `model`, `startUsage`, `blockToTool map[int]int`, `nextToolIdx`, `unknownBlocks map[int]bool`.

### 5.1 Event handling

| Event | Behavior |
|---|---|
| `message_start` | Record `msgID` / `model` / `startUsage`; **when a `container` is present, immediately emit a chunk carrying only `Container`**, otherwise emit nothing |
| `content_block_start` (`tool_use`) | Allocate a tool index, record `blockToTool[block index] = tool index`, emit a tool-call chunk carrying `ID` / `Name` |
| `content_block_start` (`text` / `thinking`) | Skip |
| `content_block_start` (unknown type) | Record `unknownBlocks[block index] = true`, emit an `ExtraBlocks` chunk carrying the raw `content_block` |
| `content_block_delta` (block index in `unknownBlocks`) | Whatever the delta's own type, emit an `ExtraBlocks` chunk carrying the raw `delta` |
| `content_block_delta` / `text_delta` | Emit `Delta.Content` |
| `content_block_delta` / `thinking_delta` | Emit `Delta.Thinking` |
| `content_block_delta` / `input_json_delta` | Look the tool index up via `blockToTool`, emit a `Function.Arguments` fragment; skip when not found |
| `content_block_delta` / `signature_delta` | Skip |
| `content_block_delta` (unknown delta type on a **known** block) | Emit an `ExtraBlocks` chunk carrying the raw `delta` |
| `message_delta` | Emit the terminal chunk: `FinishReason` (via `mapAnthropicStopReason`) + `StopDetails`; when it carries `usage`, fold it into `startUsage` via `mergeAnthropicUsage` and produce the full `Usage` via `anthropicCanonicalUsage` |
| `message_stop` | Return `io.EOF` |
| `error` | Return `*APIError{Type, Message}` |
| `ping` / `content_block_stop` | Skip |

### 5.2 Index remapping

Anthropic uses one monotonically increasing index across **all** content blocks (text, thinking, tool_use), whereas the canonical `Message.AppendDelta` expects indices scoped **to tool calls only**. `blockToTool` is that remapping table: it is populated when a `tool_use` `content_block_start` arrives, and later `input_json_delta` events use it to deliver argument fragments to the correct tool call.

### 5.3 Two-part usage assembly

Anthropic puts the input-side counts (including cache read/write, `inference_geo`, `service_tier`, `server_tool_use`) on `message_start` and the final `output_tokens` on `message_delta`. The parser therefore stashes the former in `startUsage`, merges the terminal event in via `mergeAnthropicUsage`, and only then converts — which is what guarantees streaming and non-streaming callers get a **structurally identical** `Usage`.

Merging rather than replacing is the crucial part: the terminal event typically carries only `output_tokens`, so `mergeAnthropicUsage` updates only the fields that event **actually carries**. Otherwise zero values would wipe out the input, cache, geography, service-tier and server-tool information already established at `message_start`.

### 5.4 Container ID is emitted early

See [../design/streaming.md](../design/streaming.md) §3.

## 6. Error handling

`provider.ParseErrorResponse`: the pipeline reads the body (capped at 1 MB) and hands the bytes to the provider, which decodes `{"type":"error","error":{"type":…,"message":…}}` → fill in `APIError{StatusCode, Type, Message}`. When the JSON fails to decode or `message` is empty, the **raw body goes into `Message` verbatim**, so diagnostics are never lost.

An `error` event on the streaming path likewise produces an `*APIError`, but without an HTTP status code (`StatusCode` is 0).

## 7. Known unmapped fields

These canonical fields have no Anthropic counterpart and are silently ignored during translation: `N`, `FrequencyPenalty`, `PresencePenalty`, `Seed`, `User`, `Verbosity`, `Logprobs` / `TopLogprobs` / `LogitBias`, `ServiceTier` (request side only — the response's `usage.service_tier` *is* mapped), `Store`, `Metadata`, `PromptCacheKey`, `Modalities` / `Audio`, `StreamOptions` (Anthropic streaming always returns usage).

`ResponseFormat` is mapped only in its JSON-schema shape (§3.6); other shapes are ignored like the rest, with no fabricated config.

In the other direction, `LogProbs` and `Message.Audio` never appear in an Anthropic response and stay `nil`.
