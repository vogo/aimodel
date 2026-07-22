# OpenAI Chat Completions — Wrapper Design & Implementation

- **Official protocol**: OpenAI Chat Completions API (`POST {baseURL}/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Implementation**: `provider/openai/openai.go` (request building and error parsing), `provider/openai/stream.go` (SSE parsing), `core/schema.go` (the canonical types *are* the OpenAI shape)
- **Change log**: [openai-api-changes.md](./openai-api-changes.md)

Canonical type semantics live in [../design/data-model.md](../design/data-model.md); this document covers what is specific to the OpenAI path.

---

## 1. The zero-translation path

The SDK uses the **OpenAI Chat Completions format as its canonical representation** (see [../api.md](../api.md) §2), so the OpenAI path has **no translation layer at all**:

```
ChatRequest ──json.Marshal──▶ POST {baseURL}/chat/completions
ChatResponse ◀──json.Decode── response body
```

`ChatRequest` / `ChatResponse` / `Message` / `Choice` / `Usage` in `schema.go` are the OpenAI wire structures. Which means: **adding a new OpenAI request parameter is just one `omitempty` field on `ChatRequest`** — no protocol code changes.

The same property makes it fit every OpenAI-compatible backend (DeepSeek, Kimi/Moonshot, GLM, Qwen, Doubao, MiniMax, Gemini's OpenAI-compatible endpoint, …): a backend's private extension parameters pass straight through as long as the shape matches, and unknown fields are ignored by the backend itself.

## 2. Sending the request (`doRequest`)

```
POST {baseURL}/chat/completions
Content-Type: application/json
Authorization: Bearer {apiKey}
```

- An empty `baseURL` returns `ErrNoBaseURL` immediately — unlike Anthropic, there are far too many OpenAI-compatible backends to pick a default.
- `baseURL` already had its trailing `/` stripped by `WithBaseURL` / the environment reader, so joining never produces `//`.
- The endpoint path is fixed at `/chat/completions`, so the caller's `baseURL` is expected to include the version segment (e.g. `https://api.openai.com/v1`).

## 3. Non-streaming (`openaiChatCompletion`)

1. `req.clone()`, forcing `Stream = false`;
2. fill in the default model;
3. send the request;
4. **non-200 status** → `parseErrorResponse` (§6);
5. decode into `ChatResponse`;
6. **200 but the body contains `error`** → still build an `APIError`;
7. empty `Choices` → `ErrEmptyResponse`.

Step 6 is a necessary defence: OpenAI-compatible implementations are not consistent about how they report errors.

## 4. Streaming (`openaiChatCompletionStream`)

Same flow as non-streaming, with two differences:

1. `Stream = true` is forced;
2. **when `StreamOptions` is `nil`, `{IncludeUsage: true}` is added automatically** — otherwise the terminal chunk carries no usage and `Stream.Usage()` returns nothing. An explicit caller setting is respected.

On success it returns `newStream(resp.Body)`; on failure it closes the body before returning the error.

### 4.1 SSE parsing (`openaiRecvFunc`)

OpenAI's SSE is a **stateless line-by-line `data:` stream**:

| Line | Handling |
|---|---|
| Empty | Skip |
| Starts with `:` (SSE comment / heartbeat) | Skip |
| Not prefixed `data: ` | Skip |
| `data: [DONE]` | Return `io.EOF` |
| `data: {json}` | Parse and emit a chunk |

Each chunk is decoded **once** via `streamChunkOrError` (an embedded `StreamChunk` plus an optional `Error`): if it carries an `error` field, an `*APIError` is returned directly (with no HTTP status code); otherwise the chunk is returned. A JSON failure returns a wrapped `decode stream chunk` error.

After the scan ends: return the `Scanner`'s error if it has one, otherwise `io.EOF` — which tolerates compatible backends that never send `[DONE]`.

Line cap `maxStreamLineSize = 1 MB`, buffer starting at 64 KB.

### 4.2 Delta merging

Streaming deltas accumulate through the canonical `Message.AppendDelta` / `ToolCall.Merge` — see [../design/streaming.md](../design/streaming.md) §2.

## 5. OpenAI-specific field notes

Canonical field semantics are in [../design/data-model.md](../design/data-model.md). Two details are specific to this path:

### 5.1 Nested usage fields are promoted

OpenAI puts two breakdown counts inside nested objects; `Usage.UnmarshalJSON` **promotes them to the top-level canonical fields**:

| OpenAI wire path | Canonical field |
|---|---|
| `prompt_tokens_details.cached_tokens` | `CacheReadTokens` |
| `completion_tokens_details.reasoning_tokens` | `ReasoningTokens` |

The rule is **explicit top-level wins** — the nested value is only used when the top-level field is 0.

OpenAI has no notion of cache-write billing, so the `CacheWrite*` fields are always 0 (omitted) on this path.

### 5.2 Prompt caching

OpenAI caches prefixes over ~1024 tokens **automatically**, with no request-side marker. That is why the Anthropic-only switches (`Message.CacheBreakpoint`, `Tool.CacheBreakpoint`, `ChatRequest.AutoCache`) are all `json:"-"` and never appear in an OpenAI request body. To improve hit rates, use `PromptCacheKey`. See [../design/prompt-caching.md](../design/prompt-caching.md).

## 6. Error handling (`parseErrorResponse`)

1. Read the response body, capped at `maxErrorBodySize = 1 MB` (`io.LimitReader`);
2. read failure → `APIError{StatusCode, Message:"failed to read error response", Err}`;
3. decode as `{"error":{code,message,param,type}}`;
4. **decode failure or missing `error` → put the raw body into `Message`**, so diagnostics are never lost;
5. success → `APIError{StatusCode, Code, Message, Type}`.

## 7. Fields with no Anthropic counterpart

These canonical fields are silently ignored when the client switches to `ProtocolAnthropic`: `N`, `FrequencyPenalty`, `PresencePenalty`, `Seed`, `User`, `Verbosity`, `Logprobs` / `TopLogprobs` / `LogitBias`, `ServiceTier` (request side only — the response's `usage.service_tier` *is* mapped), `Store`, `Metadata`, `PromptCacheKey`, `Modalities` / `Audio`, `StreamOptions`.

`ResponseFormat` is mapped only in its JSON-schema shape (to Anthropic's `output_config.format`); other shapes are ignored like the rest.

When writing cross-protocol code, treat all of these as best-effort parameters.
