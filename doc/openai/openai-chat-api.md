# OpenAI Chat Completions — Wrapper Design & Implementation

- **Official protocol**: OpenAI Chat Completions API (`POST {baseURL}/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Implementation**: `provider/openai/wire.go` (native wire types), `provider/openai/native.go` (native client), `provider/openai/translate.go` (canonical translation), `provider/openai/openai.go` and `stream.go` (provider boundary)
- **Change log**: [openai-api-changes.md](./openai-api-changes.md)

Canonical type semantics live in [../design/data-model.md](../design/data-model.md); this document covers what is specific to the OpenAI path.

---

## 1. Provider mapping

The OpenAI provider explicitly translates shared canonical fields into its independent Chat Completions wire model:

```
ChatRequest ──toOpenAIRequest──▶ ChatCompletionRequest ──▶ POST {baseURL}/chat/completions
ChatResponse ◀─fromOpenAIResponse─ ChatCompletionResponse ◀── response body
```

New OpenAI-only parameters must be added to the provider's native surface, not `ais.ChatRequest`. Canonical admission still requires a verified mapping in at least two providers.

### 1.1 Native client

`openai.NewClient(apiKey, ...ClientOption)` returns a native `Client`. `WithBaseURL` and `WithHTTPClient` configure it; the default base URL is `https://api.openai.com/v1`. `ChatCompletions` returns `*ChatCompletionResponse`, while `ChatCompletionsStream` returns a stream whose `Recv` exposes every `*ChatCompletionChunk` in wire order and whose `Close` is idempotent. Both methods copy the request before forcing the appropriate `stream` value, so caller state is unchanged. These calls bypass canonical translation and are the entry point for logprobs, audio, file input, metadata, storage, prompt-cache routing and other OpenAI-only features.

## 2. Sending the request (`doRequest`)

```
POST {baseURL}/chat/completions
Content-Type: application/json
Authorization: Bearer {apiKey}
```

- An empty `baseURL` returns `ErrNoBaseURL` immediately — unlike Anthropic, there are far too many OpenAI-compatible backends to pick a default.
- `baseURL` already had its trailing `/` stripped by `WithBaseURL` / the environment reader, so joining never produces `//`.
- The endpoint path is fixed at `/chat/completions`, so the caller's `baseURL` is expected to include the version segment (e.g. `https://api.openai.com/v1`).

## 3. Non-streaming

The shared pipeline (`chat.go`) drives every call; the OpenAI provider supplies the vendor steps:

1. `req.Clone()`, forcing `Stream = false` (pipeline);
2. fill in the default model (pipeline);
3. `provider.NewChatRequest` builds the request; the pipeline sends it;
4. **non-2xx status** → `provider.ParseErrorResponse` (§6);
5. `provider.ParseChatResponse` decodes `ChatCompletionResponse` and translates it into `ChatResponse`;
6. **2xx but the body contains `error`** → still build an `APIError`;
7. empty `Choices` → `ErrEmptyResponse`.

Step 6 is a necessary defence: OpenAI-compatible implementations are not consistent about how they report errors.

## 4. Streaming

Same flow as non-streaming, with two differences:

1. `Stream = true` is forced by the pipeline;
2. `provider.NewChatRequest` always adds the wire-only `stream_options.include_usage=true` for a streaming request, so the terminal usage-only chunk remains observable. Non-streaming requests omit it.

On success the pipeline wraps the body in a `Stream` backed by `provider.NewStreamDecoder`; on failure it closes the body before returning the error.

### 4.1 SSE parsing (`streamDecoder.Next`)

OpenAI's SSE is a **stateless line-by-line `data:` stream**:

| Line | Handling |
|---|---|
| Empty | Skip |
| Starts with `:` (SSE comment / heartbeat) | Skip |
| Not prefixed `data: ` | Skip |
| `data: [DONE]` | Return `io.EOF` |
| `data: {json}` | Parse and emit a chunk |

Each chunk is decoded into `ChatCompletionChunk`, checked for a body-level error, and translated by `fromOpenAIChunk`. An error becomes `*APIError` (with no HTTP status code); a JSON failure returns a wrapped `decode stream chunk` error.

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

OpenAI caches prefixes automatically, with no canonical request-side control. Anthropic cache controls live in its extension API and never appear in an OpenAI request body. See [../design/prompt-caching.md](../design/prompt-caching.md).

## 6. Error handling (`provider.ParseErrorResponse`)

1. The pipeline reads the response body, capped at `maxErrorBodySize = 1 MB` (`io.LimitReader`), and hands the bytes to the provider (a read failure yields `APIError{StatusCode, Message:"failed to read error response", Err}` before the provider is called);
2. the provider decodes as `{"error":{code,message,param,type}}`;
3. **decode failure or missing `error` → put the raw body into `Message`**, so diagnostics are never lost;
4. success → `APIError{StatusCode, Code, Message, Type}`.

## 7. Mapping boundary

OpenAI-only request fields, log probabilities, audio/file payloads and generated-audio response data are intentionally absent from canonical types. Use the OpenAI native API for those capabilities. `ResponseFormat` is shared only in its JSON-schema shape; unsupported shapes do not produce an Anthropic output format.
