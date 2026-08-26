# Anthropic Messages API — Change Log

This file records **how aimodel's Anthropic wrapper tracks the official Messages API**: what changed upstream, and how the wrapper followed.

- **Official protocol**: Anthropic Messages API (`POST /v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Implementation notes**: [anthropic-message-api.md](./anthropic-message-api.md)

**Maintenance convention**: see [../architecture.md](../architecture.md) §6. Every entry carries at least a date, the official change, and a wrapper change summary.

Newest first.

---

## 2026-08-26 — tool_result 错误语义补全

**Official change**:无新增协议变更;`tool_result.is_error` 是 Messages API 已有的 request-side 字段。

**Wrapper change**

- `ContentBlock` 新增 `IsError bool`,以 `is_error,omitempty` 原样表达失败工具结果。
- 增加 marshal/unmarshal round-trip 测试,确保 `is_error:true` 不在请求构建过程中丢失。

---

## 2026-08-24 — Claude 5 family compatibility: adaptive thinking, 400 semantics, model retirements

**Official change**: Claude Sonnet 5 (2026-06-30) and Claude Opus 5 (2026-07-24) shipped breaking Messages API changes, recorded here at the 2026-08 baseline:

- Manual extended thinking (`thinking:{enabled,budget_tokens}`) returns **400** on the Claude 5 family; adaptive thinking (`thinking:{type:"adaptive"}`) is the on-mode and is the default when `thinking` is omitted. Opus 4.7/4.8 also reject `budget_tokens`.
- Non-default sampling parameters (`temperature`/`top_p`/`top_k`) return **400** on the Claude 5 family (and on Opus 4.7/4.8); only the 4.6 family and older accept them.
- On Opus 5, `thinking:{type:"disabled"}` is accepted only at effort `low`/`medium`/`high`; pairing it with `xhigh`/`max` returns 400.
- The 1M-context beta header `context-1m-2025-08-07` is retired (2026-04-30) — 1M context is GA and needs no beta header.
- Prompt-cache minimums lowered: 512 tokens (Opus 5 / Fable 5 / Mythos 5), 1024 (Sonnet 5), 4096 (Haiku 4.5); the new tokenizer yields roughly 30% more tokens for the same text.
- Retired models: `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022` (2026-02-19), `claude-3-haiku-20240307` (2026-04-20), `claude-sonnet-4-20250514`, `claude-opus-4-20250514` (2026-06-15).

**Wrapper change**

- **`ModelClaudeOpus5` constant added** (`provider/anthropic/model.go`).
- **Golden baseline corrected**: `TestNativeThinkingMatchesGoldenBaseline` now sends the Claude 5 shape — Sonnet 5 + `thinking:{type:"adaptive", display:"omitted"}` + `output_config.effort:"high"` — replacing the manual `enabled`+`budget_tokens` form, which returns 400 on the Claude 5 family. `testdata/golden/thinking.json` updated to match, and a round-trip test pins the adaptive shape (no `budget_tokens` beside it).
- **400 semantics documented, not intercepted.** The wrapper stays a zero-validation pass-through: `Thinking`/`BudgetTokens`/`Temperature`/`TopP`/`TopK`/`OutputConfig.Effort` are unchanged and are not rejected by model name. The per-family rules are recorded in `anthropic-message-api.md` §2.3 (thinking/effort table) and §7 (sampling), with model, cache-minimum and tokenizer notes.
- **Retired beta example removed**: the `context-1m-2025-08-07` value is gone from the three doc examples; `WithBeta` infrastructure stays.
- **Wire format unchanged**: existing requests serialize exactly as before, so 4.x-family usage is unaffected.

---

## 2026-08-02 — Native-only public API: observable stream usage merging, timeouts, structural errors

**Official change**: none — this is a change to the wrapper's own surface, recorded here because it changes how every Anthropic-side capability is reached.

**Wrapper change**

- **`anthropic` is the entry point.** `anthropic.NewClient(apiKey, ...)` with `Messages` / `MessagesStream`; there is no unified client and no translation in front of them. Every Anthropic-specific surface is an ordinary field of the native types — `MessagesRequest.CacheControl` / `Container` / `InferenceGeo`, per-block and per-tool `cache_control`, `MessagesResponse.StopDetails` / `Container`, and the cache/server-tool/geography counts on `MessagesUsage`. Reasoning: [ADR 0002](../adr/0002-provider-native-as-the-only-public-interface.md).
- **`MessageStream.Usage()` makes the two-part usage merge observable.** Anthropic reports a baseline on `message_start` and the final counts on the terminal `message_delta`; the merge is field-wise, so a terminal event carrying only `output_tokens` does not blank out the input, cache, geography, tier or server-tool numbers.
- **`MessageStream.Message()`** returns the assembled message: content blocks in index order, text and thinking deltas concatenated, tool inputs reassembled from their partial-JSON fragments. `ResponseContentBlock.Raw` still holds each block as it first arrived.
- **Unmodelled blocks are the response's own blocks.** A server-tool result or a future block type is an element of `MessagesResponse.Content` with its verbatim JSON in `Raw`.
- **`WithTimeout(d)`**: bounds a whole call, copying the client configured so far so the caller's `*http.Client` is never mutated.
- **`HTTPError` implements `StatusCode() int`**; the exported field is renamed `Status`. **Breaking** for code reading the field directly.
- **`MessagesUsage.TotalInputTokens()`** is exported: this protocol reports cache counts *alongside* `input_tokens` rather than inside it, so the billable input is their sum.
- **Model, role, effort, thinking, content-source and cache-TTL constants** move into this package (`model.go`).

---

## 2026-07-22 — Public native Messages client

**Official change**

None. This release exposes the already-audited 2026-07-21 Messages API baseline as a public Go surface; the default protocol header remains `anthropic-version: 2023-06-01`.

**Wrapper change**

`anthropic` now exports `MessagesRequest`, `MessagesResponse`, their content/tool/thinking/output/cache/container/usage/error types, and all Messages SSE payload types. `NewClient` returns a `Client` with `Messages` and `MessagesStream`; these methods force the appropriate stream flag on a copy and apply no defaults of their own. Unknown native events retain their complete JSON payload. This does not add Batches, Files, Token Counting, retries, validation, or automatic beta enablement.

## 2026-07-21 — `output_config`, usage extensions, `container`/`inference_geo`, tool fields, unknown-block preservation, profile header

**Official change**

Six additions since the 2026-06-02 sync, tracked here as a **single version item**:

1. Reasoning depth and structured outputs merged into one `output_config` object — `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`) supersedes the former top-level `effort`, and `output_config.format` (`{type:"json_schema", schema:…}`) replaces the deprecated `output_format`.
2. Response `usage` gained `output_tokens_details.thinking_tokens` (thinking cost), `server_tool_use` (`{web_search_requests, web_fetch_requests}`), `inference_geo`, and `service_tier`.
3. Requests accept `container` (reuse a server-side code-execution container) and `inference_geo` (data-residency routing); responses and `message_start` return `container` as `{id, expires_at}`.
4. `content[]` may contain server-tool blocks (`server_tool_use`, `web_search_tool_result`, `code_execution_tool_result`, …) and `citations` annotations.
5. Tool definitions accept `type` (versioned built-in tools; default `custom`), `strict`, `defer_loading`, `allowed_callers`, `eager_input_streaming`, and `input_examples`.
6. The `anthropic-user-profile-id` request header associates requests with an end-user profile.

**Wrapper change**

- **`output_config`**: `MessagesRequest.OutputConfig *OutputConfig` holds `Effort string` and `Format *OutputFormat{Type string, Schema any}`. The schema passes through unvalidated and unrewritten; the object is omitted when both halves are empty. The wire now carries `output_config.effort` rather than the deprecated top-level `effort`.
- **Usage**: `MessagesUsage` gained `OutputTokensDetails`, `ServerToolUse` (`{web_search_requests, web_fetch_requests}`), `InferenceGeo` and `ServiceTier`. Streaming does not overwrite just `output_tokens`: the merge folds in exactly the fields the terminal `message_delta` carries, so the input/cache/geo/tier/server-tool numbers from `message_start` survive.
- **`container` / `inference_geo`**: `MessagesRequest` gained both as `omitempty` strings. `MessagesResponse.Container` decodes into `ResponseContainer{ID, ExpiresAt string}` — `ExpiresAt` stays the server string, with no expiry parsing, auto-renewal or retry. Streaming carries it on the `message_start` event, so tool-only or immediately-ending streams do not lose the ID.
- **Unknown content blocks**: `ResponseContentBlock` decodes the known fields *and* retains each block's verbatim JSON in `Raw`, because re-marshalling a decoded block would lose unmodelled fields. A `text` block carrying `citations` still contributes its text and keeps its whole original block. The stream does the same per event: an unrecognized `content_block_start` keeps its raw `content_block`, and every later delta on that index keeps its raw `delta`, in arrival order. **This also fixes a decode failure**: request-side `content` on a tool result is a string, but response-side `content` is polymorphic (server-tool results carry an array, code-execution results an object), so a response containing one previously failed to decode entirely; the response block type keeps it as `json.RawMessage`.
- **Tool fields**: `MessagesTool` gained `Type string`, `Strict *bool`, `DeferLoading *bool`, `AllowedCallers []string`, `EagerInputStreaming *bool` and `InputExamples []any`, all `omitempty`. Versioned built-in tool types (e.g. `web_search_20260209`) pass through unenumerated and unvalidated; an empty `type` is Anthropic's default custom tool.
- **`anthropic-user-profile-id`**: `WithUserProfileID(string)` (empty ignored, matching `WithVersion`); the header is emitted only when non-empty.
- **Deliberately not implemented**: Anthropic's versioned built-in tool types are not enumerated (type names stay opaque strings); `citations` are not parsed into dedicated fields (the raw block is preserved instead); the `tool_reference` / `tool_search` meta-protocol is deferred.

## 2026-06-02 — Request-root automatic caching + per-TTL `cache_creation` in usage

**Official change**

Since 2026-02-19, **automatic caching** lets a request carry a single request-root `cache_control` (`{type:"ephemeral"}`, optional `ttl:"1h"`) instead of per-block markers: the server caches the last cacheable block and advances the breakpoint forward as the conversation grows. The response `usage` additionally returns a `cache_creation` object breaking cache writes down by TTL — `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`, summing to `cache_creation_input_tokens`.

The wrapper previously supported only block-level `cache_control` and exposed no cache-write counts at all.

**Wrapper change**

- `MessagesRequest.CacheControl *CacheControl` carries the request-root marker, and `CacheControl` gained `TTL string`. Empty `ttl` → omitted → default 5 minutes; `"1h"` → the 1-hour cache. It coexists independently with per-block `cache_control`.
- `MessagesUsage` gained `CacheCreationInputTokens` (the total) and `CacheCreation *CacheCreation`, which breaks it down per TTL.
- Both the non-streaming and the streaming path carry the `message_start` usage forward, so the counts survive to the end of the stream.

## 2026-06-02 — Pass through the `top_k` sampling parameter

**Official change**: the Messages API natively supports `top_k` (top-k truncation sampling — restrict sampling to the K most-likely tokens at each step).

**Wrapper change**: `MessagesRequest` gained `TopK *int`, next to `TopP` and `omitempty`, passed through verbatim.

## 2026-06-02 — Configurable `anthropic-beta` / `anthropic-version` headers

**Official change**: many Anthropic capabilities (compaction, context-editing, structured-outputs during beta, fast-mode, advisor, …) are opt-in via the `anthropic-beta` request header (multiple values comma-joined); the `anthropic-version` header (default `2023-06-01`) selects the API version.

**Wrapper change**: infrastructure only — no specific beta capability is wired up. `WithBeta(values ...string)` accumulates across calls, ignores empty strings and is comma-joined on the wire; `WithVersion(string)` (empty keeps the default). `anthropic-version` is always sent, `anthropic-beta` only when non-empty.

## 2026-06-02 — New `stop_reason` constants and `stop_details` (refusal classification)

**Official change**

`stop_reason` gained three values:

- `model_context_window_exceeded` (2025-09-29) — input + output exceeded the model's context window, **distinct from** hitting the requested `max_tokens`;
- `pause_turn` — a long-running / server-tool turn was paused and may be replayed;
- `refusal` (2026-05-28, Opus 4.8) — streaming classifiers intervened on a potential policy violation.

When `stop_reason` is `refusal`, the response and the terminal `message_delta` carry `stop_details` (`{type, category, explanation}`) with the classification.

**Wrapper change**

Added the `StopReasonModelContextWindowExceeded` / `StopReasonRefusal` / `StopReasonPauseTurn` constants, whose values are the verbatim Anthropic strings; `MessagesResponse.StopReason` stays an open string, so an unlisted value still passes through. Added `StopDetails{Type, Category, Explanation}` on `MessagesResponse` and on the terminal `message_delta`.

## 2026-06-02 — Map `effort`, support `thinking.display`, deprecate `budget_tokens`

**Official change**: since 2026-02-05 the top-level `effort` parameter GA'd and supersedes `thinking.budget_tokens` for reasoning depth (also enabling `thinking.type:"adaptive"`, where the model sizes its own thinking). Since 2026-03-16, `thinking.display:"omitted"` suppresses thinking content to speed up streaming.

**Wrapper change**: `MessagesThinking` gained `Display string`; `Type` stays a `string` so `"adaptive"` passes through; `BudgetTokens` carries a deprecation note pointing at `effort` / `adaptive`.

> Superseded by the 2026-07-21 entry, which moves `effort` into `output_config`.

## 2026-06-02 — `tool_choice` `"none"` mapping and `disable_parallel_tool_use`

**Official change**: since 2024-10-03, `tool_choice` accepts `disable_parallel_tool_use` (at most one tool call per turn) alongside `auto`/`any`/`tool`. Since 2025-02-27, `tool_choice:{type:"none"}` explicitly forbids any tool call — distinct from omitting the field, which lets the model choose.

**Wrapper change**: `ToolChoice` models `{type:"none"}` — distinct from omitting the field — and gained `DisableParallelToolUse *bool`. Anthropic rejects a `tool_choice` on a tool-less request, and the flag is meaningless on `{type:"none"}`.

## [Baseline] 2026-06-02

- **Official protocol**: Anthropic Messages API (`/v1/messages`, keyed by the official documentation endpoint)
- **Summary**: wrapped the non-streaming and streaming Messages API; system instructions carried in the top-level `system` field; streaming uses Anthropic SSE event types (`content_block_delta`, `message_delta`, …).
