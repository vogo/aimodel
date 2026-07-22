# Anthropic Messages API — Change Log

This file records **how aimodel's Anthropic wrapper tracks the official Messages API**: what changed upstream, and how the wrapper followed.

- **Official protocol**: Anthropic Messages API (`POST /v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Implementation notes**: [anthropic-message-api.md](./anthropic-message-api.md)
- **Index of both protocols**: [../../CHANGES.md](../../CHANGES.md)

**Maintenance convention**: see [../architecture.md](../architecture.md) §6. Every entry carries at least a date, the official change, and a wrapper change summary.

Newest first.

---

## 2026-07-22 — Canonical de-vendoring: Anthropic-only surfaces move to the unified extension channel

**Official change**

None — this is a wrapper-side refactor. The wire behavior is unchanged: with no extension set, request JSON and headers are byte-for-byte identical to the previous release, and every previously observable Anthropic response detail remains reachable.

**Wrapper change (breaking, compile-time)**

All single-vendor canonical surfaces moved out of the canonical/root packages (now `ais`) into this package, reachable through the unified extension channel `ais.Extensions` (see [../architecture.md](../architecture.md) §2). Migration is one-to-one:

| Before (canonical) | After (`provider/anthropic`) |
|---|---|
| `ChatRequest.AutoCache` / `AutoCacheTTL` | `anthropic.ExtendRequest(req, &anthropic.RequestExtension{AutoCache: true, AutoCacheTTL: "1h"})` |
| `ChatRequest.Container` / `InferenceGeo` | `anthropic.RequestExtension{Container: …, InferenceGeo: …}` |
| `Message.CacheBreakpoint` | `anthropic.ExtendMessage(&msg, &anthropic.MessageExtension{CacheBreakpoint: true})` |
| `Message.ExtraBlocks` | `anthropic.MessageExtensionOf(&msg).ExtraBlocks` |
| `Tool.CacheBreakpoint` / `DeferLoading` / `AllowedCallers` / `EagerInputStreaming` / `InputExamples` | `anthropic.ExtendTool(&tool, &anthropic.ToolExtension{…})` (`Tool.Strict` stays canonical — OpenAI `function.strict` / Anthropic `strict`) |
| `Choice.StopDetails` / `StreamChunkChoice.StopDetails` (and the `StopDetails` type) | `anthropic.ChoiceExtensionOf(&choice).StopDetails` / `anthropic.ChunkChoiceExtensionOf(&chunkChoice).StopDetails` (`anthropic.StopDetails`) |
| `ChatResponse.Container` / `StreamChunk.Container` (and `ResponseContainer`) | `anthropic.ResponseExtensionOf(resp).Container` / `anthropic.ChunkExtensionOf(chunk).Container` (`anthropic.ResponseContainer`) |
| `Usage.CacheWriteTokens` / `CacheWrite5mTokens` / `CacheWrite1hTokens` / `ServerToolUse` / `InferenceGeo` (and `ServerToolUse`) | `anthropic.UsageExtensionOf(&usage)` (`anthropic.UsageExtension` / `anthropic.ServerToolUse`); `Usage.CacheReadTokens`, `ReasoningTokens`, `ServiceTier` stay canonical |
| `aimodel.FinishReasonRefusal` / `FinishReasonPauseTurn` / `FinishReasonModelContextWindowExceeded` | `anthropic.FinishReasonRefusal` / `FinishReasonPauseTurn` / `FinishReasonModelContextWindowExceeded` (the open `FinishReason` string still passes the values through) |

Client-level configuration is unchanged: `anthropic.Options{Beta, Version, UserProfileID}` via `aimodel.WithProviderOptions` remains the only client extension entry (the old root `WithAnthropic*` options were already removed in the provider-subpackage refactor).

Contract notes: extension values are read-only once attached; `ChatRequest.Clone` copies every node's extension map; `Message.AppendDelta` merges this provider's namespace through `ais.ExtensionMerger` (copy-on-write, arrival order preserved); a wrong-typed value in the `anthropic` namespace fails `NewChatRequest` with a `*ais.ExtensionTypeError` naming the node, before any network I/O.

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

- **`output_config`**: `anthropicRequest` gained `OutputConfig *anthropicOutputConfig` (holding `Effort string` and `Format *anthropicOutputFormat{Type, Schema any}`). New `toAnthropicOutputConfig` / `toAnthropicOutputFormat` helpers map `ReasoningEffort` → `effort` and a JSON-schema `ResponseFormat` → `format`, accepting both OpenAI's nested `json_schema.schema` and the flat `schema`; the schema passes through unvalidated and unrewritten, and a shape with no extractable schema (e.g. `{type:"json_object"}`) produces no `format` rather than a fabricated one. The field is omitted when both halves are empty. **Behavior change**: the top-level `anthropicRequest.Effort` is marked deprecated and no longer assigned, so the wire now carries `output_config.effort`; the field itself is retained for internal source compatibility.
- **Usage**: `anthropicUsage` gained `OutputTokensDetails` / `ServerToolUse` / `InferenceGeo` / `ServiceTier`. The canonical `Usage` gained `ServerToolUse *ServerToolUse` (zero counts `omitempty`), `InferenceGeo`, and `ServiceTier`; `ReasoningTokens` now also sources Anthropic's `output_tokens_details.thinking_tokens` (an explicit top-level `reasoning_tokens` still wins). All three round-trip through `usageJSON`; `Usage.Add` accumulates the server-tool counts but leaves geo/tier alone (they describe a single request). Streaming no longer overwrites just `output_tokens`: the new `mergeAnthropicUsage` folds in exactly the fields the terminal `message_delta` carries, so the input/cache/geo/tier/server-tool information from `message_start` survives.
- **`container` / `inference_geo`**: `ChatRequest` gained two `omitempty` fields passed straight through to `anthropicRequest`. `anthropicResponse.Container` deserializes directly into the new canonical `ResponseContainer{ID, ExpiresAt string}` (identical field shape) — `ExpiresAt` stays the server string, with no expiry parsing, auto-renewal, or retry. Non-streaming exposes it on `ChatResponse.Container`; streaming adds `StreamChunk.Container`, emitted **once** the moment `message_start` is parsed, so tool-only or immediately-ending streams do not lose the ID.
- **Unknown content blocks**: `Message` gained `ExtraBlocks []json.RawMessage` (`json:"-"`, runtime-only). `anthropicResponse.Content` changed from `[]anthropicContentBlock` to the new `[]anthropicResponseBlock`, which decodes the known fields *and* retains each block's verbatim JSON (re-marshalling a decoded block would lose unmodelled fields). `fromAnthropicResponse` gained a `default` branch appending unrecognized blocks; a `text` block carrying `citations` still contributes its text and additionally keeps its whole original block. `anthropic_stream.go` does the same per event: an unrecognized `content_block_start` emits its raw `content_block` and records the index, so every later delta on that index — plus any unknown delta type on a known block — emits its raw `delta`, in arrival order. **Also fixes a decode failure**: `anthropicContentBlock.ResultContent` is a `string` tagged `content`, but response-side `content` is polymorphic (server-tool results carry an array, code-execution results an object), so a response containing one previously failed to decode entirely; `anthropicResponseBlock` shadows it with a `json.RawMessage`. `Message.AppendDelta` appends `ExtraBlocks` without parsing or merging. `signature_delta` stays ignored and `text` / `thinking` / `tool_use` / `input_json_delta` are unchanged.
- **Tool fields**: `Tool` and `anthropicTool` both gained `Strict *bool`, `DeferLoading *bool`, `AllowedCallers []string`, `EagerInputStreaming *bool`, `InputExamples []any` (all `omitempty`), copied verbatim in the tool-conversion loop; `anthropicTool` also gained `Type string`. `ChatRequest.clone()` now duplicates each tool's `AllowedCallers` / `InputExamples` slices (elements stay shallow, matching the existing `any` contract). **Deviation from the plan**: mapping `Tool.Type` through unconditionally was the original intent, but the canonical `Type` is OpenAI's `"function"` for every ordinary tool and sending that to Anthropic is invalid. `"function"` and empty are therefore treated as Anthropic's default custom tool and omitted; any other value (versioned built-ins like `web_search_20260209`) passes through unenumerated and unvalidated.
- **`anthropic-user-profile-id`**: `Client` gained `anthropicUserProfileID` plus `WithAnthropicUserProfileID(string)` (empty ignored, matching `WithAnthropicVersion`); `setAnthropicHeaders` emits the header only when non-empty.
- **Default behavior unchanged**: with none of the new fields set, the request wire JSON is identical to before except that `ReasoningEffort` now lands in `output_config.effort` instead of the top-level `effort`; the new `json:"-"` members never reach the OpenAI body; responses without the new members leave every added field zero/nil and omitted from JSON; no new header is sent by default.
- **Deliberately not implemented**: Anthropic's versioned built-in tool types are not enumerated (type names stay opaque strings); `citations` are not parsed into canonical fields (the raw block is preserved instead); the `tool_reference` / `tool_search` meta-protocol is deferred; the deprecated top-level `effort` field is retained (unassigned) rather than removed this release.

## 2026-07-10 — Merge consecutive parallel `tool_result` blocks into one `user` message

**Official change**

The Messages API requires all `tool_result` blocks for one assistant turn's parallel `tool_use` to arrive inside a **single** `role:"user"` message immediately after that turn. Emitting one `user` message per result — the naive 1:1 mapping of canonical `role:"tool"` messages — makes the endpoint reject the extra results (`without tool_result blocks immediately after: call_xxx`), breaking any parallel-tool-use request (notably on DeepSeek's Anthropic-compatible endpoint, which validates this strictly).

**Wrapper change**

`toAnthropicRequest` now detects a run of consecutive canonical `RoleTool` messages and serializes it once into a single `role:"user"` `anthropicMessage` whose content array holds every `tool_result` block in the original order; the loop index skips the consumed run. The per-block builder was extracted into a `toolResultBlock` helper shared by the run-merge path (`toAnthropicToolResultMessage`) and the single-message fallback, so `CacheBreakpoint` still attaches `cache_control` to exactly the flagged block and the wire shape for a lone tool result is unchanged. A non-consecutive `RoleTool` (separated by a user/assistant/system turn) keeps its own user message; mid-conversation system handling and assistant `tool_use` serialization are untouched. The non-streaming and streaming paths share `toAnthropicRequest`, so both benefit with no separate change. **Default behavior unchanged**: a single tool result still emits one `role:"user"` with a one-element `tool_result` array.

## 2026-06-02 — Request-root automatic caching + per-TTL `cache_creation` in usage

**Official change**

Since 2026-02-19, **automatic caching** lets a request carry a single request-root `cache_control` (`{type:"ephemeral"}`, optional `ttl:"1h"`) instead of per-block markers: the server caches the last cacheable block and advances the breakpoint forward as the conversation grows. The response `usage` additionally returns a `cache_creation` object breaking cache writes down by TTL — `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`, summing to `cache_creation_input_tokens`.

The wrapper previously supported only block-level `cache_control` and exposed no cache-write counts at all.

**Wrapper change**

- `ChatRequest` gained `AutoCache bool` / `AutoCacheTTL string`, both `json:"-"` (struct-local, never on the OpenAI-shape body, mirroring `CacheBreakpoint`).
- When `AutoCache` is true, `toAnthropicRequest` sets the new top-level `anthropicRequest.CacheControl`; `anthropicCacheControl` gained `TTL string`. Empty `ttl` → omitted → default 5 minutes; `"1h"` → the 1-hour cache. It coexists independently with per-block `CacheBreakpoint`.
- `anthropicUsage` gained `CacheCreation *anthropicCacheCreation`; the canonical `Usage` gained `CacheWriteTokens` (← `cache_creation_input_tokens`, total — previously unexposed), `CacheWrite5mTokens`, and `CacheWrite1hTokens`, all accumulated by `Usage.Add` and round-tripped through JSON.
- The non-streaming path (`fromAnthropicResponse`) and the streaming path (`anthropic_stream.go`, refactored to carry the `message_start` usage forward) share the new `anthropicCanonicalUsage`.
- **Default behavior unchanged**: with `AutoCache` false no request-root `cache_control` is emitted; OpenAI responses (no `cache_creation`) leave the new fields at 0/omitted.

## 2026-06-02 — Pass through the `top_k` sampling parameter

**Official change**: the Messages API natively supports `top_k` (top-k truncation sampling — restrict sampling to the K most-likely tokens at each step). The canonical `ChatRequest` had no corresponding field.

**Wrapper change**: `ChatRequest` gained `TopK *int` (placed next to `TopP`), `anthropicRequest` gained the matching field, and `toAnthropicRequest` maps it straight through. OpenAI's Chat Completions has no `top_k`, and the canonical request *is* the OpenAI shape — so the `omitempty` field is simply omitted when unset and passed through verbatim when set (compatible backends that accept it honour it; the rest ignore the unknown field).

## 2026-06-02 — Configurable `anthropic-beta` / `anthropic-version` headers

**Official change**: many Anthropic capabilities (compaction, context-editing, structured-outputs during beta, fast-mode, advisor, …) are opt-in via the `anthropic-beta` request header (multiple values comma-joined); the `anthropic-version` header (default `2023-06-01`) selects the API version.

**Wrapper change**: infrastructure only — no specific beta capability is wired up. `Client` gained `anthropicBeta []string` / `anthropicVersion string`, plus `WithAnthropicBeta(values ...string)` (accumulates across calls, ignores empty strings, comma-joined on the wire) and `WithAnthropicVersion(string)` (empty keeps the default). `setAnthropicHeaders` emits `anthropic-version` always and `anthropic-beta` only when non-empty. **Default behavior unchanged.**

## 2026-06-02 — New `stop_reason` constants and `stop_details` (refusal classification)

**Official change**

`stop_reason` gained three values:

- `model_context_window_exceeded` (2025-09-29) — input + output exceeded the model's context window, **distinct from** hitting the requested `max_tokens`;
- `pause_turn` — a long-running / server-tool turn was paused and may be replayed;
- `refusal` (2026-05-28, Opus 4.8) — streaming classifiers intervened on a potential policy violation.

When `stop_reason` is `refusal`, the response and the terminal `message_delta` carry `stop_details` (`{type, category, explanation}`) with the classification.

**Wrapper change**

`mapAnthropicStopReason` previously mapped only `end_turn`/`stop_sequence`/`max_tokens`/`tool_use` and passed everything else through as a raw string. It now maps the three new values to named constants `FinishReasonModelContextWindowExceeded` / `FinishReasonRefusal` / `FinishReasonPauseTurn`, whose values are the verbatim Anthropic strings (purely additive — they already flowed through as raw strings, and they are **not** folded into `content_filter`/`length`, preserving Anthropic semantics). Added the canonical `StopDetails{Type, Category, Explanation}`; `Choice` and `StreamChunkChoice` gained `StopDetails *StopDetails`. `anthropicResponse` and `anthropicMessageDeltaData` deserialize straight into the canonical type (identical field shape).

## 2026-06-02 — Map `effort`, support `thinking.display`, deprecate `budget_tokens`

**Official change**: since 2026-02-05 the top-level `effort` parameter GA'd and supersedes `thinking.budget_tokens` for reasoning depth (also enabling `thinking.type:"adaptive"`, where the model sizes its own thinking). Since 2026-03-16, `thinking.display:"omitted"` suppresses thinking content to speed up streaming.

**Wrapper change**: `toAnthropicRequest` maps the canonical `ChatRequest.ReasoningEffort` to the new top-level `anthropicRequest.Effort` (empty omitted; the same field still drives OpenAI's `reasoning_effort`). `Thinking` gained `Display string`; `Type` stays a `string` so `"adaptive"` passes through; `BudgetTokens` got a deprecation note pointing at `effort` / `adaptive`.

> Superseded by the 2026-07-21 entry, which moves `effort` into `output_config`.

## 2026-06-02 — `tool_choice` `"none"` mapping and `disable_parallel_tool_use`

**Official change**: since 2024-10-03, `tool_choice` accepts `disable_parallel_tool_use` (at most one tool call per turn) alongside `auto`/`any`/`tool`. Since 2025-02-27, `tool_choice:{type:"none"}` explicitly forbids any tool call — distinct from omitting the field, which lets the model choose.

**Wrapper change**: `convertToolChoice` previously returned `nil` for `"none"`, dropping the field entirely (model-chooses semantics); it now maps to `{type:"none"}`. `anthropicToolChoice` gained `DisableParallelToolUse *bool`. The assembly folds in the canonical `ParallelToolCalls`: an explicit `false` sets `disable_parallel_tool_use:true`, defaulting the choice to `{type:"auto"}` to carry the flag when none is named but tools are present (a `tool_choice` on a tool-less request is rejected), and **never** attaching it to `{type:"none"}`. Unset or `true` leaves the choice untouched.

## 2026-06-02 — Preserve mid-conversation `system` messages (hoisting fix)

**Official change**: since 2026-05-28 (Opus 4.8), `messages` may contain `role:"system"` entries at non-leading positions, letting callers change instructions mid-session while keeping prompt-cache hits.

**Wrapper change**: `toAnthropicRequest` previously hoisted **every** `RoleSystem` message into the top-level `system` regardless of position, wrongly lifting mid-conversation instructions to the front and dropping their position semantics. Now only the **leading** run (before the first non-system message) is extracted; a system message appearing mid-conversation falls through to `toAnthropicMessage` and stays in place as a `role:"system"` message. `CacheBreakpoint` and the block-vs-string behavior are unchanged.

## [Baseline] 2026-06-02

- **Official protocol**: Anthropic Messages API (`/v1/messages`, keyed by the official documentation endpoint)
- **Summary**: wrapped the non-streaming and streaming Messages API, with bidirectional translation to/from the canonical OpenAI-compatible types; system messages extracted into a separate `system` field; streaming uses Anthropic SSE event types (`content_block_delta`, `message_delta`, …).
