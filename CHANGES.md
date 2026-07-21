# CHANGES

This file records the **sync status between aimodel's wrapper code and the official API protocols**, so maintainers and AI can track "which official API a wrapper maps to, and which version it is currently synced with".

The official API documentation entries are listed in the "Official API References" section of [README.md](./README.md) / [CLAUDE.md](./CLAUDE.md).

**Maintenance convention**: when an official API changes, update all three in sync — the wrapper code, the documentation (README.md / CLAUDE.md), and this file — keeping them consistent and continuously up to date. Each entry must include at least: date, target official protocol and version, change summary.

---

## 2026-07-21 — Anthropic: `output_config`, usage extensions, `container`/`inference_geo`, tool fields, unknown-block preservation, profile header

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Six additions since the 2026-06-02 sync, shipped here as one version item. (1) Reasoning depth and structured outputs moved into a single `output_config` object — `output_config.effort` (`low`/`medium`/`high`/`xhigh`/`max`) supersedes the former top-level `effort`, and `output_config.format` (`{type:"json_schema", schema:…}`) replaces the deprecated `output_format`. (2) Response `usage` gained `output_tokens_details.thinking_tokens` (thinking cost), `server_tool_use` (`{web_search_requests, web_fetch_requests}`), `inference_geo`, and `service_tier`. (3) Requests accept `container` (reuse a server-side code-execution container) and `inference_geo` (data-residency routing); responses and `message_start` return `container` as `{id, expires_at}`. (4) Server-side tool blocks (`server_tool_use`, `web_search_tool_result`, `code_execution_tool_result`, …) and `citations` annotations appear in `content[]`. (5) Tool definitions accept `type` (versioned built-in tools; default `custom`), `strict`, `defer_loading`, `allowed_callers`, `eager_input_streaming`, and `input_examples`. (6) The `anthropic-user-profile-id` request header associates requests with an end-user profile.
- **Change summary**:
  - **`output_config`** — `anthropicRequest` gained `OutputConfig *anthropicOutputConfig` (`json:"output_config,omitempty"`) holding `Effort string` and `Format *anthropicOutputFormat` (`{Type string, Schema any}`). `toAnthropicRequest` (`anthropic.go`) now maps `ChatRequest.ReasoningEffort` → `output_config.effort` and a JSON-schema `ChatRequest.ResponseFormat` → `output_config.format` via the new `toAnthropicOutputConfig` / `toAnthropicOutputFormat` helpers; both OpenAI's nested `{type:"json_schema", json_schema:{schema:…}}` and the flat `{type:"json_schema", schema:…}` are recognized, the schema is passed through unvalidated, and a format with no extractable schema (e.g. `{type:"json_object"}`) produces no `format` rather than a fabricated one. `output_config` is omitted when both halves are empty. **Behavior change**: `anthropicRequest.Effort` is now marked deprecated and no longer assigned, so the wire carries `output_config.effort` instead of the old top-level `effort` — the field itself is retained for internal source compatibility.
  - **Usage** — `anthropicUsage` gained `OutputTokensDetails *anthropicOutputTokensDetails` (`{thinking_tokens}`), `ServerToolUse *anthropicServerToolUse` (`{web_search_requests, web_fetch_requests}`), `InferenceGeo string` and `ServiceTier string`. The canonical `Usage` (`schema.go`) gained `ServerToolUse *ServerToolUse` (zero counts `omitempty`), `InferenceGeo string`, `ServiceTier string`; `Usage.ReasoningTokens` now also sources Anthropic's `output_tokens_details.thinking_tokens` (an explicit top-level `reasoning_tokens` still wins). All three round-trip through `usageJSON` / `UnmarshalJSON`; `Usage.Add` accumulates the server-tool counts but leaves geo/tier alone (they describe a single request). `anthropicCanonicalUsage` maps them for both protocols' paths. The streaming path no longer overwrites only `output_tokens` on the terminal `message_delta`: the new `mergeAnthropicUsage` folds in exactly the fields that event carries, so the input/cache/geo/tier/server-tool information from `message_start` survives.
  - **`container` / `inference_geo`** — `ChatRequest` gained `Container string` and `InferenceGeo string` (both `omitempty`), passed straight through to the matching new `anthropicRequest` fields. `anthropicResponse` gained `Container *ResponseContainer` (`json:"container"`), deserializing straight into the new canonical `ResponseContainer{ID, ExpiresAt string}` — `ExpiresAt` stays the server-supplied string, with no expiry parsing, auto-renewal, or retry. `ChatResponse.Container` carries it non-streaming; `StreamChunk.Container` (new) carries it in streaming, emitted **once** as soon as `message_start` is parsed so that tool-only or immediately-ending streams still surface the ID for the next turn.
  - **Unknown content blocks** — `Message` gained `ExtraBlocks []json.RawMessage` (`json:"-"`, runtime-only). `anthropicResponse.Content` changed from `[]anthropicContentBlock` to the new `[]anthropicResponseBlock`, which decodes the known fields *and* retains each block's verbatim JSON (re-marshalling a decoded block would lose unmodelled fields). `fromAnthropicResponse` gained a `default` branch appending unrecognized blocks to `ExtraBlocks`; a `text` block carrying `citations` still contributes its text and additionally keeps its whole original block. `anthropic_stream.go` does the same per event: an unrecognized `content_block_start` emits its raw `content_block` and records the index, so every later delta on that index — plus any unknown delta type on a known block — emits its raw `delta`, in arrival order. **Fixes a decode failure**: `anthropicContentBlock.ResultContent` is a `string` tagged `content`, but response-side `content` is polymorphic (server-tool results carry an array, code-execution results an object), so a response containing one previously failed to decode entirely; `anthropicResponseBlock` shadows it with a `json.RawMessage`. `Message.AppendDelta` appends `ExtraBlocks` without parsing or merging. `signature_delta` stays ignored and `text`/`thinking`/`tool_use`/`input_json_delta` are unchanged.
  - **Tool fields** — `Tool` (`schema.go`) and `anthropicTool` both gained `Strict *bool`, `DeferLoading *bool`, `AllowedCallers []string`, `EagerInputStreaming *bool`, `InputExamples []any` (all `omitempty`), copied verbatim in the tool-conversion loop; `anthropicTool` also gained `Type string` (`omitempty`). `ChatRequest.clone()` now duplicates each tool's `AllowedCallers` / `InputExamples` slices (elements stay shallow, matching the existing `any` contract). **Deviation from the spec**: the spec had `Tool.Type` map through unconditionally, but the canonical `Type` is OpenAI's `"function"` for every ordinary tool, and sending that to Anthropic is invalid. `"function"` and empty are therefore treated as Anthropic's default custom tool and omitted; any other value (versioned built-ins like `web_search_20260209`) passes through unenumerated and unvalidated.
  - **`anthropic-user-profile-id`** — `Client` gained `anthropicUserProfileID string` plus `WithAnthropicUserProfileID(string)` (empty is ignored, matching `WithAnthropicVersion`); `setAnthropicHeaders` (`anthropic_chat.go`) emits the header only when non-empty.
  - **Default behavior unchanged**: with none of the new fields set, the request wire JSON is identical to before except that `ReasoningEffort` now lands in `output_config.effort` instead of the top-level `effort`; the new `json:"-"` members never reach the OpenAI body; responses without the new members leave every added field zero/nil and omitted from JSON; no new header is sent by default.
  - Added tests: `anthropic_output_config_test.go` (`TestToAnthropicRequest_OutputConfig` table over neither/format-only/effort-only/both/`json_object`/flat-schema, `_ContainerAndInferenceGeo`, `_ToolExtensions`, `_ToolExtensionsOmitted`, `TestChatRequestClone_ToolSlices`), `anthropic_usage_test.go` (`TestFromAnthropicResponse_UsageExtensions` / `_UsageExtensionsAbsent` / `_Container`, `TestUsage_ServerToolUseJSONRoundTrip`, `TestUsage_ReasoningTokensPrecedence`, `TestUsageAdd_ServerToolUse`, `TestMergeAnthropicUsage`), `anthropic_extra_blocks_test.go` (`TestFromAnthropicResponse_ExtraBlocks` / `_TextBlockCitations` / `_NoExtraBlocks`, `TestAnthropicStream_ExtraBlocks` / `_KnownBlocksNoExtra` / `_ContainerAndUsageMerge` / `_NoContainer`), and `TestSetAnthropicHeadersUserProfileID` / `_OverWire` (`anthropic_chat_test.go`). `TestToAnthropicRequestEffort` and `TestToAnthropicRequestThinkingAdaptive` were updated to assert the new `output_config.effort` location and that the deprecated top-level `effort` is not emitted.
- **Not implemented (deliberate)**: Anthropic's versioned built-in tool types are not enumerated (type names stay opaque `string`s); `citations` are not parsed into canonical fields (the raw block is preserved instead); the `tool_reference` / `tool_search` meta-protocol is deferred; the deprecated top-level `effort` field is retained (unassigned) rather than removed this release.

---

## 2026-07-10 — Anthropic: merge consecutive parallel `tool_result` into one `user` message

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Anthropic Messages API requires all `tool_result` blocks for one assistant turn's parallel `tool_use` to arrive inside a **single** `role:"user"` message immediately after the assistant turn. Emitting one `user` message per result (the naive 1:1 mapping of canonical `role:"tool"` messages) makes the endpoint reject the extra results (`without tool_result blocks immediately after: call_xxx`), failing any parallel-tool-use request (notably on DeepSeek's Anthropic-compatible endpoint, which strictly validates this).
- **Change summary**: `toAnthropicRequest` (`anthropic.go`) now detects a run of consecutive canonical `RoleTool` messages and serializes them once into a single `role:"user"` `anthropicMessage` whose content array holds all `tool_result` blocks in original order; the loop index skips the consumed run. The per-block builder is extracted into a `toolResultBlock` helper shared by the run-merge path (`toAnthropicToolResultMessage`) and the single-message fallback, so `CacheBreakpoint` still attaches `cache_control` to exactly the flagged block and the wire shape for a lone tool result is unchanged. A non-consecutive `RoleTool` (separated by user/assistant/system) keeps its own user message; mid-conversation system and assistant `tool_use` serialization are untouched. The non-streaming `anthropicChatCompletion` and streaming `anthropicChatCompletionStream` paths share `toAnthropicRequest`, so both benefit with no separate change. **Default behavior unchanged**: a single tool result still emits one `role:"user"` with a one-element `tool_result` array. Added tests: `TestToAnthropicRequestConsecutiveToolResults` (merge + order + ids), `TestToAnthropicRequestConsecutiveToolResultsCache` (`CacheBreakpoint` survives merge, unflagged blocks stay clean), `TestToAnthropicRequestNonConsecutiveToolResults` (separated runs + inline system not regressed), `TestToAnthropicRequestParallelToolUseRound` (assistant multi-`tool_use` immediately followed by one merged `user`).

---

## 2026-06-02 — Anthropic: top-level automatic caching + usage `cache_creation` TTL breakdown

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Since 2026-02-19, **automatic caching** lets a request carry a single request-root `cache_control` (`{type:"ephemeral"}`, optional `ttl:"1h"`) instead of per-block markers: the server caches the last cacheable block and advances the breakpoint forward as the conversation grows. The response `usage` additionally returns a `cache_creation` object breaking cache writes down by TTL — `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens` (their sum equals `cache_creation_input_tokens`). The wrapper previously supported only block-level `cache_control` (via `Message.CacheBreakpoint` / `Tool.CacheBreakpoint`) and exposed no cache-write counts at all.
- **Change summary**: `ChatRequest` (`schema.go`) gained `AutoCache bool` and `AutoCacheTTL string` (both `json:"-"`, struct-local — never serialised on the canonical OpenAI-shape body, mirroring `CacheBreakpoint`). When `AutoCache` is true, `toAnthropicRequest` (`anthropic.go`) sets the new top-level `anthropicRequest.CacheControl` (`json:"cache_control,omitempty"`) to `{type:"ephemeral", ttl:AutoCacheTTL}` (empty `ttl` → omitted → default 5m; `"1h"` → 1-hour cache); `anthropicCacheControl` gained `TTL string` (`json:"ttl,omitempty"`). This coexists with the independent per-block `CacheBreakpoint` markers. For usage: `anthropicUsage` gained `CacheCreation *anthropicCacheCreation` (`json:"cache_creation,omitempty"`, `{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}`); the canonical `Usage` (`schema.go`) gained `CacheWriteTokens` (← `cache_creation_input_tokens`, total — previously unexposed), `CacheWrite5mTokens`, `CacheWrite1hTokens` (all `json:",omitempty"`, paired with `CacheReadTokens`), accumulated by `Usage.Add` and round-tripped through `usageJSON` / `UnmarshalJSON`. Both the non-streaming path (`fromAnthropicResponse`) and the streaming path (`anthropic_stream.go`, refactored to carry the message_start usage forward) populate them via a shared `anthropicCanonicalUsage` helper. Cache-creation/read tokens still fold into `PromptTokens` as before (the new fields are additive observability, subsets of `PromptTokens`, like `CacheReadTokens`). **Default behavior unchanged**: with `AutoCache` false no request-root `cache_control` is emitted, the switch never leaks into the OpenAI body, and OpenAI responses (no `cache_creation`) leave the new fields at 0/omitted. Added tests: `TestToAnthropicRequest_AutoCacheDefault` / `_AutoCache1h` / `_AutoCacheOff` (coexistence with block-level) / `TestChatRequest_OpenAIShape_NoAutoCacheLeak`, `TestFromAnthropicResponse_CacheCreationBreakdown` / `_NoCacheCreation`, `TestAnthropicStreamCacheCreationBreakdown`, `TestUsageAddWithCacheWriteTokens` / `TestUsageCacheWriteTokensJSONRoundTrip` / `TestUsageCacheWriteTokensOmittedWhenZero`.

---

## 2026-06-02 — Anthropic: pass through `top_k` sampling parameter

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Anthropic's Messages API natively supports `top_k` (top-k truncation sampling — restrict sampling to the K most-likely tokens at each step). The canonical `ChatRequest` had no corresponding field, and the Anthropic wrapper never emitted it.
- **Change summary**: `ChatRequest` (`schema.go`) gained `TopK *int` (`json:"top_k,omitempty"`), placed next to `TopP`. `anthropicRequest` (`anthropic.go`) gained the matching `TopK *int` (`json:"top_k,omitempty"`) and `toAnthropicRequest` maps `req.TopK` straight through. OpenAI's Chat Completions has no `top_k`: the canonical request is the OpenAI shape itself, so the `omitempty` field is simply omitted when unset and passed through verbatim when set (OpenAI-compatible backends that accept it honour it; the rest ignore the unknown field). **Default behavior unchanged**: unset `TopK` is absent from both wire formats. Added `TestToAnthropicRequestTopK` (set → mapped + serialized `top_k`; unset → `nil` + omitted).

---

## 2026-06-02 — Anthropic: configurable `anthropic-beta` / `anthropic-version` headers

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Many Anthropic capabilities (compaction, context-editing, structured-outputs (during beta), fast-mode, advisor, etc.) are opt-in via the `anthropic-beta` request header (multiple values comma-joined); the `anthropic-version` header (default `2023-06-01`) selects the API version.
- **Change summary**: Infrastructure only — adds Client-level configurability without wiring any specific beta capability. `Client` (`client.go`) gained `anthropicBeta []string` and `anthropicVersion string`, plus two options following the existing pattern: `WithAnthropicBeta(values ...string)` (appends across calls, ignores empty strings, comma-joined on the wire) and `WithAnthropicVersion(string)` (empty string keeps the default). `setAnthropicHeaders` (`anthropic_chat.go`) now emits `anthropic-version` from `anthropicVersionHeader()` (configured value or `anthropicAPIVersion`) and sets `anthropic-beta` only when `anthropicBetaHeader()` is non-empty. **Default behavior unchanged**: with no option, version stays `2023-06-01` and no `anthropic-beta` header is sent. Added `TestSetAnthropicHeaders` (default / single beta / multiple beta in one call / accumulation across calls / custom version / version+beta).

---

## 2026-06-02 — Anthropic: new `stop_reason` constants and `stop_details` (refusal classification)

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: `stop_reason` gained `model_context_window_exceeded` (2025-09-29; input + output exceeded the model's context window, distinct from hitting the requested `max_tokens`), `pause_turn` (a long-running / server-tool turn was paused and may be replayed), and `refusal` (2026-05-28, Opus 4.8; streaming classifiers intervened on a potential policy violation). When `stop_reason` is `refusal`, the response and the terminal streaming `message_delta` carry `stop_details` (`{type, category, explanation}`) with the refusal classification.
- **Change summary**: `mapAnthropicStopReason` (`anthropic.go`) previously only mapped `end_turn`/`stop_sequence`/`max_tokens`/`tool_use` and passed everything else through as the raw string. It now maps the three new reasons to dedicated named constants — `FinishReasonModelContextWindowExceeded` / `FinishReasonRefusal` / `FinishReasonPauseTurn` (`schema.go`), whose values are the verbatim Anthropic strings (purely additive: these previously already flowed through as raw strings, so no behavior change — they are NOT folded into `content_filter`/`length`, preserving Anthropic semantics). Added a canonical `StopDetails struct {Type, Category, Explanation string}` (all `omitempty`); `Choice` and `StreamChunkChoice` (`schema.go`) gained `StopDetails *StopDetails` (`json:"stop_details,omitempty"`). `anthropicResponse` (`stop_details`) and `anthropicMessageDeltaData` (`stop_details,omitempty`) deserialize straight into the canonical `*StopDetails` (identical field shape); `fromAnthropicResponse` and the stream `message_delta` branch (`anthropic_stream.go`) propagate it. Added tests: `TestMapAnthropicStopReason` extended with the three new reasons, `TestFromAnthropicResponseRefusalStopDetails`, `TestFromAnthropicResponseNoStopDetails`, `TestAnthropicStreamRefusalStopDetails`.

---

## 2026-06-02 — Anthropic: map `effort`, support `thinking.display`, deprecate `budget_tokens`

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Since 2026-02-05, the top-level `effort` parameter GA'd and supersedes `thinking.budget_tokens` for controlling reasoning depth (also enabling `thinking.type:"adaptive"`, where the model sizes its own thinking). Since 2026-03-16, `thinking.display:"omitted"` suppresses thinking content to speed up streaming.
- **Change summary**: `toAnthropicRequest` (`anthropic.go`) now maps the canonical `ChatRequest.ReasoningEffort` to a new top-level `anthropicRequest.Effort string` (`json:"effort,omitempty"`) — empty stays omitted; the same field still drives OpenAI's `reasoning_effort`. `Thinking` (`schema.go`) gained `Display string` (`json:"display,omitempty"`) for `"omitted"`; its `Type` stays a plain string so `"adaptive"` passes through; `BudgetTokens` got a deprecation note pointing callers at `effort`/`adaptive`. Added tests `TestToAnthropicRequestEffort`, `TestToAnthropicRequestEffortOmittedWhenEmpty`, `TestToAnthropicRequestThinkingDisplayOmitted`, `TestToAnthropicRequestThinkingAdaptive`.

---

## 2026-06-02 — Anthropic: `tool_choice` "none" mapping and `disable_parallel_tool_use`

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Since 2024-10-03, `tool_choice` accepts `disable_parallel_tool_use` (cap the model to at most one tool call per turn) alongside `auto`/`any`/`tool`. Since 2025-02-27, `tool_choice:{type:"none"}` explicitly forbids any tool call (distinct from omitting the field, which lets the model choose).
- **Change summary**: `convertToolChoice` (`anthropic.go`) previously returned `nil` for `"none"`, dropping the `tool_choice` field entirely (model-chooses semantics) instead of emitting `{type:"none"}` (forbid-all). Now `"none"` maps to `{type:"none"}`. The `anthropicToolChoice` struct gained `DisableParallelToolUse *bool` (`disable_parallel_tool_use,omitempty`). The tool-choice assembly now folds in the canonical `ChatRequest.ParallelToolCalls`: when explicitly `false`, it sets `disable_parallel_tool_use:true` on the resulting choice — defaulting to `{type:"auto"}` to carry the flag when no choice is named but tools are present, and never attaching it to `{type:"none"}` (where it is meaningless). `ParallelToolCalls` unset/`true` leaves the choice untouched. Added `TestToAnthropicRequestParallelToolCalls` (auto/any/tool/none × parallel false/true/unset) and updated the `"none"` case of `TestToAnthropicRequestToolChoice`.

---

## 2026-06-02 — Anthropic: preserve mid-conversation `system` messages (fix hoisting)

- **Official protocol**: Anthropic Messages API (`/v1/messages`)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Official change**: Since 2026-05-28 (Opus 4.8), `messages` may contain `role:"system"` entries at non-leading positions (mid-conversation system messages), letting callers change instructions mid-session while keeping prompt-cache hits.
- **Change summary**: `toAnthropicRequest` (`anthropic.go`) previously hoisted **every** `RoleSystem` message into the top-level `system` field regardless of position, wrongly lifting mid-conversation system messages to the front and dropping their position semantics. Now only the **leading** run of system messages (before the first non-system message) is extracted into `system` (CacheBreakpoint / block-vs-string behavior unchanged); a system message appearing mid-conversation falls through to `toAnthropicMessage` and stays inline as a `role:"system"` message in its original order. Added tests `TestToAnthropicRequestSystemMidConversation` and `TestToAnthropicRequestSystemLeadingAndMid` (the leading-only case stays covered by `TestToAnthropicRequestSystemExtraction`).

---

## 2026-06-02 — OpenAI: multimodal input/output (`input_audio`/`file` parts, `modalities`/`audio`)

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Official change**: Chat Completions supports audio/file multimodal input via content parts `input_audio` and `file`, requests audio output via `modalities` + `audio` (voice/format), and returns generated audio on `choices[].message.audio`.
- **Change summary**: `ContentPart` extended with `InputAudio *InputAudio` (`input_audio`, `{data, format}`) and `File *FilePart` (`file`, `{file_id | filename + file_data}`), both `omitempty`; the string/parts polymorphism stays on `Content` (`MarshalJSON`/`UnmarshalJSON` unchanged). `ChatRequest` gained `Modalities []string` (`modalities`) and `Audio *AudioConfig` (`audio`, `{voice, format}`), both `omitempty`; `clone()` now deep-copies the `Modalities` slice. `Message` gained `Audio *MessageAudio` (`audio`, `{id, data, transcript, expires_at}`, all `omitempty`) parsing assistant-generated audio. Anthropic translation untouched (no counterparts; new part types fall through its `text`/`image_url` switch safely).

---

## 2026-06-02 — OpenAI: extend `ChatRequest` with common request fields (+ response `logprobs`)

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Official change**: Chat Completions exposes the request parameters `logprobs` / `top_logprobs`, `logit_bias`, `parallel_tool_calls`, `service_tier`, `store`, `metadata`, `prompt_cache_key`, and returns `choices[].logprobs` when `logprobs` is set.
- **Change summary**: Added eight `omitempty` `ChatRequest` fields — `Logprobs *bool` (`logprobs`), `TopLogprobs *int` (`top_logprobs`), `LogitBias map[string]int` (`logit_bias`), `ParallelToolCalls *bool` (`parallel_tool_calls`), `ServiceTier string` (`service_tier`), `Store *bool` (`store`), `Metadata map[string]string` (`metadata`), `PromptCacheKey string` (`prompt_cache_key`). `clone()` now deep-copies the `LogitBias` / `Metadata` maps (via `maps.Clone`) so a copy's mutations never affect the original. Added response parsing: `Choice.LogProbs *LogProbs` with `LogProbs{Content, Refusal []TokenLogprob}`, `TokenLogprob{Token, Logprob, Bytes, TopLogprobs}`, `TopLogprob{Token, Logprob, Bytes}`.

---

## 2026-06-02 — OpenAI: sync `reasoning_effort` values, add `verbosity`

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Official change**: `reasoning_effort` accepted values extended to `none / minimal / low / medium / high / xhigh` (GPT-5.1 defaults to `none`); new `verbosity` parameter (`low / medium / high`) controls how detailed the output is.
- **Change summary**: Added `ReasoningEffort*` constants (`none/minimal/low/medium/high/xhigh`) and `Verbosity*` constants (`low/medium/high`); the fields stay plain `string` for pass-through to non-OpenAI backends. Added `ChatRequest.Verbosity string` (`json:"verbosity,omitempty"`); `clone()` needs no change (scalar). 

---

## 2026-06-02 — OpenAI: response type alignment (`reasoning_tokens`, `finish_reason` constants)

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Official change**: Response `usage` now carries `completion_tokens_details.reasoning_tokens` (internal thinking cost of reasoning models); `finish_reason` officially includes `content_filter` (and the legacy `function_call`).
- **Change summary**: Added `Usage.ReasoningTokens int` (`json:"reasoning_tokens,omitempty"`), parsed from nested `completion_tokens_details.reasoning_tokens` (explicit top-level `reasoning_tokens` takes precedence, mirroring `cached_tokens`); `Usage.Add` now accumulates it. Added `FinishReasonContentFilter` (`content_filter`) and `FinishReasonFunctionCall` (`function_call`, legacy compat) constants.

---

## 2026-06-02 — OpenAI: support `max_completion_tokens`, deprecate `max_tokens`

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Official change**: OpenAI deprecated `max_tokens` on Chat Completions; reasoning models (o-series, GPT-5.x) reject `max_tokens` and require `max_completion_tokens`, whose limit covers both visible output tokens and internal reasoning tokens.
- **Change summary**: Added `ChatRequest.MaxCompletionTokens *int` (`json:"max_completion_tokens,omitempty"`); annotated the retained `MaxTokens` as deprecated / incompatible with reasoning models. The Anthropic translator now emits `max_tokens` preferring `MaxCompletionTokens` over `MaxTokens`, defaulting to 4096.

---

## [Baseline] 2026-06-02

Records the current synced state as the baseline.

### OpenAI (OpenAI-compatible)

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`; no standalone version number, keyed by endpoint)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Change summary**: Wrapped non-streaming `ChatCompletion` and streaming `ChatCompletionStream` (SSE), using the OpenAI-compatible format as the canonical representation.

### Anthropic Messages API

- **Official protocol**: Anthropic Messages API (`/v1/messages`; keyed by the official documentation endpoint)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Change summary**: Wrapped the non-streaming and streaming Messages API, with bidirectional translation to/from the canonical OpenAI-compatible types; system messages are extracted into a separate `system` field; streaming uses Anthropic SSE event types (`content_block_delta`, `message_delta`, etc.).
