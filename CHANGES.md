# CHANGES

This file records the **sync status between aimodel's wrapper code and the official API protocols**, so maintainers and AI can track "which official API a wrapper maps to, and which version it is currently synced with".

The official API documentation entries are listed in the "Official API References" section of [README.md](./README.md) / [CLAUDE.md](./CLAUDE.md).

**Maintenance convention**: when an official API changes, update all three in sync — the wrapper code, the documentation (README.md / CLAUDE.md), and this file — keeping them consistent and continuously up to date. Each entry must include at least: date, target official protocol and version, change summary.

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
