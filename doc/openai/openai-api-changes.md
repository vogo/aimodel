# OpenAI Chat Completions API — Change Log

> Historical record: entries below describe the API surface when each change landed. OpenAI-only audio/file, log-probability, verbosity, storage/metadata, prompt-cache routing, generation-count and request-tier members were removed from the canonical schema in July 2026; they are not current `ais` capabilities. Entries dated before 2026-07-22 also predate the native wire model and the canonical translation layer — where they describe canonical types as the OpenAI wire shape, read that as the design at the time, not as a current invariant ([ADR 0005](../adr/0005-canonical-shared-semantics-over-provider-native-wire.md)).

This file records **how aimodel's OpenAI wrapper tracks the official Chat Completions API**: what changed upstream, and how the wrapper followed.

- **Official protocol**: OpenAI Chat Completions API (`POST /chat/completions`; no standalone version number — keyed by the endpoint)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Implementation notes**: [openai-chat-api.md](./openai-chat-api.md)
- **Index of both protocols**: [../../CHANGES.md](../../CHANGES.md)

**Maintenance convention**: see [../architecture.md](../architecture.md) §6. Every entry carries at least a date, the official change, and a wrapper change summary.

Since 2026-07-22 the OpenAI path has its own native wire model, so an OpenAI-side change lands in `provider/openai/wire.go` first. It reaches `ais/schema.go` only when the "≥ 2 providers" attribution test passes, and then it also needs a mapping in `provider/openai/translate.go` — canonical no longer picks up OpenAI fields for free.

Newest first.

---

## 2026-07-22 — Public native Chat Completions client and explicit canonical translation layer

**Official change**: none — this is a wrapper restructuring, driven by the customization principle (every vendor exposes a public, full-fidelity native API) and by the tightened canonical field-attribution rule that had just removed the OpenAI-only members from `ais`.

**Wrapper change**

- **Native wire model** (`provider/openai/wire.go`): independent exported types for the full Chat Completions surface — `ChatCompletionRequest` / `ChatCompletionResponse` / `ChatCompletionChunk`, `ChatCompletionMessage`, `ChatCompletionContentPart`, `ChatCompletionUsage` with its nested `PromptTokensDetails` / `CompletionTokensDetails`, tools, `StreamOptions`, `Thinking`, and the OpenAI-only members that are not canonical (log probabilities, audio/file, storage/metadata, prompt-cache routing, generation counts, request-side tier).
- **Native client** (`provider/openai/native.go`): `openai.NewClient(apiKey, ...ClientOption)` with `WithBaseURL` / `WithHTTPClient`, `ChatCompletions` and `ChatCompletionsStream`. Native types end to end, canonical translation bypassed; default base URL `https://api.openai.com/v1`.
- **Canonical translation** (`provider/openai/translate.go`): `toOpenAIRequest`, `fromOpenAIResponse`, `fromOpenAIChunk`, `fromOpenAIMessage`, `fromOpenAIUsage`. The unified-client path now goes canonical → native → wire instead of marshalling `ais.ChatRequest` directly, and decodes into `ChatCompletionResponse` / `ChatCompletionChunk` before normalizing back to canonical.
- `provider.NewChatRequest` still adds the wire-only `stream_options.include_usage=true` on streaming requests; it is now set on the native request in `toOpenAIRequest`.
- Usage promotion moved with it: `fromOpenAIUsage` reads the nested native details into `Usage.CacheReadTokens` / `ReasoningTokens`. `ais.Usage.UnmarshalJSON` keeps the same promotion for direct canonical-JSON decoding.
- **Integration examples**: native non-streaming and streaming samples added under `integrations/`.
- **No canonical change**: no `ais` field was added, removed, renamed or remapped, and the serialized request/response bodies are unchanged. Purely additive on the public surface (the native client and types).
- **Architecture consequence**: this retired the "OpenAI-compatible path has no translation layer" invariant from ADR 0002. Recorded on 2026-07-25 in [ADR 0005](../adr/0005-canonical-shared-semantics-over-provider-native-wire.md), which supersedes it; `doc/architecture.md` §2, `openai-chat-api.md` and `CLAUDE.md` were reconciled at the same time. The seam is hand-written on both sides and has **no reflective field-coverage test** — see [openai-chat-api.md](./openai-chat-api.md) §7 for why, and what a canonical field addition must do instead.

## 2026-06-02 — Multimodal input/output (`input_audio` / `file` content parts, `modalities` / `audio`)

**Official change**: Chat Completions supports audio/file input via the `input_audio` and `file` content parts, requests audio output via `modalities` + `audio` (voice/format), and returns the generated audio on `choices[].message.audio`.

**Wrapper change**

- `ContentPart` gained `InputAudio *InputAudio` (`{data, format}`) and `File *FilePart` (`{file_id | filename + file_data}`), both `omitempty`; the string/array polymorphism stays on `Content`'s `MarshalJSON` / `UnmarshalJSON`, unchanged.
- `ChatRequest` gained `Modalities []string` and `Audio *AudioConfig{Voice, Format}`; `clone()` now deep-copies the `Modalities` slice.
- `Message` gained `Audio *MessageAudio` (`{id, data, transcript, expires_at}`), parsing assistant-generated audio.
- Anthropic translation untouched: no counterpart exists, and the new part types fall safely through its `text` / `image_url` switch.

## 2026-06-02 — Extend `ChatRequest` with common request fields (+ response `logprobs`)

**Official change**: Chat Completions exposes the request parameters `logprobs` / `top_logprobs`, `logit_bias`, `parallel_tool_calls`, `service_tier`, `store`, `metadata`, `prompt_cache_key`, and returns `choices[].logprobs` when `logprobs` is set.

**Wrapper change**

- `ChatRequest` gained eight `omitempty` fields: `Logprobs *bool`, `TopLogprobs *int`, `LogitBias map[string]int`, `ParallelToolCalls *bool`, `ServiceTier string`, `Store *bool`, `Metadata map[string]string`, `PromptCacheKey string`.
- `clone()` deep-copies `LogitBias` / `Metadata` via `maps.Clone`, so a copy's mutations never affect the original.
- Response side: `Choice.LogProbs *LogProbs`, with `LogProbs{Content, Refusal []TokenLogprob}`, `TokenLogprob{Token, Logprob, Bytes, TopLogprobs}`, `TopLogprob{Token, Logprob, Bytes}`.

`ParallelToolCalls` was later reused by the Anthropic translation as `disable_parallel_tool_use` — see the [Anthropic change log](../anthropic/anthropic-api-changes.md).

## 2026-06-02 — Sync `reasoning_effort` values, add `verbosity`

**Official change**: `reasoning_effort` accepted values extended to `none` / `minimal` / `low` / `medium` / `high` / `xhigh` (GPT-5.1 defaults to `none`); the new `verbosity` parameter (`low` / `medium` / `high`) controls how detailed the output is.

**Wrapper change**: added the `ReasoningEffort*` and `Verbosity*` constants; both fields stay `string` so they pass through to non-OpenAI backends. Added `ChatRequest.Verbosity string`; `clone()` needed no change (scalar field).

`ReasoningEffort` was later reused by the Anthropic translation — first as the top-level `effort`, now as `output_config.effort`.

## 2026-06-02 — Response type alignment (`reasoning_tokens`, `finish_reason` constants)

**Official change**: response `usage` now carries `completion_tokens_details.reasoning_tokens` (the internal thinking cost of reasoning models); `finish_reason` officially includes `content_filter` (and the legacy `function_call`).

**Wrapper change**: added `Usage.ReasoningTokens int`, parsed from the nested `completion_tokens_details.reasoning_tokens` (**an explicit top-level field wins**, mirroring `cached_tokens`); `Usage.Add` accumulates it. Added the `FinishReasonContentFilter` and `FinishReasonFunctionCall` (legacy compatibility) constants.

## 2026-06-02 — Support `max_completion_tokens`, deprecate `max_tokens`

**Official change**: OpenAI deprecated `max_tokens` on Chat Completions; reasoning models (o-series, GPT-5.x) reject it and require `max_completion_tokens`, whose limit covers both visible output tokens and internal reasoning tokens.

**Wrapper change**: added `ChatRequest.MaxCompletionTokens *int`; the retained `MaxTokens` is annotated as deprecated and incompatible with reasoning models. The Anthropic translator emits `max_tokens` preferring `MaxCompletionTokens` over `MaxTokens`, defaulting to 4096.

## [Baseline] 2026-06-02

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`, no standalone version number — keyed by the endpoint)
- **Summary**: wrapped the non-streaming `ChatCompletion` and streaming `ChatCompletionStream` (SSE), using the OpenAI-compatible format as the canonical representation.
