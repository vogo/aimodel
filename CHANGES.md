# CHANGES

This file records the **sync status between aimodel's wrapper code and the official API protocols**, so maintainers and AI can track "which official API a wrapper maps to, and which version it is currently synced with".

The official API documentation entries are listed in the "Official API References" section of [README.md](./README.md) / [CLAUDE.md](./CLAUDE.md).

**Maintenance convention**: when an official API changes, update all three in sync — the wrapper code, the documentation (README.md / CLAUDE.md), and this file — keeping them consistent and continuously up to date. Each entry must include at least: date, target official protocol and version, change summary.

---

## 2026-06-02 — OpenAI: sync `reasoning_effort` values, add `verbosity`

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Official change**: `reasoning_effort` accepted values extended to `none / minimal / low / medium / high / xhigh` (GPT-5.1 defaults to `none`); new `verbosity` parameter (`low / medium / high`) controls how detailed the output is.
- **Change summary**: Added `ReasoningEffort*` constants (`none/minimal/low/medium/high/xhigh`) and `Verbosity*` constants (`low/medium/high`); the fields stay plain `string` for pass-through to non-OpenAI backends. Added `ChatRequest.Verbosity string` (`json:"verbosity,omitempty"`); `clone()` needs no change (scalar). 
- **Affected files**: `schema.go` (constants, `ChatRequest.Verbosity`), `openai_chat_test.go` (request-body serialization coverage), `README.md`, `CLAUDE.md`.

---

## 2026-06-02 — OpenAI: response type alignment (`reasoning_tokens`, `finish_reason` constants)

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`)
- **Official docs**: https://platform.openai.com/docs/api-reference/chat
- **Official change**: Response `usage` now carries `completion_tokens_details.reasoning_tokens` (internal thinking cost of reasoning models); `finish_reason` officially includes `content_filter` (and the legacy `function_call`).
- **Change summary**: Added `Usage.ReasoningTokens int` (`json:"reasoning_tokens,omitempty"`), parsed from nested `completion_tokens_details.reasoning_tokens` (explicit top-level `reasoning_tokens` takes precedence, mirroring `cached_tokens`); `Usage.Add` now accumulates it. Added `FinishReasonContentFilter` (`content_filter`) and `FinishReasonFunctionCall` (`function_call`, legacy compat) constants.
- **Affected files**: `schema.go` (types, `usageJSON`, `UnmarshalJSON`, `Add`, `FinishReason` constants), `schema_test.go` (coverage), `README.md`, `CLAUDE.md`.

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
- **Affected files**: `openai_chat.go`, `openai_stream.go`, `chat.go` (protocol dispatch), `schema.go` (canonical types)

### Anthropic Messages API

- **Official protocol**: Anthropic Messages API (`/v1/messages`; keyed by the official documentation endpoint)
- **Official docs**: https://platform.claude.com/docs/en/api/messages
- **Change summary**: Wrapped the non-streaming and streaming Messages API, with bidirectional translation to/from the canonical OpenAI-compatible types; system messages are extracted into a separate `system` field; streaming uses Anthropic SSE event types (`content_block_delta`, `message_delta`, etc.).
- **Affected files**: `anthropic.go`, `anthropic_chat.go`, `anthropic_stream.go`, `chat.go` (protocol dispatch), `schema.go` (canonical types)
