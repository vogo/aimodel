# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`github.com/vogo/aimodel` — A unified Go SDK for AI model APIs with multi-protocol support (OpenAI-compatible, Anthropic) and composable multi-model dispatching. Zero external dependencies.

This SDK is a **thin API wrapper** — it translates requests, manages connections, and normalizes responses across protocols. 
It intentionally does **not** include retry, rate limiting, request validation, caching / persistence, logging / metrics.

## Official API References

aimodel is a thin wrapper over the following official protocols. Each wrapper maps one-to-one to its official documentation:

| Protocol | Official docs | Wrapper code |
|------|------|------|
| OpenAI (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | `openai_chat.go` / `openai_stream.go` |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | `anthropic.go` / `anthropic_chat.go` / `anthropic_stream.go` |

The sync status against the official APIs (target version, change summary) is recorded in [CHANGES.md](./CHANGES.md).

**Maintenance convention**: when an official API changes, update all three in sync — the wrapper code, this document (README.md / CLAUDE.md), and CHANGES.md — keeping them consistent and continuously up to date.

## Rules

- delete build binary after test
- current file only contains core api/model dispatching logic, and core rules, not add any other logic.

## Build & Test Commands

```bash
make build          # Full pipeline: license-check → format → lint → test
make test           # Run all tests with coverage (outputs coverage.out)
make lint           # golangci-lint
make format         # goimports + go fmt + gofumpt
make license-check  # Apache license header check

# Run a single test
go test -run TestFunctionName ./...
go test -run TestFunctionName ./composes/

# Coverage report
go tool cover -func=coverage.out
```

## Architecture

### Protocol-Based Dispatch

`Client` implements `ChatCompleter` and routes requests based on `Protocol`:
- **ProtocolOpenAI** (default): OpenAI-compatible `/chat/completions` endpoint
- **ProtocolAnthropic**: Anthropic Messages API with bidirectional format translation

The dispatch happens in `chat.go`. Protocol-specific logic is isolated in `openai_*.go` and `anthropic_*.go` files.

### Key Types (schema.go)

- `ChatRequest` / `ChatResponse` — Use OpenAI-compatible format as canonical representation. `ReasoningEffort` (`reasoning_effort`) and `Verbosity` (`verbosity`) stay plain `string` for pass-through; use the `ReasoningEffort*` (`none/minimal/low/medium/high/xhigh`) and `Verbosity*` (`low/medium/high`) constants. Common request fields (all `omitempty`): `TopK *int` (top-k truncation sampling; maps to Anthropic's `top_k`, omitted/passed-through on OpenAI which has no native `top_k`) / `Logprobs *bool` / `TopLogprobs *int` / `LogitBias map[string]int` / `ParallelToolCalls *bool` / `ServiceTier string` / `Store *bool` / `Metadata map[string]string` / `PromptCacheKey string`. `clone()` deep-copies the `LogitBias` / `Metadata` maps so a copy's mutations never leak into the original.
- `Stream` — Concurrent-safe SSE reader with `Recv()` / `Close()` (mutex + atomic bool)
- `Usage` — Normalizes token counts; `CacheReadTokens` parses OpenAI's nested `prompt_tokens_details.cached_tokens` and `ReasoningTokens` parses `completion_tokens_details.reasoning_tokens` (explicit top-level fields take precedence). `CacheWriteTokens` reports prompt-cache writes (Anthropic `cache_creation_input_tokens`, total); `CacheWrite5mTokens` / `CacheWrite1hTokens` break it down by TTL (Anthropic `usage.cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}`, summing to `CacheWriteTokens`). Cache read/write tokens are subsets of `PromptTokens` surfaced separately for observability. `Add` accumulates all counts.
- `FinishReason` — Mirrors OpenAI's `finish_reason`: `stop` / `length` / `tool_calls` / `content_filter` / legacy `function_call`. Anthropic-only stop reasons with no OpenAI canonical pass through verbatim as named constants: `model_context_window_exceeded` (distinct from `length`), `refusal`, `pause_turn`.
- `StopDetails` — Anthropic's structured stop classification (`Type` / `Category` / `Explanation`, all `omitempty`), returned alongside `stop_reason:"refusal"` and surfaced on `Choice.StopDetails` (non-streaming) and `StreamChunkChoice.StopDetails` (terminal `message_delta`); `nil` when absent.

### Multi-Model Composition (composes/)

`ComposeClient` wraps multiple `ChatCompleter` instances with selection strategies:
- **Failover** — Sequential with health tracking
- **Random** — Shuffle active models per request
- **Weighted** — Proportional selection by weight

Health tracking uses exponential backoff (`2^min(errorCount-1, 6)` × base interval) with recovery probes.

### Client Configuration

Option pattern: `WithAPIKey()`, `WithBaseURL()`, `WithProtocol()`, `WithTimeout()`, `WithDefaultModel()`, `WithHTTPClient()`.

Anthropic-only header options (no effect on OpenAI protocol): `WithAnthropicBeta(...string)` enables beta features via the `anthropic-beta` header — values append across calls, empty strings are ignored, and they are comma-joined on the wire (header omitted when none set); `WithAnthropicVersion(string)` overrides the `anthropic-version` header (empty keeps the default `2023-06-01`). These are infrastructure for opting into beta capabilities (compaction, context-editing, structured-outputs, fast-mode, advisor, etc.); `setAnthropicHeaders` (`anthropic_chat.go`) emits them.

Environment variable fallback order:
- API Key: `AI_API_KEY` > `OPENAI_API_KEY` > `ANTHROPIC_API_KEY`
- Base URL: `AI_BASE_URL` > `OPENAI_BASE_URL` > `ANTHROPIC_BASE_URL`

### Anthropic Translation

API reference: see the "Official API References" section above.

Anthropic types are private (`anthropicRequest`, `anthropicResponse`, etc.) and translated to/from the canonical OpenAI-compatible types. Only the **leading** run of system messages (those before the first non-system message) is extracted into the separate top-level `system` field; a system message appearing mid-conversation is kept inline as a `role:"system"` message in its original position (mid-conversation system messages, supported since Opus 4.8) to preserve position semantics and prompt-cache hits. A run of consecutive canonical `RoleTool` messages (one assistant turn's parallel tool results) is merged into a single `role:"user"` `anthropicMessage` whose content array holds all `tool_result` blocks in original order — Anthropic requires the parallel results of one turn to share one user message (otherwise the endpoint rejects the extras as `without tool_result blocks immediately after`); a lone or non-consecutive tool result keeps its own one-element `user` message. The merge is driven purely by adjacency (no `tool_use` id pairing/sorting); the shared `toolResultBlock` helper carries `CacheBreakpoint` per block so cache breakpoints survive the merge. `tool_choice` translation: `"auto"`→`{type:"auto"}`, `"required"`→`{type:"any"}`, `"none"`→`{type:"none"}` (forbid all calls — distinct from omitting the field), specific function→`{type:"tool",name}`; `ParallelToolCalls=false` sets `disable_parallel_tool_use:true` on the choice (defaulting to `auto` when none named and tools present, never on `none`). `ReasoningEffort` maps to the top-level `effort` field (GA 2026-02-05, supersedes `thinking.budget_tokens`; empty stays omitted). `Thinking` carries `Type` (`enabled`/`disabled`/`adaptive`), the deprecated `BudgetTokens`, and `Display` (`"omitted"` suppresses thinking content for faster streaming, since 2026-03-16). Streaming uses Anthropic-specific SSE event types (`content_block_delta`, `message_delta`, etc.). `mapAnthropicStopReason` maps `end_turn`/`stop_sequence`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`, and surfaces the Anthropic-only `model_context_window_exceeded`/`refusal`/`pause_turn` as their verbatim named constants (anything else still passes through). The response and the terminal `message_delta` parse `stop_details` straight into the canonical `StopDetails` (same field shape), exposing the refusal classification on `Choice.StopDetails` / `StreamChunkChoice.StopDetails`. Prompt caching has two coexisting modes: per-block markers (`Message.CacheBreakpoint` / `Tool.CacheBreakpoint`, both `json:"-"`, attach `cache_control` to the last block / flagged tool) and **automatic caching** (`ChatRequest.AutoCache bool` + `AutoCacheTTL string`, both `json:"-"`) which emits a single request-root `cache_control` (`{type:"ephemeral", ttl:…}`; empty TTL → default 5m, `"1h"` → 1-hour cache) and lets the server advance the breakpoint as the conversation grows. The non-streaming `fromAnthropicResponse` and the streaming usage chunk share `anthropicCanonicalUsage`, which maps `cache_read_input_tokens` / `cache_creation_input_tokens` and the per-TTL `cache_creation` breakdown onto the canonical `Usage` cache fields.

## Packages

- Root package `aimodel` — Client, schemas, protocol implementations, streaming
- `composes/` — Multi-model dispatch strategies and health tracking
- `examples/` — Usage examples for openai, anthropic, and compose patterns
