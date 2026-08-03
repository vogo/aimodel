# aimodel Documentation

`aimodel` is a set of Go clients for AI model APIs, one per protocol — a zero-dependency **thin API wrapper** that builds requests, manages connections and decodes responses. It carries no retry, rate limiting, validation, caching, or logging/metrics.

This directory holds the design documentation. The root [README.md](../README.md) covers usage.

## Architecture

| Document | Contents |
|---|---|
| [architecture.md](./architecture.md) | **Start here** — design scope, the two independent protocol clients, the neutrality test and the guards behind it, repository layout, maintenance convention |
| [adr.md](./adr.md) | Architecture Decision Record index — accepted decisions and their rationale |

## Protocols

| Document | Contents |
|---|---|
| [openai/openai-chat-api.md](./openai/openai-chat-api.md) | OpenAI Chat Completions: client, wire types, `ExtraBody`, SSE and stream accumulation, usage and prompt caching, errors |
| [openai/openai-response-api.md](./openai/openai-response-api.md) | OpenAI Responses API: dated wire baseline, items, typed SSE events, hosted tools |
| [openai/openai-api-changes.md](./openai/openai-api-changes.md) | OpenAI change log (Chat Completions and Responses) |
| [anthropic/anthropic-message-api.md](./anthropic/anthropic-message-api.md) | Anthropic Messages API: client and headers, content blocks, tools, prompt caching, SSE events, two-part usage merging, errors |
| [anthropic/anthropic-api-changes.md](./anthropic/anthropic-api-changes.md) | Anthropic change log — official changes and how the wrapper followed |

## Tools

| Document | Contents |
|---|---|
| [design/compose.md](./design/compose.md) | Dispatch across several OpenAI-compatible backends: selection strategies, health tracking, recovery probes, cancellation, aggregate errors |

## Root documents

- [../README.md](../README.md) — usage: installation, chat completions, streaming, tools, multimodal input, the Anthropic protocol, the Responses API, multi-backend compose.
- [../MIGRATION.md](../MIGRATION.md) — migrating off the canonical API removed in v0.7.0.
- [../CLAUDE.md](../CLAUDE.md) — build/test commands, repository rules, and a map from code area to the document covering it (for AI assistants).
- [../CHANGES.md](../CHANGES.md) — release index and the merged timeline of both protocols' change logs.

## Official API references

| Protocol | Official docs | Provider package |
|---|---|---|
| OpenAI Chat Completions (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | `provider/openai/` |
| OpenAI Responses | https://platform.openai.com/docs/api-reference/responses | `provider/openai/` |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | `provider/anthropic/` |

## Maintenance convention

When an official API changes, update these in sync:

1. the provider's wire types and client;
2. the relevant document here — the protocol's own page, and [architecture.md](./architecture.md) if a package boundary moved;
3. that protocol's change log (`*-api-changes.md`) plus the [CHANGES.md](../CHANGES.md) index.

When a step does not apply, say so explicitly rather than skipping it silently. The root `README.md` / `CLAUDE.md` change only when the public usage surface or the agent-facing guidance does — they link here rather than restating design.
