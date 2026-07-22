# aimodel Documentation

`aimodel` is a Go SDK for multi-protocol (OpenAI-compatible, Anthropic) AI model APIs — a zero-dependency **thin API wrapper** that only translates requests, manages connections, and normalizes responses. It carries no retry, rate limiting, validation, caching, or logging/metrics.

This directory holds the design documentation. The root [README.md](../README.md) covers usage.

## Architecture

| Document | Contents |
|---|---|
| [architecture.md](./architecture.md) | **Start here** — design scope, the OpenAI-shaped canonical representation, client construction and protocol dispatch, repository layout, maintenance convention |
| [adr.md](./adr.md) | Architecture Decision Record index — accepted decisions and their rationale |

## Design topics (cross-protocol)

| Document | Contents |
|---|---|
| [design/data-model.md](./design/data-model.md) | Canonical `ChatRequest` / `Message` / `Content` / `ChatResponse` / `Usage`, field by field |
| [design/streaming.md](./design/streaming.md) | The `Stream` abstraction, delta merging, `ExtraBlocks`, stream interception |
| [design/tool-use.md](./design/tool-use.md) | Tool definitions and their Anthropic extensions, `tool_choice`, parallel tool results |
| [design/prompt-caching.md](./design/prompt-caching.md) | Per-block breakpoints, automatic caching, cache accounting |
| [design/errors.md](./design/errors.md) | Sentinel errors, `APIError`, `ModelError`, `MultiError` |
| [design/compose.md](./design/compose.md) | Selection strategies, health tracking, recovery probes, cancellation |

## Protocols

| Document | Contents |
|---|---|
| [anthropic/anthropic-message-api.md](./anthropic/anthropic-message-api.md) | Anthropic Messages API: bidirectional translation, headers, SSE events |
| [anthropic/anthropic-api-changes.md](./anthropic/anthropic-api-changes.md) | Anthropic change log — official changes and how the wrapper followed |
| [openai/openai-chat-api.md](./openai/openai-chat-api.md) | OpenAI Chat Completions: the zero-translation path, field alignment, SSE |
| [openai/openai-api-changes.md](./openai/openai-api-changes.md) | OpenAI change log |

## Root documents

- [../README.md](../README.md) — usage: installation, chat completion, reasoning effort, multimodal, streaming, the Anthropic protocol, client options, multi-model compose.
- [../CLAUDE.md](../CLAUDE.md) — build/test commands, repository rules, and a map from code area to the document covering it (for AI assistants).
- [../CHANGES.md](../CHANGES.md) — index and merged timeline of both protocols' change logs.

## Official API references

| Protocol | Official docs | Provider package |
|---|---|---|
| OpenAI (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | `provider/openai/` |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | `provider/anthropic/` |

## Maintenance convention

When an official API changes, update these in sync:

1. the wrapper code;
2. the relevant document here — a `design/` topic and/or the protocol's `*-chat-api.md`;
3. that protocol's change log (`*-api-changes.md`);
4. the root `README.md` / `CLAUDE.md` **only if** the public usage surface or the agent-facing guidance changed — they link here rather than restating design.
