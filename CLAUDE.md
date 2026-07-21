# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`github.com/vogo/aimodel` — A unified Go SDK for AI model APIs with multi-protocol support (OpenAI-compatible, Anthropic) and composable multi-model dispatching. Zero external dependencies.

This SDK is a **thin API wrapper** — it translates requests, manages connections, and normalizes responses across protocols.
It intentionally does **not** include retry, rate limiting, request validation, caching / persistence, logging / metrics.

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

## Where the design lives

**Read the relevant `doc/` page before changing behavior in these areas** — this file deliberately does not restate the design.

| If you are touching… | Read | Code |
|---|---|---|
| Anything cross-cutting: canonical representation, protocol dispatch, client options, env fallback | [doc/api.md](./doc/api.md) | `client.go`, `chat.go`, `chat_client.go` |
| Request/response types, `Usage`, `Content`, `clone()` | [doc/design/data-model.md](./doc/design/data-model.md) | `schema.go` |
| `Stream`, SSE, `AppendDelta`, `ExtraBlocks`, interception | [doc/design/streaming.md](./doc/design/streaming.md) | `stream.go`, `intercept.go`, `*_stream.go` |
| Tool definitions, `tool_choice`, parallel tool results | [doc/design/tool-use.md](./doc/design/tool-use.md) | `schema.go`, `anthropic.go` |
| Prompt caching (breakpoints, auto-cache, cache accounting) | [doc/design/prompt-caching.md](./doc/design/prompt-caching.md) | `schema.go`, `anthropic.go` |
| Errors | [doc/design/errors.md](./doc/design/errors.md) | `errors.go` |
| Multi-model dispatch, health tracking | [doc/design/compose.md](./doc/design/compose.md) | `composes/` |
| Anthropic request/response translation, SSE events | [doc/anthropic/anthropic-chat-api.md](./doc/anthropic/anthropic-chat-api.md) | `anthropic*.go` |
| OpenAI path (zero-translation), SSE parsing | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) | `openai*.go` |

## Architecture at a glance

`Client` implements `ChatCompleter` and routes on `Protocol`:

- **ProtocolOpenAI** (default) — the canonical types *are* the OpenAI wire shape, so this path serializes directly with no translation layer.
- **ProtocolAnthropic** — bidirectional translation; all Anthropic types are package-private and never exposed.

Dispatch happens in `chat.go`; protocol-specific logic is isolated in `openai_*.go` and `anthropic_*.go`.

Packages:

- Root package `aimodel` — client, schema, protocol implementations, streaming
- `composes/` — multi-model dispatch strategies and health tracking
- `examples/` — usage examples for openai, anthropic, and compose patterns

## Official API References

| Protocol | Official docs | Change log |
|---|---|---|
| OpenAI (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | [doc/openai/openai-api-changes.md](./doc/openai/openai-api-changes.md) |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | [doc/anthropic/anthropic-api-changes.md](./doc/anthropic/anthropic-api-changes.md) |

**Maintenance convention**: when an official API changes, update the wrapper code, the relevant `doc/` design or protocol document, and that protocol's change log. See [doc/api.md](./doc/api.md) §6 — root docs link to `doc/`, they do not duplicate it.
