# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`github.com/vogo/aimodel` — A unified Go SDK for AI model APIs with multi-protocol support (OpenAI-compatible, Anthropic) and composable multi-model dispatching. Zero external dependencies.

This SDK is a **thin API wrapper** — it translates requests, manages connections, and normalizes responses across protocols. 
It intentionally does **not** include retry, rate limiting, request validation, caching / persistence, logging / metrics.

## Rules

- delete build binary after test

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

- `ChatRequest` / `ChatResponse` — Use OpenAI-compatible format as canonical representation
- `Content` — Polymorphic: marshals as string (plain text) or `[]ContentPart` (multimodal)
- `Stream` — Concurrent-safe SSE reader with `Recv()` / `Close()` (mutex + atomic bool)

### Multi-Model Composition (composes/)

`ComposeClient` wraps multiple `ChatCompleter` instances with selection strategies:
- **Failover** — Sequential with health tracking
- **Random** — Shuffle active models per request
- **Weighted** — Proportional selection by weight

Health tracking uses exponential backoff (`2^min(errorCount-1, 6)` × base interval) with recovery probes.

### Client Configuration

Option pattern: `WithAPIKey()`, `WithBaseURL()`, `WithProtocol()`, `WithTimeout()`.

Environment variable fallback order:
- API Key: `AI_API_KEY` > `OPENAI_API_KEY` > `ANTHROPIC_API_KEY`
- Base URL: `AI_BASE_URL` > `OPENAI_BASE_URL` > `ANTHROPIC_BASE_URL`

### Anthropic Translation

Anthropic types are private (`anthropicRequest`, `anthropicResponse`, etc.) and translated to/from the canonical OpenAI-compatible types. System messages are extracted into the separate `system` field. Streaming uses Anthropic-specific SSE event types (`content_block_delta`, `message_delta`, etc.).

## Packages

- Root package `aimodel` — Client, schemas, protocol implementations, streaming
- `composes/` — Multi-model dispatch strategies and health tracking
- `examples/` — Usage examples for openai, anthropic, and compose patterns
