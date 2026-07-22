# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`github.com/vogo/aimodel` — A unified Go SDK for AI model APIs with multi-protocol support (OpenAI-compatible, Anthropic) and composable multi-model dispatching. Zero external dependencies.

This SDK is a **thin API wrapper** — it translates requests, manages connections, and normalizes responses across protocols.
It intentionally does **not** include retry, rate limiting, request validation, caching / persistence, logging / metrics.

## Rules

- delete build binary after test
- current file only contains core api/model dispatching logic, and core rules, not add any other logic.

## Design Principles

Three principles arbitrate every interface decision. When a new vendor or a new interaction form arrives, judge the change against them in this order.

1. **Universality (通用性)** — the canonical interface is the stable layer. Adding a vendor or a new interaction form must require **zero changes** to existing canonical types and signatures; new capability surfaces are **additive only** (new interfaces / new optional fields), never modifications to existing exported signatures.
2. **Extensibility (扩展性)** — vendor protocols are isolated from each other and evolve independently. A vendor API change touches only that vendor's subpackage (`provider/anthropic/` / `provider/openai/`), never the core abstraction. Vendor-specific parameters go through the unified extension channel — `ais.Extensions`, a per-provider-namespace map (`json:"-"`) on every extendable canonical node, with strongly-typed values and set/read helpers owned by the provider package (e.g. `anthropic.ExtendRequest` / `anthropic.RequestExtensionOf`); client-level vendor options go through `WithProviderOptions` (e.g. `anthropic.Options`). See [doc/architecture.md](./doc/architecture.md) §2. Vendor concepts never appear in canonical types — enforced by `ais/schema_vendor_test.go` (no vendor-named identifiers in `ais`; the only `json:"-"` fields are the `Extensions` channel itself).
3. **Customization (定制化)** — each vendor provides a public, native, full-fidelity client and types that pursue complete coverage of the official API and stay continuously synced with it. The canonical translation layer is built **on top of** the native layer, not the other way around.

**Field-attribution test**: a semantic enters the canonical types only when **≥ 2 vendors** share a mappable common semantic. Similar names or wire shapes alone are **not** consensus, and a field is never promoted to canonical for implementation convenience — nor kept there because "it's the OpenAI shape" when only one vendor implements it (fields that are part of the OpenAI-compatible protocol surface count as multi-vendor via protocol adoption; record the evidence in [doc/architecture.md](./doc/architecture.md) §2). Everything single-vendor lives in that provider's package and rides the `Extensions` channel (request side) or is written into it by the provider's response translation (response side); vendor convenience constants (e.g. Anthropic pass-through finish reasons) are named in the provider package. Canonical is the greatest common denominator; completeness is the native layer's responsibility.

**Four-way sync**: when an official API changes, update in order — ① the vendor's native layer → ② the canonical translation → ③ the relevant `doc/` document → ④ the protocol's change log plus the `CHANGES.md` index. When a step does not apply, state so explicitly — never shortcut by writing a vendor-specific change directly into canonical. Details in [doc/architecture.md](./doc/architecture.md) §6.

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
| Anything cross-cutting: canonical representation, registry dispatch, client options, env fallback | [doc/architecture.md](./doc/architecture.md) | `client.go`, `chat.go`, `ais/provider.go`, `ais/registry.go` |
| Request/response types, `Usage`, `Content`, `Clone()` | [doc/design/data-model.md](./doc/design/data-model.md) | `ais/schema.go` |
| `Stream`, SSE, `AppendDelta`, `ExtraBlocks`, interception | [doc/design/streaming.md](./doc/design/streaming.md) | `stream.go`, `intercept.go`, `provider/*/stream.go` |
| Tool definitions, `tool_choice`, parallel tool results | [doc/design/tool-use.md](./doc/design/tool-use.md) | `ais/schema.go`, `provider/anthropic/anthropic.go` |
| Prompt caching (breakpoints, auto-cache, cache accounting) | [doc/design/prompt-caching.md](./doc/design/prompt-caching.md) | `ais/schema.go`, `provider/anthropic/anthropic.go` |
| Errors | [doc/design/errors.md](./doc/design/errors.md) | `ais/errors.go` |
| Multi-model dispatch, health tracking | [doc/design/compose.md](./doc/design/compose.md) | `composes/` |
| Anthropic request/response translation, SSE events | [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md) | `provider/anthropic/` |
| OpenAI path (zero-translation), SSE parsing | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) | `provider/openai/` |

## Architecture at a glance

`Client` implements the `ChatCompleter` capability and delegates to a **provider** resolved by name from a registry at construction time. The default provider (`openai.Name`) is OpenAI-compatible; `anthropic.Name` selects Anthropic. Both built-ins register themselves on import.

- **openai** (default) — the canonical types *are* the OpenAI wire shape, so this path serializes directly with no translation layer.
- **anthropic** — public native `/v1/messages` client and wire types, with bidirectional translation layered on the same types for canonical calls.

`chat.go` runs one shared pipeline (clone → default model → build → single HTTP call → parse/stream); the vendor boundary is the `ais.ChatProvider` contract, implemented per subpackage. Adding a protocol = new subpackage that calls `ais.Register` in `init`, with **zero root-package change**. New interaction forms are added as new capability interfaces, never by widening `ChatCompleter`.

Packages:

- `ais/` — vendor-neutral canonical schema, error model, `ChatProvider` contract, and registry (no vendor deps)
- Root package `aimodel` — `Client` facade, shared pipeline, capability interface, `Stream`; canonical types used directly from `ais`
- `provider/openai/`, `provider/anthropic/` — the two built-in providers (each self-registers on import)
- `composes/` — multi-model dispatch strategies and health tracking (depends on the root capability interface + canonical `ais/schema` types)
- `integrations/` — integration tests and usage examples for openai, anthropic, and compose patterns

## Official API References

| Protocol | Official docs | Change log |
|---|---|---|
| OpenAI (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | [doc/openai/openai-api-changes.md](./doc/openai/openai-api-changes.md) |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | [doc/anthropic/anthropic-api-changes.md](./doc/anthropic/anthropic-api-changes.md) |

**Maintenance convention**: follow the four-way sync in [Design Principles](#design-principles) (native layer → canonical translation → `doc/` → change log). See [doc/architecture.md](./doc/architecture.md) §6 — root docs link to `doc/`, they do not duplicate it.
