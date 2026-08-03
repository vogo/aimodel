# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`github.com/vogo/aimodel` — Go clients for AI model APIs. Each supported protocol gets its own
complete, independent client: OpenAI-compatible (`provider/openai`, covering Chat Completions and
Responses) and Anthropic Messages (`provider/anthropic`). Zero external dependencies.

This SDK is a **thin API wrapper** — it builds requests, manages connections, and decodes responses.
It intentionally does **not** include rate limiting, request validation, caching / persistence, logging / metrics.
Retry is likewise out of the provider packages; the one exception is the `composes` routing layer, which
retries an endpoint before judging it dead (ADR 0004).

## Rules

- delete build binary after test
- current file only contains core api/model dispatching logic, and core rules, not add any other logic.
- code comment not ref documentation and develop phase number.

## Design Principles

Three principles arbitrate every interface decision. When a new vendor or a new interaction form arrives, judge the change against them in this order.

1. **Fidelity (协议保真性)** — each provider package expresses its official API completely and
   losslessly. Nothing is withheld pending another vendor's equivalent, nothing is reshaped to
   resemble another vendor's spelling, and no field is dropped because it has no counterpart
   elsewhere. There is no greatest common denominator to respect: the API a package wraps is the
   whole specification for that package. Unmodelled response shapes are preserved verbatim (raw
   JSON) rather than discarded.
2. **Isolation (隔离性)** — provider protocols are isolated from each other and evolve
   independently. `provider/openai` and `provider/anthropic` import neither each other nor the root
   package, and a vendor API change touches only that vendor's subpackage. Vendor-specific
   parameters are ordinary fields of that vendor's types; client configuration is that package's own
   options (e.g. `anthropic.WithBeta`, `openai.WithBaseURL`). The one controlled escape hatch is
   `openai.ChatCompletionRequest.ExtraBody`, for private top-level parameters of OpenAI-*compatible*
   backends — additive only, and a collision with a modelled field is a marshal-time error.
3. **Customization (定制化)** — a provider owns its whole surface: client, wire types, SSE decoding,
   usage aggregation, errors and options. Capabilities are narrow method sets on that provider's
   client (a new interaction form gets a new method set, never a widened existing one). A provider
   that lacks a capability simply does not have the method — absence is a compile error, not a
   runtime error value.

**Neutrality test** — the rule that decides whether code may live outside a provider package:

> An enhancement is legitimate outside a provider package **if and only if** it can be implemented
> inside a single package without introducing a semantic data type that another provider imports.
> Anything requiring a shared request/response model, bidirectional field mapping, or a
> cross-provider decision about which fields to keep **is a canonical layer being rebuilt** — reject
> it.

Consequences to apply directly:

- **Duplication between the two providers is expected and accepted.** Timeout options, stream
  aggregation, SSE scanning and error parsing exist twice, on purpose. Do not factor them into a
  shared package; "removing duplication" is not a sufficient reason to create a cross-provider type.
- **Nothing is vendor-neutral by default.** Model names, finish/stop reasons, usage shapes and error
  bodies are protocol facts and belong to the provider package that serves them.
- **Errors are matched structurally.** There is no shared error type. Both `*HTTPError` types
  implement `interface { StatusCode() int }`; a consumer declares that interface locally and uses
  `errors.As`. `composes` does exactly this — it classifies and aggregates backend errors from every
  protocol without naming, or importing, any provider's error type.
- **Routing mechanism may be shared; protocol semantics may not.** `composes` is a protocol-neutral
  routing core (active endpoint, strategies, retries, health, aliases, attribution) whose entire interface is endpoint
  indices, opaque strings, scalars and closures; `composes/openais` and `composes/anthropics` bind it
  to their own wire types and never import each other. The test for any shared type: *if I add a
  field to it, does a provider package have to learn about it?* Pools never mix protocols, and there
  is no cross-protocol failover ([ADR 0003](./doc/adr/0003-shared-routing-core-across-protocol-wrappers.md)).

Guard tests enforce this in CI rather than leaving it to convention: providers import neither each
other nor the root package; no public API references a shared semantic package; both `*HTTPError`
types satisfy `StatusCode() int`; packages declared vendor-neutral contain no protocol-semantic
identifiers (`message`, `content`, `tool`, `usage`, …), checked over the AST — `composes` is one of
them since v0.8.0, its wrappers deliberately are not; the routing core imports nothing from this
module and exports no type from it; the two compose wrappers import neither each other nor the
other's provider; and every public wire type survives a marshal → unmarshal → marshal round trip
unchanged.

**Three-way sync**: when an official API changes, update in order — ① the provider's wire types and
client → ② the relevant `doc/` document → ③ the protocol's change log (`doc/*/**-api-changes.md`).
When a step does not apply, state so explicitly. Details in [doc/architecture.md](./doc/architecture.md).

Step ② includes the ADRs: if a change contradicts an invariant an accepted ADR states, the ADR is
part of the sync, not an afterthought. Accepted ADRs are immutable — add a new ADR that supersedes
it and update the [ADR index](./doc/adr.md), rather than rewriting the old decision. The index holds
only decisions in force: once nothing in force depends on a superseded ADR, its document is removed
and its record stays in git history.

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
| Anything cross-cutting: package boundaries, what may be shared, why there is no unified client | [doc/architecture.md](./doc/architecture.md) | `provider/*/`, `composes/` |
| OpenAI Chat Completions: wire types, client, SSE, usage, errors, `ExtraBody` | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) | `provider/openai/native.go`, `wire.go` |
| OpenAI Responses: wire types, typed SSE events, hosted tools | [doc/openai/openai-response-api.md](./doc/openai/openai-response-api.md) | `provider/openai/responses*.go` |
| Anthropic Messages: wire types, client, SSE events, usage merging, prompt caching | [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md) | `provider/anthropic/native.go`, `wire.go` |
| Multi-backend dispatch, health tracking, adding a protocol wrapper | [doc/design/compose.md](./doc/design/compose.md) | `composes/`, `composes/openais/`, `composes/anthropics/` |

## Architecture at a glance

There is no unified client and no shared schema. A caller picks a protocol by importing its package:

- **`provider/openai`** — native `/chat/completions` and `/v1/responses` clients over their own wire
  types (`native.go` / `wire.go` / `responses*.go`). Serves OpenAI and every OpenAI-compatible
  backend; `ExtraBody` carries backend-private top-level parameters.
- **`provider/anthropic`** — native `/v1/messages` client and wire types, including SSE event
  decoding and stream usage merging.
- **`composes`** — the protocol-neutral routing core (no provider imports), with `composes/openais`
  and `composes/anthropics` binding it to their wire types.

Adding a protocol = a new subpackage with its own client, with **zero change** to any existing
package. There is nothing to register with and no contract to satisfy — and correspondingly no
automatic interoperability between protocols.

Packages:

- `provider/openai`, `provider/anthropic` — the two protocol clients, mutually independent
- `composes/` — the neutral routing core: strategies, health, probes, aliases, observers, attribution
- `composes/openais/` — OpenAI-wire pools: Chat Completions **and** Responses
- `composes/anthropics/` — Anthropic Messages pools
- `integrations/` — integration tests and usage examples per provider and for compose patterns

## Official API References

| Protocol | Official docs | Change log |
|---|---|---|
| OpenAI Chat Completions (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | [doc/openai/openai-api-changes.md](./doc/openai/openai-api-changes.md) |
| OpenAI Responses | https://platform.openai.com/docs/api-reference/responses | [doc/openai/openai-api-changes.md](./doc/openai/openai-api-changes.md) |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | [doc/anthropic/anthropic-api-changes.md](./doc/anthropic/anthropic-api-changes.md) |

**Maintenance convention**: follow the three-way sync in [Design Principles](#design-principles) (provider code → `doc/` → change log). See [doc/architecture.md](./doc/architecture.md) — root docs link to `doc/`, they do not duplicate it.
