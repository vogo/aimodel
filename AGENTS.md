# AGENTS.md

## Project

`github.com/vogo/aimodel` — Go clients for AI model APIs, one complete and independent client per
protocol: `provider/openai` (Chat Completions + Responses) and `provider/anthropic` (Messages).
Zero external dependencies.

A **thin API wrapper**: builds requests, manages connections, decodes responses. Deliberately
excluded — rate limiting, request validation, caching / persistence, logging / metrics, and retry.
The single exception is `composes`, which retries an endpoint before judging it dead (ADR 0004).

## Rules

- delete build binary after test
- current file only contains core api/model dispatching logic, and core rules, not add any other logic.
- code comment not ref documentation and develop phase number.
- remove deprecated features description from document and code comment.

## Design Principles

Three principles arbitrate every interface decision, in this order:

1. **Fidelity (协议保真性)** — a provider package expresses its official API completely and
   losslessly; no field is dropped, renamed or withheld because another vendor lacks it. Unmodelled
   response shapes are preserved verbatim as raw JSON.
2. **Isolation (隔离性)** — `provider/openai` and `provider/anthropic` import neither each other nor
   the root package; a vendor API change touches only that vendor's subpackage. The one escape hatch
   is `openai.ChatCompletionRequest.ExtraBody`, for private top-level parameters of
   OpenAI-*compatible* backends — additive only, collision with a modelled field is a marshal error.
3. **Customization (定制化)** — a provider owns its whole surface: client, wire types, SSE decoding,
   usage aggregation, errors, options. Capabilities are narrow method sets; a missing capability is a
   missing method (compile error), never a runtime error value.

**Neutrality test** — decides whether code may live outside a provider package:

> Legitimate **if and only if** it can be implemented without introducing a semantic data type that
> another provider imports. Anything needing a shared request/response model, bidirectional field
> mapping, or a cross-provider decision about which fields to keep **is a canonical layer being
> rebuilt** — reject it.

Consequences that override the usual instincts:

- **Duplication between providers is intentional.** Timeout options, stream aggregation, SSE scanning
  and error parsing exist twice on purpose. "Removing duplication" is never a reason to create a
  cross-provider type.
- **Nothing is vendor-neutral by default.** Model names, finish/stop reasons, usage shapes and error
  bodies are protocol facts owned by their provider package.
- **Errors are matched structurally.** No shared error type: both `*HTTPError` types implement
  `interface { StatusCode() int }`, and consumers declare that interface locally for `errors.As`.
- **Routing mechanism may be shared; protocol semantics may not.** `composes` is protocol-neutral —
  its interface is endpoint indices, opaque strings, scalars and closures. Test for any shared type:
  *if I add a field to it, must a provider package learn about it?* Pools never mix protocols and
  there is no cross-protocol failover ([ADR 0003](./doc/adr/0003-shared-routing-core-across-protocol-wrappers.md)).

Guard tests in CI enforce the above (import boundaries, structural errors, AST-level neutrality of
`composes`, wire-type round trips) — run `make test` after touching package boundaries.

**Three-way sync** when an official API changes: ① provider wire types and client → ② the relevant
`doc/` page → ③ the protocol's change log (`doc/*/*-api-changes.md`). State explicitly when a step
does not apply. Step ② includes ADRs: accepted ADRs are immutable — supersede with a new ADR and
update the [ADR index](./doc/adr.md), never rewrite. The index holds only decisions in force; a
superseded document is deleted once nothing depends on it, its record staying in git history.

## Build

`make build` (license-check → format → lint → test), `make test` (coverage.out), `make lint`,
`make format`, `make license-check`.

## Where the design lives

**Read the relevant page before changing behavior** — this file does not restate the design.

| If you are touching… | Read | Code |
|---|---|---|
| Package boundaries, what may be shared, why there is no unified client | [doc/architecture.md](./doc/architecture.md) | `provider/*/`, `composes/` |
| OpenAI Chat Completions: wire types, client, SSE, usage, errors, `ExtraBody` | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) | `provider/openai/native.go`, `wire.go` |
| OpenAI Responses: wire types, typed SSE events, hosted tools | [doc/openai/openai-response-api.md](./doc/openai/openai-response-api.md) | `provider/openai/responses*.go` |
| Anthropic Messages: wire types, client, SSE events, usage merging, prompt caching | [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md) | `provider/anthropic/native.go`, `wire.go` |
| Multi-backend dispatch, health tracking, adding a protocol wrapper | [doc/design/compose.md](./doc/design/compose.md) | `composes/`, `composes/openais/`, `composes/anthropics/` |

Also: `integrations/` — integration tests and usage examples per provider and for compose patterns.

## Official API References

- OpenAI Chat Completions — https://platform.openai.com/docs/api-reference/chat
- OpenAI Responses — https://platform.openai.com/docs/api-reference/responses
- Anthropic Messages — https://platform.claude.com/docs/en/api/messages
