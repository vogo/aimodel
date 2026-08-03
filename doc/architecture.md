# Architecture

`github.com/vogo/aimodel` — Go clients for AI model APIs, one per protocol, with zero external dependencies.

This document covers the **cross-cutting architecture**: what the SDK is for, where the package boundaries are and why nothing crosses them, and how the repository is maintained. Protocol detail lives with the protocol:

| Topic | Document |
|---|---|
| OpenAI Chat Completions: wire types, client, SSE, usage, errors, `ExtraBody` | [openai/openai-chat-api.md](./openai/openai-chat-api.md) |
| OpenAI Responses: wire types, typed SSE events, hosted tools | [openai/openai-response-api.md](./openai/openai-response-api.md) |
| Anthropic Messages: wire types, client, SSE events, usage merging, prompt caching | [anthropic/anthropic-message-api.md](./anthropic/anthropic-message-api.md) |
| Multi-backend dispatch and health tracking | [design/compose.md](./design/compose.md) |

The decisions behind this architecture are recorded in the [ADR index](./adr.md); [ADR 0002](./adr/0002-provider-native-as-the-only-public-interface.md) is the one that shapes everything below.

---

## 1. Design scope

aimodel is a **thin API wrapper**. Its responsibilities are strictly limited to three things:

1. **Request building** — turn a protocol's own request type into its wire body;
2. **Connection management** — HTTP client, timeouts, auth headers, SSE reading;
3. **Response decoding** — decode that protocol's responses and stream events, losslessly.

It **deliberately excludes** rate limiting, request validation, caching / persistence, and logging / metrics. Those belong to the caller or a framework above: putting them in the SDK introduces implicit behavior and costs the caller cannot control. Retry is excluded from the provider clients on the same grounds — one call, one HTTP request — with one bounded exception: `composes` retries an endpoint before judging it dead, because deciding that a backend is unusable is the whole job of that layer ([ADR 0004](./adr/0004-stateful-active-endpoint-with-in-call-retry.md)).

Consequences that follow directly:

- Open-string parameters (`ReasoningEffort`, `Model`, stop/finish reasons, tool types) are **not** enum-validated. Constants exist for convenience; a value the SDK has never heard of still reaches the backend.
- Request structures carry no side-effecting state — a request value is safe to reuse, and a client never mutates the one it is given.
- One call = one HTTP request. The multi-backend failover path is the only exception ([design/compose.md](./design/compose.md)).

## 2. Two independent protocol clients

There is no unified client and no shared request/response model. A caller picks a protocol by importing its package:

```
  caller ──▶ provider/openai   ──▶ POST {baseURL}/chat/completions
         │                     └──▶ POST {baseURL}/responses
         │
         ├──▶ provider/anthropic ──▶ POST {baseURL}/v1/messages
         │
         └──▶ composes ─┬─▶ composes/openais    ──▶ several OpenAI-compatible backends
                        └─▶ composes/anthropics ──▶ several Anthropic backends
```

Each package owns its whole surface: client, options, wire types, SSE decoding, stream accumulation, usage and errors. `provider/openai` and `provider/anthropic` import neither each other nor the root package, and no third package sits between them.

There is deliberately no vendor-neutral layer holding a shared schema for both protocols to translate to and from. Its one differentiating capability — delivering one request to either protocol — is needed nowhere here, while its admission rule ("a field is shared when ≥ 2 providers map it") keeps most of each vendor's API out of reach, and everything excluded has to travel through a `map[string]any` side channel. The reasoning, the evidence and the trade-offs accepted are in [ADR 0002](./adr/0002-provider-native-as-the-only-public-interface.md).

### 2.1 What the three principles mean here

**Fidelity.** A provider package expresses its official API completely. Nothing is withheld pending another vendor's equivalent, nothing is renamed to resemble another vendor's spelling, and response shapes the package does not model are preserved verbatim rather than dropped — `anthropic.ResponseContentBlock.Raw`, `openai.ResponseTool.Raw`, `openai.ChatCompletionRequest.ExtraBody`. `TestWireTypesRoundTripLosslessly` states this as a check: every exported wire type must survive marshal → unmarshal → marshal unchanged.

**Isolation.** A vendor API change touches one subpackage. Vendor parameters are ordinary fields of that vendor's types; client configuration is that package's own options (`anthropic.WithBeta`, `openai.WithBaseURL`). The single controlled escape hatch is `openai.ChatCompletionRequest.ExtraBody`, for the private top-level parameters OpenAI-*compatible* backends add (`enable_thinking`, `chat_template_kwargs`, …) — additive only, and a key colliding with a modelled field fails at marshal time.

**Customization.** Capabilities are narrow method sets on a provider's own client. A new interaction form gets a new method set rather than a widened existing one — `Responses`/`ResponsesStream` sit alongside `ChatCompletions`/`ChatCompletionsStream` on `*openai.Client` instead of extending it. A provider that lacks a capability simply does not have the method, so absence is a compile error rather than a runtime error value.

### 2.2 The neutrality test

The rule that decides whether code may live outside a provider package:

> An enhancement is legitimate outside a provider package **if and only if** it can be implemented inside a single package without introducing a semantic data type that another provider imports. Anything requiring a shared request/response model, bidirectional field mapping, or a cross-provider decision about which fields to keep **is a canonical layer being rebuilt** — reject it.

Applied consequences:

- **Duplication between the two providers is expected.** Timeout options, stream accumulation, SSE scanning and error parsing exist twice, on purpose. "Removing duplication" is not a sufficient reason to create a cross-provider type.
- **Nothing is vendor-neutral by default.** Model names, finish/stop reasons, usage shapes and error bodies are protocol facts and live with the protocol that serves them.
- **Errors are matched structurally.** There is no shared error type. Both `*HTTPError` types implement `interface { StatusCode() int }`, which a consumer declares locally:

  ```go
  type statusCoder interface{ StatusCode() int }

  var sc statusCoder
  if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests {
      // back off
  }
  ```

- **Routing mechanism may be shared; protocol semantics may not.** `composes` is a neutral routing core — the active endpoint, strategies, retries, health, aliases, attribution — whose entire interface is endpoint indices, opaque strings, scalars and closures. `composes/openais` and `composes/anthropics` bind it to their own wire types and never meet. The test to apply to any shared type: *if I add a field to it, does a provider package have to learn about it?* ([ADR 0003](./adr/0003-shared-routing-core-across-protocol-wrappers.md))

### 2.3 Guards

The failure mode this architecture risks is gradual: duplicated helpers get factored into a shared package, and the shared package acquires a request type. Each step reads as reasonable, so the boundary is enforced by tests rather than by review (`dependency_test.go`, `provider/*/roundtrip_test.go`):

1. Providers import neither each other nor the root package.
2. No package from this module is imported by both providers.
3. Both `*HTTPError` types satisfy `interface { StatusCode() int }` (compile-time assertion).
4. Packages declared vendor-neutral declare no protocol-semantic identifier (`message`, `content`, `tool`, `usage`, `chat`, `prompt`, `token`, `choice`, `completion`), checked over the AST so comments and string literals cannot trip it. `composes` is one of those packages; its two wrappers deliberately are not.
5. Every exported wire type round-trips losslessly, and every exported struct is either round-tripped or explicitly declared not to be a wire type.
6. The routing core imports nothing from this module — not one provider, not two — and its exported API references no type from this module.
7. The two compose wrappers import neither each other nor the other's provider, and neither imports the root package.

## 3. Provider packages

### 3.1 `provider/openai`

Serves OpenAI and every OpenAI-compatible backend, over two interaction forms:

| Form | Methods | Types |
|---|---|---|
| Chat Completions | `ChatCompletions`, `ChatCompletionsStream` | `ChatCompletionRequest` / `ChatCompletionResponse` / `ChatCompletionChunk` |
| Responses | `Responses`, `ResponsesStream` | `ResponsesRequest` / `Response` / `ResponseStreamEvent` |

```go
client := openai.NewClient(apiKey,
    openai.WithBaseURL("https://api.openai.com/v1"),
    openai.WithTimeout(90*time.Second),
)
```

| Option | Purpose |
|---|---|
| `WithBaseURL(string)` | API base URL; trailing `/` stripped. Defaults to `https://api.openai.com/v1` |
| `WithHTTPClient(*http.Client)` | Full transport control; `nil` panics (a programming error) |
| `WithTimeout(time.Duration)` | Bounds a whole call. Copies the client configured so far, so the caller's own `*http.Client` is never mutated and an earlier transport survives |

`NewClient` does not return an error: there is nothing left to validate at construction. `Model` is required on the request.

### 3.2 `provider/anthropic`

Serves the Messages API:

```go
client := anthropic.NewClient(apiKey,
    anthropic.WithBeta("context-1m-2025-08-07"),
    anthropic.WithVersion("2023-06-01"),
    anthropic.WithUserProfileID("user_abc123"),
)
```

Same three transport options as OpenAI, plus the three header options above. `MaxTokens` is required by the API and by the wire type.

### 3.3 Streams

Both native streams accumulate while the caller reads. Every `Recv` folds its event into an in-progress result, so after `io.EOF` the assembled message and the token accounting are available without the caller tracking deltas:

| | OpenAI | Anthropic |
|---|---|---|
| Read one event | `Recv() (*ChatCompletionChunk, error)` | `Recv() (*StreamEvent, error)` |
| Assembled result | `Response() *ChatCompletionResponse` | `Message() *MessagesResponse` |
| Token accounting | `Usage() *ChatCompletionUsage` | `Usage() *MessagesUsage` |

Before the stream ends both accessors return a live snapshot, in which a tool call's arguments may still be a partial JSON fragment. `Close` is idempotent and safe to call concurrently with `Recv`.

## 4. `composes` and its wrappers

Multi-backend dispatch is two layers. `composes` is the protocol-neutral routing core: the pool's single active endpoint, the five selection strategies that choose it, in-call exponential retries, the health machine on a fixed recovery timer (`available` / `dead` stored, `probation` derived for an endpoint the clock restored but nothing has confirmed), alias identity, capability filtering over opaque labels, attempt observers, `Stats()` snapshots and `MultiError` attribution. It sees no request, response or stream type — a wrapper hands it `Dispatch[T](ctx, router, call, attempt)` and owns everything protocol-shaped inside that closure.

| Package | Pool | Methods |
|---|---|---|
| `composes/openais` | OpenAI-compatible backends | `ChatCompletions`, `ChatCompletionsStream`, `Responses`, `ResponsesStream` |
| `composes/anthropics` | Anthropic backends | `Messages`, `MessagesStream` |

`ModelEntry.Client` is the wrapper's own interface, satisfied by that provider's native client and by the wrapper's `*ComposeClient`, so pools nest. The two pools are disjoint: they share how a candidate is chosen and how health is recorded, never what a request is, so there is no cross-protocol failover. Details: [design/compose.md](./design/compose.md).

## 5. Repository layout

| Path | Contents |
|---|---|
| Root package `aimodel` | Package clause and the module's architectural guard tests. Exports nothing |
| `provider/openai/` | Chat Completions and Responses: client and options (`native.go`, `responses.go`), wire types (`wire.go`, `responses_wire.go`), typed stream events (`responses_events.go`), stream accumulation (`accumulate.go`), model and discriminator constants (`model.go`, `responses_const.go`) |
| `provider/anthropic/` | Messages: client and options (`native.go`), wire types (`wire.go`), stream accumulation and usage merging (`accumulate.go`), model and discriminator constants (`model.go`, `const.go`) |
| `composes/` | The neutral routing core: selection strategies (`strategy.go`), the two-state health machine (`health.go`), the active endpoint, retries and dispatch loop (`router.go`), endpoint metadata and capability filtering (`endpoint.go`), aggregate errors (`errors.go`) |
| `composes/openais/` | OpenAI-wire wrapper: compose client and both interaction forms (`openais.go`), entries and declarative specs (`endpoint.go`), capability predicates (`capability.go`) |
| `composes/anthropics/` | Anthropic-wire wrapper, same file layout |
| `integrations/` | Per-provider examples and offline integration tests |

## 6. Maintenance convention

When an official API changes, update these in sync:

1. the provider's wire types and client;
2. the relevant `doc/` document — the protocol's own page, and this one if a package boundary moved;
3. the protocol's change log — [anthropic/anthropic-api-changes.md](./anthropic/anthropic-api-changes.md) or [openai/openai-api-changes.md](./openai/openai-api-changes.md).

When a step does not apply, say so explicitly rather than skipping it silently.

When an architectural decision changes, add an ADR under [`doc/adr/`](./adr/) and update the [ADR index](./adr.md). This is part of step 2, not optional cleanup: a change that contradicts an invariant an accepted ADR states is not synced until that ADR is superseded. Accepted ADRs are immutable, so record the new decision in a new ADR and mark the old one superseded rather than editing its decision text. The index carries only decisions in force: once nothing in force depends on a superseded ADR, its document is removed and the record of it stays in git history and in the protocol change logs.
