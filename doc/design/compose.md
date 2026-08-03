# Multi-Backend Composition

- **Implementation**: `composes/` (routing core), `composes/openais/`, `composes/anthropics/` (protocol wrappers)

Multi-backend dispatch is two layers. The **routing core** decides *which endpoint to try next and what
that attempt did to its health*. A **protocol wrapper** decides *what the request looks like and which
client receives it*. The core serves every protocol; a wrapper serves exactly one.

```
composes/openais       ChatCompletions · ChatCompletionsStream · Responses · ResponsesStream
composes/anthropics    Messages · MessagesStream
        │  required labels + a per-candidate closure          candidate index
        ▼                                                              ▲
composes               capability filter → strategy ordering → recovery probes → attempts
                       health state · aliases · observers · Stats() · MultiError
```

The line between them is the rule from [ADR 0008](../adr/0008-shared-routing-core-across-protocol-wrappers.md):

> **Routing mechanism may be shared across protocols. Protocol semantics may not.**

The core sees endpoint indices, opaque capability labels, ordering metadata and a closure. It never sees
a request, a response or a stream — not through a field, not through a type parameter. That is why it can
be shared without becoming the canonical layer [ADR 0007](../adr/0007-provider-native-as-the-only-public-interface.md)
removed, and why the two wrappers form **separate pools**: one shares *how a candidate is chosen*, never
*what a request is*. There is no cross-protocol failover, and a pool may not mix endpoints of two protocols.

> **Note for readers of earlier versions.** Up to v0.7.x, `composes` itself carried `ChatCompleter`,
> `ModelEntry`, `NewComposeClient` and `NewFromEndpoints` over `provider/openai` types. v0.8.0 moved all of
> them, unchanged in behaviour, to `composes/openais`, and the root package kept only the neutral core — with
> no compatibility aliases. See [MIGRATION.md](../../MIGRATION.md) for the symbol-by-symbol table.

## 1. Choosing a package

| You dispatch across… | Import | Methods |
|---|---|---|
| OpenAI-compatible backends | `composes` + `composes/openais` | `ChatCompletions`, `ChatCompletionsStream`, `Responses`, `ResponsesStream` |
| Anthropic backends | `composes` + `composes/anthropics` | `Messages`, `MessagesStream` |

Both wrappers take the core's `Strategy` and `Option` values, so a pool's operational behaviour is described
the same way whatever it serves:

```go
// OpenAI-compatible pool — manual entries, full control over each client.
cc, err := openais.NewComposeClient(composes.StrategyFailover, []openais.ModelEntry{
    {Name: "gpt-4o",       Client: openai.NewClient(openaiKey), Weight: 3},
    {Name: "qwen3.7-plus", Client: openai.NewClient(qwenKey, openai.WithBaseURL(qwenURL)), Weight: 1},
}, composes.WithRecoveryInterval(30*time.Second))

response, err := cc.ChatCompletions(ctx, &openai.ChatCompletionRequest{ /* … */ })

// The same pool, the same health, the other interaction form.
answer, err := cc.Responses(ctx, &openai.ResponsesRequest{ /* … */ })

// Anthropic pool — declarative, one endpoint per credential.
ac, err := anthropics.NewFromEndpoints(composes.StrategyWeight, []anthropics.EndpointSpec{
    {Alias: "key-1", APIKey: k1, Model: "claude-opus-5", Weight: 1},
    {Alias: "key-2", APIKey: k2, Model: "claude-opus-5", Weight: 1},
})

message, err := ac.Messages(ctx, &anthropic.MessagesRequest{ /* … */ })
```

Each wrapper has two construction entry points:

- **Manual** — `NewComposeClient(strategy, []ModelEntry{...})` with hand-built clients. Use this when you
  need provider-private options or a custom client implementation.
- **Declarative** — `NewFromEndpoints(strategy, []EndpointSpec{...})` for the common case of *same model,
  different BaseURL + key* (multi-key aggregation, canary migration, region/tier sharding). It builds one
  independent provider client per spec.

Construction validates that the entry list is non-empty and every client is non-nil. Native clients perform
no construction-time validation, so an empty key or base URL surfaces at request time, not at construction.

A non-empty `ModelEntry.Name` overrides the request's `Model`; an empty one leaves the request's own model in
place. The request is copied by value first, so the caller's object is untouched — and each interaction form
copies its own request type, with no translation between them.

`ModelEntry.Client` is the wrapper's own interface — `openais.ChatCompleter`, `anthropics.Messenger` — satisfied
by that provider's native client as-is, and by the wrapper's own `*ComposeClient`, so pools nest (a fast pool
that falls back to a cheaper one).

### Interaction forms are separate method sets

`openais.ComposeClient` serves both OpenAI interaction forms over one pool and one health state. They are
separate method sets, not a widened one: `Responses` requires a client that implements `openais.Responder`
(`*openai.Client` and a nested `*ComposeClient` both do). An entry whose client implements only
`ChatCompleter` keeps serving chat and is skipped for Responses; when *no* entry can serve Responses, the
call fails with a `*composes.CapabilityError` naming `"responses"` — before any network I/O.

## 2. Alias and model are distinct

Every endpoint carries two identities that must not be confused:

| Field | Meaning |
|---|---|
| `Name` | the **model** name sent to the backend in the request's `Model` field. |
| `Alias` | the endpoint's **operational identity**, used for health snapshots, sticky routing, and error attribution. |

`Alias` is required and unique on the declarative path (errors name the offending position). On the manual
path an empty alias is derived — from `Name` when free, otherwise `entry-<index>` — so every entry is always
stably addressable; explicit aliases must still be unique. The alias is what lets several endpoints for the
*same* model be told apart in errors and health snapshots.

---

## 3. Routing order

Every call flows through the same chain, in the core:

```
capability filter → strategy ordering → recovery probes → per-endpoint attempts
```

1. **Capability filter** (§6) drops endpoints that have *declared* they cannot serve the call, and endpoints
   the wrapper marked ineligible. Endpoints that declare nothing are never filtered. Runs before health and
   strategy.
2. **Strategy ordering** (§4) orders the capable, health-available candidates.
3. **Recovery probes** (§5) prepend errored endpoints whose backoff has elapsed.
4. **Attempts** try candidates in order until one succeeds; each failure updates health (§5) and emits an
   `AttemptResult` (§7), then fails over.

All strategies return an **ordered candidate list** rather than a single endpoint, so failover applies
uniformly to every strategy. An empty candidate list returns `composes.ErrNoActiveModels`; every candidate
failing returns a `*composes.MultiError` of `EndpointError` (in attempt order).

For streaming, only the call that opens the stream is covered: once a backend has started streaming, a
mid-stream error reaches the caller rather than triggering a retry elsewhere. It is not re-observed and does
not update health.

## 4. Selection strategies

| Strategy | Behavior |
|---|---|
| `StrategyFailover` (default) | Healthy endpoints in declaration order |
| `StrategyRandom` | Healthy endpoints, shuffled |
| `StrategyWeight` | A full ordering sampled **without replacement** in proportion to weight; `Weight <= 0` counts as 1 |
| `StrategySticky` | Pins a session to a stable endpoint (see below); non-default |
| `StrategyCost` | Ascending static cost from `EndpointCost`; unpriced endpoints sort last; alias tie-break |
| `StrategyLatency` | Ascending injected `Latency`; endpoints without latency sort last; alias tie-break |

**Sticky routing.** Attach a session id with `composes.WithSessionID(ctx, id)`. The preferred alias is an FNV
hash over the session id plus the sorted alias set, so it is reproducible across processes and instances and
does not jitter as health changes. The preferred endpoint leads when available; otherwise failover proceeds in
a deterministic order. Without a session id, sticky falls back to the configured fallback strategy
(`composes.WithStickyFallback`, default `StrategyFailover`) — never a randomly generated affinity key.

**Economic routing.** `StrategyCost` / `StrategyLatency` read static metadata on the entry
(`composes.EndpointCost{InputPrice, OutputPrice}`, `Latency`). Missing metadata is never treated as
zero/cheapest — it sorts *after* endpoints that carry data. The output side of the cost key is scaled by
`Call.OutputUnits`, which each wrapper fills from its own protocol's output cap (`max_completion_tokens`,
`max_output_tokens`, `max_tokens`), defaulting to one unit. So the request's own cap can reorder the pool
without the core ever seeing the request. Dynamic pricing is out of scope.

## 5. Health tracking, cooling & recovery probes

The core records `state`, `lastError`, `errorTime` and `errorCount` per endpoint. There are three states:

| State | Entered by | Selection behavior |
|---|---|---|
| `active` | success (or start) | always selectable |
| `cooling` | HTTP **429** | skipped until the cooling interval elapses, then rejoins regular rotation |
| `error` | HTTP **5xx** / transport failure | skipped until an exponential-backoff recovery probe is due |

Failure classification reads the status code through the structural `interface{ StatusCode() int }` — which is
how one state machine classifies failures from *every* protocol without importing any of them:

- **429** → cooling (does **not** advance the failure count);
- **5xx**, or no status (transport failure, or an error embedded in a body the HTTP layer accepted) → error;
- **other 4xx** → request failure: attributed and failed over, but the endpoint stays healthy.

An endpoint whose backoff has elapsed is **prepended** to the candidate list, forming a "probe first, keep
backing off on failure" self-healing loop. The backoff is `interval × 2^min(errorCount-1, 6)` — capped at 64×
the base interval. A 429 answering a probe keeps the endpoint in `error` at its current backoff level rather
than demoting it to the much shorter cooling interval.

Health is per **endpoint**, not per interaction form: a Chat Completions failure cools the same endpoint a
Responses call would have routed to.

## 6. Capability filtering

Each wrapper declares capabilities in its own protocol's terms — `openais.Capability{Tools, Vision,
MaxContextTokens}`, `anthropics.Capability{...}` — set on the entry (`ModelEntry.Capability` /
`EndpointSpec.Capability`) or by a client implementing that wrapper's `CapabilityProvider`. An explicit entry
declaration always wins. At construction the wrapper translates the declaration into the core's opaque label
set (`composes.Declare("tools", …)`); the core compares strings and attaches no meaning to them.

Filtering is **opt-in**: an endpoint that declares nothing is *unknown*, not *incapable*, and is never
excluded. Only an explicit declaration can drop an endpoint. The router excludes incapable endpoints but never
strips tools, rewrites the request, or downgrades. A call whose required labels no endpoint satisfies fails
fast with `*composes.CapabilityError` (matching `composes.ErrCapabilityNotSatisfied`) before any network I/O.

Each wrapper derives requirements from **its own protocol's fields only**:

| Wrapper | tools | vision |
|---|---|---|
| `openais` (Chat) | `tools`, or a `tool_choice` other than `none` | an `image_url` content part |
| `openais` (Responses) | `tools` (including hosted tools), or a `tool_choice` other than `none` | an `input_image` content part |
| `anthropics` | `tools`, or a `tool_choice.type` other than `none` | an `image` content block |

Nothing is translated between these readings: a chat request carrying tools says nothing about a Responses
request that carries none. The label strings are defined per wrapper (`openais.CapabilityTools`,
`anthropics.CapabilityTools`) because naming them centrally would make the core protocol-aware; the pools are
disjoint, so the two never meet.

## 7. Observers & health snapshots

`composes.WithAttemptObserver(fn)` registers a callback invoked when each endpoint attempt finishes, with a
`composes.AttemptResult{Alias, Success, Err, Stream}`. For streaming calls it fires when the stream is
established or fails to establish — post-establishment SSE errors are surfaced by the stream itself, not
re-reported. Observers run synchronously on the request path under no internal lock; they must be fast and
non-blocking.

`Stats()` returns an immutable `[]composes.EndpointStat{Alias, Status, ErrorCount, LastError, ErrorTime}`
snapshot, safe to call concurrently with dispatch. Status reflects routing behavior: a cooling endpoint whose
interval has elapsed is reported `active`.

## 8. Errors

When every candidate fails, the result is a `*composes.MultiError` carrying one `EndpointError{Alias, Err}` per
attempt, in order. It implements `Unwrap() []error`, so `errors.Is` / `errors.As` match **any** backend's
failure — including through to that provider's own error type:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests { /* every backend is rate-limited */ }
```

Declaring that interface locally, rather than importing a provider's error type, is the pattern the core itself
uses. When the candidate list is empty — every endpoint cooling or errored and none due for a probe — the
result is `composes.ErrNoActiveModels`.

## 9. Adding a protocol

A new protocol is a new wrapper package next to `openais` and `anthropics`. It writes:

1. the client interface(s) its endpoints must satisfy — one narrow method set per interaction form;
2. `ModelEntry` / `EndpointSpec` and their alias derivation;
3. a `Capability` type plus the predicates that read *its* request fields, translated into
   `composes.Declare(...)` labels;
4. per-endpoint request copying with the model override;
5. dispatch methods that call `composes.Dispatch` with a closure.

It inherits every operational feature and changes no existing package. It gets no interoperability with the
other pools — correctly, since there is none to give.
