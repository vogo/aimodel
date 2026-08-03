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
composes               capability filter → active endpoint → retries → failover
                       active state · health · aliases · observers · Stats() · MultiError
```

The line between them is the rule from [ADR 0003](../adr/0003-shared-routing-core-across-protocol-wrappers.md):

> **Routing mechanism may be shared across protocols. Protocol semantics may not.**

The core sees endpoint indices, opaque capability labels, ordering metadata and a closure. It never sees
a request, a response or a stream — not through a field, not through a type parameter. That is why it can
be shared without becoming the canonical layer [ADR 0002](../adr/0002-provider-native-as-the-only-public-interface.md)
removed, and why the two wrappers form **separate pools**: one shares *how a candidate is chosen*, never
*what a request is*. There is no cross-protocol failover, and a pool may not mix endpoints of two protocols.

> **Note for readers of earlier versions.** Up to v0.7.x, `composes` itself carried `ChatCompleter`,
> `ModelEntry`, `NewComposeClient` and `NewFromEndpoints` over `provider/openai` types. v0.8.0 moved all of
> them, unchanged in behaviour, to `composes/openais`, and the root package kept only the neutral core — with
> no compatibility aliases.

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
}, composes.WithRetryPolicy(time.Second, 3), composes.WithRecoverTime(5*time.Minute))

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
| `Alias` | the endpoint's **operational identity**, used for health snapshots and error attribution. |

`Alias` is required and unique on the declarative path (errors name the offending position). On the manual
path an empty alias is derived — from `Name` when free, otherwise `entry-<index>` — so every entry is always
stably addressable; explicit aliases must still be unique. The alias is what lets several endpoints for the
*same* model be told apart in errors and health snapshots.

---

## 3. The active endpoint

A pool serves its calls from **one endpoint at a time** — its *active* endpoint. The strategy chooses that
endpoint when the pool has none, or when the current one has been judged dead; it does not run per call. So a
run of successful calls all land on the same backend, whatever the strategy, and a caller gets a stable
attribution rather than a fresh draw each time.

Every call flows through the same chain, in the core:

```
capability filter → active endpoint (reuse · select · switch) → attempt + retries → next endpoint
```

1. **Capability filter** (§6) drops endpoints that have *declared* they cannot serve the call, and endpoints
   the wrapper marked ineligible. Endpoints that declare nothing are never filtered. It runs first — before
   health, before the active endpoint is even consulted.
2. **The active endpoint** serves the call whenever it is available and capable of it. Three cases divert:
   - *no active endpoint yet* — the strategy picks one among the capable available endpoints and it is
     committed as the pool's active;
   - *the active endpoint is dead* — the strategy picks a replacement and commits it;
   - *the active endpoint is healthy but cannot serve this call* — the strategy picks among the capable
     available endpoints **for this call only**. The pool's active endpoint is left where it is.
3. **Attempt and retries** (§5) run against the chosen endpoint: the first attempt, then up to `maxRetries`
   more with a doubling wait.
4. **Next endpoint** — exhausting the retries (or a credential failure) marks the endpoint dead. If another
   capable available endpoint remains, the same call continues on it; the switch is committed as the new
   active unless the failing endpoint was only a temporary pick.

A call that finds nothing to try returns `composes.ErrNoActiveModels`. A call that tried endpoints and ran out
returns a `*composes.MultiError` with **one `EndpointError` per endpoint**, in the order they were tried — a
retried endpoint contributes the error it finally failed with, not one entry per retry.

For streaming, only the call that opens the stream is covered: once a backend has started streaming, a
mid-stream error reaches the caller rather than triggering a retry or a switch. It is not re-observed and does
not update health.

### One call at a time

A pool belongs to **one conversation** and serves it **one call at a time**. A second call arriving while one
is in flight does not queue — it is rejected immediately with `composes.ErrCallInProgress`, having contacted
no endpoint and changed no health state. Concurrency on a single pool is a usage error, not a load to smooth
out, so it is reported rather than hidden behind latency.

```go
_, err := cc.ChatCompletions(ctx, request)
if errors.Is(err, composes.ErrCallInProgress) {
    // Another call owns this pool. Build one pool per concurrent conversation.
}
```

Three properties round this out:

- **The capability filter runs before the busy check.** A call no endpoint can serve is reported as a
  `*CapabilityError`, never as a busy pool — the two failures mean different things to a caller.
- **The slot is released on every exit path**, including `*MultiError` and `ErrNoActiveModels`, so a failed
  call never strands the pool.
- **`Stats()` does not contend for it.** It reads pool state under a different lock and answers while a call
  is in flight.

Callers that genuinely need parallel requests build **one pool per concurrent conversation**. Pools are cheap,
and each keeps its own active endpoint and health — which is the point: a shared pool would give the two
conversations a shared active endpoint anyway.

> **Streaming holds the pool only until the stream is established.** The slot is released when `…Stream`
> returns, so the caller reads the stream while a subsequent call may proceed. Holding it until the stream was
> closed would need a completion hook the provider stream types do not expose, and an abandoned stream would
> then wedge the pool permanently. For the one-conversation-per-pool usage this package targets, the caller is
> reading the stream before it issues the next call anyway.

Within a call, the active endpoint and its *generation* change together under a separate short-held lock, and
a caller may only retire the active endpoint it observed at the generation it saw. That guard is what keeps
the switch count at one even when calls race for the pool. No lock is held across a network attempt or a
retry wait.

## 4. Selection strategies

A strategy decides **who becomes the active endpoint**, not who serves each call. That is the substantive
change from earlier versions: `StrategyRandom` and `StrategyWeight` no longer spread load across requests —
they draw once, when the pool needs an endpoint. A pool is not a load balancer.

| Strategy | Behavior when selecting |
|---|---|
| `StrategyFailover` (default) | The first available endpoint in declaration order |
| `StrategyRandom` | A uniformly random available endpoint |
| `StrategyWeight` | An available endpoint drawn in proportion to weight; `Weight <= 0` counts as 1 |
| `StrategyCost` | The lowest static cost from `EndpointCost`; unpriced endpoints sort last; alias tie-break |
| `StrategyLatency` | The lowest injected `Latency`; endpoints without latency sort last; alias tie-break |

Each strategy still produces a full ordering, which is also the deterministic sequence a single call walks
through as endpoints die under it.

**Economic routing.** `StrategyCost` / `StrategyLatency` read static metadata on the entry
(`composes.EndpointCost{InputPrice, OutputPrice}`, `Latency`). Missing metadata is never treated as
zero/cheapest — it sorts *after* endpoints that carry data. The output side of the cost key is scaled by
`Call.OutputUnits`, which each wrapper fills from its own protocol's output cap (`max_completion_tokens`,
`max_output_tokens`, `max_tokens`), defaulting to one unit. So the request's own cap can decide *which
endpoint the pool settles on*, without the core ever seeing the request. Dynamic pricing is out of scope.

## 5. Retries, health and recovery

The core records `state`, `lastError`, `errorTime` and `errorCount` per endpoint. There are exactly two
states:

| State | Entered by | Selection behavior |
|---|---|---|
| `available` | success, or start | selectable — as the active endpoint, its replacement, or a temporary pick |
| `dead` | exhausted retries, or HTTP **401/403** | not selectable until `recover_time` has elapsed |

Failure classification reads the status code through the structural `interface{ StatusCode() int }` — which is
how one state machine classifies failures from *every* protocol without importing any of them:

- **401 / 403** → the endpoint's credentials do not work. No retry, no wait: it is dead at once.
- **everything else** — 429, 400 and other 4xx, 5xx, transport failures, an error embedded in a body the HTTP
  layer accepted — → retryable. The same endpoint is attempted again under the retry policy, and only an
  exhausted round judges it.

**In-call retries.** `composes.WithRetryPolicy(base, maxRetries)` makes retry *k* wait `base × 2^(k-1)`, so an
endpoint is attempted at most `1 + maxRetries` times and the worst-case **synchronous** wait a caller pays is
`base × (2^maxRetries − 1)`. The waits are interruptible: a cancelled context ends the wait and the call. This
is the one retry in this module — provider packages remain retry-free ([ADR 0001](../adr/0001-keep-the-sdk-a-thin-wrapper.md),
[ADR 0004](../adr/0004-stateful-active-endpoint-with-in-call-retry.md)).

**Recovery.** `composes.WithRecoverTime(d)` sets how long a dead endpoint stays out. When `d` has elapsed it is
a candidate again — that is *all*: recovery never takes the pool back from a healthy incumbent, so an endpoint
returning causes no switch of its own. There is no recovery probe and no exponential health backoff.

`NewRouter` (and therefore both wrappers' constructors) rejects `recover_time <= base × 2^maxRetries`, along
with a non-positive base or recover time and a negative retry count. A recovery window shorter than the backoff
scale the retries themselves reach would put an endpoint back into rotation inside the very interval it just
failed through.

**Cancellation** never counts as a failure: it does not change health, does not retry and does not switch. The
attempt is still attributed to its alias for observation.

Health is per **endpoint**, not per interaction form: a Chat Completions failure kills the same endpoint a
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

`Stats()` returns an immutable `[]composes.EndpointStat{Alias, Status, Active, ErrorCount, LastError,
ErrorTime}` snapshot, safe to call concurrently with dispatch. `Status` reflects routing behavior: a dead
endpoint whose recover time has elapsed is already reported `available`. `Active` marks the one endpoint
currently serving the pool, and is independent of `Status` — a call routed around the active endpoint on
capability grounds does not move it.

## 8. Errors

When every candidate fails, the result is a `*composes.MultiError` carrying one `EndpointError{Alias, Err}` per
**endpoint**, in the order they were tried. It implements `Unwrap() []error`, so `errors.Is` / `errors.As` match **any** backend's
failure — including through to that provider's own error type:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests { /* every backend is rate-limited */ }
```

Declaring that interface locally, rather than importing a provider's error type, is the pattern the core itself
uses. When there was nothing to try at all — every relevant endpoint dead and none recovered — the result is
`composes.ErrNoActiveModels` instead. Two further sentinels never reach a backend at all:
`composes.ErrCapabilityNotSatisfied` (wrapped by `*CapabilityError`) and `composes.ErrCallInProgress`.

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
