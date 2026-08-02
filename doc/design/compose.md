# Multi-Backend Composition

- **Implementation**: `composes/`

`composes` dispatches one request across several backends **of one wire format** — OpenAI-compatible — with failover, health tracking, recovery probes, capability filtering, and economic routing.

It is deliberately not a vendor-neutral package. Dispatching across protocols would need a request model both protocols agree on, which is the abstraction this SDK removed ([ADR 0007](../adr/0007-provider-native-as-the-only-public-interface.md)). Composing Anthropic backends means the same loop written against `*anthropic.Client` — a couple of hundred lines, and keeping it separate is what stops a shared message model from growing back.

> **Note for readers of earlier versions.** Up to v0.6.0 the endpoint spec carried a `Provider` field and one compose client could mix OpenAI and Anthropic endpoints, because dispatch went through the canonical request model. That model is gone, and so is cross-protocol mixing: every endpoint here is OpenAI-compatible, and the `Provider` field was removed. The operational machinery — aliases, capability filtering, sticky routing, cost/latency strategies, cooling, observers, health snapshots — is unchanged and now runs over `provider/openai` types.

There are two construction entry points:

- **Manual** — `NewComposeClient(strategy, []ModelEntry{...})` with hand-built `composes.ChatCompleter` values. Use this when you need provider-private options or a custom `ChatCompleter`.
- **Declarative** — `NewFromEndpoints(strategy, []EndpointSpec{...})` for the common case of *same model, different BaseURL + key* (multi-key aggregation, canary migration, region/tier sharding). It builds one independent `openai.Client` per spec, so N endpoints no longer need N copies of construction code.

```go
// Manual entry — full control over each client.
cc, err := composes.NewComposeClient(composes.StrategyFailover, []composes.ModelEntry{
    {Name: "gpt-4o",       Client: openai.NewClient(openaiKey), Weight: 3},
    {Name: "qwen3.7-plus", Client: openai.NewClient(qwenKey, openai.WithBaseURL(qwenURL)), Weight: 1},
}, composes.WithRecoveryInterval(30*time.Second))

// Declarative entry — two endpoints, canary split.
cc, err := composes.NewFromEndpoints(composes.StrategyWeight, []composes.EndpointSpec{
    {Alias: "stable", BaseURL: stableURL, APIKey: stableKey, Model: "gpt-4o", Weight: 9},
    {Alias: "canary", BaseURL: canaryURL, APIKey: canaryKey, Model: "gpt-4o", Weight: 1},
})

response, err := cc.ChatCompletions(ctx, &openai.ChatCompletionRequest{ /* … */ })
```

`ModelEntry.Client` is `composes.ChatCompleter`, a two-method interface satisfied by `*openai.Client` as-is — and by `*ComposeClient` itself, so pools nest (a fast pool that falls back to a cheaper one). Construction validates that the entry list is non-empty and every `Client` is non-nil. The native client performs no construction-time validation, so an empty key or base URL surfaces at request time, not at construction.

A non-empty `ModelEntry.Name` overrides the request's `Model`; an empty one leaves the request's own model in place. The request is copied by value first, so the caller's object is untouched.

## Alias and model are distinct

Every endpoint carries two identities that must not be confused:

| Field | Meaning |
|---|---|
| `Name` | the **model** name sent to the backend in `ChatCompletionRequest.Model`. |
| `Alias` | the endpoint's **operational identity**, used for health snapshots, sticky routing, and error attribution. |

`Alias` is required and unique on the declarative path (errors name the offending position). On the manual path an empty alias is derived — from `Name` when free, otherwise `entry-<index>` — so every entry is always stably addressable; explicit aliases must still be unique. The alias is what lets several endpoints for the *same* model be told apart in errors and health snapshots.

---

## 1. Routing order

Every request flows through the same chain:

```
capability filter → strategy ordering → recovery probes → per-endpoint attempts
```

1. **Capability filter** (§4) drops endpoints that have *declared* they cannot serve the request (e.g. a tools request drops endpoints declaring `Tools: false`). Endpoints that declare no capability are never filtered. Runs before health and strategy.
2. **Strategy ordering** (§2) orders the capable, health-available candidates.
3. **Recovery probes** (§3) prepend errored endpoints whose backoff has elapsed.
4. **Attempts** try candidates in order until one succeeds; each failure updates health (§3) and emits an `AttemptResult` (§5), then fails over.

All strategies return an **ordered candidate list** rather than a single endpoint — the dispatch loop tries them in turn, so failover applies uniformly to every strategy. When the candidate list is empty it returns `ErrNoActiveModels`; when every candidate fails it returns a composes-owned `*MultiError` of `EndpointError` (in attempt order).

For streaming, only the call that opens the stream is covered: once a backend has started streaming, a mid-stream error reaches the caller rather than triggering a retry elsewhere.

## 2. Selection strategies

| Strategy | Behavior |
|---|---|
| `StrategyFailover` (default) | Healthy endpoints in declaration order |
| `StrategyRandom` | Healthy endpoints, shuffled |
| `StrategyWeight` | A full ordering sampled **without replacement** in proportion to weight; `Weight <= 0` counts as 1 |
| `StrategySticky` | Pins a session to a stable endpoint (see below); non-default |
| `StrategyCost` | Ascending static cost from `EndpointCost`; unpriced endpoints sort last; alias tie-break |
| `StrategyLatency` | Ascending injected `Latency`; endpoints without latency sort last; alias tie-break |

**Sticky routing.** Attach a session id with `composes.WithSessionID(ctx, id)`. The preferred alias is an FNV hash over the session id plus the sorted alias set, so it is reproducible across processes and instances and does not jitter as health changes. The preferred endpoint leads when available; otherwise failover proceeds in a deterministic order. Without a session id, sticky falls back to the configured fallback strategy (`WithStickyFallback`, default `StrategyFailover`) — never a randomly generated affinity key.

**Economic routing.** `StrategyCost`/`StrategyLatency` read static metadata on the entry (`EndpointCost{InputPrice, OutputPrice}`, `Latency`). Missing metadata is never treated as zero/cheapest — it sorts *after* endpoints that carry data. Both strategies still honour capability filtering, health skipping, and per-candidate failover. Dynamic pricing is out of scope.

## 3. Health tracking, cooling & recovery probes

`modelHealth` records `state`, `lastError`, `errorTime`, and `errorCount`. There are three states:

| State | Entered by | Selection behavior |
|---|---|---|
| `active` | success (or start) | always selectable |
| `cooling` | HTTP **429** | skipped until the cooling interval elapses, then rejoins regular rotation |
| `error` | HTTP **5xx** / transport failure | skipped until an exponential-backoff recovery probe is due |

Failure classification (`classifyHealth`) reads the status code through the structural `interface{ StatusCode() int }` — so it works with the provider's error type without importing it:

- **429** → cooling (does **not** advance the failure count);
- **5xx**, or no status (transport failure, or an error embedded in a body the HTTP layer accepted) → error;
- **other 4xx** → request failure: attributed and failed over, but the endpoint stays healthy.

A backend whose backoff has elapsed is **prepended** to the candidate list by `prependRecoveryProbes`, forming a "probe first, keep backing off on failure" self-healing loop. The backoff is `interval × 2^min(errorCount-1, 6)` — capped at 64× the base interval.

## 4. Capability filtering

`Capability{Tools, Vision, MaxContextTokens}` is a strong-typed contract an endpoint exposes to the router. It is declared on the entry (`ModelEntry.Capability` / `EndpointSpec.Capability`) or, when a client implements `CapabilityProvider`, by the client itself (`ComposeCapability()`). An explicit entry declaration always wins.

Filtering is **opt-in**: an endpoint that declares nothing is *unknown*, not *incapable*, and is never excluded. Only an explicit declaration can drop an endpoint. The router excludes incapable endpoints but never strips tools, rewrites the request, or downgrades. A request whose required capabilities no endpoint satisfies fails fast with `*CapabilityError` (matching `ErrCapabilityNotSatisfied`) before any network I/O.

A request requires **tools** when it defines any, or sets a `tool_choice` other than `none`; it requires **vision** when any message carries an image part.

## 5. Observers & health snapshots

`WithAttemptObserver(fn)` registers a callback invoked when each endpoint attempt finishes, with an `AttemptResult{Alias, Success, Err, Stream}`. For streaming calls it fires when the stream is established or fails to establish — post-establishment SSE errors are surfaced by the stream itself, not re-reported. Observers run synchronously on the request path under no internal lock; they must be fast and non-blocking.

`Stats()` returns an immutable `[]EndpointStat{Alias, Status, ErrorCount, LastError, ErrorTime}` snapshot, safe to call concurrently with dispatch. Status reflects routing behavior: a cooling endpoint whose interval has elapsed is reported `active`.

## 6. Errors

When every candidate fails, the result is a `*MultiError` carrying one `EndpointError{Alias, Err}` per attempt, in order. It implements `Unwrap() []error`, so `errors.Is` / `errors.As` match **any** backend's failure — including through to the provider's own error type:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests { /* every backend is rate-limited */ }
```

When the candidate list is empty — every endpoint cooling or errored and none due for a probe — the result is `ErrNoActiveModels`.
