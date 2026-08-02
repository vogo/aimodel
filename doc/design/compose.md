# Multi-Model Composition

- **Implementation**: `composes/`

`ComposeClient` also implements `aimodel.ChatCompleter`, so it is **nestable** — one compose client can be a member of another.

There are two construction entry points:

- **Manual** — `NewComposeClient(strategy, []ModelEntry{...})` with hand-built `aimodel.ChatCompleter` values. Use this when you need provider-private options (`WithProviderOptions`) or a custom `ChatCompleter`.
- **Declarative** — `NewFromEndpoints(strategy, []EndpointSpec{...})` for the common case of *same protocol, same model, different BaseURL + key* (multi-key aggregation, canary migration, region/tier sharding). It builds one independent `aimodel.Client` per spec, so N endpoints no longer need N copies of construction code.

```go
// Manual entry — full control over each client.
cc, err := composes.NewComposeClient(composes.StrategyFailover, []composes.ModelEntry{
    {Name: "gpt-4o",        Client: openaiClient,    Weight: 3},
    {Name: "claude-opus-4", Client: anthropicClient, Weight: 1},
}, composes.WithRecoveryInterval(30*time.Second))

// Declarative entry — same provider, two endpoints, canary split.
cc, err := composes.NewFromEndpoints(composes.StrategyWeight, []composes.EndpointSpec{
    {Alias: "stable", BaseURL: stableURL, APIKey: stableKey, Model: "gpt-4o", Weight: 9},
    {Alias: "canary", BaseURL: canaryURL, APIKey: canaryKey, Model: "gpt-4o", Weight: 1},
})
```

A non-empty `ModelEntry.Name` overrides the request's `Model` (the request is copied by value first, so the caller's object is untouched); an empty one uses the underlying client's default model. Construction validates that the entry list is non-empty and every `Client` is non-nil.

## Alias, provider, and model are distinct

Every endpoint carries three identities that must not be confused:

| Field | Meaning |
|---|---|
| `Provider` (EndpointSpec) | the **protocol** resolved from the registry (e.g. `openai.Name`, `anthropic.Name`). Multiple endpoints share a provider; they never forge or duplicate a registry name. |
| `Name` | the **model** name sent to the backend in `ChatRequest.Model`. |
| `Alias` | the endpoint's **operational identity**, used for health snapshots, sticky routing, and error attribution. |

`Alias` is required and unique on the declarative path (errors name the offending position). On the manual path an empty alias is derived — from `Name` when free, otherwise `entry-<index>` — so every entry is always stably addressable; explicit aliases must still be unique. `EndpointSpec`/`Alias`/`EndpointError` live entirely in `composes` — nothing here enters the canonical `ais` types.

---

## 1. Routing order

Every request flows through the same chain:

```
capability filter → strategy ordering → recovery probes → per-endpoint attempts
```

1. **Capability filter** (§4) drops endpoints that cannot serve the request (e.g. a tools request keeps only `Tools`-capable endpoints). Runs before health and strategy.
2. **Strategy ordering** (§2) orders the capable, health-available candidates.
3. **Recovery probes** (§3) prepend errored endpoints whose backoff has elapsed.
4. **Attempts** try candidates in order until one succeeds; each failure updates health (§3) and emits an `AttemptResult` (§5), then fails over.

All strategies return an **ordered candidate list** rather than a single model — the dispatch loop tries them in turn, so failover applies uniformly to every strategy. When the candidate list is empty it returns `ErrNoActiveModels`; when every candidate fails it returns a composes-owned `*MultiError` of `EndpointError` (in attempt order).

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

Failure classification (`classifyHealth`):

- **429** → `cooling`. Recorded with the error and time, but it does **not** advance the consecutive-failure count, so rate limiting never drives the error backoff.
- **5xx** (or no `APIError`, e.g. a transport failure; or status 0 from a stream error) → `error`, `errorCount++`.
- **other 4xx** → attributed and failed over, but the endpoint is **not** judged unhealthy (it stays `active`).
- **context cancellation** and request validation/capability mismatch never change health.

Recovery:

- Success → `markActive()`: state `active`, error count reset to 0, last error cleared.
- `error` recovery probe `shouldProbe`: the wait is an **exponential backoff** of `interval × 2^min(errorCount-1, 6)` — at most 64× the base interval (default 60s → up to 64 minutes). A probed endpoint is **prepended** to the candidate list, forming a "probe first, keep backing off on failure" self-healing loop.
- `cooling` rejoin uses a fixed `coolingInterval` (default 10s, shorter than the recovery interval; set via `WithCoolingInterval`) — no half-open or sliding-window machinery.

`Stats()` returns an immutable per-endpoint snapshot (`[]EndpointStat`: `Alias`, `Status`, `ErrorCount`, `LastError`, `ErrorTime`), safe to read concurrently with dispatch. Zero time / nil error means "never failed or recovered". Mutating the returned slice does not affect internal state.

## 4. Capability filtering

`Capability{Tools, Vision, MaxContextTokens}` is a strong-typed contract declared per endpoint (`ModelEntry.Capability` / `EndpointSpec.Capability`) or by a client implementing `CapabilityProvider`. It is never inferred from `Tags` or the provider name.

Filtering runs first and only **excludes** candidates:

- A request with `Tools` (or an explicit `tool_choice` other than `"none"`) keeps only `Tools`-capable endpoints.
- A request carrying an image content part keeps only `Vision`-capable endpoints.
- `MaxContextTokens` is part of the contract for future budget routing; the SDK does not estimate tokens, so it is not filtered today.

If a request requires a capability no endpoint has, `composes` returns a `*CapabilityError` (matching `ErrCapabilityNotSatisfied`) **before any network I/O**. Filtering never strips tools, rewrites the request, switches dialect, or downgrades to plain chat. Missing capability metadata is treated conservatively as "not supported".

## 5. Attempt observation

`WithAttemptObserver(func(composes.AttemptResult))` registers a callback fired when each attempt finishes — for non-streaming calls when the call returns, for streaming calls when the stream is established or fails to establish. Post-establishment SSE errors are surfaced by the `Stream` itself, not re-reported. `AttemptResult` carries the endpoint `Alias`, `Success`, `Err`, and a `Stream` flag. Observers run synchronously on the request path under no internal lock: they must be fast and must not call blocking `ComposeClient` operations.

Per-endpoint failures are wrapped in `EndpointError{Alias, Err}` (unwraps to the underlying error). When every candidate fails, the composes-owned `*MultiError` preserves each `EndpointError` in attempt order — same-model endpoints are distinguished by alias, not by `ais.ModelError`. It implements Go 1.20+ `Unwrap() []error`, so `errors.Is`/`errors.As` reach any underlying error.

## 6. Context cancellation semantics

The dispatch loop checks `ctx.Err()` before and after each attempt: **cancellation never pollutes health state**, it returns `ctx.Err()` directly. Otherwise a client-side cancel would wrongly mark healthy endpoints as failed. If an attempt was already made, its `AttemptResult` still names the alias.
