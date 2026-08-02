# Multi-Backend Composition

- **Implementation**: `composes/`

`composes` dispatches one request across several backends **of one wire format** — OpenAI-compatible — with failover, health tracking and recovery probes.

It is deliberately not a vendor-neutral package. Dispatching across protocols would need a request model both protocols agree on, which is the abstraction this SDK removed ([ADR 0007](../adr/0007-provider-native-as-the-only-public-interface.md)). Composing Anthropic backends means the same loop written against `*anthropic.Client` — a couple of hundred lines, and keeping it separate is what stops a shared message model from growing back.

```go
cc, err := composes.NewComposeClient(composes.StrategyFailover, []composes.ModelEntry{
    {Name: "gpt-4o",       Client: openai.NewClient(openaiKey), Weight: 3},
    {Name: "qwen3.7-plus", Client: openai.NewClient(qwenKey, openai.WithBaseURL(qwenURL)), Weight: 1},
}, composes.WithRecoveryInterval(30*time.Second))

response, err := cc.ChatCompletions(ctx, &openai.ChatCompletionRequest{ /* … */ })
```

`ModelEntry.Client` is `composes.ChatCompleter`, a two-method interface satisfied by `*openai.Client` as-is — and by `*ComposeClient` itself, so pools nest (a fast pool that falls back to a cheaper one). Construction validates that the entry list is non-empty and every `Client` is non-nil.

A non-empty `ModelEntry.Name` overrides the request's `Model`; an empty one leaves the request's own model in place. The request is copied by value first, so the caller's object is untouched.

---

## 1. Selection strategies

| Strategy | Behavior |
|---|---|
| `StrategyFailover` (default) | Return every **healthy** backend in declaration order |
| `StrategyRandom` | Healthy backends, shuffled |
| `StrategyWeight` | A full ordering sampled **without replacement** in proportion to weight; `Weight <= 0` counts as 1 |

All three return an **ordered candidate list** rather than a single backend — the dispatch loop tries them in turn until one succeeds, so failover applies uniformly to every strategy.

For streaming, only the call that opens the stream is covered: once a backend has started streaming, a mid-stream error reaches the caller rather than triggering a retry elsewhere.

## 2. Health tracking & recovery probes

`modelHealth` records `state` (active/error), `lastError`, `errorTime` and `errorCount`.

- Success → `markActive()`, error count reset to 0.
- Failure → `markError()`, `errorCount++`.
- Recovery check `shouldProbe`: the wait is an **exponential backoff** of `interval × 2^min(errorCount-1, 6)` — at most 64× the base interval (default 60s → up to 64 minutes).

A backend whose backoff has elapsed is **prepended** to the candidate list by `prependRecoveryProbes`, forming a "probe first, keep backing off on failure" self-healing loop.

Any error marks the backend. This package does not classify failures by status code: a 4xx from one backend is as good a reason to try the next as a 5xx, and guessing which errors are retryable is exactly the kind of policy a thin wrapper leaves to its caller.

## 3. Context cancellation semantics

The dispatch loop checks `ctx.Err()` before and after each attempt: **cancellation never pollutes health state**, it returns `ctx.Err()` directly. Otherwise a client-side cancel would wrongly mark healthy backends as failed.

## 4. Errors

When every candidate fails, the result is a `*MultiError` carrying one `ModelError` per backend. It implements `Unwrap() []error`, so `errors.Is` / `errors.As` match **any** backend's failure — including through to the provider's own error type:

```go
type statusCoder interface{ StatusCode() int }

var sc statusCoder
if errors.As(err, &sc) && sc.StatusCode() == http.StatusTooManyRequests { /* every backend is rate-limited */ }
```

That is how status codes stay reachable without this package importing a provider's error type.

When the candidate list is empty — every backend unhealthy and none due for a probe — the result is `ErrNoActiveModels`.
