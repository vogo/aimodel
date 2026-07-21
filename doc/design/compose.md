# Multi-Model Composition

- **Implementation**: `composes/`

`ComposeClient` also implements `aimodel.ChatCompleter`, so it is **nestable** — one compose client can be a member of another.

```go
cc, err := composes.NewComposeClient(composes.StrategyFailover, []composes.ModelEntry{
    {Name: "gpt-4o",        Client: openaiClient,    Weight: 3},
    {Name: "claude-opus-4", Client: anthropicClient, Weight: 1},
}, composes.WithRecoveryInterval(30*time.Second))
```

A non-empty `ModelEntry.Name` overrides the request's `Model` (the request is copied by value first, so the caller's object is untouched); an empty one uses the underlying client's default model. Construction validates that the entry list is non-empty and every `Client` is non-nil.

---

## 1. Selection strategies

| Strategy | Behavior |
|---|---|
| `StrategyFailover` (default) | Return every **healthy** model in declaration order |
| `StrategyRandom` | Healthy models, shuffled |
| `StrategyWeight` | A full ordering sampled **without replacement** in proportion to weight; `Weight <= 0` counts as 1 |

All three return an **ordered candidate list** rather than a single model — the dispatch loop tries them in turn until one succeeds, so failover applies uniformly to every strategy.

## 2. Health tracking & recovery probes

`modelHealth` records `state` (active/error), `lastError`, `errorTime`, and `errorCount`.

- Success → `markActive()`, error count reset to 0.
- Failure → `markError()`, `errorCount++`.
- Recovery check `shouldProbe`: the wait is an **exponential backoff** of `interval × 2^min(errorCount-1, 6)` — at most 64× the base interval (default 60s → up to 64 minutes).

A model whose backoff has elapsed is **prepended** to the candidate list by `prependRecoveryProbes`, forming a "probe first, keep backing off on failure" self-healing loop.

## 3. Context cancellation semantics

The dispatch loop checks `ctx.Err()` before and after each attempt: **cancellation never pollutes health state**, it returns `ctx.Err()` directly. Otherwise a client-side cancel would wrongly mark healthy models as failed.

When every candidate fails it returns a `*MultiError` ([errors.md](./errors.md)); when the candidate list is empty it returns `ErrNoActiveModels`.
