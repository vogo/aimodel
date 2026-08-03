# ADR 0009: A pool has one active endpoint, retried in place before it is replaced

- Status: Accepted
- Date: 2026-08-03

## Context

Up to v0.8.x, `composes.Dispatch` was stateless. Every call re-ran the whole chain — capability filter,
strategy ordering, recovery probes — and produced a fresh ordered candidate list; "failover" existed only
*inside* one call. The `Router` held no notion of which endpoint was currently in use, because there was none.

That is a load balancer, and it is not what composing several backends was for. The observable consequences:

- `StrategyRandom` and `StrategyWeight` could route consecutive calls to different backends, so no run of
  calls had a stable owner. Anything a caller wants to keep on one backend — a warm cache, a rate-limit
  budget, a billing attribution, a conversation's continuity — had no way to express it.
- `StrategyFailover` looked like it stuck to one endpoint, but only because it always preferred the first
  declared one. A recovery probe for that endpoint was *prepended*, so a briefly failing preferred endpoint
  reclaimed traffic from a healthy one at every backoff boundary.
- `StrategySticky` addressed a different problem than it appeared to: its anchor was `FNV(sessionID + full
  alias set)`, a hash-derived *preference* per session, not a memory of what actually served the last call. It
  needed a session id threaded through the context, and it duplicated the "stay put" intent with a second,
  conflicting source of stickiness.
- The three-state health model spent its complexity in the wrong place. A 429 got a short private cooling
  window; a 5xx got an exponential probe backoff; a non-429 4xx (a malformed request, say) left health
  untouched *and* failed over — so one bad request was delivered to every endpoint in the pool in turn,
  spending quota at each and producing a `MultiError` that blamed the backends for the caller's request.

Two further facts shaped what could replace it. [ADR 0008](./0008-shared-routing-core-across-protocol-wrappers.md)
requires that whatever state the core holds be expressible in indices, opaque labels and scalars — no request
may leak in. And [ADR 0001](./0001-keep-the-sdk-a-thin-wrapper.md) says this SDK does not do retry.

## Decision

**The router holds one active endpoint per pool, and a call is served by it.**

1. **Pool-level active endpoint.** The `Router` owns at most one active endpoint index plus a *generation*.
   The strategy runs only when there is no active endpoint or the current one has been judged dead. Successive
   successful calls therefore land on the same backend under every strategy, including the randomising ones.

2. **In-call exponential retry.** A failing attempt against the chosen endpoint is retried in place: retry *k*
   waits `base × 2^(k-1)`, up to `maxRetries` retries, so an endpoint is attempted at most `1 + maxRetries`
   times. Exhausting them is what judges an endpoint dead; the call then continues on the next capable
   available endpoint without failing.

3. **Two health states.** `available` and `dead` replace `active` / `cooling` / `error`. A dead endpoint
   becomes available again once a fixed `recover_time` has elapsed — on the clock alone, with no probe request
   and no exponential health backoff.

4. **Failures split two ways, not three.** HTTP 401/403 means the endpoint's credentials do not work:
   no retry, no wait, dead immediately. Everything else — 429, 400 and other 4xx, 5xx, transport errors —
   is retryable and only kills the endpoint once the retries are spent. Context cancellation is not a failure
   at all: it changes no health and triggers no switch.

5. **Recovery restores candidacy, never the crown.** An endpoint returning from dead rejoins the candidate
   pool and nothing more. A healthy active endpoint is never displaced by one coming back.

6. **Construction validates the timing.** `recover_time` must be strictly greater than `base × 2^maxRetries`
   (a bound that also exceeds one round's total wait, `base × (2^maxRetries − 1)`). A violation, a non-positive
   base or recover time, or a negative retry count is a construction error naming the actual and required
   values.

7. **Capability filtering outranks the active endpoint, without moving it.** The filter still runs first. When
   the active endpoint cannot serve a call, the strategy picks among the capable available endpoints *for that
   call only*; the pool's active endpoint is untouched. A pool that has no active endpoint yet may commit one
   from a capability-restricted call — a pool has to start somewhere — but an established one is never
   displaced by a rarer capability's traffic.

`StrategySticky`, `WithSessionID`, `WithStickyFallback`, `WithRecoveryInterval`, `WithCoolingInterval` and the
recovery-probe mechanism are deleted outright in v0.9.0, with no compatibility aliases — the same treatment
v0.7.0 gave the canonical layer. `WithRetryPolicy` and `WithRecoverTime` replace the two interval options;
`EndpointStat` gains `Active` and its `Status` values become `available` / `dead`.

### Why this does not breach ADR 0008

The active endpoint is an `int` index and a `uint64` generation. The retry policy is two scalars. Nothing the
core learned in this change has a protocol shape, and `Call` gained no field. The guard tests still hold: the
core imports nothing from this module and declares no protocol-semantic identifier — indeed the first draft of
this change named a type `endpointChoice` and the guard rejected it for containing "choice".

### Why this is a bounded exception to ADR 0001

ADR 0001 keeps retry out of the SDK because retry policy is application-specific and hides latency and cost.
That reasoning is about the *provider clients*, and it still binds them: `provider/openai` and
`provider/anthropic` issue exactly one HTTP request per call and gain nothing here.

The exception is confined to `composes`, which ADR 0001 already carves out as "the explicitly multi-model
compose path" — a package whose entire purpose is to decide what happens when a backend misbehaves. Judging an
endpoint dead requires evidence, and a single failed request is weak evidence; without retries the choice is
between killing an endpoint on one transient blip or never killing it at all. The cost is stated rather than
hidden: a caller pays at most `base × (2^maxRetries − 1)` of synchronous wait, the waits are interruptible by
context, and the policy is explicit at construction.

## Consequences

- **A pool is not a load balancer.** `StrategyRandom` and `StrategyWeight` no longer spread requests; they draw
  once, when the pool needs an endpoint. Callers wanting per-request distribution are not served by this
  package, and that is not a gap to be closed later — it is the trade this ADR makes.

- **A call can block for seconds.** The default policy (`500ms`, 3 retries) waits up to 3.5s before failing
  over. Latency-sensitive callers should lower it; the worst case is documented on `WithRetryPolicy` rather
  than left to be discovered.

- **A pool-wide credential outage fails fast and stays failed.** If every endpoint answers 401, one call marks
  the whole pool dead and subsequent calls return `ErrNoActiveModels` until `recover_time` elapses. This is
  chosen, not incidental: retrying or rotating through endpoints with bad credentials only multiplies the
  rejections.

- **429 loses its short private cooling window.** A rate-limited endpoint now recovers on the same
  `recover_time` as any other dead one. Pools that lean on rate-limit rotation should set `recover_time` from
  their providers' limits.

- **A malformed request no longer tours the pool.** It is still retried against one endpoint before that
  endpoint is judged, which spends more quota there than v0.8 did — but it is no longer delivered to every
  backend, and the `MultiError` names the endpoints actually tried, one entry each.

- **`Stats()` gained a second axis.** `Active` and `Status` are independent; consumers reading `Status ==
  "active"` from v0.8 must be updated, since that string no longer exists.

- **Concurrency is now a stated guarantee, not an emergent one.** Active-endpoint transitions are
  generation-conditioned under a single lock, so concurrent failures of one endpoint commit exactly one
  switch. No lock is held across a network attempt or a retry wait. A `-race` test asserts the switch count.

## References

- [Multi-backend composition](../design/compose.md)
- [ADR 0001 — keep the SDK a thin wrapper](./0001-keep-the-sdk-a-thin-wrapper.md)
- [ADR 0008 — a shared routing core, protocol wrappers on top](./0008-shared-routing-core-across-protocol-wrappers.md)
