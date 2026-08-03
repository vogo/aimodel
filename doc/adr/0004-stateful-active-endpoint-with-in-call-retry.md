# ADR 0004: A pool has one active endpoint, serves one conversation at a time, and retries in place before replacing it

- Status: Accepted
- Date: 2026-08-03

## Context

Composing several backends can be read two ways: as a load balancer that picks a backend per request,
or as a pool that serves one conversation from one backend and moves only when that backend fails.

A stateless dispatcher implements the first reading — it re-runs the capability filter and the strategy
on every call, produces a fresh ordered candidate list, and holds no notion of which endpoint is
currently in use, because there is none. That is not what composing several backends is for here, and
it costs three things:

- **No run of calls has a stable owner.** Anything a caller wants to keep on one backend — a warm
  cache, a rate-limit budget, a billing attribution, a conversation's continuity — has no way to be
  expressed.
- **Preference is not the same as memory.** A dispatcher that re-derives its ordering can only express
  "stay put" as a preference for one declared endpoint or as a hash of some session key. Neither is a
  memory of what actually served the last call, so a briefly failing preferred endpoint reclaims
  traffic from a healthy one as soon as its ordering is recomputed.
- **A caller's bad request is charged to the backends.** If the health model treats a malformed
  request as a reason to move on without judging the endpoint, that one request is delivered to every
  endpoint in the pool in turn, spending quota at each and producing a `MultiError` that blames the
  backends for it.

Two further facts shaped what could replace it. [ADR 0003](./0003-shared-routing-core-across-protocol-wrappers.md)
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

3. **Two health states.** An endpoint is either `available` or `dead`. A dead endpoint becomes
   available again once a fixed `recover_time` has elapsed — on the clock alone, with no probe request.

4. **Failures split two ways.** HTTP 401/403 means the endpoint's credentials do not work:
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

8. **One conversation, one call at a time.** A pool serves a single conversation, which is inherently serial:
   a turn is issued, awaited, and followed by the next. The router holds one call slot, and a second
   concurrent call is **rejected with `ErrCallInProgress`** rather than queued — it contacts no endpoint and
   changes no health state. Queueing was considered and rejected: it would convert a caller's mistake into
   unbounded latency, and it would silently make a pool look like a load balancer again, which decision 1
   exists to stop. The capability filter runs *before* the busy check, so an unservable call is reported as
   unservable rather than as a busy pool, and `Stats()` uses a different lock so it answers throughout.

   The serialisation covers a streaming call **up to establishment**, not for the stream's lifetime — see the
   consequences below.

The pool's timing is configured at construction: `WithRetryPolicy` sets the retry base and count,
`WithRecoverTime` sets how long an endpoint stays dead. `EndpointStat` reports `Active` alongside a
`Status` of `available` / `dead`.

### Why this does not breach ADR 0003

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

- **A pool is not a load balancer.** `StrategyRandom` and `StrategyWeight` do not spread requests; they draw
  once, when the pool needs an endpoint. Callers wanting per-request distribution are not served by this
  package, and that is not a gap to be closed later — it is the trade this ADR makes.

- **A call can block for seconds.** The default policy (`500ms`, 3 retries) waits up to 3.5s before failing
  over. Latency-sensitive callers should lower it; the worst case is documented on `WithRetryPolicy` rather
  than left to be discovered.

- **A pool-wide credential outage fails fast and stays failed.** If every endpoint answers 401, one call marks
  the whole pool dead and subsequent calls return `ErrNoActiveModels` until `recover_time` elapses. This is
  chosen, not incidental: retrying or rotating through endpoints with bad credentials only multiplies the
  rejections.

- **429 gets no special treatment.** A rate-limited endpoint recovers on the same `recover_time` as any
  other dead one. Pools that lean on rate-limit rotation should set `recover_time` from their providers'
  limits.

- **A malformed request does not tour the pool.** It is retried against one endpoint before that endpoint
  is judged — which spends quota there — but it is never delivered to every backend, and the `MultiError`
  names the endpoints actually tried, one entry each.

- **`Stats()` has two independent axes.** `Active` says which endpoint currently serves the pool;
  `Status` says whether an endpoint is `available` or `dead`.

- **Concurrency is a stated guarantee, not an emergent one.** Active-endpoint transitions are
  generation-conditioned under a single lock, so concurrent failures of one endpoint commit exactly one
  switch. No lock is held across a network attempt or a retry wait. A `-race` test asserts the switch count.

- **A pool is single-occupancy, and says so.** Throughput per pool is one request. Callers wanting parallelism
  build one pool per conversation — cheap, and each keeps its own active endpoint and health, which is what
  they wanted anyway: sharing a pool would share the active endpoint too. A caller that fans out concurrent
  calls on one client gets `ErrCallInProgress` instead of parallelism. That is a loud
  failure by design; the alternative, queueing, would have turned it into a silent latency cliff behind a
  retry round of up to `base × (2^maxRetries − 1)`.

- **A streaming call holds the pool only until the stream is established.** Holding it until the caller closed
  the stream would be the stricter reading of "one call at a time", and it is not implemented, because it
  cannot be done within this ADR's other constraints. `*openai.ChatCompletionStream` and
  `*anthropic.MessageStream` expose no completion hook — `Close` merely closes a private body — so the only
  ways to learn that a stream ended are to add a hook to the provider packages (which would make them carry
  the compose layer's concurrency semantics, against ADR 0002's isolation rule) or to return a wrapper type
  from `…Stream` (which would break the method sets that let pools nest). Both were rejected.

  The gap is narrow in the usage this ADR targets: one conversation reads its stream before issuing the next
  turn, so nothing overlaps in practice. And the unimplemented option carries its own hazard — a caller who
  abandons a stream without closing it would hold the slot for the pool's lifetime, turning a leaked stream
  into a pool that answers `ErrCallInProgress` forever.

## References

- [Multi-backend composition](../design/compose.md)
- [ADR 0001 — keep the SDK a thin wrapper](./0001-keep-the-sdk-a-thin-wrapper.md)
- [ADR 0003 — a shared routing core, protocol wrappers on top](./0003-shared-routing-core-across-protocol-wrappers.md)
