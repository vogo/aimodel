# ADR 0005: Clock recovery is provisional — a recovered endpoint gets one attempt, not a retry round

- Status: Accepted
- Date: 2026-08-03

## Context

Two of [ADR 0004](./0004-stateful-active-endpoint-with-in-call-retry.md)'s decisions compose badly at exactly
one point. Decision 2 gives every endpoint a call is dispatched to a full retry round — `1 + maxRetries` attempts
and up to `base × (2^maxRetries − 1)` of synchronous wait. Decision 3 restores a dead endpoint to plain
availability once `recover_time` has elapsed, on the clock alone.

So an endpoint whose recover window has just run out is indistinguishable, to the code that picks it, from one
that answered successfully a second ago — even though nothing has happened in between except time passing. If
it is still down, the caller pays the whole round (four attempts and 3.5s at the defaults) to rediscover the
fact the recover window already failed to establish, and only then does the walk continue to the next candidate.

The cost lands narrowly, which is what makes it worth fixing cheaply rather than elaborately: decision 5 keeps a
recovered endpoint from displacing a healthy incumbent, so a clock-recovered endpoint is only ever reached when
the pool has no confirmed candidate ahead of it in the ordering. That is precisely the moment a caller is
already paying for a failover, and the one where added latency is least affordable.

### The rejected alternative: an asynchronous probe

The obvious fix is a ping-pong probe: when `recover_time` elapses, the pool issues a small request of its own in
the background and promotes the endpoint only if it answers. It was rejected on four counts.

- **It makes the SDK originate traffic.** ADR 0001 keeps this SDK a thin wrapper, and ADR 0004's carve-out is
  for *retrying a request the caller made* — a cost the caller can see and bound. A probe is a request nobody
  asked for: billed, quota-consuming, and appearing in the caller's own logs and dashboards.
- **It needs a lifecycle this module does not have.** `Router` and both wrappers hold no goroutine and expose no
  `Close`. A background prober would either add one — a contract change where forgetting to call it leaks — or
  fire lazily off the dispatch path, which is where the caller's real request already is.
- **It would widen the neutral core.** [ADR 0003](./0003-shared-routing-core-across-protocol-wrappers.md) forbids
  the core from building a request, so the probe body would have to be a wrapper-injected closure: a new option
  on `composes` and a synthetic request in each wrapper, maintained forever against two evolving protocols.
- **Its evidence is weaker than it looks.** A minimal probe is a good signal for credential rejection and for
  connectivity, and a poor one for the failure that actually dominates a pool: rate limiting and quota, where a
  trivial request passes and the caller's real one is still refused. A probe would upgrade an unconditional
  false positive into a *validated-looking* false positive, which is worse to operate against.

The caller's next real call is a better probe than any probe: it is the exact workload in question, it costs
nothing extra, and it arrives exactly when the answer is needed. What was missing was not a probe — it was the
admission that the clock proves nothing, and a cheaper price for finding out.

## Decision

**An endpoint restored by the clock re-enters rotation on probation, and the retry policy does not apply to it.**

1. **Probation is derived, never stored.** Health still stores exactly `available` and `dead`; probation is
   `dead` plus an elapsed recover window, computed at read time. The failure that took the endpoint out stays on
   record throughout, and no new transition is introduced.

2. **One attempt, no waits.** An attempt against an endpoint on probation treats `maxRetries` as zero. The
   credential fast path is unchanged and simply coincides with it.

3. **A successful call is the only promotion.** It marks the endpoint available and clears its failure
   accounting, exactly as any success does. There is no separate confirmation step and no synthetic request.

4. **A failed attempt restarts the recover window.** The endpoint is judged dead again with a fresh
   `errorTime`, and the call continues to the next candidate immediately rather than after a retry round.

5. **`Stats()` reports `probation`.** A dead endpoint whose window has elapsed no longer reports `available`:
   an operator can distinguish "confirmed working" from "back on the clock, unverified".

6. **Nothing else moves.** Recovery is still clock-only with no probe request. Candidacy rules, the capability
   filter, the active endpoint and its generation, the credential classification and the construction-time
   timing validation are all untouched.

This amends decisions 2 and 3 of ADR 0004 — the retry round is no longer unconditional, and clock recovery no
longer restores plain availability. Every other decision in that ADR stands as written.

## Consequences

- **The failover path gets cheaper by the ratio of the retry policy.** At the defaults a still-dead recovered
  endpoint costs the caller one attempt instead of four, and none of the 3.5s of wait. What remains is one
  request's worth of latency, which is the irreducible price of using the real call as the evidence.

- **The evidence is the caller's actual request, so a failure spends it.** If the confirming call is large, its
  input cost is spent at a backend that could not serve it. That was already true — it is one request now
  instead of `1 + maxRetries`.

- **`ErrorCount` climbs on a long outage.** A backend down for an hour under a 60s recover time accumulates
  roughly one count per window instead of one per full round. Each is a genuine judgement rather than noise, but
  code that reads `ErrorCount` as "distinct incidents" should read it as "times judged dead".

- **There is still no health backoff.** A permanently dead endpoint is retried once per `recover_time`, forever,
  at constant cost. Exponential health backoff was considered and left out: ADR 0004's single recovery timer is
  the whole model, and one attempt per window is cheap enough not to buy complexity with.

- **`Status` gained a third value.** Consumers that treated `Status == "available"` as "selectable" must now
  accept `"probation"` as well. `EndpointStat` is unchanged structurally; only the string domain widened — and
  `StatusAvailable` / `StatusDead` / `StatusProbation` are exported alongside it, so the next widening is a
  compile-time question rather than a literal to grep for.

- **No endpoint can tell.** Probation changes no request, adds no option and reaches no wire — it is entirely a
  fact about how many attempts the router is willing to spend.

## References

- [Multi-backend composition](../design/compose.md)
- [ADR 0001 — keep the SDK a thin wrapper](./0001-keep-the-sdk-a-thin-wrapper.md)
- [ADR 0003 — a shared routing core, protocol wrappers on top](./0003-shared-routing-core-across-protocol-wrappers.md)
- [ADR 0004 — one active endpoint, in-call retry](./0004-stateful-active-endpoint-with-in-call-retry.md)
