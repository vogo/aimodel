# Architecture Decision Records

This index lists the architectural decisions that shape `aimodel`. ADRs record why a decision was made, its trade-offs, and its consequences; [architecture.md](./architecture.md) describes the resulting system as a whole.

Every ADR listed here is in force. Decisions that have been reversed or absorbed are not kept as documents — the record of what changed and when lives in git history and in the per-protocol change logs.

| ADR | Status | Decision |
|---|---|---|
| [0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) | Accepted | Keep the SDK a thin API wrapper |
| [0002](./adr/0002-provider-native-as-the-only-public-interface.md) | Accepted | Provider-native clients are the only public interface |
| [0003](./adr/0003-shared-routing-core-across-protocol-wrappers.md) | Accepted | A shared routing core, protocol wrappers on top |
| [0004](./adr/0004-stateful-active-endpoint-with-in-call-retry.md) | Accepted | A pool has one active endpoint, serves one conversation at a time, and retries in place before replacing it |
| [0005](./adr/0005-clock-recovery-is-provisional.md) | Accepted | Clock recovery is provisional — a recovered endpoint gets one attempt, not a retry round |

[0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) sets the scope: request building, connection management and response decoding, with resilience and observability left to the caller. One SDK call is one HTTP request — [0004](./adr/0004-stateful-active-endpoint-with-in-call-retry.md) carves out the single bounded exception, in the compose layer only.

[0002](./adr/0002-provider-native-as-the-only-public-interface.md) shapes the SDK: there is no vendor-neutral request/response model, no canonical translation and no provider registry. `provider/openai` and `provider/anthropic` are complete, mutually independent clients over their own wire protocols, and shared code is admitted only when it can be written without a semantic type crossing between them.

[0003](./adr/0003-shared-routing-core-across-protocol-wrappers.md) applies that admission rule to multi-backend dispatch, and splits it in two: `composes` is a protocol-neutral routing core — strategies, health, aliases, attribution — and `composes/openais` / `composes/anthropics` bind it to their own wire types. Routing *mechanism* is shared; a request/response model is not, and pools never mix.

[0004](./adr/0004-stateful-active-endpoint-with-in-call-retry.md) settles what that core *does*, not where its boundary lies. A pool holds one active endpoint, retried in place under an exponential policy and replaced only when judged dead, with health reduced to `available` / `dead` on a fixed recovery timer and dispatch serialised to one call at a time (a concurrent second call is rejected with `ErrCallInProgress`, not queued). The neutrality rule is untouched — the state it adds is an index, a generation and two durations.

[0005](./adr/0005-clock-recovery-is-provisional.md) amends two clauses of that model and leaves the rest standing. An endpoint the clock puts back is *unconfirmed*, so it re-enters on probation and is attempted once instead of under the retry policy; the caller's next real call is the confirmation, and a failure restarts the recover window for the price of one request. This is the alternative to an asynchronous probe, which was rejected for making the SDK originate traffic nobody asked for — and for answering the wrong question on the rate-limit failures that dominate a pool.

## Adding an ADR

Create `doc/adr/NNNN-short-title.md` from this structure: title, status, context, decision, consequences, and references. ADRs are immutable after acceptance except for clarifications; a changed decision gets a new ADR that supersedes the old one, and the superseded document is removed once nothing in force depends on it.
