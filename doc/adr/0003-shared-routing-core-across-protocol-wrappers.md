# ADR 0003: A shared routing core, protocol wrappers on top

- Status: Accepted
- Date: 2026-08-03
- Refines: [ADR 0002](./0002-provider-native-as-the-only-public-interface.md) §3 — it applies the
  neutrality test to multi-backend routing. Everything else in ADR 0002 stands unchanged.
- Unaffected: [ADR 0001](./0001-keep-the-sdk-a-thin-wrapper.md)

## Context

[ADR 0002](./0002-provider-native-as-the-only-public-interface.md) leaves each protocol with a
complete, independent client and forbids a shared request/response model. Multi-backend dispatch
then raises the question of how far that prohibition reaches: if `composes` may not name a request,
may it still hold the machinery that decides *which backend* serves one?

The narrow reading — every protocol writes its own dispatch loop — was rejected.

1. **The operational machinery is not protocol-shaped.** What `composes` actually contains is
   strategy ordering, a health state machine, recovery timing, alias identity and uniqueness,
   candidate construction, attempt observation, `Stats()` snapshots and `MultiError` attribution.
   None of it reads a message, a tool definition or a usage block. The only protocol-shaped code is
   the type of one parameter and two small predicates ("does this request need tools / vision").

2. **The narrow reading costs three copies.** `provider/openai` has both Chat Completions and
   Responses, and `provider/anthropic` has Messages. Writing the loop per protocol means the same
   state machine, the same timing arithmetic and the same strategy ordering written three times —
   each evolving, breaking and being tested separately. ADR 0002 accepts duplication as the price of
   removing the shared model, and that price is right for SSE scanning and error parsing, which are
   genuinely per-protocol. A health state machine is not.

3. **The neutrality test already draws the right line, one layer up.** ADR 0002 §3 admits an
   enhancement outside a provider package *if and only if* it introduces no semantic data type that
   another provider imports. A shared dispatch loop introduces no such type: nothing crosses between
   `provider/openai` and `provider/anthropic`, and neither imports the other or the core. The
   temptation to forbid it comes from the loop and the request type being one piece of code —
   separate them and the test passes cleanly.

The failure mode ADR 0002 exists to prevent is a shared *request/response model* growing back. That
risk is unchanged by this decision, and the guards below are what keep it unchanged.

## Decision

**Routing mechanism may be shared across protocols. Protocol semantics may not.**

1. **`composes` is a protocol-neutral routing core.** It owns endpoint identity (aliases, weights,
   tags), the selection strategies, the health state machine, the active endpoint, retries, attempt
   observation, health snapshots and failure attribution. It imports no provider, declares no
   protocol concept, and exposes no request or response type.

2. **Each protocol gets a wrapper package.** `composes/openais` binds the core to the OpenAI wire —
   both Chat Completions and Responses — and `composes/anthropics` binds it to Anthropic Messages.
   A wrapper owns its entry types, its client interfaces, its capability declarations and the
   per-endpoint request copy. The wrappers import neither each other nor the other's provider.

3. **The core and a wrapper meet at one seam.** `Dispatch[T](ctx, router, call, attempt)` runs one
   full dispatch; `attempt(ctx, endpointIndex)` is the wrapper's closure and is the only place a
   request exists. The core's inputs are a `Call` of opaque strings (`Requires`), endpoint indices
   (`Eligible`), a scalar (`OutputUnits`) and a flag (`Stream`). No request, response or stream type
   appears in the core's API, generically or otherwise.

4. **Capability declarations are opaque labels.** An endpoint declares `[]string`; a call requires
   `[]string`; the core performs set containment and attaches no meaning to either side. `nil` means
   *undeclared* (unknown, never filtered) and a non-nil empty declaration means "serves nothing",
   which is why the core exposes `Declare(...)` to make the difference explicit.

5. **Pools do not mix, and there is no cross-protocol failover.** An OpenAI pool and an Anthropic
   pool are separate routers with separate health. A caller who needs a request to fail over from
   one protocol to the other still writes that mapping, with full knowledge of both protocols. This
   is the same conclusion ADR 0002 reaches; the reason simply narrows, from "there is no shared
   anything" to "there is no shared *request*".

### The neutrality test, sharpened

ADR 0002 §3 is unchanged in wording and now has an explicit reading for the mechanism/semantics
distinction:

> An enhancement may live outside a provider package **if and only if** it can be implemented
> without introducing a semantic data type that another provider imports.

- **A shared type that carries protocol content** — a request, a response, a message, a tool
  definition, a usage block, or a mapping between any of them — is a canonical layer. Rejected.
- **A shared mechanism whose whole interface is indices, opaque strings, scalars and closures** is
  not. Admitted.

The operational question to ask of any candidate: *if I add a field to this shared type, does a
provider package have to learn about it?* For `Endpoint`, `Call`, `AttemptResult` and `EndpointStat`
the answer is no — no provider imports `composes`, and none ever will.

### What this gives up

`composes` is not usable on its own: an OpenAI-wire pool is built through `composes/openais`, an
Anthropic pool through `composes/anthropics`. The routing package deliberately offers no convenience
entry point of its own, because such an entry point would be a wire-format symbol living in the
package this decision exists to keep neutral.

Two smaller costs, recorded so they are not mistaken for oversights:

- **The dispatch loop is one level more abstract** than direct per-protocol code: a generic function
  plus a closure instead of an inline call. The review standard that justifies it is "one state
  machine, one dispatch loop, one set of tests"; a wrapper that starts reimplementing health or
  ordering has broken the arrangement, whatever it looks like locally.
- **Capability label constants live in each wrapper, not centrally.** Centralising the strings would
  require the neutral core to declare `CapabilityTools` — an identifier the AST guard rejects, and
  rightly: naming the label is what makes a package protocol-aware. `openais.CapabilityTools` and
  `anthropics.CapabilityTools` therefore both spell `"tools"` independently. They are never
  exchanged (the pools are disjoint), so the duplication carries no coupling; the drift risk is
  accepted and covered by each wrapper's own tests.

### Guards

ADR 0002's five guards stand — with guard 4 now applying to `composes` itself — and two are added:

1. `provider/openai` and `provider/anthropic` import neither each other nor the root package.
2. No public API mentions a shared semantic package.
3. Both `*HTTPError` types satisfy `interface { StatusCode() int }` (compile-time assertion).
4. Packages declared vendor-neutral declare no protocol-semantic identifier (`message`, `content`,
   `tool`, `usage`, `chat`, `prompt`, `token`, `choice`, `completion`), checked over the AST.
   **`composes` is such a package**; the wrappers are deliberately not.
5. Every public wire type survives a marshal → unmarshal → marshal round trip unchanged.
6. **The routing core imports nothing from this module** — not one provider, not two. Importing a
   single provider would privilege one wire format; importing both would be a canonical layer with
   extra steps.
7. **The routing core's exported API references no type from this module**, checked over the AST, so
   a shared request model cannot re-enter through a public signature.

## Consequences

- **Three dispatch paths, one implementation.** OpenAI Chat Completions, OpenAI Responses and
  Anthropic Messages all get the same strategies, health handling, capability filtering, observers,
  `Stats()` and `MultiError` attribution. A fix to the routing arithmetic lands once.

- **The two layers move at different speeds.** A change to routing behaviour is a change to
  `composes` and reaches every protocol at once; a change to a wire format touches one wrapper and
  its provider. `provider/openai` and `provider/anthropic` are untouched by this arrangement — no
  wire type, client, option or serialization depends on it.

- **A new interaction form is a method set on a wrapper, not a widened one.** Responses arrived as
  `Responses` / `ResponsesStream` alongside the chat methods, per ADR 0002 §6. An entry whose client
  implements only `ChatCompleter` keeps serving chat and is simply not eligible for Responses
  routing — expressed through `Call.Eligible`, and reported as a capability error before any network
  I/O when no entry qualifies.

- **A new protocol is a new wrapper package.** It writes its entry types, its capability predicates
  and its per-endpoint request copy — a few hundred lines — and inherits every operational feature.
  It still gets no interoperability with the other pools, correctly, since there is none to give.

- **The regrowth risk moves, and is guarded where it moved to.** The tempting shortcut is no longer
  "share a request model"; it is "let the core peek at the request, just this once" — a `Call` field
  holding a message, or a type parameter constrained to something protocol-shaped. Guards 4, 6 and 7
  fail in CI on all three shapes.

## References

- [Multi-backend composition](../design/compose.md)
- [Architecture](../architecture.md)
- [ADR 0001 — keep the SDK a thin wrapper](./0001-keep-the-sdk-a-thin-wrapper.md)
- [ADR 0002 — provider-native clients are the only public interface](./0002-provider-native-as-the-only-public-interface.md)
- [ADR 0004 — one active endpoint, one call at a time, retried in place](./0004-stateful-active-endpoint-with-in-call-retry.md)
