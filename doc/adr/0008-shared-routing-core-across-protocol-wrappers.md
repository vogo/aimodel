# ADR 0008: A shared routing core, protocol wrappers on top

- Status: Accepted
- Date: 2026-08-03
- Partially supersedes: [ADR 0007](./0007-provider-native-as-the-only-public-interface.md) §4 — the
  decision that composing another protocol's backends means "an isomorphic loop written in
  `provider/anthropic` or in the caller's code". Everything else in ADR 0007 stands unchanged.
- Unaffected: [ADR 0001](./0001-keep-the-sdk-a-thin-wrapper.md)

## Context

[ADR 0007](./0007-provider-native-as-the-only-public-interface.md) removed the canonical layer and
narrowed `composes` to one wire format. Its §4 then answered the obvious follow-up question — what
about Anthropic backends? — with: write the same loop again, in that package or in your own code.

That answer was correct about the thing it was defending and wrong about the thing it prescribed.

1. **The operational machinery is not protocol-shaped.** What `composes` actually contains is
   strategy ordering, a three-state health machine with 429 cooling and exponential-backoff recovery
   probes, alias identity and uniqueness, candidate construction, attempt observation, `Stats()`
   snapshots and `MultiError` attribution. None of it reads a message, a tool definition or a usage
   block. The only protocol-shaped code was the type of one parameter and two small predicates
   ("does this request need tools / vision").

2. **§4's prescription cost three copies.** `provider/openai` had gained Responses and
   `provider/anthropic` had Messages; neither could reuse any of the above. Following §4 literally
   meant the same state machine, the same backoff arithmetic and the same strategy ordering written
   three times — chat, responses, messages — each evolving, breaking and being tested separately.
   ADR 0007 accepts duplication as "the price" of removing the shared model, and that price is right
   for SSE scanning and error parsing, which are genuinely per-protocol. A health state machine is
   not.

3. **The neutrality test already drew the right line; §4 applied it one layer too low.** ADR 0007 §3
   admits an enhancement outside a provider package *if and only if* it introduces no semantic data
   type that another provider imports. A shared candidate loop introduces no such type: nothing
   crosses between `provider/openai` and `provider/anthropic`, and neither imports the other or the
   core. §4 nonetheless forbade the sharing, because at the time the loop and the request type were
   the same piece of code and could not be judged apart.

The failure mode ADR 0007 exists to prevent is a shared *request/response model* growing back. That
risk is unchanged by this decision, and the guards below are what keep it unchanged.

## Decision

**Routing mechanism may be shared across protocols. Protocol semantics may not.**

1. **`composes` is a protocol-neutral routing core.** It owns endpoint identity (aliases, weights,
   tags), the six selection strategies, the health state machine, recovery probes, sticky session
   routing, attempt observation, health snapshots and failure attribution. It imports no provider,
   declares no protocol concept, and exposes no request or response type.

2. **Each protocol gets a wrapper package.** `composes/openais` binds the core to the OpenAI wire —
   both Chat Completions and Responses — and `composes/anthropics` binds it to Anthropic Messages.
   A wrapper owns its entry types, its client interfaces, its capability declarations and the
   per-endpoint request copy. The wrappers import neither each other nor the other's provider.

3. **The core and a wrapper meet at one seam.** `Dispatch[T](ctx, router, call, attempt)` runs one
   full candidate loop; `attempt(ctx, endpointIndex)` is the wrapper's closure and is the only place
   a request exists. The core's inputs are a `Call` of opaque strings (`Requires`), endpoint indices
   (`Eligible`), a scalar (`OutputUnits`) and a flag (`Stream`). No request, response or stream type
   appears in the core's API, generically or otherwise.

4. **Capability declarations are opaque labels.** An endpoint declares `[]string`; a call requires
   `[]string`; the core performs set containment and attaches no meaning to either side. `nil` still
   means *undeclared* (unknown, never filtered) and a non-nil empty declaration still means "serves
   nothing", which is why the core exposes `Declare(...)` to make the difference explicit.

5. **Pools do not mix, and there is no cross-protocol failover.** An OpenAI pool and an Anthropic
   pool are separate routers with separate health. A caller who needs a request to fail over from
   one protocol to the other still writes that mapping, with full knowledge of both protocols. This
   is the same conclusion ADR 0007 reached; only the reason narrows, from "there is no shared
   anything" to "there is no shared *request*".

### The neutrality test, sharpened

ADR 0007 §3 is unchanged in wording and now has an explicit reading for the mechanism/semantics
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

`composes` stops being usable on its own: every existing caller of `composes.NewComposeClient`,
`composes.ChatCompleter`, `composes.ModelEntry`, `composes.EndpointSpec` and `composes.NewFromEndpoints`
must move to `composes/openais`. There is **no deprecation window** — unlike v0.6.1 → v0.7.0, which
marked every removed symbol `Deprecated:` for one release so `staticcheck` could report call sites
first. The trade is deliberate: the package boundary lands in one step, because a transitional
alias in the root package would be an OpenAI-wire symbol living in the package this decision
exists to keep neutral.

Two smaller costs, recorded so they are not mistaken for oversights:

- **The dispatch loop is one level more abstract** than the direct code it replaces: a generic
  function plus a closure instead of an inline call. The review standard that justifies it is "one
  state machine, one candidate loop, one set of tests"; a wrapper that starts reimplementing health
  or ordering has broken the arrangement, whatever it looks like locally.
- **Capability label constants live in each wrapper, not centrally.** Centralising the strings would
  require the neutral core to declare `CapabilityTools` — an identifier the AST guard rejects, and
  rightly: naming the label is what makes a package protocol-aware. `openais.CapabilityTools` and
  `anthropics.CapabilityTools` therefore both spell `"tools"` independently. They are never
  exchanged (the pools are disjoint), so the duplication carries no coupling; the drift risk is
  accepted and covered by each wrapper's own tests.

### Guards

ADR 0007's five guards stand. Guard 4's exemption for `composes` is withdrawn, and two guards are
added:

1. `provider/openai` and `provider/anthropic` import neither each other nor the root package.
2. No public API mentions a shared semantic package.
3. Both `*HTTPError` types satisfy `interface { StatusCode() int }` (compile-time assertion).
4. Packages declared vendor-neutral declare no protocol-semantic identifier (`message`, `content`,
   `tool`, `usage`, `chat`, `prompt`, `token`, `choice`, `completion`), checked over the AST.
   **`composes` is now such a package**; the wrappers are deliberately not.
5. Every public wire type survives a marshal → unmarshal → marshal round trip unchanged.
6. **The routing core imports nothing from this module** — not one provider, not two. Importing a
   single provider would privilege one wire format; importing both would be a canonical layer with
   extra steps.
7. **The routing core's exported API references no type from this module**, checked over the AST, so
   a shared request model cannot re-enter through a public signature.

## Consequences

- **v0.8.0 is a breaking release** for `composes` callers only. `provider/openai` and
  `provider/anthropic` are untouched — no wire type, client, option or serialization changes.
  Every moved symbol keeps its name and behaviour under `composes/openais`.

- **Three dispatch paths, one implementation.** OpenAI Chat Completions, OpenAI Responses and
  Anthropic Messages all get failover, the six strategies, 429 cooling, backoff recovery probes,
  capability filtering, sticky routing, observers, `Stats()` and `MultiError` attribution. A fix to
  the backoff arithmetic now lands once.

- **A new interaction form is a method set on a wrapper, not a widened one.** Responses arrived as
  `Responses` / `ResponsesStream` alongside the chat methods, per ADR 0004's surviving
  small-interface principle. An entry whose client implements only `ChatCompleter` keeps serving
  chat and is simply not eligible for Responses routing — expressed through `Call.Eligible`, and
  reported as a capability error before any network I/O when no entry qualifies.

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
- [ADR 0004 — model capabilities with small interfaces](./0004-model-capabilities-with-small-interfaces.md)
- [ADR 0007 — provider-native clients are the only public interface](./0007-provider-native-as-the-only-public-interface.md)
