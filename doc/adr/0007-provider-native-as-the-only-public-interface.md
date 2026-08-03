# ADR 0007: Provider-native clients are the only public interface

- Status: Accepted
- Date: 2026-08-02
- Supersedes: [ADR 0003](./0003-dispatch-providers-through-a-registry.md), [ADR 0005](./0005-canonical-shared-semantics-over-provider-native-wire.md)
- Partially supersedes: [ADR 0004](./0004-model-capabilities-with-small-interfaces.md) (the small-interface principle survives; "capability" narrows to a method set inside one provider package)
- Absorbs: [ADR 0006](./0006-responses-capability-on-provider-native-types.md) (native types stop being the single-vendor exception and become the rule)
- Unaffected: [ADR 0001](./0001-keep-the-sdk-a-thin-wrapper.md)

## Context

[ADR 0005](./0005-canonical-shared-semantics-over-provider-native-wire.md) put a provider-neutral
canonical layer (`ais`) on top of per-vendor native wire models, and [ADR
0006](./0006-responses-capability-on-provider-native-types.md) carved out the first exception to
it. Reviewing the result against actual use, the canonical layer costs more than it returns:

1. **Its one differentiating capability was never used.** A shared request model exists so the
   same request can be delivered to different protocols. At the time this analysis was written, no
   caller in this repository did that: `composes/compose_client_test.go` imported only `anthropic`,
   and all three strategies in `integrations/compose_tests` passed `WithProvider(anthropic.Name)`.
   There was no mixed OpenAI/Anthropic dispatch anywhere; the real requirement was failover across
   several backends speaking *one* wire format. (v0.6.0 later shipped a cross-protocol endpoint
   capability — see the note after "The constraint this decision gives up". It was built on exactly
   the shared model this decision removes, and it did not change the assessment: the documented
   real-world need remained same-wire failover, and the capability rode the canonical request
   rather than expressing either protocol faithfully.)

2. **It is lossy by construction, and the loss grows.** The "≥ 2 providers share a mappable
   semantic" admission rule (ADR 0005) is correct for a shared type and, precisely because it is
   correct, keeps most of each vendor's API out. Everything excluded had to travel through
   `ais.Extensions` — a `map[string]any` side channel with per-provider typed accessors, merge
   rules for streaming, and its own error type. At that point the "portable" request is a partial
   request plus a vendor-specific bag, which is not portability.

3. **It delays access to shipped features.** A field documented upstream today cannot reach
   canonical callers until a second vendor ships a mappable equivalent. The native layer already
   had it on day one; the canonical layer's rule made the wrapper slower than the API it wraps.

4. **The exception was already spreading.** ADR 0006 had to put Responses on the unified client
   using OpenAI-native types because no canonical shape could honestly be minted from one vendor.
   That is the general case, not a special one: interaction forms and fields arrive per vendor and
   converge later, if ever.

5. **The seam is unguarded.** ADR 0005 records this itself: a canonical field not wired into a
   translation is dropped silently, compiles, and returns 200. The mitigation is procedural (a
   four-way sync convention) plus a field-count sentinel that only ensures somebody is *asked*.

The alternative considered and rejected was extracting `ais` plus the translations into their own
module so existing users keep a maintained canonical API. There is no evidence of external
canonical users, and two modules cost more to maintain than the option is worth. If a concrete
need appears, the v0.5.0 snapshot can be published as a separate module later; that decision does
not block this one.

## Decision

**The public interface of `aimodel` is the set of provider-native clients. There is no
vendor-neutral request/response model, no canonical translation, and no provider registry.**

1. **Each provider package is complete and self-contained.** `provider/openai` owns Chat
   Completions and Responses — client, wire types, SSE, usage aggregation, errors and options.
   `provider/anthropic` owns Messages the same way. Neither imports the other, the root package,
   or any shared semantic layer.

2. **Fidelity replaces universality.** A provider package expresses its official API completely
   and without compromise. Nothing is withheld pending a second vendor's equivalent, and no field
   is reshaped to resemble another vendor's spelling.

3. **The neutrality test governs shared code.** An enhancement may live outside a provider package
   **if and only if** it can be implemented inside a single package without introducing a semantic
   data type that another provider imports. Anything requiring a shared request/response model,
   bidirectional field mapping, or a cross-provider decision about which fields to keep *is* a
   canonical layer being rebuilt, and is rejected. Duplicating a small amount of logic between the
   two providers is the accepted price.

4. **`composes` narrows to one wire format.** It dispatches across several OpenAI-compatible
   backends, consuming and returning `provider/openai` types, and keeps failover, random and
   weighted strategies, health tracking, recovery probes, per-entry model override and aggregated
   per-backend errors. It is explicitly *not* a vendor-neutral package. Composing Anthropic
   backends is an isomorphic loop written in `provider/anthropic` or in the caller's code — not a
   shared abstraction over both.

5. **Errors are matched structurally, not nominally.** There is no shared error type. Each
   provider's `*HTTPError` implements `interface { StatusCode() int }`, which a consumer declares
   locally and matches with `errors.As`. This is how `composes` stays free of any provider import
   for error handling.

   One gap, stated so it is not mistaken for a bug: the structural match only classifies
   errors that carry an HTTP status. An error surfaced *mid-stream* (OpenAI delivers a chunk-level
   `error` object as an `*HTTPError` with status 0; Anthropic delivers it as `StreamEvent.Error`,
   not a Go error at all) is not an HTTP-level rejection, and `StatusCode()` returns 0 there. A
   consumer testing `sc.StatusCode() == 429` will not match a mid-stream failure — that is
   intentional. Connection-establishment failures and non-2xx responses do carry their real status,
   so the 4xx/5xx classification works for the cases `composes` acts on (failover, 429 cooling).

6. **Vendor parameters are ordinary fields.** `ais.Extensions` and the Anthropic `Extend*` / `*Of`
   helpers are removed; what they carried are fields of the native types. The single escape hatch
   is `openai.ChatCompletionRequest.ExtraBody`, for private top-level parameters of
   OpenAI-*compatible* backends (`enable_thinking`, `chat_template_kwargs`, …). It may only add
   keys: a collision with a modelled field is rejected at marshal time.

### The constraint this decision gives up

`composes` may now import `provider/openai`. The previous rule — `composes` depends on the root
capability interface and canonical types, never on a provider — existed to keep dispatch
vendor-neutral. With canonical dispatch gone, the rule protects nothing, and its dependency test
is rewritten as "`composes` depends on `provider/openai` and on no other provider". This is the
one pre-existing architectural constraint deliberately abandoned here; it is recorded so a future
reader does not mistake it for an oversight.

### Relationship to v0.6.0's cross-protocol endpoint dispatch (PR #17)

Between this analysis being written and it being implemented, v0.6.0 (PR #17) shipped a
declarative endpoint API for `composes`: `EndpointSpec` carried a `Provider` field (a registry
protocol name), and `NewFromEndpoints` built one `aimodel.Client` per spec, so a single compose
client could mix OpenAI-compatible and Anthropic endpoints behind one `ChatCompletion` call.

That mixing is realized **only** through the canonical request model — one `ais.ChatRequest`
translated differently per endpoint. Remove the model and the mixing has nothing to stand on: a
`ChatCompletionRequest` is an OpenAI-wire value and cannot be handed to an Anthropic endpoint. So
this decision deliberately reverses the cross-protocol dimension of PR #17, one release after it
shipped:

- **Dropped**: the `EndpointSpec.Provider` field and the ability to mix protocols in one compose
  client. This is the same capability the canonical layer existed to provide, and it is retired for
  the same reason — it was built on a shared request model that expresses neither protocol
  faithfully, and the documented need is same-wire failover.
- **Kept and ported**: everything in PR #17 that is provider-neutral — endpoint aliases, capability
  filtering (`Capability`/`CapabilityProvider`), sticky routing, cost and latency strategies, 429
  cooling with `classifyHealth`, attempt observers, and `Stats()` health snapshots. These run over
  `provider/openai` types now, unchanged in behavior. `classifyHealth` reads the status through the
  structural `interface{ StatusCode() int }` rather than `*ais.APIError`, which is what lets it
  drop the canonical import.

The cost is explicit: a deployment that genuinely needs to fail one request over from an
OpenAI-compatible backend to Anthropic must now run two compose clients (or map the request
itself), where v0.6.0 let it declare both endpoints in one list. That is the price of removing the
shared model, accepted here on the evidence that no caller in this repository exercised the mixing.
If a concrete cross-protocol requirement appears, the right response is a mapping the caller writes
with full knowledge of both protocols — not a revival of the canonical layer.

### Guards

The decision is enforced by tests rather than by convention, because the failure mode is gradual
re-growth of a shared layer:

1. `provider/openai` and `provider/anthropic` import neither each other nor the root package.
2. No public API mentions a shared semantic package.
3. Both `*HTTPError` types satisfy `interface { StatusCode() int }` (compile-time assertion).
4. Packages declared vendor-neutral contain no identifier carrying protocol semantics
   (`message`, `content`, `tool`, `usage`, …), checked over the AST so comments and strings do not
   cause false positives. `composes` is *not* such a package — it is openly an OpenAI-wire tool.
5. Every public wire type survives a marshal → unmarshal → marshal round trip unchanged. This
   replaces the canonical contract tests and states the fidelity principle as an executable check.

## Consequences

- **v0.7.0 is a breaking release.** `aimodel.Client`, `ais.*`, `WithProvider`, the unified
  `Stream`, the shared error model, extensions and the default-model behavior are gone.
  [MIGRATION.md](../../MIGRATION.md) maps every removed symbol; v0.6.1 marks them all
  `Deprecated:` first so `staticcheck` reports call sites before the removal lands.

- **Delivering one request to two protocols is no longer possible, and is not a goal.** Callers
  needing that write the mapping themselves, where they have the context to decide what each field
  should become — the decision the canonical layer had to make blindly.

- **Duplication is now expected.** Timeout options, stream aggregation, SSE scanning and error
  parsing exist twice. The honest risk is that in a few years someone factors them back into a
  shared package and the canonical layer regrows under a new name. Guards 4 and 5 exist to make
  that visible in CI rather than gradual; the neutrality test above is the standard to apply.

- **Test coverage must be moved before it is deleted, not after.** The canonical layer carried 21
  e2e test functions and 10 integration cases; the native layer had two example files per
  provider. Tool calls, image content, thinking, prompt caching and streaming usage merging are
  re-established against the native entry points — replaying the same request/response bytes the
  canonical tests used — *before* the canonical tests are removed. This step is not compressible.

- **Anthropic's native surface gains what the canonical layer used to hide.** The
  "start usage + terminal delta" merge becomes an observable `MessageStream.Usage()` instead of a
  private helper, and unmodelled content blocks are the response's own blocks rather than entries
  in an extension bag.

- **Adding a provider is cheaper and more local.** A new vendor is a new package with no contract
  to satisfy, no registration, and no negotiation about which of its fields are "shared enough".
  It also gets no automatic interoperability — correctly, since there is none to give.

- **ADR 0004's small-interface principle survives in reduced scope.** New interaction forms still
  get their own narrow method sets rather than widening an existing one; those method sets now
  live on a provider's client instead of describing a cross-vendor capability, and capability
  absence is a compile error instead of `*ais.CapabilityError`.

## References

- [MIGRATION.md](../../MIGRATION.md)
- [Architecture](../architecture.md)
- [ADR 0001 — keep the SDK a thin wrapper](./0001-keep-the-sdk-a-thin-wrapper.md)
- [ADR 0004 — model capabilities with small interfaces](./0004-model-capabilities-with-small-interfaces.md)
- [ADR 0005 — canonical shared semantics over provider-native wire](./0005-canonical-shared-semantics-over-provider-native-wire.md)
- [ADR 0006 — a single-vendor interaction form gets a capability on provider-native types](./0006-responses-capability-on-provider-native-types.md)
