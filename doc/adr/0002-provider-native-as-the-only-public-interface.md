# ADR 0002: Provider-native clients are the only public interface

- Status: Accepted
- Date: 2026-08-02
- Unaffected: [ADR 0001](./0001-keep-the-sdk-a-thin-wrapper.md)

## Context

The obvious way to serve two protocols from one API is a provider-neutral canonical layer: one shared
request/response schema, plus a bidirectional translation per provider. That layer is rejected here,
because it costs more than it returns.

1. **Its one differentiating capability has no demand.** A shared request model exists so the same
   request can be delivered to different protocols. No caller in this repository does that — the
   real requirement is failover across several backends speaking *one* wire format.

2. **It is lossy by construction, and the loss grows.** A shared type can only admit a field when two
   or more providers have a real, verifiable mapping for it. That rule is correct for a shared type
   and, precisely because it is correct, keeps most of each vendor's API out. Everything excluded has
   to travel through a `map[string]any` side channel with per-provider typed accessors, merge rules
   for streaming, and its own error type. At that point the "portable" request is a partial request
   plus a vendor-specific bag, which is not portability.

3. **It delays access to shipped features.** A field documented upstream today cannot reach a
   canonical caller until a second vendor ships a mappable equivalent. A native surface has it on day
   one; the shared layer's admission rule makes the wrapper slower than the API it wraps.

4. **Its exceptions spread.** OpenAI's Responses API can only be exposed on provider-native types,
   because no honest shared shape can be minted from one vendor. That is the general case, not a
   special one: interaction forms and fields arrive per vendor and converge later, if ever.

5. **Its seam is unguarded.** A canonical field not wired into a translation is dropped silently,
   compiles, and returns 200. The only mitigation is procedural — a sync convention plus a sentinel
   that ensures somebody is *asked* when a shared shape changes.

## Decision

**The public interface of `aimodel` is the set of provider-native clients. There is no
vendor-neutral request/response model, no canonical translation, and no provider registry.**

1. **Each provider package is complete and self-contained.** `openai` owns Chat
   Completions and Responses — client, wire types, SSE, usage aggregation, errors and options.
   `anthropic` owns Messages the same way. Neither imports the other, the root package,
   or any shared semantic layer.

2. **Fidelity replaces universality.** A provider package expresses its official API completely
   and without compromise. Nothing is withheld pending a second vendor's equivalent, and no field
   is reshaped to resemble another vendor's spelling.

3. **The neutrality test governs shared code.** An enhancement may live outside a provider package
   **if and only if** it can be implemented inside a single package without introducing a semantic
   data type that another provider imports. Anything requiring a shared request/response model,
   bidirectional field mapping, or a cross-provider decision about which fields to keep *is* a
   canonical layer, and is rejected. Duplicating a small amount of logic between the
   two providers is the accepted price.

4. **Errors are matched structurally, not nominally.** There is no shared error type. Each
   provider's `*HTTPError` implements `interface { StatusCode() int }`, which a consumer declares
   locally and matches with `errors.As`.

   One gap, stated so it is not mistaken for a bug: the structural match only classifies errors
   that carry an HTTP status. An error surfaced *mid-stream* (OpenAI delivers a chunk-level `error`
   object as an `*HTTPError` with status 0; Anthropic delivers it as `StreamEvent.Error`, not a Go
   error at all) is not an HTTP-level rejection, and `StatusCode()` returns 0 there. A consumer
   testing `sc.StatusCode() == 429` will not match a mid-stream failure — that is intentional.
   Connection-establishment failures and non-2xx responses do carry their real status.

5. **Vendor parameters are ordinary fields.** A vendor-specific parameter is a field of that
   vendor's native type. The single escape hatch is `openai.ChatCompletionRequest.ExtraBody`, for
   private top-level parameters of OpenAI-*compatible* backends (`enable_thinking`,
   `chat_template_kwargs`, …). It may only add keys: a collision with a modelled field is rejected
   at marshal time.

6. **A capability is a narrow method set inside one provider package.** A new interaction form gets
   a new method set on that provider's client instead of widening an existing interface, and a
   provider that lacks a capability simply does not have the method — absence is a compile error,
   not a runtime error value.

### Guards

The decision is enforced by tests rather than by convention, because the failure mode is the gradual
growth of a shared layer:

1. `openai` and `anthropic` import neither each other nor the root package.
2. No public API mentions a shared semantic package.
3. Both `*HTTPError` types satisfy `interface { StatusCode() int }` (compile-time assertion).
4. Packages declared vendor-neutral contain no identifier carrying protocol semantics
   (`message`, `content`, `tool`, `usage`, …), checked over the AST so comments and strings do not
   cause false positives.
5. Every public wire type survives a marshal → unmarshal → marshal round trip unchanged. This
   states the fidelity principle as an executable check.

## Consequences

- **Delivering one request to two protocols is not possible, and is not a goal.** A caller needing
  that writes the mapping, where it has the context to decide what each field should become — the
  decision a shared layer would have to make blindly. Multi-backend failover across several
  backends of one wire format is handled by [vage/largemodel](https://github.com/vogo/vage), not
  by mapping one request across protocols.

- **Duplication is expected.** Timeout options, stream aggregation, SSE scanning and error parsing
  exist twice. The honest risk is that someone factors them into a shared package and a canonical
  layer grows there under a new name. Guards 4 and 5 make that visible in CI rather than
  gradual; the neutrality test above is the standard to apply.

- **A native surface exposes what a shared one would have to hide.** Anthropic's "start usage +
  terminal delta" merge is an observable `MessageStream.Usage()`, and unmodelled content blocks are
  the response's own blocks.

- **Adding a provider is cheap and local.** A new vendor is a new package with no contract to
  satisfy, no registration, and no negotiation about which of its fields are "shared enough". It
  also gets no automatic interoperability — correctly, since there is none to give.

## References

- [Architecture](../architecture.md)
- [ADR 0001 — keep the SDK a thin wrapper](./0001-keep-the-sdk-a-thin-wrapper.md)
