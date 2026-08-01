# ADR 0006: A single-vendor interaction form gets a capability on provider-native types

- Status: Accepted
- Date: 2026-08-01
- Narrows: [ADR 0005](./0005-canonical-shared-semantics-over-provider-native-wire.md) (still in force for chat)

## Context

[ADR 0005](./0005-canonical-shared-semantics-over-provider-native-wire.md) settled the chat path: canonical `ais` types are the provider-neutral shared semantic layer, every provider owns a native wire model, and it states two entry points — the unified client is "canonical in, canonical out", the native client is native end to end.

OpenAI's Responses API (`POST /v1/responses`) does not fit that split. It is the interface OpenAI is moving the ecosystem to — the Assistants API retires 2026-08-26, and hosted tools, `previous_response_id` chaining, server-side conversations, reasoning items and the newer cache controls exist only there — so the SDK has to cover it. But:

1. **Only one provider has it.** Anthropic has no equivalent interaction form. The field-attribution rule (Architecture §2, restated by ADR 0005) admits a semantic to `ais` only when **at least two providers have a real, verifiable mapping**. Responses' request, item, tool and event shapes therefore cannot enter `ais` today.
2. **A canonical shape invented from one vendor is the mistake ADR 0002 already made.** ADR 0005 exists because "canonical *is* one vendor's wire format" broke down. Minting `ais.ResponseRequest` from OpenAI's `/v1/responses` would repeat it, and every field would have to be re-litigated when a second vendor's version arrives with different semantics.
3. **Hiding it behind the native client only is not enough.** Callers who configure one `aimodel.Client` — API key, base URL, HTTP client, timeout — would have to build and configure a second, parallel client to reach the endpoint their provider already serves.

So the choice is: leave Responses off the unified client entirely, invent a premature canonical shape, or let the unified client expose it on native types.

## Decision

**A single-vendor interaction form may be exposed on the unified client as its own capability interface, using that provider's native types, without entering `ais`.**

Concretely, for Responses:

- The root package gains `Responder` (`Responses` / `ResponsesStream`) whose parameters and results are `provider/openai` types. `*aimodel.Client` implements it. `ChatCompleter` and `ais.ChatProvider` are **not** widened — this follows [ADR 0004](./0004-model-capabilities-with-small-interfaces.md): a new interaction form gets a new interface.
- `ais` gains **no** Responses types or fields. The canonical schema, its JSON contract and its field-count sentinel are untouched.
- Dispatch is a type assertion on the resolved provider, not a registry concept: a provider that implements the internal Responses method set serves the call with the client's configured transport; one that does not returns `*ais.CapabilityError` (matching `ais.ErrCapabilityNotSupported`) **before any network I/O**.
- `composes` is not extended. Dispatching this form would force `composes` to import `provider/openai`, which its dependency test forbids and its narrow-capability design rejects.
- The exception is scoped to interaction forms that fail the two-provider test. It is **not** a licence to move canonical chat fields onto native types, and not a general "root may use vendor types" rule.

This narrows ADR 0005's "the unified client is canonical in, canonical out" to **the chat capability**, which is what that ADR was reasoning about. Everything else in ADR 0005 stands unchanged, including for OpenAI chat.

## Consequences

- **The root package now has a typed dependency on `provider/openai`.** It already imported it for default registration, so no new import edge appears and `TestRootProviderImportsAreBuiltInsOnly` still passes — but the dependency is now part of the root's *public* signature, not just its `init`. That is the real, deliberate cost of this decision.
- **Callers of `Responder` are coupled to OpenAI's wire model.** If Responses semantics later become canonical, their code changes. That is honest: the shape they are using *is* OpenAI's, and pretending otherwise through a one-vendor canonical type would not have made it portable.
- **The upgrade path is defined.** When a second provider ships an equivalent form, its shared semantics can be promoted into `ais` under the two-provider rule, with a canonical capability and per-provider translation — via a new ADR that supersedes this one, not by editing it.
- **Capability absence is a first-class, local outcome.** `ais.CapabilityError` / `ErrCapabilityNotSupported` give every future capability a stable way to say "this provider does not do that" without a network round trip and without a per-capability error type.
- **Full fidelity is the native layer's job, as always.** The customization principle applies unchanged: the Responses wire model pursues complete coverage of the official API, and unmodeled item/tool/event variants are preserved as raw JSON rather than dropped. See [openai-response-api.md](../openai/openai-response-api.md).
- **This does not weaken the attribution rule.** No field was promoted for convenience; the rule is what forced the native-type capability in the first place.

## References

- [Architecture §2 — canonical representation and field attribution](../architecture.md#2-canonical-representation-shared-provider-semantics)
- [Architecture §3.4 — registry dispatch and the provider contract](../architecture.md#34-registry-dispatch-and-the-provider-contract)
- [ADR 0004 — model capabilities with small interfaces](./0004-model-capabilities-with-small-interfaces.md)
- [ADR 0005 — canonical shared semantics over provider-native wire](./0005-canonical-shared-semantics-over-provider-native-wire.md)
- [OpenAI Responses API wrapper design](../openai/openai-response-api.md)
