# ADR 0005: Canonical shared semantics layered over each provider's native wire model

- Status: Accepted
- Date: 2026-07-25
- Supersedes: [ADR 0002](./0002-use-openai-chat-as-the-canonical-model.md)

## Context

[ADR 0002](./0002-use-openai-chat-as-the-canonical-model.md) made the OpenAI Chat Completions shape *be* the canonical model, so the OpenAI-compatible path could serialize `ais.ChatRequest` directly and only Anthropic paid a translation cost. Two things invalidated that premise:

1. **Canonical stopped being the OpenAI shape.** Field attribution was tightened to the "≥ 2 providers have a real, verifiable mapping" rule (Architecture §2), which removed the OpenAI-only members — log probabilities, audio/file payloads, storage/metadata, prompt-cache routing, generation-count and request-side tier — from `ais`. Canonical is now the greatest common denominator of the supported providers, not one vendor's wire format.
2. **Each provider needs a full-fidelity native surface.** The customization principle requires every vendor to expose a public native client and types that pursue complete coverage of the official API. OpenAI therefore gained its own independent wire model (`provider/openai/wire.go`) and native client (`native.go`), the same way Anthropic already had one.

Once both hold, "canonical == OpenAI wire" cannot survive: a canonical type that is deliberately smaller than the OpenAI API cannot also be the type that fully represents it. Something has to bridge the two, and leaving canonical serialization pointed at the OpenAI endpoint would either drag OpenAI-only fields back into `ais` or silently ship a request body that omits what the native client supports.

## Decision

Canonical types in `ais` are a provider-neutral **shared semantic layer**, and **every** provider — OpenAI included — owns a private-to-it native wire model plus an explicit bidirectional translation between canonical and that model. The canonical layer is built **on top of** the native layer, never the other way around.

Concretely:

- A field is admitted to `ais` only when at least two providers have a real, verifiable mapping for it. Similar spelling, popularity across OpenAI-compatible backends, direct-serialization convenience, or a backend tolerating unknown JSON is not evidence. Provider-only capabilities stay in that provider's native API (or ride the `ais.Extensions` channel where an established extension scenario already exists).
- The OpenAI provider translates through `toOpenAIRequest` / `fromOpenAIResponse` / `fromOpenAIChunk` (`provider/openai/translate.go`), exactly as the Anthropic provider translates through its own pair. No provider is privileged with a zero-translation path.
- There are two distinct entry points, and they are documented as such:
  - the **unified client** (`aimodel.Client`) — canonical in, canonical out, translated at the provider boundary;
  - the **native client** (e.g. `openai.NewClient`, `anthropic.NewClient`) — native types end to end, bypassing canonical translation, and the way to reach vendor-only features.

## Consequences

- Adding a provider or a vendor API change touches only that provider's subpackage; canonical types and signatures stay stable. This is what makes the universality and extensibility principles enforceable rather than aspirational.
- Every canonical call pays a mapping cost on both protocols, including OpenAI. The cost is a struct-to-struct copy, no extra network or serialization round trip.
- **The canonical↔wire seam is now hand-written on both paths.** Struct embedding no longer passes unknown canonical fields through automatically: a new canonical field that is not wired into `toOpenAIRequest` / `fromOpenAIResponse` is dropped silently — a composite literal need not list every field, so it compiles, marshals to a valid body with the field simply absent, and gets a 200 back. Go offers no compile-time protection here, and the mitigation is mostly procedural: the four-way sync in Architecture §6 requires the native layer and the canonical translation to move together, and each protocol's mapping-boundary section records what is intentionally absent.
- **A field-*coverage* guard is rejected; a field-*count* sentinel is not.** A reflective test asserting that every canonical field round-trips through a provider translation was considered and rejected: canonical and native are deliberately *not* isomorphic contracts, so an unmapped field is as likely to be an intended boundary (Architecture §2 lists several) as an oversight. Such a test would encode "full coverage" as the correctness standard, contradicting the admission rule above, and would need a per-field allowlist that drifts on its own. What *is* in place is `TestCanonicalNodeFieldCountsAreStable` (`ais/schema_sentinel_test.go`): it pins the field count of every canonical node a translation walks, and fails when one changes. It renders no verdict on whether a field should be mapped — it only guarantees nobody changes a canonical node's shape without being sent to the seams. Deciding a field belongs nowhere is a fine outcome; never being asked is not.
- **Canonical types stop parsing vendor wire shapes.** Under ADR 0002 it was reasonable for `ais` to understand OpenAI's nested `prompt_tokens_details` directly, since canonical *was* the wire format. It no longer is, so that knowledge belongs to the provider translations — each reads its own native usage type and fills the canonical counts. `ais.Usage.UnmarshalJSON` and its wire-shaped helpers were removed on 2026-07-26 once this made them unreachable; see the [OpenAI change log](../openai/openai-api-changes.md). Canonical types decode as plain structs.
- ADR 0002's naming consequence is void: canonical naming follows OpenAI where the shared semantic happens to match it, but that is convention and history, not a constraint.

## References

- [Architecture §2 — canonical representation](../architecture.md#2-canonical-representation-shared-provider-semantics)
- [Architecture §6 — maintenance convention](../architecture.md#6-maintenance-convention)
- [OpenAI Chat Completions wrapper design](../openai/openai-chat-api.md)
- [Anthropic Messages API wrapper design](../anthropic/anthropic-message-api.md)
- [Canonical data model](../design/data-model.md)
