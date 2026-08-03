# Architecture Decision Records

This index lists the architectural decisions that shape `aimodel`. ADRs record why a decision was made, its trade-offs, and its consequences; [architecture.md](./architecture.md) describes the resulting system as a whole.

| ADR | Status | Decision |
|---|---|---|
| [0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) | Accepted | Keep the SDK a thin API wrapper |
| [0002](./adr/0002-use-openai-chat-as-the-canonical-model.md) | **Superseded** by [0005](./adr/0005-canonical-shared-semantics-over-provider-native-wire.md) | Use OpenAI Chat Completions as the canonical model |
| [0003](./adr/0003-dispatch-providers-through-a-registry.md) | **Superseded** by [0007](./adr/0007-provider-native-as-the-only-public-interface.md) | Dispatch providers through a registry |
| [0004](./adr/0004-model-capabilities-with-small-interfaces.md) | **Partially superseded** by [0007](./adr/0007-provider-native-as-the-only-public-interface.md) | Model capabilities with small interfaces |
| [0005](./adr/0005-canonical-shared-semantics-over-provider-native-wire.md) | **Superseded** by [0007](./adr/0007-provider-native-as-the-only-public-interface.md) | Canonical shared semantics layered over each provider's native wire model |
| [0006](./adr/0006-responses-capability-on-provider-native-types.md) | **Absorbed** into [0007](./adr/0007-provider-native-as-the-only-public-interface.md) | A single-vendor interaction form gets a capability on provider-native types |
| [0007](./adr/0007-provider-native-as-the-only-public-interface.md) | Accepted (§4 **partially superseded** by [0008](./adr/0008-shared-routing-core-across-protocol-wrappers.md)) | Provider-native clients are the only public interface |
| [0008](./adr/0008-shared-routing-core-across-protocol-wrappers.md) | Accepted | A shared routing core, protocol wrappers on top |
| [0009](./adr/0009-stateful-active-endpoint-with-in-call-retry.md) | Accepted | A pool has one active endpoint, retried in place before it is replaced |

[0007](./adr/0007-provider-native-as-the-only-public-interface.md) is the decision in force for the shape of the SDK: there is no vendor-neutral request/response model, no canonical translation and no provider registry. `provider/openai` and `provider/anthropic` are complete, mutually independent clients over their own wire protocols, and shared code is admitted only when it can be written without a semantic type crossing between them.

[0008](./adr/0008-shared-routing-core-across-protocol-wrappers.md) refines where that line falls for multi-backend dispatch. It replaces 0007 §4 ("composing another protocol's backends is an isomorphic loop") with a two-layer split: `composes` is a protocol-neutral routing core — strategies, health, aliases, attribution — and `composes/openais` / `composes/anthropics` bind it to their own wire types. Routing *mechanism* is shared; a request/response model still is not, and pools still never mix. The rest of 0007 is untouched.

[0009](./adr/0009-stateful-active-endpoint-with-in-call-retry.md) changes what the routing core *does*, not where its boundary lies. `composes` goes from re-ranking every candidate on every call to holding one active endpoint per pool, retried in place under an exponential policy and replaced only when judged dead — with health reduced to `available` / `dead` on a fixed recovery timer. It is also the one bounded exception to [0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md)'s "no retry in the SDK": the exception lives in the compose layer only, and the provider clients still issue exactly one request per call. 0008's neutrality rule is untouched — the new state is an index, a generation and two durations.

Read 0002, 0003 and 0005 as history. Their canonical model, registry dispatch and canonical-over-native layering describe the SDK up to v0.5.x, not the current design. 0006 is absorbed rather than reversed: "a capability speaks provider-native types" went from being its single documented exception to being the only rule, and the capability now lives on the provider's own client rather than on a unified one. From 0004, the small-interface principle survives — a new interaction form gets a new narrow method set instead of widening an existing one — but a "capability" is now a method set inside one provider package, and a provider that lacks one is a compile error rather than a runtime `*ais.CapabilityError`. [0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) is unaffected and still binding.

## Adding an ADR

Create `doc/adr/NNNN-short-title.md` from this structure: title, status, context, decision, consequences, and references. ADRs are immutable after acceptance except for clarifications; a changed decision gets a new ADR that supersedes the old one.
