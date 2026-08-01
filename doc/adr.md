# Architecture Decision Records

This index lists the architectural decisions that shape `aimodel`. ADRs record why a decision was made, its trade-offs, and its consequences; [architecture.md](./architecture.md) describes the resulting system as a whole.

| ADR | Status | Decision |
|---|---|---|
| [0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) | Accepted | Keep the SDK a thin API wrapper |
| [0002](./adr/0002-use-openai-chat-as-the-canonical-model.md) | **Superseded** by [0005](./adr/0005-canonical-shared-semantics-over-provider-native-wire.md) | Use OpenAI Chat Completions as the canonical model |
| [0003](./adr/0003-dispatch-providers-through-a-registry.md) | Accepted | Dispatch providers through a registry |
| [0004](./adr/0004-model-capabilities-with-small-interfaces.md) | Accepted | Model capabilities with small interfaces |
| [0005](./adr/0005-canonical-shared-semantics-over-provider-native-wire.md) | Accepted | Canonical shared semantics layered over each provider's native wire model |
| [0006](./adr/0006-responses-capability-on-provider-native-types.md) | Accepted | A single-vendor interaction form gets a capability on provider-native types |

For the canonical representation, [0005](./adr/0005-canonical-shared-semantics-over-provider-native-wire.md) is the decision in force: canonical types in `ais` are the provider-neutral shared semantic layer, and every provider — OpenAI included — owns a native wire model with explicit bidirectional translation. [0002](./adr/0002-use-openai-chat-as-the-canonical-model.md) is retained as history only; do not read its "canonical *is* the OpenAI shape / no translation layer on the OpenAI path" text as a current constraint.

[0006](./adr/0006-responses-capability-on-provider-native-types.md) narrows 0005's "the unified client is canonical in, canonical out" to the **chat** capability. An interaction form only one provider has — today, the OpenAI Responses API — is exposed as its own capability interface on that provider's native types, and nothing about it enters `ais`. 0005 is otherwise unaffected.

## Adding an ADR

Create `doc/adr/NNNN-short-title.md` from this structure: title, status, context, decision, consequences, and references. ADRs are immutable after acceptance except for clarifications; a changed decision gets a new ADR that supersedes the old one.
