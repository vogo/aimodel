# Architecture Decision Records

This index lists the architectural decisions that shape `aimodel`. ADRs record why a decision was made, its trade-offs, and its consequences; [architecture.md](./architecture.md) describes the resulting system as a whole.

| ADR | Status | Decision |
|---|---|---|
| [0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) | Accepted | Keep the SDK a thin API wrapper |
| [0002](./adr/0002-use-openai-chat-as-the-canonical-model.md) | Accepted | Use OpenAI Chat Completions as the canonical model |
| [0003](./adr/0003-dispatch-providers-through-a-registry.md) | Accepted | Dispatch providers through a registry |
| [0004](./adr/0004-model-capabilities-with-small-interfaces.md) | Accepted | Model capabilities with small interfaces |

## Adding an ADR

Create `doc/adr/NNNN-short-title.md` from this structure: title, status, context, decision, consequences, and references. ADRs are immutable after acceptance except for clarifications; a changed decision gets a new ADR that supersedes the old one.
