# Architecture Decision Records

This index lists the architectural decisions that shape `aimodel`. ADRs record why a decision was made, its trade-offs, and its consequences; [architecture.md](./architecture.md) describes the resulting system as a whole.

Every ADR listed here is in force. Decisions that have been reversed or absorbed are not kept as documents — the record of what changed and when lives in git history and in the per-protocol change logs.

| ADR | Status | Decision |
|---|---|---|
| [0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) | Accepted | Keep the SDK a thin API wrapper |
| [0002](./adr/0002-provider-native-as-the-only-public-interface.md) | Accepted | Provider-native clients are the only public interface |

[0001](./adr/0001-keep-the-sdk-a-thin-wrapper.md) sets the scope: request building, connection management and response decoding, with resilience and observability left to the caller. One SDK call is one HTTP request.

[0002](./adr/0002-provider-native-as-the-only-public-interface.md) shapes the SDK: there is no vendor-neutral request/response model, no canonical translation and no provider registry. `openai` and `anthropic` are complete, mutually independent clients over their own wire protocols, and shared code is admitted only when it can be written without a semantic type crossing between them.

Multi-backend routing, in-call retries and endpoint health live in [vage/largemodel](https://github.com/vogo/vage) — not in this module.

## Adding an ADR

Create `doc/adr/NNNN-short-title.md` from this structure: title, status, context, decision, consequences, and references. ADRs are immutable after acceptance except for clarifications; a changed decision gets a new ADR that supersedes the old one, and the superseded document is removed once nothing in force depends on it.
