# CHANGES

This file is the **index** of aimodel's sync status against the official API protocols. The entries themselves live with the protocol they belong to:

| Protocol | Official docs | Change log | Implementation notes |
|---|---|---|---|
| OpenAI Chat Completions (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | [doc/openai/openai-api-changes.md](./doc/openai/openai-api-changes.md) | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) |
| OpenAI Responses | https://platform.openai.com/docs/api-reference/responses | [doc/openai/openai-api-changes.md](./doc/openai/openai-api-changes.md) | [doc/openai/openai-response-api.md](./doc/openai/openai-response-api.md) |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | [doc/anthropic/anthropic-api-changes.md](./doc/anthropic/anthropic-api-changes.md) | [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md) |

Each protocol's change log is ordered newest-first, and every entry records at least: the date, the official change, and the wrapper change summary.

For the cross-cutting design behind those changes, start at [doc/architecture.md](./doc/architecture.md).

**Maintenance convention**: when an official API changes, update the provider's wire types and client, the relevant `doc/` document, and that protocol's change log — see [doc/architecture.md](./doc/architecture.md) §6. If the change contradicts an invariant an accepted ADR states, supersede that ADR too ([ADR index](./doc/adr.md)).

---

## Releases

Protocol-independent changes to the SDK's own surface.

| Version | Change |
|---|---|
| v0.6.1 | Deprecate the canonical API. Every symbol removed in v0.7.0 — package `ais`, the root canonical client / stream / interception / `Responder` surface, and both providers' registry and extension entry points — now carries a `Deprecated:` comment pointing at [MIGRATION.md](./MIGRATION.md). No behavior, signature or serialization change. |
| v0.7.0 | Remove the canonical API. `provider/openai` and `provider/anthropic` are the only public interface, each expressing its protocol completely and independently; `composes` narrows to dispatch within the OpenAI-compatible wire format. Both `HTTPError` types implement `StatusCode() int` (the field is renamed `Status`), both native streams accumulate and report usage, and `openai.ChatCompletionRequest.ExtraBody` carries backend-private parameters. Migration table: [MIGRATION.md](./MIGRATION.md); reasoning: [ADR 0007](./doc/adr/0007-provider-native-as-the-only-public-interface.md). |

---

## Timeline

Both protocols merged, newest first. Follow a link for the full entry.

### Anthropic Messages API

| Date | Change |
|---|---|
| 2026-08-02 | [Native-only public API: observable stream usage merging, timeouts, structural errors](./doc/anthropic/anthropic-api-changes.md) |
| 2026-07-22 | [Public native Messages client and exported 2026-07-21 baseline wire schema](./doc/anthropic/anthropic-api-changes.md) |
| 2026-07-22 | [Canonical de-vendoring: Anthropic-only surfaces move to the unified extension channel (breaking; migration table inside)](./doc/anthropic/anthropic-api-changes.md) |
| 2026-07-21 | [`output_config`, usage extensions, `container`/`inference_geo`, tool fields, unknown-block preservation, profile header](./doc/anthropic/anthropic-api-changes.md) |
| 2026-07-10 | [Merge consecutive parallel `tool_result` blocks into one `user` message](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [Request-root automatic caching + per-TTL `cache_creation` in usage](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [Pass through the `top_k` sampling parameter](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [Configurable `anthropic-beta` / `anthropic-version` headers](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [New `stop_reason` constants and `stop_details`](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [Map `effort`, support `thinking.display`, deprecate `budget_tokens`](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [`tool_choice` `"none"` mapping and `disable_parallel_tool_use`](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [Preserve mid-conversation `system` messages](./doc/anthropic/anthropic-api-changes.md) |
| 2026-06-02 | [Baseline](./doc/anthropic/anthropic-api-changes.md) |

### OpenAI (Chat Completions and Responses)

| Date | Change |
|---|---|
| 2026-08-02 | [Native-only public API: `ExtraBody`, stream accumulation, timeouts, structural errors](./doc/openai/openai-api-changes.md) |
| 2026-08-01 | [Support the Responses API: native `/v1/responses` client and the root `Responder` capability](./doc/openai/openai-api-changes.md) |
| 2026-07-26 | [Remove `ais.Usage.UnmarshalJSON` (breaking for raw-wire decoding)](./doc/openai/openai-api-changes.md) |
| 2026-07-22 | [Public native Chat Completions client and explicit canonical translation layer](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Multimodal input/output (`input_audio` / `file` parts, `modalities` / `audio`)](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Extend `ChatRequest` with common request fields (+ response `logprobs`)](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Sync `reasoning_effort` values, add `verbosity`](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Response type alignment (`reasoning_tokens`, `finish_reason` constants)](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Support `max_completion_tokens`, deprecate `max_tokens`](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Baseline](./doc/openai/openai-api-changes.md) |
