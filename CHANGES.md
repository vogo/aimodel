# CHANGES

This file is the **index** of aimodel's sync status against the official API protocols. The entries themselves live with the protocol they belong to:

| Protocol | Official docs | Change log | Implementation notes |
|---|---|---|---|
| OpenAI (OpenAI-compatible) | https://platform.openai.com/docs/api-reference/chat | [doc/openai/openai-api-changes.md](./doc/openai/openai-api-changes.md) | [doc/openai/openai-chat-api.md](./doc/openai/openai-chat-api.md) |
| Anthropic Messages API | https://platform.claude.com/docs/en/api/messages | [doc/anthropic/anthropic-api-changes.md](./doc/anthropic/anthropic-api-changes.md) | [doc/anthropic/anthropic-message-api.md](./doc/anthropic/anthropic-message-api.md) |

Each protocol's change log is ordered newest-first, and every entry records at least: the date, the official change, and the wrapper change summary.

For the cross-cutting design behind those changes, start at [doc/api.md](./doc/api.md).

**Maintenance convention**: when an official API changes, update the wrapper code, the relevant `doc/` design or protocol document, and that protocol's change log — see [doc/api.md](./doc/api.md) §6.

---

## Timeline

Both protocols merged, newest first. Follow a link for the full entry.

### Anthropic Messages API

| Date | Change |
|---|---|
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

### OpenAI Chat Completions API

| Date | Change |
|---|---|
| 2026-06-02 | [Multimodal input/output (`input_audio` / `file` parts, `modalities` / `audio`)](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Extend `ChatRequest` with common request fields (+ response `logprobs`)](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Sync `reasoning_effort` values, add `verbosity`](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Response type alignment (`reasoning_tokens`, `finish_reason` constants)](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Support `max_completion_tokens`, deprecate `max_tokens`](./doc/openai/openai-api-changes.md) |
| 2026-06-02 | [Baseline](./doc/openai/openai-api-changes.md) |
