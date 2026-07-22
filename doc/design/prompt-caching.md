# Prompt Caching

How prompt-cache intent is expressed across protocols, and how cache accounting comes back.

- **Canonical mechanism**: the unified provider extension channel `core.Extensions` ([../api.md](../api.md) §2)
- **Anthropic extension API**: `provider/anthropic/extension.go` (`RequestExtension`, `MessageExtension`, `ToolExtension`, `UsageExtension`)
- **Anthropic wire mapping**: [../anthropic/anthropic-message-api.md](../anthropic/anthropic-message-api.md) §3.7

---

## 1. Two protocols, two philosophies

| | OpenAI-compatible | Anthropic |
|---|---|---|
| How caching is triggered | **Automatic** — prefixes over ~1024 tokens are cached with no request-side marker | **Explicit** — the request marks where the cacheable prefix ends |
| Caller controls | `PromptCacheKey` (canonical; routes requests sharing a prefix to the same cache) | Per-block breakpoints and/or request-root automatic caching, via the `anthropic` extension API |
| Accounting | canonical `CacheReadTokens` only | canonical `CacheReadTokens` + `anthropic.UsageExtension` cache-write counts |

Because only Anthropic needs request-side markers, no caching switch lives on the canonical types. All Anthropic cache intent travels through the node's `Extensions` map (`json:"-"`), set with the `anthropic` package helpers — it can never leak into the OpenAI-shape request body, and adding another provider's cache semantics later needs no canonical change.

---

## 2. Anthropic: two coexisting modes

| Mode | Extension API | On the wire |
|---|---|---|
| **Per-block breakpoint** | `anthropic.ExtendMessage(&msg, &anthropic.MessageExtension{CacheBreakpoint: true})` / `anthropic.ExtendTool(&tool, &anthropic.ToolExtension{CacheBreakpoint: true})` | `cache_control:{type:"ephemeral"}` attached to the corresponding block / tool |
| **Automatic caching** | `anthropic.ExtendRequest(req, &anthropic.RequestExtension{AutoCache: true, AutoCacheTTL: …})` | A single request-root `cache_control:{type:"ephemeral", ttl:…}` |

They are **independent switches and may coexist**.

### 2.1 Per-block breakpoints

Anthropic caches everything **up to and including** the marked block, so on any block-array shape the marker attaches to the **last** block of the flagged message (or to the flagged tool, caching every tool definition up to and including it).

Flagging a plain-text message promotes it to the single-element block-array form, purely so there is a block to attach `cache_control` to.

### 2.2 Automatic caching

Available since 2026-02-19. Instead of the caller maintaining breakpoints by hand, the **server** places the breakpoint on the last cacheable block and advances it forward as the conversation grows.

- `AutoCacheTTL` empty → `ttl` omitted → the default 5-minute cache.
- `AutoCacheTTL: "1h"` → the 1-hour cache.
- No `RequestExtension` (or `AutoCache: false`) → no request-root `cache_control` at all (default behavior).

---

## 3. Cache accounting

Cache **reads** have cross-provider consensus and stay canonical; cache **writes** are Anthropic-only and come back on the usage extension:

| Field | Source |
|---|---|
| `Usage.CacheReadTokens` (canonical) | OpenAI `prompt_tokens_details.cached_tokens`; Anthropic `cache_read_input_tokens` |
| `anthropic.UsageExtensionOf(&resp.Usage).CacheWriteTokens` | Anthropic `cache_creation_input_tokens` (total). OpenAI has no cache-write accounting, so the extension is absent there. |
| `UsageExtension.CacheWrite5mTokens` / `CacheWrite1hTokens` | Anthropic `usage.cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}`; the two sum to `CacheWriteTokens` |

**Cache read and write tokens are subsets of `PromptTokens`**, surfaced separately for observability. Do not add them again when computing cost. `Usage.Add` sums only the canonical counts; the extension describes one request and is left untouched.

If `CacheReadTokens` stays 0 across repeated requests that should share a prefix, something is invalidating it — a per-request timestamp or ID early in the prompt, a non-deterministic map serialization, or a changed tool list (tools are serialized before messages, so any tool change invalidates everything after).
