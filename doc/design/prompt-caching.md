# Prompt Caching

How the canonical request expresses prompt-cache intent across protocols, and how cache accounting comes back.

- **Canonical fields**: `ais/schema.go` (`Message.CacheBreakpoint`, `Tool.CacheBreakpoint`, `ChatRequest.AutoCache` / `AutoCacheTTL`)
- **Anthropic wire mapping**: [../anthropic/anthropic-message-api.md](../anthropic/anthropic-message-api.md) §3.7

---

## 1. Two protocols, two philosophies

| | OpenAI-compatible | Anthropic |
|---|---|---|
| How caching is triggered | **Automatic** — prefixes over ~1024 tokens are cached with no request-side marker | **Explicit** — the request marks where the cacheable prefix ends |
| Caller controls | `PromptCacheKey` (routes requests sharing a prefix to the same cache) | Per-block breakpoints and/or request-root automatic caching |
| Accounting | `CacheReadTokens` only | `CacheReadTokens` + `CacheWriteTokens` (with per-TTL breakdown) |

Because only Anthropic needs request-side markers, every caching switch on the canonical types is a **struct-local field** marked `json:"-"`. It can never leak into the OpenAI-shape request body, and no separate public Anthropic request type is needed.

---

## 2. Anthropic: two coexisting modes

| Mode | Canonical fields | On the wire |
|---|---|---|
| **Per-block breakpoint** | `Message.CacheBreakpoint`, `Tool.CacheBreakpoint` (both `json:"-"`) | `cache_control:{type:"ephemeral"}` attached to the corresponding block / tool |
| **Automatic caching** | `ChatRequest.AutoCache` + `AutoCacheTTL` (both `json:"-"`) | A single request-root `cache_control:{type:"ephemeral", ttl:…}` |

They are **independent fields and may coexist**.

### 2.1 Per-block breakpoints

Anthropic caches everything **up to and including** the marked block, so on any block-array shape the marker attaches to the **last** block of the flagged message (or to the flagged tool, caching every tool definition up to and including it).

Setting `CacheBreakpoint` on a plain-text message promotes it to the single-element block-array form, purely so there is a block to attach `cache_control` to.

### 2.2 Automatic caching

Available since 2026-02-19. Instead of the caller maintaining breakpoints by hand, the **server** places the breakpoint on the last cacheable block and advances it forward as the conversation grows.

- `AutoCacheTTL` empty → `ttl` omitted → the default 5-minute cache.
- `AutoCacheTTL: "1h"` → the 1-hour cache.
- `AutoCache: false` → no request-root `cache_control` at all (default behavior).

---

## 3. Cache accounting

Both protocols report cache activity through the canonical `Usage` ([data-model.md](./data-model.md) §4):

| Field | Source |
|---|---|
| `CacheReadTokens` | OpenAI `prompt_tokens_details.cached_tokens`; Anthropic `cache_read_input_tokens` |
| `CacheWriteTokens` | Anthropic `cache_creation_input_tokens` (total). OpenAI has no cache-write accounting, so it stays 0. |
| `CacheWrite5mTokens` / `CacheWrite1hTokens` | Anthropic `usage.cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}`; the two sum to `CacheWriteTokens` |

**Cache read and write tokens are subsets of `PromptTokens`**, surfaced separately for observability. Do not add them again when computing cost.

If `CacheReadTokens` stays 0 across repeated requests that should share a prefix, something is invalidating it — a per-request timestamp or ID early in the prompt, a non-deterministic map serialization, or a changed tool list (tools are serialized before messages, so any tool change invalidates everything after).
