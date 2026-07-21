# Architecture Overview

`github.com/vogo/aimodel` — a unified Go SDK for AI model APIs across multiple protocols (OpenAI-compatible, Anthropic), with zero external dependencies.

This document covers the **cross-cutting architecture**: design scope, the canonical representation, the client and its protocol dispatch, and the repository layout. Everything below the architecture level lives in its own document:

| Topic | Document |
|---|---|
| Canonical request/response types, `Usage` | [design/data-model.md](./design/data-model.md) |
| `Stream`, delta merging, `ExtraBlocks`, interception | [design/streaming.md](./design/streaming.md) |
| Tool definitions, `tool_choice`, parallel tool results | [design/tool-use.md](./design/tool-use.md) |
| Prompt-cache modes and accounting | [design/prompt-caching.md](./design/prompt-caching.md) |
| Sentinel errors, `APIError`, `MultiError` | [design/errors.md](./design/errors.md) |
| Multi-model dispatch strategies and health tracking | [design/compose.md](./design/compose.md) |
| Per-protocol wire mapping | [anthropic/anthropic-chat-api.md](./anthropic/anthropic-chat-api.md) · [openai/openai-chat-api.md](./openai/openai-chat-api.md) |

---

## 1. Design scope

aimodel is a **thin API wrapper**. Its responsibilities are strictly limited to three things:

1. **Request translation** — turn one unified request structure into each protocol's wire format;
2. **Connection management** — HTTP client, timeouts, auth headers, SSE reading;
3. **Response normalization** — reduce each protocol's responses and stream events back to one structure.

It **deliberately excludes** retry, rate limiting, request validation, caching / persistence, and logging / metrics. Those belong to the caller or a framework above: putting them in the SDK introduces implicit behavior and costs the caller cannot control.

**Consequences (design constraints):**

- Parameters are not validated and unknown values pass through verbatim (`ReasoningEffort` / `Verbosity` / `ServiceTier` stay `string` rather than enums), so each OpenAI-compatible backend's private extensions keep working.
- Request structures carry no side-effecting state — a `ChatRequest` is safe to reuse.
- One call = one HTTP request, the multi-model failover path being the sole exception ([design/compose.md](./design/compose.md)).

## 2. Canonical representation: OpenAI-shaped

The SDK uses the **OpenAI Chat Completions format as its canonical representation**:

```
                     ┌─────────────────────────┐
   ChatRequest ─────▶│  Protocol = openai      │──▶ POST {baseURL}/chat/completions
  (OpenAI shape)     │  (direct serialization) │
        │            └─────────────────────────┘
        │            ┌─────────────────────────┐
        └───────────▶│  Protocol = anthropic   │──▶ POST {baseURL}/v1/messages
                     │  toAnthropicRequest()   │
                     └─────────────────────────┘
                                  │
   ChatResponse ◀── fromAnthropicResponse() ◀───┘
  (OpenAI shape)
```

The reasoning: the OpenAI format is the de-facto standard and the overwhelming majority of backends (DeepSeek, Kimi, GLM, Qwen, Doubao, MiniMax, …) speak it natively. Choosing it as the canonical representation makes the OpenAI path **zero-translation**, leaving only the Anthropic path to translate in both directions.

**How protocol-specific fields are handled:**

| Situation | Approach | Example |
|---|---|---|
| Both protocols have it | Canonical field + bidirectional mapping | `TopP`, `Stop` ↔ `stop_sequences` |
| Anthropic-only, and not part of the wire body | **Struct-local** field marked `json:"-"`, read only by the Anthropic translator | `Message.CacheBreakpoint`, `Tool.CacheBreakpoint`, `ChatRequest.AutoCache` / `AutoCacheTTL`, `Message.ExtraBlocks` |
| Anthropic-only, but semantically normalizable | Canonical field + mapping at translation time | `TopK` (OpenAI has no `top_k`; omitted when unset), `Container`, `InferenceGeo` |
| Anthropic-only semantics with no OpenAI value | Pass through verbatim as a named constant | `FinishReasonRefusal` / `PauseTurn` / `ModelContextWindowExceeded`, `StopDetails` |
| OpenAI-only | Canonical field, ignored by the Anthropic translator | `LogitBias`, `Store`, `Metadata`, `Modalities` / `Audio` |

The `json:"-"` struct-local convention matters: it guarantees a switch **can never leak into the OpenAI-shape request body**, while avoiding a second public request type just for Anthropic.

## 3. Client (`client.go` / `chat.go`)

### 3.1 Construction & options

```go
client, err := aimodel.NewClient(
    aimodel.WithAPIKey("sk-..."),
    aimodel.WithBaseURL("https://api.openai.com/v1"),
    aimodel.WithProtocol(aimodel.ProtocolOpenAI),
    aimodel.WithDefaultModel(aimodel.ModelOpenaiGPT4o),
    aimodel.WithTimeout(90*time.Second),
)
```

| Option | Purpose | Notes |
|---|---|---|
| `WithAPIKey(string)` | Auth key | Missing → `ErrNoAPIKey` |
| `WithBaseURL(string)` | API base URL | Trailing `/` stripped automatically |
| `WithProtocol(Protocol)` | Protocol selection | Zero value = `ProtocolOpenAI` |
| `WithDefaultModel(string)` | Default model | Fills in an empty request `Model` |
| `WithTimeout(time.Duration)` | HTTP timeout | Default 60s; **applied after all options**, so option order does not matter |
| `WithHTTPClient(*http.Client)` | Custom HTTP client | `nil` panics outright (a programming error) |
| `WithAnthropicBeta(...string)` | `anthropic-beta` header | Anthropic only; accumulates across calls, empty strings ignored, comma-joined on the wire |
| `WithAnthropicVersion(string)` | `anthropic-version` header | Anthropic only; empty keeps the default `2023-06-01` |
| `WithAnthropicUserProfileID(string)` | `anthropic-user-profile-id` header | Anthropic only; associates requests with an end-user profile, empty sends no header |

### 3.2 Environment-variable fallback

`NewClient` reads the environment first, then applies explicit options (**explicit options win**):

| Setting | Fallback order |
|---|---|
| Model | `AI_MODEL` |
| API key | `AI_API_KEY` > `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` |
| Base URL | `AI_BASE_URL` > `OPENAI_BASE_URL` > `ANTHROPIC_BASE_URL` |

Implemented by `GetEnv(keys ...string)`, which returns the first non-empty value.

### 3.3 Construction-time validation

- Empty API key → `ErrNoAPIKey`;
- `ProtocolOpenAI` with an empty base URL → `ErrNoBaseURL` (there are too many OpenAI-compatible backends to pick a default);
- `ProtocolAnthropic` allows an empty base URL — it falls back to `https://api.anthropic.com` at request time;
- Any other protocol value → `unsupported protocol %q`.

### 3.4 Protocol dispatch

`Client` implements `ChatCompleter`; `chat.go` is the single dispatch point, keyed on `Protocol`:

```go
type ChatCompleter interface {
    ChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
    ChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error)
}
```

Both paths share the same preamble:

1. `req.clone()` — deep-copy the request so the SDK's own rewrites (`Stream`, default model) never mutate the caller's object ([design/data-model.md](./design/data-model.md) §1.10);
2. set the `Stream` flag;
3. `applyDefaultModel` fills an empty `Model`;
4. on the streaming path, the OpenAI side auto-adds `{include_usage:true}` when `StreamOptions` is unset, so the terminal chunk carries usage.

## 4. Model constants (`model.go`)

Plain string constants covering commonly used model names across OpenAI, DeepSeek, Gemini, Anthropic, MiniMax, Moonshot/Kimi, Zhipu GLM, Doubao, Qwen, and others. They are a writing convenience only — `ChatRequest.Model` accepts any string.

## 5. Repository layout

| Path | Contents |
|---|---|
| Root package `aimodel` | Client, schema, protocol implementations, streaming |
| `chat.go` / `chat_client.go` | Protocol dispatch, the `ChatCompleter` interface, `Protocol` constants |
| `schema.go` | Canonical request/response types |
| `openai_chat.go` / `openai_stream.go` | OpenAI-compatible protocol implementation |
| `anthropic.go` / `anthropic_chat.go` / `anthropic_stream.go` | Anthropic types, translation, HTTP, SSE |
| `stream.go` / `intercept.go` | Streaming abstraction and interception |
| `errors.go` / `model.go` / `util.go` | Errors, model constants, environment helpers |
| `composes/` | Multi-model dispatch strategies and health tracking |
| `examples/` / `integrations/` | Usage examples and integration tests |

## 6. Maintenance convention

When an official API changes, update these in sync:

1. the wrapper code;
2. the relevant `doc/` document — a `doc/design/` topic and/or the protocol's `*-chat-api.md`;
3. the protocol's change log — [anthropic/anthropic-api-changes.md](./anthropic/anthropic-api-changes.md) or [openai/openai-api-changes.md](./openai/openai-api-changes.md);
4. the root `README.md` / `CLAUDE.md` **only if** the public usage surface or the agent-facing guidance changed — they link here rather than restating design.
