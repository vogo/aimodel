# Architecture

`github.com/vogo/aimodel` — a unified Go SDK for AI model APIs across multiple protocols (OpenAI-compatible, Anthropic), with zero external dependencies.

This document covers the **cross-cutting architecture**: design scope, the canonical representation, the client and its protocol dispatch, and the repository layout. Everything below the architecture level lives in its own document:

The decisions behind this architecture are recorded in the [ADR index](./adr.md).

| Topic | Document |
|---|---|
| Canonical request/response types, `Usage` | [design/data-model.md](./design/data-model.md) |
| `Stream`, delta merging, `ExtraBlocks`, interception | [design/streaming.md](./design/streaming.md) |
| Tool definitions, `tool_choice`, parallel tool results | [design/tool-use.md](./design/tool-use.md) |
| Prompt-cache modes and accounting | [design/prompt-caching.md](./design/prompt-caching.md) |
| Sentinel errors, `APIError`, `MultiError` | [design/errors.md](./design/errors.md) |
| Multi-model dispatch strategies and health tracking | [design/compose.md](./design/compose.md) |
| Per-protocol wire mapping (implemented in `provider/anthropic` · `provider/openai`) | [anthropic/anthropic-message-api.md](./anthropic/anthropic-message-api.md) · [openai/openai-chat-api.md](./openai/openai-chat-api.md) |

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

The SDK uses the **OpenAI Chat Completions format as its canonical representation**. The canonical types live in the vendor-neutral `ais` package; callers, provider subpackages, and `composes` all use them directly (`ais.ChatRequest`, `ais.Message`, etc.) with the root `aimodel` package as the client facade. A `Client` resolves one registered **provider** at construction time and delegates every call to it:

```
                     ┌──────────────────────────────┐
   ChatRequest ─────▶│ provider "openai"            │──▶ POST {baseURL}/chat/completions
  (OpenAI shape)     │ (direct serialization)       │
        │            └──────────────────────────────┘
        │            ┌──────────────────────────────┐
        └───────────▶│ provider "anthropic"         │──▶ POST {baseURL}/v1/messages
                     │ toAnthropicRequest()          │
                     └──────────────────────────────┘
                                  │
   ChatResponse ◀── fromAnthropicResponse() ◀────────┘
  (OpenAI shape)
```

The reasoning: the OpenAI format is the de-facto standard and the overwhelming majority of backends (DeepSeek, Kimi, GLM, Qwen, Doubao, MiniMax, …) speak it natively. Choosing it as the canonical representation makes the OpenAI path **zero-translation**, leaving only the Anthropic path to translate in both directions. Each provider's translation lives entirely in its own subpackage (`provider/openai`, `provider/anthropic`), so the root package holds no vendor wire types.

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

Default (OpenAI-compatible) client:

```go
client, err := aimodel.NewClient(
    aimodel.WithAPIKey("sk-..."),
    aimodel.WithBaseURL("https://api.openai.com/v1"),
    aimodel.WithDefaultModel(ais.ModelOpenaiGPT41),
    aimodel.WithTimeout(90*time.Second),
)
```

Anthropic client — select the provider by name and pass its vendor options through the unified `WithProviderOptions` channel:

```go
import "github.com/vogo/aimodel/provider/anthropic"

client, err := aimodel.NewClient(
    aimodel.WithAPIKey("sk-ant-..."),
    aimodel.WithProvider(anthropic.Name),
    aimodel.WithProviderOptions(anthropic.Options{
        Beta:          []string{"context-1m-2025-08-07"},
        Version:       "2023-06-01",
        UserProfileID: "user_abc123",
    }),
)
```

| Option | Purpose | Notes |
|---|---|---|
| `WithAPIKey(string)` | Auth key | Missing → `ErrNoAPIKey` |
| `WithBaseURL(string)` | API base URL | Trailing `/` stripped automatically |
| `WithProvider(string)` | Provider selection by registered name | Unset = `openai.Name` (OpenAI-compatible); e.g. `anthropic.Name` |
| `WithProviderOptions(any)` | Provider-specific configuration | Forwarded to the provider factory; type defined by the provider package (e.g. `anthropic.Options`). A type the provider does not recognize fails construction |
| `WithDefaultModel(string)` | Default model | Fills in an empty request `Model` |
| `WithTimeout(time.Duration)` | HTTP timeout | Default 60s; **applied after all options**, so option order does not matter |
| `WithHTTPClient(*http.Client)` | Custom HTTP client | `nil` panics outright (a programming error) |

The built-in `openai` and `anthropic` providers register themselves on import (the root package imports both by default). A third protocol is added by writing a subpackage that implements the provider contract and calls `ais.Register` in its `init` — **no root-package change required**. See §3.4.

### 3.2 Environment-variable fallback

`NewClient` reads the environment first, then applies explicit options (**explicit options win**):

| Setting | Fallback order |
|---|---|
| Model | `AI_MODEL` |
| API key | `AI_API_KEY` > `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` |
| Base URL | `AI_BASE_URL` > `OPENAI_BASE_URL` > `ANTHROPIC_BASE_URL` |

Implemented by `GetEnv(keys ...string)`, which returns the first non-empty value.

### 3.3 Construction-time validation

`NewClient` reads generic config (key, base URL, model, timeout, HTTP client), resolves the named provider from the registry, then hands the generic config plus `WithProviderOptions` to the provider factory. Failures surface here, at construction:

- Empty API key → `ErrNoAPIKey`;
- Unknown provider name → `unknown provider %q`;
- The **provider factory** validates its own requirements — the OpenAI factory rejects an empty base URL with `ErrNoBaseURL` (too many OpenAI-compatible backends to pick a default); the Anthropic factory accepts an empty base URL and defaults to `https://api.anthropic.com` at request time;
- A `WithProviderOptions` value of a type the provider does not recognize → factory error.

### 3.4 Registry dispatch and the provider contract

Capabilities are modeled as small per-interaction-form interfaces. `Client` implements `ChatCompleter` (the chat capability); a new interaction form is added by introducing a **new** capability interface and matching client method — never by widening this one:

```go
type ChatCompleter interface {
    ChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
    ChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error)
}
```

`chat.go` runs one shared execution pipeline for both paths and delegates the vendor-specific steps to the resolved `ais.ChatProvider`:

1. `req.Clone()` — deep-copy the request so the SDK's own rewrites (`Stream`, default model) never mutate the caller's object ([design/data-model.md](./design/data-model.md) §1.10);
2. set the `Stream` flag and `applyDefaultModel` fills an empty `Model`;
3. `provider.NewChatRequest` builds the URL, body, and headers (the OpenAI provider auto-adds `{include_usage:true}` on a stream request when `StreamOptions` is unset);
4. the core layer sends the single HTTP request;
5. a non-2xx response is read (under the shared size limit) and handed to `provider.ParseErrorResponse`; a success is normalized by `provider.ParseChatResponse`, or wrapped in a `Stream` driven by `provider.NewStreamDecoder`.

The provider contract (in `ais`) is exactly this vendor boundary — request building, response parsing, error parsing, and per-event SSE decoding:

```go
type ChatProvider interface {
    NewChatRequest(ctx context.Context, req *ChatRequest) (*http.Request, error)
    ParseChatResponse(body io.Reader) (*ChatResponse, error)
    ParseErrorResponse(statusCode int, body []byte) error
    NewStreamDecoder(body io.Reader) StreamDecoder
}
```

Providers are addressed by a stable string name through a concurrency-safe registry. `ais.Register(name, factory)` is monotonic: an empty name, a nil factory, or a duplicate name panics, so dispatch never depends on import order. The registry only resolves a name to a factory — it never guesses a protocol from the model and takes no part in `composes`' multi-model selection.

## 4. Model constants (`model.go`)

Plain string constants covering commonly used model names across OpenAI, DeepSeek, Gemini, Anthropic, MiniMax, Moonshot/Kimi, Zhipu GLM, Doubao, Qwen, and others. They are a writing convenience only — `ChatRequest.Model` accepts any string.

## 5. Repository layout

| Path | Contents |
|---|---|
| `ais/` | Vendor-neutral foundation: canonical schema (`schema.go`), error model (`errors.go`), the provider contract (`provider.go`), and the registry (`registry.go`). No vendor dependencies |
| Root package `aimodel` | `Client` facade + options (`client.go`), the shared execution pipeline and `ChatCompleter` capability interface (`chat.go`), `Stream` / interception (`stream.go` / `intercept.go`), model constants (`model.go`), env helpers (`util.go`). Canonical types come from the `ais` package |
| `provider/openai/` | OpenAI-compatible provider: request building, response/error parsing, SSE decoder. Registers `openai.Name` on import |
| `provider/anthropic/` | Anthropic provider: native wire types, bidirectional translation, headers, SSE decoder, `anthropic.Options`. Registers `anthropic.Name` on import |
| `composes/` | Multi-model dispatch strategies and health tracking (depends only on the root capability interface) |
| `examples/` / `integrations/` | Usage examples and integration tests |

## 6. Maintenance convention

When an official API changes, update these in sync:

1. the wrapper code;
2. the relevant `doc/` document — a `doc/design/` topic and/or the protocol's `*-chat-api.md`;
3. the protocol's change log — [anthropic/anthropic-api-changes.md](./anthropic/anthropic-api-changes.md) or [openai/openai-api-changes.md](./openai/openai-api-changes.md);
4. the root `README.md` / `CLAUDE.md` **only if** the public usage surface or the agent-facing guidance changed — they link here rather than restating design.

When an architectural decision changes, add an ADR under [`doc/adr/`](./adr/) and update the [ADR index](./adr.md).
