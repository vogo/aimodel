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

- Canonical open-string parameters such as `ReasoningEffort` are not enum-validated; provider-only parameters belong to that provider's native API or established extension surface.
- Request structures carry no side-effecting state — a `ChatRequest` is safe to reuse.
- One call = one HTTP request, the multi-model failover path being the sole exception ([design/compose.md](./design/compose.md)).

## 2. Canonical representation: shared provider semantics

The canonical types in `ais` are a provider-neutral shared semantic layer. A field is admitted only when **at least two providers have a real, verifiable mapping for it**. Similar spelling, popularity in OpenAI-compatible APIs, direct-serialization convenience, or a backend ignoring unknown JSON is not evidence of shared semantics. Callers, provider subpackages, and `composes` use this deliberately small contract through the root `aimodel` facade:

```
                     ┌──────────────────────────────┐
   ChatRequest ─────▶│ provider "openai"            │──▶ POST {baseURL}/chat/completions
  (shared shape)     │ toOpenAIRequest()            │
        │            └──────────────────────────────┘
        │            ┌──────────────────────────────┐
        └───────────▶│ provider "anthropic"         │──▶ POST {baseURL}/v1/messages
                     │ toAnthropicRequest()          │
                     └──────────────────────────────┘
                                  │
   ChatResponse ◀── provider response translations ─┘
  (shared shape)
```

Each provider owns its wire construction and response normalization. Provider-only capabilities belong to its native API. Existing provider-specific extension scenarios continue through the unified extension channel, but extensions are not a loophole for reintroducing removed canonical request fields.

**Field attribution — the “≥ 2 providers” test.** This is the sole admission rule:

| Situation | Approach | Example |
|---|---|---|
| At least two providers map the same semantic | Canonical field + provider mappings | `TopP`, `Stop` ↔ `stop_sequences`, `ReasoningEffort`, `CacheReadTokens`; response-side `Usage.ServiceTier` |
| Vendor extension adopted by ≥ 2 vendors | Canonical field, pass-through where native | `TopK` (Anthropic native; several OpenAI-compatible backends accept it), `Thinking` (Anthropic + Qwen/GLM/DeepSeek-style backends) |
| Single-provider semantics | **Provider extension value** under the node's `Extensions` namespace, defined and read only by that provider's package | `anthropic.RequestExtension` (`AutoCache` / `AutoCacheTTL` / `Container` / `InferenceGeo`), `anthropic.MessageExtension` (`CacheBreakpoint`, `ExtraBlocks`), `anthropic.ToolExtension`, `anthropic.ChoiceExtension` (`StopDetails`), `anthropic.ResponseExtension` (`Container`), `anthropic.UsageExtension` (cache writes, server-tool counts, geography) |
| Single-provider convenience constants | Named in the provider package; the open canonical string passes the value through verbatim | `anthropic.FinishReasonRefusal` / `PauseTurn` / `ModelContextWindowExceeded` |

Attribution evidence for retained fields that are not obviously two-sided: response-side `Usage.ServiceTier` maps OpenAI and Anthropic usage responses; `Strict` on `Tool` maps OpenAI's `function.strict` and Anthropic's tool-level `strict`; `Stop` maps `stop` ↔ `stop_sequences`. Request-side service tier and OpenAI-only log probabilities, storage/metadata, prompt-cache routing, audio/file and generation-count controls are not canonical.

**The extension channel (`ais.Extensions`).** Every extendable node — `ChatRequest`, `Message`, `Tool`, `ChatResponse`, `Choice`, `Usage`, `StreamChunk`, `StreamChunkChoice` — carries an `Extensions map[string]any` tagged `json:"-"`, keyed by registered provider name. The contract:

- Canonical JSON is **never** affected: the map is not serialized, and providers ignore every foreign namespace.
- Each provider package defines one strongly-typed value per node and public set/read helpers (e.g. `anthropic.ExtendRequest` / `anthropic.RequestExtensionOf`); wire types stay private. A value of the wrong type fails request translation with a `*ais.ExtensionTypeError` naming the node — before any network I/O.
- The core layer owns only the container lifecycle: `Clone()` copies the maps at every node, and `Message.AppendDelta` merges same-name namespaces through the minimal `ais.ExtensionMerger` interface (copy-on-write; a value that does not implement it is replaced). Values are read-only once attached.
- Extensions are an **in-process translation contract**, not a cross-process JSON contract. Callers needing the full vendor payload persist it from the provider's native surface, not by marshalling canonical types.
- A third-party provider adds proprietary parameters by defining its own extension value and reading `Extensions[itsName]`, without changing `ais/schema.go`, the root package, or any other provider.

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
3. `provider.NewChatRequest` builds the URL, body, and headers (the OpenAI provider adds wire-only `stream_options.include_usage=true` on stream requests);
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
| `provider/openai/` | OpenAI-compatible provider: public native wire types/client, bidirectional canonical translation, error parsing and SSE decoder. Registers `openai.Name` on import |
| `provider/anthropic/` | Anthropic provider: native wire types, bidirectional translation, headers, SSE decoder, `anthropic.Options`, and the public extension surface (`extension.go`). Registers `anthropic.Name` on import |
| `composes/` | Multi-model dispatch strategies and health tracking (depends only on the root capability interface) |
| `examples/` / `integrations/` | Usage examples and integration tests |

## 6. Maintenance convention

When an official API changes, update these in sync:

1. the wrapper code;
2. the relevant `doc/` document — a `doc/design/` topic and/or the protocol's `*-chat-api.md`;
3. the protocol's change log — [anthropic/anthropic-api-changes.md](./anthropic/anthropic-api-changes.md) or [openai/openai-api-changes.md](./openai/openai-api-changes.md);
4. the root `README.md` / `CLAUDE.md` **only if** the public usage surface or the agent-facing guidance changed — they link here rather than restating design.

When an architectural decision changes, add an ADR under [`doc/adr/`](./adr/) and update the [ADR index](./adr.md).
