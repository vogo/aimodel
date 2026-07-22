# Error Model

- **Implementation**: `ais/errors.go`
- **Per-protocol parsing**: [../openai/openai-chat-api.md](../openai/openai-chat-api.md) §6 · [../anthropic/anthropic-message-api.md](../anthropic/anthropic-message-api.md) §6

---

## 1. Sentinel errors

`ErrNoAPIKey`, `ErrNoBaseURL`, `ErrStreamClosed`, `ErrEmptyResponse`, `ErrNoActiveModels` — match with `errors.Is`.

## 2. `APIError`

Carries the HTTP status code and the server's error body:

```go
type APIError struct {
    StatusCode int
    Code, Message, Type string
    Err error
}
```

The parsing strategy is uniform across protocols:

1. Read the response body, capped at `maxErrorBodySize = 1 MB` (`io.LimitReader`).
2. Try to decode the protocol's error JSON — OpenAI `{"error":{code,message,param,type}}`, Anthropic `{"type":"error","error":{type,message}}`.
3. **If decoding fails, or the message is empty, put the raw body into `Message`** — diagnostic information is never discarded.

Two extra defences on the non-streaming OpenAI path: a **200 response whose body still contains an `error` field** is turned into an `APIError` anyway (compatible backends are inconsistent about this), and an empty `Choices` array returns `ErrEmptyResponse`.

Errors surfaced from an SSE `error` event carry no HTTP status code (`StatusCode` is 0).

## 3. `ModelError`

`{Model, Err}` — associates an error with the specific model name that produced it. Implements `Unwrap`, so `errors.Is` / `errors.As` reach the underlying error.

## 4. `MultiError`

The collection of errors from a multi-model attempt. It implements Go 1.20+ `Unwrap() []error`, so `errors.Is` / `errors.As` match **any** of the underlying model errors. An empty collection degrades to `ErrNoActiveModels`.

See [compose.md](./compose.md).
