# Error Model

- **Implementation**: `ais/errors.go`
- **Per-protocol parsing**: [../openai/openai-chat-api.md](../openai/openai-chat-api.md) §6 · [../anthropic/anthropic-message-api.md](../anthropic/anthropic-message-api.md) §6

---

## 1. Sentinel errors

`ErrNoAPIKey`, `ErrNoBaseURL`, `ErrStreamClosed`, `ErrEmptyResponse`, `ErrNoActiveModels`, `ErrCapabilityNotSupported` — match with `errors.Is`.

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

## 3. `CapabilityError`

```go
type CapabilityError struct {
    Provider, Capability string
}
```

Returned when the client's resolved provider does not implement the capability being called — for example `Responses` on an Anthropic client. It names both sides and unwraps to `ErrCapabilityNotSupported`, so `errors.Is(err, ais.ErrCapabilityNotSupported)` is the general check and `errors.As` gets the specifics.

This is a **local** failure: it is returned before any HTTP request is built, so an unsupported capability never reaches the network and never falls back to a different endpoint. See [ADR 0006](../adr/0006-responses-capability-on-provider-native-types.md).

## 4. `ModelError`

`{Model, Err}` — associates an error with the specific model name that produced it. Implements `Unwrap`, so `errors.Is` / `errors.As` reach the underlying error.

## 5. `MultiError`

The collection of errors from a multi-model attempt. It implements Go 1.20+ `Unwrap() []error`, so `errors.Is` / `errors.As` match **any** of the underlying model errors. An empty collection degrades to `ErrNoActiveModels`.

See [compose.md](./compose.md).
