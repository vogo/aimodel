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

## 4. Multi-endpoint aggregation lives in `composes`

Compose dispatch aggregates its failures with **composes-owned** types, not with the `ais` ones:

- `composes.EndpointError{Alias, Err}` attributes one attempt to a stable endpoint **alias**, and
- `composes.MultiError{Errors []*EndpointError}` collects them in attempt order.

The alias is what makes this necessary: the multi-endpoint case routes several endpoints to the *same* model, so a model name cannot distinguish them. `MultiError` implements Go 1.20+ `Unwrap() []error`, so `errors.Is` / `errors.As` still reach any underlying `*APIError`. See [compose.md](./compose.md) §5.

**Breaking change** (this replaced `*ais.MultiError` as compose's aggregate error): `errors.Is` keeps working unchanged, but a type assertion must be retargeted —

```go
var me *ais.MultiError      // before
var me *composes.MultiError // after — .Errors is []*EndpointError, keyed by Alias
errors.As(err, &me)
```

## 5. `ais.ModelError` / `ais.MultiError`

`ModelError{Model, Err}` associates an error with a model name; `MultiError{Errors []ModelError}` collects several, unwrapping to `ErrNoActiveModels` when empty. Both implement `Unwrap`, so `errors.Is` / `errors.As` reach the underlying errors.

**Status**: they remain exported for compatibility and for callers that aggregate by model name, but nothing in this repository produces them any more — compose dispatch uses the alias-keyed types in §4. They are not part of any provider's response path and receive no new features; new code should prefer `composes.MultiError`.
