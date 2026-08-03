# OpenAI API — Change Log

> Historical record: entries below describe the API surface when each change landed. OpenAI-only audio/file, log-probability, verbosity, storage/metadata, prompt-cache routing, generation-count and request-tier members were removed from the canonical schema in July 2026; they are not current `ais` capabilities. Entries dated before 2026-07-22 also predate the native wire model and the canonical translation layer — where they describe canonical types as the OpenAI wire shape, read that as the design at the time, not as a current invariant.

This file records **how aimodel's OpenAI wrapper tracks the official OpenAI APIs**: what changed upstream, and how the wrapper followed.

- **Official protocols**: Chat Completions (`POST /chat/completions`) and, since 2026-08-01, Responses (`POST /v1/responses`) — neither carries a standalone version number, both are keyed by the endpoint
- **Official docs**: https://platform.openai.com/docs/api-reference/chat · https://platform.openai.com/docs/api-reference/responses
- **Implementation notes**: [openai-chat-api.md](./openai-chat-api.md) · [openai-response-api.md](./openai-response-api.md)

**Maintenance convention**: see [../architecture.md](../architecture.md) §6. Every entry carries at least a date, the official change, and a wrapper change summary.

Since 2026-07-22 the OpenAI path has its own native wire model, so an OpenAI-side change lands in `provider/openai/wire.go` first. It reaches `ais/schema.go` only when the "≥ 2 providers" attribution test passes, and then it also needs a mapping in `provider/openai/translate.go` — canonical no longer picks up OpenAI fields for free.

Newest first.

---

## 2026-08-02 — Native-only public API: `ExtraBody`, stream accumulation, timeouts, structural errors

**Official change**: none — this is a change to the wrapper's own surface, recorded here because it changes how every OpenAI-side capability is reached.

**Wrapper change**

- **The canonical layer is gone** (v0.7.0). `provider/openai` is now reached directly: `openai.NewClient(apiKey, ...)` with `ChatCompletions` / `ChatCompletionsStream` and `Responses` / `ResponsesStream`. The registered provider (`Name`, `New`), the canonical translation (`toOpenAIRequest` / `fromOpenAIResponse` / `fromOpenAIChunk`) and the canonical SSE decoder are deleted, along with the root `Responder` capability, which only forwarded to the native client. Reasoning: [ADR 0002](../adr/0002-provider-native-as-the-only-public-interface.md).
- **`ChatCompletionRequest.ExtraBody`**: a controlled channel for the private top-level parameters OpenAI-*compatible* backends add (`enable_thinking`, `chat_template_kwargs`, …). Additive only — a key colliding with a modelled field, an empty key, or a value that is not valid JSON fails at marshal time, before any network I/O. Decoding fills it with every unmodelled key, so a request body round-trips losslessly. The modelled key set is derived from the struct tags and cannot drift.
- **`ChatCompletionStream` accumulates while the caller reads**: `Response()` returns the assembled completion (text, reasoning content, refusals and tool-call arguments concatenated in arrival order; identity fields last-non-empty-wins; choices grown by index) and `Usage()` the token accounting. This replaces the canonical `Message.AppendDelta`.
- **`WithTimeout(d)`**: bounds a whole call. It copies the client configured so far, so the caller's `*http.Client` is never mutated and a transport from an earlier `WithHTTPClient` survives.
- **`HTTPError` implements `StatusCode() int`**. Go forbids a field and a method sharing a name, so the exported field is renamed `Status`. Consumers now match any provider's transport error through a locally declared `interface{ StatusCode() int }` without importing this package. **Breaking** for code reading the field directly.
- **Model, role, finish-reason, reasoning-effort, content-part and tool constants** move into this package (`model.go`), covering OpenAI and the OpenAI-compatible backends addressed over this protocol. Model names are protocol facts, not a vendor-neutral contract.
- **Request copying is shallow** rather than a JSON round trip when forcing the stream flag — faster, and it no longer erases `ExtraBody` in transit.

---

## 2026-08-01 — Support the Responses API: native `/v1/responses` client and the root `Responder` capability

**Official change**: OpenAI positions the Responses API (`POST /v1/responses`) as its primary interface. The Assistants API retires **2026-08-26**, and hosted tools (`web_search` / `file_search` / `code_interpreter`), `previous_response_id` chaining, server-side conversations, reasoning items with encrypted content, and the newer prompt-cache controls exist only on Responses. Chat Completions keeps working but receives no new capabilities.

**Wire baseline verified 2026-08-01** — full inventory, official links and verification method in [openai-response-api.md](./openai-response-api.md).

**Wrapper change**

- **Native wire model** (`provider/openai/responses_wire.go`, `responses_events.go`, `responses_const.go`): `ResponsesRequest` and `Response` covering every documented field; the polymorphic `ResponseInput` (string or item array) and `ResponseMessageContent` (string or content parts); discriminated `ResponseInputItem` / `ResponseOutputItem` unions with dedicated payloads for `message`, `reasoning`, `function_call`, `function_call_output`, `web_search_call`, `file_search_call`, `code_interpreter_call` and `item_reference`; the flat `ResponseContent` part type with annotations and logprobs; `ResponseTool` with typed `function` / `web_search` / `file_search` / `code_interpreter` parameters; `ResponseUsage` with `input_tokens_details.{cached_tokens,cache_write_tokens}` and `output_tokens_details.reasoning_tokens`; the 53-type SSE event taxonomy on `ResponseStreamEvent`, listed by `ResponseStreamEventTypes()`.
- **Forward compatibility**: an item, tool or event type outside the baseline keeps its verbatim payload on `Raw` and is re-encoded byte-for-byte. A **known** event that fails to decode is still an error, so drift on a modeled shape stays loud.
- **Native client** (`provider/openai/responses.go`): `Client.Responses` and `Client.ResponsesStream`, both `POST {baseURL}/responses` with bearer auth, forcing the stream mode on a copy so the caller's request is never mutated. `ResponseStream.Recv` decodes the SSE event protocol (blank-line boundaries, comments, multi-line `data:`, payload `type` discriminator, no `[DONE]` sentinel), returns `io.EOF` at the end, turns an `error` event into `*HTTPError`, and closes the body exactly once on every terminal path. Non-2xx handling reuses the existing bounded reader and `*HTTPError` contract.
- **`Response.OutputText`**: an SDK convenience tagged `json:"-"`, derived at decode time from the `output_text` parts in output order; the items themselves are untouched and it is never serialized.
- **Root capability** (`responder.go`): `Responder` (`Responses` / `ResponsesStream`) with `*Client` implementing it. It delegates through the resolved provider using the client's API key, base URL, HTTP client and timeout. No default model, no canonical translation, no interception, no compose failover — the request's own `Model` is authoritative.
- **Capability errors** (`ais/errors.go`): new `ErrCapabilityNotSupported` sentinel and `CapabilityError{Provider, Capability}`. A non-OpenAI provider (Anthropic today) fails locally with no network call.
- **No canonical change**: no `ais` schema field was added, removed or remapped; `ChatCompleter`, `ais.ChatProvider`, the chat pipeline, `Stream`, interception and `composes` are untouched. `TestCanonicalNodeFieldCountsAreStable` and `ais/schema_vendor_test` are unaffected.
- **Architecture consequence**: the unified client's public surface now uses provider types for this one capability. Recorded at the time in a since-retired ADR that narrowed "canonical in, canonical out" to the chat capability — a narrowing later generalised into [ADR 0002](../adr/0002-provider-native-as-the-only-public-interface.md); `doc/architecture.md` §2/§3.4/§5 and `CLAUDE.md` were reconciled at the same time.
- **Tests**: offline throughout — request/response round-trip fixtures across every documented union, `httptest` non-streaming call, an SSE sweep over all 53 baseline events, typed payload dispatch, unknown-event preservation, `error` event, malformed known event, invalid JSON, oversized-line scan failure, idempotent close, and both structured and unstructured non-2xx bodies. Root tests cover delegation and the no-network capability error. Runnable native and unified examples added under `integrations/openai_tests/`.
- **Out of scope**: canonical translation of Responses, `composes` dispatch, `GET`/`DELETE`/cancel on `/v1/responses`, conversation-resource CRUD, background-response polling, and the Assistants and Realtime APIs.

## 2026-07-26 — Remove `ais.Usage.UnmarshalJSON` (breaking for raw-wire decoding)

**Official change**: none — dead-code removal following the 2026-07-22 restructuring.

**Wrapper change**

- Deleted `ais.Usage.UnmarshalJSON` and its private helpers `usageJSON` / `promptTokensDetails` / `completionTokensDetails` from `ais/schema.go`. `Usage` now decodes as a plain struct.
- **Why**: after 2026-07-22 neither provider reached it. OpenAI decodes `ChatCompletionUsage` and promotes through `fromOpenAIUsage`; Anthropic decodes `MessagesUsage` and promotes through `anthropicCanonicalUsage`. The canonical method was a third, unreachable copy of the same promotion — and the only remaining place in `ais` that hard-coded a vendor wire shape (`prompt_tokens_details`, `completion_tokens_details`), which the canonical layer is not supposed to know about.
- **Breaking**: decoding a raw provider `usage` object directly into `ais.Usage` no longer promotes the nested breakdowns — `CacheReadTokens` / `ReasoningTokens` come back 0. Decode the provider's native usage type instead, or go through the unified client. **Not affected**: canonical JSON produced by this SDK (`Usage` has no custom `MarshalJSON`, so it only ever emits the flat canonical fields, which still round-trip), and every unified-client or native-client call.
- The "explicit top-level wins" precedence rule disappears with the method; it only ever arbitrated between a flat field and a nested one in the same document, which no provider payload contains.
- Tests: the four nested-promotion cases in `ais/schema_test.go` and `TestUsage_ReasoningTokensPrecedence` in `provider/anthropic` (which tested canonical behavior through a test-only `Usage = ais.Usage` alias) were replaced by `TestUsageDecodesCanonicalFieldsOnly`, which pins the new boundary. Provider-side promotion stays covered by each provider's own usage tests.
- **Field-count sentinel added**: `TestCanonicalNodeFieldCountsAreStable` (`ais/schema_sentinel_test.go`) pins the field count of the 14 canonical nodes a provider translation walks, and fails with a pointer to the translation layers when one changes. This is deliberately *not* the rejected field-coverage guard — it asserts nothing about whether a field is mapped, only that a canonical shape change cannot land without the author being sent to the seams. Leaving a field unmapped stays a legitimate outcome; it just has to be a decision. Rationale in the canonical-layer ADR of the time, retired with that layer.

## 2026-07-22 — Public native Chat Completions client and explicit canonical translation layer

**Official change**: none — this is a wrapper restructuring, driven by the customization principle (every vendor exposes a public, full-fidelity native API) and by the tightened canonical field-attribution rule that had just removed the OpenAI-only members from `ais`.

**Wrapper change**

- **Native wire model** (`provider/openai/wire.go`): independent exported types for the full Chat Completions surface — `ChatCompletionRequest` / `ChatCompletionResponse` / `ChatCompletionChunk`, `ChatCompletionMessage`, `ChatCompletionContentPart`, `ChatCompletionUsage` with its nested `PromptTokensDetails` / `CompletionTokensDetails`, tools, `StreamOptions`, `Thinking`, and the OpenAI-only members that are not canonical (log probabilities, audio/file, storage/metadata, prompt-cache routing, generation counts, request-side tier).
- **Native client** (`provider/openai/native.go`): `openai.NewClient(apiKey, ...ClientOption)` with `WithBaseURL` / `WithHTTPClient`, `ChatCompletions` and `ChatCompletionsStream`. Native types end to end, canonical translation bypassed; default base URL `https://api.openai.com/v1`.
- **Canonical translation** (`provider/openai/translate.go`): `toOpenAIRequest`, `fromOpenAIResponse`, `fromOpenAIChunk`, `fromOpenAIMessage`, `fromOpenAIUsage`. The unified-client path now goes canonical → native → wire instead of marshalling `ais.ChatRequest` directly, and decodes into `ChatCompletionResponse` / `ChatCompletionChunk` before normalizing back to canonical.
- `provider.NewChatRequest` still adds the wire-only `stream_options.include_usage=true` on streaming requests; it is now set on the native request in `toOpenAIRequest`.
- Usage promotion moved with it: `fromOpenAIUsage` reads the nested native details into `Usage.CacheReadTokens` / `ReasoningTokens`. (`ais.Usage.UnmarshalJSON` still carried a duplicate of that promotion at this point; it was removed on 2026-07-26, see the entry above.)
- **Integration examples**: native non-streaming and streaming samples added under `integrations/`.
- **No canonical change**: no `ais` field was added, removed, renamed or remapped, and the serialized request/response bodies are unchanged. Purely additive on the public surface (the native client and types).
- **Architecture consequence**: this retired the "OpenAI-compatible path has no translation layer" invariant. Recorded on 2026-07-25 in the canonical-layer ADR of the time, itself retired with that layer in v0.7.0; `doc/architecture.md` §2, `openai-chat-api.md` and `CLAUDE.md` were reconciled at the same time. The seam is hand-written on both sides and has **no field-coverage test** — see [openai-chat-api.md](./openai-chat-api.md) §7 for why, and what a canonical field addition must do instead. A field-count sentinel was added on 2026-07-26 (entry above) to make a canonical shape change impossible to miss.

## 2026-06-02 — Multimodal input/output (`input_audio` / `file` content parts, `modalities` / `audio`)

**Official change**: Chat Completions supports audio/file input via the `input_audio` and `file` content parts, requests audio output via `modalities` + `audio` (voice/format), and returns the generated audio on `choices[].message.audio`.

**Wrapper change**

- `ContentPart` gained `InputAudio *InputAudio` (`{data, format}`) and `File *FilePart` (`{file_id | filename + file_data}`), both `omitempty`; the string/array polymorphism stays on `Content`'s `MarshalJSON` / `UnmarshalJSON`, unchanged.
- `ChatRequest` gained `Modalities []string` and `Audio *AudioConfig{Voice, Format}`; `clone()` now deep-copies the `Modalities` slice.
- `Message` gained `Audio *MessageAudio` (`{id, data, transcript, expires_at}`), parsing assistant-generated audio.
- Anthropic translation untouched: no counterpart exists, and the new part types fall safely through its `text` / `image_url` switch.

## 2026-06-02 — Extend `ChatRequest` with common request fields (+ response `logprobs`)

**Official change**: Chat Completions exposes the request parameters `logprobs` / `top_logprobs`, `logit_bias`, `parallel_tool_calls`, `service_tier`, `store`, `metadata`, `prompt_cache_key`, and returns `choices[].logprobs` when `logprobs` is set.

**Wrapper change**

- `ChatRequest` gained eight `omitempty` fields: `Logprobs *bool`, `TopLogprobs *int`, `LogitBias map[string]int`, `ParallelToolCalls *bool`, `ServiceTier string`, `Store *bool`, `Metadata map[string]string`, `PromptCacheKey string`.
- `clone()` deep-copies `LogitBias` / `Metadata` via `maps.Clone`, so a copy's mutations never affect the original.
- Response side: `Choice.LogProbs *LogProbs`, with `LogProbs{Content, Refusal []TokenLogprob}`, `TokenLogprob{Token, Logprob, Bytes, TopLogprobs}`, `TopLogprob{Token, Logprob, Bytes}`.

`ParallelToolCalls` was later reused by the Anthropic translation as `disable_parallel_tool_use` — see the [Anthropic change log](../anthropic/anthropic-api-changes.md).

## 2026-06-02 — Sync `reasoning_effort` values, add `verbosity`

**Official change**: `reasoning_effort` accepted values extended to `none` / `minimal` / `low` / `medium` / `high` / `xhigh` (GPT-5.1 defaults to `none`); the new `verbosity` parameter (`low` / `medium` / `high`) controls how detailed the output is.

**Wrapper change**: added the `ReasoningEffort*` and `Verbosity*` constants; both fields stay `string` so they pass through to non-OpenAI backends. Added `ChatRequest.Verbosity string`; `clone()` needed no change (scalar field).

`ReasoningEffort` was later reused by the Anthropic translation — first as the top-level `effort`, now as `output_config.effort`.

## 2026-06-02 — Response type alignment (`reasoning_tokens`, `finish_reason` constants)

**Official change**: response `usage` now carries `completion_tokens_details.reasoning_tokens` (the internal thinking cost of reasoning models); `finish_reason` officially includes `content_filter` (and the legacy `function_call`).

**Wrapper change**: added `Usage.ReasoningTokens int`, parsed from the nested `completion_tokens_details.reasoning_tokens` (**an explicit top-level field wins**, mirroring `cached_tokens`); `Usage.Add` accumulates it. Added the `FinishReasonContentFilter` and `FinishReasonFunctionCall` (legacy compatibility) constants.

## 2026-06-02 — Support `max_completion_tokens`, deprecate `max_tokens`

**Official change**: OpenAI deprecated `max_tokens` on Chat Completions; reasoning models (o-series, GPT-5.x) reject it and require `max_completion_tokens`, whose limit covers both visible output tokens and internal reasoning tokens.

**Wrapper change**: added `ChatRequest.MaxCompletionTokens *int`; the retained `MaxTokens` is annotated as deprecated and incompatible with reasoning models. The Anthropic translator emits `max_tokens` preferring `MaxCompletionTokens` over `MaxTokens`, defaulting to 4096.

## [Baseline] 2026-06-02

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`, no standalone version number — keyed by the endpoint)
- **Summary**: wrapped the non-streaming `ChatCompletion` and streaming `ChatCompletionStream` (SSE), using the OpenAI-compatible format as the canonical representation.
