# OpenAI API — Change Log

This file records **how aimodel's OpenAI wrapper tracks the official OpenAI APIs**: what changed upstream, and how the wrapper followed.

- **Official protocols**: Chat Completions (`POST /chat/completions`) and, since 2026-08-01, Responses (`POST /v1/responses`) — neither carries a standalone version number, both are keyed by the endpoint
- **Official docs**: https://platform.openai.com/docs/api-reference/chat · https://platform.openai.com/docs/api-reference/responses
- **Implementation notes**: [openai-chat-api.md](./openai-chat-api.md) · [openai-response-api.md](./openai-response-api.md)

**Maintenance convention**: see [../architecture.md](../architecture.md) §6. Every entry carries at least a date, the official change, and a wrapper change summary.

Newest first.

---

## 2026-08-24 — Model recommendations move to the gpt-5.5 family; tool-calling constraints documented

**Official change**: OpenAI deprecates the first-gen gpt-5 / o3 dated snapshots — `gpt-5-2025-08-07`, `gpt-5-mini-2025-08-07`, `gpt-5-nano-2025-08-07`, `gpt-5-pro-2025-10-06`, `o3-2025-04-16`, `o3-pro-2025-06-10` — with retirement on **2026-12-11**, replaced by `gpt-5.5` / `gpt-5.5-pro`; the previous image-generation line (`gpt-image-1-mini`, `gpt-image-1.5`, `chatgpt-image-latest`) closes on **2026-12-01** in favour of `gpt-image-2`; the realtime preview models retired **2026-05-07**; and `gpt-4o` / `o3` / `o4-mini` are marked deprecated. On the Chat Completions endpoint, programmatic tool calling is not accepted, and on GPT-5.4+ setting `reasoning_effort` to `none` disables tool calling.

Verified **2026-08-24** against OpenAI's release notes and deprecation schedule: https://openai.com/index/introducing-gpt-5-5/ · https://platform.openai.com/docs/deprecations

**Wrapper change**

- **`ModelGPT55` / `ModelGPT55Pro` constants** added (`gpt-5.5` / `gpt-5.5-pro`); no further gpt-5.5 aliases are added because the official docs list none (mini / nano stay on the gpt-5.4 line). Existing constants, including the deprecated-but-still-served `gpt-4o` / `o3` / `o4-mini`, are kept as protocol facts.
- **Examples and recommended models** in `README.md`, `provider/openai/README.md` and `doc/design/compose.md` now recommend the gpt-5.5 family instead of `gpt-5` / `gpt-4o`.
- **Tool-calling constraints documented** in `doc/openai/openai-chat-api.md` §7 and the `ReasoningEffortNone` comment in `model.go`: both `tools` and `reasoning_effort` stay plain pass-throughs, and the conflicting combination is resolved server-side.
- **Assistants references cleaned up**: the migration link in `openai-response-api.md` is removed, and the 2026-08-01 entry now states the past-tense retirement date. Only out-of-scope boundary notes and the retired record remain.
- **Zero-reference audit**: `gpt-image-1*`, `chatgpt-image*` and `gpt-4o-realtime*` appear nowhere in this repo's code, examples or docs (this entry's retirement record aside) — confirmed, so no code change was needed.

---

## 2026-08-02 — Native-only public API: `ExtraBody`, stream accumulation, timeouts, structural errors

**Official change**: none — this is a change to the wrapper's own surface, recorded here because it changes how every OpenAI-side capability is reached.

**Wrapper change**

- **`openai` is the entry point.** `openai.NewClient(apiKey, ...)` with `ChatCompletions` / `ChatCompletionsStream` and `Responses` / `ResponsesStream`; there is no unified client and no translation in front of them. Reasoning: [ADR 0002](../adr/0002-provider-native-as-the-only-public-interface.md).
- **`ChatCompletionRequest.ExtraBody`**: a controlled channel for the private top-level parameters OpenAI-*compatible* backends add (`enable_thinking`, `chat_template_kwargs`, …). Additive only — a key colliding with a modelled field, an empty key, or a value that is not valid JSON fails at marshal time, before any network I/O. Decoding fills it with every unmodelled key, so a request body round-trips losslessly. The modelled key set is derived from the struct tags and cannot drift.
- **`ChatCompletionStream` accumulates while the caller reads**: `Response()` returns the assembled completion (text, reasoning content, refusals and tool-call arguments concatenated in arrival order; identity fields last-non-empty-wins; choices grown by index) and `Usage()` the token accounting.
- **`WithTimeout(d)`**: bounds a whole call. It copies the client configured so far, so the caller's `*http.Client` is never mutated and a transport from an earlier `WithHTTPClient` survives.
- **`HTTPError` implements `StatusCode() int`**. Go forbids a field and a method sharing a name, so the exported field is renamed `Status`. Consumers now match any provider's transport error through a locally declared `interface{ StatusCode() int }` without importing this package. **Breaking** for code reading the field directly.
- **Model, role, finish-reason, reasoning-effort, content-part and tool constants** move into this package (`model.go`), covering OpenAI and the OpenAI-compatible backends addressed over this protocol. Model names are protocol facts, not a vendor-neutral contract.
- **Request copying is shallow** rather than a JSON round trip when forcing the stream flag — faster, and it no longer erases `ExtraBody` in transit.

---

## 2026-08-01 — Support the Responses API: native `/v1/responses` client

**Official change**: OpenAI positions the Responses API (`POST /v1/responses`) as its primary interface. The Assistants API retired on **2026-08-26**, and hosted tools (`web_search` / `file_search` / `code_interpreter`), `previous_response_id` chaining, server-side conversations, reasoning items with encrypted content, and the newer prompt-cache controls exist only on Responses. Chat Completions keeps working but receives no new capabilities.

**Wire baseline verified 2026-08-01** — full inventory, official links and verification method in [openai-response-api.md](./openai-response-api.md).

**Wrapper change**

- **Native wire model** (`openai/responses_wire.go`, `responses_events.go`, `responses_const.go`): `ResponsesRequest` and `Response` covering every documented field; the polymorphic `ResponseInput` (string or item array) and `ResponseMessageContent` (string or content parts); discriminated `ResponseInputItem` / `ResponseOutputItem` unions with dedicated payloads for `message`, `reasoning`, `function_call`, `function_call_output`, `web_search_call`, `file_search_call`, `code_interpreter_call` and `item_reference`; the flat `ResponseContent` part type with annotations and logprobs; `ResponseTool` with typed `function` / `web_search` / `file_search` / `code_interpreter` parameters; `ResponseUsage` with `input_tokens_details.{cached_tokens,cache_write_tokens}` and `output_tokens_details.reasoning_tokens`; the 53-type SSE event taxonomy on `ResponseStreamEvent`, listed by `ResponseStreamEventTypes()`.
- **Forward compatibility**: an item, tool or event type outside the baseline keeps its verbatim payload on `Raw` and is re-encoded byte-for-byte. A **known** event that fails to decode is still an error, so drift on a modeled shape stays loud.
- **Native client** (`openai/responses.go`): `Client.Responses` and `Client.ResponsesStream`, both `POST {baseURL}/responses` with bearer auth, forcing the stream mode on a copy so the caller's request is never mutated. `ResponseStream.Recv` decodes the SSE event protocol (blank-line boundaries, comments, multi-line `data:`, payload `type` discriminator, no `[DONE]` sentinel), returns `io.EOF` at the end, turns an `error` event into `*HTTPError`, and closes the body exactly once on every terminal path. Non-2xx handling reuses the existing bounded reader and `*HTTPError` contract.
- **`Response.OutputText`**: an SDK convenience tagged `json:"-"`, derived at decode time from the `output_text` parts in output order; the items themselves are untouched and it is never serialized.
- **Tests**: offline throughout — request/response round-trip fixtures across every documented union, `httptest` non-streaming call, an SSE sweep over all 53 baseline events, typed payload dispatch, unknown-event preservation, `error` event, malformed known event, invalid JSON, oversized-line scan failure, idempotent close, and both structured and unstructured non-2xx bodies. Runnable native examples added under `integrations/openai_tests/`.
- **Out of scope**: `GET`/`DELETE`/cancel on `/v1/responses`, conversation-resource CRUD, background-response polling, and the Assistants and Realtime APIs.

## 2026-07-22 — Public native Chat Completions client

**Official change**: none — this is a wrapper restructuring, driven by the customization principle: every vendor exposes a public, full-fidelity native API.

**Wrapper change**

- **Native wire model** (`openai/wire.go`): independent exported types for the full Chat Completions surface — `ChatCompletionRequest` / `ChatCompletionResponse` / `ChatCompletionChunk`, `ChatCompletionMessage`, `ChatCompletionContentPart`, `ChatCompletionUsage` with its nested `PromptTokensDetails` / `CompletionTokensDetails`, tools, `StreamOptions`, `Thinking`, and the OpenAI-only members (log probabilities, audio/file, storage/metadata, prompt-cache routing, generation counts, request-side tier).
- **Native client** (`openai/native.go`): `openai.NewClient(apiKey, ...ClientOption)` with `WithBaseURL` / `WithHTTPClient`, `ChatCompletions` and `ChatCompletionsStream`. Native types end to end; default base URL `https://api.openai.com/v1`.
- **Integration examples**: native non-streaming and streaming samples added under `integrations/`.

## 2026-06-02 — Multimodal input/output (`input_audio` / `file` content parts, `modalities` / `audio`)

**Official change**: Chat Completions supports audio/file input via the `input_audio` and `file` content parts, requests audio output via `modalities` + `audio` (voice/format), and returns the generated audio on `choices[].message.audio`.

**Wrapper change**

- The content-part type gained `InputAudio` (`{data, format}`) and `File` (`{file_id | filename + file_data}`), both `omitempty`; the string/array polymorphism on the content field is unchanged.
- The request type gained `Modalities []string` and `Audio *AudioConfig{Voice, Format}`.
- The assistant message type gained `Audio` (`{id, data, transcript, expires_at}`), parsing generated audio.

## 2026-06-02 — Common request fields (+ response `logprobs`)

**Official change**: Chat Completions exposes the request parameters `logprobs` / `top_logprobs`, `logit_bias`, `parallel_tool_calls`, `service_tier`, `store`, `metadata`, `prompt_cache_key`, and returns `choices[].logprobs` when `logprobs` is set.

**Wrapper change**

- The request type gained eight `omitempty` fields: `Logprobs *bool`, `TopLogprobs *int`, `LogitBias map[string]int`, `ParallelToolCalls *bool`, `ServiceTier string`, `Store *bool`, `Metadata map[string]string`, `PromptCacheKey string`.
- Response side: the choice type gained `Logprobs`, with `ChoiceLogprobs{Content, Refusal []TokenLogprob}`, `TokenLogprob{Token, Logprob, Bytes, TopLogprobs}`, `TopLogprob{Token, Logprob, Bytes}`.

## 2026-06-02 — Sync `reasoning_effort` values, add `verbosity`

**Official change**: `reasoning_effort` accepted values extended to `none` / `minimal` / `low` / `medium` / `high` / `xhigh` (GPT-5.1 defaults to `none`); the new `verbosity` parameter (`low` / `medium` / `high`) controls how detailed the output is.

**Wrapper change**: added the `ReasoningEffort*` and `Verbosity*` constants; both fields stay `string` so they pass through to non-OpenAI backends. Added `Verbosity string` on the request type.

## 2026-06-02 — Response type alignment (`reasoning_tokens`, `finish_reason` constants)

**Official change**: response `usage` now carries `completion_tokens_details.reasoning_tokens` (the internal thinking cost of reasoning models); `finish_reason` officially includes `content_filter` (and the legacy `function_call`).

**Wrapper change**: added `ReasoningTokens` on the usage type, parsed from the nested `completion_tokens_details.reasoning_tokens`. Added the `FinishReasonContentFilter` and `FinishReasonFunctionCall` (legacy compatibility) constants.

## 2026-06-02 — Support `max_completion_tokens`, deprecate `max_tokens`

**Official change**: OpenAI deprecated `max_tokens` on Chat Completions; reasoning models (o-series, GPT-5.x) reject it and require `max_completion_tokens`, whose limit covers both visible output tokens and internal reasoning tokens.

**Wrapper change**: added `MaxCompletionTokens *int` on the request type; the retained `MaxTokens` is annotated as deprecated and incompatible with reasoning models.

## [Baseline] 2026-06-02

- **Official protocol**: OpenAI Chat Completions API (`/chat/completions`, no standalone version number — keyed by the endpoint)
- **Summary**: wrapped the non-streaming and streaming (SSE) Chat Completions calls over the OpenAI-compatible wire format.
