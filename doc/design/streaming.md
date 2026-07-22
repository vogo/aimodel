# Streaming

The protocol-independent streaming abstraction, delta merging, unmodelled-block preservation, and stream interception.

- **Implementation**: `stream.go`, `intercept.go` (the `Stream` type and interception; both root); per-event decoding lives in each provider's `stream.go` behind `core.StreamDecoder`
- **Per-protocol SSE parsing**: [../openai/openai-chat-api.md](../openai/openai-chat-api.md) §4.1 · [../anthropic/anthropic-message-api.md](../anthropic/anthropic-message-api.md) §5

---

## 1. `Stream`

```go
type Stream struct { /* mu sync.Mutex; closed atomic.Bool; recv func() (*StreamChunk, error); usage *Usage; onClose func(*Usage) */ }

func (s *Stream) Recv() (*StreamChunk, error)  // io.EOF signals normal end
func (s *Stream) Usage() *Usage
func (s *Stream) Close() error                 // idempotent, safe alongside Recv
```

Design points:

- **Protocol differences are absorbed by the provider's SSE decoder.** The `Stream` struct itself is protocol-agnostic: it wraps a `core.StreamDecoder` (whose `Next()` becomes the stream's `recv`). The OpenAI provider's decoder does line-by-line `data:` parsing (`[DONE]` → `io.EOF`); the Anthropic provider's decoder does paired `event:` + `data:` parsing (`message_stop` → `io.EOF`).
- **Concurrency safety**: `Recv` serializes on a mutex; `Close` uses `CompareAndSwap` to run exactly once and **closes the underlying reader directly** to unblock an in-flight `Recv` (`http.Response.Body.Close` is safe to call concurrently). After closing, `Recv` returns `ErrStreamClosed`.
- **Usage capture**: any chunk carrying a `Usage` is recorded into `s.usage`; `Usage()` returns it once the stream ends.
- **Container ID**: `StreamChunk.Container` (`*ResponseContainer`) is emitted **once**, on the Anthropic path, as soon as `message_start` is read — see §3.
- **SSE line cap** `maxStreamLineSize = 1 MB` (`bufio.Scanner` buffer starts at 64 KB).

`StreamChunk` mirrors `ChatResponse` for the incremental case: `{ID, Object, Created, Model, Choices []StreamChunkChoice, Usage *Usage, Container *ResponseContainer}`, with `StreamChunkChoice{Index, Delta Message, FinishReason *string, StopDetails *StopDetails}`.

---

## 2. Delta merging

```go
func (m *Message) AppendDelta(delta *Message)
func (tc *ToolCall) Merge(delta *ToolCall)
```

`AppendDelta` concatenates text and thinking, and merges tool calls in place by `ToolCall.Index` (growing the slice with placeholder elements when needed). `Merge` uses "non-empty overwrite" for ID / Type / Name and **string append** for `Arguments`, because tool-argument JSON is streamed in fragments.

`AppendDelta` also appends `Message.ExtraBlocks` in arrival order — see §4.

---

## 3. Container ID is emitted early

The Anthropic execution container is already known at `message_start`, but the streaming path never produces a `ChatResponse` — so if the ID only rode along with a text delta, a stream that produces only tool events (or ends immediately) would lose it entirely, and the caller would have no way to obtain the ID it needs to reuse the container next turn.

`StreamChunk.Container` therefore carries it, and the parser emits a chunk holding just that field the moment `message_start` is parsed. Ordinary deltas never repeat it — it appears exactly once per stream, and not at all when the response carries no container.

---

## 4. Unmodelled content blocks (`ExtraBlocks`)

`Message.ExtraBlocks []json.RawMessage` (`json:"-"`, runtime-only) preserves protocol content blocks this wrapper does not model, instead of dropping them silently. Elements are the **verbatim** response / SSE sub-objects — never decoded-then-re-marshalled, so unknown nested fields survive intact.

What lands there (Anthropic path):

| Source | Preserved |
|---|---|
| Non-streaming: an unrecognized block in `content[]` | The whole original block (`server_tool_use`, `web_search_tool_result`, `code_execution_tool_result`, any future type) |
| Non-streaming: a `text` block carrying `citations` | The whole original block — **in addition to** contributing its text to `Content` |
| Streaming: an unrecognized `content_block_start` | Its original `content_block` |
| Streaming: any later delta on that block's index | Its original `delta` |
| Streaming: an unknown delta type on a **known** block | Its original `delta` |

The wrapper deliberately does **not** try to reassemble a complete block from the streamed start + deltas. Guessing at merge rules for a type it does not understand would corrupt the data; emitting each event's raw sub-object in arrival order is lossless and stable, at the cost of the caller understanding the increment order. `AppendDelta` appends them without parsing, merging, or rewriting, so callers reassemble them however they need.

Regression guarantees: `signature_delta` stays ignored, and `text` / `thinking` / `tool_use` / `input_json_delta` on known blocks behave exactly as before.

---

## 5. Stream interception (`intercept.go`)

Two **additive** decorators that do not change the consumer-side API:

```go
// Callback on close, carrying the final usage.
func WrapStream(s *Stream, onClose func(*Usage)) *Stream

// Per-chunk observation plus a one-shot completion callback.
func InterceptStream(s *Stream, onChunk func(*StreamChunk), onDone func(err error)) *Stream
```

`InterceptStream` wraps `s.recv`: every non-nil chunk fires `onChunk`; `onDone` is guarded by `sync.Once` and fires **exactly once** — on the first non-nil error (including `io.EOF`) or on `Close`, whichever comes first. It chains any previously installed `onClose`. Both decorators defend against `s == nil` by firing the callbacks with zero values immediately and returning `nil`.

Constraints: callbacks must be cheap, and **must not call `Recv` / `Close`** (that deadlocks).
