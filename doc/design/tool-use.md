# Tool Use

The canonical tool definition, tool choice, and the cross-protocol rules for parallel tool calls.

- **Canonical types**: `core/schema.go` (`Tool`, `FunctionDefinition`, `ToolCall`, `FunctionCall`)
- **Anthropic wire mapping**: [../anthropic/anthropic-message-api.md](../anthropic/anthropic-message-api.md) §3.5

---

## 1. `Tool`

```go
type Tool struct {
    Type     string             // see §1.1
    Function FunctionDefinition // {Name, Description, Parameters any}
    Strict   *bool              // exact input-schema validation (OpenAI + Anthropic)

    Extensions core.Extensions `json:"-"`  // provider extension channel
}

// Anthropic-only tool controls, attached via anthropic.ExtendTool:
type ToolExtension struct {  // package anthropic
    CacheBreakpoint     bool
    DeferLoading        *bool
    AllowedCallers      []string
    EagerInputStreaming *bool
    InputExamples       []any
}
```

### 1.1 `Type` doubles as the Anthropic tool type

`Type` is OpenAI's tool kind, which is `"function"` for every tool OpenAI defines. Anthropic instead uses `type` to select the tool *kind* — the default custom tool, or a versioned built-in.

The translation therefore treats **`"function"` and empty alike as Anthropic's default custom tool and does not send the field**; sending the literal `"function"` would be rejected. Any other value passes through verbatim, so versioned built-ins work without this wrapper enumerating them:

```go
{Type: "function", Function: …}              // → Anthropic default custom tool (type omitted)
{Type: "web_search_20260209", Function: …}   // → type: "web_search_20260209"
{Type: "code_execution_20260521", …}         // → type: "code_execution_20260521"
```

Built-in type names are neither enumerated nor validated — they are opaque strings, so a newly released tool version works without an SDK update.

### 1.2 Extension fields

`Strict` is canonical (OpenAI carries it inside `function`, Anthropic at the tool level). The Anthropic-only controls live on `anthropic.ToolExtension` and are copied verbatim into the Anthropic tool object:

| Field | Wire | Meaning |
|---|---|---|
| `Strict *bool` (canonical) | `strict` | Guarantee the tool input validates exactly against the declared schema. |
| `ToolExtension.CacheBreakpoint` | `cache_control` | Prompt-cache marker — see [prompt-caching.md](./prompt-caching.md). |
| `ToolExtension.DeferLoading` | `defer_loading` | Keep this tool's schema out of the initial context for on-demand discovery by tool search. At least one tool must stay loaded. |
| `ToolExtension.AllowedCallers` | `allowed_callers` | Restrict who may invoke the tool, e.g. `["code_execution_20260120"]` for programmatic tool calling. |
| `ToolExtension.EagerInputStreaming` | `eager_input_streaming` | Stream this tool's input as partial JSON instead of buffering it. |
| `ToolExtension.InputExamples` | `input_examples` | Sample inputs demonstrating a complex schema. |

`ChatRequest.Clone()` duplicates each tool's extension map; the extension value itself is shared read-only configuration ([data-model.md](./data-model.md) §1.9).

---

## 2. `ToolChoice`

The canonical field is `any`, accepting OpenAI's shapes. The Anthropic translation:

| Canonical value | Anthropic |
|---|---|
| `"auto"` | `{type:"auto"}` |
| `"required"` | `{type:"any"}` |
| `"none"` | `{type:"none"}` — **forbid every tool call**, which differs from omitting the field (model chooses) |
| `{"function":{"name":"x"}}` | `{type:"tool", name:"x"}` |
| anything else | `nil` (field omitted) |

### 2.1 `ParallelToolCalls` folds into the choice

Anthropic puts the switch **inside** `tool_choice`, so the canonical top-level `ParallelToolCalls` has to be folded in:

- Only an explicit `false` has any effect; it sets `disable_parallel_tool_use:true`.
- When no `tool_choice` was named but tools **are** present, the translator defaults to `{type:"auto"}` purely to carry the flag — a `tool_choice` on a tool-less request is rejected, so it is never fabricated there.
- It is **never** attached to `{type:"none"}`, where no call is allowed and the flag is meaningless.
- Unset or `true` leaves `tool_choice` completely untouched.

---

## 3. Tool results

A tool result is a canonical `Message` with `Role: RoleTool`, `ToolCallID` set to the originating call's ID, and the result text in `Content`. A missing `ToolCallID` is an error, not a silent skip.

### 3.1 Parallel results must share one message (Anthropic)

Anthropic requires **all** `tool_result` blocks for one assistant turn's parallel `tool_use` to arrive inside a **single** `role:"user"` message immediately after that turn. The naive 1:1 mapping of canonical `RoleTool` messages emits one user message per result, and the endpoint rejects the extras (`without tool_result blocks immediately after: …`).

The translator therefore detects a run of **consecutive** `RoleTool` messages and serializes it once, into one `role:"user"` message whose content array holds every `tool_result` block in the original order. A lone or non-consecutive tool result (separated by a user/assistant/system turn) keeps its own one-element user message.

The merge is driven purely by **adjacency in the input array** — no `tool_use` ID pairing or sorting is attempted, which keeps the rule predictable and order-preserving. A per-message cache breakpoint (`anthropic.MessageExtension`) survives the merge, so a cache boundary still lands on exactly the flagged block.

---

## 4. Server-side tools

Anthropic's server-executed tools (web search, web fetch, code execution) return content blocks this wrapper does not model — `server_tool_use`, `web_search_tool_result`, `code_execution_tool_result`, and so on. They are preserved verbatim on the message's Anthropic extension (`anthropic.MessageExtensionOf(&msg).ExtraBlocks`) rather than dropped; see [streaming.md](./streaming.md) §4.

Their billed invocation counts arrive on `anthropic.UsageExtensionOf(&usage).ServerToolUse` (`{WebSearchRequests, WebFetchRequests}`); see [data-model.md](./data-model.md) §4.
