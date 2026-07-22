/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/vogo/aimodel/core"
)

// This file is the public Anthropic extension surface of the unified provider
// extension channel (core.Extensions). Request-side values configure
// Anthropic-only parameters without touching the canonical schema;
// response-side values carry Anthropic-only response metadata written by this
// provider's translators. Every value lives under the Name namespace of the
// node's Extensions map; wire types stay private to this package.
//
// Extension values are read-only once attached: the same value may be shared
// by a request and its pipeline clones, so callers must not mutate a value
// after setting it, and accessors return the stored value without copying.

// FinishReason values Anthropic surfaces verbatim — they have no OpenAI
// canonical equivalent, so mapAnthropicStopReason passes them through instead
// of folding them into stop/length/content_filter. Named here for
// readability; callers should treat any non-canonical FinishReason as opaque.
const (
	// FinishReasonModelContextWindowExceeded maps Anthropic's
	// "model_context_window_exceeded" (input + output exceeded the model's
	// context window — distinct from hitting the requested max_tokens /
	// "length").
	FinishReasonModelContextWindowExceeded core.FinishReason = "model_context_window_exceeded"
	// FinishReasonRefusal maps Anthropic's "refusal" (streaming classifiers
	// intervened on a potential policy violation). The classification, when
	// present, is carried on the choice's ChoiceExtension.StopDetails.
	FinishReasonRefusal core.FinishReason = "refusal"
	// FinishReasonPauseTurn maps Anthropic's "pause_turn" (a long-running
	// turn — e.g. a server-side tool — was paused; the client may replay it
	// to continue).
	FinishReasonPauseTurn core.FinishReason = "pause_turn"
)

// RequestExtension carries the Anthropic-only request parameters. Attach it
// with ExtendRequest; the translator reads it before building the wire body.
type RequestExtension struct {
	// AutoCache enables Anthropic's automatic prompt caching: a single
	// cache_control at the request root. The server places the cache
	// breakpoint on the last cacheable block and advances it forward as the
	// conversation grows — no per-block breakpoint needed. It coexists with
	// the explicit per-block MessageExtension / ToolExtension breakpoints.
	AutoCache bool

	// AutoCacheTTL selects the TTL for AutoCache. Empty means the default
	// 5-minute ephemeral cache; "1h" requests the 1-hour cache. Ignored when
	// AutoCache is false.
	AutoCacheTTL string

	// Container reuses a server-side execution container across requests,
	// keeping the code-execution state alive. Pass back the ID from a
	// previous response's ResponseExtension.Container. Empty omits the field.
	Container string

	// InferenceGeo pins where inference runs for data-residency purposes
	// (e.g. "us" / "eu"). Kept a plain string for pass-through; empty omits
	// the field.
	InferenceGeo string
}

// MessageExtension carries the Anthropic-only per-message extension. On the
// request side, CacheBreakpoint marks a prompt-cache boundary. On the
// response side, this provider stores the content blocks the canonical layer
// does not model in ExtraBlocks.
type MessageExtension struct {
	// CacheBreakpoint asks the translator to emit a cache boundary at the end
	// of this message's content blocks (cache_control on the last block).
	CacheBreakpoint bool

	// ExtraBlocks preserves the raw JSON of native content blocks the
	// canonical layer does not model — server-tool blocks (server_tool_use,
	// web_search_tool_result, code_execution_tool_result, …), any future
	// block type, and text blocks carrying citations (the text is still
	// extracted into Content; the whole original block is kept here so the
	// annotations are not lost).
	//
	// Elements are the verbatim response/SSE sub-objects in arrival order —
	// never re-marshalled — so unknown fields survive. Streaming appends the
	// original content_block of an unrecognized content_block_start and the
	// original delta of any subsequent unrecognized content_block_delta as
	// separate elements; callers reassemble them if they need to.
	ExtraBlocks []json.RawMessage
}

// MergeExtension implements core.ExtensionMerger so streaming deltas
// accumulate: ExtraBlocks concatenate in arrival order and the breakpoint
// flag sticks. It returns a fresh value — neither the receiver nor the delta
// is mutated, so previously delivered chunks stay intact.
func (e *MessageExtension) MergeExtension(delta any) any {
	d, ok := delta.(*MessageExtension)
	if !ok || d == nil {
		return e
	}

	merged := &MessageExtension{
		CacheBreakpoint: e.CacheBreakpoint || d.CacheBreakpoint,
	}

	merged.ExtraBlocks = make([]json.RawMessage, 0, len(e.ExtraBlocks)+len(d.ExtraBlocks))
	merged.ExtraBlocks = append(merged.ExtraBlocks, e.ExtraBlocks...)
	merged.ExtraBlocks = append(merged.ExtraBlocks, d.ExtraBlocks...)

	return merged
}

// ToolExtension carries the Anthropic-only per-tool parameters. Attach it
// with ExtendTool.
type ToolExtension struct {
	// CacheBreakpoint marks this tool as the end of a cacheable prefix:
	// Anthropic caches every tool up to and including the one flagged.
	CacheBreakpoint bool

	// DeferLoading keeps this tool's schema out of the initial context, to be
	// discovered on demand by Anthropic's tool-search tools. Not every tool
	// may be deferred — at least one must stay loaded.
	DeferLoading *bool

	// AllowedCallers restricts who may invoke this tool, e.g.
	// ["code_execution_20260120"] for programmatic tool calling (the model
	// calls it from inside the code-execution sandbox).
	AllowedCallers []string

	// EagerInputStreaming enables fine-grained streaming of this tool's
	// input (partial JSON is emitted as it is generated rather than
	// buffered).
	EagerInputStreaming *bool

	// InputExamples are sample tool inputs shown to the model to demonstrate
	// correct usage of a complex schema. Elements are arbitrary JSON values.
	InputExamples []any
}

// StopDetails carries Anthropic's structured stop classification, returned
// alongside stop_reason "refusal". All fields are best-effort and may be
// empty; Explanation in particular is not guaranteed stable across model
// versions.
type StopDetails struct {
	// Type discriminates the stop classification, e.g. "refusal".
	Type string `json:"type,omitempty"`
	// Category is the policy category that triggered the stop, e.g. "cyber" /
	// "bio"; empty when the stop maps to no named category.
	Category string `json:"category,omitempty"`
	// Explanation is a human-readable description of the stop; may be empty
	// and is not guaranteed stable across versions.
	Explanation string `json:"explanation,omitempty"`
}

// ChoiceExtension carries the Anthropic-only per-choice response metadata,
// written by this provider on core.Choice (unary) and on the terminal
// core.StreamChunkChoice (streaming).
type ChoiceExtension struct {
	// StopDetails is the structured stop classification (e.g. the refusal
	// category); nil when the response carries none.
	StopDetails *StopDetails
}

// ResponseContainer is the server-side execution container returned
// alongside a response. ExpiresAt is kept as the server-supplied string —
// this wrapper neither parses nor acts on the expiry.
type ResponseContainer struct {
	ID        string `json:"id"`
	ExpiresAt string `json:"expires_at,omitempty"`
}

// ResponseExtension carries the Anthropic-only response-level metadata,
// written by this provider on core.ChatResponse (unary) and on the
// core.StreamChunk that reports the message_start information (streaming).
type ResponseExtension struct {
	// Container identifies the server-side execution container this response
	// used. Pass its ID back via RequestExtension.Container to reuse the
	// container on the next turn; nil when the response carries none.
	Container *ResponseContainer
}

// ServerToolUse counts server-side tool invocations billed with a request.
type ServerToolUse struct {
	WebSearchRequests int
	WebFetchRequests  int
}

// UsageExtension carries the Anthropic-only usage accounting, written by this
// provider on core.Usage. The cross-provider counts (prompt/completion/cache
// read/reasoning tokens, service tier) stay on core.Usage itself.
type UsageExtension struct {
	// CacheWriteTokens reports tokens written to the prompt cache
	// (cache_creation_input_tokens, the total across TTLs). Like the
	// canonical CacheReadTokens it is a subset of PromptTokens.
	CacheWriteTokens int

	// CacheWrite5mTokens / CacheWrite1hTokens break CacheWriteTokens down by
	// TTL (usage.cache_creation ephemeral_5m/1h_input_tokens). Their sum
	// equals CacheWriteTokens; both stay zero when Anthropic returns no
	// breakdown.
	CacheWrite5mTokens int
	CacheWrite1hTokens int

	// ServerToolUse counts the server-side tool invocations billed with this
	// request (usage.server_tool_use); nil when the API reports none.
	ServerToolUse *ServerToolUse

	// InferenceGeo reports the geography inference actually ran in
	// (usage.inference_geo, e.g. "us").
	InferenceGeo string
}

// --- setters (request side) ---

// ExtendRequest attaches the Anthropic request extension to a canonical
// request. Passing nil removes a previously attached extension.
func ExtendRequest(r *core.ChatRequest, ext *RequestExtension) {
	setExtension(&r.Extensions, ext)
}

// ExtendMessage attaches the Anthropic message extension to a canonical
// message. Passing nil removes a previously attached extension.
func ExtendMessage(m *core.Message, ext *MessageExtension) {
	setExtension(&m.Extensions, ext)
}

// ExtendTool attaches the Anthropic tool extension to a canonical tool.
// Passing nil removes a previously attached extension.
func ExtendTool(t *core.Tool, ext *ToolExtension) {
	setExtension(&t.Extensions, ext)
}

// setExtension stores ext under this provider's namespace; a typed nil
// deletes the entry so the translator sees a genuinely absent extension.
func setExtension[T any](exts *core.Extensions, ext *T) {
	if ext == nil {
		delete(*exts, Name)

		return
	}

	exts.Set(Name, ext)
}

// --- accessors ---

// RequestExtensionOf returns the Anthropic request extension attached to r,
// or nil when absent. A value of any other type also yields nil — the
// translator rejects such a value with a *core.ExtensionTypeError before any
// network I/O, so it cannot silently take effect.
func RequestExtensionOf(r *core.ChatRequest) *RequestExtension {
	ext, _ := extensionOf[RequestExtension](r.Extensions, "")

	return ext
}

// MessageExtensionOf returns the Anthropic message extension attached to m,
// or nil when absent (same type contract as RequestExtensionOf).
func MessageExtensionOf(m *core.Message) *MessageExtension {
	ext, _ := extensionOf[MessageExtension](m.Extensions, "")

	return ext
}

// ToolExtensionOf returns the Anthropic tool extension attached to t, or nil
// when absent (same type contract as RequestExtensionOf).
func ToolExtensionOf(t *core.Tool) *ToolExtension {
	ext, _ := extensionOf[ToolExtension](t.Extensions, "")

	return ext
}

// ChoiceExtensionOf returns the Anthropic per-choice response metadata of a
// unary choice, or nil when the response carries none.
func ChoiceExtensionOf(c *core.Choice) *ChoiceExtension {
	ext, _ := extensionOf[ChoiceExtension](c.Extensions, "")

	return ext
}

// ChunkChoiceExtensionOf returns the Anthropic per-choice response metadata
// of a stream chunk choice (populated on the terminal chunk), or nil.
func ChunkChoiceExtensionOf(c *core.StreamChunkChoice) *ChoiceExtension {
	ext, _ := extensionOf[ChoiceExtension](c.Extensions, "")

	return ext
}

// ResponseExtensionOf returns the Anthropic response-level metadata of a
// unary response, or nil when the response carries none.
func ResponseExtensionOf(r *core.ChatResponse) *ResponseExtension {
	ext, _ := extensionOf[ResponseExtension](r.Extensions, "")

	return ext
}

// ChunkExtensionOf returns the Anthropic chunk-level metadata of a stream
// chunk (populated on the chunk carrying the message_start information), or
// nil.
func ChunkExtensionOf(c *core.StreamChunk) *ResponseExtension {
	ext, _ := extensionOf[ResponseExtension](c.Extensions, "")

	return ext
}

// UsageExtensionOf returns the Anthropic usage accounting attached to u, or
// nil when the response carries none.
func UsageExtensionOf(u *core.Usage) *UsageExtension {
	ext, _ := extensionOf[UsageExtension](u.Extensions, "")

	return ext
}

// extensionOf reads this provider's namespace from an extension map. A
// missing or nil entry is equivalent to a zero value (nil, no error). A value
// of any other type yields a *core.ExtensionTypeError naming the canonical
// node — the request translator propagates it before any network I/O; the
// public accessors drop it and report absence.
func extensionOf[T any](exts core.Extensions, node string) (*T, error) {
	v, ok := exts[Name]
	if !ok || v == nil {
		return nil, nil
	}

	ext, ok := v.(*T)
	if !ok {
		return nil, &core.ExtensionTypeError{
			Provider: Name,
			Node:     node,
			Want:     fmt.Sprintf("*%T", *new(T)),
			Value:    v,
		}
	}

	return ext, nil
}
