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

	"github.com/vogo/aimodel/ais"
)

// --- Anthropic request types ---

type anthropicRequest struct {
	Model         string               `json:"model"`
	Messages      []anthropicMessage   `json:"messages"`
	System        json.RawMessage      `json:"system,omitempty"`
	MaxTokens     int                  `json:"max_tokens"`
	Temperature   *float64             `json:"temperature,omitempty"`
	TopP          *float64             `json:"top_p,omitempty"`
	TopK          *int                 `json:"top_k,omitempty"`
	StopSequences []string             `json:"stop_sequences,omitempty"`
	Stream        bool                 `json:"stream,omitempty"`
	Tools         []anthropicTool      `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice `json:"tool_choice,omitempty"`
	Thinking      *ais.Thinking        `json:"thinking,omitempty"`
	// Effort is Anthropic's former top-level reasoning-depth control.
	//
	// Deprecated: superseded by OutputConfig.Effort — reasoning depth now
	// lives inside output_config. Kept only so existing internal callers keep
	// compiling; toAnthropicRequest no longer assigns it, so it is never sent
	// alongside output_config.effort.
	Effort string `json:"effort,omitempty"`
	// OutputConfig carries Anthropic's output configuration: the reasoning
	// effort (mapped from ais.ChatRequest.ReasoningEffort) and the structured
	// output format (mapped from ais.ChatRequest.ResponseFormat). Omitted when
	// both are absent.
	OutputConfig *anthropicOutputConfig `json:"output_config,omitempty"`
	// Container reuses a server-side execution container across requests,
	// mapped straight through from RequestExtension.Container.
	Container string `json:"container,omitempty"`
	// InferenceGeo pins the inference geography for data residency, mapped
	// straight through from RequestExtension.InferenceGeo.
	InferenceGeo string `json:"inference_geo,omitempty"`
	// CacheControl, when set, is the request-root automatic-caching marker
	// (mapped from RequestExtension.AutoCache). The server caches the last
	// cacheable block and advances the breakpoint as the conversation grows.
	// It coexists with per-block cache_control markers.
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// anthropicOutputConfig is the request-side output configuration: how deeply
// the model reasons, and how its answer is shaped.
type anthropicOutputConfig struct {
	// Effort selects the reasoning depth (low/medium/high/xhigh/max).
	Effort string `json:"effort,omitempty"`
	// Format constrains the response to a schema (structured outputs).
	Format *anthropicOutputFormat `json:"format,omitempty"`
}

// anthropicOutputFormat is the structured-output format inside output_config.
// Schema holds the caller's JSON Schema as-is — this wrapper never validates
// or rewrites it.
type anthropicOutputFormat struct {
	Type   string `json:"type"`
	Schema any    `json:"schema,omitempty"`
}

type anthropicContentBlock struct {
	Type      string                  `json:"type"`
	Text      string                  `json:"text,omitempty"`
	Thinking  string                  `json:"thinking,omitempty"`
	ID        string                  `json:"id,omitempty"`
	Name      string                  `json:"name,omitempty"`
	Input     json.RawMessage         `json:"input,omitempty"`
	ToolUseID string                  `json:"tool_use_id,omitempty"`
	Source    *anthropicContentSource `json:"source,omitempty"`
	// ResultContent holds the content for tool_result blocks.
	ResultContent string `json:"content,omitempty"`
	// CacheControl, when set, marks this block as a prompt-cache
	// boundary. Anthropic caches everything up to and including this
	// block for the ephemeral TTL (default 5 minutes).
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

// anthropicCacheControl is the request-side prompt-cache hint.
type anthropicCacheControl struct {
	Type string `json:"type"` // always "ephemeral" today
	// TTL selects the cache lifetime: empty defaults to 5 minutes, "1h"
	// requests the 1-hour cache. Used by the request-root automatic-caching
	// marker; per-block markers leave it empty.
	TTL string `json:"ttl,omitempty"`
}

// ephemeralCache returns the canonical 5-minute ephemeral marker used on
// both message content blocks and tool definitions.
func ephemeralCache() *anthropicCacheControl {
	return &anthropicCacheControl{Type: "ephemeral"}
}

// anthropicContentSource represents the source of an image or document in Anthropic's API format.
// Supported source types: "base64", "url", "text", "content".
type anthropicContentSource struct {
	Type      string                  `json:"type"`
	MediaType string                  `json:"media_type,omitempty"`
	Data      string                  `json:"data,omitempty"`
	URL       string                  `json:"url,omitempty"`
	Content   []anthropicContentBlock `json:"content,omitempty"`
}

type anthropicTool struct {
	// Type selects the tool kind. Empty means the default custom tool;
	// versioned built-in types ("web_search_20260209",
	// "code_execution_20260521", …) pass through unvalidated.
	Type         string                 `json:"type,omitempty"`
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	InputSchema  any                    `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
	// The following mirror the canonical Tool fields one-to-one; see their
	// documentation on Tool (schema.go).
	Strict              *bool    `json:"strict,omitempty"`
	DeferLoading        *bool    `json:"defer_loading,omitempty"`
	AllowedCallers      []string `json:"allowed_callers,omitempty"`
	EagerInputStreaming *bool    `json:"eager_input_streaming,omitempty"`
	InputExamples       []any    `json:"input_examples,omitempty"`
}

type anthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
	// DisableParallelToolUse maps the canonical ParallelToolCalls=false:
	// when set true, Anthropic emits at most one tool call per turn.
	DisableParallelToolUse *bool `json:"disable_parallel_tool_use,omitempty"`
}

// --- Anthropic response types ---

type anthropicResponse struct {
	ID           string                   `json:"id"`
	Type         string                   `json:"type"`
	Role         string                   `json:"role"`
	Model        string                   `json:"model"`
	Content      []anthropicResponseBlock `json:"content"`
	StopReason   string                   `json:"stop_reason"`
	StopSequence *string                  `json:"stop_sequence"`
	// StopDetails carries the structured stop classification (e.g. the refusal
	// category) returned alongside stop_reason "refusal". The public extension
	// type's JSON tags match the wire shape, so it deserializes directly.
	StopDetails *StopDetails   `json:"stop_details"`
	Usage       anthropicUsage `json:"usage"`
	// Container is the server-side execution container the response used. The
	// public extension type's JSON tags match the wire shape; nil when absent
	// or null.
	Container *ResponseContainer `json:"container"`
}

// anthropicResponseBlock is a response-side content block: the known fields
// decoded into anthropicContentBlock, plus the verbatim JSON of the whole
// block. Keeping the original bytes is what lets unmodelled blocks (server
// tool results, future types) and text-block citations survive into
// Message.ExtraBlocks without a lossy decode/re-encode round trip.
type anthropicResponseBlock struct {
	anthropicContentBlock
	// Citations, when present, holds the raw citation annotations of a text
	// block. This wrapper does not interpret them — the field only signals
	// that the original block carries more than the extracted text.
	Citations json.RawMessage `json:"citations,omitempty"`

	// Content shadows the embedded block's request-side ResultContent (also
	// tagged "content", but a string). Response-side "content" is polymorphic
	// — server-tool result blocks carry an array, code-execution results an
	// object — so decoding it as a string would fail the whole response
	// instead of preserving the block. Shallower fields win in encoding/json,
	// so this one takes the value and ResultContent stays request-only.
	Content json.RawMessage `json:"content,omitempty"`

	raw json.RawMessage
}

// UnmarshalJSON decodes the known fields and retains the original bytes.
func (b *anthropicResponseBlock) UnmarshalJSON(data []byte) error {
	// alias drops the method set, so decoding it does not recurse.
	type alias anthropicResponseBlock

	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}

	*b = anthropicResponseBlock(a)
	b.raw = append(json.RawMessage(nil), data...)

	return nil
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	// CacheCreation breaks CacheCreationInputTokens down by TTL. Anthropic
	// returns it when 1-hour caching or mixed TTLs are in play; nil otherwise.
	CacheCreation *anthropicCacheCreation `json:"cache_creation,omitempty"`
	// OutputTokensDetails breaks the output tokens down; its thinking_tokens
	// is Anthropic's source for the canonical Usage.ReasoningTokens.
	OutputTokensDetails *anthropicOutputTokensDetails `json:"output_tokens_details,omitempty"`
	// ServerToolUse counts the server-side tool invocations billed with this
	// request; nil when no server tool ran.
	ServerToolUse *anthropicServerToolUse `json:"server_tool_use,omitempty"`
	// InferenceGeo reports the geography inference ran in (e.g. "us").
	InferenceGeo string `json:"inference_geo,omitempty"`
	// ServiceTier reports the tier that served the request.
	ServiceTier string `json:"service_tier,omitempty"`
}

// anthropicOutputTokensDetails is the per-category breakdown of output tokens.
type anthropicOutputTokensDetails struct {
	ThinkingTokens int `json:"thinking_tokens"`
}

// anthropicServerToolUse counts server-side tool invocations. Its fields match
// the canonical ServerToolUse exactly.
type anthropicServerToolUse struct {
	WebSearchRequests int `json:"web_search_requests"`
	WebFetchRequests  int `json:"web_fetch_requests"`
}

// anthropicCacheCreation is the per-TTL breakdown of cache writes; the two
// fields sum to cache_creation_input_tokens.
type anthropicCacheCreation struct {
	Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
}

// totalInputTokens returns the total input tokens including cached tokens.
func (u anthropicUsage) totalInputTokens() int {
	return u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
}

type anthropicErrorResponse struct {
	Type  string         `json:"type"`
	Error anthropicError `json:"error"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// --- Anthropic streaming types ---

type anthropicMessageStart struct {
	Type    string            `json:"type"`
	Message anthropicResponse `json:"message"`
}

type anthropicContentBlockStart struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	// ContentBlock retains its raw JSON so an unrecognized block can be
	// preserved verbatim on Message.ExtraBlocks.
	ContentBlock anthropicResponseBlock `json:"content_block"`
}

type anthropicContentBlockDelta struct {
	Type  string         `json:"type"`
	Index int            `json:"index"`
	Delta anthropicDelta `json:"delta"`
}

type anthropicDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`

	// raw is the verbatim delta sub-object, kept for the same reason as
	// anthropicResponseBlock.raw.
	raw json.RawMessage
}

// UnmarshalJSON decodes the known fields and retains the original bytes.
func (d *anthropicDelta) UnmarshalJSON(data []byte) error {
	type alias anthropicDelta

	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}

	*d = anthropicDelta(a)
	d.raw = append(json.RawMessage(nil), data...)

	return nil
}

type anthropicMessageDelta struct {
	Type  string                    `json:"type"`
	Delta anthropicMessageDeltaData `json:"delta"`
	Usage *anthropicUsage           `json:"usage,omitempty"`
}

type anthropicMessageDeltaData struct {
	StopReason   string       `json:"stop_reason,omitempty"`
	StopSequence *string      `json:"stop_sequence,omitempty"`
	StopDetails  *StopDetails `json:"stop_details,omitempty"`
}
