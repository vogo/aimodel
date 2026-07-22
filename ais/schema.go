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

package ais

import (
	"encoding/json"
	"strings"
)

// Role represents the role of a chat message participant.
type Role string

// Role constants for chat messages.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// FinishReason represents the reason a model stopped generating.
type FinishReason string

// FinishReason constants cover the values with a cross-provider consensus
// mapping. FinishReason stays an open string: providers pass values outside
// this set through verbatim (naming vendor-specific convenience constants is
// the provider package's job — e.g. anthropic.FinishReasonRefusal), and
// callers should treat any non-canonical value as opaque.
const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonContentFilter FinishReason = "content_filter"
	// FinishReasonFunctionCall is the legacy value emitted by the deprecated
	// OpenAI functions API; retained for backward compatibility.
	FinishReasonFunctionCall FinishReason = "function_call"
)

// ReasoningEffort constants enumerate the official OpenAI reasoning_effort
// values that constrain how many reasoning tokens a model spends. GPT-5.1
// defaults to ReasoningEffortNone. ChatRequest.ReasoningEffort stays a plain
// string so callers can pass through values these constants don't cover (other
// OpenAI-compatible backends may define their own).
const (
	ReasoningEffortNone    = "none"
	ReasoningEffortMinimal = "minimal"
	ReasoningEffortLow     = "low"
	ReasoningEffortMedium  = "medium"
	ReasoningEffortHigh    = "high"
	ReasoningEffortXHigh   = "xhigh"
)

// Thinking configures extended thinking (Anthropic) or reasoning (OpenAI-compatible) behavior.
type Thinking struct {
	// Type selects the thinking mode. Anthropic accepts "enabled", "disabled",
	// and (since the effort GA) "adaptive"; kept a plain string for pass-through.
	Type string `json:"type"`

	// BudgetTokens caps the thinking tokens for type "enabled".
	//
	// Deprecated: Anthropic's top-level `effort` parameter (mapped from
	// ChatRequest.ReasoningEffort) GA'd on 2026-02-05 and supersedes
	// budget_tokens for new models; prefer setting ReasoningEffort, or use
	// type "adaptive" to let the model size its own thinking. BudgetTokens
	// remains for models/callers that still pin an explicit budget.
	BudgetTokens int `json:"budget_tokens,omitempty"`

	// Display controls whether thinking content is streamed back. Set to
	// "omitted" (Anthropic, since 2026-03-16) to suppress thinking blocks and
	// speed up streaming; empty streams thinking as usual.
	Display string `json:"display,omitempty"`
}

// ChatRequest represents a request to the chat completions API.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`

	// MaxTokens is the legacy OpenAI cap on generated tokens.
	//
	// Deprecated: OpenAI has deprecated max_tokens on Chat Completions, and
	// reasoning models (the o-series, GPT-5.x, …) reject it outright — they
	// require max_completion_tokens instead. Keep MaxTokens only for older /
	// non-reasoning models that still accept it; new code should set
	// MaxCompletionTokens.
	MaxTokens *int `json:"max_tokens,omitempty"`

	// MaxCompletionTokens is the OpenAI cap that supersedes MaxTokens. Its
	// limit covers both visible output tokens and internal reasoning tokens,
	// and it is the only token-cap field accepted by reasoning models
	// (o-series, GPT-5.x, …). Prefer it over MaxTokens.
	MaxCompletionTokens *int `json:"max_completion_tokens,omitempty"`

	TopP *float64 `json:"top_p,omitempty"`

	// TopK limits sampling to the K most-likely tokens at each step (top-k
	// truncation). It maps directly to Anthropic's top_k. OpenAI's Chat
	// Completions has no top_k, so it is simply omitted there when unset and
	// passed through verbatim when set (OpenAI-compatible backends that do
	// accept it will honour it; the rest ignore the unknown field).
	TopK *int `json:"top_k,omitempty"`

	Stop           []string  `json:"stop,omitempty"`
	ResponseFormat any       `json:"response_format,omitempty"`
	Stream         bool      `json:"stream,omitempty"`
	Tools          []Tool    `json:"tools,omitempty"`
	ToolChoice     any       `json:"tool_choice,omitempty"`
	Thinking       *Thinking `json:"thinking,omitempty"`

	// ReasoningEffort controls how many reasoning tokens the model spends.
	// It maps to OpenAI's reasoning_effort and to Anthropic's top-level
	// effort (GA'd 2026-02-05, supersedes thinking.budget_tokens). Use the
	// ReasoningEffort* constants (none/minimal/low/medium/high/xhigh) or pass
	// any value a custom backend accepts.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	// ParallelToolCalls toggles whether the model may emit multiple tool
	// calls in a single turn. Defaults to true server-side when unset.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// Extensions is the unified provider extension channel for request-level
	// vendor parameters (e.g. anthropic.RequestExtension set via that
	// package's helpers). See the Extensions type for the contract; never
	// serialized on the canonical body.
	Extensions Extensions `json:"-"`
}

// ChatResponse represents a response from the chat completions API.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
	Error   *Error   `json:"error,omitempty"`

	// Extensions carries provider-scoped response metadata with no
	// cross-provider consensus (e.g. anthropic.ResponseExtension); read it
	// through the provider package's typed accessors.
	Extensions Extensions `json:"-"`
}

// Choice represents a single completion choice.
type Choice struct {
	Index        int          `json:"index"`
	Message      Message      `json:"message"`
	FinishReason FinishReason `json:"finish_reason"`
	// Extensions carries provider-scoped per-choice metadata (e.g.
	// anthropic.ChoiceExtension with the structured stop details).
	Extensions Extensions `json:"-"`
}

// Content represents chat message content that can be either a plain string
// or an array of content parts (text, image_url, etc.) for multimodal input.
type Content struct {
	text  string
	parts []ContentPart
}

// ContentPart represents a single part in a multimodal content array.
// Exactly one of the payload fields is set, selected by Type:
// "text" → Text, "image_url" → ImageURL.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL in a content part.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// NewTextContent creates a Content from a plain string.
func NewTextContent(text string) Content {
	return Content{text: text}
}

// NewPartsContent creates a Content from multiple content parts.
func NewPartsContent(parts ...ContentPart) Content {
	return Content{parts: parts}
}

// Parts returns the content parts for multimodal content, or nil for plain text content.
func (c Content) Parts() []ContentPart { return c.parts }

// Text returns the text content. For multimodal content, it concatenates all text parts.
func (c Content) Text() string {
	if c.parts == nil {
		return c.text
	}

	var b strings.Builder

	for _, p := range c.parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}

	return b.String()
}

// MarshalJSON implements json.Marshaler.
// Outputs a plain string when content is text-only, or an array for multimodal.
func (c Content) MarshalJSON() ([]byte, error) {
	if c.parts != nil {
		return json.Marshal(c.parts)
	}

	return json.Marshal(c.text)
}

// UnmarshalJSON implements json.Unmarshaler.
// Accepts both a plain string and an array of content parts.
func (c *Content) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		return json.Unmarshal(data, &c.text)
	}

	if len(data) > 0 && data[0] == '[' {
		return json.Unmarshal(data, &c.parts)
	}

	// Handle null.
	c.text = ""
	c.parts = nil

	return nil
}

// Message represents a chat message.
type Message struct {
	Role       Role       `json:"role"`
	Content    Content    `json:"content"`
	Thinking   string     `json:"reasoning_content,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`

	// Extensions carries provider-scoped message extensions: request-side
	// markers (e.g. the anthropic.MessageExtension cache breakpoint) and
	// response-side payloads the canonical layer does not model (e.g.
	// unrecognized native content blocks). Never serialized on the canonical
	// body; use the provider package's typed accessors.
	Extensions Extensions `json:"-"`
}

// AppendDelta merges a streaming delta message into this message. Provider
// extension namespaces are merged through the ExtensionMerger contract; the
// canonical layer does not interpret the values.
func (m *Message) AppendDelta(delta *Message) {
	m.Content.text += delta.Content.text
	m.Thinking += delta.Thinking
	m.Extensions.mergeDelta(delta.Extensions)

	for _, dtc := range delta.ToolCalls {
		idx := dtc.Index

		// Grow the slice with placeholders if needed.
		for idx >= len(m.ToolCalls) {
			m.ToolCalls = append(m.ToolCalls, ToolCall{Index: len(m.ToolCalls)})
		}

		m.ToolCalls[idx].Merge(&dtc)
	}
}

// ToolCall represents a function call requested by the model.
type ToolCall struct {
	Index    int          `json:"index"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// Merge appends delta data into this tool call.
func (tc *ToolCall) Merge(delta *ToolCall) {
	if delta.ID != "" {
		tc.ID = delta.ID
	}

	if delta.Type != "" {
		tc.Type = delta.Type
	}

	if delta.Function.Name != "" {
		tc.Function.Name = delta.Function.Name
	}

	tc.Function.Arguments += delta.Function.Arguments
}

// FunctionCall contains the function name and arguments.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// Tool represents a tool definition for the API.
type Tool struct {
	// Type is OpenAI's tool kind — "function" for every tool OpenAI defines.
	// For Anthropic it doubles as the tool type: "function" (and empty) means
	// the default custom tool and is not sent, while any other value is passed
	// through verbatim so versioned server tools ("web_search_20260209",
	// "code_execution_20260521", …) work without this wrapper enumerating them.
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`

	// Strict requests exact schema validation of the tool input, guaranteeing
	// the arguments match the declared schema (Anthropic's top-level `strict`
	// on the tool definition; OpenAI carries the same flag inside `function`).
	Strict *bool `json:"strict,omitempty"`

	// Extensions carries provider-scoped per-tool parameters (e.g. the
	// anthropic.ToolExtension cache breakpoint and tool controls). Never
	// serialized on the canonical body.
	Extensions Extensions `json:"-"`
}

// FunctionDefinition describes a function available to the model.
type FunctionDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// Clone returns a deep copy of the ChatRequest, duplicating slice and map
// fields so that mutations to the copy do not affect the original. The
// execution pipeline clones every caller request before handing the working
// copy to a provider, so providers may rewrite the copy without touching
// caller state. Extension maps are copied at every node (request, messages,
// tools); the extension values themselves are shared read-only configuration.
func (r *ChatRequest) Clone() ChatRequest {
	c := *r

	if len(r.Messages) > 0 {
		c.Messages = make([]Message, len(r.Messages))
		copy(c.Messages, r.Messages)

		for i := range c.Messages {
			c.Messages[i].Extensions = c.Messages[i].Extensions.Clone()
		}
	}

	if len(r.Stop) > 0 {
		c.Stop = make([]string, len(r.Stop))
		copy(c.Stop, r.Stop)
	}

	if len(r.Tools) > 0 {
		c.Tools = make([]Tool, len(r.Tools))
		copy(c.Tools, r.Tools)

		for i := range c.Tools {
			c.Tools[i].Extensions = c.Tools[i].Extensions.Clone()
		}
	}

	c.Extensions = r.Extensions.Clone()

	return c
}

// StreamChunk represents a single chunk in a streaming response.
type StreamChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []StreamChunkChoice `json:"choices"`
	Usage   *Usage              `json:"usage,omitempty"`

	// Extensions carries provider-scoped chunk-level metadata (e.g.
	// anthropic.ResponseExtension on the chunk that reports the execution
	// container); read it through the provider package's typed accessors.
	Extensions Extensions `json:"-"`
}

// StreamChunkChoice represents a choice within a stream chunk.
type StreamChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`

	// Extensions carries provider-scoped per-choice metadata (e.g. the
	// anthropic.ChoiceExtension stop details on the terminal chunk).
	Extensions Extensions `json:"-"`
}

// Usage tracks token usage for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// CacheReadTokens reports prompt tokens served from the provider's prompt
	// cache (OpenAI prompt_tokens_details.cached_tokens, Anthropic
	// cache_read_input_tokens). It is a subset of PromptTokens, surfaced
	// separately for observability.
	CacheReadTokens int `json:"cache_read_tokens,omitempty"`
	// ReasoningTokens reports tokens consumed by a reasoning model's internal
	// thinking, parsed from OpenAI's completion_tokens_details.reasoning_tokens
	// and from Anthropic's usage.output_tokens_details.thinking_tokens (an
	// explicit top-level reasoning_tokens takes precedence over either nested
	// source). Like the OpenAI count it is a subset of CompletionTokens.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`

	// ServiceTier reports the latency/throughput tier that served the request
	// (e.g. "standard", "priority", "batch"; OpenAI response service_tier,
	// Anthropic usage.service_tier). It describes one request and is not
	// combined by Usage.Add.
	ServiceTier string `json:"service_tier,omitempty"`

	// Extensions carries provider-scoped usage accounting with no
	// cross-provider consensus (e.g. anthropic.UsageExtension with the cache
	// write totals). It describes one request: Usage.Add does not combine it.
	Extensions Extensions `json:"-"`
}

// usageJSON is the JSON representation used for unmarshaling Usage,
// including nested OpenAI prompt_tokens_details.
type usageJSON struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	CacheReadTokens         int                      `json:"cache_read_tokens,omitempty"`
	ReasoningTokens         int                      `json:"reasoning_tokens,omitempty"`
	ServiceTier             string                   `json:"service_tier,omitempty"`
	PromptTokensDetails     *promptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// promptTokensDetails captures OpenAI's nested prompt token details.
type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// completionTokensDetails captures OpenAI's nested completion token details.
type completionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// UnmarshalJSON implements json.Unmarshaler for Usage.
// It extracts OpenAI's nested prompt_tokens_details.cached_tokens
// into CacheReadTokens when present.
func (u *Usage) UnmarshalJSON(data []byte) error {
	var raw usageJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	u.PromptTokens = raw.PromptTokens
	u.CompletionTokens = raw.CompletionTokens
	u.TotalTokens = raw.TotalTokens
	u.CacheReadTokens = raw.CacheReadTokens
	u.ReasoningTokens = raw.ReasoningTokens
	u.ServiceTier = raw.ServiceTier

	// Extract OpenAI's cached_tokens from nested prompt_tokens_details.
	if u.CacheReadTokens == 0 && raw.PromptTokensDetails != nil {
		u.CacheReadTokens = raw.PromptTokensDetails.CachedTokens
	}

	// Extract OpenAI's reasoning_tokens from nested completion_tokens_details.
	if u.ReasoningTokens == 0 && raw.CompletionTokensDetails != nil {
		u.ReasoningTokens = raw.CompletionTokensDetails.ReasoningTokens
	}

	return nil
}

// Add accumulates token counts from another Usage into this one. Counts are
// summed; ServiceTier and Extensions describe a single request and are left
// untouched.
func (u *Usage) Add(other *Usage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.TotalTokens += other.TotalTokens
	u.CacheReadTokens += other.CacheReadTokens
	u.ReasoningTokens += other.ReasoningTokens
}

// Error represents an error in the API response body.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	Type    string `json:"type"`
}
