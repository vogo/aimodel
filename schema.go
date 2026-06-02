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

package aimodel

import (
	"encoding/json"
	"maps"
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

// FinishReason constants.
const (
	FinishReasonStop          FinishReason = "stop"
	FinishReasonLength        FinishReason = "length"
	FinishReasonToolCalls     FinishReason = "tool_calls"
	FinishReasonContentFilter FinishReason = "content_filter"
	// FinishReasonFunctionCall is the legacy value emitted by the deprecated
	// OpenAI functions API; retained for backward compatibility.
	FinishReasonFunctionCall FinishReason = "function_call"

	// The following are Anthropic-specific stop_reason values surfaced verbatim
	// (they have no OpenAI canonical equivalent, so they pass through rather than
	// being folded into stop/length/content_filter). They are named here only for
	// readability; callers should treat any non-canonical FinishReason as opaque.

	// FinishReasonModelContextWindowExceeded maps Anthropic's
	// "model_context_window_exceeded" (input + output exceeded the model's
	// context window — distinct from hitting the requested max_tokens / "length").
	FinishReasonModelContextWindowExceeded FinishReason = "model_context_window_exceeded"
	// FinishReasonRefusal maps Anthropic's "refusal" (streaming classifiers
	// intervened on a potential policy violation). The classification, when
	// present, is carried on Choice.StopDetails.
	FinishReasonRefusal FinishReason = "refusal"
	// FinishReasonPauseTurn maps Anthropic's "pause_turn" (a long-running turn —
	// e.g. a server-side tool — was paused; the client may replay it to continue).
	FinishReasonPauseTurn FinishReason = "pause_turn"
)

// StopDetails carries Anthropic's structured stop classification, returned
// alongside stop_reason "refusal" (and surfaced on Choice / StreamChunkChoice).
// All fields are best-effort and may be empty; Explanation in particular is not
// guaranteed stable across model versions.
type StopDetails struct {
	// Type discriminates the stop classification, e.g. "refusal".
	Type string `json:"type,omitempty"`
	// Category is the policy category that triggered the stop, e.g. "cyber" /
	// "bio"; empty when the stop maps to no named category.
	Category string `json:"category,omitempty"`
	// Explanation is a human-readable description of the stop; may be empty and
	// is not guaranteed stable across versions.
	Explanation string `json:"explanation,omitempty"`
}

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

// Verbosity constants enumerate the official OpenAI verbosity values that
// control how detailed the model's output is. ChatRequest.Verbosity stays a
// plain string for the same pass-through reason as ReasoningEffort.
const (
	VerbosityLow    = "low"
	VerbosityMedium = "medium"
	VerbosityHigh   = "high"
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

// StreamOptions configures streaming behavior.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
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

	N                *int           `json:"n,omitempty"`
	Stop             []string       `json:"stop,omitempty"`
	FrequencyPenalty *float64       `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64       `json:"presence_penalty,omitempty"`
	Seed             *int           `json:"seed,omitempty"`
	User             string         `json:"user,omitempty"`
	ResponseFormat   any            `json:"response_format,omitempty"`
	Stream           bool           `json:"stream,omitempty"`
	StreamOptions    *StreamOptions `json:"stream_options,omitempty"`
	Tools            []Tool         `json:"tools,omitempty"`
	ToolChoice       any            `json:"tool_choice,omitempty"`
	Thinking         *Thinking      `json:"thinking,omitempty"`

	// ReasoningEffort controls how many reasoning tokens the model spends.
	// It maps to OpenAI's reasoning_effort and to Anthropic's top-level
	// effort (GA'd 2026-02-05, supersedes thinking.budget_tokens). Use the
	// ReasoningEffort* constants (none/minimal/low/medium/high/xhigh) or pass
	// any value a custom backend accepts.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`

	// Verbosity maps to OpenAI's verbosity and controls how detailed the
	// output is. Use the Verbosity* constants (low/medium/high).
	Verbosity string `json:"verbosity,omitempty"`

	// Logprobs requests per-token log probabilities in the response. When
	// true, each Choice carries a LogProbs payload.
	Logprobs *bool `json:"logprobs,omitempty"`

	// TopLogprobs (0–20) asks for that many most-likely tokens at each
	// position, each with a log probability. Requires Logprobs to be true.
	TopLogprobs *int `json:"top_logprobs,omitempty"`

	// LogitBias nudges the likelihood of specific tokens. Keys are token IDs
	// (as strings); values are bias amounts in [-100, 100].
	LogitBias map[string]int `json:"logit_bias,omitempty"`

	// ParallelToolCalls toggles whether the model may emit multiple tool
	// calls in a single turn. Defaults to true server-side when unset.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// ServiceTier selects the latency/throughput tier (e.g. "auto",
	// "default", "flex", "priority"). Kept a plain string for pass-through.
	ServiceTier string `json:"service_tier,omitempty"`

	// Store asks OpenAI to persist this completion for later retrieval
	// (e.g. dashboards, evals).
	Store *bool `json:"store,omitempty"`

	// Metadata attaches up to 16 string key/value pairs to the request,
	// surfaced alongside a stored completion.
	Metadata map[string]string `json:"metadata,omitempty"`

	// PromptCacheKey routes requests sharing a cacheable prefix to the same
	// cache, improving prompt cache hit rates.
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`

	// Modalities lists the output types the model may generate, e.g.
	// ["text"] or ["text", "audio"]. Required when requesting audio output.
	Modalities []string `json:"modalities,omitempty"`

	// Audio configures audio output (voice and format); required when
	// "audio" is included in Modalities.
	Audio *AudioConfig `json:"audio,omitempty"`
}

// AudioConfig configures the audio output of a request (ChatRequest.Audio).
// Voice selects the speaker (e.g. "alloy"); Format is the output encoding
// (e.g. "wav", "mp3", "flac", "opus", "pcm16"). Both stay plain strings for
// pass-through to OpenAI-compatible backends.
type AudioConfig struct {
	Voice  string `json:"voice"`
	Format string `json:"format"`
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
}

// Choice represents a single completion choice.
type Choice struct {
	Index        int          `json:"index"`
	Message      Message      `json:"message"`
	FinishReason FinishReason `json:"finish_reason"`
	// LogProbs carries per-token log probabilities when the request set
	// Logprobs to true; nil otherwise.
	LogProbs *LogProbs `json:"logprobs,omitempty"`
	// StopDetails carries Anthropic's structured stop classification (e.g. the
	// refusal category); nil when absent.
	StopDetails *StopDetails `json:"stop_details,omitempty"`
}

// LogProbs holds the token log-probability payload for a Choice, mirroring
// OpenAI's choices[].logprobs object.
type LogProbs struct {
	Content []TokenLogprob `json:"content,omitempty"`
	Refusal []TokenLogprob `json:"refusal,omitempty"`
}

// TokenLogprob is the log probability of a single output token, plus the
// most-likely alternatives at that position (TopLogprobs).
type TokenLogprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes,omitempty"`
	TopLogprobs []TopLogprob `json:"top_logprobs,omitempty"`
}

// TopLogprob is one alternative token and its log probability at a position.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// Content represents chat message content that can be either a plain string
// or an array of content parts (text, image_url, etc.) for multimodal input.
type Content struct {
	text  string
	parts []ContentPart
}

// ContentPart represents a single part in a multimodal content array.
// Exactly one of the payload fields is set, selected by Type:
// "text" → Text, "image_url" → ImageURL, "input_audio" → InputAudio,
// "file" → File.
type ContentPart struct {
	Type       string      `json:"type"`
	Text       string      `json:"text,omitempty"`
	ImageURL   *ImageURL   `json:"image_url,omitempty"`
	InputAudio *InputAudio `json:"input_audio,omitempty"`
	File       *FilePart   `json:"file,omitempty"`
}

// ImageURL represents an image URL in a content part.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// InputAudio represents an audio input in a content part (type "input_audio").
// Data is the base64-encoded audio; Format is the encoding ("wav" or "mp3").
type InputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

// FilePart represents a file input in a content part (type "file"). Reference
// an already-uploaded file via FileID, or inline one with Filename + FileData
// (base64-encoded contents).
type FilePart struct {
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
	FileData string `json:"file_data,omitempty"`
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

	// Audio carries the generated audio of an assistant message when audio
	// output was requested (ChatRequest.Modalities includes "audio").
	Audio *MessageAudio `json:"audio,omitempty"`

	// CacheBreakpoint tells protocol backends that support explicit prompt
	// caching (Anthropic) to emit a cache boundary at the end of this
	// message's content blocks. OpenAI-compatible backends silently ignore
	// the field — OpenAI caches 1024-token+ prefixes automatically with no
	// request-side marker.
	//
	// Struct-local: marked `json:"-"` so the canonical (OpenAI-shape)
	// request body never carries the field. Only the Anthropic translator
	// reads it.
	CacheBreakpoint bool `json:"-"`
}

// AppendDelta merges a streaming delta message into this message.
func (m *Message) AppendDelta(delta *Message) {
	m.Content.text += delta.Content.text
	m.Thinking += delta.Thinking

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

// MessageAudio is the generated audio attached to an assistant message
// (Message.Audio), mirroring OpenAI's message.audio object. Data is the
// base64-encoded audio; Transcript is its text transcript; ID references the
// audio for multi-turn reuse; ExpiresAt is its Unix expiry time.
type MessageAudio struct {
	ID         string `json:"id,omitempty"`
	Data       string `json:"data,omitempty"`
	Transcript string `json:"transcript,omitempty"`
	ExpiresAt  int64  `json:"expires_at,omitempty"`
}

// Tool represents a tool definition for the API.
type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`

	// CacheBreakpoint marks this tool as the end of a cacheable prefix for
	// Anthropic requests. Anthropic caches every tool up to and including
	// the one flagged. OpenAI-compatible backends ignore the field.
	// Struct-local: never serialised on the canonical request body.
	CacheBreakpoint bool `json:"-"`
}

// FunctionDefinition describes a function available to the model.
type FunctionDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// clone returns a deep copy of the ChatRequest, duplicating slice fields
// so that mutations to the copy do not affect the original.
func (r *ChatRequest) clone() ChatRequest {
	c := *r

	if len(r.Messages) > 0 {
		c.Messages = make([]Message, len(r.Messages))
		copy(c.Messages, r.Messages)
	}

	if len(r.Stop) > 0 {
		c.Stop = make([]string, len(r.Stop))
		copy(c.Stop, r.Stop)
	}

	if len(r.Modalities) > 0 {
		c.Modalities = make([]string, len(r.Modalities))
		copy(c.Modalities, r.Modalities)
	}

	if len(r.Tools) > 0 {
		c.Tools = make([]Tool, len(r.Tools))
		copy(c.Tools, r.Tools)
	}

	if len(r.LogitBias) > 0 {
		c.LogitBias = maps.Clone(r.LogitBias)
	}

	if len(r.Metadata) > 0 {
		c.Metadata = maps.Clone(r.Metadata)
	}

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
}

// StreamChunkChoice represents a choice within a stream chunk.
type StreamChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`
	// StopDetails carries Anthropic's structured stop classification (e.g. the
	// refusal category) on the terminal message_delta; nil when absent.
	StopDetails *StopDetails `json:"stop_details,omitempty"`
}

// Usage tracks token usage for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	// ReasoningTokens reports tokens consumed by a reasoning model's internal
	// thinking, parsed from OpenAI's completion_tokens_details.reasoning_tokens.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// usageJSON is the JSON representation used for unmarshaling Usage,
// including nested OpenAI prompt_tokens_details.
type usageJSON struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	CacheReadTokens         int                      `json:"cache_read_tokens,omitempty"`
	ReasoningTokens         int                      `json:"reasoning_tokens,omitempty"`
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

// Add accumulates token counts from another Usage into this one.
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
