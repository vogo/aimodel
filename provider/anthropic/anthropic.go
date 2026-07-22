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
	"strings"

	"github.com/vogo/aimodel/ais"
)

// Anthropic Messages API reference: https://platform.claude.com/docs/en/api/messages
const (
	anthropicDefaultBaseURL   = "https://api.anthropic.com"
	anthropicAPIVersion       = "2023-06-01"
	anthropicDefaultMaxTokens = 4096
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

// --- Translation functions ---

// toAnthropicRequest converts a ais.ChatRequest to an Anthropic API request.
// Anthropic-only parameters arrive through the unified extension channel
// (RequestExtension / MessageExtension / ToolExtension); a mis-typed
// extension value fails here — before any network I/O — with a
// *ais.ExtensionTypeError.
func toAnthropicRequest(req *ais.ChatRequest) (*anthropicRequest, error) {
	reqExt, err := extensionOf[RequestExtension](req.Extensions, "ChatRequest")
	if err != nil {
		return nil, err
	}

	if reqExt == nil {
		reqExt = &RequestExtension{}
	}

	ar := &anthropicRequest{
		Model:        req.Model,
		Temperature:  req.Temperature,
		TopP:         req.TopP,
		TopK:         req.TopK,
		Stream:       req.Stream,
		Container:    reqExt.Container,
		InferenceGeo: reqExt.InferenceGeo,
	}

	// MaxTokens: Anthropic always uses max_tokens. Prefer the newer
	// MaxCompletionTokens (OpenAI's reasoning-model field) over the deprecated
	// MaxTokens, falling back to the default when neither is set.
	// Reading the deprecated canonical MaxTokens is the intended
	// backward-compat fallback for older / non-reasoning models.
	switch {
	case req.MaxCompletionTokens != nil:
		ar.MaxTokens = *req.MaxCompletionTokens
	case req.MaxTokens != nil: //nolint:staticcheck // deprecated field read on purpose
		ar.MaxTokens = *req.MaxTokens //nolint:staticcheck // deprecated field read on purpose
	default:
		ar.MaxTokens = anthropicDefaultMaxTokens
	}

	// Stop -> StopSequences.
	if len(req.Stop) > 0 {
		ar.StopSequences = req.Stop
	}

	// Extract system messages and convert the rest.
	var systemTexts []string

	var systemBlocks []anthropicContentBlock

	useBlocks := false

	// anyCacheableSystem tracks whether any system message asked for a
	// cache breakpoint; when true we force the block-array form on the
	// system field so we can attach cache_control to the last block.
	var anyCacheableSystem bool

	// seenNonSystem flips once we pass the first non-system message. Only
	// the leading run of system messages (before any user/assistant turn)
	// is hoisted into the top-level `system` field; system messages that
	// appear mid-conversation are kept inline as role:"system" Anthropic
	// messages (supported since Opus 4.8 / 2026-05-28) so their position
	// and prompt-cache semantics are preserved.
	var seenNonSystem bool

	for i := 0; i < len(req.Messages); i++ {
		m := req.Messages[i]
		if m.Role == ais.RoleSystem && !seenNonSystem {
			bp, err := messageCacheBreakpoint(&m)
			if err != nil {
				return nil, err
			}

			if bp {
				anyCacheableSystem = true
				useBlocks = true
			}
			if parts := m.Content.Parts(); len(parts) > 0 {
				useBlocks = true

				for _, p := range parts {
					if p.Type == "text" {
						systemBlocks = append(systemBlocks, anthropicContentBlock{
							Type: "text",
							Text: p.Text,
						})
					}
				}
			} else {
				text := m.Content.Text()
				systemTexts = append(systemTexts, text)
				systemBlocks = append(systemBlocks, anthropicContentBlock{
					Type: "text",
					Text: text,
				})
			}

			continue
		}

		// Any non-system message ends the leading system run; subsequent
		// system messages fall through to toAnthropicMessage and stay
		// inline as role:"system".
		if m.Role != ais.RoleSystem {
			seenNonSystem = true
		}

		// Consecutive ais.RoleTool messages represent the parallel tool_result
		// blocks for one assistant turn. Anthropic requires all of them to
		// arrive inside a single role:"user" message immediately after the
		// assistant turn — emitting one user message per result (the naive
		// 1:1 mapping) makes the endpoint reject the rest as missing
		// tool_result blocks. Collect the whole run and serialize it once.
		if m.Role == ais.RoleTool {
			runStart := i
			for i+1 < len(req.Messages) && req.Messages[i+1].Role == ais.RoleTool {
				i++
			}

			am, err := toAnthropicToolResultMessage(req.Messages[runStart : i+1])
			if err != nil {
				return nil, err
			}

			ar.Messages = append(ar.Messages, am)

			continue
		}

		am, err := toAnthropicMessage(m)
		if err != nil {
			return nil, err
		}

		ar.Messages = append(ar.Messages, am)
	}

	if (useBlocks || anyCacheableSystem) && len(systemBlocks) > 0 {
		if anyCacheableSystem {
			// Attach cache_control to the last block — Anthropic caches
			// every block up to and including it.
			systemBlocks[len(systemBlocks)-1].CacheControl = ephemeralCache()
		}

		data, err := json.Marshal(systemBlocks)
		if err != nil {
			return nil, fmt.Errorf("aimodel: marshal system content: %w", err)
		}

		ar.System = data
	} else if len(systemTexts) > 0 {
		data, err := json.Marshal(strings.Join(systemTexts, "\n"))
		if err != nil {
			return nil, fmt.Errorf("aimodel: marshal system text: %w", err)
		}

		ar.System = data
	}

	// Convert tools. Anthropic-only tool controls (cache breakpoint, defer
	// loading, allowed callers, eager input streaming, input examples) arrive
	// on each tool's ToolExtension; a flagged breakpoint carries a
	// cache_control marker — Anthropic caches every tool up to and including
	// the flagged one.
	for _, t := range req.Tools {
		tExt, err := extensionOf[ToolExtension](t.Extensions, "Tool")
		if err != nil {
			return nil, err
		}

		if tExt == nil {
			tExt = &ToolExtension{}
		}

		at := anthropicTool{
			Name:                t.Function.Name,
			Description:         t.Function.Description,
			InputSchema:         t.Function.Parameters,
			Strict:              t.Strict,
			DeferLoading:        tExt.DeferLoading,
			AllowedCallers:      tExt.AllowedCallers,
			EagerInputStreaming: tExt.EagerInputStreaming,
			InputExamples:       tExt.InputExamples,
		}

		// "function" is OpenAI's only tool kind and the canonical default, so
		// it maps to Anthropic's default custom tool — omitted rather than
		// sent as a bogus type. Any other value (a versioned built-in tool)
		// passes through unvalidated.
		if t.Type != "" && t.Type != "function" {
			at.Type = t.Type
		}

		if tExt.CacheBreakpoint {
			at.CacheControl = ephemeralCache()
		}
		ar.Tools = append(ar.Tools, at)
	}

	// Convert tool choice, folding in ParallelToolCalls=false as
	// disable_parallel_tool_use. The flag lives inside tool_choice, so when
	// the caller disables parallel calls without naming a choice we default
	// to type "auto" to carry it — but only when tools are present, since a
	// tool_choice on a tool-less request is rejected. The flag is meaningless
	// for type "none" (no calls at all), so it is never attached there.
	tc := convertToolChoice(req.ToolChoice)
	if req.ParallelToolCalls != nil && !*req.ParallelToolCalls {
		if tc == nil && len(req.Tools) > 0 {
			tc = &anthropicToolChoice{Type: "auto"}
		}
		if tc != nil && tc.Type != "none" {
			disable := true
			tc.DisableParallelToolUse = &disable
		}
	}
	ar.ToolChoice = tc

	// Pass through thinking configuration.
	ar.Thinking = req.Thinking

	// Output configuration: the canonical reasoning effort and a JSON-schema
	// response format both live under output_config. The deprecated top-level
	// effort field is deliberately left unset so only one of the two is sent.
	if oc := toAnthropicOutputConfig(req); oc != nil {
		ar.OutputConfig = oc
	}

	// Automatic caching: a single request-root cache_control. The server
	// caches the last cacheable block and advances the breakpoint as the
	// conversation grows. Coexists with per-block breakpoint markers.
	if reqExt.AutoCache {
		ar.CacheControl = &anthropicCacheControl{Type: "ephemeral", TTL: reqExt.AutoCacheTTL}
	}

	return ar, nil
}

// messageCacheBreakpoint reports whether the message's Anthropic extension
// asks for a prompt-cache boundary, failing on a mis-typed extension value.
func messageCacheBreakpoint(m *ais.Message) (bool, error) {
	ext, err := extensionOf[MessageExtension](m.Extensions, "Message")
	if err != nil {
		return false, err
	}

	return ext != nil && ext.CacheBreakpoint, nil
}

// toAnthropicOutputConfig builds the output_config object from the canonical
// reasoning effort and response format. Either half may be absent; when both
// are, it returns nil so the field is omitted entirely.
func toAnthropicOutputConfig(req *ais.ChatRequest) *anthropicOutputConfig {
	format := toAnthropicOutputFormat(req.ResponseFormat)
	if req.ReasoningEffort == "" && format == nil {
		return nil
	}

	return &anthropicOutputConfig{
		Effort: req.ReasoningEffort,
		Format: format,
	}
}

// toAnthropicOutputFormat translates a canonical ResponseFormat into
// Anthropic's structured-output format. Only JSON-schema shapes are
// recognized: OpenAI's {type:"json_schema", json_schema:{schema:…}} and the
// flat {type:"json_schema", schema:…}. Anything else — including
// {type:"json_object"}, which has no Anthropic counterpart — yields nil rather
// than a fabricated config, keeping this a thin translation.
func toAnthropicOutputFormat(rf any) *anthropicOutputFormat {
	m, ok := rf.(map[string]any)
	if !ok {
		return nil
	}

	if t, _ := m["type"].(string); t != "json_schema" {
		return nil
	}

	schema := m["schema"]
	if nested, ok := m["json_schema"].(map[string]any); ok {
		schema = nested["schema"]
	}

	if schema == nil {
		return nil
	}

	return &anthropicOutputFormat{Type: "json_schema", Schema: schema}
}

// toolResultBlock builds a single tool_result content block from a canonical
// tool-result message. It is shared by the consecutive-run merge path and the
// single-message fallback so the wire shape stays identical. A missing
// ToolCallID is rejected up front.
func toolResultBlock(m ais.Message) (anthropicContentBlock, error) {
	if m.ToolCallID == "" {
		return anthropicContentBlock{}, fmt.Errorf("aimodel: tool result message missing tool_call_id")
	}

	block := anthropicContentBlock{
		Type:          "tool_result",
		ToolUseID:     m.ToolCallID,
		ResultContent: m.Content.Text(),
	}

	bp, err := messageCacheBreakpoint(&m)
	if err != nil {
		return anthropicContentBlock{}, err
	}

	if bp {
		block.CacheControl = ephemeralCache()
	}

	return block, nil
}

// toAnthropicToolResultMessage serializes a run of one or more consecutive
// canonical tool-result messages into a single Anthropic role:"user" message
// whose content array holds all the tool_result blocks in order. Anthropic
// requires the parallel results of one assistant turn to share one user
// message; merging here keeps the request valid for parallel tool use.
func toAnthropicToolResultMessage(msgs []ais.Message) (anthropicMessage, error) {
	blocks := make([]anthropicContentBlock, 0, len(msgs))
	for _, m := range msgs {
		block, err := toolResultBlock(m)
		if err != nil {
			return anthropicMessage{}, err
		}

		blocks = append(blocks, block)
	}

	data, err := json.Marshal(blocks)
	if err != nil {
		return anthropicMessage{}, fmt.Errorf("aimodel: marshal tool result: %w", err)
	}

	return anthropicMessage{Role: "user", Content: data}, nil
}

func toAnthropicMessage(m ais.Message) (anthropicMessage, error) {
	am := anthropicMessage{
		Role: string(m.Role),
	}

	// Tool result messages become user messages with tool_result content blocks.
	if m.Role == ais.RoleTool {
		return toAnthropicToolResultMessage([]ais.Message{m})
	}

	cacheBreakpoint, err := messageCacheBreakpoint(&m)
	if err != nil {
		return anthropicMessage{}, err
	}

	// Assistant messages with thinking, tool calls, or both require content-block format.
	if m.Role == ais.RoleAssistant && (m.Thinking != "" || len(m.ToolCalls) > 0) {
		var blocks []anthropicContentBlock

		if m.Thinking != "" {
			blocks = append(blocks, anthropicContentBlock{
				Type:     "thinking",
				Thinking: m.Thinking,
			})
		}

		text := m.Content.Text()
		if text != "" {
			blocks = append(blocks, anthropicContentBlock{
				Type: "text",
				Text: text,
			})
		}

		for _, tc := range m.ToolCalls {
			blocks = append(blocks, anthropicContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			})
		}

		if cacheBreakpoint && len(blocks) > 0 {
			blocks[len(blocks)-1].CacheControl = ephemeralCache()
		}

		data, err := json.Marshal(blocks)
		if err != nil {
			return anthropicMessage{}, fmt.Errorf("aimodel: marshal assistant content: %w", err)
		}

		am.Content = data

		return am, nil
	}

	// Multimodal content with parts.
	if parts := m.Content.Parts(); len(parts) > 0 {
		var blocks []anthropicContentBlock

		for _, p := range parts {
			switch p.Type {
			case "text":
				blocks = append(blocks, anthropicContentBlock{
					Type: "text",
					Text: p.Text,
				})
			case "image_url":
				if p.ImageURL == nil {
					continue
				}

				block := anthropicContentBlock{Type: "image"}

				if mediaType, b64Data, ok := parseDataURI(p.ImageURL.URL); ok {
					block.Source = &anthropicContentSource{
						Type:      "base64",
						MediaType: mediaType,
						Data:      b64Data,
					}
				} else {
					block.Source = &anthropicContentSource{
						Type: "url",
						URL:  p.ImageURL.URL,
					}
				}

				blocks = append(blocks, block)
			}
		}

		if cacheBreakpoint && len(blocks) > 0 {
			blocks[len(blocks)-1].CacheControl = ephemeralCache()
		}

		data, err := json.Marshal(blocks)
		if err != nil {
			return anthropicMessage{}, fmt.Errorf("aimodel: marshal multimodal content: %w", err)
		}

		am.Content = data

		return am, nil
	}

	// Plain text message. Promote to block-array form when the caller
	// flagged CacheBreakpoint so we can attach cache_control.
	if cacheBreakpoint {
		block := anthropicContentBlock{
			Type:         "text",
			Text:         m.Content.Text(),
			CacheControl: ephemeralCache(),
		}
		data, err := json.Marshal([]anthropicContentBlock{block})
		if err != nil {
			return anthropicMessage{}, fmt.Errorf("aimodel: marshal cached message content: %w", err)
		}
		am.Content = data
		return am, nil
	}

	data, err := json.Marshal(m.Content.Text())
	if err != nil {
		return anthropicMessage{}, fmt.Errorf("aimodel: marshal message content: %w", err)
	}

	am.Content = data

	return am, nil
}

func convertToolChoice(tc any) *anthropicToolChoice {
	switch v := tc.(type) {
	case string:
		switch v {
		case "auto":
			return &anthropicToolChoice{Type: "auto"}
		case "required":
			return &anthropicToolChoice{Type: "any"}
		case "none":
			// Explicit "none" forbids any tool call; an omitted tool_choice
			// would instead let the model choose, so emit {type:"none"}.
			return &anthropicToolChoice{Type: "none"}
		}
	case map[string]any:
		if fn, ok := v["function"].(map[string]any); ok {
			if name, ok := fn["name"].(string); ok {
				return &anthropicToolChoice{Type: "tool", Name: name}
			}
		}
	}

	return nil
}

// fromAnthropicResponse converts an Anthropic API response to a ais.ChatResponse.
// Anthropic-only response information — unmodelled content blocks, structured
// stop details, the execution container, cache-write accounting — is written
// into this provider's extension namespaces instead of canonical fields.
func fromAnthropicResponse(ar *anthropicResponse) *ais.ChatResponse {
	msg := ais.Message{
		Role: ais.RoleAssistant,
	}

	var textParts []string

	var thinkingParts []string

	var extraBlocks []json.RawMessage

	for _, block := range ar.Content {
		switch block.Type {
		case "thinking":
			thinkingParts = append(thinkingParts, block.Thinking)
		case "text":
			textParts = append(textParts, block.Text)

			// A text block may carry citation annotations this wrapper does
			// not promote to the canonical layer. The text is still extracted
			// above; keep the whole original block so the annotations remain
			// reachable.
			if len(block.Citations) > 0 {
				extraBlocks = append(extraBlocks, block.raw)
			}
		case "tool_use":
			msg.ToolCalls = append(msg.ToolCalls, ais.ToolCall{
				Index: len(msg.ToolCalls),
				ID:    block.ID,
				Type:  "function",
				Function: ais.FunctionCall{
					Name:      block.Name,
					Arguments: string(block.Input),
				},
			})
		default:
			// Server-tool blocks (server_tool_use, web_search_tool_result,
			// code_execution_tool_result, …) and any block type added after
			// this wrapper was written. Preserve the original JSON instead of
			// dropping it silently.
			extraBlocks = append(extraBlocks, block.raw)
		}
	}

	if len(thinkingParts) > 0 {
		msg.Thinking = strings.Join(thinkingParts, "\n")
	}

	if len(textParts) > 0 {
		msg.Content = ais.NewTextContent(strings.Join(textParts, "\n"))
	}

	if len(extraBlocks) > 0 {
		msg.Extensions.Set(Name, &MessageExtension{ExtraBlocks: extraBlocks})
	}

	choice := ais.Choice{
		Index:        0,
		Message:      msg,
		FinishReason: mapAnthropicStopReason(ar.StopReason),
	}

	if ar.StopDetails != nil {
		choice.Extensions.Set(Name, &ChoiceExtension{StopDetails: ar.StopDetails})
	}

	cr := &ais.ChatResponse{
		ID:      ar.ID,
		Object:  "chat.completion",
		Model:   ar.Model,
		Choices: []ais.Choice{choice},
		Usage:   anthropicCanonicalUsage(&ar.Usage),
	}

	if ar.Container != nil {
		cr.Extensions.Set(Name, &ResponseExtension{Container: ar.Container})
	}

	return cr
}

// anthropicCanonicalUsage builds a canonical Usage from an Anthropic usage
// object, folding cached/created tokens into PromptTokens (as before). The
// cross-provider counts stay canonical; cache-write totals, the per-TTL
// breakdown, server-tool counts and the inference geography go into the
// UsageExtension namespace (attached only when any of them is present).
func anthropicCanonicalUsage(u *anthropicUsage) ais.Usage {
	cu := ais.Usage{
		PromptTokens:     u.totalInputTokens(),
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.totalInputTokens() + u.OutputTokens,
		CacheReadTokens:  u.CacheReadInputTokens,
		ServiceTier:      u.ServiceTier,
	}

	if u.OutputTokensDetails != nil {
		cu.ReasoningTokens = u.OutputTokensDetails.ThinkingTokens
	}

	ext := &UsageExtension{
		CacheWriteTokens: u.CacheCreationInputTokens,
		InferenceGeo:     u.InferenceGeo,
	}

	if u.CacheCreation != nil {
		ext.CacheWrite5mTokens = u.CacheCreation.Ephemeral5mInputTokens
		ext.CacheWrite1hTokens = u.CacheCreation.Ephemeral1hInputTokens
	}

	if u.ServerToolUse != nil {
		ext.ServerToolUse = &ServerToolUse{
			WebSearchRequests: u.ServerToolUse.WebSearchRequests,
			WebFetchRequests:  u.ServerToolUse.WebFetchRequests,
		}
	}

	if ext.CacheWriteTokens != 0 || ext.InferenceGeo != "" || ext.ServerToolUse != nil ||
		ext.CacheWrite5mTokens != 0 || ext.CacheWrite1hTokens != 0 {
		cu.Extensions.Set(Name, ext)
	}

	return cu
}

// mergeAnthropicUsage folds a later usage object (the terminal message_delta)
// into the baseline captured at message_start. Only fields the later object
// actually carries are applied — a terminal event that reports just
// output_tokens must not blank out the input, cache, geo, tier or server-tool
// information already established.
func mergeAnthropicUsage(base, next *anthropicUsage) {
	if next.InputTokens != 0 {
		base.InputTokens = next.InputTokens
	}

	if next.OutputTokens != 0 {
		base.OutputTokens = next.OutputTokens
	}

	if next.CacheCreationInputTokens != 0 {
		base.CacheCreationInputTokens = next.CacheCreationInputTokens
	}

	if next.CacheReadInputTokens != 0 {
		base.CacheReadInputTokens = next.CacheReadInputTokens
	}

	if next.CacheCreation != nil {
		base.CacheCreation = next.CacheCreation
	}

	if next.OutputTokensDetails != nil {
		base.OutputTokensDetails = next.OutputTokensDetails
	}

	if next.ServerToolUse != nil {
		base.ServerToolUse = next.ServerToolUse
	}

	if next.InferenceGeo != "" {
		base.InferenceGeo = next.InferenceGeo
	}

	if next.ServiceTier != "" {
		base.ServiceTier = next.ServiceTier
	}
}

// parseDataURI parses a data URI (e.g. "data:image/jpeg;base64,/9j...")
// and returns the media type and base64-encoded data.
func parseDataURI(uri string) (mediaType, data string, ok bool) {
	const prefix = "data:"

	if !strings.HasPrefix(uri, prefix) {
		return "", "", false
	}

	// Format: data:<mediaType>;base64,<data>
	rest := uri[len(prefix):]

	semicolon := strings.Index(rest, ";")
	if semicolon < 0 {
		return "", "", false
	}

	mediaType = rest[:semicolon]

	rest = rest[semicolon+1:]
	if !strings.HasPrefix(rest, "base64,") {
		return "", "", false
	}

	data = rest[len("base64,"):]

	return mediaType, data, true
}

func mapAnthropicStopReason(reason string) ais.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return ais.FinishReasonStop
	case "max_tokens":
		return ais.FinishReasonLength
	case "tool_use":
		return ais.FinishReasonToolCalls
	case "model_context_window_exceeded":
		return FinishReasonModelContextWindowExceeded
	case "refusal":
		return FinishReasonRefusal
	case "pause_turn":
		return FinishReasonPauseTurn
	default:
		return ais.FinishReason(reason)
	}
}
