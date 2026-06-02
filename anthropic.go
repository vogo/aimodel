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
	"fmt"
	"strings"
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
	StopSequences []string             `json:"stop_sequences,omitempty"`
	Stream        bool                 `json:"stream,omitempty"`
	Tools         []anthropicTool      `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice `json:"tool_choice,omitempty"`
	Thinking      *Thinking            `json:"thinking,omitempty"`
	// Effort is Anthropic's top-level reasoning-depth control (GA 2026-02-05),
	// mapped from the canonical ChatRequest.ReasoningEffort. It supersedes
	// thinking.budget_tokens for new models.
	Effort string `json:"effort,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
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
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	InputSchema  any                    `json:"input_schema"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
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
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Model        string                  `json:"model"`
	Content      []anthropicContentBlock `json:"content"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	// StopDetails carries the structured stop classification (e.g. the refusal
	// category) returned alongside stop_reason "refusal". Its fields match the
	// canonical StopDetails exactly, so it deserializes straight into it.
	StopDetails *StopDetails   `json:"stop_details"`
	Usage       anthropicUsage `json:"usage"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
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
	Type         string                `json:"type"`
	Index        int                   `json:"index"`
	ContentBlock anthropicContentBlock `json:"content_block"`
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

// toAnthropicRequest converts a ChatRequest to an Anthropic API request.
func toAnthropicRequest(req *ChatRequest) (*anthropicRequest, error) {
	ar := &anthropicRequest{
		Model:       req.Model,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}

	// MaxTokens: Anthropic always uses max_tokens. Prefer the newer
	// MaxCompletionTokens (OpenAI's reasoning-model field) over the deprecated
	// MaxTokens, falling back to the default when neither is set.
	switch {
	case req.MaxCompletionTokens != nil:
		ar.MaxTokens = *req.MaxCompletionTokens
	case req.MaxTokens != nil:
		ar.MaxTokens = *req.MaxTokens
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

	for _, m := range req.Messages {
		if m.Role == RoleSystem && !seenNonSystem {
			if m.CacheBreakpoint {
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
		if m.Role != RoleSystem {
			seenNonSystem = true
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

	// Convert tools. Tools flagged with CacheBreakpoint carry a
	// cache_control marker; Anthropic caches every tool up to and
	// including the flagged one.
	for _, t := range req.Tools {
		at := anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		}
		if t.CacheBreakpoint {
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

	// Map the canonical reasoning effort to Anthropic's top-level effort
	// field (supersedes thinking.budget_tokens). Empty stays omitted.
	ar.Effort = req.ReasoningEffort

	return ar, nil
}

func toAnthropicMessage(m Message) (anthropicMessage, error) {
	am := anthropicMessage{
		Role: string(m.Role),
	}

	// Tool result messages become user messages with tool_result content blocks.
	if m.Role == RoleTool {
		if m.ToolCallID == "" {
			return anthropicMessage{}, fmt.Errorf("aimodel: tool result message missing tool_call_id")
		}

		am.Role = "user"

		block := anthropicContentBlock{
			Type:          "tool_result",
			ToolUseID:     m.ToolCallID,
			ResultContent: m.Content.Text(),
		}
		if m.CacheBreakpoint {
			block.CacheControl = ephemeralCache()
		}

		data, err := json.Marshal([]anthropicContentBlock{block})
		if err != nil {
			return anthropicMessage{}, fmt.Errorf("aimodel: marshal tool result: %w", err)
		}

		am.Content = data

		return am, nil
	}

	// Assistant messages with thinking, tool calls, or both require content-block format.
	if m.Role == RoleAssistant && (m.Thinking != "" || len(m.ToolCalls) > 0) {
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

		if m.CacheBreakpoint && len(blocks) > 0 {
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

		if m.CacheBreakpoint && len(blocks) > 0 {
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
	if m.CacheBreakpoint {
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

// fromAnthropicResponse converts an Anthropic API response to a ChatResponse.
func fromAnthropicResponse(ar *anthropicResponse) *ChatResponse {
	msg := Message{
		Role: RoleAssistant,
	}

	var textParts []string

	var thinkingParts []string

	for _, block := range ar.Content {
		switch block.Type {
		case "thinking":
			thinkingParts = append(thinkingParts, block.Thinking)
		case "text":
			textParts = append(textParts, block.Text)
		case "tool_use":
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				Index: len(msg.ToolCalls),
				ID:    block.ID,
				Type:  "function",
				Function: FunctionCall{
					Name:      block.Name,
					Arguments: string(block.Input),
				},
			})
		}
	}

	if len(thinkingParts) > 0 {
		msg.Thinking = strings.Join(thinkingParts, "\n")
	}

	if len(textParts) > 0 {
		msg.Content = NewTextContent(strings.Join(textParts, "\n"))
	}

	return &ChatResponse{
		ID:     ar.ID,
		Object: "chat.completion",
		Model:  ar.Model,
		Choices: []Choice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: mapAnthropicStopReason(ar.StopReason),
				StopDetails:  ar.StopDetails,
			},
		},
		Usage: Usage{
			PromptTokens:     ar.Usage.totalInputTokens(),
			CompletionTokens: ar.Usage.OutputTokens,
			TotalTokens:      ar.Usage.totalInputTokens() + ar.Usage.OutputTokens,
			CacheReadTokens:  ar.Usage.CacheReadInputTokens,
		},
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

func mapAnthropicStopReason(reason string) FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return FinishReasonStop
	case "max_tokens":
		return FinishReasonLength
	case "tool_use":
		return FinishReasonToolCalls
	case "model_context_window_exceeded":
		return FinishReasonModelContextWindowExceeded
	case "refusal":
		return FinishReasonRefusal
	case "pause_turn":
		return FinishReasonPauseTurn
	default:
		return FinishReason(reason)
	}
}
