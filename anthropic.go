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

const (
	anthropicDefaultBaseURL   = "https://api.anthropic.com"
	anthropicAPIVersion       = "2023-06-01"
	anthropicDefaultMaxTokens = 4096
)

// --- Anthropic request types ---

type anthropicRequest struct {
	Model         string               `json:"model"`
	Messages      []anthropicMessage   `json:"messages"`
	System        string               `json:"system,omitempty"`
	MaxTokens     int                  `json:"max_tokens"`
	Temperature   *float64             `json:"temperature,omitempty"`
	TopP          *float64             `json:"top_p,omitempty"`
	StopSequences []string             `json:"stop_sequences,omitempty"`
	Stream        bool                 `json:"stream,omitempty"`
	Tools         []anthropicTool      `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice `json:"tool_choice,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	// ResultContent holds the content for tool_result blocks.
	ResultContent string `json:"content,omitempty"`
}

type anthropicTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema"`
}

type anthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
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
	Usage        anthropicUsage          `json:"usage"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
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
	PartialJSON string `json:"partial_json,omitempty"`
}

type anthropicMessageDelta struct {
	Type  string                    `json:"type"`
	Delta anthropicMessageDeltaData `json:"delta"`
	Usage *anthropicUsage           `json:"usage,omitempty"`
}

type anthropicMessageDeltaData struct {
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence,omitempty"`
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

	// MaxTokens: use provided value or default.
	if req.MaxTokens != nil {
		ar.MaxTokens = *req.MaxTokens
	} else {
		ar.MaxTokens = anthropicDefaultMaxTokens
	}

	// Stop -> StopSequences.
	if len(req.Stop) > 0 {
		ar.StopSequences = req.Stop
	}

	// Extract system messages and convert the rest.
	for _, m := range req.Messages {
		if m.Role == RoleSystem {
			if ar.System != "" {
				ar.System += "\n"
			}

			ar.System += m.Content.Text()

			continue
		}

		am, err := toAnthropicMessage(m)
		if err != nil {
			return nil, err
		}

		ar.Messages = append(ar.Messages, am)
	}

	// Convert tools.
	for _, t := range req.Tools {
		ar.Tools = append(ar.Tools, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}

	// Convert tool choice.
	if req.ToolChoice != nil {
		ar.ToolChoice = convertToolChoice(req.ToolChoice)
	}

	return ar, nil
}

func toAnthropicMessage(m Message) (anthropicMessage, error) {
	am := anthropicMessage{
		Role: string(m.Role),
	}

	// Tool result messages become user messages with tool_result content blocks.
	if m.Role == RoleTool {
		am.Role = "user"

		block := anthropicContentBlock{
			Type:          "tool_result",
			ToolUseID:     m.ToolCallID,
			ResultContent: m.Content.Text(),
		}

		data, err := json.Marshal([]anthropicContentBlock{block})
		if err != nil {
			return anthropicMessage{}, fmt.Errorf("aimodel: marshal tool result: %w", err)
		}

		am.Content = data

		return am, nil
	}

	// Assistant messages with tool calls.
	if m.Role == RoleAssistant && len(m.ToolCalls) > 0 {
		var blocks []anthropicContentBlock

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

		data, err := json.Marshal(blocks)
		if err != nil {
			return anthropicMessage{}, fmt.Errorf("aimodel: marshal tool use: %w", err)
		}

		am.Content = data

		return am, nil
	}

	// Plain text message.
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
			return nil
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

	for _, block := range ar.Content {
		switch block.Type {
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
			},
		},
		Usage: Usage{
			PromptTokens:     ar.Usage.InputTokens,
			CompletionTokens: ar.Usage.OutputTokens,
			TotalTokens:      ar.Usage.InputTokens + ar.Usage.OutputTokens,
		},
	}
}

func mapAnthropicStopReason(reason string) FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return FinishReasonStop
	case "max_tokens":
		return FinishReasonLength
	case "tool_use":
		return FinishReasonToolCalls
	default:
		return FinishReason(reason)
	}
}
