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
	FinishReasonStop      FinishReason = "stop"
	FinishReasonLength    FinishReason = "length"
	FinishReasonToolCalls FinishReason = "tool_calls"
)

// ChatRequest represents a request to the chat completions API.
type ChatRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	Temperature      *float64  `json:"temperature,omitempty"`
	MaxTokens        *int      `json:"max_tokens,omitempty"`
	TopP             *float64  `json:"top_p,omitempty"`
	N                *int      `json:"n,omitempty"`
	Stop             []string  `json:"stop,omitempty"`
	FrequencyPenalty *float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64  `json:"presence_penalty,omitempty"`
	Seed             *int      `json:"seed,omitempty"`
	User             string    `json:"user,omitempty"`
	ResponseFormat   any       `json:"response_format,omitempty"`
	Stream           bool      `json:"stream,omitempty"`
	Tools            []Tool    `json:"tools,omitempty"`
	ToolChoice       any       `json:"tool_choice,omitempty"`
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
}

// Content represents chat message content that can be either a plain string
// or an array of content parts (text, image_url, etc.) for multimodal input.
type Content struct {
	text  string
	parts []ContentPart
}

// ContentPart represents a single part in a multimodal content array.
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
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// AppendDelta merges a streaming delta message into this message.
func (m *Message) AppendDelta(delta *Message) {
	m.Content.text += delta.Content.text

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
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition describes a function available to the model.
type FunctionDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// StreamChunk represents a single chunk in a streaming response.
type StreamChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []StreamChunkChoice `json:"choices"`
}

// StreamChunkChoice represents a choice within a stream chunk.
type StreamChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Usage tracks token usage for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Error represents an error in the API response body.
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
	Type    string `json:"type"`
}
