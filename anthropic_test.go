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
	"testing"
)

func TestToAnthropicRequestBasic(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hello")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if ar.Model != ModelAnthropicClaude4Sonnet {
		t.Errorf("model = %q", ar.Model)
	}
	if ar.MaxTokens != anthropicDefaultMaxTokens {
		t.Errorf("max_tokens = %d, want %d", ar.MaxTokens, anthropicDefaultMaxTokens)
	}
	if ar.System != "" {
		t.Errorf("system = %q, want empty", ar.System)
	}
	if len(ar.Messages) != 1 {
		t.Fatalf("messages len = %d", len(ar.Messages))
	}
	if ar.Messages[0].Role != "user" {
		t.Errorf("role = %q", ar.Messages[0].Role)
	}
}

func TestToAnthropicRequestSystemExtraction(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleSystem, Content: NewTextContent("You are helpful.")},
			{Role: RoleSystem, Content: NewTextContent("Be concise.")},
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if ar.System != "You are helpful.\nBe concise." {
		t.Errorf("system = %q", ar.System)
	}
	if len(ar.Messages) != 1 {
		t.Errorf("messages len = %d, want 1 (system excluded)", len(ar.Messages))
	}
}

func TestToAnthropicRequestMaxTokens(t *testing.T) {
	maxTokens := 1024
	req := &ChatRequest{
		Model:     ModelAnthropicClaude4Sonnet,
		MaxTokens: &maxTokens,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if ar.MaxTokens != 1024 {
		t.Errorf("max_tokens = %d, want 1024", ar.MaxTokens)
	}
}

func TestToAnthropicRequestStopSequences(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Stop:  []string{"END", "STOP"},
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if len(ar.StopSequences) != 2 {
		t.Fatalf("stop_sequences len = %d", len(ar.StopSequences))
	}
	if ar.StopSequences[0] != "END" || ar.StopSequences[1] != "STOP" {
		t.Errorf("stop_sequences = %v", ar.StopSequences)
	}
}

func TestToAnthropicRequestTools(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("weather?")},
		},
		Tools: []Tool{
			{
				Type: "function",
				Function: FunctionDefinition{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city": map[string]any{"type": "string"},
						},
					},
				},
			},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if len(ar.Tools) != 1 {
		t.Fatalf("tools len = %d", len(ar.Tools))
	}
	if ar.Tools[0].Name != "get_weather" {
		t.Errorf("tool name = %q", ar.Tools[0].Name)
	}
	if ar.Tools[0].Description != "Get weather" {
		t.Errorf("tool description = %q", ar.Tools[0].Description)
	}
}

func TestToAnthropicRequestToolChoice(t *testing.T) {
	tests := []struct {
		name       string
		toolChoice any
		wantType   string
		wantName   string
		wantNil    bool
	}{
		{"auto", "auto", "auto", "", false},
		{"required", "required", "any", "", false},
		{"none", "none", "", "", true},
		{
			"specific",
			map[string]any{"type": "function", "function": map[string]any{"name": "get_weather"}},
			"tool",
			"get_weather",
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &ChatRequest{
				Model: ModelAnthropicClaude4Sonnet,
				Messages: []Message{
					{Role: RoleUser, Content: NewTextContent("Hi")},
				},
				ToolChoice: tt.toolChoice,
			}

			ar, err := toAnthropicRequest(req)
			if err != nil {
				t.Fatalf("toAnthropicRequest: %v", err)
			}

			if tt.wantNil {
				if ar.ToolChoice != nil {
					t.Errorf("tool_choice should be nil")
				}

				return
			}

			if ar.ToolChoice == nil {
				t.Fatal("tool_choice is nil")
			}
			if ar.ToolChoice.Type != tt.wantType {
				t.Errorf("type = %q, want %q", ar.ToolChoice.Type, tt.wantType)
			}
			if ar.ToolChoice.Name != tt.wantName {
				t.Errorf("name = %q, want %q", ar.ToolChoice.Name, tt.wantName)
			}
		})
	}
}

func TestToAnthropicRequestToolResult(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("weather?")},
			{
				Role:       RoleTool,
				Content:    NewTextContent(`{"temp": 72}`),
				ToolCallID: "call_1",
			},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if len(ar.Messages) != 2 {
		t.Fatalf("messages len = %d", len(ar.Messages))
	}

	// Tool result should be converted to user role.
	if ar.Messages[1].Role != "user" {
		t.Errorf("role = %q, want user", ar.Messages[1].Role)
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.Messages[1].Content, &blocks); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks len = %d", len(blocks))
	}
	if blocks[0].Type != "tool_result" {
		t.Errorf("type = %q", blocks[0].Type)
	}
	if blocks[0].ToolUseID != "call_1" {
		t.Errorf("tool_use_id = %q", blocks[0].ToolUseID)
	}
}

func TestFromAnthropicResponseText(t *testing.T) {
	ar := &anthropicResponse{
		ID:    "msg_123",
		Model: ModelAnthropicClaude4Sonnet,
		Content: []anthropicContentBlock{
			{Type: "text", Text: "Hello!"},
		},
		StopReason: "end_turn",
		Usage: anthropicUsage{
			InputTokens:  10,
			OutputTokens: 5,
		},
	}

	cr := fromAnthropicResponse(ar)

	if cr.ID != "msg_123" {
		t.Errorf("id = %q", cr.ID)
	}
	if len(cr.Choices) != 1 {
		t.Fatalf("choices len = %d", len(cr.Choices))
	}
	if cr.Choices[0].Message.Content.Text() != "Hello!" {
		t.Errorf("content = %q", cr.Choices[0].Message.Content.Text())
	}
	if cr.Choices[0].FinishReason != FinishReasonStop {
		t.Errorf("finish_reason = %q", cr.Choices[0].FinishReason)
	}
	if cr.Usage.PromptTokens != 10 {
		t.Errorf("prompt_tokens = %d", cr.Usage.PromptTokens)
	}
	if cr.Usage.CompletionTokens != 5 {
		t.Errorf("completion_tokens = %d", cr.Usage.CompletionTokens)
	}
	if cr.Usage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d", cr.Usage.TotalTokens)
	}
}

func TestFromAnthropicResponseToolUse(t *testing.T) {
	ar := &anthropicResponse{
		ID:    "msg_456",
		Model: ModelAnthropicClaude4Sonnet,
		Content: []anthropicContentBlock{
			{Type: "text", Text: "Let me check the weather."},
			{
				Type:  "tool_use",
				ID:    "toolu_1",
				Name:  "get_weather",
				Input: json.RawMessage(`{"city":"NYC"}`),
			},
		},
		StopReason: "tool_use",
		Usage:      anthropicUsage{InputTokens: 20, OutputTokens: 30},
	}

	cr := fromAnthropicResponse(ar)

	if cr.Choices[0].Message.Content.Text() != "Let me check the weather." {
		t.Errorf("content = %q", cr.Choices[0].Message.Content.Text())
	}
	if len(cr.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d", len(cr.Choices[0].Message.ToolCalls))
	}
	tc := cr.Choices[0].Message.ToolCalls[0]
	if tc.ID != "toolu_1" {
		t.Errorf("id = %q", tc.ID)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("name = %q", tc.Function.Name)
	}
	if cr.Choices[0].FinishReason != FinishReasonToolCalls {
		t.Errorf("finish_reason = %q", cr.Choices[0].FinishReason)
	}
}

func TestMapAnthropicStopReason(t *testing.T) {
	tests := []struct {
		reason string
		want   FinishReason
	}{
		{"end_turn", FinishReasonStop},
		{"stop_sequence", FinishReasonStop},
		{"max_tokens", FinishReasonLength},
		{"tool_use", FinishReasonToolCalls},
		{"unknown", FinishReason("unknown")},
	}

	for _, tt := range tests {
		t.Run(tt.reason, func(t *testing.T) {
			if got := mapAnthropicStopReason(tt.reason); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}
