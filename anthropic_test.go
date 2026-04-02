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
	if ar.System != nil {
		t.Errorf("system = %s, want nil", ar.System)
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

	var system string
	if err := json.Unmarshal(ar.System, &system); err != nil {
		t.Fatalf("unmarshal system: %v", err)
	}
	if system != "You are helpful.\nBe concise." {
		t.Errorf("system = %q", system)
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

func TestToAnthropicRequestImageDataURI(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{
				Role: RoleUser,
				Content: NewPartsContent(
					ContentPart{Type: "image_url", ImageURL: &ImageURL{
						URL: "data:image/jpeg;base64,/9j/4AAQ",
					}},
				),
			},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.Messages[0].Content, &blocks); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}

	if len(blocks) != 1 {
		t.Fatalf("blocks len = %d, want 1", len(blocks))
	}
	if blocks[0].Type != "image" {
		t.Errorf("type = %q, want image", blocks[0].Type)
	}
	if blocks[0].Source == nil {
		t.Fatal("source is nil")
	}
	if blocks[0].Source.Type != "base64" {
		t.Errorf("source.type = %q, want base64", blocks[0].Source.Type)
	}
	if blocks[0].Source.MediaType != "image/jpeg" {
		t.Errorf("source.media_type = %q, want image/jpeg", blocks[0].Source.MediaType)
	}
	if blocks[0].Source.Data != "/9j/4AAQ" {
		t.Errorf("source.data = %q, want /9j/4AAQ", blocks[0].Source.Data)
	}
}

func TestToAnthropicRequestImageURL(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{
				Role: RoleUser,
				Content: NewPartsContent(
					ContentPart{Type: "image_url", ImageURL: &ImageURL{
						URL: "https://example.com/photo.png",
					}},
				),
			},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.Messages[0].Content, &blocks); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}

	if len(blocks) != 1 {
		t.Fatalf("blocks len = %d, want 1", len(blocks))
	}
	if blocks[0].Source == nil {
		t.Fatal("source is nil")
	}
	if blocks[0].Source.Type != "url" {
		t.Errorf("source.type = %q, want url", blocks[0].Source.Type)
	}
	if blocks[0].Source.URL != "https://example.com/photo.png" {
		t.Errorf("source.url = %q", blocks[0].Source.URL)
	}
}

func TestToAnthropicRequestMixedContent(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{
				Role: RoleUser,
				Content: NewPartsContent(
					ContentPart{Type: "text", Text: "What is in this image?"},
					ContentPart{Type: "image_url", ImageURL: &ImageURL{
						URL: "data:image/png;base64,iVBOR",
					}},
				),
			},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.Messages[0].Content, &blocks); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}

	if len(blocks) != 2 {
		t.Fatalf("blocks len = %d, want 2", len(blocks))
	}
	if blocks[0].Type != "text" || blocks[0].Text != "What is in this image?" {
		t.Errorf("blocks[0] = %+v", blocks[0])
	}
	if blocks[1].Type != "image" {
		t.Errorf("blocks[1].type = %q, want image", blocks[1].Type)
	}
	if blocks[1].Source.Type != "base64" {
		t.Errorf("blocks[1].source.type = %q, want base64", blocks[1].Source.Type)
	}
	if blocks[1].Source.MediaType != "image/png" {
		t.Errorf("blocks[1].source.media_type = %q", blocks[1].Source.MediaType)
	}
}

func TestToAnthropicRequestImageNilURL(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{
				Role: RoleUser,
				Content: NewPartsContent(
					ContentPart{Type: "text", Text: "hello"},
					ContentPart{Type: "image_url"}, // nil ImageURL, should be skipped
				),
			},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.Messages[0].Content, &blocks); err != nil {
		t.Fatalf("unmarshal content: %v", err)
	}

	if len(blocks) != 1 {
		t.Fatalf("blocks len = %d, want 1 (nil image_url skipped)", len(blocks))
	}
	if blocks[0].Type != "text" {
		t.Errorf("blocks[0].type = %q, want text", blocks[0].Type)
	}
}

func TestParseDataURI(t *testing.T) {
	tests := []struct {
		name      string
		uri       string
		wantMedia string
		wantData  string
		wantOK    bool
	}{
		{
			name:      "jpeg",
			uri:       "data:image/jpeg;base64,/9j/4AAQ",
			wantMedia: "image/jpeg",
			wantData:  "/9j/4AAQ",
			wantOK:    true,
		},
		{
			name:      "png",
			uri:       "data:image/png;base64,iVBOR",
			wantMedia: "image/png",
			wantData:  "iVBOR",
			wantOK:    true,
		},
		{
			name:   "not data uri",
			uri:    "https://example.com/img.png",
			wantOK: false,
		},
		{
			name:   "no semicolon",
			uri:    "data:image/jpeg",
			wantOK: false,
		},
		{
			name:   "not base64",
			uri:    "data:image/jpeg;charset=utf-8,hello",
			wantOK: false,
		},
		{
			name:      "webp",
			uri:       "data:image/webp;base64,UklGR",
			wantMedia: "image/webp",
			wantData:  "UklGR",
			wantOK:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			media, data, ok := parseDataURI(tt.uri)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if media != tt.wantMedia {
				t.Errorf("mediaType = %q, want %q", media, tt.wantMedia)
			}
			if data != tt.wantData {
				t.Errorf("data = %q, want %q", data, tt.wantData)
			}
		})
	}
}

func TestToAnthropicRequestThinking(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Think step by step")},
		},
		Thinking: &Thinking{
			Type:         "enabled",
			BudgetTokens: 10000,
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if ar.Thinking == nil {
		t.Fatal("thinking is nil")
	}

	if ar.Thinking.Type != "enabled" {
		t.Errorf("thinking.type = %q, want enabled", ar.Thinking.Type)
	}

	if ar.Thinking.BudgetTokens != 10000 {
		t.Errorf("thinking.budget_tokens = %d, want 10000", ar.Thinking.BudgetTokens)
	}
}

func TestFromAnthropicResponseThinking(t *testing.T) {
	ar := &anthropicResponse{
		ID:    "msg_think",
		Model: ModelAnthropicClaude4Sonnet,
		Content: []anthropicContentBlock{
			{Type: "thinking", Thinking: "Let me analyze this step by step."},
			{Type: "thinking", Thinking: "The answer involves calculus."},
			{Type: "text", Text: "The answer is 42."},
		},
		StopReason: "end_turn",
		Usage:      anthropicUsage{InputTokens: 20, OutputTokens: 50},
	}

	cr := fromAnthropicResponse(ar)

	if len(cr.Choices) != 1 {
		t.Fatalf("choices len = %d", len(cr.Choices))
	}

	msg := cr.Choices[0].Message
	if msg.Thinking != "Let me analyze this step by step.\nThe answer involves calculus." {
		t.Errorf("thinking = %q", msg.Thinking)
	}

	if msg.Content.Text() != "The answer is 42." {
		t.Errorf("content = %q", msg.Content.Text())
	}
}

func TestToAnthropicMessageThinkingRoundTrip(t *testing.T) {
	msg := Message{
		Role:     RoleAssistant,
		Thinking: "Step 1: analyze.\nStep 2: conclude.",
		Content:  NewTextContent("The answer is 42."),
	}

	am, err := toAnthropicMessage(msg)
	if err != nil {
		t.Fatalf("toAnthropicMessage: %v", err)
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(am.Content, &blocks); err != nil {
		t.Fatalf("unmarshal blocks: %v", err)
	}

	if len(blocks) != 2 {
		t.Fatalf("got %d blocks, want 2", len(blocks))
	}

	if blocks[0].Type != "thinking" {
		t.Errorf("block[0].type = %q, want thinking", blocks[0].Type)
	}

	if blocks[0].Thinking != "Step 1: analyze.\nStep 2: conclude." {
		t.Errorf("block[0].thinking = %q", blocks[0].Thinking)
	}

	if blocks[1].Type != "text" || blocks[1].Text != "The answer is 42." {
		t.Errorf("block[1] = %+v", blocks[1])
	}
}

func TestToAnthropicMessageThinkingWithToolCalls(t *testing.T) {
	msg := Message{
		Role:     RoleAssistant,
		Thinking: "I should use the calculator.",
		Content:  NewTextContent("Let me calculate."),
		ToolCalls: []ToolCall{
			{
				Index: 0,
				ID:    "call_1",
				Type:  "function",
				Function: FunctionCall{
					Name:      "calculator",
					Arguments: `{"expr":"2+2"}`,
				},
			},
		},
	}

	am, err := toAnthropicMessage(msg)
	if err != nil {
		t.Fatalf("toAnthropicMessage: %v", err)
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(am.Content, &blocks); err != nil {
		t.Fatalf("unmarshal blocks: %v", err)
	}

	if len(blocks) != 3 {
		t.Fatalf("got %d blocks, want 3", len(blocks))
	}

	if blocks[0].Type != "thinking" {
		t.Errorf("block[0].type = %q, want thinking", blocks[0].Type)
	}

	if blocks[1].Type != "text" {
		t.Errorf("block[1].type = %q, want text", blocks[1].Type)
	}

	if blocks[2].Type != "tool_use" {
		t.Errorf("block[2].type = %q, want tool_use", blocks[2].Type)
	}
}

func TestToAnthropicRequestToolResultMissingID(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("weather?")},
			{
				Role:    RoleTool,
				Content: NewTextContent(`{"temp": 72}`),
				// ToolCallID intentionally omitted.
			},
		},
	}

	_, err := toAnthropicRequest(req)
	if err == nil {
		t.Fatal("expected error for missing tool_call_id")
	}

	if !strings.Contains(err.Error(), "tool_call_id") {
		t.Errorf("error = %q, want mention of tool_call_id", err.Error())
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

func TestToAnthropicRequestSystemMultimodal(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{
				Role: RoleSystem,
				Content: NewPartsContent(
					ContentPart{Type: "text", Text: "You are helpful."},
				),
			},
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	// Multimodal system messages should produce a content block array.
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.System, &blocks); err != nil {
		t.Fatalf("unmarshal system blocks: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks len = %d, want 1", len(blocks))
	}
	if blocks[0].Type != "text" || blocks[0].Text != "You are helpful." {
		t.Errorf("block = %+v", blocks[0])
	}
}

func TestAnthropicUsageTotalInputTokens(t *testing.T) {
	tests := []struct {
		name  string
		usage anthropicUsage
		want  int
	}{
		{
			name:  "all zeros",
			usage: anthropicUsage{},
			want:  0,
		},
		{
			name:  "only InputTokens",
			usage: anthropicUsage{InputTokens: 15},
			want:  15,
		},
		{
			name: "all three fields set",
			usage: anthropicUsage{
				InputTokens:              10,
				CacheCreationInputTokens: 5,
				CacheReadInputTokens:     3,
			},
			want: 18,
		},
		{
			name: "only cache fields set",
			usage: anthropicUsage{
				CacheCreationInputTokens: 7,
				CacheReadInputTokens:     4,
			},
			want: 11,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.usage.totalInputTokens(); got != tt.want {
				t.Errorf("totalInputTokens() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestFromAnthropicResponseCacheTokens(t *testing.T) {
	ar := &anthropicResponse{
		ID:    "msg_cache",
		Model: ModelAnthropicClaude4Sonnet,
		Content: []anthropicContentBlock{
			{Type: "text", Text: "Cached response."},
		},
		StopReason: "end_turn",
		Usage: anthropicUsage{
			InputTokens:              10,
			CacheCreationInputTokens: 5,
			CacheReadInputTokens:     3,
			OutputTokens:             20,
		},
	}

	cr := fromAnthropicResponse(ar)

	if cr.Usage.PromptTokens != 18 {
		t.Errorf("prompt_tokens = %d, want 18 (10+5+3)", cr.Usage.PromptTokens)
	}
	if cr.Usage.CompletionTokens != 20 {
		t.Errorf("completion_tokens = %d, want 20", cr.Usage.CompletionTokens)
	}
	if cr.Usage.TotalTokens != 38 {
		t.Errorf("total_tokens = %d, want 38 (18+20)", cr.Usage.TotalTokens)
	}
}

func TestToAnthropicRequestSystemMixed(t *testing.T) {
	req := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleSystem, Content: NewTextContent("Be helpful.")},
			{
				Role: RoleSystem,
				Content: NewPartsContent(
					ContentPart{Type: "text", Text: "Be concise."},
				),
			},
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	// When any system message has parts, all should be content blocks.
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.System, &blocks); err != nil {
		t.Fatalf("unmarshal system blocks: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks len = %d, want 2", len(blocks))
	}
	if blocks[0].Text != "Be helpful." {
		t.Errorf("blocks[0].text = %q", blocks[0].Text)
	}
	if blocks[1].Text != "Be concise." {
		t.Errorf("blocks[1].text = %q", blocks[1].Text)
	}
}
