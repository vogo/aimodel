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

func TestChatRequestJSON(t *testing.T) {
	req := &ChatRequest{
		Model: ModelOpenaiGPT4o,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hello")},
		},
		Temperature: new(0.7),
		Stream:      true,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded ChatRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.Model != ModelOpenaiGPT4o {
		t.Errorf("model = %q", decoded.Model)
	}
	if len(decoded.Messages) != 1 {
		t.Fatalf("messages len = %d", len(decoded.Messages))
	}
	if decoded.Messages[0].Content.Text() != "Hello" {
		t.Errorf("content = %q", decoded.Messages[0].Content.Text())
	}
	if !decoded.Stream {
		t.Error("stream should be true")
	}
}

func TestChatRequestOmitsEmptyFields(t *testing.T) {
	req := &ChatRequest{
		Model: ModelOpenaiGPT4o,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	if _, ok := raw["temperature"]; ok {
		t.Error("temperature should be omitted when nil")
	}
	if _, ok := raw["tools"]; ok {
		t.Error("tools should be omitted when empty")
	}
	if _, ok := raw["stream"]; ok {
		t.Error("stream should be omitted when false")
	}
}

func TestChatResponseJSON(t *testing.T) {
	raw := `{
		"id": "chatcmpl-123",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": "Hello!"},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
	}`

	var resp ChatResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.ID != "chatcmpl-123" {
		t.Errorf("id = %q", resp.ID)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices len = %d", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content.Text() != "Hello!" {
		t.Errorf("content = %q", resp.Choices[0].Message.Content.Text())
	}
	if resp.Choices[0].FinishReason != FinishReasonStop {
		t.Errorf("finish_reason = %q", resp.Choices[0].FinishReason)
	}
	if resp.Usage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d", resp.Usage.TotalTokens)
	}
}

func TestMessageWithToolCalls(t *testing.T) {
	raw := `{
		"role": "assistant",
		"content": null,
		"tool_calls": [{
			"id": "call_abc",
			"type": "function",
			"function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
		}]
	}`

	var msg Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].ID != "call_abc" {
		t.Errorf("tool call id = %q", msg.ToolCalls[0].ID)
	}
	if msg.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("function name = %q", msg.ToolCalls[0].Function.Name)
	}
}

func TestToolCallMerge(t *testing.T) {
	tc := &ToolCall{
		ID:   "call_1",
		Type: "function",
		Function: FunctionCall{
			Name:      "search",
			Arguments: `{"q":`,
		},
	}

	delta := &ToolCall{
		Function: FunctionCall{
			Arguments: `"hello"}`,
		},
	}

	tc.Merge(delta)

	want := `{"q":"hello"}`
	if tc.Function.Arguments != want {
		t.Errorf("merged arguments = %q, want %q", tc.Function.Arguments, want)
	}
}

func TestMessageAppendDelta(t *testing.T) {
	msg := &Message{Role: RoleAssistant, Content: NewTextContent("Hello")}
	delta := &Message{Content: NewTextContent(" world")}
	msg.AppendDelta(delta)

	if msg.Content.Text() != "Hello world" {
		t.Errorf("content = %q", msg.Content.Text())
	}
}

func TestMessageAppendDeltaToolCalls(t *testing.T) {
	msg := &Message{Role: RoleAssistant}

	// First delta: new tool call
	delta1 := &Message{
		ToolCalls: []ToolCall{
			{
				Index: 0,
				ID:    "call_1",
				Type:  "function",
				Function: FunctionCall{
					Name:      "search",
					Arguments: `{"q":`,
				},
			},
		},
	}
	msg.AppendDelta(delta1)

	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d after first delta", len(msg.ToolCalls))
	}

	// Second delta: continue arguments
	delta2 := &Message{
		ToolCalls: []ToolCall{
			{
				Index: 0,
				Function: FunctionCall{
					Arguments: `"test"}`,
				},
			},
		},
	}
	msg.AppendDelta(delta2)

	want := `{"q":"test"}`
	if msg.ToolCalls[0].Function.Arguments != want {
		t.Errorf("arguments = %q, want %q", msg.ToolCalls[0].Function.Arguments, want)
	}
}

func TestMessageAppendDeltaNonContiguousIndex(t *testing.T) {
	msg := &Message{Role: RoleAssistant}

	// Delta with index 0
	delta1 := &Message{
		ToolCalls: []ToolCall{
			{Index: 0, ID: "call_1", Type: "function", Function: FunctionCall{Name: "fn1", Arguments: `{"a":1}`}},
		},
	}
	msg.AppendDelta(delta1)

	// Delta with index 2 (skipping 1)
	delta2 := &Message{
		ToolCalls: []ToolCall{
			{Index: 2, ID: "call_3", Type: "function", Function: FunctionCall{Name: "fn3", Arguments: `{"c":3}`}},
		},
	}
	msg.AppendDelta(delta2)

	if len(msg.ToolCalls) != 3 {
		t.Fatalf("tool_calls len = %d, want 3", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].Function.Name != "fn1" {
		t.Errorf("tool_calls[0].name = %q", msg.ToolCalls[0].Function.Name)
	}
	if msg.ToolCalls[2].Function.Name != "fn3" {
		t.Errorf("tool_calls[2].name = %q", msg.ToolCalls[2].Function.Name)
	}
}

func TestStreamChunkJSON(t *testing.T) {
	raw := `{
		"id": "chatcmpl-456",
		"object": "chat.completion.chunk",
		"created": 1700000001,
		"model": "gpt-4o",
		"choices": [{
			"index": 0,
			"delta": {"content": "Hi"},
			"finish_reason": null
		}]
	}`

	var chunk StreamChunk
	if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if chunk.ID != "chatcmpl-456" {
		t.Errorf("id = %q", chunk.ID)
	}
	if len(chunk.Choices) != 1 {
		t.Fatalf("choices len = %d", len(chunk.Choices))
	}
	if chunk.Choices[0].Delta.Content.Text() != "Hi" {
		t.Errorf("delta content = %q", chunk.Choices[0].Delta.Content.Text())
	}
}

func TestUsageJSON(t *testing.T) {
	raw := `{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}`

	var u Usage
	if err := json.Unmarshal([]byte(raw), &u); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if u.PromptTokens != 100 || u.CompletionTokens != 50 || u.TotalTokens != 150 {
		t.Errorf("usage = %+v", u)
	}
}

func TestUsageAdd(t *testing.T) {
	tests := []struct {
		name   string
		base   Usage
		other  Usage
		expect Usage
	}{
		{
			name:   "add non-zero to zero",
			base:   Usage{},
			other:  Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
			expect: Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
		},
		{
			name:   "add non-zero to non-zero",
			base:   Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
			other:  Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
			expect: Usage{PromptTokens: 15, CompletionTokens: 30, TotalTokens: 45},
		},
		{
			name:   "add zero to non-zero",
			base:   Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
			other:  Usage{},
			expect: Usage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.base.Add(&tt.other)
			if tt.base != tt.expect {
				t.Errorf("got %+v, want %+v", tt.base, tt.expect)
			}
		})
	}
}

func TestContentMarshalString(t *testing.T) {
	c := NewTextContent("hello")
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	if string(data) != `"hello"` {
		t.Errorf("got %s, want %q", data, "hello")
	}
}

func TestContentMarshalParts(t *testing.T) {
	c := NewPartsContent(
		ContentPart{Type: "text", Text: "describe this"},
		ContentPart{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/img.png"}},
	)

	data, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	// Should be an array
	if data[0] != '[' {
		t.Errorf("expected array, got %c", data[0])
	}
}

func TestContentUnmarshalString(t *testing.T) {
	var c Content
	if err := json.Unmarshal([]byte(`"hello"`), &c); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if c.Text() != "hello" {
		t.Errorf("text = %q", c.Text())
	}
}

func TestContentUnmarshalParts(t *testing.T) {
	raw := `[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"https://example.com/img.png"}}]`

	var c Content
	if err := json.Unmarshal([]byte(raw), &c); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if c.Text() != "hello" {
		t.Errorf("text = %q", c.Text())
	}
}

func TestContentUnmarshalNull(t *testing.T) {
	var c Content
	if err := json.Unmarshal([]byte("null"), &c); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if c.Text() != "" {
		t.Errorf("text = %q, want empty", c.Text())
	}
}

func TestContentParts(t *testing.T) {
	parts := []ContentPart{
		{Type: "text", Text: "describe this"},
		{Type: "image_url", ImageURL: &ImageURL{URL: "https://example.com/img.png"}},
	}
	c := NewPartsContent(parts...)

	got := c.Parts()
	if len(got) != 2 {
		t.Fatalf("Parts() len = %d, want 2", len(got))
	}
	if got[0].Type != "text" || got[0].Text != "describe this" {
		t.Errorf("Parts()[0] = %+v", got[0])
	}
	if got[1].Type != "image_url" || got[1].ImageURL.URL != "https://example.com/img.png" {
		t.Errorf("Parts()[1] = %+v", got[1])
	}

	// Plain text content should return nil parts.
	tc := NewTextContent("hello")
	if tc.Parts() != nil {
		t.Errorf("text content Parts() = %v, want nil", tc.Parts())
	}
}

func TestMessageAppendDeltaThinking(t *testing.T) {
	msg := &Message{Role: RoleAssistant}
	msg.AppendDelta(&Message{Thinking: "Let me think"})
	msg.AppendDelta(&Message{Thinking: " about this"})

	if msg.Thinking != "Let me think about this" {
		t.Errorf("thinking = %q, want %q", msg.Thinking, "Let me think about this")
	}
}

func TestChatRequestReasoningEffort(t *testing.T) {
	req := &ChatRequest{
		Model: ModelOpenaiGPT4o,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
		ReasoningEffort: "high",
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	val, ok := raw["reasoning_effort"]
	if !ok {
		t.Fatal("reasoning_effort not present in JSON")
	}

	if string(val) != `"high"` {
		t.Errorf("reasoning_effort = %s, want %q", val, "high")
	}
}

func TestOpenAIReasoningContentUnmarshal(t *testing.T) {
	raw := `{
		"role": "assistant",
		"content": "The answer is 42.",
		"reasoning_content": "I need to compute the answer to life."
	}`

	var msg Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if msg.Thinking != "I need to compute the answer to life." {
		t.Errorf("thinking = %q", msg.Thinking)
	}

	if msg.Content.Text() != "The answer is 42." {
		t.Errorf("content = %q", msg.Content.Text())
	}
}

func TestStreamChunkUsageJSON(t *testing.T) {
	raw := `{
		"id": "chatcmpl-789",
		"object": "chat.completion.chunk",
		"created": 1700000002,
		"model": "gpt-4o",
		"choices": [],
		"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
	}`

	var chunk StreamChunk
	if err := json.Unmarshal([]byte(raw), &chunk); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if chunk.Usage == nil {
		t.Fatal("usage is nil")
	}
	if chunk.Usage.PromptTokens != 10 {
		t.Errorf("prompt_tokens = %d", chunk.Usage.PromptTokens)
	}
	if chunk.Usage.CompletionTokens != 20 {
		t.Errorf("completion_tokens = %d", chunk.Usage.CompletionTokens)
	}
	if chunk.Usage.TotalTokens != 30 {
		t.Errorf("total_tokens = %d", chunk.Usage.TotalTokens)
	}
}

func TestStreamChunkUsageOmittedWhenNil(t *testing.T) {
	chunk := StreamChunk{ID: "test", Choices: []StreamChunkChoice{}}
	data, err := json.Marshal(chunk)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}
	if _, ok := raw["usage"]; ok {
		t.Error("usage should be omitted when nil")
	}
}

func TestUsageAddWithCacheReadTokens(t *testing.T) {
	base := Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, CacheReadTokens: 20}
	other := Usage{PromptTokens: 200, CompletionTokens: 100, TotalTokens: 300, CacheReadTokens: 30}
	base.Add(&other)

	if base.CacheReadTokens != 50 {
		t.Errorf("CacheReadTokens = %d, want 50", base.CacheReadTokens)
	}
	if base.PromptTokens != 300 {
		t.Errorf("PromptTokens = %d, want 300", base.PromptTokens)
	}
}

func TestUsageCacheReadTokensJSON(t *testing.T) {
	raw := `{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cache_read_tokens": 20}`
	var u Usage
	if err := json.Unmarshal([]byte(raw), &u); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if u.CacheReadTokens != 20 {
		t.Errorf("CacheReadTokens = %d, want 20", u.CacheReadTokens)
	}
}

func TestUsageCacheReadTokensOmittedWhenZero(t *testing.T) {
	u := Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	data, err := json.Marshal(u)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := raw["cache_read_tokens"]; ok {
		t.Error("cache_read_tokens should be omitted when zero")
	}
}

func TestUsageOpenAIPromptTokensDetails(t *testing.T) {
	// OpenAI includes cached tokens in a nested prompt_tokens_details field.
	raw := `{
		"prompt_tokens": 100,
		"completion_tokens": 50,
		"total_tokens": 150,
		"prompt_tokens_details": {
			"cached_tokens": 30
		}
	}`
	var u Usage
	if err := json.Unmarshal([]byte(raw), &u); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if u.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30 (from prompt_tokens_details.cached_tokens)", u.CacheReadTokens)
	}
}

func TestUsageExplicitCacheReadTokensTakesPrecedence(t *testing.T) {
	// When cache_read_tokens is explicitly set, prompt_tokens_details should not override it.
	raw := `{
		"prompt_tokens": 100,
		"completion_tokens": 50,
		"total_tokens": 150,
		"cache_read_tokens": 25,
		"prompt_tokens_details": {
			"cached_tokens": 30
		}
	}`
	var u Usage
	if err := json.Unmarshal([]byte(raw), &u); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if u.CacheReadTokens != 25 {
		t.Errorf("CacheReadTokens = %d, want 25 (explicit cache_read_tokens)", u.CacheReadTokens)
	}
}

func TestChatRequestClone(t *testing.T) {
	orig := &ChatRequest{
		Model: "gpt-4o",
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
		Stop:  []string{"END"},
		Tools: []Tool{{Type: "function", Function: FunctionDefinition{Name: "test"}}},
	}

	cloned := orig.clone()

	// Modify the clone's slices.
	cloned.Messages = append(cloned.Messages, Message{Role: RoleAssistant, Content: NewTextContent("Hello")})
	cloned.Stop = append(cloned.Stop, "STOP")
	cloned.Tools = append(cloned.Tools, Tool{Type: "function", Function: FunctionDefinition{Name: "test2"}})
	cloned.Model = "gpt-3.5"

	// Original should be unchanged.
	if len(orig.Messages) != 1 {
		t.Errorf("original messages len = %d, want 1", len(orig.Messages))
	}
	if len(orig.Stop) != 1 {
		t.Errorf("original stop len = %d, want 1", len(orig.Stop))
	}
	if len(orig.Tools) != 1 {
		t.Errorf("original tools len = %d, want 1", len(orig.Tools))
	}
	if orig.Model != "gpt-4o" {
		t.Errorf("original model = %q", orig.Model)
	}
}
