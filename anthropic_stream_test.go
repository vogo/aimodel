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
	"errors"
	"io"
	"strings"
	"testing"
)

func TestAnthropicStreamTextDelta(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var contents []string

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if len(chunk.Choices) > 0 {
			text := chunk.Choices[0].Delta.Content.Text()
			if text != "" {
				contents = append(contents, text)
			}
		}
	}

	if len(contents) != 2 {
		t.Fatalf("got %d text chunks, want 2: %v", len(contents), contents)
	}
	if contents[0] != "Hi" || contents[1] != " there" {
		t.Errorf("contents = %v", contents)
	}
}

func TestAnthropicStreamToolUse(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"NYC\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var chunks []*StreamChunk

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}

		chunks = append(chunks, chunk)
	}

	if len(chunks) < 3 {
		t.Fatalf("got %d chunks, want at least 3", len(chunks))
	}

	// First chunk: tool_use start with name.
	first := chunks[0]
	if len(first.Choices[0].Delta.ToolCalls) != 1 {
		t.Fatal("expected tool call in first chunk")
	}
	if first.Choices[0].Delta.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("name = %q", first.Choices[0].Delta.ToolCalls[0].Function.Name)
	}
	if first.Choices[0].Delta.ToolCalls[0].ID != "toolu_1" {
		t.Errorf("id = %q", first.Choices[0].Delta.ToolCalls[0].ID)
	}
}

func TestAnthropicStreamPingSkip(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_3","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}` + "\n\n" +
		"event: ping\n" +
		`data: {"type":"ping"}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	chunk, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if chunk.Choices[0].Delta.Content.Text() != "ok" {
		t.Errorf("content = %q", chunk.Choices[0].Delta.Content.Text())
	}
}

func TestAnthropicStreamError(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_4","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}` + "\n\n" +
		"event: error\n" +
		`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	_, err := s.Recv()
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.Type != "overloaded_error" {
		t.Errorf("type = %q", apiErr.Type)
	}
	if apiErr.Message != "Overloaded" {
		t.Errorf("message = %q", apiErr.Message)
	}
}

func TestAnthropicStreamClose(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_5","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	_, err := s.Recv()
	if !errors.Is(err, ErrStreamClosed) {
		t.Errorf("got %v, want ErrStreamClosed", err)
	}
}

func TestAnthropicStreamThinking(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_t1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" think..."}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc123"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 42."}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":1}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var thinkingParts []string

	var textParts []string

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			if delta.Thinking != "" {
				thinkingParts = append(thinkingParts, delta.Thinking)
			}

			if text := delta.Content.Text(); text != "" {
				textParts = append(textParts, text)
			}
		}
	}

	if len(thinkingParts) != 2 {
		t.Fatalf("got %d thinking chunks, want 2: %v", len(thinkingParts), thinkingParts)
	}

	if thinkingParts[0] != "Let me" || thinkingParts[1] != " think..." {
		t.Errorf("thinking parts = %v", thinkingParts)
	}

	if len(textParts) != 1 || textParts[0] != "The answer is 42." {
		t.Errorf("text parts = %v", textParts)
	}
}

func TestAnthropicStreamToolUseIndexMapping(t *testing.T) {
	// Anthropic uses global content block indices (text=0, tool_use=1, tool_use=2).
	// The stream must remap these to tool-call-scoped indices (0, 1).
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_idx","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		// Block 0: text
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I'll call two tools."}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		// Block 1: first tool_use (tool index 0)
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_a","name":"get_weather"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"NYC\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":1}` + "\n\n" +
		// Block 2: second tool_use (tool index 1)
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_b","name":"get_time"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"tz\":"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"UTC\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":2}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var msg Message

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if len(chunk.Choices) > 0 {
			msg.AppendDelta(&chunk.Choices[0].Delta)
		}
	}

	// Verify text content.
	if msg.Content.Text() != "I'll call two tools." {
		t.Errorf("content = %q", msg.Content.Text())
	}

	// Verify two tool calls with correct indices.
	if len(msg.ToolCalls) != 2 {
		t.Fatalf("tool_calls len = %d, want 2", len(msg.ToolCalls))
	}

	tc0 := msg.ToolCalls[0]
	if tc0.Index != 0 {
		t.Errorf("tool_calls[0].index = %d, want 0", tc0.Index)
	}
	if tc0.ID != "toolu_a" {
		t.Errorf("tool_calls[0].id = %q, want toolu_a", tc0.ID)
	}
	if tc0.Function.Name != "get_weather" {
		t.Errorf("tool_calls[0].name = %q", tc0.Function.Name)
	}
	if tc0.Function.Arguments != `{"city":"NYC"}` {
		t.Errorf("tool_calls[0].arguments = %q", tc0.Function.Arguments)
	}

	tc1 := msg.ToolCalls[1]
	if tc1.Index != 1 {
		t.Errorf("tool_calls[1].index = %d, want 1", tc1.Index)
	}
	if tc1.ID != "toolu_b" {
		t.Errorf("tool_calls[1].id = %q, want toolu_b", tc1.ID)
	}
	if tc1.Function.Name != "get_time" {
		t.Errorf("tool_calls[1].name = %q", tc1.Function.Name)
	}
	if tc1.Function.Arguments != `{"tz":"UTC"}` {
		t.Errorf("tool_calls[1].arguments = %q", tc1.Function.Arguments)
	}
}

func TestAnthropicStreamThinkingWithToolUse(t *testing.T) {
	// Thinking(0) + text(1) + tool_use(2): tool index should be 0, not 2.
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_tt","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		// Block 0: thinking
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"I need to call a tool."}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		// Block 1: text
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Calling tool."}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":1}` + "\n\n" +
		// Block 2: tool_use (should become tool index 0)
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_c","name":"calculator"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"expr\":\"2+2\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":2}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":15}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var msg Message

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if len(chunk.Choices) > 0 {
			msg.AppendDelta(&chunk.Choices[0].Delta)
		}
	}

	if msg.Thinking != "I need to call a tool." {
		t.Errorf("thinking = %q", msg.Thinking)
	}

	if msg.Content.Text() != "Calling tool." {
		t.Errorf("content = %q", msg.Content.Text())
	}

	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d, want 1", len(msg.ToolCalls))
	}

	tc := msg.ToolCalls[0]
	if tc.Index != 0 {
		t.Errorf("tool_calls[0].index = %d, want 0 (not the Anthropic block index 2)", tc.Index)
	}
	if tc.ID != "toolu_c" {
		t.Errorf("tool_calls[0].id = %q", tc.ID)
	}
	if tc.Function.Name != "calculator" {
		t.Errorf("tool_calls[0].name = %q", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"expr":"2+2"}` {
		t.Errorf("tool_calls[0].arguments = %q", tc.Function.Arguments)
	}
}

func TestAnthropicStreamFinishReason(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_6","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":100}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	// Skip text chunk.
	_, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv text: %v", err)
	}

	// Get finish reason chunk.
	chunk, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv delta: %v", err)
	}
	if chunk.Choices[0].FinishReason == nil {
		t.Fatal("expected finish reason")
	}
	if *chunk.Choices[0].FinishReason != string(FinishReasonLength) {
		t.Errorf("finish_reason = %q, want %q", *chunk.Choices[0].FinishReason, FinishReasonLength)
	}

	// Verify usage is included in the message_delta chunk.
	if chunk.Usage == nil {
		t.Fatal("expected usage in message_delta chunk")
	}
	if chunk.Usage.PromptTokens != 5 {
		t.Errorf("prompt_tokens = %d, want 5", chunk.Usage.PromptTokens)
	}
	if chunk.Usage.CompletionTokens != 100 {
		t.Errorf("completion_tokens = %d, want 100", chunk.Usage.CompletionTokens)
	}
	if chunk.Usage.TotalTokens != 105 {
		t.Errorf("total_tokens = %d, want 105", chunk.Usage.TotalTokens)
	}
}
