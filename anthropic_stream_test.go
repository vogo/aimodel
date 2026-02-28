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
}
