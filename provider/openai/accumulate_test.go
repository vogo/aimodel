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

package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// streamServer replays a fixed SSE body, so accumulation is tested against
// exact wire bytes rather than a live backend.
func streamServer(t *testing.T, body string) *Client {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	return NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
}

// drain reads a stream to completion, the way a caller does.
func drain(t *testing.T, stream *ChatCompletionStream) {
	t.Helper()

	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}
}

func TestStreamAccumulatesTextReasoningAndToolCallFragments(t *testing.T) {
	const body = `data: {"id":"chat-1","object":"chat.completion.chunk","created":7,"model":"gpt-5","system_fingerprint":"fp_1","choices":[{"index":0,"delta":{"role":"assistant","content":"He"}}]}

data: {"id":"chat-1","choices":[{"index":0,"delta":{"content":"llo","reasoning_content":"let me "}}]}

data: {"id":"chat-1","choices":[{"index":0,"delta":{"reasoning_content":"think","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"ci"}}]}}]}

data: {"id":"chat-1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"SF\"}"}}]},"finish_reason":"tool_calls"}]}

data: {"id":"chat-1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":3}}}

data: [DONE]

`

	stream, err := streamServer(t, body).ChatCompletionsStream(context.Background(), &ChatCompletionRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	drain(t, stream)

	response := stream.Response()
	if response == nil {
		t.Fatal("Response is nil after a stream that carried chunks")
	}

	if response.ID != "chat-1" || response.Model != "gpt-5" || response.Created != 7 || response.SystemFingerprint != "fp_1" {
		t.Errorf("envelope = %+v", response)
	}

	if response.Object != objectChatCompletion {
		t.Errorf("object = %q, want the unary discriminator %q", response.Object, objectChatCompletion)
	}

	if len(response.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(response.Choices))
	}

	choice := response.Choices[0]
	if got := choice.Message.Content.Text(); got != "Hello" {
		t.Errorf("content = %q, want %q", got, "Hello")
	}

	if choice.Message.ReasoningContent != "let me think" {
		t.Errorf("reasoning content = %q", choice.Message.ReasoningContent)
	}

	if choice.FinishReason == nil || *choice.FinishReason != "tool_calls" {
		t.Errorf("finish reason = %v", choice.FinishReason)
	}

	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(choice.Message.ToolCalls))
	}

	call := choice.Message.ToolCalls[0]
	if call.ID != "call_1" || call.Type != "function" || call.Function.Name != "get_weather" {
		t.Errorf("tool call identity = %+v", call)
	}

	if call.Function.Arguments != `{"city":"SF"}` {
		t.Errorf("tool call arguments = %q, want the reassembled JSON", call.Function.Arguments)
	}

	usage := stream.Usage()
	if usage == nil || usage.TotalTokens != 15 || usage.CompletionTokensDetails.ReasoningTokens != 3 {
		t.Errorf("usage = %+v", usage)
	}

	if response.Usage != usage {
		t.Error("Response().Usage and Usage() must report the same object")
	}
}

func TestStreamAccumulatesParallelChoicesAndRefusals(t *testing.T) {
	const body = `data: {"id":"chat-2","choices":[{"index":1,"delta":{"role":"assistant","content":"second"}}]}

data: {"id":"chat-2","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I can"}}]}

data: {"id":"chat-2","choices":[{"index":0,"delta":{"refusal":"not"},"finish_reason":"content_filter"}]}

data: [DONE]

`

	stream, err := streamServer(t, body).ChatCompletionsStream(context.Background(), &ChatCompletionRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	drain(t, stream)

	response := stream.Response()
	if len(response.Choices) != 2 {
		t.Fatalf("choices = %d, want 2 (an out-of-order index must not drop the lower one)", len(response.Choices))
	}

	if response.Choices[0].Message.Refusal != "I cannot" {
		t.Errorf("refusal = %q", response.Choices[0].Message.Refusal)
	}

	if got := response.Choices[1].Message.Content.Text(); got != "second" {
		t.Errorf("choice 1 content = %q", got)
	}

	if response.Choices[0].Index != 0 || response.Choices[1].Index != 1 {
		t.Errorf("indexes = %d,%d", response.Choices[0].Index, response.Choices[1].Index)
	}

	if stream.Usage() != nil {
		t.Error("Usage must stay nil when the backend reports none")
	}
}

func TestStreamAccumulationIsAvailableWhileReading(t *testing.T) {
	const body = `data: {"id":"chat-3","choices":[{"index":0,"delta":{"content":"partial"}}]}

data: {"id":"chat-3","choices":[{"index":0,"delta":{"content":" and rest"}}]}

data: [DONE]

`

	stream, err := streamServer(t, body).ChatCompletionsStream(context.Background(), &ChatCompletionRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	if stream.Response() != nil {
		t.Error("Response must be nil before the first chunk")
	}

	if _, err = stream.Recv(); err != nil {
		t.Fatal(err)
	}

	if got := stream.Response().Choices[0].Message.Content.Text(); got != "partial" {
		t.Errorf("live snapshot = %q, want the first chunk only", got)
	}

	drain(t, stream)

	if got := stream.Response().Choices[0].Message.Content.Text(); got != "partial and rest" {
		t.Errorf("final content = %q", got)
	}
}
