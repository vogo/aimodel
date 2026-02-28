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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestAnthropicChatCompletion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify endpoint.
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %s, want /v1/messages", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %s", r.Method)
		}

		// Verify headers.
		if r.Header.Get("x-api-key") != "sk-ant-test" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != anthropicAPIVersion {
			t.Errorf("anthropic-version = %q", r.Header.Get("anthropic-version"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %q", r.Header.Get("Content-Type"))
		}

		// Verify request body is in Anthropic format.
		var ar anthropicRequest
		if err := json.NewDecoder(r.Body).Decode(&ar); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if ar.Model != ModelAnthropicClaude4Sonnet {
			t.Errorf("model = %q", ar.Model)
		}
		if ar.System != "You are helpful." {
			t.Errorf("system = %q", ar.System)
		}
		if len(ar.Messages) != 1 {
			t.Errorf("messages len = %d", len(ar.Messages))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(anthropicResponse{
			ID:    "msg_test",
			Model: ModelAnthropicClaude4Sonnet,
			Content: []anthropicContentBlock{
				{Type: "text", Text: "Hello!"},
			},
			StopReason: "end_turn",
			Usage:      anthropicUsage{InputTokens: 10, OutputTokens: 5},
		})
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-ant-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	result, err := c.AnthropicChatCompletion(context.Background(), &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Messages: []Message{
			{Role: RoleSystem, Content: NewTextContent("You are helpful.")},
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	})
	if err != nil {
		t.Fatalf("AnthropicChatCompletion: %v", err)
	}

	if result.ID != "msg_test" {
		t.Errorf("id = %q", result.ID)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("choices len = %d", len(result.Choices))
	}
	if result.Choices[0].Message.Content.Text() != "Hello!" {
		t.Errorf("content = %q", result.Choices[0].Message.Content.Text())
	}
	if result.Choices[0].FinishReason != FinishReasonStop {
		t.Errorf("finish_reason = %q", result.Choices[0].FinishReason)
	}
	if result.Usage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d", result.Usage.TotalTokens)
	}
}

func TestAnthropicChatCompletionAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(anthropicErrorResponse{
			Type: "error",
			Error: anthropicError{
				Type:    "authentication_error",
				Message: "invalid x-api-key",
			},
		})
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-bad"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.AnthropicChatCompletion(context.Background(), &ChatRequest{
		Model:    ModelAnthropicClaude4Sonnet,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 401 {
		t.Errorf("status = %d", apiErr.StatusCode)
	}
	if apiErr.Type != "authentication_error" {
		t.Errorf("type = %q", apiErr.Type)
	}
}

func TestAnthropicChatCompletionDefaultBaseURL(t *testing.T) {
	c, err := NewClient(WithAPIKey("sk-ant-test"))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	// Verify anthropicBaseURL returns default when baseURL is empty.
	if got := c.anthropicBaseURL(); got != anthropicDefaultBaseURL {
		t.Errorf("anthropicBaseURL() = %q, want %q", got, anthropicDefaultBaseURL)
	}
}

func TestAnthropicChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %s", r.URL.Path)
		}

		// Verify stream is set in request body.
		var ar anthropicRequest
		if err := json.NewDecoder(r.Body).Decode(&ar); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if !ar.Stream {
			t.Error("stream should be true")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		events := []string{
			"event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}`,
			"event: content_block_start\ndata: " + `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			"event: ping\ndata: " + `{"type":"ping"}`,
			"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`,
			"event: content_block_stop\ndata: " + `{"type":"content_block_stop","index":0}`,
			"event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}`,
			"event: message_stop\ndata: " + `{"type":"message_stop"}`,
		}

		for _, e := range events {
			_, _ = fmt.Fprintf(w, "%s\n\n", e)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-ant-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := c.AnthropicChatCompletionStream(context.Background(), &ChatRequest{
		Model:    ModelAnthropicClaude4Sonnet,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("AnthropicChatCompletionStream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	var contents []string
	var gotFinishReason bool

	for {
		chunk, err := stream.Recv()
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
			if chunk.Choices[0].FinishReason != nil {
				gotFinishReason = true
			}
		}
	}

	if len(contents) != 2 {
		t.Fatalf("got %d text chunks, want 2: %v", len(contents), contents)
	}
	if contents[0] != "Hello" || contents[1] != " world" {
		t.Errorf("contents = %v", contents)
	}
	if !gotFinishReason {
		t.Error("expected finish reason chunk")
	}
}

func TestAnthropicChatCompletionStreamAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(anthropicErrorResponse{
			Type: "error",
			Error: anthropicError{
				Type:    "rate_limit_error",
				Message: "Rate limited",
			},
		})
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-ant-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.AnthropicChatCompletionStream(context.Background(), &ChatRequest{
		Model:    ModelAnthropicClaude4Sonnet,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 429 {
		t.Errorf("status = %d", apiErr.StatusCode)
	}
}

func TestAnthropicEnvFallback(t *testing.T) {
	t.Setenv("AI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-env")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_BASE_URL", "https://custom.anthropic.com")

	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.apiKey != "sk-ant-env" {
		t.Errorf("apiKey = %q, want sk-ant-env", c.apiKey)
	}
	if c.baseURL != "https://custom.anthropic.com" {
		t.Errorf("baseURL = %q", c.baseURL)
	}
}
