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

func TestChatCompletion(t *testing.T) {
	resp := ChatResponse{
		ID:      "chatcmpl-test",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   ModelOpenaiGPT4o,
		Choices: []Choice{
			{
				Index:        0,
				Message:      Message{Role: RoleAssistant, Content: NewTextContent("Hello!")},
				FinishReason: FinishReasonStop,
			},
		},
		Usage: Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer sk-test" {
			t.Errorf("auth = %s", r.Header.Get("Authorization"))
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %s", r.Header.Get("Content-Type"))
		}

		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != ModelOpenaiGPT4o {
			t.Errorf("request model = %q", req.Model)
		}
		if req.Stream {
			t.Error("stream should be false for non-streaming")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	result, err := c.ChatCompletion(context.Background(), &ChatRequest{
		Model: ModelOpenaiGPT4o,
		Messages: []Message{
			{Role: RoleUser, Content: NewTextContent("Hi")},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if result.ID != "chatcmpl-test" {
		t.Errorf("id = %q", result.ID)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("choices len = %d", len(result.Choices))
	}
	if result.Choices[0].Message.Content.Text() != "Hello!" {
		t.Errorf("content = %q", result.Choices[0].Message.Content.Text())
	}
	if result.Usage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d", result.Usage.TotalTokens)
	}
}

func TestChatCompletionAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "Rate limit exceeded",
				"type":    "tokens",
				"code":    "rate_limit_exceeded",
			},
		})
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ChatRequest{
		Model:    ModelOpenaiGPT4o,
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
	if apiErr.Code != "rate_limit_exceeded" {
		t.Errorf("code = %q", apiErr.Code)
	}
}

func TestChatCompletionEmptyChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ChatResponse{
			ID:      "chatcmpl-empty",
			Choices: []Choice{},
		})
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ChatRequest{
		Model:    ModelOpenaiGPT4o,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("Hi")}},
	})
	if !errors.Is(err, ErrEmptyResponse) {
		t.Errorf("got %v, want ErrEmptyResponse", err)
	}
}

func TestChatCompletionCancelledContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request should not reach server")
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = c.ChatCompletion(ctx, &ChatRequest{
		Model:    ModelOpenaiGPT4o,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}

		if !req.Stream {
			t.Error("stream should be true")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}

		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := c.ChatCompletionStream(context.Background(), &ChatRequest{
		Model:    ModelOpenaiGPT4o,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	var contents []string
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if len(chunk.Choices) > 0 {
			contents = append(contents, chunk.Choices[0].Delta.Content.Text())
		}
	}

	if len(contents) != 3 {
		t.Fatalf("got %d chunks, want 3", len(contents))
	}
	if contents[0] != "Hello" || contents[1] != " world" {
		t.Errorf("contents = %v", contents)
	}
}

func TestChatCompletionStreamAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "Invalid API key",
				"type":    "invalid_request_error",
				"code":    "invalid_api_key",
			},
		})
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-bad"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletionStream(context.Background(), &ChatRequest{
		Model:    ModelOpenaiGPT4o,
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
}

func TestChatCompletionWithTools(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}

		if len(req.Tools) != 1 {
			t.Errorf("tools len = %d", len(req.Tools))
		}
		if req.Tools[0].Function.Name != "get_weather" {
			t.Errorf("tool name = %q", req.Tools[0].Function.Name)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ChatResponse{
			ID: "chatcmpl-tools",
			Choices: []Choice{
				{
					Index: 0,
					Message: Message{
						Role: RoleAssistant,
						ToolCalls: []ToolCall{
							{
								ID:   "call_1",
								Type: "function",
								Function: FunctionCall{
									Name:      "get_weather",
									Arguments: `{"city":"NYC"}`,
								},
							},
						},
					},
					FinishReason: FinishReasonToolCalls,
				},
			},
		})
	}))
	defer srv.Close()

	c, err := NewClient(WithAPIKey("sk-test"), WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	result, err := c.ChatCompletion(context.Background(), &ChatRequest{
		Model:    ModelOpenaiGPT4o,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("weather?")}},
		Tools: []Tool{
			{
				Type: "function",
				Function: FunctionDefinition{
					Name:        "get_weather",
					Description: "Get current weather",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"city": map[string]any{"type": "string"},
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if len(result.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d", len(result.Choices[0].Message.ToolCalls))
	}
	tc := result.Choices[0].Message.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("function name = %q", tc.Function.Name)
	}
}
