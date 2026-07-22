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

package e2e_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/ais"
)

func TestChatCompletion(t *testing.T) {
	resp := ais.ChatResponse{
		ID:      "chatcmpl-test",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   ais.ModelOpenaiGPT4o,
		Choices: []ais.Choice{
			{
				Index:        0,
				Message:      ais.Message{Role: ais.RoleAssistant, Content: ais.NewTextContent("Hello!")},
				FinishReason: ais.FinishReasonStop,
			},
		},
		Usage: ais.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
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

		var req ais.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != ais.ModelOpenaiGPT4o {
			t.Errorf("request model = %q", req.Model)
		}
		if req.Stream {
			t.Error("stream should be false for non-streaming")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	result, err := c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model: ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{
			{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")},
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

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *ais.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *ais.APIError, got %T: %v", err, err)
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
		_ = json.NewEncoder(w).Encode(ais.ChatResponse{
			ID:      "chatcmpl-empty",
			Choices: []ais.Choice{},
		})
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if !errors.Is(err, ais.ErrEmptyResponse) {
		t.Errorf("got %v, want ais.ErrEmptyResponse", err)
	}
}

func TestChatCompletionCancelledContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("request should not reach server")
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = c.ChatCompletion(ctx, &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ais.ChatRequest
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

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	stream, err := c.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
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

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-bad"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	_, err = c.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *ais.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *ais.APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 401 {
		t.Errorf("status = %d", apiErr.StatusCode)
	}
}

func TestChatCompletionStreamAutoInjectsStreamOptions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream        bool `json:"stream"`
			StreamOptions *struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}

		if !req.Stream {
			t.Error("Stream = false, want true")
		} else if req.StreamOptions == nil {
			t.Error("StreamOptions = nil, want non-nil")
		} else if !req.StreamOptions.IncludeUsage {
			t.Error("StreamOptions.IncludeUsage = false, want true")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	stream, err := c.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	// Drain the stream.
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}
}

func TestChatCompletionStreamUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":16,"total_tokens":24}}`,
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}

		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	stream, err := c.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("Usage() = nil, want non-nil")
	}
	if usage.PromptTokens != 8 {
		t.Errorf("prompt_tokens = %d, want 8", usage.PromptTokens)
	}
	if usage.CompletionTokens != 16 {
		t.Errorf("completion_tokens = %d, want 16", usage.CompletionTokens)
	}
	if usage.TotalTokens != 24 {
		t.Errorf("total_tokens = %d, want 24", usage.TotalTokens)
	}
}

func TestChatCompletionStreamUsageCacheReadTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":50,"completion_tokens":10,"total_tokens":60,"prompt_tokens_details":{"cached_tokens":15}}}`,
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}

		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	stream, err := c.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("Usage() = nil, want non-nil")
	}
	if usage.CacheReadTokens != 15 {
		t.Errorf("CacheReadTokens = %d, want 15", usage.CacheReadTokens)
	}
}

func TestChatCompletionWithCacheReadTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-cache",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "gpt-4o",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello!"},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 20,
				"total_tokens": 120,
				"prompt_tokens_details": {
					"cached_tokens": 30
				}
			}
		}`))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	result, err := c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if result.Usage.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", result.Usage.CacheReadTokens)
	}
	if result.Usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", result.Usage.PromptTokens)
	}
}

func TestChatRequestMaxTokensSerialization(t *testing.T) {
	maxCompletion := 256
	maxTokens := 512

	tests := []struct {
		name            string
		req             ais.ChatRequest
		wantContains    []string
		wantNotContains []string
	}{
		{
			name: "only MaxCompletionTokens",
			req: ais.ChatRequest{
				Model:               ais.ModelOpenaiGPT4o,
				MaxCompletionTokens: &maxCompletion,
			},
			wantContains:    []string{`"max_completion_tokens":256`},
			wantNotContains: []string{"max_tokens"},
		},
		{
			name: "only MaxTokens",
			req: ais.ChatRequest{
				Model:     ais.ModelOpenaiGPT4o,
				MaxTokens: &maxTokens,
			},
			wantContains:    []string{`"max_tokens":512`},
			wantNotContains: []string{"max_completion_tokens"},
		},
		{
			name: "neither set",
			req: ais.ChatRequest{
				Model: ais.ModelOpenaiGPT4o,
			},
			wantNotContains: []string{"max_tokens", "max_completion_tokens"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(&tt.req)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			got := string(data)

			for _, want := range tt.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("body %s does not contain %q", got, want)
				}
			}
			for _, notWant := range tt.wantNotContains {
				if strings.Contains(got, notWant) {
					t.Errorf("body %s should not contain %q", got, notWant)
				}
			}
		})
	}
}

func TestChatCompletionWithTools(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ais.ChatRequest
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
		_ = json.NewEncoder(w).Encode(ais.ChatResponse{
			ID: "chatcmpl-tools",
			Choices: []ais.Choice{
				{
					Index: 0,
					Message: ais.Message{
						Role: ais.RoleAssistant,
						ToolCalls: []ais.ToolCall{
							{
								ID:   "call_1",
								Type: "function",
								Function: ais.FunctionCall{
									Name:      "get_weather",
									Arguments: `{"city":"NYC"}`,
								},
							},
						},
					},
					FinishReason: ais.FinishReasonToolCalls,
				},
			},
		})
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	result, err := c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("weather?")}},
		Tools: []ais.Tool{
			{
				Type: "function",
				Function: ais.FunctionDefinition{
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

// captureRequestBody starts a test server that records the raw request body of
// the next chat completion and returns a minimal valid response.
func captureRequestBody(t *testing.T) (*httptest.Server, *map[string]json.RawMessage) {
	t.Helper()

	var captured map[string]json.RawMessage

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		// Reset to a fresh map so a prior request's keys don't linger
		// (json.Unmarshal merges into an existing map rather than replacing it).
		captured = nil
		if err := json.Unmarshal(body, &captured); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ais.ChatResponse{
			ID: "chatcmpl-capture",
			Choices: []ais.Choice{
				{Index: 0, Message: ais.Message{Role: ais.RoleAssistant, Content: ais.NewTextContent("ok")}, FinishReason: ais.FinishReasonStop},
			},
		})
	}))

	return srv, &captured
}

func TestOpenAIChatRequestReasoningEffortValues(t *testing.T) {
	srv, captured := captureRequestBody(t)
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	efforts := []string{
		ais.ReasoningEffortNone,
		ais.ReasoningEffortMinimal,
		ais.ReasoningEffortLow,
		ais.ReasoningEffortMedium,
		ais.ReasoningEffortHigh,
		ais.ReasoningEffortXHigh,
	}

	for _, effort := range efforts {
		t.Run(effort, func(t *testing.T) {
			if _, err := c.ChatCompletion(context.Background(), &ais.ChatRequest{
				Model:           ais.ModelOpenaiGPT4o,
				Messages:        []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
				ReasoningEffort: effort,
			}); err != nil {
				t.Fatalf("ChatCompletion: %v", err)
			}

			val, ok := (*captured)["reasoning_effort"]
			if !ok {
				t.Fatal("reasoning_effort not present in request body")
			}
			if string(val) != `"`+effort+`"` {
				t.Errorf("reasoning_effort = %s, want %q", val, effort)
			}
		})
	}

	// Verify the constants match the official literal values.
	want := []string{"none", "minimal", "low", "medium", "high", "xhigh"}
	for i, effort := range efforts {
		if effort != want[i] {
			t.Errorf("constant[%d] = %q, want %q", i, effort, want[i])
		}
	}
}

func TestOpenAIChatRequestCommonFields(t *testing.T) {
	srv, captured := captureRequestBody(t)
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	parallel := false

	if _, err := c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:             ais.ModelOpenaiGPT4o,
		Messages:          []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
		ParallelToolCalls: &parallel,
	}); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	wantKeys := map[string]string{
		"parallel_tool_calls": `false`,
	}

	for key, want := range wantKeys {
		val, ok := (*captured)[key]
		if !ok {
			t.Errorf("%s not present in request body", key)
			continue
		}
		if string(val) != want {
			t.Errorf("%s = %s, want %s", key, val, want)
		}
	}
}

func TestOpenAIChatRequestCommonFieldsOmitEmpty(t *testing.T) {
	srv, captured := captureRequestBody(t)
	defer srv.Close()

	c, err := aimodel.NewClient(aimodel.WithAPIKey("sk-test"), aimodel.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("aimodel.NewClient: %v", err)
	}

	// No optional fields set: all of them must be omitted.
	if _, err := c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    ais.ModelOpenaiGPT4o,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	}); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	for _, key := range []string{"parallel_tool_calls"} {
		if _, ok := (*captured)[key]; ok {
			t.Errorf("%s should be omitted when empty", key)
		}
	}
}
