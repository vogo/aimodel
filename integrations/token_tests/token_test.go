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

package token_tests

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vogo/aimodel"
)

// Test 1a: Anthropic sync response with cache_read_input_tokens.
// Sends a chat completion request to a mock Anthropic server that returns
// known cache_read_input_tokens and verifies CacheReadTokens is populated.
func TestIntegration_Anthropic_CacheReadTokens_Sync(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Anthropic response with cache_read_input_tokens.
		_, _ = w.Write([]byte(`{
			"id": "msg_01",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4",
			"content": [{"type": "text", "text": "Hello"}],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 50,
				"cache_creation_input_tokens": 10,
				"cache_read_input_tokens": 30,
				"output_tokens": 20
			}
		}`))
	}))
	defer srv.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-test"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProtocol(aimodel.ProtocolAnthropic),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	resp, err := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	// PromptTokens should be totalInputTokens() = input_tokens + cache_creation + cache_read = 50 + 10 + 30 = 90.
	if resp.Usage.PromptTokens != 90 {
		t.Errorf("PromptTokens = %d, want 90", resp.Usage.PromptTokens)
	}

	if resp.Usage.CompletionTokens != 20 {
		t.Errorf("CompletionTokens = %d, want 20", resp.Usage.CompletionTokens)
	}

	// CacheReadTokens should be 30.
	if resp.Usage.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", resp.Usage.CacheReadTokens)
	}
}

// Test 1b: Anthropic streaming response with cache_read_input_tokens.
// Sends a streaming request to a mock Anthropic server and verifies
// CacheReadTokens is propagated through the stream usage.
func TestIntegration_Anthropic_CacheReadTokens_Stream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		events := []string{
			`event: message_start` + "\n" +
				`data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"stop_reason":null,"usage":{"input_tokens":50,"cache_creation_input_tokens":10,"cache_read_input_tokens":30,"output_tokens":0}}}` + "\n\n",
			`event: content_block_start` + "\n" +
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n",
			`event: content_block_delta` + "\n" +
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}` + "\n\n",
			`event: content_block_stop` + "\n" +
				`data: {"type":"content_block_stop","index":0}` + "\n\n",
			`event: message_delta` + "\n" +
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}` + "\n\n",
			`event: message_stop` + "\n" +
				`data: {"type":"message_stop"}` + "\n\n",
		}

		for _, event := range events {
			_, _ = fmt.Fprint(w, event)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-test"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProtocol(aimodel.ProtocolAnthropic),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := client.ChatCompletionStream(context.Background(), &aimodel.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}

	// Drain stream.
	for {
		_, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			t.Fatalf("Recv: %v", recvErr)
		}
	}

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("stream.Usage() = nil, want non-nil")
	}

	// PromptTokens = totalInputTokens() = 50 + 10 + 30 = 90.
	if usage.PromptTokens != 90 {
		t.Errorf("PromptTokens = %d, want 90", usage.PromptTokens)
	}

	if usage.CompletionTokens != 20 {
		t.Errorf("CompletionTokens = %d, want 20", usage.CompletionTokens)
	}

	// CacheReadTokens should be 30.
	if usage.CacheReadTokens != 30 {
		t.Errorf("CacheReadTokens = %d, want 30", usage.CacheReadTokens)
	}

	_ = stream.Close()
}

// Test 1c: OpenAI sync response with prompt_tokens_details.cached_tokens.
// Sends a chat completion to a mock OpenAI server returning cached_tokens
// in the nested prompt_tokens_details and verifies CacheReadTokens.
func TestIntegration_OpenAI_CacheReadTokens_Sync(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-01",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "gpt-4o",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello"},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 50,
				"total_tokens": 150,
				"prompt_tokens_details": {
					"cached_tokens": 40
				}
			}
		}`))
	}))
	defer srv.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-test"),
		aimodel.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	resp, err := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Model:    "gpt-4o",
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if resp.Usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", resp.Usage.PromptTokens)
	}

	if resp.Usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", resp.Usage.CompletionTokens)
	}

	// CacheReadTokens should be extracted from prompt_tokens_details.cached_tokens.
	if resp.Usage.CacheReadTokens != 40 {
		t.Errorf("CacheReadTokens = %d, want 40", resp.Usage.CacheReadTokens)
	}
}

// Test 1d: OpenAI streaming response with prompt_tokens_details.cached_tokens.
// Sends a streaming request to a mock OpenAI server and verifies
// CacheReadTokens is extracted from the final usage chunk.
func TestIntegration_OpenAI_CacheReadTokens_Stream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		chunks := []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":40}}}`,
		}

		for _, chunk := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			flusher.Flush()
		}

		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-test"),
		aimodel.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := client.ChatCompletionStream(context.Background(), &aimodel.ChatRequest{
		Model:    "gpt-4o",
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}

	// Drain stream.
	for {
		_, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			t.Fatalf("Recv: %v", recvErr)
		}
	}

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("stream.Usage() = nil, want non-nil")
	}

	if usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", usage.PromptTokens)
	}

	if usage.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", usage.CompletionTokens)
	}

	// CacheReadTokens should be 40.
	if usage.CacheReadTokens != 40 {
		t.Errorf("CacheReadTokens = %d, want 40", usage.CacheReadTokens)
	}

	_ = stream.Close()
}

// Test 1e: OpenAI sync response without prompt_tokens_details.
// Verifies CacheReadTokens defaults to 0 when no cached_tokens present.
func TestIntegration_OpenAI_NoCacheTokens_Sync(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-02",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "gpt-4o",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello"},
				"finish_reason": "stop"
			}],
			"usage": {
				"prompt_tokens": 100,
				"completion_tokens": 50,
				"total_tokens": 150
			}
		}`))
	}))
	defer srv.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-test"),
		aimodel.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	resp, err := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Model:    "gpt-4o",
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	// No cache tokens present, should be 0.
	if resp.Usage.CacheReadTokens != 0 {
		t.Errorf("CacheReadTokens = %d, want 0", resp.Usage.CacheReadTokens)
	}
}

// Test 1f: Anthropic sync response with zero cache_read_input_tokens.
// Verifies CacheReadTokens is 0 when no caching occurred.
func TestIntegration_Anthropic_ZeroCacheTokens_Sync(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_02",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4",
			"content": [{"type": "text", "text": "Hello"}],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 50,
				"cache_creation_input_tokens": 0,
				"cache_read_input_tokens": 0,
				"output_tokens": 20
			}
		}`))
	}))
	defer srv.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-test"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProtocol(aimodel.ProtocolAnthropic),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	resp, err := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if resp.Usage.CacheReadTokens != 0 {
		t.Errorf("CacheReadTokens = %d, want 0", resp.Usage.CacheReadTokens)
	}
}
