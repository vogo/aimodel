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

package aimodel_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/ais"
	"github.com/vogo/aimodel/provider/anthropic"
)

// These tests exercise the full client pipeline against the Anthropic provider
// through httptest. They assert observable wire behavior (endpoint, headers,
// request shape, response mapping) without referencing provider-internal wire
// types, which now live in the provider/anthropic package.

const anthropicVersion = "2023-06-01"

func TestAnthropicChatCompletion(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %s, want /v1/messages", r.URL.Path)
		}

		if r.Method != http.MethodPost {
			t.Errorf("method = %s", r.Method)
		}

		if r.Header.Get("x-api-key") != "sk-ant-test" {
			t.Errorf("x-api-key = %q", r.Header.Get("x-api-key"))
		}

		if r.Header.Get("anthropic-version") != anthropicVersion {
			t.Errorf("anthropic-version = %q", r.Header.Get("anthropic-version"))
		}

		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("content-type = %q", r.Header.Get("Content-Type"))
		}

		var ar struct {
			Model    string            `json:"model"`
			System   json.RawMessage   `json:"system"`
			Messages []json.RawMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&ar); err != nil {
			t.Fatalf("decode request: %v", err)
		}

		if ar.Model != "claude-sonnet-4" {
			t.Errorf("model = %q", ar.Model)
		}

		var system string
		if err := json.Unmarshal(ar.System, &system); err != nil {
			t.Errorf("unmarshal system: %v", err)
		}

		if system != "You are helpful." {
			t.Errorf("system = %q", system)
		}

		if len(ar.Messages) != 1 {
			t.Errorf("messages len = %d", len(ar.Messages))
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_test","model":"claude-sonnet-4","content":[{"type":"text","text":"Hello!"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-ant-test"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProvider(anthropic.Name),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	result, err := c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []ais.Message{
			{Role: ais.RoleSystem, Content: ais.NewTextContent("You are helpful.")},
			{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")},
		},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
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

	if result.Choices[0].FinishReason != ais.FinishReasonStop {
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
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-bad"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProvider(anthropic.Name),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *ais.APIError
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

// TestAnthropicUserProfileIDOverWire verifies a provider option reaches the
// server on a real request through the pipeline.
func TestAnthropicUserProfileIDOverWire(t *testing.T) {
	var gotProfile, gotVersion string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotProfile = r.Header.Get("anthropic-user-profile-id")
		gotVersion = r.Header.Get("anthropic-version")

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-sonnet-4","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-ant-test"),
		aimodel.WithProvider(anthropic.Name),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProviderOptions(anthropic.Options{UserProfileID: "user_abc123"}),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if gotProfile != "user_abc123" {
		t.Errorf("anthropic-user-profile-id = %q, want user_abc123", gotProfile)
	}

	if gotVersion != anthropicVersion {
		t.Errorf("anthropic-version = %q, want %q", gotVersion, anthropicVersion)
	}
}

func TestAnthropicChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %s", r.URL.Path)
		}

		var ar struct {
			Stream bool `json:"stream"`
		}
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

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-ant-test"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProvider(anthropic.Name),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := c.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
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
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}`))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("sk-ant-test"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProvider(anthropic.Name),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *ais.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}

	if apiErr.StatusCode != 429 {
		t.Errorf("status = %d", apiErr.StatusCode)
	}
}

// TestAnthropicEnvFallback verifies ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL sit
// in the generic env fallback chain: a client built purely from those env vars
// (with the Anthropic provider selected) sends to the configured base URL.
func TestAnthropicEnvFallback(t *testing.T) {
	var reached bool

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true

		if r.Header.Get("x-api-key") != "sk-ant-env" {
			t.Errorf("x-api-key = %q, want sk-ant-env", r.Header.Get("x-api-key"))
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-sonnet-4","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()

	t.Setenv("AI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-env")
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_BASE_URL", srv.URL)

	c, err := aimodel.NewClient(aimodel.WithProvider(anthropic.Name))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if !reached {
		t.Error("request did not reach the env-configured base URL")
	}
}
