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

package composes

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vogo/aimodel"
)

// newTestServer creates an httptest server that returns a valid OpenAI chat response
// with the given model name in the response.
func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the model is set correctly in the request.
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		resp := aimodel.ChatResponse{
			ID:    "test-id",
			Model: fmt.Sprintf("%v", req["model"]),
			Choices: []aimodel.Choice{
				{
					Index:        0,
					Message:      aimodel.Message{Role: aimodel.RoleAssistant, Content: aimodel.NewTextContent("hello from " + fmt.Sprintf("%v", req["model"]))},
					FinishReason: aimodel.FinishReasonStop,
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
}

// newFailServer creates a server that always returns a 500 error.
func newFailServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": "internal server error",
				"type":    "server_error",
			},
		})
	}))
}

// newStreamServer creates a server that returns a valid SSE stream.
func newStreamServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		modelName := fmt.Sprintf("%v", req["model"])

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)

		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		chunk := aimodel.StreamChunk{
			ID:    "chunk-1",
			Model: modelName,
			Choices: []aimodel.StreamChunkChoice{
				{
					Index: 0,
					Delta: aimodel.Message{
						Role:    aimodel.RoleAssistant,
						Content: aimodel.NewTextContent("hello stream"),
					},
				},
			},
		}

		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
}

func newClientForServer(t *testing.T, server *httptest.Server) aimodel.ChatClient {
	t.Helper()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("test-key"),
		aimodel.WithBaseURL(server.URL),
	)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	return c
}

func testRequest() *aimodel.ChatRequest {
	return &aimodel.ChatRequest{
		Model: "placeholder",
		Messages: []aimodel.Message{
			{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("hi")},
		},
	}
}

func TestNewComposeClient_EmptyEntries(t *testing.T) {
	_, err := NewComposeClient(StrategyFailover, nil)
	if err == nil {
		t.Fatal("expected error for empty entries")
	}
}

func TestNewComposeClient_NilClient(t *testing.T) {
	_, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: nil},
	})
	if err == nil {
		t.Fatal("expected error for nil client")
	}
}

func TestNewComposeClient_EmptyName_UsesClientDefault(t *testing.T) {
	var receivedModel string

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		receivedModel = fmt.Sprintf("%v", req["model"])

		w.Header().Set("Content-Type", "application/json")

		resp := aimodel.ChatResponse{
			ID:    "test",
			Model: receivedModel,
			Choices: []aimodel.Choice{
				{
					Index:        0,
					Message:      aimodel.Message{Role: aimodel.RoleAssistant, Content: aimodel.NewTextContent("ok")},
					FinishReason: aimodel.FinishReasonStop,
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer s.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("test-key"),
		aimodel.WithBaseURL(s.URL),
		aimodel.WithDefaultModel("my-default-model"),
	)
	if err != nil {
		t.Fatal(err)
	}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "", Client: client},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Use empty model in request so the client default is applied.
	_, err = cc.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Messages: []aimodel.Message{
			{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("hi")},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if receivedModel != "my-default-model" {
		t.Fatalf("model = %s, want my-default-model", receivedModel)
	}
}

func TestNewComposeClient_WithRecoveryInterval(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	c, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s)},
	}, WithRecoveryInterval(30*time.Second))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if c.recoveryInterval != 30*time.Second {
		t.Fatalf("recoveryInterval = %v, want 30s", c.recoveryInterval)
	}
}

func TestFailover_FirstModelSucceeds(t *testing.T) {
	s0 := newTestServer(t)
	defer s0.Close()

	s1 := newTestServer(t)
	defer s1.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s0)},
		{Name: "m1", Client: newClientForServer(t, s1)},
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Model != "m0" {
		t.Fatalf("model = %s, want m0", resp.Model)
	}
}

func TestFailover_Fallback(t *testing.T) {
	sFail := newFailServer(t)
	defer sFail.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, sFail)},
		{Name: "m1", Client: newClientForServer(t, sOK)},
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %s, want m1", resp.Model)
	}
}

func TestFailover_AllFail(t *testing.T) {
	s0 := newFailServer(t)
	defer s0.Close()

	s1 := newFailServer(t)
	defer s1.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s0)},
		{Name: "m1", Client: newClientForServer(t, s1)},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = cc.ChatCompletion(context.Background(), testRequest())
	if err == nil {
		t.Fatal("expected error when all models fail")
	}

	var me *aimodel.MultiError
	if !errors.As(err, &me) {
		t.Fatalf("expected MultiError, got %T: %v", err, err)
	}

	if len(me.Errors) != 2 {
		t.Fatalf("expected 2 model errors, got %d", len(me.Errors))
	}
}

func TestFailover_StreamFallback(t *testing.T) {
	sFail := newFailServer(t)
	defer sFail.Close()

	sStream := newStreamServer(t)
	defer sStream.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, sFail)},
		{Name: "m1", Client: newClientForServer(t, sStream)},
	})
	if err != nil {
		t.Fatal(err)
	}

	stream, err := cc.ChatCompletionStream(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv()
	if err != nil {
		t.Fatalf("recv error: %v", err)
	}

	if chunk.Model != "m1" {
		t.Fatalf("stream model = %s, want m1", chunk.Model)
	}

	// Drain remaining chunks.
	for {
		_, err := stream.Recv()
		if err == io.EOF {
			break
		}

		if err != nil {
			t.Fatalf("unexpected stream error: %v", err)
		}
	}
}

func TestFailover_RecoveryProbe(t *testing.T) {
	var callCount atomic.Int64

	sProbe := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount.Add(1)

		w.Header().Set("Content-Type", "application/json")

		resp := aimodel.ChatResponse{
			ID:    "probe-id",
			Model: "m0",
			Choices: []aimodel.Choice{
				{
					Index:        0,
					Message:      aimodel.Message{Role: aimodel.RoleAssistant, Content: aimodel.NewTextContent("recovered")},
					FinishReason: aimodel.FinishReasonStop,
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer sProbe.Close()

	s1 := newTestServer(t)
	defer s1.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, sProbe)},
		{Name: "m1", Client: newClientForServer(t, s1)},
	}, WithRecoveryInterval(time.Second))
	if err != nil {
		t.Fatal(err)
	}

	// Mark primary as error.
	now := time.Now()
	cc.nowFunc = func() time.Time { return now }
	cc.health[0].markError(errors.New("initial failure"), now)

	// Request should use m1 (primary is errored).
	resp, err := cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %s, want m1", resp.Model)
	}

	// Advance time past recovery interval.
	cc.nowFunc = func() time.Time { return now.Add(2 * time.Second) }

	// Now primary should be probed first and succeed.
	resp, err = cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model != "m0" {
		t.Fatalf("model after probe = %s, want m0", resp.Model)
	}

	// Verify that the primary was actually called.
	if callCount.Load() == 0 {
		t.Fatal("expected primary to be called during probe")
	}
}

func TestRandom_RecoveryProbe(t *testing.T) {
	var probeCount atomic.Int64

	sProbe := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probeCount.Add(1)

		w.Header().Set("Content-Type", "application/json")

		resp := aimodel.ChatResponse{
			ID:    "probe-id",
			Model: "m0",
			Choices: []aimodel.Choice{
				{
					Index:        0,
					Message:      aimodel.Message{Role: aimodel.RoleAssistant, Content: aimodel.NewTextContent("recovered")},
					FinishReason: aimodel.FinishReasonStop,
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer sProbe.Close()

	s1 := newTestServer(t)
	defer s1.Close()

	cc, err := NewComposeClient(StrategyRandom, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, sProbe)},
		{Name: "m1", Client: newClientForServer(t, s1)},
	}, WithRecoveryInterval(time.Second))
	if err != nil {
		t.Fatal(err)
	}

	// Mark m0 as error.
	now := time.Now()
	cc.nowFunc = func() time.Time { return now }
	cc.health[0].markError(errors.New("fail"), now)

	// Advance time past recovery interval and probe should fire.
	cc.nowFunc = func() time.Time { return now.Add(2 * time.Second) }

	resp, err := cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	// Recovery probe is prepended, so m0 should be tried first.
	if resp.Model != "m0" {
		t.Fatalf("model = %s, want m0 (recovery probe)", resp.Model)
	}

	if probeCount.Load() == 0 {
		t.Fatal("expected m0 to be probed during random strategy")
	}
}

func TestModelOverride(t *testing.T) {
	var receivedModel string

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		receivedModel = fmt.Sprintf("%v", req["model"])

		w.Header().Set("Content-Type", "application/json")

		resp := aimodel.ChatResponse{
			ID:    "test",
			Model: receivedModel,
			Choices: []aimodel.Choice{
				{
					Index:        0,
					Message:      aimodel.Message{Role: aimodel.RoleAssistant, Content: aimodel.NewTextContent("ok")},
					FinishReason: aimodel.FinishReasonStop,
				},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "custom-model-v2", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	req := &aimodel.ChatRequest{
		Model: "original-model",
		Messages: []aimodel.Message{
			{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("hi")},
		},
	}

	_, err = cc.ChatCompletion(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}

	if receivedModel != "custom-model-v2" {
		t.Fatalf("sent model = %s, want custom-model-v2", receivedModel)
	}

	// Original request should be unchanged.
	if req.Model != "original-model" {
		t.Fatalf("original request model mutated to %s", req.Model)
	}
}

func TestAnthropicChatCompletion(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check that Anthropic headers are set.
		if r.Header.Get("x-api-key") == "" {
			http.Error(w, "missing api key", http.StatusUnauthorized)
			return
		}

		if !strings.Contains(r.URL.Path, "/v1/messages") {
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		resp := map[string]any{
			"id":    "msg-test",
			"type":  "message",
			"role":  "assistant",
			"model": "claude-test",
			"content": []map[string]any{
				{"type": "text", "text": "hello from anthropic"},
			},
			"stop_reason": "end_turn",
			"usage": map[string]any{
				"input_tokens":  10,
				"output_tokens": 5,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer s.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("test-key"),
		aimodel.WithBaseURL(s.URL),
	)
	if err != nil {
		t.Fatal(err)
	}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "claude-test", Client: client, Protocol: aimodel.ProtocolAnthropic},
	})
	if err != nil {
		t.Fatal(err)
	}

	// ChatCompletion with Protocol=Anthropic should route to AnthropicChatCompletion.
	resp, err := cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}

	if resp.Choices[0].Message.Content.Text() != "hello from anthropic" {
		t.Fatalf("got content = %s", resp.Choices[0].Message.Content.Text())
	}
}

func TestConcurrentRequests(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup

	errCh := make(chan error, 50)

	for range 50 {
		wg.Add(1)

		go func() {
			defer wg.Done()

			_, err := cc.ChatCompletion(context.Background(), testRequest())
			if err != nil {
				errCh <- err
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent request error: %v", err)
	}
}

func TestContextCancellation_DoesNotPoisonHealth(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Cancel context before making request.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = cc.ChatCompletion(ctx, testRequest())
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}

	// Model should remain active (not poisoned by the cancellation).
	if !cc.health[0].isActive() {
		t.Fatal("model health should remain active after context cancellation")
	}
}

func TestMultiError_UnwrapAll(t *testing.T) {
	s0 := newFailServer(t)
	defer s0.Close()

	s1 := newFailServer(t)
	defer s1.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s0)},
		{Name: "m1", Client: newClientForServer(t, s1)},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = cc.ChatCompletion(context.Background(), testRequest())

	var me *aimodel.MultiError
	if !errors.As(err, &me) {
		t.Fatalf("expected MultiError, got %T", err)
	}

	// errors.As should find APIError from any model's error in the chain.
	var apiErr *aimodel.APIError
	if !errors.As(err, &apiErr) {
		t.Fatal("expected errors.As to find APIError in multi-error chain")
	}
}

func TestNoActiveModels_Error(t *testing.T) {
	s0 := newFailServer(t)
	defer s0.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s0)},
	})
	if err != nil {
		t.Fatal(err)
	}

	// First call fails and marks model as error.
	_, _ = cc.ChatCompletion(context.Background(), testRequest())

	// Second call should return ErrNoActiveModels since no active models remain.
	_, err = cc.ChatCompletion(context.Background(), testRequest())
	if !errors.Is(err, aimodel.ErrNoActiveModels) {
		t.Fatalf("expected ErrNoActiveModels, got %v", err)
	}
}
