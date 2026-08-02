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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vogo/aimodel/provider/openai"
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

		resp := openai.ChatCompletionResponse{
			ID:    "test-id",
			Model: fmt.Sprintf("%v", req["model"]),
			Choices: []openai.ChatCompletionChoice{
				{
					Index:   0,
					Message: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("hello from " + fmt.Sprintf("%v", req["model"]))},
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

		chunk := openai.ChatCompletionChunk{
			ID:    "chunk-1",
			Model: modelName,
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionMessage{
						Role:    "assistant",
						Content: openai.NewTextContent("hello stream"),
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

func newClientForServer(t *testing.T, server *httptest.Server) *openai.Client {
	t.Helper()

	return openai.NewClient("test-key", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
}

func testRequest() *openai.ChatCompletionRequest {
	return &openai.ChatCompletionRequest{
		Model: "placeholder",
		Messages: []openai.ChatCompletionMessage{
			{Role: "user", Content: openai.NewTextContent("hi")},
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

// TestEmptyEntryName_KeepsTheRequestModel verifies an entry without a name
// leaves the request's own model alone. There is no client-level default model
// to fall back to.
func TestEmptyEntryName_KeepsTheRequestModel(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	response, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if response.Model != "placeholder" {
		t.Fatalf("model = %s, want the request's own model", response.Model)
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

	resp, err := cc.ChatCompletions(context.Background(), testRequest())
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

	resp, err := cc.ChatCompletions(context.Background(), testRequest())
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

	_, err = cc.ChatCompletions(context.Background(), testRequest())
	if err == nil {
		t.Fatal("expected error when all models fail")
	}

	var me *MultiError
	if !errors.As(err, &me) {
		t.Fatalf("expected composes.MultiError, got %T: %v", err, err)
	}

	if len(me.Errors) != 2 {
		t.Fatalf("expected 2 endpoint errors, got %d", len(me.Errors))
	}

	// Each failure is attributed to a distinct alias.
	if me.Errors[0].Alias != "m0" || me.Errors[1].Alias != "m1" {
		t.Fatalf("aliases = %q, %q; want m0, m1", me.Errors[0].Alias, me.Errors[1].Alias)
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

	stream, err := cc.ChatCompletionsStream(context.Background(), testRequest())
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

		resp := openai.ChatCompletionResponse{
			ID:    "probe-id",
			Model: "m0",
			Choices: []openai.ChatCompletionChoice{
				{
					Index:   0,
					Message: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("recovered")},
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
	resp, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %s, want m1", resp.Model)
	}

	// Advance time past recovery interval.
	cc.nowFunc = func() time.Time { return now.Add(2 * time.Second) }

	// Now primary should be probed first and succeed.
	resp, err = cc.ChatCompletions(context.Background(), testRequest())
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

		resp := openai.ChatCompletionResponse{
			ID:    "probe-id",
			Model: "m0",
			Choices: []openai.ChatCompletionChoice{
				{
					Index:   0,
					Message: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("recovered")},
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

	resp, err := cc.ChatCompletions(context.Background(), testRequest())
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

		resp := openai.ChatCompletionResponse{
			ID:    "test",
			Model: receivedModel,
			Choices: []openai.ChatCompletionChoice{
				{
					Index:   0,
					Message: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("ok")},
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

	req := &openai.ChatCompletionRequest{
		Model: "original-model",
		Messages: []openai.ChatCompletionMessage{
			{Role: "user", Content: openai.NewTextContent("hi")},
		},
	}

	_, err = cc.ChatCompletions(context.Background(), req)
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

// TestNestedComposeClients verifies a ComposeClient is itself a backend, so
// pools can be layered (for example a fast pool that falls back to a slower
// one).
func TestNestedComposeClients(t *testing.T) {
	sFail := newFailServer(t)
	defer sFail.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	inner, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "inner", Client: newClientForServer(t, sFail)},
	})
	if err != nil {
		t.Fatal(err)
	}

	outer, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "pool", Client: inner},
		{Name: "m1", Client: newClientForServer(t, sOK)},
	})
	if err != nil {
		t.Fatal(err)
	}

	response, err := outer.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if response.Model != "m1" {
		t.Fatalf("model = %s, want the outer fallback m1", response.Model)
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
		wg.Go(func() {
			_, err := cc.ChatCompletions(context.Background(), testRequest())
			if err != nil {
				errCh <- err
			}
		})
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

	_, err = cc.ChatCompletions(ctx, testRequest())
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

	_, err = cc.ChatCompletions(context.Background(), testRequest())

	var me *MultiError
	if !errors.As(err, &me) {
		t.Fatalf("expected composes.MultiError, got %T", err)
	}

	// errors.As reaches a per-endpoint error and its alias.
	var ee *EndpointError
	if !errors.As(err, &ee) {
		t.Fatal("expected errors.As to find EndpointError in multi-error chain")
	}

	if ee.Alias != "m0" && ee.Alias != "m1" {
		t.Fatalf("endpoint alias = %q, want m0 or m1", ee.Alias)
	}

	// errors.As should find APIError from any endpoint's error in the chain.
	var apiErr *openai.HTTPError
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
	_, _ = cc.ChatCompletions(context.Background(), testRequest())

	// Second call should return ErrNoActiveModels since no active models remain.
	_, err = cc.ChatCompletions(context.Background(), testRequest())
	if !errors.Is(err, ErrNoActiveModels) {
		t.Fatalf("expected ErrNoActiveModels, got %v", err)
	}
}

// NewComposeClient must own its entries: deriving aliases may not write back
// into the caller's slice, and later mutations by the caller must not reach the
// client's routing table.
func TestNewComposeClient_DoesNotMutateCallerEntries(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	entries := []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s)},
		{Name: "m1", Client: newClientForServer(t, s)},
	}

	cc, err := NewComposeClient(StrategyFailover, entries)
	if err != nil {
		t.Fatal(err)
	}

	if entries[0].Alias != "" || entries[1].Alias != "" {
		t.Fatalf("caller entries were written back: %q, %q", entries[0].Alias, entries[1].Alias)
	}

	if cc.entries[0].Alias != "m0" || cc.entries[1].Alias != "m1" {
		t.Fatalf("client aliases = %q, %q, want m0, m1", cc.entries[0].Alias, cc.entries[1].Alias)
	}

	// The client does not share the caller's backing array.
	entries[0].Name = "mutated"

	if cc.entries[0].Name != "m0" {
		t.Fatalf("client entry follows caller mutation: %q", cc.entries[0].Name)
	}
}
