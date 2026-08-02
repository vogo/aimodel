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

// newTestServer creates an httptest server that echoes the requested model in
// an OpenAI-shaped completion, so a test can tell which backend served it.
func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)

			return
		}

		model := fmt.Sprintf("%v", request["model"])

		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w,
			`{"id":"test-id","object":"chat.completion","model":%q,"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hello from %s"}}]}`,
			model, model)
	}))
}

// newFailServer creates a server that always returns a 500 error.
func newFailServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"error":{"message":"internal server error","type":"server_error"}}`)
	}))
}

// newStreamServer creates a server that returns a valid SSE stream naming the
// model it served.
func newStreamServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)

			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w,
			"data: {\"id\":\"chunk-1\",\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello stream\"}}]}\n\ndata: [DONE]\n\n",
			fmt.Sprintf("%v", request["model"]))
	}))
}

func newClientForServer(t *testing.T, server *httptest.Server) *openai.Client {
	t.Helper()

	return openai.NewClient("test-key", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))
}

func testRequest() *openai.ChatCompletionRequest {
	return &openai.ChatCompletionRequest{
		Model:    "placeholder",
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: openai.NewTextContent("hi")}},
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

	response, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if response.Model != "m0" {
		t.Fatalf("model = %s, want m0", response.Model)
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

	response, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if response.Model != "m1" {
		t.Fatalf("model = %s, want m1", response.Model)
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

	var multi *MultiError
	if !errors.As(err, &multi) {
		t.Fatalf("expected MultiError, got %T: %v", err, err)
	}

	if len(multi.Errors) != 2 {
		t.Fatalf("expected 2 model errors, got %d", len(multi.Errors))
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

	for {
		_, err = stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("unexpected stream error: %v", err)
		}
	}

	// The accumulated result stays available through the dispatching client.
	if got := stream.Response().Choices[0].Message.Content.Text(); got != "hello stream" {
		t.Errorf("accumulated content = %q", got)
	}
}

func TestFailover_RecoveryProbe(t *testing.T) {
	var callCount atomic.Int64

	sProbe := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount.Add(1)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"probe-id","model":"m0","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"recovered"}}]}`)
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
	response, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if response.Model != "m1" {
		t.Fatalf("model = %s, want m1", response.Model)
	}

	// Advance time past recovery interval.
	cc.nowFunc = func() time.Time { return now.Add(2 * time.Second) }

	// Now primary should be probed first and succeed.
	response, err = cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if response.Model != "m0" {
		t.Fatalf("model after probe = %s, want m0", response.Model)
	}

	if callCount.Load() == 0 {
		t.Fatal("expected primary to be called during probe")
	}
}

func TestRandom_RecoveryProbe(t *testing.T) {
	var probeCount atomic.Int64

	sProbe := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		probeCount.Add(1)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"probe-id","model":"m0","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"recovered"}}]}`)
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

	response, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	// Recovery probe is prepended, so m0 should be tried first.
	if response.Model != "m0" {
		t.Fatalf("model = %s, want m0 (recovery probe)", response.Model)
	}

	if probeCount.Load() == 0 {
		t.Fatal("expected m0 to be probed during random strategy")
	}
}

func TestModelOverride(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "custom-model-v2", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	request := testRequest()
	request.Model = "original-model"

	response, err := cc.ChatCompletions(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}

	if response.Model != "custom-model-v2" {
		t.Fatalf("sent model = %s, want custom-model-v2", response.Model)
	}

	if request.Model != "original-model" {
		t.Fatalf("original request model mutated to %s", request.Model)
	}
}

// TestNestedComposeClients verifies a ComposeClient is itself a backend, so
// pools can be layered (for example a fast pool that falls back to a slow one).
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

// TestMultiErrorUnwrapsToTheBackendError verifies the aggregate keeps every
// backend's error reachable, including its status code — matched through a
// locally declared interface, so this package imports no provider error type.
func TestMultiErrorUnwrapsToTheBackendError(t *testing.T) {
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

	var multi *MultiError
	if !errors.As(err, &multi) {
		t.Fatalf("expected MultiError, got %T", err)
	}

	var modelErr *ModelError
	if !errors.As(err, &modelErr) {
		t.Fatal("expected errors.As to find a ModelError in the chain")
	}

	type statusCoder interface{ StatusCode() int }

	var status statusCoder
	if !errors.As(err, &status) {
		t.Fatal("expected errors.As to reach a provider error carrying a status code")
	}

	if status.StatusCode() != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", status.StatusCode())
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
