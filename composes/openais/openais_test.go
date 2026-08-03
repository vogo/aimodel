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

package openais

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

	"github.com/vogo/aimodel/composes"
	"github.com/vogo/aimodel/provider/openai"
)

// newTestServer creates an httptest server that echoes the requested model back
// in a valid Chat Completions response.
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
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:    "test-id",
			Model: model,
			Choices: []openai.ChatCompletionChoice{{
				Index:   0,
				Message: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("hello from " + model)},
			}},
		})
	}))
}

// newStatusServer returns a server that always replies with the given HTTP
// status and an OpenAI-shaped error body, plus a hit counter.
func newStatusServer(t *testing.T, status int) (*httptest.Server, *atomic.Int64) {
	t.Helper()

	var hits atomic.Int64

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"message": "boom", "type": "error"},
		})
	}))

	return s, &hits
}

// newFailServer creates a server that always returns a 500 error.
func newFailServer(t *testing.T) *httptest.Server {
	t.Helper()

	s, _ := newStatusServer(t, http.StatusInternalServerError)

	return s
}

// newStreamServer creates a server that returns a valid chat SSE stream.
func newStreamServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)

			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)

			return
		}

		chunk := openai.ChatCompletionChunk{
			ID:    "chunk-1",
			Model: fmt.Sprintf("%v", request["model"]),
			Choices: []openai.ChatCompletionChunkChoice{{
				Index: 0,
				Delta: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("hello stream")},
			}},
		}

		data, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()

		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
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

// recordingObserver collects attempt results safely.
type recordingObserver struct {
	mu      sync.Mutex
	results []composes.AttemptResult
}

func (r *recordingObserver) fn(res composes.AttemptResult) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.results = append(r.results, res)
}

func (r *recordingObserver) seen() []composes.AttemptResult {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]composes.AttemptResult(nil), r.results...)
}

func TestNewComposeClient_EmptyEntries(t *testing.T) {
	if _, err := NewComposeClient(composes.StrategyFailover, nil); err == nil {
		t.Fatal("expected an error for empty entries")
	}
}

func TestNewComposeClient_NilClient(t *testing.T) {
	_, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{{Name: "m0", Client: nil}})
	if err == nil {
		t.Fatal("expected an error for a nil client")
	}
}

// An entry without a name leaves the request's own model alone. There is no
// client-level default model to fall back to.
func TestEmptyEntryName_KeepsTheRequestModel(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
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

func TestFailover_FirstEndpointSucceeds(t *testing.T) {
	s0, s1 := newTestServer(t), newTestServer(t)
	defer s0.Close()
	defer s1.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
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
	sFail, sOK := newFailServer(t), newTestServer(t)
	defer sFail.Close()
	defer sOK.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
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

func TestFailover_AllFailAttributesEveryAlias(t *testing.T) {
	s0, s1 := newFailServer(t), newFailServer(t)
	defer s0.Close()
	defer s1.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s0)},
		{Name: "m1", Client: newClientForServer(t, s1)},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = cc.ChatCompletions(context.Background(), testRequest())

	var multi *composes.MultiError
	if !errors.As(err, &multi) {
		t.Fatalf("expected *composes.MultiError, got %T: %v", err, err)
	}

	if len(multi.Errors) != 2 {
		t.Fatalf("endpoint errors = %d, want 2", len(multi.Errors))
	}

	if multi.Errors[0].Alias != "m0" || multi.Errors[1].Alias != "m1" {
		t.Fatalf("aliases = %q, %q; want m0, m1", multi.Errors[0].Alias, multi.Errors[1].Alias)
	}

	// errors.As reaches the endpoint attribution and the provider's own error.
	var endpointErr *composes.EndpointError
	if !errors.As(err, &endpointErr) {
		t.Fatal("expected errors.As to find a composes.EndpointError in the chain")
	}

	var apiErr *openai.HTTPError
	if !errors.As(err, &apiErr) {
		t.Fatal("expected errors.As to find *openai.HTTPError in the chain")
	}
}

func TestFailover_StreamFallback(t *testing.T) {
	sFail, sStream := newFailServer(t), newStreamServer(t)
	defer sFail.Close()
	defer sStream.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
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
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("unexpected stream error: %v", err)
		}
	}
}

// Health recovery end to end: a 5xx errors the endpoint, the elapsed backoff
// prepends it as a probe, and a successful probe returns it to active.
func TestHealthRecovery_ProbeReturnsEndpointToActive(t *testing.T) {
	var hits atomic.Int64

	sFlaky := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if hits.Add(1) == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": "boom"}})

			return
		}

		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID: "id", Model: "m0",
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("recovered")},
			}},
		})
	}))
	defer sFlaky.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	// A one-nanosecond recovery interval makes the backoff elapsed by the time
	// the next call routes, without any clock injection.
	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "flaky", Client: newClientForServer(t, sFlaky)},
		{Name: "m1", Alias: "steady", Client: newClientForServer(t, sOK)},
	}, composes.WithRecoveryInterval(time.Nanosecond))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %s, want the failover m1", resp.Model)
	}

	if s := cc.Stats()[0]; s.Status != "error" || s.ErrorCount != 1 {
		t.Fatalf("stats after 5xx = %+v, want error/1", s)
	}

	// The next call probes the errored endpoint first, and it recovers.
	resp, err = cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model != "m0" {
		t.Fatalf("model after the probe = %s, want m0", resp.Model)
	}

	if s := cc.Stats()[0]; s.Status != "active" || s.ErrorCount != 0 || s.LastError != nil {
		t.Fatalf("stats after a successful probe = %+v, want active/0/nil", s)
	}
}

// A 429 cools the endpoint out of rotation without advancing the error backoff.
func TestCooling_429SkippedDuringWindow(t *testing.T) {
	s429, hits429 := newStatusServer(t, http.StatusTooManyRequests)
	defer s429.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "limited", Client: newClientForServer(t, s429)},
		{Name: "m1", Alias: "ok", Client: newClientForServer(t, sOK)},
	}, composes.WithCoolingInterval(time.Minute))
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %q, want m1 (failover from the cooled endpoint)", resp.Model)
	}

	if s := cc.Stats()[0]; s.Status != "cooling" || s.ErrorCount != 0 {
		t.Fatalf("stats after 429 = %+v, want cooling/0", s)
	}

	// Inside the window the cooled endpoint is not contacted again.
	if _, err := cc.ChatCompletions(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	if hits429.Load() != 1 {
		t.Fatalf("the cooled endpoint was hit %d times, want 1", hits429.Load())
	}
}

// A non-429 4xx is attributed and failed over, but leaves the endpoint healthy.
func TestRequestFailure_4xxStaysActive(t *testing.T) {
	s400, _ := newStatusServer(t, http.StatusBadRequest)
	defer s400.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "badreq", Client: newClientForServer(t, s400)},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = cc.ChatCompletions(context.Background(), testRequest())

	var endpointErr *composes.EndpointError
	if !errors.As(err, &endpointErr) || endpointErr.Alias != "badreq" {
		t.Fatalf("expected an attributed EndpointError for badreq, got %v", err)
	}

	if s := cc.Stats()[0]; s.Status != "active" || s.ErrorCount != 0 || s.LastError != nil {
		t.Fatalf("stats after 4xx = %+v, want active/0/nil", s)
	}
}

func TestNoActiveModels_Error(t *testing.T) {
	s := newFailServer(t)
	defer s.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	// The first call fails and errors the only endpoint.
	_, _ = cc.ChatCompletions(context.Background(), testRequest())

	// The second finds no candidate at all.
	if _, err = cc.ChatCompletions(context.Background(), testRequest()); !errors.Is(err, composes.ErrNoActiveModels) {
		t.Fatalf("expected composes.ErrNoActiveModels, got %v", err)
	}
}

// Every strategy routes through the same core loop, so each one fails over.
func TestEveryStrategy_FailsOverToTheHealthyEndpoint(t *testing.T) {
	strategies := []composes.Strategy{
		composes.StrategyFailover,
		composes.StrategyRandom,
		composes.StrategyWeight,
		composes.StrategySticky,
		composes.StrategyCost,
		composes.StrategyLatency,
	}

	for _, strategy := range strategies {
		t.Run(string(strategy), func(t *testing.T) {
			sFail, sOK := newFailServer(t), newTestServer(t)
			defer sFail.Close()
			defer sOK.Close()

			cheap, slow := 10*time.Millisecond, 50*time.Millisecond

			cc, err := NewComposeClient(strategy, []ModelEntry{
				{
					Name: "m0", Alias: "broken", Weight: 9, Client: newClientForServer(t, sFail),
					Cost: &composes.EndpointCost{InputPrice: 1, OutputPrice: 1}, Latency: &cheap,
				},
				{
					Name: "m1", Alias: "healthy", Weight: 1, Client: newClientForServer(t, sOK),
					Cost: &composes.EndpointCost{InputPrice: 5, OutputPrice: 5}, Latency: &slow,
				},
			})
			if err != nil {
				t.Fatal(err)
			}

			ctx := composes.WithSessionID(context.Background(), "session-1")

			resp, err := cc.ChatCompletions(ctx, testRequest())
			if err != nil {
				t.Fatalf("%s: expected failover to succeed, got %v", strategy, err)
			}

			if resp.Model != "m1" {
				t.Fatalf("%s: model = %q, want m1", strategy, resp.Model)
			}
		})
	}
}

// Sticky routing pins a session to one endpoint across calls.
func TestSticky_SameSessionKeepsOneEndpoint(t *testing.T) {
	obs := &recordingObserver{}

	entries := make([]ModelEntry, 3)

	for i, alias := range []string{"a", "b", "c"} {
		s := newTestServer(t)
		t.Cleanup(s.Close)

		entries[i] = ModelEntry{Name: "m" + alias, Alias: alias, Client: newClientForServer(t, s)}
	}

	cc, err := NewComposeClient(composes.StrategySticky, entries, composes.WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	ctx := composes.WithSessionID(context.Background(), "session-xyz")

	first, err := cc.ChatCompletions(ctx, testRequest())
	if err != nil {
		t.Fatal(err)
	}

	for range 10 {
		resp, err := cc.ChatCompletions(ctx, testRequest())
		if err != nil {
			t.Fatal(err)
		}

		if resp.Model != first.Model {
			t.Fatalf("sticky routing drifted: %q then %q", first.Model, resp.Model)
		}
	}

	// Every attempt was attributed to the same alias, and all succeeded. The
	// entries are named "m<alias>", so the alias is the model name's tail.
	wantAlias := first.Model[1:]

	for _, res := range obs.seen() {
		if !res.Success || res.Alias != wantAlias {
			t.Fatalf("unexpected attempt %+v, want a success on alias %q", res, wantAlias)
		}
	}
}

func TestModelOverride_LeavesTheCallerRequestUntouched(t *testing.T) {
	var received atomic.Value

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		_ = json.NewDecoder(r.Body).Decode(&request)

		model := fmt.Sprintf("%v", request["model"])
		received.Store(model)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{ID: "test", Model: model})
	}))
	defer s.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "custom-model-v2", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	request := testRequest()
	request.Model = "original-model"

	if _, err = cc.ChatCompletions(context.Background(), request); err != nil {
		t.Fatal(err)
	}

	if got := received.Load(); got != "custom-model-v2" {
		t.Fatalf("sent model = %v, want custom-model-v2", got)
	}

	if request.Model != "original-model" {
		t.Fatalf("the caller's request was mutated to %s", request.Model)
	}
}

// A ComposeClient is itself a backend, so pools nest (for example a fast pool
// that falls back to a slower one).
func TestNestedComposeClients(t *testing.T) {
	sFail, sOK := newFailServer(t), newTestServer(t)
	defer sFail.Close()
	defer sOK.Close()

	inner, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "inner", Client: newClientForServer(t, sFail)},
	})
	if err != nil {
		t.Fatal(err)
	}

	outer, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
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

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup

	errCh := make(chan error, 50)

	for range 50 {
		wg.Go(func() {
			if _, err := cc.ChatCompletions(context.Background(), testRequest()); err != nil {
				errCh <- err
			}

			_ = cc.Stats()
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

	obs := &recordingObserver{}

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "m0", Client: newClientForServer(t, s)},
	}, composes.WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err = cc.ChatCompletions(ctx, testRequest()); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	if s := cc.Stats()[0]; s.Status != "active" {
		t.Fatalf("status after cancellation = %q, want active", s.Status)
	}
}

func TestAttemptObserver_AttributesFailoverAndStreams(t *testing.T) {
	sFail, sStream := newFailServer(t), newStreamServer(t)
	defer sFail.Close()
	defer sStream.Close()

	obs := &recordingObserver{}

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "bad", Client: newClientForServer(t, sFail)},
		{Name: "m1", Alias: "good", Client: newClientForServer(t, sStream)},
	}, composes.WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	stream, err := cc.ChatCompletionsStream(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	_ = stream.Close()

	results := obs.seen()
	if len(results) != 2 {
		t.Fatalf("observations = %d, want 2", len(results))
	}

	if results[0].Alias != "bad" || results[0].Success || !results[0].Stream {
		t.Fatalf("first observation = %+v, want bad/failure/stream", results[0])
	}

	if results[1].Alias != "good" || !results[1].Success || !results[1].Stream {
		t.Fatalf("second observation = %+v, want good/success/stream", results[1])
	}
}

// Both interaction forms share one pool, so the health a chat failure records
// is the health Responses routing sees.
func TestStats_SharedAcrossInteractionForms(t *testing.T) {
	sFail, sOK := newFailServer(t), newTestServer(t)
	defer sFail.Close()
	defer sOK.Close()

	cc, err := NewComposeClient(composes.StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "broken", Client: newClientForServer(t, sFail)},
		{Name: "m1", Alias: "healthy", Client: newClientForServer(t, sOK)},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := cc.ChatCompletions(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	stats := cc.Stats()
	if len(stats) != 2 {
		t.Fatalf("stats len = %d, want 2", len(stats))
	}

	byAlias := map[string]composes.EndpointStat{}
	for _, s := range stats {
		byAlias[s.Alias] = s
	}

	if byAlias["broken"].Status != "error" {
		t.Fatalf("broken status = %q, want error", byAlias["broken"].Status)
	}

	if byAlias["healthy"].Status != "active" || byAlias["healthy"].LastError != nil {
		t.Fatalf("healthy stats = %+v, want active with zeroed error fields", byAlias["healthy"])
	}
}
