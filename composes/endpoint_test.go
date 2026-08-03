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
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/vogo/aimodel/provider/openai"
)

func specForServer(s *httptest.Server, alias, model string) EndpointSpec {
	return EndpointSpec{
		BaseURL: s.URL,
		APIKey:  "key-" + alias,
		Model:   model,
		Alias:   alias,
	}
}

func TestNewFromEndpoints_Success(t *testing.T) {
	s0 := newTestServer(t)
	defer s0.Close()

	s1 := newTestServer(t)
	defer s1.Close()

	cc, err := NewFromEndpoints(StrategyWeight, []EndpointSpec{
		specForServer(s0, "primary", "gpt-4o"),
		{BaseURL: s1.URL, APIKey: "k", Model: "gpt-4o", Alias: "canary", Weight: 1},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(cc.entries) != 2 {
		t.Fatalf("entries = %d, want 2", len(cc.entries))
	}

	if cc.entries[0].Alias != "primary" || cc.entries[1].Alias != "canary" {
		t.Fatalf("aliases = %q,%q; want primary,canary", cc.entries[0].Alias, cc.entries[1].Alias)
	}

	// Same provider + model, distinct endpoints — a request must succeed.
	resp, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	if resp.Model != "gpt-4o" {
		t.Fatalf("model = %q, want gpt-4o", resp.Model)
	}
}

func TestNewFromEndpoints_Empty(t *testing.T) {
	if _, err := NewFromEndpoints(StrategyFailover, nil); err == nil {
		t.Fatal("expected error for empty specs")
	}
}

func TestNewFromEndpoints_MissingAlias(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	_, err := NewFromEndpoints(StrategyFailover, []EndpointSpec{
		{BaseURL: s.URL, APIKey: "k", Model: "m"},
	})
	if err == nil || !strings.Contains(err.Error(), "alias is required") {
		t.Fatalf("expected missing-alias error, got %v", err)
	}
}

func TestNewFromEndpoints_DuplicateAlias(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	_, err := NewFromEndpoints(StrategyFailover, []EndpointSpec{
		specForServer(s, "dup", "m"),
		specForServer(s, "dup", "m"),
	})
	if err == nil || !strings.Contains(err.Error(), "duplicate alias") {
		t.Fatalf("expected duplicate-alias error, got %v", err)
	}
}

// The native client performs no construction-time validation, so an empty
// credential does not fail NewFromEndpoints — it surfaces at request time.
// This records that behavior; there is also no global env fallback, so the
// endpoint uses exactly the key it was given.
func TestNewFromEndpoints_EmptyAPIKeyConstructs(t *testing.T) {
	cc, err := NewFromEndpoints(StrategyFailover, []EndpointSpec{
		{BaseURL: "http://localhost:1", APIKey: "", Model: "m", Alias: "nokey"},
	})
	if err != nil {
		t.Fatalf("native construction does not validate credentials: %v", err)
	}

	if len(cc.entries) != 1 || cc.entries[0].Alias != "nokey" {
		t.Fatalf("endpoint not built: %+v", cc.entries)
	}
}

func TestNewFromEndpoints_CredentialReachesEndpoint(t *testing.T) {
	var gotAuth string

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(openai.ChatCompletionResponse{
			ID:      "id",
			Model:   "m",
			Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Role: "assistant", Content: openai.NewTextContent("ok")}}},
		})
	}))
	defer s.Close()

	cc, err := NewFromEndpoints(StrategyFailover, []EndpointSpec{
		{BaseURL: s.URL, APIKey: "secret-key", Model: "m", Alias: "a"},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := cc.ChatCompletions(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(gotAuth, "secret-key") {
		t.Fatalf("endpoint did not receive its explicit key, auth = %q", gotAuth)
	}
}

func TestEndpointError_Attribution(t *testing.T) {
	s0 := newFailServer(t)
	defer s0.Close()

	s1 := newFailServer(t)
	defer s1.Close()

	cc, err := NewFromEndpoints(StrategyFailover, []EndpointSpec{
		specForServer(s0, "ep-a", "m"),
		specForServer(s1, "ep-b", "m"),
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = cc.ChatCompletions(context.Background(), testRequest())

	var me *MultiError
	if !errors.As(err, &me) {
		t.Fatalf("expected composes.MultiError, got %T", err)
	}

	if len(me.Errors) != 2 {
		t.Fatalf("endpoint errors = %d, want 2", len(me.Errors))
	}

	if me.Errors[0].Alias != "ep-a" || me.Errors[1].Alias != "ep-b" {
		t.Fatalf("aliases = %q,%q; want ep-a,ep-b", me.Errors[0].Alias, me.Errors[1].Alias)
	}

	// The underlying APIError is reachable through the endpoint error.
	var apiErr *openai.HTTPError
	if !errors.As(me.Errors[0], &apiErr) {
		t.Fatal("expected to unwrap APIError from EndpointError")
	}
}

// recordingObserver collects attempt results safely for sequential tests.
type recordingObserver struct {
	mu      sync.Mutex
	results []AttemptResult
}

func (r *recordingObserver) fn(res AttemptResult) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.results = append(r.results, res)
}

func TestAttemptObserver_NonStreamingSuccess(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	obs := &recordingObserver{}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "a0", Client: newClientForServer(t, s)},
	}, WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := cc.ChatCompletions(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	if len(obs.results) != 1 {
		t.Fatalf("observations = %d, want 1", len(obs.results))
	}

	if obs.results[0].Alias != "a0" || !obs.results[0].Success || obs.results[0].Stream {
		t.Fatalf("unexpected observation: %+v", obs.results[0])
	}
}

func TestAttemptObserver_FailoverAttributesBoth(t *testing.T) {
	sFail := newFailServer(t)
	defer sFail.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	obs := &recordingObserver{}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "bad", Client: newClientForServer(t, sFail)},
		{Name: "m1", Alias: "good", Client: newClientForServer(t, sOK)},
	}, WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := cc.ChatCompletions(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	if len(obs.results) != 2 {
		t.Fatalf("observations = %d, want 2", len(obs.results))
	}

	if obs.results[0].Alias != "bad" || obs.results[0].Success {
		t.Fatalf("first observation = %+v, want bad/failure", obs.results[0])
	}

	if obs.results[1].Alias != "good" || !obs.results[1].Success {
		t.Fatalf("second observation = %+v, want good/success", obs.results[1])
	}
}

func TestAttemptObserver_StreamEstablished(t *testing.T) {
	s := newStreamServer(t)
	defer s.Close()

	obs := &recordingObserver{}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "streamer", Client: newClientForServer(t, s)},
	}, WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	stream, err := cc.ChatCompletionsStream(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	_ = stream.Close()

	if len(obs.results) != 1 {
		t.Fatalf("observations = %d, want 1", len(obs.results))
	}

	if obs.results[0].Alias != "streamer" || !obs.results[0].Success || !obs.results[0].Stream {
		t.Fatalf("unexpected observation: %+v", obs.results[0])
	}
}

func TestAttemptObserver_StreamEstablishFail(t *testing.T) {
	sFail := newFailServer(t)
	defer sFail.Close()

	sStream := newStreamServer(t)
	defer sStream.Close()

	obs := &recordingObserver{}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "bad", Client: newClientForServer(t, sFail)},
		{Name: "m1", Alias: "good", Client: newClientForServer(t, sStream)},
	}, WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	stream, err := cc.ChatCompletionsStream(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	_ = stream.Close()

	// Establishment failure on "bad" is reported; "good" establishes.
	if len(obs.results) != 2 {
		t.Fatalf("observations = %d, want 2", len(obs.results))
	}

	if obs.results[0].Alias != "bad" || obs.results[0].Success || !obs.results[0].Stream {
		t.Fatalf("first observation = %+v, want bad/failure/stream", obs.results[0])
	}

	if obs.results[1].Alias != "good" || !obs.results[1].Success || !obs.results[1].Stream {
		t.Fatalf("second observation = %+v, want good/success/stream", obs.results[1])
	}
}

func TestWeightFailover_Combo(t *testing.T) {
	sFail := newFailServer(t)
	defer sFail.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	// Heavy weight on the failing endpoint: weight affects first-choice
	// probability, but failure still fails over to the healthy endpoint.
	cc, err := NewComposeClient(StrategyWeight, []ModelEntry{
		{Name: "m0", Alias: "heavy", Weight: 9, Client: newClientForServer(t, sFail)},
		{Name: "m1", Alias: "light", Weight: 1, Client: newClientForServer(t, sOK)},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Seed a deterministic rng so the heavy endpoint is tried first.
	cc.mu.Lock()
	cc.rng = newRand(1)
	cc.mu.Unlock()

	resp, err := cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("expected failover success, got %v", err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %q, want m1 after failover", resp.Model)
	}

	// The heavy endpoint is now errored; subsequent calls go straight to m1.
	resp, err = cc.ChatCompletions(context.Background(), testRequest())
	if err != nil {
		t.Fatal(err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %q, want m1", resp.Model)
	}
}

func TestManualEntry_BackwardCompatible(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	// Hand-built clients with no explicit alias keep working; aliases derive.
	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	if cc.entries[0].Alias != "m0" {
		t.Fatalf("derived alias = %q, want m0", cc.entries[0].Alias)
	}

	if _, err := cc.ChatCompletions(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}
}

func TestResolveAliases_SameModelUnique(t *testing.T) {
	entries := []ModelEntry{
		{Name: "gpt-4o"},
		{Name: "gpt-4o"}, // same model: canary case, must still be unique
		{Name: ""},
	}
	if err := resolveAliases(entries); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	seen := map[string]bool{}
	for i, e := range entries {
		if e.Alias == "" {
			t.Fatalf("entry %d has empty alias", i)
		}

		if seen[e.Alias] {
			t.Fatalf("duplicate derived alias %q", e.Alias)
		}

		seen[e.Alias] = true
	}
}

func TestNewComposeClient_DuplicateExplicitAlias(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	_, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "x", Client: newClientForServer(t, s)},
		{Name: "m1", Alias: "x", Client: newClientForServer(t, s)},
	})
	if err == nil || !strings.Contains(err.Error(), "duplicate alias") {
		t.Fatalf("expected duplicate-alias error, got %v", err)
	}
}

func TestContextCancellation_ObserverStillAttributes(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	obs := &recordingObserver{}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "slow", Client: newClientForServer(t, s)},
	}, WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before dispatch

	_, err = cc.ChatCompletions(ctx, testRequest())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	// Health must not be poisoned by cancellation.
	if !cc.health[0].isActive() {
		t.Fatal("health should remain active after cancellation")
	}
}
