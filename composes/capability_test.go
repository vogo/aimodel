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
	"sync/atomic"
	"testing"
	"time"

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/ais"
)

func toolsRequest() *ais.ChatRequest {
	return &ais.ChatRequest{
		Model: "placeholder",
		Messages: []ais.Message{
			{Role: ais.RoleUser, Content: ais.NewTextContent("what is the weather?")},
		},
		Tools: []ais.Tool{
			{Type: "function", Function: ais.FunctionDefinition{Name: "get_weather"}},
		},
	}
}

func visionRequest() *ais.ChatRequest {
	return &ais.ChatRequest{
		Model: "placeholder",
		Messages: []ais.Message{
			{Role: ais.RoleUser, Content: ais.NewPartsContent(
				ais.ContentPart{Type: "text", Text: "describe"},
				ais.ContentPart{Type: "image_url", ImageURL: &ais.ImageURL{URL: "https://x/y.png"}},
			)},
		},
	}
}

func selectReq(c *ComposeClient, req *ais.ChatRequest) []int {
	return c.selectModels(context.Background(), req, c.capableIndices(req))
}

// newToolsCapturingServer records whether the incoming request carried tools.
func newToolsCapturingServer(t *testing.T, gotTools *atomic.Bool) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)

		if tools, ok := req["tools"].([]any); ok && len(tools) > 0 {
			gotTools.Store(true)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ais.ChatResponse{
			ID:      "id",
			Model:   "m",
			Choices: []ais.Choice{{Message: ais.Message{Role: ais.RoleAssistant, Content: ais.NewTextContent("ok")}}},
		})
	}))
}

func TestCapability_ToolsFilterOnlyCapableEndpoints(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "no-tools", Capability: &Capability{Tools: false}},
		{Name: "m1", Alias: "with-tools", Capability: &Capability{Tools: true}},
	}
	c := newTestComposeClient(StrategyFailover, entries)

	got := selectReq(c, toolsRequest())
	assertIntSlice(t, got, []int{1})

	// A plain request keeps both eligible.
	got = selectReq(c, testRequest())
	assertIntSlice(t, got, []int{0, 1})
}

func TestCapability_AllIncapableErrorBeforeNetwork(t *testing.T) {
	var hits atomic.Int64

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
	}))
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "a", Capability: &Capability{Tools: false}, Client: newClientForServer(t, s)},
		{Name: "m1", Alias: "b", Capability: &Capability{Tools: false}, Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = cc.ChatCompletion(context.Background(), toolsRequest())

	if !errors.Is(err, ErrCapabilityNotSatisfied) {
		t.Fatalf("expected ErrCapabilityNotSatisfied, got %v", err)
	}

	var capErr *CapabilityError
	if !errors.As(err, &capErr) {
		t.Fatalf("expected CapabilityError, got %T", err)
	}

	if len(capErr.Required) != 1 || capErr.Required[0] != "tools" {
		t.Fatalf("required = %v, want [tools]", capErr.Required)
	}

	// Zero network calls: the failure happens before any request is sent.
	if hits.Load() != 0 {
		t.Fatalf("expected zero network calls, got %d", hits.Load())
	}
}

func TestCapability_NoDowngradeToolsPreserved(t *testing.T) {
	var gotTools atomic.Bool

	sTools := newToolsCapturingServer(t, &gotTools)
	defer sTools.Close()

	sPlain := newTestServer(t)
	defer sPlain.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "plain", Capability: &Capability{Tools: false}, Client: newClientForServer(t, sPlain)},
		{Name: "m1", Alias: "tools", Capability: &Capability{Tools: true}, Client: newClientForServer(t, sTools)},
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := cc.ChatCompletion(context.Background(), toolsRequest()); err != nil {
		t.Fatal(err)
	}

	// The capable endpoint received the tools untouched — no stripping/downgrade.
	if !gotTools.Load() {
		t.Fatal("tools were stripped before reaching the capable endpoint")
	}
}

func TestCapability_VisionFilter(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "text-only", Capability: &Capability{Vision: false}},
		{Name: "m1", Alias: "vision", Capability: &Capability{Vision: true}},
	}
	c := newTestComposeClient(StrategyFailover, entries)

	got := selectReq(c, visionRequest())
	assertIntSlice(t, got, []int{1})
}

func TestCapability_ToolChoiceNoneDoesNotRequireTools(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "no-tools", Capability: &Capability{Tools: false}},
	}
	c := newTestComposeClient(StrategyFailover, entries)

	req := testRequest()
	req.ToolChoice = "none"

	got := selectReq(c, req)
	assertIntSlice(t, got, []int{0})
}

// fakeCompleter is a ChatCompleter that also declares a Capability, exercising
// the CapabilityProvider fallback for clients that expose their own capability.
type fakeCompleter struct {
	cap Capability
}

func (f *fakeCompleter) ChatCompletion(context.Context, *ais.ChatRequest) (*ais.ChatResponse, error) {
	return &ais.ChatResponse{}, nil
}

func (f *fakeCompleter) ChatCompletionStream(context.Context, *ais.ChatRequest) (*aimodel.Stream, error) {
	return nil, nil
}

func (f *fakeCompleter) ComposeCapability() Capability { return f.cap }

func TestCapability_ProviderDeclaration(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "declared", Client: &fakeCompleter{cap: Capability{Tools: true}}},
		{Name: "m1", Alias: "silent", Client: &fakeCompleter{cap: Capability{}}},
	}
	c := newTestComposeClient(StrategyFailover, entries)

	got := selectReq(c, toolsRequest())
	assertIntSlice(t, got, []int{0})
}

func TestCostStrategy_DeterministicOrder(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "pricey", Cost: &EndpointCost{InputPrice: 5, OutputPrice: 5}},
		{Name: "m1", Alias: "cheap", Cost: &EndpointCost{InputPrice: 1, OutputPrice: 1}},
		{Name: "m2", Alias: "mid", Cost: &EndpointCost{InputPrice: 3, OutputPrice: 3}},
	}
	c := newTestComposeClient(StrategyCost, entries)

	// Ascending cost: cheap(2) < mid(6) < pricey(10) → indices 1, 2, 0.
	got := selectReq(c, testRequest())
	assertIntSlice(t, got, []int{1, 2, 0})

	// Deterministic across calls.
	for range 20 {
		assertIntSlice(t, selectReq(c, testRequest()), []int{1, 2, 0})
	}
}

func TestCostStrategy_MissingDataLast(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "unpriced"}, // nil cost → after priced endpoints
		{Name: "m1", Alias: "priced", Cost: &EndpointCost{InputPrice: 9, OutputPrice: 9}},
	}
	c := newTestComposeClient(StrategyCost, entries)

	got := selectReq(c, testRequest())
	assertIntSlice(t, got, []int{1, 0})
}

func TestCostStrategy_AliasTieBreak(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "zeta", Cost: &EndpointCost{InputPrice: 1, OutputPrice: 1}},
		{Name: "m1", Alias: "alpha", Cost: &EndpointCost{InputPrice: 1, OutputPrice: 1}},
	}
	c := newTestComposeClient(StrategyCost, entries)

	// Equal cost → alias tie-break: alpha before zeta → indices 1, 0.
	assertIntSlice(t, selectReq(c, testRequest()), []int{1, 0})
}

func TestCostStrategy_FailoverAfterCheapestFails(t *testing.T) {
	sFail := newFailServer(t)
	defer sFail.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	cc, err := NewComposeClient(StrategyCost, []ModelEntry{
		{Name: "m0", Alias: "cheap", Cost: &EndpointCost{InputPrice: 1, OutputPrice: 1}, Client: newClientForServer(t, sFail)},
		{Name: "m1", Alias: "pricey", Cost: &EndpointCost{InputPrice: 9, OutputPrice: 9}, Client: newClientForServer(t, sOK)},
	})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("expected failover to pricey endpoint, got %v", err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %q, want m1 after cheapest failed", resp.Model)
	}
}

func TestLatencyStrategy_DeterministicOrder(t *testing.T) {
	slow := 50 * time.Millisecond
	fast := 10 * time.Millisecond
	mid := 30 * time.Millisecond

	entries := []ModelEntry{
		{Name: "m0", Alias: "slow", Latency: &slow},
		{Name: "m1", Alias: "fast", Latency: &fast},
		{Name: "m2", Alias: "mid", Latency: &mid},
	}
	c := newTestComposeClient(StrategyLatency, entries)

	// Ascending latency: fast(10) < mid(30) < slow(50) → indices 1, 2, 0.
	assertIntSlice(t, selectReq(c, testRequest()), []int{1, 2, 0})
}

func TestLatencyStrategy_MissingDataLast(t *testing.T) {
	slow := 50 * time.Millisecond

	entries := []ModelEntry{
		{Name: "m0", Alias: "unknown"}, // nil latency → last
		{Name: "m1", Alias: "known", Latency: &slow},
	}
	c := newTestComposeClient(StrategyLatency, entries)

	assertIntSlice(t, selectReq(c, testRequest()), []int{1, 0})
}

func TestLatencyStrategy_AliasTieBreak(t *testing.T) {
	same := 20 * time.Millisecond

	entries := []ModelEntry{
		{Name: "m0", Alias: "zeta", Latency: &same},
		{Name: "m1", Alias: "alpha", Latency: &same},
	}
	c := newTestComposeClient(StrategyLatency, entries)

	assertIntSlice(t, selectReq(c, testRequest()), []int{1, 0})
}

func TestCapabilityFilter_RunsBeforeHealth(t *testing.T) {
	// The incapable endpoint is healthy but must still be excluded; the capable
	// endpoint is errored, so nothing is available → ErrNoActiveModels (not a
	// capability error, since a capable endpoint does exist).
	entries := []ModelEntry{
		{Name: "m0", Alias: "healthy-incapable", Capability: &Capability{Tools: false}},
		{Name: "m1", Alias: "errored-capable", Capability: &Capability{Tools: true}},
	}
	c := newTestComposeClient(StrategyFailover, entries)
	c.health[1].markError(errors.New("down"), time.Now())

	got := selectReq(c, toolsRequest())
	if len(got) != 0 {
		t.Fatalf("expected no available capable candidates, got %v", got)
	}
}
