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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vogo/aimodel/ais"
)

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

func TestClassifyHealth(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want healthOutcome
	}{
		{"429 cools", &ais.APIError{StatusCode: 429}, outcomeCooling},
		{"500 errors", &ais.APIError{StatusCode: 500}, outcomeError},
		{"503 errors", &ais.APIError{StatusCode: 503}, outcomeError},
		{"400 request failure", &ais.APIError{StatusCode: 400}, outcomeRequestFailure},
		{"404 request failure", &ais.APIError{StatusCode: 404}, outcomeRequestFailure},
		{"status 0 errors", &ais.APIError{StatusCode: 0}, outcomeError},
		{"transport errors", errors.New("connection refused"), outcomeError},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classifyHealth(tc.err); got != tc.want {
				t.Fatalf("classifyHealth = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestModelHealth_CoolingDoesNotAdvanceBackoff(t *testing.T) {
	h := newModelHealth()
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	h.markCooling(errors.New("rate limited"), now)

	snap := h.snapshot()
	if snap.state != stateCooling {
		t.Fatalf("state = %s, want cooling", snap.state)
	}

	if snap.errorCount != 0 {
		t.Fatalf("errorCount after cooling = %d, want 0", snap.errorCount)
	}

	// Cooling never becomes an error probe.
	if h.shouldProbe(now.Add(time.Hour), time.Second) {
		t.Fatal("cooling endpoint must not be treated as an error probe")
	}
}

func TestModelHealth_AvailableAfterCooling(t *testing.T) {
	h := newModelHealth()
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	cooling := 10 * time.Second

	h.markCooling(errors.New("429"), now)

	if h.available(now, cooling) {
		t.Fatal("cooling endpoint should be unavailable during the cooling window")
	}

	if !h.available(now.Add(cooling), cooling) {
		t.Fatal("cooling endpoint should rejoin rotation once the window elapses")
	}
}

func TestCooling_429SkippedThenRejoin(t *testing.T) {
	s429, hits429 := newStatusServer(t, http.StatusTooManyRequests)
	defer s429.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "limited", Client: newClientForServer(t, s429)},
		{Name: "m1", Alias: "ok", Client: newClientForServer(t, sOK)},
	}, WithCoolingInterval(time.Second))
	if err != nil {
		t.Fatal(err)
	}

	cc.nowFunc = func() time.Time { return now }

	// First request: ep0 hits 429 → cooling; failover to ep1 succeeds.
	resp, err := cc.ChatCompletion(context.Background(), testRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Model != "m1" {
		t.Fatalf("model = %q, want m1 (failover from cooled ep0)", resp.Model)
	}

	if hits429.Load() != 1 {
		t.Fatalf("429 endpoint hits = %d, want 1", hits429.Load())
	}

	// ep0 is cooling: within the window it is not selectable.
	got := selectAll(cc)
	assertIntSlice(t, got, []int{1})

	// A request inside the cooling window goes only to ep1.
	if _, err := cc.ChatCompletion(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	if hits429.Load() != 1 {
		t.Fatalf("cooled endpoint was hit again during window: %d", hits429.Load())
	}

	// After the cooling window the endpoint rejoins regular rotation.
	cc.nowFunc = func() time.Time { return now.Add(2 * time.Second) }

	got = selectAll(cc)
	assertIntSlice(t, got, []int{0, 1})
}

func TestError_5xxEntersErrorAndBackoff(t *testing.T) {
	s500, _ := newStatusServer(t, http.StatusInternalServerError)
	defer s500.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "broken", Client: newClientForServer(t, s500)},
		{Name: "m1", Alias: "ok", Client: newClientForServer(t, sOK)},
	}, WithRecoveryInterval(time.Minute))
	if err != nil {
		t.Fatal(err)
	}

	cc.nowFunc = func() time.Time { return now }

	if _, err := cc.ChatCompletion(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	stats := cc.Stats()
	if stats[0].Status != "error" {
		t.Fatalf("ep0 status = %q, want error", stats[0].Status)
	}

	if stats[0].ErrorCount != 1 {
		t.Fatalf("ep0 errorCount = %d, want 1", stats[0].ErrorCount)
	}

	if stats[0].LastError == nil {
		t.Fatal("ep0 lastError should be recorded")
	}

	// A second 5xx advances the consecutive count. Advance past the first
	// backoff (1x the recovery interval) so the errored endpoint is probed.
	cc.nowFunc = func() time.Time { return now.Add(time.Minute + time.Second) }

	if _, err := cc.ChatCompletion(context.Background(), testRequest()); err != nil {
		t.Fatal(err)
	}

	if cc.Stats()[0].ErrorCount != 2 {
		t.Fatalf("ep0 errorCount = %d, want 2", cc.Stats()[0].ErrorCount)
	}
}

func TestRequestFailure_4xxStaysActive(t *testing.T) {
	s400, _ := newStatusServer(t, http.StatusBadRequest)
	defer s400.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "badreq", Client: newClientForServer(t, s400)},
	})
	if err != nil {
		t.Fatal(err)
	}

	// The request fails and is attributed...
	_, err = cc.ChatCompletion(context.Background(), testRequest())

	var ee *EndpointError
	if !errors.As(err, &ee) || ee.Alias != "badreq" {
		t.Fatalf("expected attributed EndpointError for badreq, got %v", err)
	}

	// ...but the endpoint is not judged unhealthy.
	if !cc.health[0].isActive() {
		t.Fatal("4xx request failure must not mark the endpoint unhealthy")
	}

	if s := cc.Stats()[0]; s.Status != "active" || s.ErrorCount != 0 || s.LastError != nil {
		t.Fatalf("stats after 4xx = %+v, want active/0/nil", s)
	}
}

func TestStats_SnapshotPerAlias(t *testing.T) {
	s500, _ := newStatusServer(t, http.StatusInternalServerError)
	defer s500.Close()

	sOK := newTestServer(t)
	defer sOK.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "broken", Client: newClientForServer(t, s500)},
		{Name: "m1", Alias: "healthy", Client: newClientForServer(t, sOK)},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, _ = cc.ChatCompletion(context.Background(), testRequest())

	stats := cc.Stats()
	if len(stats) != 2 {
		t.Fatalf("stats len = %d, want 2", len(stats))
	}

	byAlias := map[string]EndpointStat{}
	for _, s := range stats {
		byAlias[s.Alias] = s
	}

	if byAlias["broken"].Status != "error" {
		t.Fatalf("broken status = %q, want error", byAlias["broken"].Status)
	}

	if byAlias["healthy"].Status != "active" || byAlias["healthy"].LastError != nil || !byAlias["healthy"].ErrorTime.IsZero() {
		t.Fatalf("healthy stats = %+v, want active with zeroed error fields", byAlias["healthy"])
	}
}

func TestStats_SuccessResets(t *testing.T) {
	h := newModelHealth()
	now := time.Now()

	h.markError(errors.New("fail"), now)
	h.markActive()

	snap := h.snapshot()
	if snap.state != stateActive || snap.errorCount != 0 || snap.lastError != nil || !snap.errorTime.IsZero() {
		t.Fatalf("after recovery: state=%s count=%d err=%v time=%v", snap.state, snap.errorCount, snap.lastError, snap.errorTime)
	}
}

func TestStats_ImmutableSnapshot(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "a", Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	stats := cc.Stats()
	stats[0].Alias = "tampered"
	stats[0].ErrorCount = 999

	// Internal state must be unaffected by mutating the returned container.
	if cc.entries[0].Alias != "a" {
		t.Fatalf("internal alias mutated to %q", cc.entries[0].Alias)
	}

	if cc.Stats()[0].ErrorCount != 0 {
		t.Fatal("internal errorCount mutated via snapshot")
	}
}

func TestStats_ConcurrentSafe(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	s500, _ := newStatusServer(t, http.StatusInternalServerError)
	defer s500.Close()

	cc, err := NewComposeClient(StrategyRandom, []ModelEntry{
		{Name: "m0", Alias: "ok", Client: newClientForServer(t, s)},
		{Name: "m1", Alias: "flaky", Client: newClientForServer(t, s500)},
	})
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup

	for range 30 {
		wg.Go(func() {
			_, _ = cc.ChatCompletion(context.Background(), testRequest())
			_ = cc.Stats()
		})
	}

	wg.Wait()
}

// A cooling endpoint whose interval has elapsed is already selectable again, so
// Stats() must report it as active instead of leaving a stale "cooling" until
// the next success — otherwise the reported state contradicts routing.
func TestStats_ElapsedCoolingReportsActive(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "a", Client: newClientForServer(t, s)},
	}, WithCoolingInterval(10*time.Second))
	if err != nil {
		t.Fatal(err)
	}

	now := time.Now()
	cc.nowFunc = func() time.Time { return now }
	cc.health[0].markCooling(&ais.APIError{StatusCode: 429}, now)

	if got := cc.Stats()[0].Status; got != "cooling" {
		t.Fatalf("status right after 429 = %q, want cooling", got)
	}

	// Interval elapsed: available() would select it, so Stats must agree.
	cc.nowFunc = func() time.Time { return now.Add(11 * time.Second) }

	if got := cc.Stats()[0].Status; got != "active" {
		t.Fatalf("status after the cooling interval = %q, want active", got)
	}

	if !cc.health[0].available(now.Add(11*time.Second), cc.coolingInterval) {
		t.Fatal("endpoint should be available once cooling elapsed")
	}
}

// A 429 answering a recovery probe must not demote a long error backoff to the
// much shorter cooling interval, which would let the endpoint skip its backoff.
func TestModelHealth_429OnErroredKeepsBackoff(t *testing.T) {
	h := newModelHealth()
	now := time.Now()

	for range 3 {
		h.markError(errors.New("5xx"), now)
	}

	// The probe fires after the 4x backoff (errorCount=3 → 2^2) and gets a 429.
	probeAt := now.Add(4 * time.Minute)
	h.markCooling(&ais.APIError{StatusCode: 429}, probeAt)

	snap := h.snapshot()
	if snap.state != stateError {
		t.Fatalf("state = %s, want error (a 429 must not clear the error state)", snap.state)
	}

	if snap.errorCount != 3 {
		t.Fatalf("errorCount = %d, want 3 (cooling never advances the backoff)", snap.errorCount)
	}

	// Not selectable via the cooling path, and the next probe waits a full
	// backoff from the probe attempt rather than the 10s cooling interval.
	if h.available(probeAt.Add(30*time.Second), 10*time.Second) {
		t.Fatal("an errored endpoint must not rejoin rotation through cooling")
	}

	if h.shouldProbe(probeAt.Add(time.Minute), time.Minute) {
		t.Fatal("next probe must wait the full 4x backoff, not one interval")
	}

	if !h.shouldProbe(probeAt.Add(4*time.Minute), time.Minute) {
		t.Fatal("probe should be due once the 4x backoff elapsed")
	}
}
