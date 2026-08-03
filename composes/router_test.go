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
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// statusError is a stand-in for a provider's transport error: the router only
// ever reads a status code off it, structurally.
type statusError struct {
	status int
}

func (e *statusError) Error() string   { return fmt.Sprintf("status %d", e.status) }
func (e *statusError) StatusCode() int { return e.status }

// attemptLog records which endpoints an attempt closure was invoked for.
type attemptLog struct {
	mu    sync.Mutex
	calls []int
}

func (a *attemptLog) record(idx int) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.calls = append(a.calls, idx)
}

func (a *attemptLog) seen() []int {
	a.mu.Lock()
	defer a.mu.Unlock()

	return append([]int(nil), a.calls...)
}

// newTestRouter builds a router with a deterministic rng so ordering
// assertions do not depend on the process seed.
func newTestRouter(t *testing.T, strategy Strategy, endpoints []Endpoint, opts ...Option) *Router {
	t.Helper()

	r, err := NewRouter(strategy, endpoints, opts...)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	r.rng = newRand(42)

	return r
}

// endpointsNamed builds simple endpoints with explicit aliases.
func endpointsNamed(aliases ...string) []Endpoint {
	endpoints := make([]Endpoint, len(aliases))
	for i, a := range aliases {
		endpoints[i] = Endpoint{Alias: a}
	}

	return endpoints
}

// dispatchTo runs one dispatch whose attempt succeeds on the endpoints listed
// in ok and fails with the given error elsewhere, returning the endpoint that
// served it (or the dispatch error).
func dispatchTo(ctx context.Context, r *Router, call Call, log *attemptLog, fail error, ok ...int) (int, error) {
	okSet := make(map[int]bool, len(ok))
	for _, idx := range ok {
		okSet[idx] = true
	}

	return Dispatch(ctx, r, call, func(_ context.Context, endpoint int) (int, error) {
		log.record(endpoint)

		if !okSet[endpoint] {
			return 0, fail
		}

		return endpoint, nil
	})
}

func assertIntSlice(t *testing.T, got, want []int) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len=%d, want %d: got %v", len(got), len(want), got)
	}

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestNewRouter_EmptyEndpoints(t *testing.T) {
	if _, err := NewRouter(StrategyFailover, nil); err == nil {
		t.Fatal("expected error for empty endpoint list")
	}
}

func TestNewRouter_DerivesAndValidatesAliases(t *testing.T) {
	r, err := NewRouter(StrategyFailover, []Endpoint{{}, {Alias: "named"}, {}})
	if err != nil {
		t.Fatal(err)
	}

	assertStringSlice(t, r.Aliases(), []string{"entry-0", "named", "entry-2"})
}

func TestNewRouter_DuplicateAlias(t *testing.T) {
	_, err := NewRouter(StrategyFailover, endpointsNamed("x", "x"))
	if err == nil || !strings.Contains(err.Error(), "duplicate alias") {
		t.Fatalf("expected duplicate-alias error, got %v", err)
	}
}

// The router owns its endpoints: deriving aliases may not write back into the
// caller's slice, and later mutations by the caller must not reach routing.
func TestNewRouter_DoesNotMutateCallerEndpoints(t *testing.T) {
	endpoints := []Endpoint{{Weight: 1}, {Weight: 2}}

	r := newTestRouter(t, StrategyFailover, endpoints)

	if endpoints[0].Alias != "" || endpoints[1].Alias != "" {
		t.Fatalf("caller endpoints were written back: %q, %q", endpoints[0].Alias, endpoints[1].Alias)
	}

	endpoints[0].Weight = 99

	if r.endpoints[0].Weight != 1 {
		t.Fatalf("router endpoint follows caller mutation: %d", r.endpoints[0].Weight)
	}
}

func TestDispatch_FirstCandidateSucceeds(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a", "b"))
	log := &attemptLog{}

	got, err := dispatchTo(context.Background(), r, Call{}, log, errors.New("boom"), 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	if got != 0 {
		t.Fatalf("served by %d, want 0", got)
	}

	assertIntSlice(t, log.seen(), []int{0})
}

func TestDispatch_FailoverToNextCandidate(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a", "b"))
	log := &attemptLog{}

	got, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500}, 1)
	if err != nil {
		t.Fatal(err)
	}

	if got != 1 {
		t.Fatalf("served by %d, want 1", got)
	}

	assertIntSlice(t, log.seen(), []int{0, 1})
}

func TestDispatch_AllFailAttributesEveryAlias(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a", "b"))
	log := &attemptLog{}

	_, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500})

	var multi *MultiError
	if !errors.As(err, &multi) {
		t.Fatalf("expected *MultiError, got %T: %v", err, err)
	}

	if len(multi.Errors) != 2 {
		t.Fatalf("endpoint errors = %d, want 2", len(multi.Errors))
	}

	if multi.Errors[0].Alias != "a" || multi.Errors[1].Alias != "b" {
		t.Fatalf("aliases = %q, %q; want a, b", multi.Errors[0].Alias, multi.Errors[1].Alias)
	}

	// errors.As reaches through the aggregate to a single endpoint failure and
	// on to the backend's own error type.
	var endpointErr *EndpointError
	if !errors.As(err, &endpointErr) {
		t.Fatal("expected errors.As to find an EndpointError in the chain")
	}

	var status *statusError
	if !errors.As(err, &status) || status.StatusCode() != 500 {
		t.Fatal("expected errors.As to reach the backend error through the chain")
	}
}

func TestDispatch_NoActiveEndpoints(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a"), WithRecoveryInterval(time.Hour))
	log := &attemptLog{}

	// The only endpoint fails with a 5xx and enters the error state.
	if _, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500}); err == nil {
		t.Fatal("expected the first dispatch to fail")
	}

	// Its backoff has not elapsed, so the next dispatch has no candidate at all.
	_, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500})
	if !errors.Is(err, ErrNoActiveModels) {
		t.Fatalf("expected ErrNoActiveModels, got %v", err)
	}

	assertIntSlice(t, log.seen(), []int{0})
}

func TestDispatch_CapabilityFilterFailsFastBeforeAnyAttempt(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, []Endpoint{
		{Alias: "plain", Declares: Declare()},
		{Alias: "also-plain", Declares: Declare("vision")},
	})
	log := &attemptLog{}

	_, err := dispatchTo(context.Background(), r, Call{Requires: []string{"tools"}}, log, nil, 0, 1)

	if !errors.Is(err, ErrCapabilityNotSatisfied) {
		t.Fatalf("expected ErrCapabilityNotSatisfied, got %v", err)
	}

	var capErr *CapabilityError
	if !errors.As(err, &capErr) {
		t.Fatalf("expected *CapabilityError, got %T", err)
	}

	assertStringSlice(t, capErr.Required, []string{"tools"})
	assertStringSlice(t, capErr.Considered, []string{"plain", "also-plain"})

	if len(log.seen()) != 0 {
		t.Fatalf("capability failure must precede every attempt, saw %v", log.seen())
	}
}

// A declared endpoint that satisfies the labels serves the call; an undeclared
// one is unknown rather than incapable and stays in the list.
func TestDispatch_CapabilityFilterKeepsUndeclared(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, []Endpoint{
		{Alias: "no-tools", Declares: Declare()},
		{Alias: "unknown"},
		{Alias: "tools", Declares: Declare("tools")},
	})

	got := r.capableIndices(Call{Requires: []string{"tools"}})
	assertIntSlice(t, got, []int{1, 2})
}

// Eligible carries a fact about the caller's backend object, not about the
// protocol: endpoints outside it are never attempted.
func TestDispatch_EligibleRestrictsCandidates(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a", "b", "c"))
	log := &attemptLog{}

	got, err := dispatchTo(context.Background(), r, Call{Eligible: []int{1, 2}}, log, errors.New("boom"), 0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}

	if got != 1 {
		t.Fatalf("served by %d, want 1 (endpoint 0 is not eligible)", got)
	}

	assertIntSlice(t, log.seen(), []int{1})
}

func TestDispatch_EligibleCombinesWithCapabilityFilter(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, []Endpoint{
		{Alias: "a", Declares: Declare("tools")},
		{Alias: "b", Declares: Declare()},
		{Alias: "c", Declares: Declare("tools")},
	})

	got := r.capableIndices(Call{Requires: []string{"tools"}, Eligible: []int{1, 2}})
	assertIntSlice(t, got, []int{2})
}

func TestDispatch_CancelledBeforeAttempt(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a"))
	log := &attemptLog{}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := dispatchTo(ctx, r, Call{}, log, nil, 0)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	if len(log.seen()) != 0 {
		t.Fatalf("no attempt should be made on a cancelled context, saw %v", log.seen())
	}

	if !r.health[0].isActive() {
		t.Fatal("health must not be poisoned by cancellation")
	}
}

// Cancellation mid-attempt is still attributed to its alias, but leaves health
// untouched: the endpoint did not misbehave.
func TestDispatch_CancelledMidAttemptAttributesButKeepsHealth(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("slow", "other"))

	var seen []AttemptResult

	r.attemptObservers = append(r.attemptObservers, func(res AttemptResult) {
		seen = append(seen, res)
	})

	ctx, cancel := context.WithCancel(context.Background())

	_, err := Dispatch(ctx, r, Call{}, func(_ context.Context, endpoint int) (int, error) {
		cancel() // the attempt is in flight when the caller gives up

		return 0, fmt.Errorf("aborted at %d", endpoint)
	})

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	if len(seen) != 1 || seen[0].Alias != "slow" || seen[0].Success {
		t.Fatalf("observations = %+v, want one failed attempt for slow", seen)
	}

	if !r.health[0].isActive() {
		t.Fatal("health must not be poisoned by mid-attempt cancellation")
	}
}

func TestDispatch_429CoolsAndRejoinsRotation(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	r := newTestRouter(t, StrategyFailover, endpointsNamed("limited", "ok"), WithCoolingInterval(time.Second))
	r.nowFunc = func() time.Time { return now }

	log := &attemptLog{}

	// The rate-limited endpoint cools; the call fails over and succeeds.
	if _, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 429}, 1); err != nil {
		t.Fatal(err)
	}

	if s := r.Stats()[0]; s.Status != "cooling" || s.ErrorCount != 0 {
		t.Fatalf("stats after 429 = %+v, want cooling with errorCount 0", s)
	}

	// Inside the window it is not selectable at all.
	assertIntSlice(t, selectAll(r), []int{1})

	// Once the window elapses it rejoins regular rotation without a probe.
	r.nowFunc = func() time.Time { return now.Add(2 * time.Second) }
	assertIntSlice(t, selectAll(r), []int{0, 1})

	if s := r.Stats()[0]; s.Status != "active" {
		t.Fatalf("elapsed cooling reports %q, want active", s.Status)
	}
}

func TestDispatch_5xxErrorsAndProbesAfterBackoff(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	r := newTestRouter(t, StrategyFailover, endpointsNamed("broken", "ok"), WithRecoveryInterval(time.Minute))
	r.nowFunc = func() time.Time { return now }

	log := &attemptLog{}

	if _, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500}, 1); err != nil {
		t.Fatal(err)
	}

	stats := r.Stats()
	if stats[0].Status != "error" || stats[0].ErrorCount != 1 || stats[0].LastError == nil {
		t.Fatalf("stats after 5xx = %+v, want error/1/non-nil", stats[0])
	}

	// Before the backoff elapses only the healthy endpoint is tried.
	log = &attemptLog{}
	if _, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500}, 1); err != nil {
		t.Fatal(err)
	}

	assertIntSlice(t, log.seen(), []int{1})

	// Once it elapses the errored endpoint is probed *first*.
	r.nowFunc = func() time.Time { return now.Add(time.Minute + time.Second) }

	log = &attemptLog{}
	if _, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500}, 0, 1); err != nil {
		t.Fatal(err)
	}

	assertIntSlice(t, log.seen(), []int{0})

	if s := r.Stats()[0]; s.Status != "active" || s.ErrorCount != 0 {
		t.Fatalf("stats after a successful probe = %+v, want active/0", s)
	}
}

// A non-429 4xx is the caller's fault, not the endpoint's: it is attributed and
// failed over, but the endpoint stays healthy.
func TestDispatch_4xxDoesNotDowngradeHealth(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("badreq"))
	log := &attemptLog{}

	_, err := dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 400})

	var endpointErr *EndpointError
	if !errors.As(err, &endpointErr) || endpointErr.Alias != "badreq" {
		t.Fatalf("expected an attributed EndpointError for badreq, got %v", err)
	}

	if s := r.Stats()[0]; s.Status != "active" || s.ErrorCount != 0 || s.LastError != nil {
		t.Fatalf("stats after 4xx = %+v, want active/0/nil", s)
	}
}

// A transport failure carries no status and is treated as an endpoint fault.
func TestDispatch_TransportFailureErrorsEndpoint(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("gone", "ok"))
	log := &attemptLog{}

	if _, err := dispatchTo(context.Background(), r, Call{}, log, errors.New("connection refused"), 1); err != nil {
		t.Fatal(err)
	}

	if s := r.Stats()[0]; s.Status != "error" || s.ErrorCount != 1 {
		t.Fatalf("stats after a transport failure = %+v, want error/1", s)
	}
}

func TestDispatch_ObserverSeesEveryAttempt(t *testing.T) {
	var seen []AttemptResult

	r := newTestRouter(t, StrategyFailover, endpointsNamed("bad", "good"),
		WithAttemptObserver(func(res AttemptResult) { seen = append(seen, res) }))

	log := &attemptLog{}

	if _, err := dispatchTo(context.Background(), r, Call{Stream: true}, log, &statusError{status: 500}, 1); err != nil {
		t.Fatal(err)
	}

	if len(seen) != 2 {
		t.Fatalf("observations = %d, want 2", len(seen))
	}

	if seen[0].Alias != "bad" || seen[0].Success || !seen[0].Stream || seen[0].Err == nil {
		t.Fatalf("first observation = %+v, want bad/failure/stream", seen[0])
	}

	if seen[1].Alias != "good" || !seen[1].Success || !seen[1].Stream || seen[1].Err != nil {
		t.Fatalf("second observation = %+v, want good/success/stream", seen[1])
	}
}

// A nil observer is ignored rather than panicking on the request path.
func TestWithAttemptObserver_NilIgnored(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a"), WithAttemptObserver(nil))

	if len(r.attemptObservers) != 0 {
		t.Fatalf("observers = %d, want 0", len(r.attemptObservers))
	}
}

func TestStats_ImmutableSnapshot(t *testing.T) {
	r := newTestRouter(t, StrategyFailover, endpointsNamed("a"))

	stats := r.Stats()
	stats[0].Alias = "tampered"
	stats[0].ErrorCount = 999

	if r.endpoints[0].Alias != "a" {
		t.Fatalf("internal alias mutated to %q", r.endpoints[0].Alias)
	}

	if r.Stats()[0].ErrorCount != 0 {
		t.Fatal("internal errorCount mutated via the snapshot")
	}
}

func TestStats_ConcurrentWithDispatch(t *testing.T) {
	r := newTestRouter(t, StrategyRandom, endpointsNamed("ok", "flaky"))

	var wg sync.WaitGroup

	for range 30 {
		wg.Go(func() {
			log := &attemptLog{}
			_, _ = dispatchTo(context.Background(), r, Call{}, log, &statusError{status: 500}, 0)
			_ = r.Stats()
		})
	}

	wg.Wait()
}

func TestErrorStrings(t *testing.T) {
	endpointErr := &EndpointError{Alias: "ep", Err: errors.New("boom")}
	if !strings.Contains(endpointErr.Error(), "ep") || !strings.Contains(endpointErr.Error(), "boom") {
		t.Fatalf("EndpointError.Error() = %q", endpointErr.Error())
	}

	multi := &MultiError{Errors: []*EndpointError{
		{Alias: "a", Err: errors.New("x")},
		{Alias: "b", Err: errors.New("y")},
	}}

	if msg := multi.Error(); !strings.Contains(msg, "a: x") || !strings.Contains(msg, "b: y") {
		t.Fatalf("MultiError.Error() = %q", msg)
	}

	if (&MultiError{}).Error() == "" {
		t.Fatal("empty MultiError.Error() should not be empty")
	}

	capErr := &CapabilityError{Required: []string{"vision"}, Considered: []string{"text-only"}}
	if !strings.Contains(capErr.Error(), "vision") || !strings.Contains(capErr.Error(), "text-only") {
		t.Fatalf("CapabilityError.Error() = %q", capErr.Error())
	}
}

func assertStringSlice(t *testing.T, got, want []string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len=%d, want %d: got %v", len(got), len(want), got)
	}

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %q, want %q", i, got[i], want[i])
		}
	}
}
