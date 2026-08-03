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
	"errors"
	"sync"
	"testing"
	"time"
)

func TestClassifyHealth(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want healthOutcome
	}{
		{"429 cools", &statusError{status: 429}, outcomeCooling},
		{"500 errors", &statusError{status: 500}, outcomeError},
		{"503 errors", &statusError{status: 503}, outcomeError},
		{"400 request failure", &statusError{status: 400}, outcomeRequestFailure},
		{"404 request failure", &statusError{status: 404}, outcomeRequestFailure},
		{"status 0 errors", &statusError{status: 0}, outcomeError},
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

func TestEndpointHealth_NewIsActive(t *testing.T) {
	if !newEndpointHealth().isActive() {
		t.Fatal("a new endpointHealth should be active")
	}
}

func TestEndpointHealth_StateTransitions(t *testing.T) {
	h := newEndpointHealth()
	now := time.Now()

	h.markError(errors.New("fail"), now)

	if h.isActive() {
		t.Fatal("should be in the error state after markError")
	}

	h.markActive()

	if !h.isActive() {
		t.Fatal("should be active after markActive")
	}
}

func TestEndpointHealth_SuccessResetsAccounting(t *testing.T) {
	h := newEndpointHealth()
	now := time.Now()

	h.markError(errors.New("e1"), now)
	h.markError(errors.New("e2"), now.Add(time.Second))

	if snap := h.snapshot(); snap.errorCount != 2 {
		t.Fatalf("error count = %d, want 2", snap.errorCount)
	}

	h.markActive()

	snap := h.snapshot()
	if snap.state != stateActive || snap.errorCount != 0 || snap.lastError != nil || !snap.errorTime.IsZero() {
		t.Fatalf("after recovery: state=%s count=%d err=%v time=%v",
			snap.state, snap.errorCount, snap.lastError, snap.errorTime)
	}
}

func TestEndpointHealth_CoolingDoesNotAdvanceBackoff(t *testing.T) {
	h := newEndpointHealth()
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
		t.Fatal("a cooling endpoint must not be treated as an error probe")
	}
}

func TestEndpointHealth_AvailableAfterCooling(t *testing.T) {
	h := newEndpointHealth()
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	cooling := 10 * time.Second

	h.markCooling(errors.New("429"), now)

	if h.available(now, cooling) {
		t.Fatal("a cooling endpoint should be unavailable during the cooling window")
	}

	if !h.available(now.Add(cooling), cooling) {
		t.Fatal("a cooling endpoint should rejoin rotation once the window elapses")
	}
}

func TestEndpointHealth_ShouldProbe(t *testing.T) {
	h := newEndpointHealth()
	interval := 60 * time.Second
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	if h.shouldProbe(now, interval) {
		t.Fatal("an active endpoint should not need probing")
	}

	// First error: backoff = 1x interval (60s).
	h.markError(errors.New("fail"), now)

	if h.shouldProbe(now.Add(30*time.Second), interval) {
		t.Fatal("should not probe before the interval elapses")
	}

	if !h.shouldProbe(now.Add(60*time.Second), interval) {
		t.Fatal("should probe at the interval boundary")
	}

	if !h.shouldProbe(now.Add(90*time.Second), interval) {
		t.Fatal("should probe after the interval")
	}
}

func TestEndpointHealth_ShouldProbe_ExponentialBackoff(t *testing.T) {
	h := newEndpointHealth()
	interval := 10 * time.Second
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// 1st error: backoff = 1x = 10s.
	h.markError(errors.New("fail1"), now)

	if !h.shouldProbe(now.Add(10*time.Second), interval) {
		t.Fatal("1st error: should probe after 10s")
	}

	// 2nd error: backoff = 2x = 20s.
	h.markError(errors.New("fail2"), now)

	if h.shouldProbe(now.Add(15*time.Second), interval) {
		t.Fatal("2nd error: should not probe after 15s (backoff=20s)")
	}

	if !h.shouldProbe(now.Add(20*time.Second), interval) {
		t.Fatal("2nd error: should probe after 20s")
	}

	// 3rd error: backoff = 4x = 40s.
	h.markError(errors.New("fail3"), now)

	if h.shouldProbe(now.Add(30*time.Second), interval) {
		t.Fatal("3rd error: should not probe after 30s (backoff=40s)")
	}

	if !h.shouldProbe(now.Add(40*time.Second), interval) {
		t.Fatal("3rd error: should probe after 40s")
	}
}

func TestEndpointHealth_ShouldProbe_BackoffCap(t *testing.T) {
	h := newEndpointHealth()
	interval := 10 * time.Second
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// 100 consecutive errors — the backoff caps at 64x.
	for range 100 {
		h.markError(errors.New("fail"), now)
	}

	maxBackoff := interval * 64

	if h.shouldProbe(now.Add(maxBackoff-time.Second), interval) {
		t.Fatal("should not probe before the capped backoff")
	}

	if !h.shouldProbe(now.Add(maxBackoff), interval) {
		t.Fatal("should probe at the capped backoff")
	}
}

// A 429 answering a recovery probe must not demote a long error backoff to the
// much shorter cooling interval, which would let the endpoint skip its backoff.
func TestEndpointHealth_429OnErroredKeepsBackoff(t *testing.T) {
	h := newEndpointHealth()
	now := time.Now()

	for range 3 {
		h.markError(errors.New("5xx"), now)
	}

	// The probe fires after the 4x backoff (errorCount=3 → 2^2) and gets a 429.
	probeAt := now.Add(4 * time.Minute)
	h.markCooling(&statusError{status: 429}, probeAt)

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
		t.Fatal("the next probe must wait the full 4x backoff, not one interval")
	}

	if !h.shouldProbe(probeAt.Add(4*time.Minute), time.Minute) {
		t.Fatal("the probe should be due once the 4x backoff elapsed")
	}
}

func TestEndpointHealth_ConcurrentAccess(t *testing.T) {
	h := newEndpointHealth()
	now := time.Now()

	var wg sync.WaitGroup

	for i := range 100 {
		wg.Add(1)

		go func(n int) {
			defer wg.Done()

			if n%2 == 0 {
				h.markError(errors.New("fail"), now)
			} else {
				h.markActive()
			}

			h.isActive()
			h.shouldProbe(now, time.Minute)
		}(i)
	}

	wg.Wait()
}
