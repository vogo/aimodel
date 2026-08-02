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
	"time"

	"github.com/vogo/aimodel/ais"
)

// modelState represents the health state of a model endpoint.
//
//   - active:  participating in regular selection.
//   - cooling: a 429 rate-limit response put the endpoint to sleep; it is
//     skipped until the cooling interval elapses, then rejoins regular
//     rotation. Cooling never counts toward the consecutive-failure backoff.
//   - error:   a 5xx (or transport) failure; the endpoint is skipped until an
//     exponential-backoff recovery probe is due.
type modelState string

const (
	stateActive  modelState = "active"
	stateCooling modelState = "cooling"
	stateError   modelState = "error"
)

// healthOutcome classifies how a failed attempt affects endpoint health.
type healthOutcome int

const (
	// outcomeError marks the endpoint errored (5xx or transport failure) and
	// advances the consecutive-failure backoff.
	outcomeError healthOutcome = iota
	// outcomeCooling puts the endpoint into rate-limit cooling (HTTP 429); it
	// does not count as a health failure.
	outcomeCooling
	// outcomeRequestFailure is a client-side request failure (non-429 4xx). The
	// error is attributed and failover continues, but the endpoint is not
	// marked unhealthy.
	outcomeRequestFailure
)

// classifyHealth decides how an attempt error affects endpoint health.
//
//   - HTTP 429                       → cooling
//   - HTTP 5xx (or status 0, e.g. an
//     SSE error event)                → error
//   - other HTTP 4xx                 → request failure (endpoint stays healthy)
//   - no APIError (transport, etc.)  → error
func classifyHealth(err error) healthOutcome {
	var apiErr *ais.APIError
	if !errors.As(err, &apiErr) {
		return outcomeError
	}

	switch {
	case apiErr.StatusCode == 429:
		return outcomeCooling
	case apiErr.StatusCode >= 400 && apiErr.StatusCode < 500:
		return outcomeRequestFailure
	default:
		// 5xx, or status 0 (stream-level error without an HTTP code).
		return outcomeError
	}
}

// modelHealth tracks the health state of a single model endpoint.
type modelHealth struct {
	mu         sync.RWMutex
	state      modelState
	lastError  error
	errorTime  time.Time
	errorCount int
}

func newModelHealth() *modelHealth {
	return &modelHealth{state: stateActive}
}

// markActive records a successful attempt: the endpoint returns to active and
// all failure accounting is cleared.
func (h *modelHealth) markActive() {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.state = stateActive
	h.lastError = nil
	h.errorCount = 0
	h.errorTime = time.Time{}
}

// markError records a health-relevant failure (5xx / transport): the endpoint
// enters the error state and the consecutive-failure count advances.
func (h *modelHealth) markError(err error, now time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.state = stateError
	h.lastError = err
	h.errorTime = now
	h.errorCount++
}

// markCooling records a rate-limit (429) failure: the endpoint enters cooling
// with the error and timestamp recorded, but the consecutive-failure count is
// left untouched so cooling never drives the error backoff.
//
// An endpoint already in the error state stays there: a 429 answering a recovery
// probe must not demote a long backoff to the much shorter cooling interval. The
// probe's timestamp is recorded so the next probe waits another full backoff at
// the current level, and errorCount still does not advance.
func (h *modelHealth) markCooling(err error, now time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.lastError = err
	h.errorTime = now

	if h.state != stateError {
		h.state = stateCooling
	}
}

// isActive reports whether the endpoint is in the active state.
func (h *modelHealth) isActive() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return h.state == stateActive
}

// available reports whether the endpoint may be selected now. An active
// endpoint is always available; a cooling endpoint becomes available again once
// the cooling interval has elapsed; an errored endpoint is never available
// through this path (it returns via recovery probes instead).
func (h *modelHealth) available(now time.Time, cooling time.Duration) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	switch h.state {
	case stateActive:
		return true
	case stateCooling:
		return now.Sub(h.errorTime) >= cooling
	default:
		return false
	}
}

// healthSnapshot is a consistent copy of one endpoint's health accounting.
type healthSnapshot struct {
	state      modelState
	errorCount int
	lastError  error
	errorTime  time.Time
}

// snapshot returns a consistent copy of the health accounting for Stats().
func (h *modelHealth) snapshot() healthSnapshot {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return healthSnapshot{
		state:      h.state,
		errorCount: h.errorCount,
		lastError:  h.lastError,
		errorTime:  h.errorTime,
	}
}

// maxBackoffShift caps exponential backoff at 2^6 = 64x the base interval.
const maxBackoffShift = 6

// shouldProbe returns true if enough time has passed since the last error
// for a recovery probe attempt. The required wait time grows exponentially
// with consecutive errors, capped at 64x the base interval. Only the error
// state is probed; cooling endpoints rejoin rotation on their own timer.
func (h *modelHealth) shouldProbe(now time.Time, interval time.Duration) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.state != stateError {
		return false
	}

	shift := max(h.errorCount-1, 0)
	shift = min(shift, maxBackoffShift)

	backoff := interval * time.Duration(1<<shift)

	return now.Sub(h.errorTime) >= backoff
}
