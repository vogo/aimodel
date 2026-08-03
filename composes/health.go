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
)

// statusCoder is the structural contract a backend's transport error satisfies.
// Declaring it locally — rather than importing a provider's error type — is what
// lets one health state machine classify failures from every protocol this
// module wraps without importing any of them.
type statusCoder interface{ StatusCode() int }

// endpointState is the health of one endpoint. There are exactly two states:
//
//   - available: the endpoint may be selected — as the pool's active endpoint,
//     as its replacement, or as a temporary pick for a call the active cannot
//     serve.
//   - dead: the endpoint exhausted its in-call retries, or answered with a
//     credential failure. It is not selectable until the recover time elapses,
//     at which point it becomes available again on the clock alone — there is no
//     recovery probe and no exponential health backoff.
type endpointState string

const (
	stateAvailable endpointState = "available"
	stateDead      endpointState = "dead"
)

// failureOutcome classifies how a failed attempt is handled.
type failureOutcome int

const (
	// outcomeRetryable is every failure that may be transient: 429, 5xx, 400 and
	// other 4xx, and transport errors alike. The attempt is repeated against the
	// same endpoint under the exponential retry policy, and the endpoint is only
	// judged dead once those retries are exhausted.
	outcomeRetryable failureOutcome = iota
	// outcomeCredential is HTTP 401/403: the endpoint's credentials do not work,
	// so repeating the same call cannot help. It skips retries entirely and the
	// endpoint is judged dead at once.
	outcomeCredential
)

// classifyFailure decides how an attempt error is handled.
//
//   - HTTP 401 / 403 → credential failure (no retry, immediate death)
//   - anything else  → retryable
//
// The deliberate simplicity here is a reversal of the older three-way split
// (cooling / error / request-failure): 429 no longer has its own short cooling
// window, and a 4xx no longer leaves the endpoint healthy while still failing
// the call. One retry path and one recovery timer describe the whole model.
func classifyFailure(err error) failureOutcome {
	var sc statusCoder
	if !errors.As(err, &sc) {
		return outcomeRetryable
	}

	switch sc.StatusCode() {
	case 401, 403:
		return outcomeCredential
	default:
		return outcomeRetryable
	}
}

// endpointHealth tracks the health state of a single endpoint.
type endpointHealth struct {
	mu         sync.RWMutex
	state      endpointState
	lastError  error
	errorTime  time.Time
	errorCount int
}

func newEndpointHealth() *endpointHealth {
	return &endpointHealth{state: stateAvailable}
}

// markSuccess records a successful attempt: the endpoint is available and all
// failure accounting is cleared.
func (h *endpointHealth) markSuccess() {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.state = stateAvailable
	h.lastError = nil
	h.errorCount = 0
	h.errorTime = time.Time{}
}

// markDead records the failure that took the endpoint out of rotation: either a
// credential rejection or the last error of an exhausted retry round. The
// timestamp starts the recovery window.
func (h *endpointHealth) markDead(err error, now time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.state = stateDead
	h.lastError = err
	h.errorTime = now
	h.errorCount++
}

// available reports whether the endpoint may be selected now. A dead endpoint
// becomes available again purely on the clock, once recover has elapsed since
// the failure — recovery restores candidacy, nothing more.
func (h *endpointHealth) available(now time.Time, recover time.Duration) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.state == stateAvailable {
		return true
	}

	return now.Sub(h.errorTime) >= recover
}

// healthSnapshot is a consistent copy of one endpoint's health accounting.
type healthSnapshot struct {
	state      endpointState
	errorCount int
	lastError  error
	errorTime  time.Time
}

// snapshot returns a consistent copy of the health accounting for Stats().
func (h *endpointHealth) snapshot() healthSnapshot {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return healthSnapshot{
		state:      h.state,
		errorCount: h.errorCount,
		lastError:  h.lastError,
		errorTime:  h.errorTime,
	}
}
