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
	"sync"
	"time"
)

// modelState represents the health state of a model.
type modelState string

const (
	stateActive modelState = "active"
	stateError  modelState = "error"
)

// modelHealth tracks the health state of a single model entry.
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

func (h *modelHealth) markActive() {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.state = stateActive
	h.lastError = nil
	h.errorCount = 0
	h.errorTime = time.Time{}
}

func (h *modelHealth) markError(err error, now time.Time) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.state = stateError
	h.lastError = err
	h.errorTime = now
	h.errorCount++
}

func (h *modelHealth) isActive() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return h.state == stateActive
}

// maxBackoffShift caps exponential backoff at 2^6 = 64x the base interval.
const maxBackoffShift = 6

// shouldProbe returns true if enough time has passed since the last error
// for a recovery probe attempt. The required wait time grows exponentially
// with consecutive errors, capped at 64x the base interval.
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
