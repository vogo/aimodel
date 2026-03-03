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

func TestModelHealth_NewIsActive(t *testing.T) {
	h := newModelHealth()
	if !h.isActive() {
		t.Fatal("new modelHealth should be active")
	}
}

func TestModelHealth_StateTransitions(t *testing.T) {
	h := newModelHealth()
	now := time.Now()

	// active → error
	h.markError(errors.New("fail"), now)
	if h.isActive() {
		t.Fatal("should be in error state after markError")
	}

	// error → active
	h.markActive()
	if !h.isActive() {
		t.Fatal("should be active after markActive")
	}
}

func TestModelHealth_ErrorCountResets(t *testing.T) {
	h := newModelHealth()
	now := time.Now()

	h.markError(errors.New("e1"), now)
	h.markError(errors.New("e2"), now.Add(time.Second))

	h.mu.RLock()
	count := h.errorCount
	h.mu.RUnlock()

	if count != 2 {
		t.Fatalf("error count = %d, want 2", count)
	}

	h.markActive()

	h.mu.RLock()
	count = h.errorCount
	h.mu.RUnlock()

	if count != 0 {
		t.Fatalf("error count after markActive = %d, want 0", count)
	}
}

func TestModelHealth_ShouldProbe(t *testing.T) {
	h := newModelHealth()
	interval := 60 * time.Second
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// Active model should not probe.
	if h.shouldProbe(now, interval) {
		t.Fatal("active model should not need probing")
	}

	// First error: backoff = 1x interval (60s).
	h.markError(errors.New("fail"), now)
	if h.shouldProbe(now.Add(30*time.Second), interval) {
		t.Fatal("should not probe before interval elapses")
	}

	if !h.shouldProbe(now.Add(60*time.Second), interval) {
		t.Fatal("should probe at interval boundary")
	}

	if !h.shouldProbe(now.Add(90*time.Second), interval) {
		t.Fatal("should probe after interval")
	}
}

func TestModelHealth_ShouldProbe_ExponentialBackoff(t *testing.T) {
	h := newModelHealth()
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

func TestModelHealth_ShouldProbe_BackoffCap(t *testing.T) {
	h := newModelHealth()
	interval := 10 * time.Second
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// Mark 100 consecutive errors — backoff should cap at 64x = 640s.
	for range 100 {
		h.markError(errors.New("fail"), now)
	}

	maxBackoff := interval * 64
	if h.shouldProbe(now.Add(maxBackoff-time.Second), interval) {
		t.Fatal("should not probe before capped backoff")
	}

	if !h.shouldProbe(now.Add(maxBackoff), interval) {
		t.Fatal("should probe at capped backoff")
	}
}

func TestModelHealth_ConcurrentAccess(t *testing.T) {
	h := newModelHealth()
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
