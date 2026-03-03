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
	"math"
	"testing"
	"time"
)

func newTestComposeClient(strategy Strategy, entries []ModelEntry) *ComposeClient {
	health := make([]*modelHealth, len(entries))
	for i := range health {
		health[i] = newModelHealth()
	}

	return &ComposeClient{
		entries:          entries,
		health:           health,
		strategy:         strategy,
		recoveryInterval: defaultRecoveryInterval,
		nowFunc:          time.Now,
		rng:              newRand(42),
	}
}

func TestSelectFailover_AllActive(t *testing.T) {
	c := newTestComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0"}, {Name: "m1"}, {Name: "m2"},
	})

	got := c.selectModels()
	want := []int{0, 1, 2}

	assertIntSlice(t, got, want)
}

func TestSelectFailover_SkipError(t *testing.T) {
	c := newTestComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0"}, {Name: "m1"}, {Name: "m2"},
	})
	c.health[1].markError(errors.New("fail"), time.Now())

	got := c.selectModels()
	want := []int{0, 2}

	assertIntSlice(t, got, want)
}

func TestSelectFailover_AllError(t *testing.T) {
	c := newTestComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0"}, {Name: "m1"},
	})
	now := time.Now()
	c.health[0].markError(errors.New("fail"), now)
	c.health[1].markError(errors.New("fail"), now)

	got := c.selectModels()
	if len(got) != 0 {
		t.Fatalf("expected empty list, got %v", got)
	}
}

func TestSelectRandom_AllActive(t *testing.T) {
	c := newTestComposeClient(StrategyRandom, []ModelEntry{
		{Name: "m0"}, {Name: "m1"}, {Name: "m2"},
	})

	got := c.selectModels()
	if len(got) != 3 {
		t.Fatalf("expected 3 indices, got %d", len(got))
	}

	// All indices should be present.
	seen := make(map[int]bool)
	for _, idx := range got {
		seen[idx] = true
	}

	for i := range 3 {
		if !seen[i] {
			t.Fatalf("missing index %d in %v", i, got)
		}
	}
}

func TestSelectRandom_SkipError(t *testing.T) {
	c := newTestComposeClient(StrategyRandom, []ModelEntry{
		{Name: "m0"}, {Name: "m1"}, {Name: "m2"},
	})
	c.health[0].markError(errors.New("fail"), time.Now())

	got := c.selectModels()
	if len(got) != 2 {
		t.Fatalf("expected 2 indices, got %d", len(got))
	}

	for _, idx := range got {
		if idx == 0 {
			t.Fatal("should not include errored model index 0")
		}
	}
}

func TestSelectRandom_Distribution(t *testing.T) {
	c := newTestComposeClient(StrategyRandom, []ModelEntry{
		{Name: "m0"}, {Name: "m1"}, {Name: "m2"},
	})

	// Run many iterations to check distribution.
	counts := make(map[int]int)
	iterations := 3000

	for range iterations {
		got := c.selectModels()
		counts[got[0]]++
	}

	// Each model should be selected first roughly 1/3 of the time.
	expected := float64(iterations) / 3.0

	for i := range 3 {
		ratio := float64(counts[i]) / expected
		if ratio < 0.7 || ratio > 1.3 {
			t.Fatalf("model %d selected %d times (expected ~%.0f), ratio=%.2f", i, counts[i], expected, ratio)
		}
	}
}

func TestSelectWeighted_ProportionalDistribution(t *testing.T) {
	c := newTestComposeClient(StrategyWeight, []ModelEntry{
		{Name: "m0", Weight: 3},
		{Name: "m1", Weight: 1},
	})

	counts := make(map[int]int)
	iterations := 4000

	for range iterations {
		got := c.selectModels()
		counts[got[0]]++
	}

	// m0 has weight 3, m1 has weight 1 → m0 should be ~75%.
	ratio := float64(counts[0]) / float64(iterations)
	if math.Abs(ratio-0.75) > 0.05 {
		t.Fatalf("m0 selected ratio=%.3f, expected ~0.75", ratio)
	}
}

func TestSelectWeighted_ZeroWeightTreatedAsOne(t *testing.T) {
	c := newTestComposeClient(StrategyWeight, []ModelEntry{
		{Name: "m0", Weight: 0},
		{Name: "m1", Weight: 1},
	})

	counts := make(map[int]int)
	iterations := 2000

	for range iterations {
		got := c.selectModels()
		counts[got[0]]++
	}

	// Both have effective weight 1 → ~50% each.
	ratio := float64(counts[0]) / float64(iterations)
	if math.Abs(ratio-0.5) > 0.08 {
		t.Fatalf("m0 selected ratio=%.3f, expected ~0.50", ratio)
	}
}

func TestSelectWeighted_SkipError(t *testing.T) {
	c := newTestComposeClient(StrategyWeight, []ModelEntry{
		{Name: "m0", Weight: 10},
		{Name: "m1", Weight: 1},
	})
	c.health[0].markError(errors.New("fail"), time.Now())

	got := c.selectModels()
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("expected [1], got %v", got)
	}
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
