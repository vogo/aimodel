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
	"testing"
	"time"

	"github.com/vogo/aimodel/ais"
)

func stickyClient(aliases ...string) *ComposeClient {
	entries := make([]ModelEntry, len(aliases))
	for i, a := range aliases {
		entries[i] = ModelEntry{Name: "model-" + a, Alias: a}
	}

	return newTestComposeClient(StrategySticky, entries)
}

func selectWithSession(c *ComposeClient, sessionID string) []int {
	ctx := context.Background()
	if sessionID != "" {
		ctx = WithSessionID(ctx, sessionID)
	}

	req := &ais.ChatRequest{}

	return c.selectModels(ctx, req, c.capableIndices(req))
}

func TestSticky_SameSessionStable(t *testing.T) {
	c := stickyClient("a", "b", "c")

	first := selectWithSession(c, "user-1")[0]

	for range 100 {
		got := selectWithSession(c, "user-1")
		if got[0] != first {
			t.Fatalf("sticky first choice changed: %d vs %d", got[0], first)
		}

		if len(got) != 3 {
			t.Fatalf("expected full candidate order, got %v", got)
		}
	}
}

func TestSticky_DifferentSessionsDistribute(t *testing.T) {
	c := stickyClient("a", "b", "c")

	seen := map[int]bool{}

	for i := range 200 {
		got := selectWithSession(c, fmt.Sprintf("session-%d", i))
		seen[got[0]] = true
	}

	// 200 distinct ids across 3 endpoints must hit more than one endpoint.
	if len(seen) < 2 {
		t.Fatalf("sticky distribution collapsed to %v", seen)
	}
}

func TestSticky_CrossInstanceStable(t *testing.T) {
	c1 := stickyClient("a", "b", "c")
	c2 := stickyClient("a", "b", "c")

	for i := range 50 {
		sid := fmt.Sprintf("user-%d", i)
		if c1.stickyPreferredAlias(sid) != c2.stickyPreferredAlias(sid) {
			t.Fatalf("instances disagree for %q: %q vs %q",
				sid, c1.stickyPreferredAlias(sid), c2.stickyPreferredAlias(sid))
		}
	}
}

func TestSticky_NoSessionFallsBackToFailover(t *testing.T) {
	c := stickyClient("a", "b", "c")

	// No session id: falls back to the default (failover → definition order),
	// not a random affinity.
	got := selectWithSession(c, "")
	assertIntSlice(t, got, []int{0, 1, 2})

	// Repeated calls stay deterministic.
	for range 20 {
		assertIntSlice(t, selectWithSession(c, ""), []int{0, 1, 2})
	}
}

func TestSticky_PreferredUnhealthyFailover(t *testing.T) {
	c := stickyClient("a", "b", "c")

	sid := "user-42"
	preferred := c.stickyPreferredAlias(sid)

	// Find and disable the preferred endpoint.
	prefIdx := -1

	for i, a := range []string{"a", "b", "c"} {
		if a == preferred {
			prefIdx = i
		}
	}

	c.health[prefIdx].markError(errors.New("down"), time.Now())

	got := selectWithSession(c, sid)

	for _, idx := range got {
		if idx == prefIdx {
			t.Fatalf("unhealthy preferred endpoint %d still selected", prefIdx)
		}
	}

	if len(got) != 2 {
		t.Fatalf("expected 2 remaining candidates, got %v", got)
	}

	// Deterministic: same result every time while health is unchanged.
	for range 20 {
		assertIntSlice(t, selectWithSession(c, sid), got)
	}
}

func TestSticky_FallbackCoercedFromSticky(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategySticky, []ModelEntry{
		{Name: "m0", Alias: "a", Client: newClientForServer(t, s)},
	}, WithStickyFallback(StrategySticky))
	if err != nil {
		t.Fatal(err)
	}

	if cc.stickyFallback == StrategySticky {
		t.Fatal("sticky fallback must be coerced away from sticky to avoid recursion")
	}
}

func TestSticky_IntegrationRoutesToPreferred(t *testing.T) {
	aliases := []string{"a", "b", "c"}

	entries := make([]ModelEntry, len(aliases))
	for i, a := range aliases {
		s := newTestServer(t)
		t.Cleanup(s.Close)

		entries[i] = ModelEntry{Name: "m" + a, Alias: a, Client: newClientForServer(t, s)}
	}

	cc, err := NewComposeClient(StrategySticky, entries)
	if err != nil {
		t.Fatal(err)
	}

	obs := &recordingObserver{}
	cc.attemptObservers = append(cc.attemptObservers, obs.fn)

	ctx := WithSessionID(context.Background(), "session-xyz")

	resp, err := cc.ChatCompletion(ctx, testRequest())
	if err != nil {
		t.Fatal(err)
	}

	want := cc.stickyPreferredAlias("session-xyz")

	// The successful attempt must be the deterministic preferred alias.
	var successAlias string

	for _, r := range obs.results {
		if r.Success {
			successAlias = r.Alias
		}
	}

	if successAlias != want {
		t.Fatalf("routed to %q, want preferred %q", successAlias, want)
	}

	// Response model reflects the preferred endpoint's model name.
	if resp.Model != "m"+want {
		t.Fatalf("model = %q, want m%s", resp.Model, want)
	}
}
