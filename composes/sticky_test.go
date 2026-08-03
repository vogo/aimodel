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
)

func stickyRouter(t *testing.T, aliases ...string) *Router {
	t.Helper()

	return newTestRouter(t, StrategySticky, endpointsNamed(aliases...))
}

func selectWithSession(r *Router, sessionID string) []int {
	ctx := context.Background()
	if sessionID != "" {
		ctx = WithSessionID(ctx, sessionID)
	}

	call := Call{}

	return r.selectEndpoints(ctx, call, r.capableIndices(call))
}

func TestSticky_SameSessionStable(t *testing.T) {
	r := stickyRouter(t, "a", "b", "c")

	first := selectWithSession(r, "user-1")[0]

	for range 100 {
		got := selectWithSession(r, "user-1")
		if got[0] != first {
			t.Fatalf("sticky first choice changed: %d vs %d", got[0], first)
		}

		if len(got) != 3 {
			t.Fatalf("expected the full candidate order, got %v", got)
		}
	}
}

func TestSticky_DifferentSessionsDistribute(t *testing.T) {
	r := stickyRouter(t, "a", "b", "c")

	seen := map[int]bool{}

	for i := range 200 {
		seen[selectWithSession(r, fmt.Sprintf("session-%d", i))[0]] = true
	}

	if len(seen) < 2 {
		t.Fatalf("sticky distribution collapsed to %v", seen)
	}
}

func TestSticky_CrossInstanceStable(t *testing.T) {
	r1 := stickyRouter(t, "a", "b", "c")
	r2 := stickyRouter(t, "a", "b", "c")

	for i := range 50 {
		sid := fmt.Sprintf("user-%d", i)
		if r1.stickyPreferredAlias(sid) != r2.stickyPreferredAlias(sid) {
			t.Fatalf("instances disagree for %q: %q vs %q",
				sid, r1.stickyPreferredAlias(sid), r2.stickyPreferredAlias(sid))
		}
	}
}

func TestSticky_NoSessionFallsBackToFailover(t *testing.T) {
	r := stickyRouter(t, "a", "b", "c")

	// No session id: falls back to the default (failover → declaration order),
	// not a random affinity.
	for range 20 {
		assertIntSlice(t, selectWithSession(r, ""), []int{0, 1, 2})
	}
}

func TestSticky_ConfiguredFallbackApplies(t *testing.T) {
	slow, fast := 50*time.Millisecond, 10*time.Millisecond

	r := newTestRouter(t, StrategySticky, []Endpoint{
		{Alias: "slow", Latency: &slow},
		{Alias: "fast", Latency: &fast},
	}, WithStickyFallback(StrategyLatency))

	assertIntSlice(t, selectWithSession(r, ""), []int{1, 0})
}

func TestSticky_PreferredUnhealthyFailover(t *testing.T) {
	r := stickyRouter(t, "a", "b", "c")

	sid := "user-42"
	preferred := r.stickyPreferredAlias(sid)

	prefIdx := -1

	for i, a := range []string{"a", "b", "c"} {
		if a == preferred {
			prefIdx = i
		}
	}

	r.health[prefIdx].markError(errors.New("down"), time.Now())

	got := selectWithSession(r, sid)

	for _, idx := range got {
		if idx == prefIdx {
			t.Fatalf("unhealthy preferred endpoint %d still selected", prefIdx)
		}
	}

	if len(got) != 2 {
		t.Fatalf("expected 2 remaining candidates, got %v", got)
	}

	// Deterministic: the same result every time while health is unchanged.
	for range 20 {
		assertIntSlice(t, selectWithSession(r, sid), got)
	}
}

func TestSticky_FallbackCoercedFromSticky(t *testing.T) {
	r := newTestRouter(t, StrategySticky, endpointsNamed("a"), WithStickyFallback(StrategySticky))

	if r.stickyFallback == StrategySticky {
		t.Fatal("the sticky fallback must be coerced away from sticky to avoid recursion")
	}
}

// The defensive coercion inside selectSticky, reached only if the field is set
// past the option-time coercion.
func TestSticky_FallbackCoercionAtRuntime(t *testing.T) {
	r := stickyRouter(t, "a", "b")
	r.stickyFallback = StrategySticky

	assertIntSlice(t, selectWithSession(r, ""), []int{0, 1})
}

func TestSticky_NoAvailableCandidates(t *testing.T) {
	r := stickyRouter(t, "a", "b")

	now := r.nowFunc()
	r.health[0].markError(errors.New("down"), now)
	r.health[1].markError(errors.New("down"), now)

	if got := selectWithSession(r, "user-1"); len(got) != 0 {
		t.Fatalf("expected no candidates when all are unhealthy, got %v", got)
	}
}

func TestSticky_NilContext(t *testing.T) {
	if id := sessionIDFromContext(nil); id != "" { //nolint:staticcheck // deliberately exercising the nil-context guard
		t.Fatalf("nil-context session id = %q, want empty", id)
	}
}

// The digest must be reduced in uint32 space: int(h.Sum32()) is negative for
// digests above MaxInt32 wherever int is 32 bits (GOARCH=386/arm/mips), which
// would make the modulo negative and panic on the alias lookup.
func TestSticky_PreferredAliasAlwaysInRange(t *testing.T) {
	for n := 1; n <= 8; n++ {
		aliases := make([]string, n)
		for i := range aliases {
			aliases[i] = fmt.Sprintf("ep-%d", i)
		}

		r := stickyRouter(t, aliases...)

		valid := make(map[string]bool, n)
		for _, a := range aliases {
			valid[a] = true
		}

		for i := range 500 {
			got := r.stickyPreferredAlias(fmt.Sprintf("session-%d", i))
			if !valid[got] {
				t.Fatalf("n=%d session-%d: preferred alias %q is outside the configured set", n, i, got)
			}
		}
	}
}

// Sticky routing must reach the dispatch loop, not just the ordering helper:
// the endpoint that actually serves the call is the preferred one.
func TestSticky_DispatchRoutesToPreferred(t *testing.T) {
	r := stickyRouter(t, "a", "b", "c")

	sid := "session-xyz"
	preferred := r.stickyPreferredAlias(sid)

	log := &attemptLog{}

	served, err := dispatchTo(WithSessionID(context.Background(), sid), r, Call{}, log, errors.New("boom"), 0, 1, 2)
	if err != nil {
		t.Fatal(err)
	}

	if r.endpoints[served].Alias != preferred {
		t.Fatalf("routed to %q, want the preferred %q", r.endpoints[served].Alias, preferred)
	}
}
