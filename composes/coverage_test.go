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
	"strings"
	"testing"

	"github.com/vogo/aimodel/provider/openai"
)

// blockingCompleter blocks in ChatCompletion until the context is cancelled,
// signalling via started once the call is in flight. It returns ctx.Err(),
// deterministically exercising the mid-flight cancellation path.
type blockingCompleter struct {
	started chan struct{}
}

func (b blockingCompleter) ChatCompletions(ctx context.Context, _ *openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
	close(b.started)
	<-ctx.Done()

	return nil, ctx.Err()
}

func (b blockingCompleter) ChatCompletionsStream(ctx context.Context, _ *openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error) {
	<-ctx.Done()

	return nil, ctx.Err()
}

// TestCancellation_MidFlightAttributesAlias covers the path where a request is
// already in flight when the context is cancelled: the attempt is attributed to
// its alias, health is not poisoned, and ctx.Err() is returned.
func TestCancellation_MidFlightAttributesAlias(t *testing.T) {
	obs := &recordingObserver{}

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "slow", Client: blockingCompleter{started: make(chan struct{})}},
	}, WithAttemptObserver(obs.fn))
	if err != nil {
		t.Fatal(err)
	}

	started := cc.entries[0].Client.(blockingCompleter).started

	ctx, cancel := context.WithCancel(context.Background())

	errCh := make(chan error, 1)

	go func() {
		_, err := cc.ChatCompletions(ctx, testRequest())
		errCh <- err
	}()

	<-started // the attempt is genuinely in flight, past the pre-call ctx check
	cancel()

	err = <-errCh
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	// The in-flight attempt was still attributed to its alias.
	if len(obs.results) != 1 || obs.results[0].Alias != "slow" || obs.results[0].Success {
		t.Fatalf("observations = %+v, want one failed attempt for slow", obs.results)
	}

	// Cancellation must not poison health.
	if !cc.health[0].isActive() {
		t.Fatal("health should remain active after mid-flight cancellation")
	}
}

// TestCapabilityError_VisionRequired covers the vision branch of the capability
// error path (a vision request with no capable endpoint).
func TestCapabilityError_VisionRequired(t *testing.T) {
	s := newTestServer(t)
	defer s.Close()

	cc, err := NewComposeClient(StrategyFailover, []ModelEntry{
		{Name: "m0", Alias: "text-only", Capability: &Capability{Vision: false}, Client: newClientForServer(t, s)},
	})
	if err != nil {
		t.Fatal(err)
	}

	_, err = cc.ChatCompletions(context.Background(), visionRequest())

	var capErr *CapabilityError
	if !errors.As(err, &capErr) {
		t.Fatalf("expected CapabilityError, got %v", err)
	}

	if len(capErr.Required) != 1 || capErr.Required[0] != "vision" {
		t.Fatalf("required = %v, want [vision]", capErr.Required)
	}

	// The error string names the capability and the considered aliases.
	if !strings.Contains(capErr.Error(), "vision") || !strings.Contains(capErr.Error(), "text-only") {
		t.Fatalf("error string missing detail: %q", capErr.Error())
	}
}

// TestCostStrategy_OutputCapAffectsKey exercises costKey's MaxCompletionTokens
// and MaxTokens branches by showing the output cap change the ranking.
func TestCostStrategy_OutputCapAffectsKey(t *testing.T) {
	entries := []ModelEntry{
		{Name: "m0", Alias: "input-heavy", Cost: &EndpointCost{InputPrice: 100, OutputPrice: 0}},
		{Name: "m1", Alias: "output-heavy", Cost: &EndpointCost{InputPrice: 0, OutputPrice: 1}},
	}
	c := newTestComposeClient(StrategyCost, entries)

	// No output cap (units=1): input-heavy=100, output-heavy=1 → output-heavy first.
	assertIntSlice(t, selectReq(c, testRequest()), []int{1, 0})

	// Large output cap via MaxCompletionTokens (units=1000): input-heavy=100,
	// output-heavy=1000 → the order flips.
	big := testRequest()
	max := 1000
	big.MaxCompletionTokens = &max

	assertIntSlice(t, selectReq(c, big), []int{0, 1})

	// The legacy MaxTokens fallback drives the same branch and the same flip.
	legacy := testRequest()
	legacy.MaxTokens = &max //nolint:staticcheck // intentionally exercising the legacy fallback

	assertIntSlice(t, selectReq(c, legacy), []int{0, 1})
}

// TestErrorStrings covers the Error() formatting of EndpointError and
// MultiError, including the empty MultiError case.
func TestErrorStrings(t *testing.T) {
	ee := &EndpointError{Alias: "ep", Err: errors.New("boom")}
	if !strings.Contains(ee.Error(), "ep") || !strings.Contains(ee.Error(), "boom") {
		t.Fatalf("EndpointError.Error() = %q", ee.Error())
	}

	me := &MultiError{Errors: []*EndpointError{
		{Alias: "a", Err: errors.New("x")},
		{Alias: "b", Err: errors.New("y")},
	}}

	msg := me.Error()
	if !strings.Contains(msg, "a: x") || !strings.Contains(msg, "b: y") {
		t.Fatalf("MultiError.Error() = %q", msg)
	}

	empty := &MultiError{}
	if empty.Error() == "" {
		t.Fatal("empty MultiError.Error() should not be empty")
	}
}

// TestSticky_NilContext covers the nil-context guard in sessionIDFromContext.
func TestSticky_NilContext(t *testing.T) {
	if id := sessionIDFromContext(nil); id != "" { //nolint:staticcheck // deliberately exercising the nil-context guard
		t.Fatalf("nil context session id = %q, want empty", id)
	}
}

// TestSticky_FallbackCoercionAtRuntime covers the defensive coercion inside
// selectSticky when the fallback is (impossibly) sticky.
func TestSticky_FallbackCoercionAtRuntime(t *testing.T) {
	c := stickyClient("a", "b")
	c.stickyFallback = StrategySticky // bypass the option-time coercion

	// No session id: must coerce to failover (definition order), not recurse.
	assertIntSlice(t, selectWithSession(c, ""), []int{0, 1})
}

// TestSticky_NoAvailableCandidates covers the empty-available guard when a
// session id is present but every endpoint is unhealthy.
func TestSticky_NoAvailableCandidates(t *testing.T) {
	c := stickyClient("a", "b")

	now := c.nowFunc()
	c.health[0].markError(errors.New("down"), now)
	c.health[1].markError(errors.New("down"), now)

	if got := selectWithSession(c, "user-1"); len(got) != 0 {
		t.Fatalf("expected no candidates when all unhealthy, got %v", got)
	}
}
