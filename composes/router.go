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
	"fmt"
	"math/rand"
	"slices"
	"sync"
	"time"
)

const (
	defaultRecoveryInterval = 60 * time.Second
	// defaultCoolingInterval is the default rate-limit (429) cooling duration.
	// It is deliberately shorter than the default recovery interval so a
	// rate-limited endpoint rejoins rotation quickly without a backoff probe.
	defaultCoolingInterval = 10 * time.Second
)

// Router is the protocol-neutral routing core: it owns endpoint identity,
// selection strategies, health state and failure attribution, and nothing else.
// It never sees a request or a response — a wrapper package binds the types and
// hands [Dispatch] a closure that performs one attempt against one endpoint.
//
// That boundary is the whole point: routing *mechanism* is shared, protocol
// *semantics* are not. See doc/adr/0008-shared-routing-core-across-protocol-wrappers.md.
type Router struct {
	endpoints        []Endpoint
	health           []*endpointHealth
	strategy         Strategy
	stickyFallback   Strategy
	recoveryInterval time.Duration
	coolingInterval  time.Duration
	attemptObservers []func(AttemptResult)
	nowFunc          func() time.Time
	rng              *rand.Rand
	mu               sync.Mutex // protects rng
}

// Option configures a Router. The same options serve every wrapper package, so
// a pool's operational behaviour is described once regardless of the protocol
// its endpoints speak.
type Option func(*Router)

// WithRecoveryInterval sets the duration after which an errored endpoint
// becomes eligible for a recovery probe.
func WithRecoveryInterval(d time.Duration) Option {
	return func(r *Router) {
		r.recoveryInterval = d
	}
}

// WithCoolingInterval sets how long a rate-limited (429) endpoint sleeps before
// rejoining regular rotation. It has no effect on the 5xx error backoff.
func WithCoolingInterval(d time.Duration) Option {
	return func(r *Router) {
		r.coolingInterval = d
	}
}

// WithStickyFallback sets the strategy StrategySticky falls back to when a
// request carries no session id. A sticky fallback is coerced to failover to
// avoid recursion. Defaults to StrategyFailover.
func WithStickyFallback(s Strategy) Option {
	return func(r *Router) {
		if s == StrategySticky {
			s = StrategyFailover
		}

		r.stickyFallback = s
	}
}

// WithAttemptObserver registers a callback invoked when each endpoint attempt
// finishes (see AttemptResult). Multiple observers may be registered. Observers
// run synchronously on the request path under no internal lock: they must be
// fast and must not call blocking Router operations.
func WithAttemptObserver(fn func(AttemptResult)) Option {
	return func(r *Router) {
		if fn != nil {
			r.attemptObservers = append(r.attemptObservers, fn)
		}
	}
}

// NewRouter creates a Router over the given endpoints, in the order they are
// declared. The index of an endpoint in this slice is its identity for the rest
// of its life: [Dispatch] hands that index back to the attempt closure, which is
// how a wrapper finds the client and the model name that belong to it.
//
// Endpoints are copied before aliases are resolved, so the caller's slice and
// its elements are never written back to.
func NewRouter(strategy Strategy, endpoints []Endpoint, opts ...Option) (*Router, error) {
	if len(endpoints) == 0 {
		return nil, fmt.Errorf("aimodel/composes: at least one endpoint is required")
	}

	owned := slices.Clone(endpoints)

	if err := resolveAliases(owned); err != nil {
		return nil, err
	}

	health := make([]*endpointHealth, len(owned))
	for i := range health {
		health[i] = newEndpointHealth()
	}

	r := &Router{
		endpoints:        owned,
		health:           health,
		strategy:         strategy,
		stickyFallback:   StrategyFailover,
		recoveryInterval: defaultRecoveryInterval,
		coolingInterval:  defaultCoolingInterval,
		nowFunc:          time.Now,
		rng:              newRand(time.Now().UnixNano()),
	}

	for _, opt := range opts {
		opt(r)
	}

	return r, nil
}

// resolveAliases guarantees every endpoint has a non-empty, unique alias so all
// health snapshots and errors are stably addressable. An empty alias becomes
// "entry-<index>"; a wrapper that derives readable aliases from its own metadata
// does so before calling NewRouter and leaves the rest empty. Explicit aliases
// must not collide with any other alias, derived or explicit. It writes to the
// slice it is given, which is always the router's own copy — never the caller's.
func resolveAliases(endpoints []Endpoint) error {
	seen := make(map[string]int, len(endpoints))

	for i := range endpoints {
		alias := endpoints[i].Alias
		if alias == "" {
			alias = fmt.Sprintf("entry-%d", i)
			endpoints[i].Alias = alias
		}

		if prev, dup := seen[alias]; dup {
			return fmt.Errorf("aimodel/composes: duplicate alias %q at endpoints %d and %d", alias, prev, i)
		}

		seen[alias] = i
	}

	return nil
}

// Call describes one dispatch to the router in neutral terms. A wrapper derives
// every field from its own native request; none of them carries protocol
// meaning here.
type Call struct {
	// Requires lists the opaque labels an endpoint must declare to serve this
	// call. The router compares them as strings against Endpoint.Declares and
	// attaches no meaning to either side. An endpoint that declares nothing is
	// unknown, not incapable, and is never excluded.
	Requires []string

	// Eligible optionally restricts routing to these endpoint indices. Nil means
	// every endpoint takes part. It exists for facts about the caller's backend
	// object rather than about the protocol — for example a wrapper whose entry
	// clients do not all implement the method set this call needs.
	Eligible []int

	// OutputUnits scales EndpointCost.OutputPrice when ordering by StrategyCost.
	// A wrapper sets it from whatever its protocol calls an output cap; zero or
	// negative counts as one unit. It never reaches any endpoint.
	OutputUnits float64

	// Stream reports whether the attempt establishes a stream. It is carried
	// through to AttemptResult for observation and changes no routing decision.
	Stream bool
}

// Dispatch runs one full candidate loop and returns the first successful
// attempt's value. The routing chain is: capability filter → strategy ordering
// → recovery probes → per-endpoint attempts with health and observation updates.
//
// attempt is invoked with the index of the endpoint to try; it owns everything
// protocol-shaped — building the per-endpoint request, calling the backend, and
// returning its value. The router only learns whether the attempt failed, and
// classifies that failure structurally (see classifyHealth).
//
// Failures are collected in attempt order and returned as a *MultiError; an
// empty candidate list yields ErrNoActiveModels; an unsatisfiable Requires set
// yields a *CapabilityError before any attempt is made.
func Dispatch[T any](
	ctx context.Context,
	r *Router,
	call Call,
	attempt func(ctx context.Context, endpoint int) (T, error),
) (T, error) {
	var zero T

	// 1. Capability filter runs before health and strategy. A call whose
	// required labels no endpoint declares fails fast, before any attempt.
	capable := r.capableIndices(call)
	if len(call.Requires) > 0 && len(capable) == 0 {
		return zero, &CapabilityError{Required: slices.Clone(call.Requires), Considered: r.Aliases()}
	}

	// 2. Strategy ordering over the capable, health-available candidates.
	candidates := r.selectEndpoints(ctx, call, capable)

	// 3. Prepend recovery probes for capable endpoints whose backoff elapsed.
	candidates = r.prependRecoveryProbes(candidates, capable)

	if len(candidates) == 0 {
		return zero, ErrNoActiveModels
	}

	var errs []*EndpointError

	for _, idx := range candidates {
		// Return immediately if the context is cancelled to avoid marking
		// healthy endpoints as errored due to client-side cancellation.
		if ctx.Err() != nil {
			return zero, ctx.Err()
		}

		alias := r.endpoints[idx].Alias

		result, err := attempt(ctx, idx)
		if err != nil {
			// Cancellation never poisons health; still attribute the alias if an
			// attempt was made.
			if ctx.Err() != nil {
				r.observe(AttemptResult{Alias: alias, Success: false, Err: err, Stream: call.Stream})
				return zero, ctx.Err()
			}

			r.observe(AttemptResult{Alias: alias, Success: false, Err: err, Stream: call.Stream})
			r.applyFailure(idx, err)

			errs = append(errs, &EndpointError{Alias: alias, Err: err})

			continue
		}

		r.observe(AttemptResult{Alias: alias, Success: true, Stream: call.Stream})
		r.health[idx].markActive()

		return result, nil
	}

	return zero, &MultiError{Errors: errs}
}

// applyFailure updates endpoint health according to the failure classification.
// Request failures (non-429 4xx) are attributed and failed over but leave the
// endpoint healthy.
func (r *Router) applyFailure(idx int, err error) {
	now := r.nowFunc()

	switch classifyHealth(err) {
	case outcomeCooling:
		r.health[idx].markCooling(err, now)
	case outcomeError:
		r.health[idx].markError(err, now)
	case outcomeRequestFailure:
		// Attributed and failed over, but the endpoint is not judged unhealthy.
	}
}

// observe fans an attempt result out to every registered observer. It holds no
// internal lock; observers must be fast and non-blocking.
func (r *Router) observe(result AttemptResult) {
	for _, fn := range r.attemptObservers {
		fn(result)
	}
}

// Aliases returns every endpoint alias in declaration order. A wrapper uses it
// to name the endpoints it considered in an error it raises itself, before
// dispatch begins.
func (r *Router) Aliases() []string {
	aliases := make([]string, len(r.endpoints))
	for i := range r.endpoints {
		aliases[i] = r.endpoints[i].Alias
	}

	return aliases
}

// Stats returns an immutable per-endpoint health snapshot, safe to call
// concurrently with dispatch. Mutating the returned slice does not affect
// internal state.
//
// Status reflects routing behavior, not just the stored state: a cooling
// endpoint whose interval has elapsed is already selectable again, so it is
// reported as "active" rather than staying "cooling" until the next success.
func (r *Router) Stats() []EndpointStat {
	now := r.nowFunc()
	stats := make([]EndpointStat, len(r.endpoints))

	for i := range r.endpoints {
		snap := r.health[i].snapshot()

		state := snap.state
		if state == stateCooling && now.Sub(snap.errorTime) >= r.coolingInterval {
			state = stateActive
		}

		stats[i] = EndpointStat{
			Alias:      r.endpoints[i].Alias,
			Status:     string(state),
			ErrorCount: snap.errorCount,
			LastError:  snap.lastError,
			ErrorTime:  snap.errorTime,
		}
	}

	return stats
}

// prependRecoveryProbes prepends errored, capable endpoints that are eligible
// for recovery probing to the candidate list. Cooling endpoints rejoin rotation
// on their own timer and are not probed here.
func (r *Router) prependRecoveryProbes(candidates, capable []int) []int {
	now := r.nowFunc()

	// Collect the set of already-selected candidates for quick lookup.
	selected := make(map[int]bool, len(candidates))
	for _, idx := range candidates {
		selected[idx] = true
	}

	var probes []int

	for _, idx := range capable {
		if selected[idx] {
			continue
		}

		if r.health[idx].shouldProbe(now, r.recoveryInterval) {
			probes = append(probes, idx)
		}
	}

	if len(probes) == 0 {
		return candidates
	}

	return append(probes, candidates...)
}
