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
	"sync"
	"time"

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/ais"
)

const (
	defaultRecoveryInterval = 60 * time.Second
	// defaultCoolingInterval is the default rate-limit (429) cooling duration.
	// It is deliberately shorter than the default recovery interval so a
	// rate-limited endpoint rejoins rotation quickly without a backoff probe.
	defaultCoolingInterval = 10 * time.Second
)

// ModelEntry describes a single model backend in the compose client.
type ModelEntry struct {
	// Name is the model identifier sent in ChatRequest.Model.
	// If empty, the underlying client's default model is used.
	Name string
	// Client is the underlying API client for this model.
	// Protocol routing is handled internally by each Client.
	Client aimodel.ChatCompleter
	// Weight is used by StrategyWeight. Zero is treated as 1.
	Weight int

	// Alias is the endpoint's operational identity, used for health snapshots,
	// sticky routing, and error attribution. It is distinct from Name (the model
	// sent to the backend) and from the provider (the registry protocol name).
	// When empty on a hand-built entry, a stable "entry-<index>" alias is
	// derived; explicit aliases must be unique across all entries.
	Alias string

	// Tags carry operational attributes (region/tier/workspace) for observation
	// and future strategies. They never participate in capability decisions.
	Tags map[string]string

	// Capability optionally declares the endpoint's strong-typed capability. A
	// nil value falls back to the client's CapabilityProvider declaration, then
	// to the conservative zero capability.
	Capability *Capability

	// Cost optionally declares static pricing for StrategyCost. Nil sorts after
	// priced endpoints.
	Cost *EndpointCost

	// Latency optionally declares a routing latency for StrategyLatency. Nil
	// sorts after endpoints that carry a latency.
	Latency *time.Duration
}

// EndpointStat is an immutable per-endpoint health snapshot returned by Stats().
type EndpointStat struct {
	// Alias is the endpoint's operational identity.
	Alias string
	// Status is "active", "cooling", or "error".
	Status string
	// ErrorCount is the consecutive health-failure count (5xx/transport); it
	// does not advance on cooling or request failures.
	ErrorCount int
	// LastError is the most recent health-relevant error, or nil if the
	// endpoint has never failed or has since recovered.
	LastError error
	// ErrorTime is when LastError occurred; the zero value means never/Recovered.
	ErrorTime time.Time
}

// ComposeClient dispatches chat requests across multiple model backends.
// It implements aimodel.ChatCompleter and can be nested.
type ComposeClient struct {
	entries          []ModelEntry
	health           []*modelHealth
	strategy         Strategy
	stickyFallback   Strategy
	recoveryInterval time.Duration
	coolingInterval  time.Duration
	attemptObservers []func(AttemptResult)
	nowFunc          func() time.Time
	rng              *rand.Rand
	mu               sync.Mutex // protects rng
}

// ComposeOption configures a ComposeClient.
type ComposeOption func(*ComposeClient)

// WithRecoveryInterval sets the duration after which an errored endpoint
// becomes eligible for a recovery probe.
func WithRecoveryInterval(d time.Duration) ComposeOption {
	return func(c *ComposeClient) {
		c.recoveryInterval = d
	}
}

// WithCoolingInterval sets how long a rate-limited (429) endpoint sleeps before
// rejoining regular rotation. It has no effect on the 5xx error backoff.
func WithCoolingInterval(d time.Duration) ComposeOption {
	return func(c *ComposeClient) {
		c.coolingInterval = d
	}
}

// WithStickyFallback sets the strategy StrategySticky falls back to when a
// request carries no session id. A sticky fallback is coerced to failover to
// avoid recursion. Defaults to StrategyFailover.
func WithStickyFallback(s Strategy) ComposeOption {
	return func(c *ComposeClient) {
		if s == StrategySticky {
			s = StrategyFailover
		}

		c.stickyFallback = s
	}
}

// WithAttemptObserver registers a callback invoked when each endpoint attempt
// finishes (see AttemptResult). Multiple observers may be registered. Observers
// run synchronously on the request path under no internal lock: they must be
// fast and must not call blocking ComposeClient operations.
func WithAttemptObserver(fn func(AttemptResult)) ComposeOption {
	return func(c *ComposeClient) {
		if fn != nil {
			c.attemptObservers = append(c.attemptObservers, fn)
		}
	}
}

// NewComposeClient creates a ComposeClient with the given strategy and model entries.
func NewComposeClient(strategy Strategy, entries []ModelEntry, opts ...ComposeOption) (*ComposeClient, error) {
	if len(entries) == 0 {
		return nil, fmt.Errorf("aimodel/composes: at least one model entry is required")
	}

	for i, e := range entries {
		if e.Client == nil {
			return nil, fmt.Errorf("aimodel/composes: entry %d (%q): client is nil", i, e.Name)
		}
	}

	if err := resolveAliases(entries); err != nil {
		return nil, err
	}

	health := make([]*modelHealth, len(entries))
	for i := range health {
		health[i] = newModelHealth()
	}

	c := &ComposeClient{
		entries:          entries,
		health:           health,
		strategy:         strategy,
		stickyFallback:   StrategyFailover,
		recoveryInterval: defaultRecoveryInterval,
		coolingInterval:  defaultCoolingInterval,
		nowFunc:          time.Now,
		rng:              newRand(time.Now().UnixNano()),
	}

	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// resolveAliases guarantees every entry has a non-empty, unique alias so all
// health snapshots and errors are stably addressable. An empty alias is derived
// from the model Name when it is free, otherwise as "entry-<index>" (this keeps
// hand-built entries backward compatible). Explicit aliases must not collide
// with any other alias, derived or explicit.
func resolveAliases(entries []ModelEntry) error {
	seen := make(map[string]int, len(entries))

	for i := range entries {
		alias := entries[i].Alias

		if alias == "" {
			// Prefer the model name for readable attribution; fall back to the
			// index when the name is empty or already taken (e.g. two endpoints
			// for the same model — the canary case).
			if name := entries[i].Name; name != "" {
				if _, taken := seen[name]; !taken {
					alias = name
				}
			}

			if alias == "" {
				alias = fmt.Sprintf("entry-%d", i)
			}

			entries[i].Alias = alias
		}

		if prev, dup := seen[alias]; dup {
			return fmt.Errorf("aimodel/composes: duplicate alias %q at entries %d and %d", alias, prev, i)
		}

		seen[alias] = i
	}

	return nil
}

// ChatCompletion sends a non-streaming request, routing via the configured strategy.
// Protocol routing is handled internally by each entry's Client.
func (c *ComposeClient) ChatCompletion(ctx context.Context, req *ais.ChatRequest) (*ais.ChatResponse, error) {
	return dispatchUnary(ctx, c, req, false, func(ctx context.Context, client aimodel.ChatCompleter, r *ais.ChatRequest) (*ais.ChatResponse, error) {
		return client.ChatCompletion(ctx, r)
	})
}

// ChatCompletionStream sends a streaming request, routing via the configured strategy.
// Protocol routing is handled internally by each entry's Client.
func (c *ComposeClient) ChatCompletionStream(ctx context.Context, req *ais.ChatRequest) (*aimodel.Stream, error) {
	return dispatchUnary(ctx, c, req, true, func(ctx context.Context, client aimodel.ChatCompleter, r *ais.ChatRequest) (*aimodel.Stream, error) {
		return client.ChatCompletionStream(ctx, r)
	})
}

// dispatchUnary is the generic dispatch loop shared by all public methods. The
// routing chain is: capability filter → strategy ordering → recovery probes →
// per-endpoint attempts with health + observation updates.
func dispatchUnary[T any](
	ctx context.Context,
	c *ComposeClient,
	req *ais.ChatRequest,
	stream bool,
	call func(context.Context, aimodel.ChatCompleter, *ais.ChatRequest) (T, error),
) (T, error) {
	var zero T

	// 1. Capability filter runs before health and strategy. A request whose
	// capabilities no endpoint satisfies fails fast, before any network I/O.
	capable := c.capableIndices(req)
	if reqCaps := requiredCapabilities(req); len(reqCaps) > 0 && len(capable) == 0 {
		return zero, &CapabilityError{Required: reqCaps, Considered: c.allAliases()}
	}

	// 2. Strategy ordering over the capable, health-available candidates.
	candidates := c.selectModels(ctx, req, capable)

	// 3. Prepend recovery probes for capable endpoints whose backoff elapsed.
	candidates = c.prependRecoveryProbes(candidates, capable)

	if len(candidates) == 0 {
		return zero, ais.ErrNoActiveModels
	}

	var errs []*EndpointError

	for _, idx := range candidates {
		// Return immediately if the context is cancelled to avoid
		// marking healthy models as errored due to client-side cancellation.
		if ctx.Err() != nil {
			return zero, ctx.Err()
		}

		entry := c.entries[idx]

		// Clone the request and override the model name if specified.
		r := *req
		if entry.Name != "" {
			r.Model = entry.Name
		}

		result, err := call(ctx, entry.Client, &r)
		if err != nil {
			// Cancellation never poisons health; still attribute the alias if an
			// attempt was made.
			if ctx.Err() != nil {
				c.observe(AttemptResult{Alias: entry.Alias, Success: false, Err: err, Stream: stream})
				return zero, ctx.Err()
			}

			c.observe(AttemptResult{Alias: entry.Alias, Success: false, Err: err, Stream: stream})
			c.applyFailure(idx, err)

			errs = append(errs, &EndpointError{Alias: entry.Alias, Err: err})

			continue
		}

		c.observe(AttemptResult{Alias: entry.Alias, Success: true, Stream: stream})
		c.health[idx].markActive()

		return result, nil
	}

	return zero, &MultiError{Errors: errs}
}

// applyFailure updates endpoint health according to the failure classification.
// Request failures (non-429 4xx) are attributed and failed over but leave the
// endpoint healthy.
func (c *ComposeClient) applyFailure(idx int, err error) {
	now := c.nowFunc()

	switch classifyHealth(err) {
	case outcomeCooling:
		c.health[idx].markCooling(err, now)
	case outcomeError:
		c.health[idx].markError(err, now)
	case outcomeRequestFailure:
		// Attributed and failed over, but the endpoint is not judged unhealthy.
	}
}

// observe fans an attempt result out to every registered observer. It holds no
// internal lock; observers must be fast and non-blocking.
func (c *ComposeClient) observe(result AttemptResult) {
	for _, fn := range c.attemptObservers {
		fn(result)
	}
}

// allAliases returns every entry alias in definition order.
func (c *ComposeClient) allAliases() []string {
	aliases := make([]string, len(c.entries))
	for i := range c.entries {
		aliases[i] = c.entries[i].Alias
	}

	return aliases
}

// Stats returns an immutable per-endpoint health snapshot, safe to call
// concurrently with dispatch. Mutating the returned slice does not affect
// internal state.
func (c *ComposeClient) Stats() []EndpointStat {
	stats := make([]EndpointStat, len(c.entries))

	for i := range c.entries {
		snap := c.health[i].snapshot()

		stats[i] = EndpointStat{
			Alias:      c.entries[i].Alias,
			Status:     string(snap.state),
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
func (c *ComposeClient) prependRecoveryProbes(candidates []int, capable []int) []int {
	now := c.nowFunc()

	// Collect the set of already-active candidates for quick lookup.
	active := make(map[int]bool, len(candidates))
	for _, idx := range candidates {
		active[idx] = true
	}

	var probes []int

	for _, idx := range capable {
		if active[idx] {
			continue
		}

		if c.health[idx].shouldProbe(now, c.recoveryInterval) {
			probes = append(probes, idx)
		}
	}

	if len(probes) == 0 {
		return candidates
	}

	return append(probes, candidates...)
}

// Compile-time check: ComposeClient implements aimodel.ChatCompleter.
var _ aimodel.ChatCompleter = (*ComposeClient)(nil)
