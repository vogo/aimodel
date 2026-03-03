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
)

const defaultRecoveryInterval = 60 * time.Second

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
}

// ComposeClient dispatches chat requests across multiple model backends.
// It implements aimodel.ChatCompleter and can be nested.
type ComposeClient struct {
	entries          []ModelEntry
	health           []*modelHealth
	strategy         Strategy
	recoveryInterval time.Duration
	nowFunc          func() time.Time
	rng              *rand.Rand
	mu               sync.Mutex // protects rng
}

// ComposeOption configures a ComposeClient.
type ComposeOption func(*ComposeClient)

// WithRecoveryInterval sets the duration after which an errored model
// becomes eligible for a recovery probe.
func WithRecoveryInterval(d time.Duration) ComposeOption {
	return func(c *ComposeClient) {
		c.recoveryInterval = d
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

	health := make([]*modelHealth, len(entries))
	for i := range health {
		health[i] = newModelHealth()
	}

	c := &ComposeClient{
		entries:          entries,
		health:           health,
		strategy:         strategy,
		recoveryInterval: defaultRecoveryInterval,
		nowFunc:          time.Now,
		rng:              newRand(time.Now().UnixNano()),
	}

	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// ChatCompletion sends a non-streaming request, routing via the configured strategy.
// Protocol routing is handled internally by each entry's Client.
func (c *ComposeClient) ChatCompletion(ctx context.Context, req *aimodel.ChatRequest) (*aimodel.ChatResponse, error) {
	return dispatchUnary(ctx, c, req, func(ctx context.Context, client aimodel.ChatCompleter, r *aimodel.ChatRequest) (*aimodel.ChatResponse, error) {
		return client.ChatCompletion(ctx, r)
	})
}

// ChatCompletionStream sends a streaming request, routing via the configured strategy.
// Protocol routing is handled internally by each entry's Client.
func (c *ComposeClient) ChatCompletionStream(ctx context.Context, req *aimodel.ChatRequest) (*aimodel.Stream, error) {
	return dispatchUnary(ctx, c, req, func(ctx context.Context, client aimodel.ChatCompleter, r *aimodel.ChatRequest) (*aimodel.Stream, error) {
		return client.ChatCompletionStream(ctx, r)
	})
}

// dispatchUnary is the generic dispatch loop shared by all public methods.
func dispatchUnary[T any](
	ctx context.Context,
	c *ComposeClient,
	req *aimodel.ChatRequest,
	call func(context.Context, aimodel.ChatCompleter, *aimodel.ChatRequest) (T, error),
) (T, error) {
	var zero T

	candidates := c.selectModels()

	// Recovery probe: prepend errored models that are eligible for probing.
	candidates = c.prependRecoveryProbes(candidates)

	if len(candidates) == 0 {
		return zero, aimodel.ErrNoActiveModels
	}

	var errs []aimodel.ModelError

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
			// Do not poison model health on context cancellation.
			if ctx.Err() != nil {
				return zero, ctx.Err()
			}

			c.health[idx].markError(err, c.nowFunc())
			errs = append(errs, aimodel.ModelError{Model: entry.Name, Err: err})

			continue
		}

		c.health[idx].markActive()

		return result, nil
	}

	return zero, &aimodel.MultiError{Errors: errs}
}

// prependRecoveryProbes prepends errored models that are eligible for recovery probing
// to the candidate list.
func (c *ComposeClient) prependRecoveryProbes(candidates []int) []int {
	now := c.nowFunc()

	// Collect the set of already-active candidates for quick lookup.
	active := make(map[int]bool, len(candidates))
	for _, idx := range candidates {
		active[idx] = true
	}

	var probes []int

	for i := range c.entries {
		if active[i] {
			continue
		}

		if c.health[i].shouldProbe(now, c.recoveryInterval) {
			probes = append(probes, i)
		}
	}

	if len(probes) == 0 {
		return candidates
	}

	return append(probes, candidates...)
}

// Compile-time check: ComposeClient implements aimodel.ChatCompleter.
var _ aimodel.ChatCompleter = (*ComposeClient)(nil)
