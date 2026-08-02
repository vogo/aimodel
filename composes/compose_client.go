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

	"github.com/vogo/aimodel/provider/openai"
)

const defaultRecoveryInterval = 60 * time.Second

// ChatCompleter is the method set a backend must provide to take part in
// dispatch. *openai.Client satisfies it, and so does a ComposeClient, which is
// what makes nesting work.
//
// This package dispatches within one wire format — OpenAI-compatible — rather
// than across protocols (ADR 0007). Composing Anthropic backends is the same
// loop written against that package's client; sharing one here would require a
// request model both protocols agree on, which is the abstraction this SDK
// removed.
type ChatCompleter interface {
	ChatCompletions(ctx context.Context, request *openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	ChatCompletionsStream(ctx context.Context, request *openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
}

// ModelEntry describes a single model backend in the compose client.
type ModelEntry struct {
	// Name is the model identifier sent in ChatCompletionRequest.Model.
	// If empty, the request's own model is left in place.
	Name string
	// Client is the underlying API client for this backend.
	Client ChatCompleter
	// Weight is used by StrategyWeight. Zero is treated as 1.
	Weight int
}

// ComposeClient dispatches chat requests across multiple model backends.
// It implements ChatCompleter and can be nested.
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

// ChatCompletions sends a non-streaming request, routing via the configured
// strategy.
func (c *ComposeClient) ChatCompletions(
	ctx context.Context, request *openai.ChatCompletionRequest,
) (*openai.ChatCompletionResponse, error) {
	return dispatch(ctx, c, request,
		func(ctx context.Context, client ChatCompleter, r *openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error) {
			return client.ChatCompletions(ctx, r)
		})
}

// ChatCompletionsStream sends a streaming request, routing via the configured
// strategy. Only the call that opens the stream is covered by failover: once a
// backend has started streaming, a mid-stream error reaches the caller.
func (c *ComposeClient) ChatCompletionsStream(
	ctx context.Context, request *openai.ChatCompletionRequest,
) (*openai.ChatCompletionStream, error) {
	return dispatch(ctx, c, request,
		func(ctx context.Context, client ChatCompleter, r *openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error) {
			return client.ChatCompletionsStream(ctx, r)
		})
}

// dispatch is the generic dispatch loop shared by all public methods.
func dispatch[T any](
	ctx context.Context,
	c *ComposeClient,
	req *openai.ChatCompletionRequest,
	call func(context.Context, ChatCompleter, *openai.ChatCompletionRequest) (T, error),
) (T, error) {
	var zero T

	candidates := c.selectModels()

	// Recovery probe: prepend errored models that are eligible for probing.
	candidates = c.prependRecoveryProbes(candidates)

	if len(candidates) == 0 {
		return zero, ErrNoActiveModels
	}

	var errs []ModelError

	for _, idx := range candidates {
		// Return immediately if the context is cancelled to avoid
		// marking healthy models as errored due to client-side cancellation.
		if ctx.Err() != nil {
			return zero, ctx.Err()
		}

		entry := c.entries[idx]

		// Copy the request and override the model name if specified, so the
		// caller's request is untouched and each backend sees its own model.
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
			errs = append(errs, ModelError{Model: entry.Name, Err: err})

			continue
		}

		c.health[idx].markActive()

		return result, nil
	}

	return zero, &MultiError{Errors: errs}
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

// Compile-time checks: a ComposeClient is itself a backend, so compose
// clients nest; and the native OpenAI client can be used as one directly.
var (
	_ ChatCompleter = (*ComposeClient)(nil)
	_ ChatCompleter = (*openai.Client)(nil)
)
