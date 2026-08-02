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
	"fmt"
	"maps"
	"strings"
	"time"

	"github.com/vogo/aimodel"
)

// EndpointSpec declaratively describes one same-protocol endpoint: the provider
// (a registry protocol name, never an alias), the connection coordinates, the
// model name sent to the backend, and the endpoint's operational identity and
// routing metadata. NewFromEndpoints builds an independent aimodel.Client per
// spec, so N endpoints no longer require N copies of construction code.
//
// Provider is the protocol resolved from the registry; Alias is the operational
// identity of the constructed instance. The two are not interchangeable, and
// multiple endpoints never share or forge a provider name.
type EndpointSpec struct {
	// Provider is the registered protocol name (e.g. openai.Name,
	// anthropic.Name). Empty selects the root client's default provider.
	Provider string
	// BaseURL is the endpoint's API base URL. It always overrides any
	// environment fallback, so instances never share global credentials.
	BaseURL string
	// APIKey is the endpoint's credential. It always overrides any environment
	// fallback.
	APIKey string
	// Model is the model name sent in ChatRequest.Model for this endpoint.
	Model string
	// Alias is the required, unique operational identity used for health
	// snapshots, sticky routing, and error attribution.
	Alias string
	// Weight is used by StrategyWeight; Weight <= 0 counts as 1.
	Weight int
	// Tags carry operational attributes (region/tier/workspace). They are
	// copied onto the entry for observation and later strategies; they never
	// participate in capability decisions.
	Tags map[string]string

	// Capability optionally declares the endpoint's strong-typed capability.
	// Leaving it nil keeps the endpoint out of the capability filter entirely
	// (unknown, not incapable).
	Capability *Capability
	// Cost optionally declares static pricing for StrategyCost.
	Cost *EndpointCost
	// Latency optionally declares a routing latency for StrategyLatency.
	Latency *time.Duration
}

// EndpointError wraps a single endpoint's attempt failure, attributing it to a
// stable alias. It unwraps to the underlying error, so errors.Is/As reach the
// original APIError (or any other cause).
type EndpointError struct {
	// Alias is the operational identity of the endpoint that failed.
	Alias string
	// Err is the underlying attempt error.
	Err error
}

func (e *EndpointError) Error() string {
	return fmt.Sprintf("aimodel/composes: endpoint %s: %v", e.Alias, e.Err)
}

func (e *EndpointError) Unwrap() error {
	return e.Err
}

// MultiError aggregates every endpoint failure from one dispatch, in attempt
// order. It is composes-owned: same-model endpoints are distinguished by alias
// (via EndpointError), not by ais.ModelError. It implements Go 1.20+ multi-error
// unwrapping so errors.Is/As match any underlying endpoint error.
type MultiError struct {
	Errors []*EndpointError
}

func (e *MultiError) Error() string {
	if len(e.Errors) == 0 {
		return "aimodel/composes: all endpoints failed"
	}

	var b strings.Builder

	b.WriteString("aimodel/composes: all endpoints failed: ")

	for i, ee := range e.Errors {
		if i > 0 {
			b.WriteString("; ")
		}

		fmt.Fprintf(&b, "%s: %v", ee.Alias, ee.Err)
	}

	return b.String()
}

// Unwrap returns the endpoint errors for Go 1.20+ multi-error unwrapping.
func (e *MultiError) Unwrap() []error {
	errs := make([]error, len(e.Errors))
	for i := range e.Errors {
		errs[i] = e.Errors[i]
	}

	return errs
}

// AttemptResult reports the outcome of a single endpoint attempt. It is emitted
// to the observer registered via WithAttemptObserver when each attempt finishes:
// for non-streaming calls, when the call returns; for streaming calls, when the
// stream is established or fails to establish. Post-establishment SSE errors are
// surfaced by the Stream itself, not re-reported here.
type AttemptResult struct {
	// Alias is the endpoint that was attempted.
	Alias string
	// Success reports whether the attempt succeeded (stream established for
	// streaming calls).
	Success bool
	// Err is the attempt error, nil on success.
	Err error
	// Stream reports whether the attempt was a streaming call.
	Stream bool
}

// NewFromEndpoints builds a ComposeClient from declarative endpoint specs. Each
// spec is turned into an independent aimodel.Client (via the root NewClient with
// the spec's provider, BaseURL, APIKey, and default-model options) and wrapped
// in a ModelEntry. Explicit endpoint coordinates always override environment
// fallbacks. Advanced callers needing provider-private options or a custom
// ChatCompleter keep building ModelEntry by hand via NewComposeClient.
func NewFromEndpoints(strategy Strategy, specs []EndpointSpec, opts ...ComposeOption) (*ComposeClient, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("aimodel/composes: at least one endpoint spec is required")
	}

	// Validate aliases up front so the error names the offending position
	// before any client is constructed.
	seen := make(map[string]int, len(specs))

	for i, s := range specs {
		if s.Alias == "" {
			return nil, fmt.Errorf("aimodel/composes: endpoint %d: alias is required", i)
		}

		if prev, dup := seen[s.Alias]; dup {
			return nil, fmt.Errorf("aimodel/composes: duplicate alias %q at endpoints %d and %d", s.Alias, prev, i)
		}

		seen[s.Alias] = i
	}

	entries := make([]ModelEntry, len(specs))

	for i, s := range specs {
		var clientOpts []aimodel.Option

		if s.Provider != "" {
			clientOpts = append(clientOpts, aimodel.WithProvider(s.Provider))
		}

		// APIKey and BaseURL are always applied — even when empty — so an
		// explicit spec overrides any environment fallback. This keeps instances
		// self-contained: two endpoints never accidentally share the global
		// AI_API_KEY / OPENAI_BASE_URL fallback. An empty value therefore fails
		// construction with a clear, per-endpoint error rather than silently
		// inheriting global credentials.
		clientOpts = append(clientOpts, aimodel.WithAPIKey(s.APIKey), aimodel.WithBaseURL(s.BaseURL))

		if s.Model != "" {
			clientOpts = append(clientOpts, aimodel.WithDefaultModel(s.Model))
		}

		client, err := aimodel.NewClient(clientOpts...)
		if err != nil {
			return nil, fmt.Errorf("aimodel/composes: endpoint %d (alias %q): %w", i, s.Alias, err)
		}

		tags := make(map[string]string, len(s.Tags))
		maps.Copy(tags, s.Tags)

		entries[i] = ModelEntry{
			Name:       s.Model,
			Client:     client,
			Weight:     s.Weight,
			Alias:      s.Alias,
			Tags:       tags,
			Capability: s.Capability,
			Cost:       s.Cost,
			Latency:    s.Latency,
		}
	}

	return NewComposeClient(strategy, entries, opts...)
}
