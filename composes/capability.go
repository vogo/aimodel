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
	"fmt"
	"sort"
	"strings"

	"github.com/vogo/aimodel/ais"
)

// Capability is the strong-typed contract an endpoint exposes to the router.
// It is declared by the endpoint (via ModelEntry.Capability / EndpointSpec
// .Capability) or, when a wrapped client implements CapabilityProvider, by the
// client itself. It is never inferred from Tags or the provider name, and it
// drives candidate filtering only — the router excludes incapable endpoints but
// never strips tools, rewrites the request, switches dialect, or downgrades.
//
// Filtering is opt-in: an endpoint that declares nothing is *unknown*, not
// *incapable*, and never participates in the filter. Only an explicit
// declaration — a non-nil ModelEntry.Capability or a CapabilityProvider client —
// can exclude an endpoint.
type Capability struct {
	// Tools marks the endpoint able to serve function/tool calls.
	Tools bool
	// Vision marks the endpoint able to accept image content parts.
	Vision bool
	// MaxContextTokens is the endpoint's context window. It is part of the
	// contract for future token-budget routing; this SDK does not estimate
	// token counts, so it is not used for filtering today.
	MaxContextTokens int
}

// CapabilityProvider is optionally implemented by a ChatCompleter (e.g. a
// provider's native client) that can declare its own Capability. A non-nil
// ModelEntry.Capability always overrides this declaration. Implementing it is
// itself a declaration: the returned Capability participates in filtering, so a
// zero value means "supports neither tools nor vision", not "unknown".
type CapabilityProvider interface {
	ComposeCapability() Capability
}

// EndpointCost carries static per-unit pricing used by StrategyCost. Dynamic
// pricing is out of scope; these are fixed routing metadata.
type EndpointCost struct {
	// InputPrice is the cost per input unit (e.g. per 1M input tokens).
	InputPrice float64
	// OutputPrice is the cost per output unit (e.g. per 1M output tokens).
	OutputPrice float64
}

// ErrCapabilityNotSatisfied reports that no endpoint can serve a requested
// capability. It is returned before any network I/O. Match with errors.Is; the
// wrapping CapabilityError names the required capabilities.
var ErrCapabilityNotSatisfied = errors.New("aimodel/composes: no endpoint satisfies the required capabilities")

// CapabilityError names the capabilities a request required and the endpoint
// aliases that were considered. It never triggers a downgrade.
type CapabilityError struct {
	// Required lists the unsatisfied capability names (e.g. "tools", "vision").
	Required []string
	// Considered lists the aliases of the endpoints that were evaluated.
	Considered []string
}

func (e *CapabilityError) Error() string {
	return fmt.Sprintf(
		"aimodel/composes: no endpoint satisfies required capabilities [%s] (considered: %s)",
		strings.Join(e.Required, ", "),
		strings.Join(e.Considered, ", "),
	)
}

func (e *CapabilityError) Unwrap() error {
	return ErrCapabilityNotSatisfied
}

// resolvedCapability returns the effective capability for entry i and whether it
// was declared at all: an explicit ModelEntry.Capability wins; otherwise a client
// implementing CapabilityProvider declares it. With neither, the capability is
// *unknown* (declared == false) — the entry is left out of the filter rather than
// assumed incapable, so configurations that never declare a Capability route
// exactly as they did before capability filtering existed.
func (c *ComposeClient) resolvedCapability(i int) (Capability, bool) {
	if cap := c.entries[i].Capability; cap != nil {
		return *cap, true
	}

	if cp, ok := c.entries[i].Client.(CapabilityProvider); ok {
		return cp.ComposeCapability(), true
	}

	return Capability{}, false
}

// requestRequiresTools reports whether the request needs a tools-capable
// endpoint: it defines tools, or sets an explicit tool_choice other than "none".
func requestRequiresTools(req *ais.ChatRequest) bool {
	if len(req.Tools) > 0 {
		return true
	}

	return req.ToolChoice != nil && req.ToolChoice != "none"
}

// requestRequiresVision reports whether any message carries an image part.
func requestRequiresVision(req *ais.ChatRequest) bool {
	for i := range req.Messages {
		for _, part := range req.Messages[i].Content.Parts() {
			if part.ImageURL != nil || part.Type == "image_url" {
				return true
			}
		}
	}

	return false
}

// requiredCapabilities returns the capability names the request demands.
func requiredCapabilities(req *ais.ChatRequest) []string {
	var reqs []string

	if requestRequiresTools(req) {
		reqs = append(reqs, "tools")
	}

	if requestRequiresVision(req) {
		reqs = append(reqs, "vision")
	}

	return reqs
}

// capableIndices returns the entry indices whose capability satisfies the
// request, in definition order. Endpoints that declare no capability are
// unknown, not incapable: they always stay in the candidate list. Health is
// intentionally ignored here — the capability filter runs before health skipping
// and strategy ordering.
func (c *ComposeClient) capableIndices(req *ais.ChatRequest) []int {
	needTools := requestRequiresTools(req)
	needVision := requestRequiresVision(req)

	out := make([]int, 0, len(c.entries))

	for i := range c.entries {
		cap, declared := c.resolvedCapability(i)
		if declared {
			if needTools && !cap.Tools {
				continue
			}

			if needVision && !cap.Vision {
				continue
			}
		}

		out = append(out, i)
	}

	return out
}

// costKey computes the deterministic routing cost for an entry against a
// request. Token estimation is out of scope, so the input volume is a fixed
// unit and the output volume is the request's output cap when present; both are
// constant across endpoints for a given request, so ordering reduces to the
// injected static pricing.
func costKey(e *ModelEntry, req *ais.ChatRequest) float64 {
	const inputUnits = 1.0

	outputUnits := 1.0
	if req.MaxCompletionTokens != nil {
		outputUnits = float64(*req.MaxCompletionTokens)
	} else if req.MaxTokens != nil { //nolint:staticcheck // fallback for pre-max_completion_tokens models
		outputUnits = float64(*req.MaxTokens) //nolint:staticcheck // see above
	}

	return e.Cost.InputPrice*inputUnits + e.Cost.OutputPrice*outputUnits
}

// sortByCost orders candidate indices by ascending static cost. Endpoints
// without pricing metadata sort after priced ones; equal keys tie-break on
// alias so identical inputs always yield an identical order.
func (c *ComposeClient) sortByCost(indices []int, req *ais.ChatRequest) []int {
	sort.Slice(indices, func(a, b int) bool {
		ea, eb := &c.entries[indices[a]], &c.entries[indices[b]]

		ca, cb := ea.Cost != nil, eb.Cost != nil
		if ca != cb {
			return ca // priced endpoints come first
		}

		if ca { // both priced
			ka, kb := costKey(ea, req), costKey(eb, req)
			if ka != kb {
				return ka < kb
			}
		}

		return ea.Alias < eb.Alias
	})

	return indices
}

// sortByLatency orders candidate indices by ascending injected latency.
// Endpoints without a latency value sort after those with one; equal values
// tie-break on alias.
func (c *ComposeClient) sortByLatency(indices []int) []int {
	sort.Slice(indices, func(a, b int) bool {
		ea, eb := &c.entries[indices[a]], &c.entries[indices[b]]

		la, lb := ea.Latency != nil, eb.Latency != nil
		if la != lb {
			return la // endpoints with latency data come first
		}

		if la && *ea.Latency != *eb.Latency {
			return *ea.Latency < *eb.Latency
		}

		return ea.Alias < eb.Alias
	})

	return indices
}
