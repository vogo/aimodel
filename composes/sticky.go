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
	"hash/fnv"
	"sort"

	"github.com/vogo/aimodel/ais"
)

// sessionIDKey is the context key carrying the sticky-routing session id.
type sessionIDKey struct{}

// WithSessionID attaches a session id used by StrategySticky to pin a request
// stream to a stable endpoint. It lives in composes context, never on
// ais.ChatRequest, so the canonical request type stays untouched.
func WithSessionID(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, sessionIDKey{}, sessionID)
}

// sessionIDFromContext returns the sticky session id, or "" when absent.
func sessionIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}

	id, _ := ctx.Value(sessionIDKey{}).(string)

	return id
}

// stickyPreferredAlias deterministically selects the preferred alias for a
// session id. The hash input is the session id plus the full configured alias
// set (sorted), so the choice is reproducible across processes and instances
// and does not jitter as endpoint health changes. The preferred alias may be
// momentarily unavailable; the caller then fails over in a deterministic order.
func (c *ComposeClient) stickyPreferredAlias(sessionID string) string {
	aliases := make([]string, len(c.entries))
	for i := range c.entries {
		aliases[i] = c.entries[i].Alias
	}

	sort.Strings(aliases)

	h := fnv.New32a()
	h.Write([]byte(sessionID))
	h.Write([]byte{0}) // separator so (id, set) pairs never collide

	for _, a := range aliases {
		h.Write([]byte(a))
		h.Write([]byte{0})
	}

	// Reduce the digest modulo the alias count in uint32 space: converting the
	// uint32 digest to int first would yield a negative value (and a negative
	// index) wherever int is 32 bits.
	idx := int(h.Sum32() % uint32(len(aliases)))

	return aliases[idx]
}

// selectSticky orders candidate indices for StrategySticky.
//
// Without a session id it falls back to the configured sticky-fallback strategy
// (never a randomly generated affinity key). With a session id it pins the
// preferred alias first (when it is among the available candidates) and lists
// the remaining available candidates in definition order, giving a deterministic
// failover order when the preferred endpoint is unhealthy.
func (c *ComposeClient) selectSticky(ctx context.Context, req *ais.ChatRequest, available []int) []int {
	sessionID := sessionIDFromContext(ctx)
	if sessionID == "" {
		fallback := c.stickyFallback
		if fallback == StrategySticky { // never recurse into sticky
			fallback = StrategyFailover
		}

		return c.orderByStrategy(ctx, req, fallback, available)
	}

	if len(available) == 0 {
		return available
	}

	preferred := c.stickyPreferredAlias(sessionID)

	// Locate the preferred alias within the available candidates.
	prefIdx := -1

	inAvailable := false

	for _, idx := range available {
		if c.entries[idx].Alias == preferred {
			prefIdx = idx
			inAvailable = true

			break
		}
	}

	result := make([]int, 0, len(available))

	if inAvailable {
		result = append(result, prefIdx)
	}

	// Append the rest in the deterministic order the caller supplied
	// (definition order), skipping the preferred already placed first.
	for _, idx := range available {
		if idx == prefIdx {
			continue
		}

		result = append(result, idx)
	}

	return result
}
