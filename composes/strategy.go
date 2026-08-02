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
	"math/rand"

	"github.com/vogo/aimodel/ais"
)

// Strategy determines how candidate endpoints are ordered for a request. Every
// strategy returns a full ordered candidate list (not a single pick), so the
// dispatch loop fails over uniformly regardless of strategy.
type Strategy string

const (
	// StrategyFailover selects endpoints in definition order, skipping unhealthy ones.
	StrategyFailover Strategy = "failover"
	// StrategyRandom selects active endpoints in a shuffled order.
	StrategyRandom Strategy = "random"
	// StrategyWeight orders endpoints sampled without replacement in proportion
	// to weight; Weight <= 0 counts as 1.
	StrategyWeight Strategy = "weighted"
	// StrategySticky pins a request stream to a stable endpoint by session id
	// (see WithSessionID). It is non-default; without a session id it falls back
	// to the configured sticky-fallback strategy.
	StrategySticky Strategy = "sticky"
	// StrategyCost orders endpoints by ascending static cost (EndpointCost).
	StrategyCost Strategy = "cost"
	// StrategyLatency orders endpoints by ascending injected latency.
	StrategyLatency Strategy = "latency"
)

// selectModels returns the ordered list of endpoint indices to try. It first
// narrows the capability-filtered set to the health-available endpoints, then
// orders them according to the configured strategy.
func (c *ComposeClient) selectModels(ctx context.Context, req *ais.ChatRequest, capable []int) []int {
	now := c.nowFunc()

	available := make([]int, 0, len(capable))

	for _, idx := range capable {
		if c.health[idx].available(now, c.coolingInterval) {
			available = append(available, idx)
		}
	}

	return c.orderByStrategy(ctx, req, c.strategy, available)
}

// orderByStrategy orders an already-available candidate slice per the given
// strategy. The input slice is in definition order and is not mutated.
func (c *ComposeClient) orderByStrategy(ctx context.Context, req *ais.ChatRequest, s Strategy, available []int) []int {
	switch s {
	case StrategyRandom:
		return c.orderRandom(available)
	case StrategyWeight:
		return c.orderWeighted(available)
	case StrategySticky:
		return c.selectSticky(ctx, req, available)
	case StrategyCost:
		return c.sortByCost(append([]int(nil), available...), req)
	case StrategyLatency:
		return c.sortByLatency(append([]int(nil), available...))
	default: // StrategyFailover and any unknown value.
		return append([]int(nil), available...)
	}
}

// orderRandom returns a shuffled copy of the available indices.
func (c *ComposeClient) orderRandom(available []int) []int {
	result := append([]int(nil), available...)

	c.mu.Lock()
	c.rng.Shuffle(len(result), func(i, j int) {
		result[i], result[j] = result[j], result[i]
	})
	c.mu.Unlock()

	return result
}

// orderWeighted orders the available indices by sampling without replacement in
// proportion to weight; Weight <= 0 counts as 1.
func (c *ComposeClient) orderWeighted(available []int) []int {
	type candidate struct {
		idx    int
		weight int
	}

	candidates := make([]candidate, 0, len(available))

	for _, idx := range available {
		w := c.entries[idx].Weight
		if w <= 0 {
			w = 1
		}

		candidates = append(candidates, candidate{idx: idx, weight: w})
	}

	result := make([]int, 0, len(candidates))

	c.mu.Lock()
	defer c.mu.Unlock()

	for len(candidates) > 0 {
		total := 0
		for _, cand := range candidates {
			total += cand.weight
		}

		r := c.rng.Intn(total)
		cumulative := 0

		for j, cand := range candidates {
			cumulative += cand.weight

			if r < cumulative {
				result = append(result, cand.idx)
				candidates = append(candidates[:j], candidates[j+1:]...)

				break
			}
		}
	}

	return result
}

// newRand returns a deterministic rand for testing or a seeded real rand.
func newRand(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
