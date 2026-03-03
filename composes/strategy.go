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

import "math/rand"

// Strategy determines how models are selected for requests.
type Strategy string

const (
	// StrategyFailover selects models in definition order, skipping errored ones.
	StrategyFailover Strategy = "failover"
	// StrategyRandom selects a random active model for each request.
	StrategyRandom Strategy = "random"
	// StrategyWeight selects models based on their weight proportionally.
	StrategyWeight Strategy = "weighted"
)

// selectModels returns an ordered list of model indices to try.
// The caller iterates and attempts each until one succeeds.
func (c *ComposeClient) selectModels() []int {
	switch c.strategy {
	case StrategyRandom:
		return c.selectRandom()
	case StrategyWeight:
		return c.selectWeighted()
	default:
		return c.selectFailover()
	}
}

// selectFailover returns indices in definition order, skipping error models.
func (c *ComposeClient) selectFailover() []int {
	result := make([]int, 0, len(c.entries))

	for i := range c.entries {
		if c.health[i].isActive() {
			result = append(result, i)
		}
	}

	return result
}

// selectRandom returns a shuffled list of active model indices.
func (c *ComposeClient) selectRandom() []int {
	active := make([]int, 0, len(c.entries))

	for i := range c.entries {
		if c.health[i].isActive() {
			active = append(active, i)
		}
	}

	c.mu.Lock()
	c.rng.Shuffle(len(active), func(i, j int) {
		active[i], active[j] = active[j], active[i]
	})
	c.mu.Unlock()

	return active
}

// selectWeighted selects from active models proportional to their weights.
// Returns a full ordering: pick one by weight, then repeat with remaining.
func (c *ComposeClient) selectWeighted() []int {
	type candidate struct {
		idx    int
		weight int
	}

	candidates := make([]candidate, 0, len(c.entries))

	for i := range c.entries {
		if c.health[i].isActive() {
			w := c.entries[i].Weight
			if w <= 0 {
				w = 1
			}

			candidates = append(candidates, candidate{idx: i, weight: w})
		}
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

// randSource returns a new deterministic rand for testing or real rand.
func newRand(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
