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

package compose_tests

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/vogo/aimodel/composes"
	"github.com/vogo/aimodel/provider/openai"
)

// testCanary demonstrates the declarative same-provider multi-endpoint path:
// one model served by two endpoints (a stable one and a canary) with different
// base URLs and keys. NewFromEndpoints builds both clients from specs, the
// weighted strategy splits traffic 90/10 for a canary migration, failover
// covers a failing endpoint, and the attempt observer attributes every try to a
// stable endpoint alias.
func testCanary() {
	fmt.Println("=== Compose Same-Provider Canary ===")

	// This example needs two real endpoints; skip gracefully when the canary
	// coordinates are not configured.
	if os.Getenv("CANARY_BASE_URL") == "" || os.Getenv("CANARY_API_KEY") == "" {
		fmt.Println("  (skipped: set CANARY_BASE_URL and CANARY_API_KEY to run)")
		return
	}

	model := os.Getenv("OPENAI_MODEL")

	cc, err := composes.NewFromEndpoints(composes.StrategyWeight, []composes.EndpointSpec{
		{
			Alias:   "stable",
			BaseURL: os.Getenv("OPENAI_BASE_URL"),
			APIKey:  os.Getenv("OPENAI_API_KEY"),
			Model:   model,
			Weight:  9, // 90% of traffic
			Tags:    map[string]string{"tier": "stable"},
		},
		{
			Alias:   "canary",
			BaseURL: os.Getenv("CANARY_BASE_URL"),
			APIKey:  os.Getenv("CANARY_API_KEY"),
			Model:   model,
			Weight:  1, // 10% of traffic
			Tags:    map[string]string{"tier": "canary"},
		},
	}, composes.WithAttemptObserver(func(res composes.AttemptResult) {
		status := "ok"
		if !res.Success {
			status = "fail: " + res.Err.Error()
		}

		fmt.Printf("  attempt endpoint=%s %s\n", res.Alias, status)
	}))
	if err != nil {
		log.Fatal(err)
	}

	for i := range 5 {
		resp, err := cc.ChatCompletions(context.Background(), &openai.ChatCompletionRequest{
			Messages: []openai.ChatCompletionMessage{
				{Role: "user", Content: openai.NewTextContent("Say hello!")},
			},
		})
		if err != nil {
			log.Printf("request %d: %v", i+1, err)
			continue
		}

		fmt.Printf("request %d [%s]: %s\n", i+1, resp.Model, resp.Choices[0].Message.Content.Text())
	}

	// Per-endpoint health snapshot, keyed by alias.
	for _, s := range cc.Stats() {
		fmt.Printf("  endpoint=%s status=%s errorCount=%d\n", s.Alias, s.Status, s.ErrorCount)
	}
}
