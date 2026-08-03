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
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/vogo/aimodel/composes"
	"github.com/vogo/aimodel/composes/anthropics"
	"github.com/vogo/aimodel/provider/anthropic"
)

// anthropicKeyVars names the environment variables that each carry one
// Anthropic credential. Several keys for the same model is the common shape:
// multi-key aggregation, one key serving until it is judged dead.
var anthropicKeyVars = []string{"ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "ANTHROPIC_API_KEY_3"}

// testMessages dispatches Anthropic Messages across several Anthropic
// endpoints. It is a separate pool from the OpenAI one — there is no
// cross-protocol failover — but it is the same routing core underneath: the
// same strategies, the same health state machine, the same aliases and stats.
func testMessages() {
	fmt.Println("=== Compose Anthropic Messages ===")

	specs := buildAnthropicSpecs()
	if len(specs) == 0 {
		fmt.Println("  (skipped: set ANTHROPIC_API_KEY and ANTHROPIC_MODEL to run)")

		return
	}

	cc, err := anthropics.NewFromEndpoints(composes.StrategyWeight, specs,
		composes.WithAttemptObserver(func(res composes.AttemptResult) {
			status := "ok"
			if !res.Success {
				status = "fail: " + res.Err.Error()
			}

			fmt.Printf("  attempt endpoint=%s %s\n", res.Alias, status)
		}))
	if err != nil {
		log.Fatal(err)
	}

	response, err := cc.Messages(context.Background(), &anthropic.MessagesRequest{
		MaxTokens: 64,
		Messages: []anthropic.MessagesMessage{
			{Role: anthropic.RoleUser, Content: json.RawMessage(`"Say hello!"`)},
		},
	})
	if err != nil {
		log.Printf("messages: %v", err)

		return
	}

	for _, block := range response.Content {
		if block.Type == anthropic.ContentBlockTypeText {
			fmt.Printf("[%s] %s\n", response.Model, block.Text)
		}
	}

	// Per-endpoint health snapshot, keyed by alias — identical in shape to the
	// OpenAI pool's, because it is the same core reporting it.
	for _, s := range cc.Stats() {
		fmt.Printf("  endpoint=%s status=%s errorCount=%d\n", s.Alias, s.Status, s.ErrorCount)
	}
}

// buildAnthropicSpecs turns every configured credential into one endpoint for
// the same model, and skips the rest.
func buildAnthropicSpecs() []anthropics.EndpointSpec {
	model := os.Getenv("ANTHROPIC_MODEL")
	if model == "" {
		return nil
	}

	var specs []anthropics.EndpointSpec

	for i, keyVar := range anthropicKeyVars {
		apiKey := os.Getenv(keyVar)
		if apiKey == "" {
			continue
		}

		specs = append(specs, anthropics.EndpointSpec{
			Alias:   fmt.Sprintf("key-%d", i+1),
			BaseURL: os.Getenv("ANTHROPIC_BASE_URL"),
			APIKey:  apiKey,
			Model:   model,
			Weight:  1,
			Tags:    map[string]string{"credential": keyVar},
		})
	}

	return specs
}
