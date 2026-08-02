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
	"os"
	"testing"

	"github.com/vogo/aimodel/composes"
	"github.com/vogo/aimodel/provider/openai"
)

// Compose dispatches across backends that speak one wire format. These
// examples use several OpenAI-compatible endpoints, which is the shape real
// deployments have: one request body, many endpoints that can serve it.

// backendPrefixes names the environment groups a backend is configured from:
// <PREFIX>_API_KEY, <PREFIX>_BASE_URL and <PREFIX>_MODEL.
var backendPrefixes = []string{"OPENAI", "DEEPSEEK", "QWEN"}

func TestComposeClient(t *testing.T) {
	entries := buildComposeEntries()
	if len(entries) == 0 {
		t.Skip("no OpenAI-compatible backend configured; set <PREFIX>_API_KEY and <PREFIX>_MODEL")
	}

	testFailover(entries)
	testWeight(entries)
	testRandom(entries)
}

// buildComposeEntries configures one entry per fully-specified environment
// group and skips the rest, so the examples run with whatever is available.
func buildComposeEntries() []composes.ModelEntry {
	var entries []composes.ModelEntry

	for _, prefix := range backendPrefixes {
		apiKey, model := os.Getenv(prefix+"_API_KEY"), os.Getenv(prefix+"_MODEL")
		if apiKey == "" || model == "" {
			continue
		}

		options := []openai.ClientOption{}
		if baseURL := os.Getenv(prefix + "_BASE_URL"); baseURL != "" {
			options = append(options, openai.WithBaseURL(baseURL))
		}

		entries = append(entries, composes.ModelEntry{
			Name:   model,
			Client: openai.NewClient(apiKey, options...),
		})
	}

	return entries
}
