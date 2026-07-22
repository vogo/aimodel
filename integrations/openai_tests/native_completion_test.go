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

package openai_tests

import (
	"context"
	"os"
	"testing"

	"github.com/vogo/aimodel/provider/openai"
)

func TestNativeChatCompletions(t *testing.T) {
	apiKey, model := os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_MODEL")
	if apiKey == "" || model == "" {
		t.Skip("OPENAI_API_KEY and OPENAI_MODEL are required")
	}
	options := []openai.ClientOption{}
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		options = append(options, openai.WithBaseURL(baseURL))
	}
	response, err := openai.NewClient(apiKey, options...).ChatCompletions(context.Background(), &openai.ChatCompletionRequest{
		Model: model, Messages: []openai.ChatCompletionMessage{{Role: "user", Content: openai.NewTextContent("Say hello in one sentence.")}},
		Logprobs: new(true), TopLogprobs: new(2),
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.ID == "" {
		t.Fatal("empty completion id")
	}
	if len(response.Choices) != 0 {
		t.Logf("native logprobs: %+v", response.Choices[0].Logprobs)
	}
}
