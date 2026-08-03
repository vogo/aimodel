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

	"github.com/vogo/aimodel/composes"
	"github.com/vogo/aimodel/provider/openai"
)

func testRandom(entries []composes.ModelEntry) {
	fmt.Println("=== Compose Random ===")

	cc, err := composes.NewComposeClient(composes.StrategyRandom, entries)
	if err != nil {
		log.Fatal(err)
	}

	// Send 5 requests to show the random distribution.
	for i := range 5 {
		response, err := cc.ChatCompletions(context.Background(), helloRequest())
		if err != nil {
			log.Printf("request %d: %v", i+1, err)

			continue
		}

		if len(response.Choices) == 0 {
			log.Printf("request %d: no choices", i+1)

			continue
		}

		fmt.Printf("request %d [%s]: %s\n", i+1, response.Model, response.Choices[0].Message.Content.Text())
	}
}

// helloRequest is the one request body every backend in the pool receives.
func helloRequest() *openai.ChatCompletionRequest {
	return &openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.RoleUser, Content: openai.NewTextContent("Say hello!")},
		},
	}
}
