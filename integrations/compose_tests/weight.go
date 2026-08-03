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
	"github.com/vogo/aimodel/composes/openais"
)

func testWeight(entries []openais.ModelEntry) {
	fmt.Println("=== Compose Weight ===")

	// Send three times as much traffic to the first backend as to each of the
	// others.
	weighted := make([]openais.ModelEntry, len(entries))
	for i, entry := range entries {
		entry.Weight = 1
		if i == 0 {
			entry.Weight = 3
		}

		weighted[i] = entry
	}

	cc, err := openais.NewComposeClient(composes.StrategyWeight, weighted)
	if err != nil {
		log.Fatal(err)
	}

	// Send 5 requests to show the traffic distribution.
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
