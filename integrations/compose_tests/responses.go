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
	"errors"
	"fmt"
	"io"
	"log"

	"github.com/vogo/aimodel/composes"
	"github.com/vogo/aimodel/composes/openais"
	"github.com/vogo/aimodel/provider/openai"
)

// testResponses dispatches the OpenAI Responses API across the same endpoint
// pool the chat examples use. It is the second interaction form on one compose
// client: same strategies, same health state, its own method set and its own
// request type — nothing is translated between the two.
func testResponses(entries []openais.ModelEntry) {
	fmt.Println("=== Compose Responses ===")

	cc, err := openais.NewComposeClient(composes.StrategyFailover, entries,
		composes.WithAttemptObserver(func(res composes.AttemptResult) {
			status := "ok"
			if !res.Success {
				status = "fail: " + res.Err.Error()
			}

			fmt.Printf("  attempt endpoint=%s stream=%v %s\n", res.Alias, res.Stream, status)
		}))
	if err != nil {
		log.Fatal(err)
	}

	response, err := cc.Responses(context.Background(), &openai.ResponsesRequest{
		Input: openai.NewResponseTextInput("Say hello!"),
	})
	if err != nil {
		// An endpoint pool without a Responses-capable backend fails fast, before
		// any network I/O — the same capability error shape a missing tools or
		// vision capability produces.
		if errors.Is(err, composes.ErrCapabilityNotSatisfied) {
			fmt.Printf("  (skipped: %v)\n", err)

			return
		}

		log.Printf("responses: %v", err)

		return
	}

	fmt.Printf("[%s] %s\n", response.Model, response.OutputText)

	testResponsesStream(cc)
}

// testResponsesStream shows that failover covers stream establishment only:
// once a backend starts streaming, its events reach the caller directly.
func testResponsesStream(cc *openais.ComposeClient) {
	stream, err := cc.ResponsesStream(context.Background(), &openai.ResponsesRequest{
		Input: openai.NewResponseTextInput("Count to three."),
	})
	if err != nil {
		log.Printf("responses stream: %v", err)

		return
	}

	defer func() { _ = stream.Close() }()

	for {
		event, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			log.Printf("responses stream recv: %v", err)

			return
		}

		if event.Type == openai.ResponseEventOutputTextDelta {
			fmt.Print(event.Delta)
		}
	}

	fmt.Println()

	for _, s := range cc.Stats() {
		fmt.Printf("  endpoint=%s status=%s errorCount=%d\n", s.Alias, s.Status, s.ErrorCount)
	}
}
