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

package thinking_tests

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"testing"

	"github.com/vogo/aimodel"
)

func TestThinkingClient(t *testing.T) {
	client, err := aimodel.NewClient(
		aimodel.WithAPIKey(aimodel.GetEnv("ANTHROPIC_API_KEY")),
		aimodel.WithBaseURL(aimodel.GetEnv("ANTHROPIC_BASE_URL")),
		aimodel.WithDefaultModel(aimodel.GetEnv("ANTHROPIC_MODEL")),
		aimodel.WithProtocol(aimodel.ProtocolAnthropic),
	)
	if err != nil {
		t.Logf("init client error: %v", err)
		return
	}

	testThinkingCompletion(client)
	testThinkingStream(client)
}

func testThinkingCompletion(client *aimodel.Client) {
	fmt.Println("=== Anthropic Extended Thinking ===")

	maxTokens := 16000
	resp, err := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		MaxTokens: &maxTokens,
		Messages: []aimodel.Message{
			{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("When Will AGI Arrive?")},
		},
		Thinking: &aimodel.Thinking{
			Type:         "enabled",
			BudgetTokens: 10000,
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) == 0 {
		log.Fatal("no choices in response")
	}

	msg := resp.Choices[0].Message
	if msg.Thinking != "" {
		fmt.Println("[Thinking]", msg.Thinking)
	}

	fmt.Println("[Answer]", msg.Content.Text())
}

func testThinkingStream(client *aimodel.Client) {
	fmt.Println("=== Anthropic Extended Thinking Stream ===")

	maxTokens := 16000
	stream, err := client.ChatCompletionStream(context.Background(), &aimodel.ChatRequest{
		MaxTokens: &maxTokens,
		Messages: []aimodel.Message{
			{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("When Will AGI Arrive?")},
		},
		Thinking: &aimodel.Thinking{
			Type:         "enabled",
			BudgetTokens: 10000,
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = stream.Close() }()

	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			if delta.Thinking != "" {
				fmt.Print(delta.Thinking)
			}

			if text := delta.Content.Text(); text != "" {
				fmt.Print(text)
			}
		}
	}

	fmt.Println()
}
