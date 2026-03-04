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

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/vogo/aimodel"
)

func testToolCall(client *aimodel.Client) {
	fmt.Println("=== OpenAI Tool Call ===")

	tools := []aimodel.Tool{
		{
			Type: "function",
			Function: aimodel.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather for a given location",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city name, e.g. Beijing",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	messages := []aimodel.Message{
		{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("What is the weather in Beijing today?")},
	}

	// Step 1: send the request with tools, the model should trigger a tool call.
	resp, err := client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) == 0 {
		log.Fatal("no choices in response")
	}

	choice := resp.Choices[0]

	// If the model did not request a tool call, print and exit.
	if choice.FinishReason != aimodel.FinishReasonToolCalls || len(choice.Message.ToolCalls) == 0 {
		fmt.Println(choice.Message.Content.Text())
		return
	}

	fmt.Printf("Model requested %d tool call(s)\n", len(choice.Message.ToolCalls))

	// Append the assistant message with tool calls.
	messages = append(messages, choice.Message)

	// Step 2: execute each tool call and append results.
	for _, tc := range choice.Message.ToolCalls {
		fmt.Printf("  -> %s(%s)\n", tc.Function.Name, tc.Function.Arguments)

		var args struct {
			Location string `json:"location"`
		}

		if unmarshalErr := json.Unmarshal([]byte(tc.Function.Arguments), &args); unmarshalErr != nil {
			log.Fatalf("parse tool arguments: %v", unmarshalErr)
		}

		result := getWeather(args.Location)

		messages = append(messages, aimodel.Message{
			Role:       aimodel.RoleTool,
			Content:    aimodel.NewTextContent(result),
			ToolCallID: tc.ID,
		})
	}

	// Step 3: send the tool results back, the model should produce a final answer.
	resp, err = client.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) == 0 {
		log.Fatal("no choices in response")
	}

	fmt.Println(resp.Choices[0].Message.Content.Text())
}

// getWeather simulates a weather API call.
func getWeather(location string) string {
	data := map[string]string{
		"location":    location,
		"temperature": "22",
		"unit":        "celsius",
		"condition":   "sunny",
	}

	b, _ := json.Marshal(data)

	return string(b)
}
