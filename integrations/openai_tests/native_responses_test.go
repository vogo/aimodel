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
	"errors"
	"io"
	"os"
	"testing"

	"github.com/vogo/aimodel/provider/openai"
)

// nativeResponsesClient builds the native client the Responses examples use, or
// skips when the environment has no OpenAI credentials.
func nativeResponsesClient(t *testing.T) (*openai.Client, string) {
	t.Helper()

	apiKey, model := os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_MODEL")
	if apiKey == "" || model == "" {
		t.Skip("OPENAI_API_KEY and OPENAI_MODEL are required")
	}

	options := []openai.ClientOption{}
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		options = append(options, openai.WithBaseURL(baseURL))
	}

	return openai.NewClient(apiKey, options...), model
}

// TestNativeResponses is the native non-streaming example: native types end to
// end, with the hosted web search tool enabled.
func TestNativeResponses(t *testing.T) {
	client, model := nativeResponsesClient(t)

	response, err := client.Responses(context.Background(), &openai.ResponsesRequest{
		Model:        model,
		Instructions: "Answer in one sentence.",
		Input:        openai.NewResponseTextInput("What is the Responses API?"),
		Tools:        []openai.ResponseTool{{Type: openai.ResponseToolTypeWebSearch}},
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("status: %s", response.Status)
	t.Logf("output text: %s", response.OutputText)

	for _, item := range response.Output {
		t.Logf("output item: %s", item.Type)
	}

	if response.Usage != nil {
		t.Logf("usage: in=%d out=%d total=%d", response.Usage.InputTokens, response.Usage.OutputTokens, response.Usage.TotalTokens)
	}
}

// TestNativeResponsesStream is the native streaming example: one typed event at
// a time, ending with io.EOF.
func TestNativeResponsesStream(t *testing.T) {
	client, model := nativeResponsesClient(t)

	stream, err := client.ResponsesStream(context.Background(), &openai.ResponsesRequest{
		Model: model,
		Input: openai.NewResponseTextInput("Count from one to three."),
	})
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = stream.Close() }()

	for {
		event, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			return
		}

		if recvErr != nil {
			t.Fatal(recvErr)
		}

		switch event.Type {
		case openai.ResponseEventOutputTextDelta:
			t.Logf("delta: %s", event.Delta)
		case openai.ResponseEventCompleted:
			t.Logf("completed: %s", event.Response.OutputText)
		default:
			t.Logf("event %d: %s", event.SequenceNumber, event.Type)
		}
	}
}
