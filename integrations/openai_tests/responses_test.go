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

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/provider/openai"
)

// unifiedResponder builds the unified client the Responses examples use. The
// same client also serves canonical chat calls; Responses reaches it through
// the separate Responder capability.
func unifiedResponder(t *testing.T) (aimodel.Responder, string) {
	t.Helper()

	model := os.Getenv("OPENAI_MODEL")
	if os.Getenv("OPENAI_API_KEY") == "" || model == "" {
		t.Skip("OPENAI_API_KEY and OPENAI_MODEL are required")
	}

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey(aimodel.GetEnv("OPENAI_API_KEY")),
		aimodel.WithBaseURL(aimodel.GetEnv("OPENAI_BASE_URL", "AI_BASE_URL")),
	)
	if err != nil {
		t.Skipf("init client error: %v", err)
	}

	return client, model
}

// TestResponses is the unified non-streaming example. The unified client speaks
// the OpenAI-native Responses types on this capability — Responses has no
// canonical form (ADR 0006).
func TestResponses(t *testing.T) {
	client, model := unifiedResponder(t)

	response, err := client.Responses(context.Background(), &openai.ResponsesRequest{
		Model:        model,
		Instructions: "Answer in one sentence.",
		Input:        openai.NewResponseTextInput("What is the Responses API?"),
	})
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("status: %s", response.Status)
	t.Logf("output text: %s", response.OutputText)
}

// TestResponsesStream is the unified streaming example.
func TestResponsesStream(t *testing.T) {
	client, model := unifiedResponder(t)

	stream, err := client.ResponsesStream(context.Background(), &openai.ResponsesRequest{
		Model: model,
		Input: openai.NewResponseItemsInput(
			openai.NewResponseInputMessage(openai.ResponseRoleUser, "Count from one to three."),
		),
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

		if event.Type == openai.ResponseEventOutputTextDelta {
			t.Logf("delta: %s", event.Delta)
		}
	}
}
