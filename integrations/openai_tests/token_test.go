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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vogo/aimodel/openai"
)

// Token accounting end to end against mock backends: both protocols report
// prompt-cache activity, and each one reports it in its own shape. These run
// offline, so they gate merges; the examples that need a real key live in the
// provider example files.

// mockBackend serves one canned body for every request.
func mockBackend(t *testing.T, contentType, body string) *httptest.Server {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	return server
}

func TestOpenAICacheTokensSync(t *testing.T) {
	server := mockBackend(t, "application/json", `{
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
		"usage": {
			"prompt_tokens": 100,
			"completion_tokens": 20,
			"total_tokens": 120,
			"prompt_tokens_details": {"cached_tokens": 40}
		}
	}`)

	client := openai.NewClient("sk-test", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))

	response, err := client.ChatCompletions(context.Background(), &openai.ChatCompletionRequest{
		Model:    openai.ModelGPT4o,
		Messages: []openai.ChatCompletionMessage{{Role: openai.RoleUser, Content: openai.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}

	usage := response.Usage
	if usage.PromptTokens != 100 || usage.CompletionTokens != 20 || usage.TotalTokens != 120 {
		t.Errorf("usage totals = %+v", usage)
	}

	// OpenAI reports cached tokens as a subset of prompt_tokens, the opposite
	// of Anthropic's alongside-the-input accounting.
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens != 40 {
		t.Errorf("cached tokens = %+v, want 40", usage.PromptTokensDetails)
	}
}

func TestOpenAICacheTokensStream(t *testing.T) {
	server := mockBackend(t, "text/event-stream",
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`+"\n\n"+
			`data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n"+
			`data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":40}}}`+"\n\n"+
			"data: [DONE]\n\n")

	client := openai.NewClient("sk-test", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))

	stream, err := client.ChatCompletionsStream(context.Background(), &openai.ChatCompletionRequest{
		Model:         openai.ModelGPT4o,
		Messages:      []openai.ChatCompletionMessage{{Role: openai.RoleUser, Content: openai.NewTextContent("Hi")}},
		StreamOptions: &openai.StreamOptions{IncludeUsage: new(true)},
	})
	if err != nil {
		t.Fatalf("ChatCompletionsStream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	for {
		_, err = stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("stream reported no usage")
	}

	if usage.PromptTokens != 100 || usage.CompletionTokens != 20 {
		t.Errorf("usage totals = %+v", usage)
	}

	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens != 40 {
		t.Errorf("cached tokens = %+v, want 40", usage.PromptTokensDetails)
	}
}

func TestOpenAINoCacheTokens(t *testing.T) {
	server := mockBackend(t, "application/json", `{
		"id": "chatcmpl-2",
		"object": "chat.completion",
		"model": "gpt-4o",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
	}`)

	client := openai.NewClient("sk-test", openai.WithBaseURL(server.URL), openai.WithHTTPClient(server.Client()))

	response, err := client.ChatCompletions(context.Background(), &openai.ChatCompletionRequest{
		Model:    openai.ModelGPT4o,
		Messages: []openai.ChatCompletionMessage{{Role: openai.RoleUser, Content: openai.NewTextContent("Hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}

	if response.Usage.PromptTokensDetails != nil {
		t.Errorf("prompt_tokens_details = %+v, want nil when the API omits it", response.Usage.PromptTokensDetails)
	}
}
