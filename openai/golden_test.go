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

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// The baselines under testdata/golden are the exact request bodies this client
// is expected to put on the wire for tool calls, image content and thinking. A
// request must reach its baseline byte for byte; where it cannot, the
// difference belongs in a comment, not in a relaxed comparison.

// captureNativeRequest sends one native request and returns the exact body it
// put on the wire.
func captureNativeRequest(t *testing.T, request *ChatCompletionRequest) []byte {
	t.Helper()

	var body []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chat-1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	client := NewClient("test-key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	if _, err := client.ChatCompletions(context.Background(), request); err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}

	return body
}

// assertMatchesGolden compares a request body against a recorded baseline.
func assertMatchesGolden(t *testing.T, name string, body []byte) {
	t.Helper()

	var indented bytes.Buffer
	if err := json.Indent(&indented, body, "", "  "); err != nil {
		t.Fatalf("indent: %v (body=%s)", err, body)
	}

	indented.WriteByte('\n')

	path := filepath.Join("testdata", "golden", name)

	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	if !bytes.Equal(want, indented.Bytes()) {
		t.Errorf("native request differs from the golden baseline %s\n--- want ---\n%s\n--- got ---\n%s",
			path, want, indented.Bytes())
	}
}

func TestNativeToolCallMatchesGoldenBaseline(t *testing.T) {
	request := &ChatCompletionRequest{
		Model: ModelGPT4o,
		Messages: []ChatCompletionMessage{
			{Role: "user", Content: NewTextContent("What is the weather in SF?")},
			// An assistant turn that only calls a tool still sends an explicit
			// empty content.
			{Role: "assistant", Content: NewTextContent(""), ToolCalls: []ChatCompletionToolCall{{
				ID: "call_1", Type: "function",
				Function: ChatCompletionFunctionCall{Name: "get_weather", Arguments: `{"city":"SF"}`},
			}}},
			{Role: "tool", ToolCallID: "call_1", Content: NewTextContent(`{"temp_c":18}`)},
		},
		Tools: []ChatCompletionTool{{
			Type: "function",
			Function: ChatCompletionFunction{
				Name:        "get_weather",
				Description: "Get the current weather in a city",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
					"required":   []any{"city"},
				},
				Strict: new(true),
			},
		}},
		ToolChoice:        "auto",
		ParallelToolCalls: new(false),
	}

	assertMatchesGolden(t, "tool_call.json", captureNativeRequest(t, request))
}

func TestNativeImageContentMatchesGoldenBaseline(t *testing.T) {
	request := &ChatCompletionRequest{
		Model: ModelGPT4o,
		Messages: []ChatCompletionMessage{{Role: "user", Content: NewPartsContent(
			ChatCompletionContentPart{Type: "text", Text: "What is in this image?"},
			ChatCompletionContentPart{Type: "image_url", ImageURL: &ImageURL{
				URL:    "https://example.com/cat.png",
				Detail: "high",
			}},
		)}},
		MaxCompletionTokens: new(512),
	}

	assertMatchesGolden(t, "image_content.json", captureNativeRequest(t, request))
}

func TestNativeThinkingMatchesGoldenBaseline(t *testing.T) {
	request := &ChatCompletionRequest{
		Model:               ModelGPT56,
		Messages:            []ChatCompletionMessage{{Role: "user", Content: NewTextContent("Solve it step by step.")}},
		Thinking:            &Thinking{Type: "enabled", BudgetTokens: 2048, Display: "omitted"},
		ReasoningEffort:     ReasoningEffortHigh,
		MaxCompletionTokens: new(4096),
	}

	assertMatchesGolden(t, "thinking.json", captureNativeRequest(t, request))
}

// TestNativeToolCallResponseDecodesFully covers the response half of the tool
// call path: index, id, type and the function name/arguments pair.
func TestNativeToolCallResponseDecodesFully(t *testing.T) {
	const fixture = `{"id":"chat-1","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]}}],"usage":{"prompt_tokens":50,"completion_tokens":12,"total_tokens":62}}`

	var response ChatCompletionResponse
	if err := json.Unmarshal([]byte(fixture), &response); err != nil {
		t.Fatal(err)
	}

	choice := response.Choices[0]
	if choice.FinishReason == nil || *choice.FinishReason != "tool_calls" {
		t.Errorf("finish reason = %v", choice.FinishReason)
	}

	if len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d", len(choice.Message.ToolCalls))
	}

	call := choice.Message.ToolCalls[0]
	if call.ID != "call_1" || call.Function.Name != "get_weather" || call.Function.Arguments != `{"city":"SF"}` {
		t.Errorf("tool call = %+v", call)
	}
}

// TestNativeStreamUsageFromSharedFixture reads the shared SSE fixture under
// testdata/golden, so streaming usage is measured on fixed input.
func TestNativeStreamUsageFromSharedFixture(t *testing.T) {
	fixture, err := os.ReadFile(filepath.Join("testdata", "golden", "stream_usage.sse"))
	if err != nil {
		t.Fatalf("read stream fixture: %v", err)
	}

	stream, err := streamServer(t, string(fixture)).
		ChatCompletionsStream(context.Background(), &ChatCompletionRequest{
			Model:         ModelGPT56,
			StreamOptions: &StreamOptions{IncludeUsage: new(true)},
		})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	drain(t, stream)

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("Usage is nil after a stream that reported it")
	}

	for _, tc := range []struct {
		field string
		got   int
		want  int
	}{
		{"prompt_tokens", usage.PromptTokens, 1000},
		{"completion_tokens", usage.CompletionTokens, 20},
		{"total_tokens", usage.TotalTokens, 1020},
		{"cached_tokens", usage.PromptTokensDetails.CachedTokens, 896},
		{"reasoning_tokens", usage.CompletionTokensDetails.ReasoningTokens, 8},
	} {
		if tc.got != tc.want {
			t.Errorf("usage.%s = %d, want %d", tc.field, tc.got, tc.want)
		}
	}

	response := stream.Response()
	if got := response.Choices[0].Message.Content.Text(); got != "Hello" {
		t.Errorf("content = %q", got)
	}

	if response.Choices[0].Message.ReasoningContent != "thinking" {
		t.Errorf("reasoning content = %q", response.Choices[0].Message.ReasoningContent)
	}
}

// TestNativePromptCachingUsageDecodes covers the OpenAI half of prompt
// caching, which is reported on the response rather than requested, through
// the nested prompt/completion token details.
func TestNativePromptCachingUsageDecodes(t *testing.T) {
	const fixture = `{"id":"chat-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":20,"total_tokens":1020,"prompt_tokens_details":{"cached_tokens":896,"audio_tokens":0},"completion_tokens_details":{"reasoning_tokens":8}}}`

	var response ChatCompletionResponse
	if err := json.Unmarshal([]byte(fixture), &response); err != nil {
		t.Fatal(err)
	}

	if response.Usage.PromptTokensDetails.CachedTokens != 896 {
		t.Errorf("cached tokens = %d, want 896", response.Usage.PromptTokensDetails.CachedTokens)
	}

	if response.Usage.CompletionTokensDetails.ReasoningTokens != 8 {
		t.Errorf("reasoning tokens = %d, want 8", response.Usage.CompletionTokensDetails.ReasoningTokens)
	}

	// The cached count is a subset of the prompt tokens, not an addition to
	// them.
	if response.Usage.PromptTokensDetails.CachedTokens > response.Usage.PromptTokens {
		t.Error("cached tokens must be a subset of prompt tokens")
	}
}
