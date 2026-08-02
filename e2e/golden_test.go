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

package e2e_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/ais"
	"github.com/vogo/aimodel/provider/anthropic"
	"github.com/vogo/aimodel/provider/openai"
)

// This file records what the canonical entry point actually puts on the wire
// for the behaviors that only canonical tests covered: tool calls, image
// content, thinking and prompt caching. The recorded bodies live in each
// provider's testdata/golden and are replayed by that provider's native tests,
// which is what proves the removal of the canonical layer changes the entry
// point and not the request semantics.
//
// Run `go test ./e2e -run TestGolden -update` to refresh a baseline after an
// intentional change. This file is deleted with the canonical layer; the
// golden files and the native assertions stay.

var update = flag.Bool("update", false, "rewrite the golden wire baselines")

const (
	openaiGoldenDir    = "../provider/openai/testdata/golden"
	anthropicGoldenDir = "../provider/anthropic/testdata/golden"
)

// captureRequest sends one canonical request and returns the exact body the
// provider put on the wire.
func captureRequest(t *testing.T, providerName string, request *ais.ChatRequest) []byte {
	t.Helper()

	var body []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "application/json")

		if providerName == anthropic.Name {
			_, _ = io.WriteString(w, `{"id":"msg_1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)

			return
		}

		_, _ = io.WriteString(w, `{"id":"chat-1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("test-key"),
		aimodel.WithBaseURL(server.URL),
		aimodel.WithProvider(providerName),
		aimodel.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	if _, err = client.ChatCompletion(context.Background(), request); err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	return body
}

// assertGolden compares a recorded body against its baseline, or rewrites the
// baseline under -update.
func assertGolden(t *testing.T, dir, name string, body []byte) {
	t.Helper()

	var indented bytes.Buffer
	if err := json.Indent(&indented, body, "", "  "); err != nil {
		t.Fatalf("indent %s: %v (body=%s)", name, err, body)
	}

	indented.WriteByte('\n')
	path := filepath.Join(dir, name)

	if *update {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("create %s: %v", dir, err)
		}

		if err := os.WriteFile(path, indented.Bytes(), 0o600); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}

		t.Logf("updated %s", path)

		return
	}

	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v (run with -update to create it)", path, err)
	}

	if !bytes.Equal(want, indented.Bytes()) {
		t.Errorf("%s drifted from its baseline\n--- want ---\n%s\n--- got ---\n%s", path, want, indented.Bytes())
	}
}

// --- canonical request fixtures, shared with the native replay tests ---

func canonicalToolCallRequest(model string) *ais.ChatRequest {
	return &ais.ChatRequest{
		Model: model,
		Messages: []ais.Message{
			{Role: ais.RoleUser, Content: ais.NewTextContent("What is the weather in SF?")},
			{Role: ais.RoleAssistant, ToolCalls: []ais.ToolCall{{
				Index: 0, ID: "call_1", Type: "function",
				Function: ais.FunctionCall{Name: "get_weather", Arguments: `{"city":"SF"}`},
			}}},
			{Role: ais.RoleTool, ToolCallID: "call_1", Content: ais.NewTextContent(`{"temp_c":18}`)},
		},
		Tools: []ais.Tool{{
			Type: "function",
			Function: ais.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the current weather in a city",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
					"required":   []any{"city"},
				},
			},
			Strict: new(true),
		}},
		ToolChoice:        "auto",
		ParallelToolCalls: new(false),
	}
}

func canonicalImageRequest(model string) *ais.ChatRequest {
	return &ais.ChatRequest{
		Model: model,
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewPartsContent(
			ais.ContentPart{Type: "text", Text: "What is in this image?"},
			ais.ContentPart{Type: "image_url", ImageURL: &ais.ImageURL{
				URL:    "https://example.com/cat.png",
				Detail: "high",
			}},
		)}},
		MaxCompletionTokens: new(512),
	}
}

func canonicalThinkingRequest(model string) *ais.ChatRequest {
	return &ais.ChatRequest{
		Model:               model,
		Messages:            []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("Solve it step by step.")}},
		Thinking:            &ais.Thinking{Type: "enabled", BudgetTokens: 2048, Display: "omitted"},
		ReasoningEffort:     ais.ReasoningEffortHigh,
		MaxCompletionTokens: new(4096),
	}
}

// canonicalCachingRequest exercises all three Anthropic cache placements: the
// request-root automatic breakpoint, a per-message breakpoint and a per-tool
// breakpoint.
func canonicalCachingRequest(model string) *ais.ChatRequest {
	system := ais.Message{Role: ais.RoleSystem, Content: ais.NewTextContent("A long, reusable system prompt.")}
	anthropic.ExtendMessage(&system, &anthropic.MessageExtension{CacheBreakpoint: true})

	tool := ais.Tool{
		Type:     "function",
		Function: ais.FunctionDefinition{Name: "search", Description: "Search the knowledge base"},
	}
	anthropic.ExtendTool(&tool, &anthropic.ToolExtension{CacheBreakpoint: true})

	request := &ais.ChatRequest{
		Model: model,
		Messages: []ais.Message{
			system,
			{Role: ais.RoleUser, Content: ais.NewTextContent("Answer from the cached context.")},
		},
		Tools:               []ais.Tool{tool},
		MaxCompletionTokens: new(1024),
	}
	anthropic.ExtendRequest(request, &anthropic.RequestExtension{AutoCache: true, AutoCacheTTL: "1h"})

	return request
}

// --- baselines ---

func TestGoldenOpenAIWireBaselines(t *testing.T) {
	for _, tc := range []struct {
		name    string
		request *ais.ChatRequest
	}{
		{"tool_call.json", canonicalToolCallRequest(ais.ModelOpenaiGPT4o)},
		{"image_content.json", canonicalImageRequest(ais.ModelOpenaiGPT4o)},
		{"thinking.json", canonicalThinkingRequest(ais.ModelOpenaiGPT56)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertGolden(t, openaiGoldenDir, tc.name, captureRequest(t, openai.Name, tc.request))
		})
	}
}

// --- streaming usage: the same SSE bytes through both entry points ---

// replayStream serves a fixture to one canonical streaming call and returns the
// usage the canonical layer reported.
func replayStream(t *testing.T, providerName, fixture string) *ais.Usage {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, fixture)
	}))
	defer server.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("test-key"),
		aimodel.WithBaseURL(server.URL),
		aimodel.WithProvider(providerName),
		aimodel.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := client.ChatCompletionStream(context.Background(), &ais.ChatRequest{
		Model:    "test-model",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
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

	return stream.Usage()
}

func readFixture(t *testing.T, path string) string {
	t.Helper()

	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	return string(body)
}

// TestGoldenAnthropicStreamUsageIsEntryPointIndependent replays one SSE
// fixture through the canonical stream and asserts the token accounting
// matches what the native stream reports for the same bytes — the native
// numbers are pinned in provider/anthropic/accumulate_test.go.
func TestGoldenAnthropicStreamUsageIsEntryPointIndependent(t *testing.T) {
	usage := replayStream(t, anthropic.Name, readFixture(t, anthropicGoldenDir+"/stream_usage.sse"))
	if usage == nil {
		t.Fatal("canonical stream reported no usage")
	}

	// Native: InputTokens 100, CacheCreationInputTokens 5, CacheReadInputTokens
	// 20, OutputTokens 25. The canonical prompt count is their input sum.
	if usage.PromptTokens != 125 {
		t.Errorf("prompt tokens = %d, want 100+5+20", usage.PromptTokens)
	}

	if usage.CompletionTokens != 25 {
		t.Errorf("completion tokens = %d, want the native OutputTokens", usage.CompletionTokens)
	}

	if usage.CacheReadTokens != 20 {
		t.Errorf("cache read tokens = %d, want the native CacheReadInputTokens", usage.CacheReadTokens)
	}

	if usage.ServiceTier != "standard" {
		t.Errorf("service tier = %q; the terminal event must not blank it out", usage.ServiceTier)
	}

	ext := anthropic.UsageExtensionOf(usage)
	if ext == nil || ext.CacheWriteTokens != 5 || ext.InferenceGeo != "us" {
		t.Errorf("usage extension = %+v, want the native cache-write and geo values", ext)
	}
}

// TestGoldenOpenAIStreamUsageIsEntryPointIndependent does the same for the
// OpenAI wire; the native numbers are pinned in
// provider/openai/golden_test.go.
func TestGoldenOpenAIStreamUsageIsEntryPointIndependent(t *testing.T) {
	usage := replayStream(t, openai.Name, readFixture(t, openaiGoldenDir+"/stream_usage.sse"))
	if usage == nil {
		t.Fatal("canonical stream reported no usage")
	}

	for _, tc := range []struct {
		field string
		got   int
		want  int
	}{
		{"prompt tokens", usage.PromptTokens, 1000},
		{"completion tokens", usage.CompletionTokens, 20},
		{"total tokens", usage.TotalTokens, 1020},
		{"cache read tokens", usage.CacheReadTokens, 896},
		{"reasoning tokens", usage.ReasoningTokens, 8},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.field, tc.got, tc.want)
		}
	}
}

func TestGoldenAnthropicWireBaselines(t *testing.T) {
	for _, tc := range []struct {
		name    string
		request *ais.ChatRequest
	}{
		{"tool_call.json", canonicalToolCallRequest(ais.ModelAnthropicClaudeSonnet5)},
		{"image_content.json", canonicalImageRequest(ais.ModelAnthropicClaudeSonnet5)},
		{"thinking.json", canonicalThinkingRequest(ais.ModelAnthropicClaudeSonnet5)},
		{"prompt_caching.json", canonicalCachingRequest(ais.ModelAnthropicClaudeSonnet5)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertGolden(t, anthropicGoldenDir, tc.name, captureRequest(t, anthropic.Name, tc.request))
		})
	}
}
