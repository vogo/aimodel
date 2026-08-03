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

package anthropic

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
// is expected to put on the wire for tool calls, image content, thinking and
// prompt caching. Replaying them from the native client is what shows the
// removal changed the entry point and not the request semantics.

// captureNativeRequest sends one native request and returns the exact body it
// put on the wire.
func captureNativeRequest(t *testing.T, request *MessagesRequest) []byte {
	t.Helper()

	var body []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	client := NewClient("test-key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	if _, err := client.Messages(context.Background(), request); err != nil {
		t.Fatalf("Messages: %v", err)
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

// blocks marshals content blocks into the polymorphic content position.
func blocks(t *testing.T, items ...ContentBlock) json.RawMessage {
	t.Helper()

	encoded, err := json.Marshal(items)
	if err != nil {
		t.Fatalf("marshal content blocks: %v", err)
	}

	return encoded
}

func TestNativeToolCallMatchesGoldenBaseline(t *testing.T) {
	request := &MessagesRequest{
		Model: ModelClaudeSonnet5,
		// The request states the token cap, because the API requires
		// max_tokens.
		MaxTokens: 4096,
		Messages: []MessagesMessage{
			{Role: "user", Content: json.RawMessage(`"What is the weather in SF?"`)},
			{Role: "assistant", Content: blocks(t, ContentBlock{
				Type:  ContentBlockTypeToolUse,
				ID:    "call_1",
				Name:  "get_weather",
				Input: json.RawMessage(`{"city":"SF"}`),
			})},
			// A tool result is a user turn in this protocol.
			{Role: "user", Content: blocks(t, ContentBlock{
				Type:          ContentBlockTypeToolResult,
				ToolUseID:     "call_1",
				ResultContent: `{"temp_c":18}`,
			})},
		},
		Tools: []MessagesTool{{
			Name:        "get_weather",
			Description: "Get the current weather in a city",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []any{"city"},
			},
			Strict: new(true),
		}},
		ToolChoice: &ToolChoice{Type: ToolChoiceTypeAuto, DisableParallelToolUse: new(true)},
	}

	assertMatchesGolden(t, "tool_call.json", captureNativeRequest(t, request))
}

func TestNativeImageContentMatchesGoldenBaseline(t *testing.T) {
	request := &MessagesRequest{
		Model:     ModelClaudeSonnet5,
		MaxTokens: 512,
		Messages: []MessagesMessage{{Role: "user", Content: blocks(
			t,
			ContentBlock{Type: ContentBlockTypeText, Text: "What is in this image?"},
			ContentBlock{Type: ContentBlockTypeImage, Source: &ContentSource{
				Type: ContentSourceTypeURL,
				URL:  "https://example.com/cat.png",
			}},
		)}},
	}

	assertMatchesGolden(t, "image_content.json", captureNativeRequest(t, request))
}

func TestNativeThinkingMatchesGoldenBaseline(t *testing.T) {
	request := &MessagesRequest{
		Model:        ModelClaudeSonnet5,
		MaxTokens:    4096,
		Messages:     []MessagesMessage{{Role: "user", Content: json.RawMessage(`"Solve it step by step."`)}},
		Thinking:     &MessagesThinking{Type: ThinkingTypeEnabled, BudgetTokens: 2048, Display: ThinkingDisplayOmitted},
		OutputConfig: &OutputConfig{Effort: EffortHigh},
	}

	assertMatchesGolden(t, "thinking.json", captureNativeRequest(t, request))
}

func TestNativePromptCachingMatchesGoldenBaseline(t *testing.T) {
	request := &MessagesRequest{
		Model:     ModelClaudeSonnet5,
		MaxTokens: 1024,
		Messages: []MessagesMessage{
			{Role: "user", Content: json.RawMessage(`"Answer from the cached context."`)},
		},
		System: blocks(t, ContentBlock{
			Type:         ContentBlockTypeText,
			Text:         "A long, reusable system prompt.",
			CacheControl: &CacheControl{Type: CacheControlTypeEphemeral},
		}),
		Tools: []MessagesTool{{
			Name:         "search",
			Description:  "Search the knowledge base",
			CacheControl: &CacheControl{Type: CacheControlTypeEphemeral},
		}},
		CacheControl: &CacheControl{Type: CacheControlTypeEphemeral, TTL: CacheControlTTL1h},
	}

	assertMatchesGolden(t, "prompt_caching.json", captureNativeRequest(t, request))
}

// TestNativeCachingUsageDecodes covers the response half of prompt caching:
// the per-TTL write breakdown and the read count.
func TestNativeCachingUsageDecodes(t *testing.T) {
	const fixture = `{"id":"msg_1","content":[],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":30,"cache_creation_input_tokens":1500,"cache_read_input_tokens":900,"cache_creation":{"ephemeral_5m_input_tokens":500,"ephemeral_1h_input_tokens":1000},"server_tool_use":{"web_search_requests":2,"web_fetch_requests":1},"inference_geo":"eu","service_tier":"priority"}}`

	var response MessagesResponse
	if err := json.Unmarshal([]byte(fixture), &response); err != nil {
		t.Fatal(err)
	}

	usage := response.Usage
	for _, tc := range []struct {
		field string
		got   any
		want  any
	}{
		{"cache_creation_input_tokens", usage.CacheCreationInputTokens, 1500},
		{"cache_read_input_tokens", usage.CacheReadInputTokens, 900},
		{"ephemeral_5m_input_tokens", usage.CacheCreation.Ephemeral5mInputTokens, 500},
		{"ephemeral_1h_input_tokens", usage.CacheCreation.Ephemeral1hInputTokens, 1000},
		{"web_search_requests", usage.ServerToolUse.WebSearchRequests, 2},
		{"web_fetch_requests", usage.ServerToolUse.WebFetchRequests, 1},
		{"inference_geo", usage.InferenceGeo, "eu"},
		{"service_tier", usage.ServiceTier, "priority"},
	} {
		if tc.got != tc.want {
			t.Errorf("usage.%s = %v, want %v", tc.field, tc.got, tc.want)
		}
	}

	if sum := usage.CacheCreation.Ephemeral5mInputTokens + usage.CacheCreation.Ephemeral1hInputTokens; sum != usage.CacheCreationInputTokens {
		t.Errorf("per-TTL breakdown sums to %d, want %d", sum, usage.CacheCreationInputTokens)
	}
}

// TestNativeToolUseResponseDecodesFully covers the response half of the tool
// call path, including the thinking block that accompanies it.
func TestNativeToolUseResponseDecodesFully(t *testing.T) {
	const fixture = `{"id":"msg_1","model":"claude-sonnet-5","role":"assistant","content":[{"type":"thinking","thinking":"weigh the options","signature":"sig"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"SF"}}],"stop_reason":"tool_use","usage":{"input_tokens":40,"output_tokens":25,"output_tokens_details":{"thinking_tokens":11}}}`

	var response MessagesResponse
	if err := json.Unmarshal([]byte(fixture), &response); err != nil {
		t.Fatal(err)
	}

	if response.StopReason != StopReasonToolUse {
		t.Errorf("stop reason = %q", response.StopReason)
	}

	if len(response.Content) != 2 {
		t.Fatalf("content blocks = %d, want 2", len(response.Content))
	}

	if response.Content[0].Thinking != "weigh the options" {
		t.Errorf("thinking = %q", response.Content[0].Thinking)
	}

	// The signature is not a modelled field; Raw is what keeps it reachable.
	if !bytes.Contains(response.Content[0].Raw, []byte(`"signature":"sig"`)) {
		t.Errorf("Raw lost the unmodelled signature: %s", response.Content[0].Raw)
	}

	tool := response.Content[1]
	if tool.ID != "toolu_1" || tool.Name != "get_weather" || string(tool.Input) != `{"city":"SF"}` {
		t.Errorf("tool block = %+v", tool)
	}

	if response.Usage.OutputTokensDetails.ThinkingTokens != 11 {
		t.Errorf("thinking tokens = %d", response.Usage.OutputTokensDetails.ThinkingTokens)
	}
}
