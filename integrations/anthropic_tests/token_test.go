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

package anthropic_tests

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vogo/aimodel/anthropic"
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

func TestAnthropicCacheTokensSync(t *testing.T) {
	server := mockBackend(t, "application/json", `{
		"id": "msg_01",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-5",
		"content": [{"type": "text", "text": "Hello"}],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 50,
			"cache_creation_input_tokens": 10,
			"cache_read_input_tokens": 30,
			"output_tokens": 20
		}
	}`)

	client := anthropic.NewClient("sk-test", anthropic.WithBaseURL(server.URL), anthropic.WithHTTPClient(server.Client()))

	response, err := client.Messages(context.Background(), &anthropic.MessagesRequest{
		Model:     anthropic.ModelClaudeSonnet5,
		MaxTokens: 64,
		Messages:  []anthropic.MessagesMessage{{Role: "user", Content: []byte(`"Hi"`)}},
	})
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}

	usage := response.Usage
	if usage.InputTokens != 50 || usage.OutputTokens != 20 {
		t.Errorf("input/output = %d/%d, want 50/20", usage.InputTokens, usage.OutputTokens)
	}

	// Anthropic reports the cache counts alongside input_tokens rather than
	// inside it: the billable input is the sum of all three.
	if usage.CacheCreationInputTokens != 10 || usage.CacheReadInputTokens != 30 {
		t.Errorf("cache write/read = %d/%d, want 10/30", usage.CacheCreationInputTokens, usage.CacheReadInputTokens)
	}
}

func TestAnthropicCacheTokensStream(t *testing.T) {
	server := mockBackend(t, "text/event-stream",
		"event: message_start\n"+
			`data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"stop_reason":null,"usage":{"input_tokens":50,"cache_creation_input_tokens":10,"cache_read_input_tokens":30,"output_tokens":0}}}`+"\n\n"+
			"event: content_block_start\n"+
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n"+
			"event: content_block_delta\n"+
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`+"\n\n"+
			"event: message_delta\n"+
			`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}`+"\n\n"+
			"event: message_stop\n"+
			`data: {"type":"message_stop"}`+"\n\n")

	client := anthropic.NewClient("sk-test", anthropic.WithBaseURL(server.URL), anthropic.WithHTTPClient(server.Client()))

	stream, err := client.MessagesStream(context.Background(), &anthropic.MessagesRequest{
		Model:     anthropic.ModelClaudeSonnet5,
		MaxTokens: 64,
		Messages:  []anthropic.MessagesMessage{{Role: "user", Content: []byte(`"Hi"`)}},
	})
	if err != nil {
		t.Fatalf("MessagesStream: %v", err)
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

	// The terminal event carries only output_tokens; the cache counts from
	// message_start must survive the merge.
	if usage.CacheReadInputTokens != 30 || usage.CacheCreationInputTokens != 10 {
		t.Errorf("cache read/write = %d/%d, want 30/10", usage.CacheReadInputTokens, usage.CacheCreationInputTokens)
	}

	if usage.InputTokens != 50 || usage.OutputTokens != 20 {
		t.Errorf("input/output = %d/%d, want 50/20", usage.InputTokens, usage.OutputTokens)
	}
}

func TestAnthropicZeroCacheTokens(t *testing.T) {
	server := mockBackend(t, "application/json", `{
		"id": "msg_02",
		"type": "message",
		"role": "assistant",
		"model": "claude-sonnet-5",
		"content": [{"type": "text", "text": "Hello"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 20}
	}`)

	client := anthropic.NewClient("sk-test", anthropic.WithBaseURL(server.URL), anthropic.WithHTTPClient(server.Client()))

	response, err := client.Messages(context.Background(), &anthropic.MessagesRequest{
		Model:     anthropic.ModelClaudeSonnet5,
		MaxTokens: 64,
		Messages:  []anthropic.MessagesMessage{{Role: "user", Content: []byte(`"Hi"`)}},
	})
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}

	if response.Usage.CacheReadInputTokens != 0 || response.Usage.CacheCreationInputTokens != 0 {
		t.Errorf("cache counts = %+v, want zeroes", response.Usage)
	}

	if response.Usage.CacheCreation != nil {
		t.Errorf("cache_creation = %+v, want nil when the API omits it", response.Usage.CacheCreation)
	}
}
