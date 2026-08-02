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
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// nativeStreamServer replays a fixed SSE body, so accumulation is tested
// against exact wire bytes rather than a live backend.
func nativeStreamServer(t *testing.T, body string) *Client {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	return NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
}

func drainNative(t *testing.T, stream *MessageStream) {
	t.Helper()

	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}
}

// fullStream carries every block kind the accumulator handles: text, thinking
// and a tool_use whose input arrives as partial JSON. It is the same fixture
// the canonical stream is replayed against, so both entry points are measured
// on identical bytes.
func fullStream(t *testing.T) string {
	t.Helper()

	body, err := os.ReadFile(filepath.Join("testdata", "golden", "stream_usage.sse"))
	if err != nil {
		t.Fatalf("read stream fixture: %v", err)
	}

	return string(body)
}

func TestMessageStreamAccumulatesEveryBlockKind(t *testing.T) {
	stream, err := nativeStreamServer(t, fullStream(t)).
		MessagesStream(context.Background(), &MessagesRequest{Model: "claude-sonnet-5", MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	drainNative(t, stream)

	message := stream.Message()
	if message == nil {
		t.Fatal("Message is nil after a stream that carried events")
	}

	if message.ID != "msg_1" || message.Model != "claude-sonnet-5" || message.Role != "assistant" {
		t.Errorf("envelope = %+v", message)
	}

	if message.StopReason != StopReasonToolUse {
		t.Errorf("stop reason = %q", message.StopReason)
	}

	if len(message.Content) != 3 {
		t.Fatalf("content blocks = %d, want 3", len(message.Content))
	}

	if message.Content[0].Type != ContentBlockTypeThinking || message.Content[0].Thinking != "let me check" {
		t.Errorf("thinking block = %+v", message.Content[0])
	}

	if message.Content[1].Type != ContentBlockTypeText || message.Content[1].Text != "Hello" {
		t.Errorf("text block = %+v", message.Content[1])
	}

	tool := message.Content[2]
	if tool.Type != ContentBlockTypeToolUse || tool.ID != "toolu_1" || tool.Name != "get_weather" {
		t.Errorf("tool block identity = %+v", tool)
	}

	if string(tool.Input) != `{"city":"SF"}` {
		t.Errorf("tool input = %s, want the reassembled JSON", tool.Input)
	}

	if len(message.Content[1].Raw) == 0 {
		t.Error("Raw must keep the block as it first arrived")
	}
}

// TestMessageStreamUsageMergesStartAndTerminalEvents pins the merge rule the
// canonical layer used to apply in a private helper: the terminal event
// reports only output_tokens and must not blank out what message_start
// established.
func TestMessageStreamUsageMergesStartAndTerminalEvents(t *testing.T) {
	stream, err := nativeStreamServer(t, fullStream(t)).
		MessagesStream(context.Background(), &MessagesRequest{Model: "claude-sonnet-5", MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	drainNative(t, stream)

	usage := stream.Usage()
	if usage == nil {
		t.Fatal("Usage is nil after a stream that reported it")
	}

	for _, tc := range []struct {
		field string
		got   any
		want  any
	}{
		{"input_tokens", usage.InputTokens, 100},
		{"output_tokens", usage.OutputTokens, 25},
		{"cache_read_input_tokens", usage.CacheReadInputTokens, 20},
		{"cache_creation_input_tokens", usage.CacheCreationInputTokens, 5},
		{"inference_geo", usage.InferenceGeo, "us"},
		{"service_tier", usage.ServiceTier, "standard"},
	} {
		if tc.got != tc.want {
			t.Errorf("usage.%s = %v, want %v", tc.field, tc.got, tc.want)
		}
	}

	if &stream.Message().Usage != usage {
		t.Error("Message().Usage and Usage() must report the same object")
	}
}

func TestMessageStreamUsageIsNilBeforeMessageStart(t *testing.T) {
	stream, err := nativeStreamServer(t, "event: ping\ndata: {\"type\":\"ping\"}\n\n").
		MessagesStream(context.Background(), &MessagesRequest{Model: "claude-sonnet-5", MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	if stream.Usage() != nil || stream.Message() != nil {
		t.Error("Usage and Message must be nil before the first message event")
	}

	drainNative(t, stream)

	if stream.Usage() != nil || stream.Message() != nil {
		t.Error("a stream of pings carries no message")
	}
}

// TestMessageStreamPreservesUnknownBlocks verifies an unmodelled block reaches
// the caller instead of being dropped or diverted into a side channel.
func TestMessageStreamPreservesUnknownBlocks(t *testing.T) {
	const body = `event: message_start
data: {"type":"message_start","message":{"id":"msg_2","role":"assistant","content":[],"usage":{"input_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"type":"web_search_result","url":"https://example.com"}]}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

`

	stream, err := nativeStreamServer(t, body).
		MessagesStream(context.Background(), &MessagesRequest{Model: "claude-sonnet-5", MaxTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	drainNative(t, stream)

	message := stream.Message()
	if len(message.Content) != 1 {
		t.Fatalf("content blocks = %d, want the unmodelled block kept", len(message.Content))
	}

	block := message.Content[0]
	if block.Type != "web_search_tool_result" || block.ToolUseID != "srv_1" {
		t.Errorf("block = %+v", block)
	}

	if len(block.Content) == 0 {
		t.Error("the polymorphic response-side content must be preserved verbatim")
	}
}
