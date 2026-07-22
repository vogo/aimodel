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
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
)

// sameJSON reports whether two raw JSON documents are semantically equal.
func sameJSON(t *testing.T, got json.RawMessage, want string) bool {
	t.Helper()

	var g, w any

	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("decode got %s: %v", got, err)
	}

	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("decode want %s: %v", want, err)
	}

	return reflect.DeepEqual(g, w)
}

// TestFromAnthropicResponse_ExtraBlocks verifies unmodelled blocks are kept
// verbatim — including fields this wrapper has no struct member for — while
// the known blocks keep their existing behavior.
func TestFromAnthropicResponse_ExtraBlocks(t *testing.T) {
	const serverToolUse = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"go"},"nested":{"deep":[1,2,{"k":"v"}]}}`

	const searchResult = `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://go.dev","title":"Go"}]}`

	body := `{"id":"msg_x","model":"claude-sonnet-4","stop_reason":"end_turn",
		"usage":{"input_tokens":1,"output_tokens":1},
		"content":[
			{"type":"thinking","thinking":"pondering"},
			` + serverToolUse + `,
			` + searchResult + `,
			{"type":"text","text":"Go is at go.dev"},
			{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"a":1}}
		]}`

	var ar MessagesResponse
	if err := json.Unmarshal([]byte(body), &ar); err != nil {
		t.Fatalf("decode: %v", err)
	}

	msg := fromAnthropicResponse(&ar).Choices[0].Message

	// Known blocks are unaffected.
	if msg.Thinking != "pondering" {
		t.Errorf("thinking = %q", msg.Thinking)
	}

	if msg.Content.Text() != "Go is at go.dev" {
		t.Errorf("content = %q", msg.Content.Text())
	}

	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "toolu_1" {
		t.Errorf("tool_calls = %+v", msg.ToolCalls)
	}

	// Unknown blocks are preserved, in order, with every nested field intact.
	blocks := extraBlocksOf(&msg)
	if len(blocks) != 2 {
		t.Fatalf("extra blocks len = %d, want 2: %v", len(blocks), blocks)
	}

	if !sameJSON(t, blocks[0], serverToolUse) {
		t.Errorf("extra block 0 = %s\nwant %s", blocks[0], serverToolUse)
	}

	if !sameJSON(t, blocks[1], searchResult) {
		t.Errorf("extra block 1 = %s\nwant %s", blocks[1], searchResult)
	}
}

// TestFromAnthropicResponse_TextBlockCitations verifies a text block carrying
// citations contributes its text as usual and is additionally preserved whole,
// while a plain text block adds nothing to ExtraBlocks.
func TestFromAnthropicResponse_TextBlockCitations(t *testing.T) {
	const cited = `{"type":"text","text":"Go was released in 2009.","citations":[{"type":"web_search_result_location","url":"https://go.dev","cited_text":"2009"}]}`

	body := `{"id":"msg_c","model":"claude-sonnet-4","stop_reason":"end_turn",
		"usage":{"input_tokens":1,"output_tokens":1},
		"content":[{"type":"text","text":"Answer:"},` + cited + `]}`

	var ar MessagesResponse
	if err := json.Unmarshal([]byte(body), &ar); err != nil {
		t.Fatalf("decode: %v", err)
	}

	msg := fromAnthropicResponse(&ar).Choices[0].Message

	if msg.Content.Text() != "Answer:\nGo was released in 2009." {
		t.Errorf("content = %q", msg.Content.Text())
	}

	blocks := extraBlocksOf(&msg)
	if len(blocks) != 1 {
		t.Fatalf("extra blocks len = %d, want only the cited block: %v", len(blocks), blocks)
	}

	if !sameJSON(t, blocks[0], cited) {
		t.Errorf("extra block = %s\nwant %s", blocks[0], cited)
	}
}

// TestFromAnthropicResponse_NoExtraBlocks verifies an ordinary response leaves
// ExtraBlocks nil.
func TestFromAnthropicResponse_NoExtraBlocks(t *testing.T) {
	body := `{"id":"m","model":"claude-sonnet-4","stop_reason":"end_turn",
		"usage":{"input_tokens":1,"output_tokens":1},
		"content":[{"type":"text","text":"hi"},{"type":"thinking","thinking":"t"}]}`

	var ar MessagesResponse
	if err := json.Unmarshal([]byte(body), &ar); err != nil {
		t.Fatalf("decode: %v", err)
	}

	msg := fromAnthropicResponse(&ar).Choices[0].Message
	if got := extraBlocksOf(&msg); got != nil {
		t.Errorf("extra blocks = %v, want nil", got)
	}
}

// TestAnthropicStream_ExtraBlocks drives a full SSE sequence mixing a known
// text block with an unrecognized block, and verifies the unknown block's
// start and its deltas are preserved verbatim in arrival order while the known
// text/tool paths keep working. It also checks Message.AppendDelta preserves
// that order.
func TestAnthropicStream_ExtraBlocks(t *testing.T) {
	const unknownStart = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","extra":{"k":1}}`

	const unknownDelta = `{"type":"input_json_delta","partial_json":"{\"q\":"}`

	const futureDelta = `{"type":"some_future_delta","payload":{"n":1}}`

	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Searching"}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":` + unknownStart + `}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":` + unknownDelta + `}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":` + futureDelta + `}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var acc Message

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}

		if len(chunk.Choices) > 0 {
			acc.AppendDelta(&chunk.Choices[0].Delta)
		}
	}

	if acc.Content.Text() != "Searching" {
		t.Errorf("text = %q, want Searching", acc.Content.Text())
	}

	want := []string{unknownStart, unknownDelta, futureDelta}

	blocks := extraBlocksOf(&acc)
	if len(blocks) != len(want) {
		t.Fatalf("extra blocks len = %d, want %d: %v", len(blocks), len(want), blocks)
	}

	for i, w := range want {
		if !sameJSON(t, blocks[i], w) {
			t.Errorf("extra block %d = %s\nwant %s", i, blocks[i], w)
		}
	}
}

// TestAnthropicStream_KnownBlocksNoExtra is the regression guard: a stream of
// only modelled events must produce no ExtraBlocks, and signature_delta stays
// ignored rather than being misclassified as unknown.
func TestAnthropicStream_KnownBlocksNoExtra(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"abc"}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var acc Message

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}

		if len(chunk.Choices) > 0 {
			acc.AppendDelta(&chunk.Choices[0].Delta)
		}
	}

	if got := extraBlocksOf(&acc); got != nil {
		t.Errorf("extra blocks = %v, want none", got)
	}

	if acc.Thinking != "hmm" {
		t.Errorf("thinking = %q, want hmm", acc.Thinking)
	}

	if len(acc.ToolCalls) != 1 || acc.ToolCalls[0].Function.Arguments != `{"a":1}` {
		t.Errorf("tool calls = %+v", acc.ToolCalls)
	}
}

// TestAnthropicStream_ContainerAndUsageMerge verifies the container is exposed
// exactly once, as early as message_start, and that merging the terminal usage
// does not blank out the extended information established at message_start.
func TestAnthropicStream_ContainerAndUsageMerge(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4","content":[],` +
		`"container":{"id":"container_abc","expires_at":"2026-07-21T10:00:00Z"},` +
		`"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":20,` +
		`"server_tool_use":{"web_search_requests":2},"inference_geo":"us","service_tier":"priority",` +
		`"output_tokens_details":{"thinking_tokens":5}}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	var (
		containers []*ResponseContainer
		lastUsage  *Usage
	)

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}

		if c := chunkContainer(chunk); c != nil {
			containers = append(containers, c)
		}

		if chunk.Usage != nil {
			lastUsage = chunk.Usage
		}
	}

	if len(containers) != 1 {
		t.Fatalf("container reported %d times, want exactly 1", len(containers))
	}

	if containers[0].ID != "container_abc" || containers[0].ExpiresAt != "2026-07-21T10:00:00Z" {
		t.Errorf("container = %+v", containers[0])
	}

	if lastUsage == nil {
		t.Fatal("no usage chunk")
	}

	if lastUsage.CompletionTokens != 42 {
		t.Errorf("completion_tokens = %d, want the terminal 42", lastUsage.CompletionTokens)
	}

	if lastUsage.PromptTokens != 120 {
		t.Errorf("prompt_tokens = %d, want the message_start 120", lastUsage.PromptTokens)
	}

	if lastUsage.CacheReadTokens != 20 {
		t.Errorf("cache_read_tokens = %d, want 20 preserved through the merge", lastUsage.CacheReadTokens)
	}

	uext := UsageExtensionOf(lastUsage)
	if uext == nil {
		t.Fatal("usage extension missing")
	}

	if uext.InferenceGeo != "us" || lastUsage.ServiceTier != "priority" {
		t.Errorf("geo/tier = %q/%q, want them preserved through the merge", uext.InferenceGeo, lastUsage.ServiceTier)
	}

	if uext.ServerToolUse == nil || uext.ServerToolUse.WebSearchRequests != 2 {
		t.Errorf("server_tool_use = %+v, want it preserved through the merge", uext.ServerToolUse)
	}

	if lastUsage.ReasoningTokens != 5 {
		t.Errorf("reasoning_tokens = %d, want 5 preserved through the merge", lastUsage.ReasoningTokens)
	}
}

// TestAnthropicStream_NoContainer verifies a stream without a container emits
// no container-bearing chunk at all.
func TestAnthropicStream_NoContainer(t *testing.T) {
	body := "" +
		"event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	s := newAnthropicStream(io.NopCloser(strings.NewReader(body)))

	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			t.Fatalf("Recv: %v", err)
		}

		if c := chunkContainer(chunk); c != nil {
			t.Errorf("container = %+v, want none", c)
		}
	}
}
