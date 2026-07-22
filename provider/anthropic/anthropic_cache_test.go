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
	"strings"
	"testing"
)

// TestToAnthropicRequest_SystemCacheBreakpoint verifies that a system
// message with CacheBreakpoint=true lands as a block-array system field
// whose last block carries "cache_control":{"type":"ephemeral"}.
func TestToAnthropicRequest_SystemCacheBreakpoint(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			cacheMsg(Message{
				Role:    RoleSystem,
				Content: NewTextContent("You are a helpful coder."),
			}),
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	var systemBlocks []anthropicContentBlock
	if err := json.Unmarshal(ar.System, &systemBlocks); err != nil {
		t.Fatalf("system not a block array: %v (raw=%s)", err, string(ar.System))
	}
	if len(systemBlocks) == 0 {
		t.Fatal("expected at least one system block")
	}
	last := systemBlocks[len(systemBlocks)-1]
	if last.CacheControl == nil || last.CacheControl.Type != "ephemeral" {
		t.Errorf("last system block CacheControl = %+v, want {ephemeral}", last.CacheControl)
	}
}

// TestToAnthropicRequest_ToolCacheBreakpoint verifies a tool with
// CacheBreakpoint=true emits cache_control on its translated form.
func TestToAnthropicRequest_ToolCacheBreakpoint(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "t1", Description: "first"}},
			cacheTool(Tool{Type: "function", Function: FunctionDefinition{Name: "t2", Description: "second"}}),
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	if len(ar.Tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(ar.Tools))
	}
	if ar.Tools[0].CacheControl != nil {
		t.Errorf("tool 0 CacheControl should be nil, got %+v", ar.Tools[0].CacheControl)
	}
	if ar.Tools[1].CacheControl == nil || ar.Tools[1].CacheControl.Type != "ephemeral" {
		t.Errorf("tool 1 CacheControl = %+v, want {ephemeral}", ar.Tools[1].CacheControl)
	}
}

// TestToAnthropicRequest_NoCacheBreakpoint verifies that without flags
// set, no cache_control field appears anywhere in the translated request.
func TestToAnthropicRequest_NoCacheBreakpoint(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			{Role: RoleSystem, Content: NewTextContent("You are a helpful coder.")},
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "t1"}},
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	body, err := json.Marshal(ar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(body), "cache_control") {
		t.Errorf("unexpected cache_control in request body: %s", body)
	}
}

// TestToAnthropicRequest_BothBreakpoints verifies the happy-path case we
// actually ship: one system + one last-tool marker = exactly 2 cache_control
// occurrences in the outbound JSON.
func TestToAnthropicRequest_BothBreakpoints(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			cacheMsg(Message{Role: RoleSystem, Content: NewTextContent("sys")}),
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "t1"}},
			cacheTool(Tool{Type: "function", Function: FunctionDefinition{Name: "t2"}}),
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	body, err := json.Marshal(ar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	count := strings.Count(string(body), `"cache_control":`)
	if count != 2 {
		t.Errorf("cache_control count = %d, want 2 (body=%s)", count, body)
	}
}

// TestToAnthropicRequest_SystemWithPartsAndCache verifies that a system
// message using multimodal parts still gets the marker on its last text
// block.
func TestToAnthropicRequest_SystemWithPartsAndCache(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			cacheMsg(Message{
				Role: RoleSystem,
				Content: NewPartsContent(
					ContentPart{Type: "text", Text: "first"},
					ContentPart{Type: "text", Text: "second"},
				),
			}),
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.System, &blocks); err != nil {
		t.Fatalf("system not a block array: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks = %d, want 2", len(blocks))
	}
	if blocks[0].CacheControl != nil {
		t.Errorf("first block should not carry cache_control")
	}
	if blocks[1].CacheControl == nil {
		t.Errorf("last block missing cache_control")
	}
}

// TestChatRequest_OpenAIShape_NoCacheControl verifies that marshalling the
// canonical ChatRequest (which is the on-wire body for OpenAI-compatible
// endpoints) never leaks a cache_control field, even when CacheBreakpoint
// is set on messages or tools.
func TestChatRequest_OpenAIShape_NoCacheControl(t *testing.T) {
	req := &ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			cacheMsg(Message{Role: RoleSystem, Content: NewTextContent("sys")}),
			cacheMsg(Message{Role: RoleUser, Content: NewTextContent("hi")}),
		},
		Tools: []Tool{
			cacheTool(Tool{Type: "function", Function: FunctionDefinition{Name: "t1"}}),
		},
	}
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(body), "cache_control") {
		t.Errorf("CacheBreakpoint leaked into OpenAI request body: %s", body)
	}
	if strings.Contains(string(body), "CacheBreakpoint") {
		t.Errorf("CacheBreakpoint field name leaked into OpenAI request body: %s", body)
	}
}

// TestToAnthropicRequest_AutoCacheDefault verifies AutoCache=true (no TTL)
// emits a request-root cache_control of type ephemeral with no ttl.
func TestToAnthropicRequest_AutoCacheDefault(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
	}
	autoCache(req, "")
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	if ar.CacheControl == nil || ar.CacheControl.Type != "ephemeral" {
		t.Fatalf("CacheControl = %+v, want {ephemeral}", ar.CacheControl)
	}
	if ar.CacheControl.TTL != "" {
		t.Errorf("TTL = %q, want empty (default 5m)", ar.CacheControl.TTL)
	}
	body, err := json.Marshal(ar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(body), `"cache_control":{"type":"ephemeral"}`) {
		t.Errorf("request-root cache_control not serialized as expected: %s", body)
	}
	if strings.Contains(string(body), `"ttl"`) {
		t.Errorf("ttl should be omitted when empty: %s", body)
	}
}

// TestToAnthropicRequest_AutoCache1h verifies AutoCacheTTL "1h" carries the
// ttl through onto the request-root cache_control.
func TestToAnthropicRequest_AutoCache1h(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
	}
	autoCache(req, "1h")
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	if ar.CacheControl == nil || ar.CacheControl.TTL != "1h" {
		t.Fatalf("CacheControl = %+v, want ttl=1h", ar.CacheControl)
	}
	body, err := json.Marshal(ar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(body), `"cache_control":{"type":"ephemeral","ttl":"1h"}`) {
		t.Errorf("request-root cache_control with ttl not serialized as expected: %s", body)
	}
}

// TestToAnthropicRequest_AutoCacheOff verifies that with AutoCache=false the
// request carries no request-root cache_control (default behavior unchanged),
// while a per-block CacheBreakpoint still works independently (coexistence).
func TestToAnthropicRequest_AutoCacheOff(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			cacheMsg(Message{Role: RoleSystem, Content: NewTextContent("sys")}),
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	if ar.CacheControl != nil {
		t.Errorf("request-root CacheControl should be nil when AutoCache off, got %+v", ar.CacheControl)
	}
	// The block-level breakpoint must still be present and independent.
	var systemBlocks []anthropicContentBlock
	if err := json.Unmarshal(ar.System, &systemBlocks); err != nil {
		t.Fatalf("system not a block array: %v", err)
	}
	if last := systemBlocks[len(systemBlocks)-1]; last.CacheControl == nil {
		t.Error("per-block CacheBreakpoint should still attach cache_control independently of AutoCache")
	}
}

// TestChatRequest_OpenAIShape_NoAutoCacheLeak verifies the AutoCache switch
// never leaks into the canonical (OpenAI-shape) request body.
func TestChatRequest_OpenAIShape_NoAutoCacheLeak(t *testing.T) {
	req := &ChatRequest{
		Model:    "gpt-4",
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
	}
	autoCache(req, "1h")
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, leak := range []string{"cache_control", "AutoCache", "auto_cache", `"ttl"`} {
		if strings.Contains(string(body), leak) {
			t.Errorf("AutoCache leaked %q into OpenAI request body: %s", leak, body)
		}
	}
}

// TestFromAnthropicResponse_CacheCreationBreakdown verifies the response
// usage.cache_creation 5m/1h breakdown and the total cache write count map
// onto the canonical Usage, with cached tokens still folded into PromptTokens.
func TestFromAnthropicResponse_CacheCreationBreakdown(t *testing.T) {
	raw := `{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-4-8",` +
		`"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn",` +
		`"usage":{"input_tokens":2048,"cache_read_input_tokens":1800,"cache_creation_input_tokens":248,` +
		`"output_tokens":503,"cache_creation":{"ephemeral_5m_input_tokens":148,"ephemeral_1h_input_tokens":100}}}`

	var ar anthropicResponse
	if err := json.Unmarshal([]byte(raw), &ar); err != nil {
		t.Fatalf("unmarshal anthropic response: %v", err)
	}
	if ar.Usage.CacheCreation == nil {
		t.Fatal("cache_creation not parsed into anthropicUsage")
	}
	if ar.Usage.CacheCreation.Ephemeral5mInputTokens != 148 || ar.Usage.CacheCreation.Ephemeral1hInputTokens != 100 {
		t.Errorf("cache_creation breakdown = %+v, want {148,100}", ar.Usage.CacheCreation)
	}

	cr := fromAnthropicResponse(&ar)
	u := cr.Usage

	ext := UsageExtensionOf(&u)
	if ext == nil {
		t.Fatal("usage extension missing")
	}
	if ext.CacheWriteTokens != 248 {
		t.Errorf("CacheWriteTokens = %d, want 248", ext.CacheWriteTokens)
	}
	if ext.CacheWrite5mTokens != 148 {
		t.Errorf("CacheWrite5mTokens = %d, want 148", ext.CacheWrite5mTokens)
	}
	if ext.CacheWrite1hTokens != 100 {
		t.Errorf("CacheWrite1hTokens = %d, want 100", ext.CacheWrite1hTokens)
	}
	if u.CacheReadTokens != 1800 {
		t.Errorf("CacheReadTokens = %d, want 1800", u.CacheReadTokens)
	}
	if u.PromptTokens != 2048+248+1800 {
		t.Errorf("PromptTokens = %d, want %d (folded)", u.PromptTokens, 2048+248+1800)
	}
}

// TestFromAnthropicResponse_NoCacheCreation verifies that without a
// cache_creation object the breakdown fields stay zero while the total cache
// write still reflects cache_creation_input_tokens.
func TestFromAnthropicResponse_NoCacheCreation(t *testing.T) {
	ar := anthropicResponse{
		Content:    responseBlocks([]anthropicContentBlock{{Type: "text", Text: "hi"}}),
		StopReason: "end_turn",
		Usage:      anthropicUsage{InputTokens: 10, OutputTokens: 5, CacheCreationInputTokens: 7},
	}
	u := fromAnthropicResponse(&ar).Usage

	ext := UsageExtensionOf(&u)
	if ext == nil {
		t.Fatal("usage extension missing")
	}
	if ext.CacheWriteTokens != 7 {
		t.Errorf("CacheWriteTokens = %d, want 7", ext.CacheWriteTokens)
	}
	if ext.CacheWrite5mTokens != 0 || ext.CacheWrite1hTokens != 0 {
		t.Errorf("breakdown should be zero without cache_creation, got 5m=%d 1h=%d", ext.CacheWrite5mTokens, ext.CacheWrite1hTokens)
	}
}
